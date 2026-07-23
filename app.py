"""
VWAP + EMA8 Stock Screener (Upstox API + Streamlit)
====================================================

Logic
-----
For every symbol in the watchlist:
  - Fetch today's intraday 1-minute candles -> compute running VWAP
  - Fetch recent daily candles -> compute EMA(8) on close price
  - Compare the current traded price (LTP) against VWAP and EMA8

Classification:
  BULLISH  -> price > VWAP  AND  price > EMA8   (shown in the "Above" table)
  BEARISH  -> price < VWAP  AND  price < EMA8   (shown in the "Below" table)
  NEUTRAL  -> mixed (price above one, below the other) -> shown in tables as
              neither Above nor Below, but its chart IS still rendered below
              (per your latest request: charts for ALL screened stocks)

A TradingView-style candlestick chart (Plotly) with VWAP and EMA8 overlaid
is shown for EVERY stock that returned usable data (no dropdown) — you can
order the charts by signal strength (how far price has moved from
VWAP/EMA8) or alphabetically.

Setup
-----
1. Create an app at https://developer.upstox.com/ and generate an
   OAuth2 access token (valid for the trading day).
2. Put the token in Streamlit secrets (recommended) or paste it in the
   sidebar at runtime:

   .streamlit/secrets.toml
   -----------------------
   UPSTOX_ACCESS_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOi..."

3. pip install -r requirements.txt
4. streamlit run app.py
"""

import io
import gzip
import json
import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
from datetime import datetime, timedelta
from urllib.parse import quote

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

st.set_page_config(page_title="VWAP / EMA8 Screener", layout="wide")

WATCHLIST = [
    "SHRIRAMFIN", "BHARTIARTL", "AXISBANK", "SUNPHARMA", "CIPLA",
    "HDFCLIFE", "APOLLOHOSP", "JIOFIN", "LT", "TATAMOTORS",
    "ITC", "ICICIBANK", "INDIGO", "BAJAJ-AUTO", "NESTLEIND",
    "BAJAJFINSV", "TATASTEEL", "ADANIPORTS", "DRREDDY", "GRASIM",
    "ONGC", "TRENT", "HDFCBANK", "ADANIENT", "KOTAKBANK",
    "JSWSTEEL", "ASIANPAINT", "SBILIFE", "MARUTI", "RELIANCE",
    "EICHERMOT", "ULTRACEMCO", "HINDUNILVR", "SBIN", "MAXHEALTH",
    "BAJFINANCE", "TITAN", "COALINDIA", "POWERGRID", "NTPC",
    "TATACONSUM", "M&M", "HINDALCO", "BEL", "ETERNAL",
    "TCS", "HCLTECH", "WIPRO", "INFY", "TECHM",
]
# NOTE: "TMPV" in the original list looks like a typo for TATAMOTORS -
# change it back if you meant a different symbol.

UPSTOX_V2 = "https://api.upstox.com/v2"
UPSTOX_V3 = "https://api.upstox.com/v3"
NSE_INSTRUMENTS_JSON_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz"
EMA_PERIOD = 8


# --------------------------------------------------------------------------
# AUTH
# --------------------------------------------------------------------------

def get_access_token() -> str:
    token = st.secrets.get("UPSTOX_ACCESS_TOKEN", "") if hasattr(st, "secrets") else ""
    with st.sidebar:
        st.header("Upstox Auth")
        token = st.text_input(
            "Access Token",
            value=token,
            type="password",
            help="Generate daily from https://developer.upstox.com/ (OAuth2 login flow).",
        )
        st.caption("Token is only kept in this session, never written to disk.")
    return token.strip()


def auth_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }


# --------------------------------------------------------------------------
# INSTRUMENT LOOKUP  (symbol -> instrument_key)
#
# Primary path: Upstox's authenticated Instrument Search API
# (GET /v2/instruments/search). Some Upstox token types (e.g. Analytics
# Tokens) are documented as supporting this endpoint but in practice get
# rejected with 401 / UDAPI100050 on it specifically, even while working
# fine on candle/quote endpoints — this is a known Upstox-side quirk, not
# necessarily an expired token. See:
# community.upstox.com/t/upstox-analytics-token-works-for-historical-candles-but-returns-udapi100050-on-instrument-search
#
# Fallback path: Upstox's public NSE.json.gz instrument file. It needs no
# auth at all, so it sidesteps the quirk above entirely. We try the search
# API first (fast, always fresh) and only fall back per-symbol if it fails.
# --------------------------------------------------------------------------

def _search_instrument(symbol: str, token: str) -> str | None:
    url = f"{UPSTOX_V2}/instruments/search"
    params = {"query": symbol, "exchanges": "NSE", "segments": "EQ", "records": 10}
    r = requests.get(url, headers=auth_headers(token), params=params, timeout=15)
    r.raise_for_status()
    results = r.json().get("data", [])
    if not results:
        return None
    # Prefer an exact trading_symbol match on NSE cash-market equity
    for item in results:
        if item.get("trading_symbol", "").upper() == symbol.upper() and item.get("exchange") == "NSE":
            return item.get("instrument_key")
    # Fall back to the first NSE EQ result
    for item in results:
        if item.get("exchange") == "NSE" and item.get("instrument_type") == "EQ":
            return item.get("instrument_key")
    return results[0].get("instrument_key")


def _upstox_error_detail(resp) -> str:
    """Pull Upstox's own errorCode/message out of a failed response body,
    e.g. {"errors":[{"errorCode":"UDAPI100050","message":"Invalid token..."}]}"""
    try:
        body = resp.json()
        errs = body.get("errors") or []
        if errs:
            code = errs[0].get("errorCode", "")
            msg = errs[0].get("message", "")
            return f" [{code}: {msg}]" if code or msg else ""
    except Exception:
        pass
    return ""


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_search(symbol: str, _token: str, cache_bust: str):
    """Returns (instrument_key_or_None, error_message_or_None).
    `_token` is excluded from Streamlit's cache key (leading underscore);
    `cache_bust` (today's date) naturally expires the cache each day."""
    try:
        key = _search_instrument(symbol, _token)
        if key:
            return key, None
        return None, "no matching NSE equity instrument found"
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        detail = _upstox_error_detail(e.response) if e.response is not None else ""
        return None, f"Search API HTTP {status}{detail}"
    except requests.RequestException as e:
        return None, f"network error ({e})"


@st.cache_data(ttl=12 * 3600, show_spinner="Downloading public NSE instrument list (fallback)...")
def _load_nse_instrument_file() -> dict:
    """symbol (upper) -> instrument_key, built from Upstox's public,
    unauthenticated NSE.json.gz file. Used only as a fallback."""
    r = requests.get(NSE_INSTRUMENTS_JSON_URL, timeout=30)
    r.raise_for_status()
    with gzip.open(io.BytesIO(r.content)) as f:
        data = json.load(f)
    mapping = {}
    for item in data:
        if item.get("segment") == "NSE_EQ" and item.get("exchange") == "NSE":
            sym = (item.get("trading_symbol") or "").upper()
            if sym:
                mapping[sym] = item.get("instrument_key")
    return mapping


def resolve_instrument_keys(symbols: list[str], token: str):
    """Returns (mapping, errors, used_fallback).
    Tries the authenticated Search API per symbol first. Any symbol that
    fails there is retried against the public instrument file, so a
    search-API-only quirk (auth or otherwise) doesn't take down the whole
    screener."""
    today = datetime.now().strftime("%Y-%m-%d")
    mapping, errors = {}, []
    search_dead = False  # once the search API fails once, don't keep hammering it

    for sym in symbols:
        if search_dead:
            continue
        key, err = _cached_search(sym, token, today)
        if key:
            mapping[sym] = key
        elif err:
            errors.append(f"{sym}: {err}")
            search_dead = True  # same failure mode will repeat for every symbol

    unresolved = [s for s in symbols if s not in mapping]
    used_fallback = False
    if unresolved:
        try:
            file_map = _load_nse_instrument_file()
            still_missing = []
            for sym in unresolved:
                key = file_map.get(sym.upper()) or file_map.get(f"{sym.upper()}-EQ")
                if key:
                    mapping[sym] = key
                    used_fallback = True
                else:
                    still_missing.append(sym)
            if still_missing:
                errors.append(f"Not found in fallback file either: {', '.join(still_missing)}")
        except Exception as e:
            errors.append(f"Fallback instrument file also failed: {e}")

    return mapping, errors, used_fallback


# --------------------------------------------------------------------------
# DATA FETCH
# --------------------------------------------------------------------------

def _candles_to_df(candles: list) -> pd.DataFrame:
    if not candles:
        return pd.DataFrame()
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    df = pd.DataFrame(candles, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_intraday_1min(instrument_key: str, token: str) -> pd.DataFrame:
    """Today's 1-minute candles, used to compute running VWAP.
    Empty (zero rows) whenever the market hasn't traded yet today —
    e.g. before 9:15 AM IST, on a weekend, or on an exchange holiday."""
    key = quote(instrument_key, safe="")
    url = f"{UPSTOX_V3}/historical-candle/intraday/{key}/minutes/1"
    r = requests.get(url, headers=auth_headers(token), timeout=15)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return _candles_to_df(candles)


def fetch_last_session_1min(instrument_key: str, token: str, lookback_days: int = 7):
    """Fallback for when the market is closed: pulls 1-minute candles for the
    last several calendar days and returns only the most recent trading
    session found (handles weekends/holidays automatically).
    Returns (dataframe, session_date) — dataframe is empty if nothing found."""
    key = quote(instrument_key, safe="")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{UPSTOX_V3}/historical-candle/{key}/minutes/1/{to_date}/{from_date}"
    r = requests.get(url, headers=auth_headers(token), timeout=15)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    df = _candles_to_df(candles)
    if df.empty:
        return df, None
    last_session_date = df["timestamp"].dt.date.max()
    session_df = df[df["timestamp"].dt.date == last_session_date].reset_index(drop=True)
    return session_df, last_session_date


def fetch_price_series(instrument_key: str, token: str):
    """Returns (dataframe, is_live, session_date) — tries today's live
    intraday candles first, and falls back to the last completed session
    (previous trading day) if the market is currently closed."""
    live_df = fetch_intraday_1min(instrument_key, token)
    if not live_df.empty:
        return live_df, True, datetime.now().date()
    fallback_df, session_date = fetch_last_session_1min(instrument_key, token)
    return fallback_df, False, session_date


def fetch_daily_candles(instrument_key: str, token: str, lookback_days: int = 40) -> pd.DataFrame:
    """Recent daily candles, used to compute EMA(8)."""
    key = quote(instrument_key, safe="")
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{UPSTOX_V3}/historical-candle/{key}/days/1/{to_date}/{from_date}"
    r = requests.get(url, headers=auth_headers(token), timeout=15)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    return _candles_to_df(candles)


# --------------------------------------------------------------------------
# INDICATORS
# --------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> float:
    if df.empty:
        return np.nan
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical_price * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    vwap_series = cum_pv / cum_vol.replace(0, np.nan)
    return float(vwap_series.iloc[-1])


def compute_ema8(daily_df: pd.DataFrame, current_price: float) -> float:
    if daily_df.empty:
        return np.nan
    closes = list(daily_df["close"])
    closes.append(current_price)  # include today's live price as the latest point
    series = pd.Series(closes)
    ema = series.ewm(span=EMA_PERIOD, adjust=False).mean()
    return float(ema.iloc[-1])


# --------------------------------------------------------------------------
# SCREENING PIPELINE
# --------------------------------------------------------------------------

def screen_stocks(symbols: list[str], token: str) -> pd.DataFrame:
    key_map, resolve_errors, used_fallback = resolve_instrument_keys(symbols, token)

    if used_fallback:
        st.info(
            "ℹ️ Upstox's Instrument Search API rejected one or more requests, so those "
            "symbols were resolved through Upstox's **public, unauthenticated** NSE "
            "instrument list instead. This is a known Upstox-side quirk on that specific "
            "endpoint for some token types — it doesn't necessarily mean your token is "
            "expired. Screening will continue normally."
        )
    if resolve_errors:
        with st.expander(f"⚠️ {len(resolve_errors)} lookup issue(s) — click for details"):
            for e in resolve_errors:
                st.write("•", e)

    rows = []
    progress = st.progress(0.0, text="Screening...")
    n = len(symbols)
    token_confirmed_dead = False

    for i, sym in enumerate(symbols):
        instrument_key = key_map.get(sym)
        if not instrument_key:
            progress.progress((i + 1) / n)
            continue
        try:
            price_df, is_live, session_date = fetch_price_series(instrument_key, token)
            daily = fetch_daily_candles(instrument_key, token)
            if price_df.empty:
                progress.progress((i + 1) / n)
                continue

            ltp = float(price_df["close"].iloc[-1])
            vwap = compute_vwap(price_df)
            ema8 = compute_ema8(daily, ltp)

            if np.isnan(vwap) or np.isnan(ema8):
                progress.progress((i + 1) / n)
                continue

            if ltp > vwap and ltp > ema8:
                status = "ABOVE"
            elif ltp < vwap and ltp < ema8:
                status = "BELOW"
            else:
                status = "NEUTRAL"  # excluded from the Above/Below tables, but still charted

            # Signal strength: combined % distance of price from VWAP and EMA8.
            # Larger value = price has moved further away from both lines.
            strength = abs((ltp - vwap) / vwap) + abs((ltp - ema8) / ema8)

            rows.append({
                "Symbol": sym,
                "Instrument Key": instrument_key,
                "LTP": round(ltp, 2),
                "VWAP": round(vwap, 2),
                "EMA8": round(ema8, 2),
                "Status": status,
                "Strength": strength,
                "Session": "Live" if is_live else f"Last close ({session_date})",
            })
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 401 and not token_confirmed_dead:
                token_confirmed_dead = True
                detail = _upstox_error_detail(e.response) if e.response is not None else ""
                st.error(
                    f"🔒 {sym}: 401{detail} on a **market-data** endpoint (candles), not just "
                    "the search endpoint — this does indicate the token itself is invalid/expired "
                    "for your whole account. Generate a fresh one at "
                    "[developer.upstox.com](https://developer.upstox.com/)."
                )
            else:
                st.warning(f"{sym}: API error ({e})")
        except Exception as e:
            st.warning(f"{sym}: {e}")

        progress.progress((i + 1) / n)
        time.sleep(0.05)  # gentle pacing to avoid rate limiting

    progress.empty()
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# CHART (TradingView-style candlestick + VWAP + EMA8)
# --------------------------------------------------------------------------

def build_chart(symbol: str, instrument_key: str, token: str):
    """Fetches price data and returns (figure_or_None, is_live, session_date)."""
    price_df, is_live, session_date = fetch_price_series(instrument_key, token)
    if price_df.empty:
        return None, is_live, session_date

    typical_price = (price_df["high"] + price_df["low"] + price_df["close"]) / 3
    cum_pv = (typical_price * price_df["volume"]).cumsum()
    cum_vol = price_df["volume"].cumsum()
    price_df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)
    price_df["ema8"] = price_df["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=price_df["timestamp"], open=price_df["open"], high=price_df["high"],
        low=price_df["low"], close=price_df["close"], name=symbol,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(
        x=price_df["timestamp"], y=price_df["vwap"], name="VWAP",
        line=dict(color="#ff9800", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=price_df["timestamp"], y=price_df["ema8"], name="EMA8",
        line=dict(color="#2962ff", width=1.6),
    ))

    session_note = "live, today" if is_live else f"last completed session, {session_date}"
    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} — 1min candles with VWAP & EMA8 ({session_note})",
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig, is_live, session_date


def _sort_group(group_df: pd.DataFrame, sort_choice: str) -> pd.DataFrame:
    if sort_choice == "Signal strength":
        return group_df.sort_values("Strength", ascending=False)
    return group_df.sort_values("Symbol")


def _render_chart_row(row: dict, token: str, badge: str):
    st.markdown(
        f"#### {badge} {row['Symbol']}  ·  LTP {row['LTP']}  ·  "
        f"VWAP {row['VWAP']}  ·  EMA8 {row['EMA8']}  ·  {row['Session']}"
    )
    fig, is_live, session_date = build_chart(row["Symbol"], row["Instrument Key"], token)
    if fig is None:
        st.info(f"No recent intraday data available for {row['Symbol']}.")
        return
    st.plotly_chart(fig, use_container_width=True, key=f"chart_{row['Symbol']}")
    if not is_live:
        st.caption(f"⏸️ Market is currently closed — showing the last completed session ({session_date}).")
    st.divider()


def render_all_charts(rows_df: pd.DataFrame, token: str, sort_choice: str):
    """Renders charts in three ordered sections, in this fixed order:
      1) ABOVE  (price above both VWAP & EMA8)
      2) BELOW  (price below both VWAP & EMA8)
      3) NEUTRAL (mixed signal)
    Within each section, charts are ordered either by signal strength
    (strongest move away from VWAP/EMA8 first) or alphabetically, per
    the sort_choice control. Sections with no matching stocks are skipped
    (no empty header shown)."""
    if rows_df.empty:
        st.caption("No stocks currently have usable chart data.")
        return

    sections = [
        ("ABOVE", "🟢 Price ABOVE VWAP & EMA8", "🟢"),
        ("BELOW", "🔴 Price BELOW VWAP & EMA8", "🔴"),
        ("NEUTRAL", "⚪ Neutral (mixed signal)", "⚪"),
    ]

    any_rendered = False
    for status_value, section_title, badge in sections:
        group = rows_df[rows_df["Status"] == status_value]
        if group.empty:
            continue
        any_rendered = True
        st.markdown(f"### {section_title} ({len(group)})")
        for _, row in _sort_group(group, sort_choice).iterrows():
            _render_chart_row(row, token, badge)

    if not any_rendered:
        st.caption("No stocks currently have usable chart data.")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main():
    st.title("📈 VWAP + EMA8 Screener")
    st.caption(
        "Above/Below tables show only stocks trading **fully above** VWAP & EMA8, "
        "or **fully below** both. Mixed (Neutral) signals are excluded from those "
        "two tables, but every stock that returned usable data still gets a chart below."
    )

    token = get_access_token()
    if not token:
        st.warning("Enter your Upstox access token in the sidebar to begin.")
        st.stop()

    with st.sidebar:
        st.divider()
        symbols_text = st.text_area(
            "Watchlist (comma separated)",
            value=", ".join(WATCHLIST),
            height=180,
        )
        symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
        run = st.button("🔍 Run Screener", type="primary", use_container_width=True)

        st.divider()
        st.subheader("⏱️ Auto-Refresh")
        auto_refresh = st.checkbox("Enable auto-refresh", value=False)
        refresh_secs = st.number_input(
            "Refresh interval (seconds)",
            min_value=5, max_value=3600, value=60, step=5,
            disabled=not auto_refresh,
        )
        if auto_refresh:
            st_autorefresh(interval=refresh_secs * 1000, key="autorefresh_timer")
            st.caption(f"Auto-refreshing every {refresh_secs}s.")

        st.divider()
        st.subheader("📊 Chart Scope")
        chart_scope = st.radio(
            "Which stocks get a chart?",
            ["All screened stocks", "Only Above/Below (skip Neutral)"],
            index=0,
            help="'All screened stocks' charts every symbol that returned usable "
                 "data, including Neutral ones. The other option only charts "
                 "stocks that made the Above or Below tables.",
        )

    if "results" not in st.session_state:
        st.session_state["results"] = pd.DataFrame()
    if "has_run" not in st.session_state:
        st.session_state["has_run"] = False

    if run or auto_refresh:
        st.session_state["results"] = screen_stocks(symbols, token)
        st.session_state["has_run"] = True
        st.session_state["last_updated"] = datetime.now().strftime("%H:%M:%S")

    if st.session_state.get("last_updated"):
        st.caption(f"🕒 Last updated: {st.session_state['last_updated']}")

    df = st.session_state["results"]

    if df.empty:
        if st.session_state["has_run"]:
            st.warning(
                "Ran the screener but got **zero usable rows**. Common causes: "
                "an expired/invalid access token (see any red error above), symbols "
                "that don't match an NSE trading symbol exactly, or no candle data "
                "returned for any symbol."
            )
        else:
            st.info("Click **Run Screener** in the sidebar to fetch live data.")
        return

    above_df = df[df["Status"] == "ABOVE"].drop(columns=["Status"])
    below_df = df[df["Status"] == "BELOW"].drop(columns=["Status"])

    if not df.empty and (df["Session"] != "Live").any():
        st.info(
            "⏸️ Market appears closed for one or more symbols — those rows show "
            "the **last completed session's** close vs. VWAP/EMA8 instead of a live price."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"🟢 Price ABOVE VWAP & EMA8 ({len(above_df)})")
        st.dataframe(
            above_df.drop(columns=["Instrument Key", "Strength"]),
            use_container_width=True, hide_index=True,
        )
    with col2:
        st.subheader(f"🔴 Price BELOW VWAP & EMA8 ({len(below_df)})")
        st.dataframe(
            below_df.drop(columns=["Instrument Key", "Strength"]),
            use_container_width=True, hide_index=True,
        )

    # ---- Charts — one per stock, no dropdown ----
    # By default this covers EVERY symbol the screener returned usable data
    # for (Above + Below + Neutral), not just the two filtered tables above.
    st.divider()
    if chart_scope == "All screened stocks":
        chart_universe = df.copy()
    else:
        chart_universe = df[df["Status"].isin(["ABOVE", "BELOW"])].copy()

    st.subheader(f"📊 Charts ({len(chart_universe)} of {len(df)} screened stocks)")

    sort_choice = st.radio("Chart order", ["Signal strength", "Alphabetical"], horizontal=True)
    render_all_charts(chart_universe, token, sort_choice)


if __name__ == "__main__":
    main()
