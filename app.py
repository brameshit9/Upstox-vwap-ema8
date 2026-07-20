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
  NEUTRAL  -> mixed (price above one, below the other) -> NOT shown anywhere
              (per your requirement: only pure above-both or pure below-both)

A TradingView-style candlestick chart (Plotly) with VWAP and EMA8 overlaid
is shown when you click a row / pick a symbol from the dropdown.

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

import time
import requests
import numpy as np
import pandas as pd
import streamlit as st
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
# Uses Upstox's authenticated Instrument Search API instead of downloading
# the static NSE.csv.gz file — that file has been reported blank/stale for
# some accounts, and Upstox is deprecating the CSV format in favour of this
# search endpoint. https://api.upstox.com/v2/instruments/search
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


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_search(symbol: str, _token: str, cache_bust: str):
    """Returns (instrument_key_or_None, error_message_or_None).
    `_token` is excluded from Streamlit's cache key (leading underscore);
    `cache_bust` (today's date) naturally expires the cache each day since
    Upstox tokens/instrument sets refresh daily anyway."""
    try:
        key = _search_instrument(symbol, _token)
        if key:
            return key, None
        return None, "no matching NSE equity instrument found"
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else "?"
        if status == 401:
            return None, "401 Unauthorized — access token is invalid or expired"
        return None, f"HTTP {status} error"
    except requests.RequestException as e:
        return None, f"network error ({e})"


def resolve_instrument_keys(symbols: list[str], token: str):
    """Returns (mapping, errors). Fails fast on the first request if it's a
    401, since that means every subsequent symbol will fail the same way."""
    today = datetime.now().strftime("%Y-%m-%d")
    mapping, errors = {}, []

    for sym in symbols:
        key, err = _cached_search(sym, token, today)
        if key:
            mapping[sym] = key
        elif err:
            errors.append(f"{sym}: {err}")
            if "401" in err:
                # Token is dead for every remaining symbol too — stop burning calls.
                break

    return mapping, errors


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
    key_map, resolve_errors = resolve_instrument_keys(symbols, token)

    if resolve_errors:
        auth_failed = any("401" in e for e in resolve_errors)
        if auth_failed:
            st.error(
                "🔒 Upstox rejected the access token (401 Unauthorized). "
                "Tokens expire daily — generate a fresh one at "
                "[developer.upstox.com](https://developer.upstox.com/) and paste it in the sidebar."
            )
        with st.expander(f"⚠️ {len(resolve_errors)} symbol(s) could not be resolved — click for details"):
            for e in resolve_errors:
                st.write("•", e)

    rows = []
    progress = st.progress(0.0, text="Screening...")
    n = len(symbols)

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
                status = "NEUTRAL"  # skipped from display per requirement

            rows.append({
                "Symbol": sym,
                "Instrument Key": instrument_key,
                "LTP": round(ltp, 2),
                "VWAP": round(vwap, 2),
                "EMA8": round(ema8, 2),
                "Status": status,
                "Session": "Live" if is_live else f"Last close ({session_date})",
            })
        except requests.HTTPError as e:
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

def render_chart(symbol: str, instrument_key: str, token: str):
    price_df, is_live, session_date = fetch_price_series(instrument_key, token)
    if price_df.empty:
        st.info("No recent intraday data available for this instrument.")
        return

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
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)
    if not is_live:
        st.caption(f"⏸️ Market is currently closed — showing the last completed session ({session_date}).")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

def main():
    st.title("📈 VWAP + EMA8 Screener")
    st.caption(
        "Shows only stocks trading **fully above** VWAP & EMA8, or **fully below** "
        "both. Mixed signals are hidden."
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

    if "results" not in st.session_state:
        st.session_state["results"] = pd.DataFrame()
    if "has_run" not in st.session_state:
        st.session_state["has_run"] = False

    if run:
        st.session_state["results"] = screen_stocks(symbols, token)
        st.session_state["has_run"] = True

    df = st.session_state["results"]

    if df.empty:
        if st.session_state["has_run"]:
            st.warning(
                "Ran the screener but got **zero usable rows**. Common causes: "
                "an expired/invalid access token (see any red error above), symbols "
                "that don't match an NSE trading symbol exactly, or every stock "
                "currently sitting in the 'mixed' zone (above one indicator, below "
                "the other) which is hidden by design."
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
        st.dataframe(above_df.drop(columns=["Instrument Key"]), use_container_width=True, hide_index=True)
    with col2:
        st.subheader(f"🔴 Price BELOW VWAP & EMA8 ({len(below_df)})")
        st.dataframe(below_df.drop(columns=["Instrument Key"]), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("📊 Chart")
    chart_candidates = pd.concat([above_df, below_df])["Symbol"].tolist()
    if chart_candidates:
        picked = st.selectbox("Pick a symbol to view chart", chart_candidates)
        row = df[df["Symbol"] == picked].iloc[0]
        render_chart(picked, row["Instrument Key"], token)
    else:
        st.caption("No stocks currently meet the above/below criteria.")


if __name__ == "__main__":
    main()
