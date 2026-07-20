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

import io
import time
import gzip
import requests
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

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

UPSTOX_BASE = "https://api.upstox.com/v2"
INSTRUMENTS_URL = "https://assets.upstox.com/market-quote/instruments/exchange/NSE.csv.gz"
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
# INSTRUMENT MASTER  (symbol -> instrument_key)
# --------------------------------------------------------------------------

@st.cache_data(ttl=24 * 3600, show_spinner="Downloading NSE instrument master...")
def load_instrument_master() -> pd.DataFrame:
    resp = requests.get(INSTRUMENTS_URL, timeout=30)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content)) as f:
        df = pd.read_csv(f)
    # Keep only NSE cash-market equities
    df = df[(df["exchange"] == "NSE_EQ") | (df["segment"] == "NSE_EQ")]
    return df


def resolve_instrument_keys(symbols: list[str]) -> dict:
    df = load_instrument_master()
    mapping = {}
    for sym in symbols:
        row = df[df["tradingsymbol"] == sym]
        if row.empty:
            # some symbols have a "-EQ" suffix variant in the master file
            row = df[df["tradingsymbol"] == f"{sym}-EQ"]
        if not row.empty:
            mapping[sym] = row.iloc[0]["instrument_key"]
    return mapping


# --------------------------------------------------------------------------
# DATA FETCH
# --------------------------------------------------------------------------

def fetch_intraday_1min(instrument_key: str, token: str) -> pd.DataFrame:
    """Today's 1-minute candles, used to compute running VWAP."""
    url = f"{UPSTOX_BASE}/historical-candle/intraday/{instrument_key}/1minute"
    r = requests.get(url, headers=auth_headers(token), timeout=15)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    df = pd.DataFrame(candles, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_daily_candles(instrument_key: str, token: str, lookback_days: int = 40) -> pd.DataFrame:
    """Recent daily candles, used to compute EMA(8)."""
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    url = f"{UPSTOX_BASE}/historical-candle/{instrument_key}/day/{to_date}/{from_date}"
    r = requests.get(url, headers=auth_headers(token), timeout=15)
    r.raise_for_status()
    candles = r.json().get("data", {}).get("candles", [])
    if not candles:
        return pd.DataFrame()
    cols = ["timestamp", "open", "high", "low", "close", "volume", "oi"]
    df = pd.DataFrame(candles, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fetch_ltp(instrument_keys: list[str], token: str) -> dict:
    """Batch LTP fetch. instrument_keys must be url-encoded / comma joined."""
    prices = {}
    chunk_size = 200  # Upstox allows batching; keep chunks safe
    for i in range(0, len(instrument_keys), chunk_size):
        chunk = instrument_keys[i:i + chunk_size]
        url = f"{UPSTOX_BASE}/market-quote/ltp"
        r = requests.get(url, headers=auth_headers(token), params={"instrument_key": ",".join(chunk)}, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", {})
        for _, v in data.items():
            prices[v["instrument_token"]] = v["last_price"]
    return prices


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
    key_map = resolve_instrument_keys(symbols)
    rows = []
    progress = st.progress(0.0, text="Screening...")
    n = len(symbols)

    for i, sym in enumerate(symbols):
        instrument_key = key_map.get(sym)
        if not instrument_key:
            progress.progress((i + 1) / n)
            continue
        try:
            intraday = fetch_intraday_1min(instrument_key, token)
            daily = fetch_daily_candles(instrument_key, token)
            if intraday.empty:
                progress.progress((i + 1) / n)
                continue

            ltp = float(intraday["close"].iloc[-1])
            vwap = compute_vwap(intraday)
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
    intraday = fetch_intraday_1min(instrument_key, token)
    if intraday.empty:
        st.info("No intraday data available for chart.")
        return

    typical_price = (intraday["high"] + intraday["low"] + intraday["close"]) / 3
    cum_pv = (typical_price * intraday["volume"]).cumsum()
    cum_vol = intraday["volume"].cumsum()
    intraday["vwap"] = cum_pv / cum_vol.replace(0, np.nan)
    intraday["ema8"] = intraday["close"].ewm(span=EMA_PERIOD, adjust=False).mean()

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=intraday["timestamp"], open=intraday["open"], high=intraday["high"],
        low=intraday["low"], close=intraday["close"], name=symbol,
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    fig.add_trace(go.Scatter(
        x=intraday["timestamp"], y=intraday["vwap"], name="VWAP",
        line=dict(color="#ff9800", width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=intraday["timestamp"], y=intraday["ema8"], name="EMA8",
        line=dict(color="#2962ff", width=1.6),
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{symbol} — 1min candles with VWAP & EMA8",
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


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

    if run:
        st.session_state["results"] = screen_stocks(symbols, token)

    df = st.session_state["results"]

    if df.empty:
        st.info("Click **Run Screener** in the sidebar to fetch live data.")
        return

    above_df = df[df["Status"] == "ABOVE"].drop(columns=["Status"])
    below_df = df[df["Status"] == "BELOW"].drop(columns=["Status"])

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
