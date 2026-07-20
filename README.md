# VWAP + EMA8 Stock Screener (Upstox + Streamlit)

Live screener for a fixed NSE watchlist. For every symbol it pulls today's
1-minute candles (for VWAP) and recent daily candles (for EMA8), then buckets
each stock as:

- 🟢 **ABOVE** — current price is above *both* VWAP and EMA8
- 🔴 **BELOW** — current price is below *both* VWAP and EMA8
- *(anything mixed is hidden, as requested)*

**Market-closed handling:** if the market hasn't traded yet today (before
9:15 AM IST, weekends, exchange holidays), the app automatically falls back
to the most recent completed session's candles instead of showing nothing.
Rows using this fallback are labeled `Last close (<date>)` in the **Session**
column, and a banner appears above the tables.

Clicking a symbol renders a TradingView-style candlestick chart with VWAP and
EMA8 plotted on top (via Plotly).

## 1. Get an Upstox access token

1. Sign up as a developer at https://developer.upstox.com/ and create an app.
2. Complete the OAuth2 login flow ato get a daily `access_token`
   (Upstox tokens expire every day — you'll need to refresh this each
   trading day, or automate the OAuth dance separately).

## 2. Run locally

```bash
git clone https://github.com/<your-username>/vwap-ema8-screener.git
cd vwap-ema8-screener

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Option A: paste token in the app sidebar at runtime (simplest)
streamlit run app.py

# Option B: store it in secrets so you don't retype it
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit .streamlit/secrets.toml and paste your token
streamlit run app.py
```

## 3. Push to GitHub

```bash
cd vwap-ema8-screener
git init
git add .
git commit -m "Initial commit: VWAP/EMA8 screener"
git branch -M main
git remote add origin https://github.com/<your-username>/vwap-ema8-screener.git
git push -u origin main
```

`.streamlit/secrets.toml` is already in `.gitignore` — your real token will
never be committed. Only `secrets.toml.example` (a blank template) goes to
GitHub.

## 4. Deploy on Streamlit Community Cloud

1. Go to https://share.streamlit.io/ and sign in with GitHub.
2. Click **New app** → pick your `vwap-ema8-screener` repo → branch `main`
   → main file `app.py`.
3. Before/after deploying, open **App settings → Secrets** and paste:
   ```toml
   UPSTOX_ACCESS_TOKEN = "your-daily-token"
   ```
4. Deploy. Because the token expires daily, update it in Secrets each
   morning (or wire up a refresh-token flow if you want it fully hands-off).

## Notes / things you may want to customize

- **Watchlist**: edit the `WATCHLIST` list in `app.py`, or type symbols
  directly into the sidebar text box at runtime. (I fixed what looked like a
  typo — `"TMPV"` → `"TATAMOTORS"` — change it back if that wasn't a typo.)
- **EMA period**: `EMA_PERIOD = 8` at the top of `app.py`.
- **VWAP window**: currently resets each day (standard intraday VWAP) using
  1-minute candles.
- **Symbol → instrument_key lookup**: uses Upstox's authenticated
  [Instrument Search API](https://upstox.com/developer/api-documentation/instrument-search)
  (`GET /v2/instruments/search`) rather than the static instrument CSV
  files — those are being phased out and have been reported blank for some
  accounts. Results are cached per symbol for a few hours each day.
- **Candle data**: uses Upstox's v3 historical/intraday candle endpoints.
- **Rate limits**: Upstox rate-limits historical/quote endpoints; the
  screener paces requests slightly and shows a progress bar. For 50 symbols
  this typically takes well under a minute.
- This app reads only market data — it does not place any orders.

## File structure

```
vwap-ema8-screener/
├── app.py                          # Streamlit app (screener + chart)
├── requirements.txt
├── .gitignore
└── .streamlit/
    └── secrets.toml.example        # copy -> secrets.toml locally
```
