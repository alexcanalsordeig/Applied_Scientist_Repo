# Signal — Systematic Stock Screener & Portfolio Backtester

An interactive web app that screens the S&P 500 on momentum and quality factors, constructs a portfolio, and backtests it against the SPY benchmark.

Built as a personal research framework — a repeatable way to evaluate stock selection signals, **not** a price predictor or investment advice.

---

## Live App

Run the interactive dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard lets you adjust the universe size, number of holdings, and weighting method, then see the resulting portfolio, signal breakdown, sector allocation, and backtest — all updating live.

> *Deploying on Streamlit Cloud gives a public URL — see Deployment below.*

---

## What it does

1. **Fetches** live adjusted prices and fundamental data (P/E, ROE, revenue growth) for the full S&P 500 via Yahoo Finance
2. **Scores** every stock on a composite of momentum and quality signals using percentile ranking
3. **Constructs** a portfolio of the top N stocks with configurable weighting (equal, score-weighted, or inverse volatility)
4. **Backtests** the portfolio against SPY on total return, Sharpe ratio, max drawdown, and CAGR
5. **Visualises** everything in an interactive Streamlit dashboard

---

## Two ways to run

**Interactive app (recommended):**
```bash
streamlit run app.py
```

**Command-line pipeline (generates static HTML charts in `output/`):**
```bash
python main.py            # 30-stock sample, fast
python main.py --full     # full S&P 500
python main.py --refresh  # force fresh data download
```

---

## Signal Framework

| Signal | Type | Logic |
|---|---|---|
| 12-1 Month Momentum | Price | Return from 12 months ago to 1 month ago (skips last month to avoid reversal) |
| RSI | Price | Prefer stocks near RSI 50 — not overbought, not oversold |
| Return on Equity | Fundamental | Higher ROE = better capital efficiency |
| Revenue Growth | Fundamental | YoY top-line growth |
| P/E Ratio | Fundamental | Lower is better; negative P/E excluded (loss-making) |

Each signal is **percentile-ranked** across the universe (0–100) before combining. This makes the composite score robust to outliers. Missing data is ranked last, never rewarded.

**Default weights:** Momentum 30% · RSI 15% · ROE 20% · Revenue Growth 20% · P/E 15%

---

## Project Structure

```
stock-screener/
├── app.py              # Streamlit dashboard (interactive front-end)
├── main.py             # CLI pipeline (static chart generation)
├── src/
│   ├── universe.py     # S&P 500 ticker list
│   ├── fetcher.py      # Cached data layer (prices + fundamentals)
│   ├── screener.py     # Composite factor scoring
│   ├── portfolio.py    # Portfolio construction & weighting
│   ├── backtest.py     # Performance metrics vs benchmark
│   └── visualise.py    # Plotly chart generation
├── .streamlit/         # Dark theme config
└── requirements.txt
```

---

## Design Decisions

**Why percentile ranking?** Raw signal values have different scales and outliers (a P/E of 500 vs 15). Ranking converts everything to 0–100 before combining, so no single signal dominates due to scale.

**Why 12-1 month momentum?** The 12-1 specification (excluding the most recent month) is standard in academic literature (Jegadeesh & Titman, 1993) to avoid short-term reversal contamination.

**Why equal weight as default?** Research shows equal weight often outperforms optimised portfolios out-of-sample due to estimation error (DeMiguel et al., 2009). It is also simpler and more transparent.

**Why buy-and-hold backtest?** Avoids look-ahead bias and keeps the analysis honest. The goal is to evaluate *signal quality*, not to simulate live trading. A production version would use rolling rebalancing with out-of-sample windows.

---

## Deployment

Deploy free on [Streamlit Cloud](https://share.streamlit.io):
1. Push this repo to GitHub
2. Sign in at share.streamlit.io with GitHub
3. New app → select this repo → set main file path to `stock-screener/app.py`
4. Deploy — you get a public URL

---

## Caveat

Past backtest performance does not predict future returns. This is an analytical framework built for research and learning, not investment advice.

---

## Stack

`Python` · `Streamlit` · `yfinance` · `pandas` · `numpy` · `plotly` · `scipy`