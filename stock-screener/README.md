# Systematic Stock Screener & Portfolio Backtester

A data pipeline that screens S&P 500 stocks on momentum and quality signals, constructs a portfolio, and backtests it against the SPY benchmark.

Built as a personal analytical framework to support investment research — not a trading system, not financial advice.

---

## What it does

1. **Fetches** adjusted price history and fundamental data (P/E, ROE, revenue growth) for S&P 500 stocks via Yahoo Finance
2. **Scores** each stock on a composite of momentum and quality signals using percentile ranking
3. **Constructs** a portfolio of the top N stocks with configurable weighting (equal, score-weighted, or inverse volatility)
4. **Backtests** the portfolio against SPY on total return, Sharpe ratio, max drawdown, and CAGR
5. **Visualises** results as interactive Plotly charts

---

## Signal Framework

| Signal | Type | Logic |
|---|---|---|
| 12-1 Month Momentum | Price | Return from 12 months ago to 1 month ago (skips last month to avoid reversal) |
| RSI | Price | Prefer stocks near RSI 50 — not overbought, not oversold |
| Return on Equity | Fundamental | Higher ROE = better capital efficiency |
| Revenue Growth | Fundamental | YoY top-line growth |
| P/E Ratio | Fundamental | Lower is better; negative P/E excluded (loss-making) |

Each signal is **percentile-ranked** across the universe (0–100) before combining. This makes the composite score robust to outliers.

**Default weights:**
```
momentum_12_1 : 30%
rsi_score     : 15%
roe_score     : 20%
revenue_growth: 20%
pe_score      : 15%
```

---

## Outputs

All charts saved to `output/` as standalone HTML files:

| File | Chart |
|---|---|
| `00_metrics_summary.html` | Portfolio vs SPY metrics table |
| `01_cumulative_returns.html` | Growth of $1 over backtest period |
| `02_drawdown.html` | Underwater equity curve |
| `03_signal_heatmap.html` | Heatmap of signal scores per stock |
| `04_sector_allocation.html` | Portfolio sector breakdown |
| `05_holdings.html` | Individual stock weights |

---

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Quick run — 30 sample tickers (fast, good for testing)
python main.py

# Full S&P 500 run (~10 min first run, cached after)
python main.py --full

# Force re-download fresh data
python main.py --refresh

# Score-weighted portfolio instead of equal weight
python main.py --method score

# Change number of holdings
python main.py --top_n 20
```

---

## Design Decisions

**Why percentile ranking?**
Raw signal values have different scales and outliers (a P/E of 500 vs 15). Ranking converts everything to 0–100 before combining, so no single signal dominates due to scale.

**Why 12-1 month momentum?**
The 12-1 specification (excluding the most recent month) is standard in academic literature (Jegadeesh & Titman, 1993) to avoid short-term reversal contamination.

**Why equal weight as default?**
Research shows equal weight often outperforms optimised portfolios out-of-sample due to estimation error in covariance matrices (DeMiguel et al., 2009). It is also simpler and more transparent.

**Why buy-and-hold backtest?**
Avoids look-ahead bias and makes the analysis honest. The goal is to evaluate the *screening signal quality*, not to simulate a trading strategy.

---

## Caveat

Past backtest performance does not predict future returns. This is an analytical framework built for learning and research purposes, not investment advice.

---

## Stack

`Python` · `yfinance` · `pandas` · `numpy` · `plotly` · `scipy`
