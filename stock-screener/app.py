"""
app.py
------
Streamlit front-end for the systematic stock screener.

Run locally:
    streamlit run app.py

Deploy free on Streamlit Cloud:
    1. Push this repo to GitHub
    2. Go to share.streamlit.io
    3. Point it at this repo + app.py
    4. It auto-deploys and gives you a public URL

What it shows:
- Today's top-ranked stocks by composite signal score
- Interactive controls: universe size, number of holdings, weighting method
- Signal breakdown per stock
- Backtest of the selected portfolio vs SPY
- Honest caveat banner — this is research tooling, not investment advice
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.universe import get_sp500_tickers, get_sample_tickers
from src.fetcher import fetch_prices, fetch_fundamentals
from src.screener import score_universe, select_portfolio
from src.portfolio import build_portfolio
from src.backtest import run_backtest

logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------- #
# Page config
# ---------------------------------------------------------------------- #
st.set_page_config(
    page_title="Signal — S&P 500 Stock Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark styling to match the portfolio charts
st.markdown("""
<style>
    .stApp { background-color: #0e0f11; }
    h1, h2, h3 { color: #e8eaf0; }
    .metric-card {
        background: #16181c;
        border: 1px solid #2a2d35;
        border-radius: 12px;
        padding: 16px;
    }
    .caveat {
        background: rgba(79,142,247,0.08);
        border: 1px solid rgba(79,142,247,0.25);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------- #
# Data loading (cached)
# ---------------------------------------------------------------------- #
@st.cache_data(ttl=60 * 60 * 12)  # cache for 12 hours
def load_data(full_universe: bool, refresh: bool):
    """Fetch prices + fundamentals. Cached to avoid re-downloading."""
    if full_universe:
        tickers = get_sp500_tickers()
    else:
        tickers = get_sample_tickers()

    if "SPY" not in tickers:
        tickers = tickers + ["SPY"]

    prices = fetch_prices(tickers, period="2y", force_refresh=refresh)
    fundamentals = fetch_fundamentals(
        [t for t in tickers if t != "SPY"],
        force_refresh=refresh,
    )
    return prices, fundamentals


# ---------------------------------------------------------------------- #
# Sidebar controls
# ---------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")

full_universe = st.sidebar.toggle("Full S&P 500", value=True,
                                  help="Off = 30-stock sample (faster)")
top_n = st.sidebar.slider("Number of holdings", 5, 30, 15)
method = st.sidebar.selectbox(
    "Weighting method",
    ["equal", "score", "inv_vol"],
    format_func=lambda m: {
        "equal": "Equal weight",
        "score": "Score-weighted",
        "inv_vol": "Inverse volatility",
    }[m],
)

refresh = st.sidebar.button("🔄 Refresh data", help="Download fresh prices & fundamentals")

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ---------------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------------- #
st.title("📊 Signal")
st.markdown("##### Systematic S&P 500 screener — momentum + quality factor model")

st.markdown("""
<div class="caveat">
<strong>What this is:</strong> a repeatable screening framework that ranks stocks on momentum and
quality signals, then backtests the resulting portfolio against SPY. <strong>What this is not:</strong>
a prediction of tomorrow's price or investment advice. Past performance does not predict future returns.
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ---------------------------------------------------------------------- #
# Load + compute
# ---------------------------------------------------------------------- #
with st.spinner("Loading data and scoring universe..."):
    prices, fundamentals = load_data(full_universe, refresh)
    universe_prices = prices.drop(columns=["SPY"], errors="ignore")
    scores = score_universe(universe_prices, fundamentals)
    selected = select_portfolio(scores, top_n=top_n)
    portfolio = build_portfolio(selected, universe_prices, method=method)
    results = run_backtest(prices, portfolio["weight"], benchmark="SPY")

pm = results["portfolio_metrics"]
bm = results["benchmark_metrics"]

# ---------------------------------------------------------------------- #
# Top metrics row
# ---------------------------------------------------------------------- #
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Return", f"{pm['total_return']}%", f"{round(pm['total_return'] - bm['total_return'], 1)}% vs SPY")
c2.metric("CAGR", f"{pm['cagr']}%", f"{round(pm['cagr'] - bm['cagr'], 1)}% vs SPY")
c3.metric("Sharpe Ratio", f"{pm['sharpe_ratio']}", f"{round(pm['sharpe_ratio'] - bm['sharpe_ratio'], 2)} vs SPY")
c4.metric("Max Drawdown", f"{pm['max_drawdown']}%", f"{round(pm['max_drawdown'] - bm['max_drawdown'], 1)}% vs SPY",
          delta_color="inverse")

st.markdown("---")

# ---------------------------------------------------------------------- #
# Two columns: picks table + cumulative chart
# ---------------------------------------------------------------------- #
left, right = st.columns([1, 1.3])

with left:
    st.subheader("📋 Today's Top Picks")
    display = portfolio[["shortName", "sector", "composite", "weight_pct"]].copy()
    display.columns = ["Company", "Sector", "Score", "Weight %"]
    display["Score"] = display["Score"].round(1)
    display = display.reset_index().rename(columns={"index": "Ticker"})
    st.dataframe(display, width='stretch', height=560, hide_index=True)

with right:
    st.subheader("📈 Backtest vs SPY")
    cum_port = results["cumulative_portfolio"]
    cum_bench = results["cumulative_benchmark"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values,
        name=f"Portfolio (Sharpe {pm['sharpe_ratio']})",
        line=dict(color="#4f8ef7", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=cum_bench.index, y=cum_bench.values,
        name=f"SPY (Sharpe {bm['sharpe_ratio']})",
        line=dict(color="#6b7280", width=1.5, dash="dot"),
    ))
    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis_title="Growth of $1",
    )
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# ---------------------------------------------------------------------- #
# Bottom row: signal heatmap + sector pie
# ---------------------------------------------------------------------- #
b_left, b_right = st.columns([1.4, 1])

with b_left:
    st.subheader("🔬 Signal Breakdown")
    signal_cols = ["momentum_12_1", "rsi_score", "roe_score", "revenue_growth", "pe_score"]
    available = [c for c in signal_cols if c in scores.columns]
    sel_scores = scores.loc[selected.index, available]
    labels = scores.loc[selected.index, "shortName"].fillna(pd.Series(selected.index, index=selected.index))

    heat = px.imshow(
        sel_scores.values,
        x=[c.replace("_", " ").title() for c in available],
        y=labels.values,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100, aspect="auto",
    )
    heat.update_layout(
        template="plotly_dark", height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(heat, width='stretch')

with b_right:
    st.subheader("🥧 Sector Allocation")
    sector_weights = portfolio.groupby("sector")["weight"].sum().reset_index()
    pie = px.pie(
        sector_weights, names="sector", values="weight",
        color_discrete_sequence=px.colors.qualitative.Set3,
        hole=0.4,
    )
    pie.update_layout(
        template="plotly_dark", height=480,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=True,
    )
    st.plotly_chart(pie, width='stretch')

st.markdown("---")
st.caption(
    "Signal weights: Momentum 30% · RSI 15% · ROE 20% · Revenue Growth 20% · P/E 15%. "
    "Backtest is buy-and-hold over the displayed period. Built with Python, yfinance, pandas, and Plotly."
)
