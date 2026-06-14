"""
visualise.py
------------
Generates interactive Plotly charts for the portfolio analysis.
All charts saved as standalone HTML files — no server required.

Charts produced:
1. Cumulative returns: portfolio vs SPY benchmark
2. Drawdown chart: underwater equity curve
3. Signal heatmap: stock scores across all signals
4. Sector allocation: pie chart of portfolio sector weights
5. Portfolio holdings: bar chart of individual weights
"""

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("output")


def save(fig, filename: str):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.write_html(str(path))
    logger.info(f"Saved: {path}")


def plot_cumulative_returns(results: dict):
    """Portfolio vs SPY cumulative return chart."""
    cum_port = results["cumulative_portfolio"]
    cum_bench = results["cumulative_benchmark"]
    pm = results["portfolio_metrics"]
    bm = results["benchmark_metrics"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port.values,
        name=f"Portfolio (Sharpe: {pm['sharpe_ratio']})",
        line=dict(color="#4f8ef7", width=2.5),
        hovertemplate="%{x}<br>Portfolio: %{y:.3f}x<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=cum_bench.index, y=cum_bench.values,
        name=f"SPY Benchmark (Sharpe: {bm['sharpe_ratio']})",
        line=dict(color="#6b7280", width=1.5, dash="dot"),
        hovertemplate="%{x}<br>SPY: %{y:.3f}x<extra></extra>",
    ))

    # Annotation: total return labels
    fig.add_annotation(
        x=cum_port.index[-1], y=cum_port.values[-1],
        text=f"+{pm['total_return']}%",
        showarrow=False, xanchor="left", font=dict(color="#4f8ef7", size=12)
    )
    fig.add_annotation(
        x=cum_bench.index[-1], y=cum_bench.values[-1],
        text=f"+{bm['total_return']}%",
        showarrow=False, xanchor="left", font=dict(color="#6b7280", size=12)
    )

    fig.update_layout(
        title="Cumulative Returns: Portfolio vs SPY",
        xaxis_title="Date", yaxis_title="Growth of $1",
        template="plotly_dark",
        legend=dict(x=0.01, y=0.99),
        hovermode="x unified",
    )
    save(fig, "01_cumulative_returns.html")


def plot_drawdown(results: dict):
    """Underwater equity curve (drawdown over time)."""
    for name, returns, color in [
        ("Portfolio", results["portfolio_returns"], "#ef4444"),
        ("SPY", results["benchmark_returns"], "#6b7280"),
    ]:
        cum = (1 + returns).cumprod()
        rolling_max = cum.cummax()
        drawdown = (cum - rolling_max) / rolling_max * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=drawdown.index, y=drawdown.values,
            fill="tozeroy", name=name,
            line=dict(color=color),
            hovertemplate="%{x}<br>Drawdown: %{y:.1f}%<extra></extra>",
        ))

    fig.update_layout(
        title="Drawdown Chart",
        xaxis_title="Date", yaxis_title="Drawdown (%)",
        template="plotly_dark",
        hovermode="x unified",
    )
    save(fig, "02_drawdown.html")


def plot_signal_heatmap(scores: pd.DataFrame, selected: pd.DataFrame):
    """Heatmap of signal scores for selected stocks."""
    signal_cols = ["momentum_12_1", "rsi_score", "roe_score", "revenue_growth", "pe_score"]
    available = [c for c in signal_cols if c in scores.columns]

    sel_scores = scores.loc[selected.index, available]
    labels = scores.loc[selected.index, "shortName"].fillna(selected.index.to_series())

    fig = px.imshow(
        sel_scores.values,
        x=[c.replace("_", " ").title() for c in available],
        y=labels.values,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100,
        aspect="auto",
        title="Signal Score Heatmap (0 = worst, 100 = best)",
    )
    fig.update_layout(template="plotly_dark")
    save(fig, "03_signal_heatmap.html")


def plot_sector_allocation(portfolio: pd.DataFrame):
    """Pie chart of sector weights."""
    sector_weights = portfolio.groupby("sector")["weight"].sum().reset_index()

    fig = px.pie(
        sector_weights,
        names="sector", values="weight",
        title="Portfolio Sector Allocation",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig.update_layout(template="plotly_dark")
    save(fig, "04_sector_allocation.html")


def plot_holdings(portfolio: pd.DataFrame):
    """Horizontal bar chart of individual stock weights."""
    df = portfolio.sort_values("weight_pct")
    labels = df["shortName"].fillna(df.index.to_series())

    fig = go.Figure(go.Bar(
        x=df["weight_pct"].values,
        y=labels.values,
        orientation="h",
        marker_color="#4f8ef7",
        text=[f"{w:.1f}%" for w in df["weight_pct"].values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Portfolio Holdings by Weight",
        xaxis_title="Weight (%)",
        template="plotly_dark",
        height=max(400, len(df) * 30),
    )
    save(fig, "05_holdings.html")


def plot_metrics_summary(results: dict):
    """Side-by-side metrics comparison table."""
    pm = results["portfolio_metrics"]
    bm = results["benchmark_metrics"]

    metrics = list(pm.keys())
    port_vals = list(pm.values())
    bench_vals = list(bm.values())

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Metric", "Portfolio", "SPY"],
            fill_color="#1e2128",
            font=dict(color="white", size=13),
            align="left",
        ),
        cells=dict(
            values=[metrics, port_vals, bench_vals],
            fill_color=[["#16181c"] * len(metrics)],
            font=dict(color="white", size=12),
            align="left",
        ),
    )])
    fig.update_layout(title="Performance Summary", template="plotly_dark")
    save(fig, "00_metrics_summary.html")


def run_all(results: dict, scores: pd.DataFrame, selected: pd.DataFrame, portfolio: pd.DataFrame):
    """Generate all charts."""
    logger.info("Generating visualisations...")
    plot_metrics_summary(results)
    plot_cumulative_returns(results)
    plot_drawdown(results)
    plot_signal_heatmap(scores, selected)
    plot_sector_allocation(portfolio)
    plot_holdings(portfolio)
    logger.info(f"All charts saved to {OUTPUT_DIR}/")
