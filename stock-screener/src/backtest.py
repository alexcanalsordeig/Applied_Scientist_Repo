"""
backtest.py
-----------
Backtests portfolio performance against the SPY benchmark.

Methodology:
- Buy-and-hold: weights are set at the start and held for the full period.
  This is intentional — it avoids look-ahead bias and keeps the analysis
  honest. We are NOT claiming to trade this strategy in real time.
- All returns are based on adjusted closing prices (splits + dividends).
- Performance metrics follow industry standard definitions.

Important caveat (always state this):
  Past backtest performance does not predict future returns.
  This is an analytical framework, not investment advice.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


def compute_portfolio_returns(
    prices: pd.DataFrame,
    weights: pd.Series,
    start_date: str = None,
) -> pd.Series:
    """
    Compute daily portfolio returns given prices and weights.

    Args:
        prices: DataFrame of adjusted close prices.
        weights: Series of portfolio weights (must sum to ~1).
        start_date: Optional start date string ('YYYY-MM-DD').

    Returns:
        Daily portfolio return series.
    """
    tickers = weights.index.tolist()
    port_prices = prices[tickers].copy()

    if start_date:
        port_prices = port_prices[port_prices.index >= start_date]

    # Forward-fill missing prices (holidays, halted stocks)
    port_prices = port_prices.ffill().dropna(how="all")

    # Daily returns
    returns = port_prices.pct_change().dropna()

    # Weighted portfolio return
    port_returns = (returns * weights).sum(axis=1)
    port_returns.name = "portfolio"
    return port_returns


def compute_benchmark_returns(
    prices: pd.DataFrame,
    benchmark: str = "SPY",
    start_date: str = None,
) -> pd.Series:
    """
    Extract benchmark returns from the prices DataFrame.
    SPY must be included in the downloaded tickers.
    """
    if benchmark not in prices.columns:
        raise ValueError(
            f"Benchmark '{benchmark}' not found in prices. "
            "Add it to your ticker list in main.py."
        )
    bench = prices[[benchmark]].copy()
    if start_date:
        bench = bench[bench.index >= start_date]
    bench_returns = bench[benchmark].pct_change().dropna()
    bench_returns.name = benchmark
    return bench_returns


def compute_metrics(returns: pd.Series) -> dict:
    """
    Compute standard portfolio performance metrics.

    Metrics:
    - Total return: cumulative return over the full period
    - CAGR: compound annual growth rate
    - Sharpe ratio: risk-adjusted return (assumes risk-free rate = 0)
    - Max drawdown: largest peak-to-trough decline
    - Volatility: annualised standard deviation of daily returns
    - Calmar ratio: CAGR / abs(max drawdown) — risk-adjusted return
    """
    cumulative = (1 + returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    n_years = len(returns) / TRADING_DAYS_PER_YEAR

    cagr = (1 + total_return) ** (1 / n_years) - 1
    volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = cagr / volatility if volatility > 0 else np.nan

    # Max drawdown
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan

    return {
        "total_return": round(total_return * 100, 2),
        "cagr": round(cagr * 100, 2),
        "volatility": round(volatility * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown * 100, 2),
        "calmar_ratio": round(calmar, 3),
    }


def run_backtest(
    prices: pd.DataFrame,
    weights: pd.Series,
    benchmark: str = "SPY",
    start_date: str = None,
) -> dict:
    """
    Run full backtest and return results dict.

    Returns:
        {
          'portfolio_returns': pd.Series,
          'benchmark_returns': pd.Series,
          'cumulative_portfolio': pd.Series,
          'cumulative_benchmark': pd.Series,
          'portfolio_metrics': dict,
          'benchmark_metrics': dict,
        }
    """
    port_returns = compute_portfolio_returns(prices, weights, start_date)
    bench_returns = compute_benchmark_returns(prices, benchmark, start_date)

    # Align on common dates
    common_idx = port_returns.index.intersection(bench_returns.index)
    port_returns = port_returns.loc[common_idx]
    bench_returns = bench_returns.loc[common_idx]

    cum_port = (1 + port_returns).cumprod()
    cum_bench = (1 + bench_returns).cumprod()

    port_metrics = compute_metrics(port_returns)
    bench_metrics = compute_metrics(bench_returns)

    logger.info("\n=== BACKTEST RESULTS ===")
    logger.info(f"{'Metric':<20} {'Portfolio':>12} {'SPY':>12}")
    logger.info("-" * 46)
    for key in port_metrics:
        logger.info(f"{key:<20} {str(port_metrics[key]):>12} {str(bench_metrics[key]):>12}")

    return {
        "portfolio_returns": port_returns,
        "benchmark_returns": bench_returns,
        "cumulative_portfolio": cum_port,
        "cumulative_benchmark": cum_bench,
        "portfolio_metrics": port_metrics,
        "benchmark_metrics": bench_metrics,
    }
