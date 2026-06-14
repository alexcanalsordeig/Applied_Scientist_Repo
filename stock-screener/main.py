"""
main.py
-------
End-to-end pipeline: fetch → screen → build portfolio → backtest → visualise.

Usage:
    python main.py                  # Run on sample 30 tickers (fast, for testing)
    python main.py --full           # Run on full S&P 500 (~10 min first run)
    python main.py --refresh        # Force re-download data
    python main.py --method score   # Use score-weighted portfolio
"""

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Stock Screener & Backtester")
    parser.add_argument("--full", action="store_true", help="Use full S&P 500 universe")
    parser.add_argument("--refresh", action="store_true", help="Force re-download data")
    parser.add_argument("--method", default="equal", choices=["equal", "score", "inv_vol"],
                        help="Portfolio weighting method")
    parser.add_argument("--top_n", type=int, default=15, help="Number of stocks to select")
    parser.add_argument("--period", default="2y", help="Price history period (e.g. 1y, 2y, 5y)")
    args = parser.parse_args()

    # Import here to keep top-level clean
    from src.universe import get_sp500_tickers, get_sample_tickers
    from src.fetcher import fetch_prices, fetch_fundamentals
    from src.screener import score_universe, select_portfolio
    from src.portfolio import build_portfolio
    from src.backtest import run_backtest
    from src import visualise

    # ------------------------------------------------------------------ #
    # 1. Universe
    # ------------------------------------------------------------------ #
    if args.full:
        tickers = get_sp500_tickers()
    else:
        tickers = get_sample_tickers()
        logger.info("Running on sample universe (30 tickers). Use --full for S&P 500.")

    # Always include SPY for benchmarking
    if "SPY" not in tickers:
        tickers.append("SPY")

    # ------------------------------------------------------------------ #
    # 2. Fetch data
    # ------------------------------------------------------------------ #
    prices = fetch_prices(tickers, period=args.period, force_refresh=args.refresh)
    fundamentals = fetch_fundamentals(
        [t for t in tickers if t != "SPY"],
        force_refresh=args.refresh,
    )

    # ------------------------------------------------------------------ #
    # 3. Screen universe
    # ------------------------------------------------------------------ #
    universe_prices = prices.drop(columns=["SPY"], errors="ignore")
    scores = score_universe(universe_prices, fundamentals)

    logger.info(f"\nTop 10 stocks by composite score:\n{scores.head(10)[['shortName', 'composite', 'sector']]}")

    # ------------------------------------------------------------------ #
    # 4. Select portfolio
    # ------------------------------------------------------------------ #
    selected = select_portfolio(scores, top_n=args.top_n)

    # ------------------------------------------------------------------ #
    # 5. Build portfolio weights
    # ------------------------------------------------------------------ #
    portfolio = build_portfolio(selected, universe_prices, method=args.method)

    # ------------------------------------------------------------------ #
    # 6. Backtest vs SPY
    # ------------------------------------------------------------------ #
    results = run_backtest(
        prices=prices,
        weights=portfolio["weight"],
        benchmark="SPY",
    )

    # Save metrics to JSON
    metrics_output = {
        "portfolio": results["portfolio_metrics"],
        "spy_benchmark": results["benchmark_metrics"],
    }
    Path("output").mkdir(exist_ok=True)
    with open("output/metrics.json", "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info("Metrics saved to output/metrics.json")

    # ------------------------------------------------------------------ #
    # 7. Visualise
    # ------------------------------------------------------------------ #
    visualise.run_all(results, scores, selected, portfolio)

    logger.info("\n✓ Done. Open output/ to view charts.")
    logger.info(f"  Portfolio total return : {results['portfolio_metrics']['total_return']}%")
    logger.info(f"  SPY total return       : {results['benchmark_metrics']['total_return']}%")
    logger.info(f"  Portfolio Sharpe       : {results['portfolio_metrics']['sharpe_ratio']}")
    logger.info(f"  SPY Sharpe             : {results['benchmark_metrics']['sharpe_ratio']}")


if __name__ == "__main__":
    main()
