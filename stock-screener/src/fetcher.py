"""
fetcher.py
----------
Downloads historical price data and fundamental metrics from Yahoo Finance.
Saves to local CSV files to avoid re-downloading on every run.

Design decisions:
- Cache to disk: yfinance has rate limits; caching avoids redundant calls.
- Separation of concerns: fetching is isolated from screening logic.
- Graceful degradation: tickers that fail are skipped with a warning.
"""

import os
import time
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PRICE_FILE = DATA_DIR / "prices.csv"
FUNDAMENTALS_FILE = DATA_DIR / "fundamentals.csv"


def fetch_prices(
    tickers: list[str],
    period: str = "2y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.

    Args:
        tickers: List of ticker symbols.
        period: yfinance period string ('1y', '2y', '5y').
        force_refresh: Re-download even if cached file exists.

    Returns:
        DataFrame with dates as index and tickers as columns.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if PRICE_FILE.exists() and not force_refresh:
        logger.info(f"Loading prices from cache: {PRICE_FILE}")
        return pd.read_csv(PRICE_FILE, index_col=0, parse_dates=True)

    logger.info(f"Downloading price data for {len(tickers)} tickers ({period})...")
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=True)

    # yfinance returns MultiIndex columns when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.to_csv(PRICE_FILE)
    logger.info(f"Saved prices to {PRICE_FILE} — shape: {prices.shape}")
    return prices


def fetch_fundamentals(
    tickers: list[str],
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Download fundamental metrics for each ticker via yfinance.

    Metrics collected:
    - trailingPE: Price-to-earnings (valuation)
    - priceToBook: Price-to-book (valuation)
    - revenueGrowth: YoY revenue growth (momentum)
    - returnOnEquity: ROE (quality)
    - debtToEquity: Leverage (risk)
    - trailingEps: Earnings per share
    - marketCap: Market capitalisation (for filtering micro-caps)

    Returns:
        DataFrame with tickers as index and metrics as columns.
    """
    DATA_DIR.mkdir(exist_ok=True)

    if FUNDAMENTALS_FILE.exists() and not force_refresh:
        logger.info(f"Loading fundamentals from cache: {FUNDAMENTALS_FILE}")
        return pd.read_csv(FUNDAMENTALS_FILE, index_col=0)

    logger.info(f"Downloading fundamentals for {len(tickers)} tickers...")
    records = []

    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info
            records.append({
                "ticker": ticker,
                "trailingPE": info.get("trailingPE"),
                "priceToBook": info.get("priceToBook"),
                "revenueGrowth": info.get("revenueGrowth"),
                "returnOnEquity": info.get("returnOnEquity"),
                "debtToEquity": info.get("debtToEquity"),
                "trailingEps": info.get("trailingEps"),
                "marketCap": info.get("marketCap"),
                "sector": info.get("sector"),
                "shortName": info.get("shortName"),
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  {i + 1}/{len(tickers)} done...")
                time.sleep(1)  # polite rate limiting
        except Exception as e:
            logger.warning(f"  Failed to fetch {ticker}: {e}")

    df = pd.DataFrame(records).set_index("ticker")
    df.to_csv(FUNDAMENTALS_FILE)
    logger.info(f"Saved fundamentals to {FUNDAMENTALS_FILE} — shape: {df.shape}")
    return df
