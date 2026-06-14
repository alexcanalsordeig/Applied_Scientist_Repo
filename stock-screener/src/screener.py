"""
screener.py
-----------
Scores and ranks stocks on a composite of momentum and quality signals.

Signal framework:
- Momentum (price-based): 12-1 month return, RSI
- Quality (fundamental): ROE, revenue growth
- Valuation (fundamental): P/E, P/B (lower = better, penalised if too high)

Each signal is ranked percentile-style (0–100) across the universe.
The composite score is a weighted average — weights are explicit and adjustable.

Design note: ranking rather than using raw values makes the score robust to
outliers (a P/E of 500 doesn't blow up the score, it just ranks last).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Signal weights — must sum to 1.0
WEIGHTS = {
    "momentum_12_1": 0.30,
    "rsi_score":     0.15,
    "roe_score":     0.20,
    "revenue_growth":0.20,
    "pe_score":      0.15,
}


def compute_momentum(prices: pd.DataFrame) -> pd.Series:
    if len(prices) < 252:
        logger.warning("Less than 1 year of price data — momentum signal may be noisy.")
    ret_12m = prices.iloc[-252] if len(prices) >= 252 else prices.iloc[0]
    ret_1m = prices.iloc[-21]
    momentum = (ret_1m - ret_12m) / ret_12m
    return momentum


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1]


def rank_signal(series: pd.Series, ascending: bool = True) -> pd.Series:
    """
    Rank signal values as percentiles (0-100).
    NaNs are always ranked LAST (score = 0) — never rewarded for missing data.
    """
    ranked = series.rank(ascending=ascending, pct=True, na_option="bottom") * 100
    # Explicitly set NaN raw values to 0 score
    ranked[series.isna()] = 0.0
    return ranked


def score_universe(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    scores = pd.DataFrame(index=prices.columns)

    # --- Momentum signals ---
    momentum = compute_momentum(prices)
    scores["momentum_12_1_raw"] = momentum
    scores["momentum_12_1"] = rank_signal(momentum, ascending=True)

    rsi = compute_rsi(prices)
    rsi_distance = (rsi - 50).abs()
    scores["rsi_raw"] = rsi
    scores["rsi_score"] = rank_signal(rsi_distance, ascending=False)

    # --- Fundamental signals ---
    fundamentals_clean = fundamentals[~fundamentals.index.duplicated(keep="first")]
    fund = fundamentals_clean.reindex(scores.index)

    scores["roe_raw"] = fund["returnOnEquity"]
    scores["roe_score"] = rank_signal(fund["returnOnEquity"], ascending=True)

    scores["revenue_growth_raw"] = fund["revenueGrowth"]
    scores["revenue_growth"] = rank_signal(fund["revenueGrowth"], ascending=True)

    pe = fund["trailingPE"].where(fund["trailingPE"] > 0)
    scores["pe_raw"] = pe
    scores["pe_score"] = rank_signal(pe, ascending=False)

    # --- Composite score ---
    scores["composite"] = sum(
        scores[signal] * weight
        for signal, weight in WEIGHTS.items()
    )

    # Attach metadata
    scores["sector"] = fund["sector"]
    scores["shortName"] = fund["shortName"]
    scores["marketCap"] = fund["marketCap"]

    # Drop tickers with no price data (delisted / failed downloads)
    scores = scores.dropna(subset=["momentum_12_1_raw"])

    return scores.sort_values("composite", ascending=False)


def select_portfolio(
    scores: pd.DataFrame,
    top_n: int = 15,
    min_market_cap: float = 2e9,
) -> pd.DataFrame:
    filtered = scores.dropna(subset=["composite"])

    if min_market_cap:
        filtered = filtered[
            filtered["marketCap"].fillna(0) >= min_market_cap
        ]

    selected = filtered.head(top_n)
    logger.info(f"Selected {len(selected)} stocks from {len(scores)} universe.")
    return selected