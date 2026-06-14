"""
portfolio.py
------------
Constructs portfolio weights from screener output.

Weighting schemes supported:
- Equal weight: simplest, most robust out-of-sample
- Score weight: allocate more to higher-scoring stocks
- Inverse volatility: weight by 1/vol to reduce risk concentration

Equal weight is the default — research shows it often outperforms
more complex schemes out-of-sample due to estimation error in
covariance matrices (DeMiguel et al., 2009).
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def equal_weight(selected: pd.DataFrame) -> pd.Series:
    """Allocate equal weight to each selected stock."""
    n = len(selected)
    return pd.Series(1 / n, index=selected.index, name="weight")


def score_weight(selected: pd.DataFrame) -> pd.Series:
    """
    Allocate weight proportional to composite score.
    Higher-scoring stocks get more capital.
    """
    scores = selected["composite"]
    weights = scores / scores.sum()
    return weights.rename("weight")


def inverse_volatility_weight(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.Series:
    """
    Allocate weight inversely proportional to recent volatility.
    Less volatile stocks get more weight — reduces risk concentration.

    Args:
        lookback: Number of trading days to estimate volatility.
    """
    tickers = selected.index.tolist()
    recent_prices = prices[tickers].iloc[-lookback:]
    returns = recent_prices.pct_change().dropna()
    vol = returns.std()
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum()
    return weights.rename("weight")


def build_portfolio(
    selected: pd.DataFrame,
    prices: pd.DataFrame,
    method: str = "equal",
) -> pd.DataFrame:
    """
    Build portfolio with weights.

    Args:
        selected: Output of screener.select_portfolio().
        prices: Full price DataFrame (for vol estimation).
        method: 'equal', 'score', or 'inv_vol'.

    Returns:
        DataFrame with tickers, weights, and metadata.
    """
    if method == "equal":
        weights = equal_weight(selected)
    elif method == "score":
        weights = score_weight(selected)
    elif method == "inv_vol":
        weights = inverse_volatility_weight(selected, prices)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    portfolio = selected[["shortName", "sector", "composite", "momentum_12_1_raw",
                           "roe_raw", "revenue_growth_raw", "pe_raw"]].copy()
    portfolio["weight"] = weights
    portfolio["weight_pct"] = (weights * 100).round(2)

    logger.info(f"Portfolio built with {method} weighting:")
    logger.info(f"\n{portfolio[['shortName', 'weight_pct', 'sector']].to_string()}")
    return portfolio
