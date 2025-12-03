# backtesting/var_backtest.py
"""
Backtesting of Value-at-Risk (VaR):
- Kupiec Unconditional Coverage Test (Likelihood Ratio)
- Exception counting
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import math


# ------------------------------------------------------------
# Count VaR breaches (exceptions)
# ------------------------------------------------------------
def count_exceptions(returns: pd.Series, var_series: pd.Series) -> int:
    """
    Count the number of VaR breaches: returns < -VaR.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily returns.
    var_series : pd.Series
        Daily VaR values (positive numbers representing a loss).

    Returns
    -------
    int
        Number of exceptions.
    """
    r = returns.dropna()
    v = var_series.reindex(r.index).dropna()

    # Align both
    if len(v) != len(r):
        min_len = min(len(r), len(v))
        r = r.iloc[-min_len:]
        v = v.iloc[-min_len:]

    breaches = r < -v
    return int(breaches.sum())


# ------------------------------------------------------------
# Kupiec LR test
# ------------------------------------------------------------
def kupiec_test(returns: pd.Series, var_series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Kupiec (1995) Unconditional Coverage Test.
    
    H0: The true exception probability = alpha.

    LRuc = -2 log( (1-alpha)^(T-N) * alpha^N / ( (1-p)^(T-N) * p^N ) )
    where p = N/T.

    Parameters
    ----------
    returns : pd.Series
        Strategy daily returns.
    var_series : pd.Series
        VaR time series.
    alpha : float
        Target VaR level (e.g., 0.05 for 95% VaR).

    Returns
    -------
    dict
        Keys: N (exceptions), T, p_hat, LRuc, p_value
    """
    r = returns.dropna()
    v = var_series.reindex(r.index).dropna()

    if len(v) != len(r):
        min_len = min(len(r), len(v))
        r = r.iloc[-min_len:]
        v = v.iloc[-min_len:]

    breaches = r < -v
    N = int(breaches.sum())
    T = len(r)
    if T == 0:
        return {}

    p_hat = N / T

    # Avoid log(0)
    if p_hat in (0.0, 1.0):
        return {
            "N": N,
            "T": T,
            "p_hat": p_hat,
            "LRuc": float("inf"),
            "p_value": 0.0,
        }

    num = ((1 - alpha) ** (T - N)) * (alpha ** N)
    den = ((1 - p_hat) ** (T - N)) * (p_hat ** N)
    LRuc = -2 * math.log(num / den)

    # LRuc ~ chi2(1 df)
    p_value = 1 - chi2_cdf(LRuc, df=1)

    return {
        "N": N,
        "T": T,
        "p_hat": p_hat,
        "LRuc": LRuc,
        "p_value": p_value,
    }


# ------------------------------------------------------------
# Chi-square CDF (manual small helper)
# ------------------------------------------------------------
def chi2_cdf(x: float, df: int = 1) -> float:
    """
    CDF of chi-square distribution with df degrees of freedom.
    For df = 1, CDF = erf(sqrt(x/2)).
    """
    if df != 1:
        raise NotImplementedError("Only df=1 implemented for Kupiec test.")
    return math.erf(math.sqrt(x / 2))
