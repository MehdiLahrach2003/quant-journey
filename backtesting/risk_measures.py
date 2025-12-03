# backtesting/risk_measures.py
"""
Risk measures for trading strategies:
- Historical Value-at-Risk (VaR)
- Parametric Gaussian VaR
- Expected Shortfall (CVaR)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Historical VaR
# ------------------------------------------------------------
def var_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Historical Value-at-Risk at confidence level alpha.

    Parameters
    ----------
    returns : pd.Series
        Daily strategy returns (simple returns).
    alpha : float
        Tail probability (e.g., 0.05 for 95% VaR).

    Returns
    -------
    float
        Historical VaR (positive number meaning a loss).
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    # VaR = -quantile of returns
    return float(-np.quantile(r, alpha))


# ------------------------------------------------------------
# Gaussian (parametric) VaR
# ------------------------------------------------------------
def var_gaussian(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Gaussian (parametric) VaR = -[mu + sigma * z_alpha].

    Assumes returns are iid and normally distributed.
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    mu = r.mean()
    sigma = r.std()
    z = abs(np.quantile(np.random.standard_normal(1_000_000), alpha))  # good approx

    return float(-(mu - z * sigma))


# ------------------------------------------------------------
# Expected Shortfall (CVaR)
# ------------------------------------------------------------
def cvar_historical(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Expected Shortfall (CVaR): average loss beyond VaR.

    CVaR = -mean(return | return < VaR threshold)
    """
    r = returns.dropna()
    if len(r) == 0:
        return np.nan

    threshold = np.quantile(r, alpha)
    tail = r[r < threshold]

    if len(tail) == 0:
        return np.nan

    return float(-tail.mean())


# ------------------------------------------------------------
# Combined report
# ------------------------------------------------------------
def compute_risk_measures(returns: pd.Series, alpha: float = 0.05) -> dict:
    """
    Compute VaR (historical, Gaussian) and CVaR for a return series.

    Returns
    -------
    dict
        Keys: 'VaR_hist', 'VaR_gauss', 'CVaR'
    """
    return {
        "VaR_hist": var_historical(returns, alpha),
        "VaR_gauss": var_gaussian(returns, alpha),
        "CVaR": cvar_historical(returns, alpha),
    }
