# backtesting/var_cornish.py
"""
Cornish–Fisher adjusted Value-at-Risk (VaR) to account for skewness and kurtosis.

This improves on Gaussian VaR by expanding the quantile using a polynomial
correction based on higher-order moments.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import math


def _compute_moments(returns: pd.Series) -> tuple[float, float, float]:
    """
    Compute mean, skewness, and excess kurtosis of returns.

    Returns
    -------
    (mu, skew, kurt_excess)
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan, np.nan

    mu = r.mean()
    sigma = r.std()

    if sigma == 0 or len(r) < 4:
        return mu, 0.0, 0.0  # no higher moments possible

    # Standardized returns
    z = (r - mu) / sigma

    skew = float((z**3).mean())
    kurt_excess = float((z**4).mean() - 3.0)  # excess kurtosis

    return mu, skew, kurt_excess


def cornish_fisher_var(returns: pd.Series, alpha: float = 0.05) -> float:
    """
    Cornish–Fisher VaR correction.

    Formula:
        z_cf = z
               + (1/6)(z^2 - 1) * skew
               + (1/24)(z^3 - 3z) * kurt_excess
               - (1/36)(2z^3 - 5z) * skew^2

    VaR = -(mu + sigma * z_cf)

    Parameters
    ----------
    returns : pd.Series
        Strategy simple returns.
    alpha : float
        Tail probability.

    Returns
    -------
    float
        Cornish–Fisher VaR (positive number representing a loss).
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan

    mu = r.mean()
    sigma = r.std()

    # Gaussian quantile
    z = float(np.quantile(np.random.standard_normal(500_000), alpha))

    # Higher-order moments
    _, skew, kurt_excess = _compute_moments(r)

    # Cornish–Fisher expanded quantile
    z_cf = (
        z
        + (1/6)*(z*z - 1)*skew
        + (1/24)*(z**3 - 3*z)*kurt_excess
        - (1/36)*(2*z**3 - 5*z)*skew*skew
    )

    # VaR is minus the quantile
    return float(-(mu + sigma * z_cf))


def compute_cornish_report(returns: pd.Series, alpha: float = 0.05) -> dict:
    """
    Return a dictionary with:
        - Gaussian VaR
        - Cornish–Fisher VaR
        - Difference in %
    """
    r = returns.dropna()
    if len(r) == 0:
        return {}

    # Gaussian approximation
    mu = r.mean()
    sigma = r.std()
    z = float(np.quantile(np.random.standard_normal(500_000), alpha))
    var_gauss = -(mu + sigma*z)

    # Cornish–Fisher adjusted
    var_cf = cornish_fisher_var(r, alpha)

    return {
        "VaR_gaussian": var_gauss,
        "VaR_cornish": var_cf,
        "CF_adjustment_%": 100 * (var_cf - var_gauss) / abs(var_gauss) if var_gauss != 0 else np.nan,
    }
