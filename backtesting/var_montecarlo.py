# backtesting/var_montecarlo.py
"""
Monte Carlo estimation of VaR and CVaR:
- Parametric MC (Gaussian)
- Bootstrap MC (sampling historical returns)
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Parametric Monte Carlo VaR / CVaR
# ------------------------------------------------------------
def mc_var_parametric(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 20_000
) -> tuple[float, float]:
    """
    Gaussian Monte Carlo VaR and CVaR.
    
    We simulate returns ~ N(mu, sigma^2) and compute tail losses.

    Returns
    -------
    (VaR, CVaR)
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan

    mu = r.mean()
    sigma = r.std()

    sims = np.random.normal(mu, sigma, size=n_sims)
    sims = np.sort(sims)

    # VaR = negative alpha-quantile
    var_mc = -np.percentile(sims, alpha * 100)

    # CVaR = average of worst alpha% outcomes
    cutoff = int(alpha * n_sims)
    cvar_mc = -sims[:cutoff].mean()

    return float(var_mc), float(cvar_mc)


# ------------------------------------------------------------
# Bootstrap Monte Carlo VaR / CVaR
# ------------------------------------------------------------
def mc_var_bootstrap(
    returns: pd.Series,
    alpha: float = 0.05,
    n_sims: int = 20_000
) -> tuple[float, float]:
    """
    Bootstrap Monte Carlo VaR and CVaR.
    We randomly sample returns *with replacement* from historical data.

    Returns
    -------
    (VaR, CVaR)
    """
    r = returns.dropna().astype(float)
    if len(r) == 0:
        return np.nan, np.nan

    sims = np.random.choice(r, size=n_sims, replace=True)
    sims = np.sort(sims)

    var_mc = -np.percentile(sims, alpha * 100)
    cutoff = int(alpha * n_sims)
    cvar_mc = -sims[:cutoff].mean()

    return float(var_mc), float(cvar_mc)
