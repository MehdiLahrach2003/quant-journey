# backtesting/optimizer.py
"""
Markowitz-style portfolio optimisation helpers.

- Compute annualised mean returns and covariance from BacktestResult objects
- Compute portfolio statistics (return, volatility, Sharpe)
- Solve for minimum-variance and maximum-Sharpe portfolios (long-only)
- Generate an efficient frontier (target-return grid, long-only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .engine import BacktestResult  # uses the BacktestResult defined in engine.py


@dataclass
class PortfolioStats:
    weights: np.ndarray
    ret_ann: float
    vol_ann: float
    sharpe: float


# ---------------------------------------------------------------------
# Mu / Sigma from strategy backtest results
# ---------------------------------------------------------------------
def compute_mu_cov_from_results(
    results: Dict[str, BacktestResult],
    freq_per_year: int = 252,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Build annualised mean return vector and covariance matrix from a dict
    of BacktestResult objects (one per strategy/asset).

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Keys are strategy names, values are BacktestResult with `.returns`
        as daily net returns.
    freq_per_year : int
        Number of return observations per year (252 for daily).

    Returns
    -------
    mu_ann : pd.Series
        Annualised mean returns for each strategy (index = strategy names).
    cov_ann : pd.DataFrame
        Annualised covariance matrix (index/columns = strategy names).
    """
    if not results:
        raise ValueError("`results` dict is empty.")

    # Concatenate daily returns in a DataFrame
    ret_dict = {name: res.returns for name, res in results.items()}
    df_ret = pd.DataFrame(ret_dict).dropna(how="all")

    # Daily mean & covariance
    mu_daily = df_ret.mean()
    cov_daily = df_ret.cov()

    # Annualise (simple approximation)
    mu_ann = mu_daily * float(freq_per_year)
    cov_ann = cov_daily * float(freq_per_year)

    return mu_ann, cov_ann


# ---------------------------------------------------------------------
# Portfolio statistics
# ---------------------------------------------------------------------
def portfolio_stats(
    weights: np.ndarray,
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    rf: float = 0.0,
) -> PortfolioStats:
    """
    Compute annualised return, volatility and Sharpe ratio for given weights.

    Parameters
    ----------
    weights : np.ndarray
        Portfolio weights (must sum to 1, but we normalise defensively).
    mu_ann : pd.Series
        Annualised mean returns.
    cov_ann : pd.DataFrame
        Annualised covariance matrix.
    rf : float
        Risk-free rate (annualised), default 0.

    Returns
    -------
    PortfolioStats
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be a 1D array.")

    # Ensure alignment
    assets = list(mu_ann.index)
    cov = cov_ann.loc[assets, assets].values

    # Normalise weights to sum to 1 (defensive)
    if w.sum() != 0.0:
        w = w / w.sum()
    else:
        w = np.ones_like(w) / len(w)

    ret_ann = float(np.dot(w, mu_ann.values))
    var_ann = float(w @ cov @ w)
    vol_ann = math.sqrt(var_ann) if var_ann > 0.0 else 0.0

    if vol_ann > 0.0:
        sharpe = (ret_ann - rf) / vol_ann
    else:
        sharpe = 0.0

    return PortfolioStats(weights=w, ret_ann=ret_ann, vol_ann=vol_ann, sharpe=sharpe)


# ---------------------------------------------------------------------
# Optimisers (long-only, fully invested)
# ---------------------------------------------------------------------
def _long_only_constraints(n_assets: int):
    """Return constraints and bounds for long-only, sum(weights)=1."""
    # Equality constraint: sum(weights) = 1
    cons = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    # Bounds: 0 <= w_i <= 1
    bounds = [(0.0, 1.0)] * n_assets
    return cons, bounds


def solve_min_var(mu_ann: pd.Series, cov_ann: pd.DataFrame) -> PortfolioStats:
    """
    Solve for the long-only minimum-variance portfolio.

    Returns
    -------
    PortfolioStats
    """
    n = len(mu_ann)
    cov = cov_ann.values
    cons, bounds = _long_only_constraints(n)

    def obj(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError(f"Min-variance optimisation failed: {res.message}")

    return portfolio_stats(res.x, mu_ann, cov_ann)


def solve_max_sharpe(
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    rf: float = 0.0,
) -> PortfolioStats:
    """
    Solve for the long-only maximum-Sharpe portfolio (tangency portfolio).

    Returns
    -------
    PortfolioStats
    """
    n = len(mu_ann)
    cov = cov_ann.values
    cons, bounds = _long_only_constraints(n)

    mu_vec = mu_ann.values

    def obj(w: np.ndarray) -> float:
        # Negative Sharpe ratio (we minimise)
        w = np.asarray(w, dtype=float)
        ret = float(np.dot(w, mu_vec))
        var = float(w @ cov @ w)
        vol = math.sqrt(var) if var > 0.0 else 0.0
        if vol == 0.0:
            return 1e6  # penalise degenerate solutions
        sharpe = (ret - rf) / vol
        return -sharpe

    x0 = np.ones(n) / n
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=[cons])

    if not res.success:
        raise RuntimeError(f"Max-Sharpe optimisation failed: {res.message}")

    return portfolio_stats(res.x, mu_ann, cov_ann, rf=rf)


# ---------------------------------------------------------------------
# Efficient frontier
# ---------------------------------------------------------------------
def efficient_frontier(
    mu_ann: pd.Series,
    cov_ann: pd.DataFrame,
    n_points: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a (long-only) efficient frontier by sweeping target returns
    between min(mu_ann) and max(mu_ann).

    Parameters
    ----------
    mu_ann : pd.Series
        Annualised mean returns.
    cov_ann : pd.DataFrame
        Annualised covariance matrix.
    n_points : int
        Number of frontier points.

    Returns
    -------
    vols : np.ndarray
        Annualised volatilities of efficient portfolios.
    rets : np.ndarray
        Annualised returns of efficient portfolios.
    weights_grid : np.ndarray
        Matrix of weights, shape (n_points, n_assets).
    """
    n = len(mu_ann)
    assets = list(mu_ann.index)
    mu_vec = mu_ann.values
    cov = cov_ann.loc[assets, assets].values

    mu_min = float(mu_vec.min())
    mu_max = float(mu_vec.max())
    target_grid = np.linspace(mu_min, mu_max, n_points)

    cons_sum, bounds = _long_only_constraints(n)

    vols = []
    rets = []
    weights_grid = []

    for target in target_grid:
        # Constraint: sum(w) = 1 and w·mu = target
        def cons_ret_fun(w: np.ndarray) -> float:
            return float(np.dot(w, mu_vec) - target)

        constraints = [
            cons_sum,
            {"type": "eq", "fun": cons_ret_fun},
        ]

        def obj(w: np.ndarray) -> float:
            return float(w @ cov @ w)

        x0 = np.ones(n) / n
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=constraints)

        if res.success:
            stats = portfolio_stats(res.x, mu_ann, cov_ann)
            vols.append(stats.vol_ann)
            rets.append(stats.ret_ann)
            weights_grid.append(stats.weights)
        # If optimisation fails for this target, we skip the point

    if not vols:
        raise RuntimeError("Efficient frontier optimisation failed for all target returns.")

    vols = np.asarray(vols)
    rets = np.asarray(rets)
    weights_grid = np.asarray(weights_grid)

    # Sort by volatility for a nicer plot
    order = np.argsort(vols)
    return vols[order], rets[order], weights_grid[order]
