# utils/risk_mc.py
# Simple Monte Carlo risk analysis based on a strategy's daily returns.

from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.risk import max_drawdown


def monte_carlo_from_returns(
    returns: pd.Series,
    n_paths: int = 2000,
    horizon: int = 252,
    seed: int | None = 42,
) -> pd.DataFrame:
    """
    Run a simple Monte Carlo on strategy returns.

    We resample (with replacement) from historical daily returns and
    build synthetic equity paths. For each path we compute:
    - final equity (starting from 1.0)
    - max drawdown over the path.

    Parameters
    ----------
    returns : pd.Series
        Daily returns of the strategy (e.g. equity.pct_change()).
    n_paths : int
        Number of Monte Carlo paths to simulate.
    horizon : int
        Number of days per simulated path (e.g. 252 for 1 year).
    seed : int | None
        Random seed for reproducibility (None = no fixed seed).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'final_equity'
        - 'max_drawdown'
    """
    ret = returns.dropna().to_numpy()
    if ret.size == 0:
        raise ValueError("returns series is empty – cannot run Monte Carlo.")

    rng = np.random.default_rng(seed)

    final_equity = np.empty(n_paths)
    max_dd = np.empty(n_paths)

    for i in range(n_paths):
        # Sample daily returns with replacement
        sample = rng.choice(ret, size=horizon, replace=True)

        # Build equity path starting from 1.0
        equity_path = pd.Series(np.cumprod(1.0 + sample), index=range(horizon))

        final_equity[i] = equity_path.iloc[-1]
        max_dd[i] = max_drawdown(equity_path)

    return pd.DataFrame(
        {
            "final_equity": final_equity,
            "max_drawdown": max_dd,
        }
    )
