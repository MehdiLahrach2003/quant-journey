# backtesting/risk.py
# Volatility targeting overlay for any discrete position series.

from __future__ import annotations
import numpy as np
import pandas as pd


def vol_target_positions(
    returns: pd.Series,
    positions: pd.Series,
    target_vol_annual: float = 0.10,   # e.g., 10% annual vol target
    lookback: int = 20,                # rolling window in trading days
    ann_factor: int = 252,             # trading days per year
    max_leverage: float = 3.0,         # cap on exposure magnitude
) -> pd.Series:
    """
    Scale raw positions {-1,0,+1} to target an annualized volatility.

    Parameters
    ----------
    returns : pd.Series
        Underlying daily returns (price.pct_change()) indexed by dates.
    positions : pd.Series
        Raw positions (e.g., {-1,0,+1}) indexed like `returns`.
        Must already be shifted to avoid lookahead (your SMA does it).
    target_vol_annual : float
        Desired annualized volatility for the strategy.
    lookback : int
        Rolling window (in days) for realized vol estimate.
    ann_factor : int
        Annualization factor (252 for daily data).
    max_leverage : float
        Hard cap on absolute exposure.

    Returns
    -------
    pd.Series
        Scaled exposures (floats), clipped to +/- max_leverage.
    """
    ret = returns.astype(float).fillna(0.0)

    # Realized vol estimate (annualized)
    vol_daily = ret.rolling(lookback).std()
    vol_annual = vol_daily * np.sqrt(ann_factor)

    # Scale factor = target / realized
    scale = target_vol_annual / vol_annual.replace(0.0, np.nan)
    scale = scale.clip(upper=max_leverage)
    scale = scale.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    expo = positions.astype(float) * scale
    expo.name = "exposure"

    # Do NOT shift here; assume positions were already shifted in the signal.
    return expo
