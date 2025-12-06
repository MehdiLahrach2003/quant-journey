# backtesting/risk_contrib.py
"""
Risk contribution utilities for portfolio analysis.

Given:
- a covariance matrix Σ
- a weight vector w

we can compute:
- portfolio volatility
- marginal contribution to risk (MRC)
- component contribution to risk (CRC = w_i * MRC_i)

These are standard tools for:
- risk-parity
- portfolio diagnostics
- attribution of total risk to individual assets/strategies.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np
import pandas as pd
import math


@dataclass
class RiskContributionResult:
    weights: pd.Series          # portfolio weights
    vol_port: float             # total portfolio volatility
    mrc: pd.Series              # marginal contribution to risk
    crc: pd.Series              # component contribution to risk (sums to vol_port)


def risk_contributions(
    weights: pd.Series | np.ndarray,
    cov_ann: pd.DataFrame,
) -> RiskContributionResult:
    """
    Compute portfolio volatility and risk contributions (MRC & CRC).

    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights. If a Series, its index should match cov_ann columns.
    cov_ann : pd.DataFrame
        Annualised covariance matrix of asset/strategy returns.

    Returns
    -------
    RiskContributionResult
    """
    # Convert weights to aligned Series
    if isinstance(weights, pd.Series):
        w = weights.astype(float)
        assets = list(w.index)
    else:
        # assume order is the same as cov_ann columns
        assets = list(cov_ann.columns)
        w = pd.Series(weights, index=assets, dtype=float)

    # Align covariance to assets order
    cov = cov_ann.loc[assets, assets].values
    w_vec = w.values

    # Portfolio variance and volatility
    var_port = float(w_vec @ cov @ w_vec)
    vol_port = math.sqrt(var_port) if var_port > 0.0 else 0.0

    if vol_port == 0.0:
        # Degenerate case: zero volatility
        mrc = pd.Series(0.0, index=assets)
        crc = pd.Series(0.0, index=assets)
        return RiskContributionResult(weights=w, vol_port=0.0, mrc=mrc, crc=crc)

    # Marginal contribution to risk: (Σ w)_i / σ_p
    sigma_w = cov @ w_vec  # shape (n,)
    mrc_vec = sigma_w / vol_port

    # Component contribution to risk: w_i * MRC_i
    crc_vec = w_vec * mrc_vec

    mrc = pd.Series(mrc_vec, index=assets, name="MRC")
    crc = pd.Series(crc_vec, index=assets, name="CRC")

    return RiskContributionResult(weights=w, vol_port=vol_port, mrc=mrc, crc=crc)
