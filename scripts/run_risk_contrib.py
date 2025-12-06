# scripts/run_risk_contrib.py
"""
Compute and display risk contributions for several portfolios built
from multi-asset SMA 20/100 strategies:

- Equal-weight
- Inverse-volatility
- Minimum-variance
- Maximum-Sharpe

Outputs:
- Table of weights and component risk contributions per asset.
"""

import os
import sys
import math

import numpy as np
import pandas as pd

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.optimizer import (
    compute_mu_cov_from_results,
    solve_min_var,
    solve_max_sharpe,
    portfolio_stats,
)
from backtesting.risk_contrib import risk_contributions


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_sma_strategies(df_prices: pd.DataFrame) -> dict:
    """
    For each asset in df_prices, build SMA 20/100 strategy and run a backtest.

    Returns
    -------
    dict[str, BacktestResult]
    """
    results = {}
    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()
        pos = sma_crossover_positions(price, short=20, long=100)
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)
        results[symbol] = res
    return results


def inverse_vol_weights_from_results(results: dict, asset_order: list[str]) -> np.ndarray:
    """
    Build inverse-volatility weights from BacktestResult returns.
    """
    sigmas = []
    for name in asset_order:
        r = results[name].returns
        sigma = float(r.std() * math.sqrt(252.0))
        sigmas.append(sigma)

    sigmas = np.asarray(sigmas, dtype=float)
    inv = np.where(sigmas > 0.0, 1.0 / sigmas, 0.0)
    total = inv.sum()
    if total == 0.0:
        return np.ones_like(inv) / len(inv)
    return inv / total


def print_risk_table(title: str, rc_result, mu_ann: pd.Series):
    """
    Pretty-print weights and risk contributions for a given portfolio.
    """
    print(f"\n===== {title} =====")
    df = pd.DataFrame({
        "Weight": rc_result.weights,
        "MRC": rc_result.mrc,       # marginal risk contribution
        "CRC": rc_result.crc,       # component risk contribution
        "Mu_ann": mu_ann.reindex(rc_result.weights.index),
    })

    # Normalise CRC in percentage of total risk
    if rc_result.vol_port > 0.0:
        df["CRC_%"] = 100.0 * df["CRC"] / rc_result.vol_port
    else:
        df["CRC_%"] = 0.0

    print(f"Portfolio volatility (ann.): {rc_result.vol_port:.4f}\n")
    print(df.round(4))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Universe
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # 2) Load multi-asset prices
    df_prices = load_multi_assets(symbols)

    # 3) Build SMA strategies & backtests
    results = build_sma_strategies(df_prices)

    # 4) Annualised mu and covariance in strategy space
    mu_ann, cov_ann = compute_mu_cov_from_results(results)
    assets = list(mu_ann.index)
    n = len(assets)

    # === Portfolios to analyse ===

    # Equal-weight
    w_eq = np.ones(n) / n

    # Inverse-vol
    w_inv = inverse_vol_weights_from_results(results, assets)

    # Min-variance (long-only)
    stats_minvar = solve_min_var(mu_ann, cov_ann)
    w_minvar = stats_minvar.weights

    # Max-Sharpe (long-only)
    stats_maxsharpe = solve_max_sharpe(mu_ann, cov_ann, rf=0.0)
    w_maxsharpe = stats_maxsharpe.weights

    # 5) Risk contributions for each portfolio
    rc_eq = risk_contributions(pd.Series(w_eq, index=assets), cov_ann)
    rc_inv = risk_contributions(pd.Series(w_inv, index=assets), cov_ann)
    rc_minvar = risk_contributions(pd.Series(w_minvar, index=assets), cov_ann)
    rc_maxsharpe = risk_contributions(pd.Series(w_maxsharpe, index=assets), cov_ann)

    # 6) Pretty-print
    print_risk_table("Equal-weight portfolio", rc_eq, mu_ann)
    print_risk_table("Inverse-vol portfolio", rc_inv, mu_ann)
    print_risk_table("Min-variance portfolio", rc_minvar, mu_ann)
    print_risk_table("Max-Sharpe portfolio", rc_maxsharpe, mu_ann)


if __name__ == "__main__":
    main()
