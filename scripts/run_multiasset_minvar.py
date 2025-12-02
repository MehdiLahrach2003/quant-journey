# scripts/run_multiasset_minvar.py
"""
Multi-asset SMA 20/100 strategies combined in a minimum-variance portfolio.

This script:
- loads multi-asset prices (MSFT, BTCUSD, SP500, AAPL)
- builds a SMA 20/100 strategy on each asset
- runs a backtest per asset (using backtesting.engine.run_backtest)
- computes the sample covariance matrix of strategy daily returns
- builds the unconstrained minimum-variance portfolio:
      w ∝ Σ^{-1} 1
- compares it to an equal-weight portfolio
- plots all equity curves and saves the figure to data/multiasset_minvar_equity.png
"""

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

# Make project root importable (so 'backtesting.*' and 'utils.*' work)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.portfolio import combine_backtests


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_sma_strategies(df_prices):
    """
    For each asset in df_prices, build a SMA 20/100 strategy and run a backtest.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Columns = asset symbols, index = dates, values = prices.

    Returns
    -------
    dict[str, BacktestResult]
        One BacktestResult per asset.
    """
    results = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # Build SMA crossover positions (+1 / 0 / -1)
        pos = sma_crossover_positions(price, short=20, long=100)

        # Run backtest on this single asset
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)
        results[symbol] = res

    return results


def compute_minvar_weights(results):
    """
    Compute minimum-variance portfolio weights from strategy daily returns.

    We use the classical Markowitz closed-form solution (no constraints):
        w* ∝ Σ^{-1} 1

    The weights are then normalized to sum to 1. They may be slightly
    negative (allowing mild short exposures), which is fine for a
    theoretical min-var portfolio.

    Parameters
    ----------
    results : dict[str, BacktestResult]

    Returns
    -------
    weights : dict[str, float]
        Minimum-variance weights (sum to 1).
    cov : pd.DataFrame
        Sample covariance matrix of strategy returns.
    """
    import pandas as pd

    # Build a DataFrame of strategy daily returns
    ret_df = pd.DataFrame(
        {name: res.returns.astype(float) for name, res in results.items()}
    ).dropna(how="all")

    # Sample covariance matrix Σ
    cov = ret_df.cov()

    # Vector of ones
    names = list(ret_df.columns)
    ones = np.ones(len(names))

    # Use pseudo-inverse in case Σ is ill-conditioned
    sigma_inv = np.linalg.pinv(cov.values)

    # Unconstrained min-var weights (not yet normalized)
    raw_w = sigma_inv @ ones

    # Normalize to sum to 1
    w = raw_w / raw_w.sum()

    weights = {name: float(w[i]) for i, name in enumerate(names)}
    return weights, cov


def plot_multiasset_minvar(results, ew_res, minvar_res):
    """
    Plot equity curves for each asset strategy + EW portfolio + min-var portfolio.
    """
    plt.figure(figsize=(12, 6))

    # Individual strategies
    for name, res in results.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    # Equal-weight portfolio
    ew_eq = ew_res.equity / ew_res.equity.iloc[0]
    plt.plot(ew_eq.index, ew_eq, label="Portfolio EW", linewidth=2.0, ls="--")

    # Min-variance portfolio
    mv_eq = minvar_res.equity / minvar_res.equity.iloc[0]
    plt.plot(mv_eq.index, mv_eq, label="Portfolio min-var", linewidth=2.0)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Multi-asset SMA 20/100 strategies vs portfolios (EW vs min-var)")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_minvar_equity.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Min-variance equity plot saved → {out_png}")

    plt.show()


def print_weights_and_metrics(results, ew_res, minvar_res, minvar_w, cov):
    """
    Print summary of weights and basic metrics.
    """
    import pandas as pd

    print("\n===== Minimum-variance weights (strategy space) =====")
    for name, w in minvar_w.items():
        print(f"{name:10s}  w_minvar = {w:7.3f}")

    # Annualized vol for portfolios
    def ann_vol(res):
        return float(res.returns.std() * math.sqrt(252.0))

    print("\n===== Portfolio comparison =====")
    print(f"{'Portfolio':20s} {'CumReturn':>10s} {'AnnVol':>10s} {'Sharpe?':>10s}")
    for label, res in [
        ("Equal-weight", ew_res),
        ("Min-variance", minvar_res),
    ]:
        eq = res.equity
        cum_ret = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
        vol = ann_vol(res)
        # Sharpe stored in metrics under 'Sharpe Ratio'
        sharpe = res.metrics.get("Sharpe Ratio", float("nan"))
        print(f"{label:20s} {cum_ret:10.4f} {vol:10.4f} {sharpe:10.4f}")

    print("\n===== Covariance matrix of strategy returns =====")
    # Nice compact print
    cov_round = cov.round(4)
    with pd.option_context("display.width", 120, "display.max_columns", None):
        print(cov_round)


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    # 1) Choose the universe
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # 2) Load multi-asset prices (one column per symbol)
    df_prices = load_multi_assets(symbols)

    # 3) Build SMA strategies and run backtests per asset
    results = build_sma_strategies(df_prices)

    # 4) Build an equal-weight portfolio as a baseline
    n = len(results)
    ew_weights = {name: 1.0 / n for name in results.keys()}
    ew_res = combine_backtests(results, weights=ew_weights)

    # 5) Compute minimum-variance weights from strategy returns
    minvar_w, cov = compute_minvar_weights(results)

    # 6) Combine BacktestResult objects into a min-var portfolio
    minvar_res = combine_backtests(results, weights=minvar_w)

    # 7) Print weights + metrics
    print_weights_and_metrics(results, ew_res, minvar_res, minvar_w, cov)

    # 8) Plot equity curves
    plot_multiasset_minvar(results, ew_res, minvar_res)


if __name__ == "__main__":
    main()
