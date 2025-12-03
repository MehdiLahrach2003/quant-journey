# scripts/run_multiasset_frontier.py
"""
Multi-asset efficient frontier for SMA 20/100 strategies on several assets.

Steps:
- Load multi-asset prices (MSFT, BTCUSD, SP500, AAPL)
- Build SMA 20/100 strategies for each asset and run backtests
- Compute annualised mean returns and covariance in strategy space
- Compute equal-weight, inverse-vol, min-variance and max-Sharpe portfolios
- Generate a long-only efficient frontier
- Plot everything in (vol, return) space
"""

import os
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

# Make project root importable
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest
from backtesting.optimizer import (
    compute_mu_cov_from_results,
    portfolio_stats,
    solve_min_var,
    solve_max_sharpe,
    efficient_frontier,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def build_sma_strategies(df_prices):
    """
    Build SMA 20/100 strategies and run backtests for each column in df_prices.

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


def compute_inverse_vol_weights_from_results(results, assets):
    """
    Build inverse-volatility weights from BacktestResult returns.

    Parameters
    ----------
    results : dict[str, BacktestResult]
    assets : list[str]
        Order of assets (must match mu/cov indices later).

    Returns
    -------
    np.ndarray
        Normalised inverse-vol weights (sum to 1).
    """
    sigmas = []
    for name in assets:
        res = results[name]
        sigma = float(res.returns.std() * math.sqrt(252.0))
        sigmas.append(sigma)

    sigmas = np.asarray(sigmas, dtype=float)
    inv = np.where(sigmas > 0.0, 1.0 / sigmas, 0.0)
    total = inv.sum()
    if total == 0.0:
        return np.ones_like(inv) / len(inv)
    return inv / total


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------
def plot_frontier(
    results,
    mu_ann,
    cov_ann,
    stats_eq,
    stats_inv,
    stats_minvar,
    stats_maxsharpe,
    vols_ef,
    rets_ef,
):
    """
    Plot strategies, special portfolios and efficient frontier.
    """
    plt.figure(figsize=(10, 6))

    # Individual strategies
    for name, res in results.items():
        # realised annualised stats
        r_daily = res.returns
        mu = float(r_daily.mean() * 252.0)
        vol = float(r_daily.std() * math.sqrt(252.0))
        plt.scatter(vol, mu, label=name, s=50)

    # Special portfolios
    plt.scatter(
        stats_eq.vol_ann,
        stats_eq.ret_ann,
        marker="*", s=120, label="Equal-weight",
    )
    plt.scatter(
        stats_inv.vol_ann,
        stats_inv.ret_ann,
        marker="*", s=120, label="Inverse-vol",
    )
    plt.scatter(
        stats_minvar.vol_ann,
        stats_minvar.ret_ann,
        marker="D", s=80, label="Min-Var",
    )
    plt.scatter(
        stats_maxsharpe.vol_ann,
        stats_maxsharpe.ret_ann,
        marker="D", s=80, label="Max-Sharpe",
    )

    # Efficient frontier
    plt.plot(vols_ef, rets_ef, lw=2.0, label="Efficient frontier")

    plt.xlabel("Annualised Volatility")
    plt.ylabel("Annualised Return")
    plt.title("Efficient frontier – SMA 20/100 multi-asset strategies")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_efficient_frontier.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Efficient frontier plot saved → {out_png}")

    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    # 1) Universe
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # 2) Load multi-asset prices (one column per symbol)
    df_prices = load_multi_assets(symbols)

    # 3) Build SMA 20/100 strategies & backtests
    results = build_sma_strategies(df_prices)

    # 4) Mean returns and covariance in strategy space
    mu_ann, cov_ann = compute_mu_cov_from_results(results)
    assets = list(mu_ann.index)
    n = len(assets)

    # 5) Equal-weight portfolio
    w_eq = np.ones(n) / n
    stats_eq = portfolio_stats(w_eq, mu_ann, cov_ann)

    # 6) Inverse-vol portfolio (based on realised strategy vol)
    w_inv = compute_inverse_vol_weights_from_results(results, assets)
    stats_inv = portfolio_stats(w_inv, mu_ann, cov_ann)

    # 7) Min-var and Max-Sharpe portfolios
    stats_minvar = solve_min_var(mu_ann, cov_ann)
    stats_maxsharpe = solve_max_sharpe(mu_ann, cov_ann, rf=0.0)

    # 8) Efficient frontier
    vols_ef, rets_ef, _ = efficient_frontier(mu_ann, cov_ann, n_points=40)

    # 9) Plot
    plot_frontier(
        results,
        mu_ann,
        cov_ann,
        stats_eq,
        stats_inv,
        stats_minvar,
        stats_maxsharpe,
        vols_ef,
        rets_ef,
    )


if __name__ == "__main__":
    main()
