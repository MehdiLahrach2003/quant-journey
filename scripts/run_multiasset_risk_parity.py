# scripts/run_multiasset_risk_parity.py
"""
Multi-asset SMA 20/100 strategies combined in an inverse-volatility (risk-parity)
portfolio.

Assumptions:
- You already have daily close data in data/MSFT.csv, data/AAPL.csv, etc.
- utils.data_loader exposes a `load_multi_assets(symbols)` function that
  returns a DataFrame with one price column per symbol.
"""

import os
import sys
import math

import matplotlib.pyplot as plt
import numpy as np

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


def compute_inverse_vol_weights(results):
    """
    Compute inverse-volatility weights from strategy returns.

    Parameters
    ----------
    results : dict[str, BacktestResult]

    Returns
    -------
    weights : dict[str, float]
        Normalized inverse-vol weights (sum to 1).
    vols : dict[str, float]
        Annualized volatilities used for the weighting.
    """
    vols = {}
    inv_vol = {}

    for name, res in results.items():
        # Daily strategy returns → annualized volatility
        sigma = float(res.returns.std() * math.sqrt(252.0))
        vols[name] = sigma

        if sigma > 0.0:
            inv_vol[name] = 1.0 / sigma
        else:
            inv_vol[name] = 0.0

    total = sum(inv_vol.values())
    if total == 0.0:
        # Fallback: equal weights if all vols are zero for some reason
        n = len(inv_vol)
        weights = {k: 1.0 / n for k in inv_vol.keys()}
    else:
        weights = {k: v / total for k, v in inv_vol.items()}

    return weights, vols


def plot_multiasset_equity(results, portfolio_res, title_suffix="(inverse-vol weights)"):
    """
    Plot equity curves for each asset strategy and the risk-parity portfolio.
    """
    plt.figure(figsize=(12, 6))

    # Individual strategies
    for name, res in results.items():
        eq = res.equity / res.equity.iloc[0]
        plt.plot(eq.index, eq, label=name)

    # Portfolio
    port_eq = portfolio_res.equity / portfolio_res.equity.iloc[0]
    plt.plot(port_eq.index, port_eq, label="Portfolio inv-vol", linewidth=2.0)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title(f"Multi-asset SMA 20/100 strategies vs portfolio {title_suffix}")
    plt.ylabel("Equity (rebased to 1.0)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "multiasset_risk_parity_equity.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Risk-parity equity plot saved → {out_png}")

    plt.show()


def print_metrics_table(results, portfolio_res, weights, vols):
    """
    Print a small summary table for all strategies and the portfolio.
    """
    print("\n===== Inverse-vol weights (strategy space) =====")
    for name in results.keys():
        w = weights[name]
        v = vols[name]
        print(f"{name:10s}  weight = {w:6.3f}   vol_ann = {v:6.3f}")

    print("\n===== Strategy metrics =====")
    for name, res in results.items():
        print(f"\n--- {name} ---")
        for k, v in res.metrics.items():
            print(f"{k:25s}: {v:.4f}")
        print(f"{'Final equity':25s}: {res.equity.iloc[-1]:.4f}")

    print("\n===== Portfolio (inverse-vol) =====")
    for k, v in portfolio_res.metrics.items():
        print(f"{k:25s}: {v:.4f}")
    print(f"{'Final equity':25s}: {portfolio_res.equity.iloc[-1]:.4f}")


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

    # 4) Compute inverse-volatility weights from strategy returns
    weights, vols = compute_inverse_vol_weights(results)

    # 5) Combine BacktestResult objects into a risk-parity portfolio
    portfolio_res = combine_backtests(results, weights=weights)

    # 6) Print metrics and weights
    print_metrics_table(results, portfolio_res, weights, vols)

    # 7) Plot equity curves
    plot_multiasset_equity(results, portfolio_res)


if __name__ == "__main__":
    main()
