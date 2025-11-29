# scripts/run_portfolio_risk_parity.py
"""
Risk-parity portfolio between SMA 20/100 and Breakout 50d strategies.

What this script does:
- Load underlying price data
- Build two strategies:
    * SMA 20/100 crossover
    * 50-day breakout
- Run the backtest engine for each strategy separately
- Estimate their return volatility
- Build static "risk-parity" weights ~ 1 / vol
- Combine both strategies into a portfolio using these weights
- Print metrics and plot equity curves (individual + portfolio)
- Save the figure to data/portfolio_risk_parity.png
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# Make project root importable so backtesting.* and utils.* work
# -------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from backtesting.engine import run_backtest
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.trend_breakout import breakout_positions


def build_strategies(df, cost_bps: float = 1.0):
    """
    Build positions and run backtests for:
    - SMA 20/100 crossover
    - 50-day breakout
    """
    price = df["price"]

    # SMA 20/100
    sma_pos = sma_crossover_positions(price, short=20, long=100)
    sma_res = run_backtest(price, sma_pos, cost_bps=cost_bps, initial_capital=1.0)

    # Breakout 50d (long when price > 50d high, flat otherwise)
    brk_pos = breakout_positions(price, lookback=50)
    brk_res = run_backtest(price, brk_pos, cost_bps=cost_bps, initial_capital=1.0)

    return sma_res, brk_res


def compute_risk_parity_weights(sma_res, brk_res):
    """
    Compute static risk-parity weights proportional to 1 / volatility.

    Returns
    -------
    w_sma : float
        Weight for SMA strategy.
    w_brk : float
        Weight for breakout strategy.
    """
    # Use daily net returns from the engine
    r_sma = sma_res.returns.dropna()
    r_brk = brk_res.returns.dropna()

    vol_sma = r_sma.std()
    vol_brk = r_brk.std()

    if vol_sma == 0 or vol_brk == 0:
        # Degenerate case – fall back to equal weights
        print("[WARN] One strategy has zero volatility, using 50/50 weights.")
        return 0.5, 0.5

    inv_vol_sma = 1.0 / vol_sma
    inv_vol_brk = 1.0 / vol_brk

    total = inv_vol_sma + inv_vol_brk
    w_sma = inv_vol_sma / total
    w_brk = inv_vol_brk / total

    return float(w_sma), float(w_brk)


def build_portfolio_returns(sma_res, brk_res, w_sma: float, w_brk: float):
    """
    Combine daily returns of the two strategies with fixed weights.

    Returns
    -------
    portfolio_ret : pd.Series
        Daily portfolio returns.
    """
    import pandas as pd

    # Align indices (intersection)
    idx = sma_res.returns.index.intersection(brk_res.returns.index)

    r_sma = sma_res.returns.reindex(idx)
    r_brk = brk_res.returns.reindex(idx)

    portfolio_ret = w_sma * r_sma + w_brk * r_brk
    portfolio_ret.name = "portfolio_ret"

    return portfolio_ret


def summarize_portfolio(portfolio_ret):
    """
    Compute simple metrics for the portfolio: cumulative return,
    annualized volatility and Sharpe ratio (assuming daily data).
    """
    r = portfolio_ret.dropna()
    if r.empty:
        return {"Cumulative Return": 0.0, "Annualized Vol": 0.0, "Sharpe": 0.0}

    cum_ret = float((1.0 + r).prod() - 1.0)
    ann_vol = float(r.std() * np.sqrt(252))
    if ann_vol == 0:
        sharpe = 0.0
    else:
        sharpe = float(r.mean() / r.std() * np.sqrt(252))

    return {
        "Cumulative Return": cum_ret,
        "Annualized Vol": ann_vol,
        "Sharpe": sharpe,
    }


def plot_equity_curves(sma_res, brk_res, portfolio_ret, w_sma: float, w_brk: float):
    """
    Plot individual strategy equity curves and the risk-parity portfolio.
    """
    import pandas as pd

    # Rebuild portfolio equity from returns
    eq_port = (1.0 + portfolio_ret).cumprod()
    eq_port.name = "Portfolio"

    # Rebase all to 1.0 for comparison
    eq_sma = sma_res.equity / sma_res.equity.iloc[0]
    eq_brk = brk_res.equity / brk_res.equity.iloc[0]
    eq_port = eq_port / eq_port.iloc[0]

    # Align indices for plotting
    idx_all = eq_sma.index.union(eq_brk.index).union(eq_port.index)
    eq_sma = eq_sma.reindex(idx_all)
    eq_brk = eq_brk.reindex(idx_all)
    eq_port = eq_port.reindex(idx_all)

    plt.figure(figsize=(10, 6))
    plt.plot(eq_sma.index, eq_sma, label="SMA 20/100", lw=1.2)
    plt.plot(eq_brk.index, eq_brk, label="Breakout 50d", lw=1.2)
    plt.plot(eq_port.index, eq_port, label=f"Risk-parity ({w_sma:.0%} SMA / {w_brk:.0%} Breakout)", lw=1.8)

    plt.axhline(1.0, color="black", lw=0.8, ls="--")
    plt.title("Risk-parity portfolio vs individual strategies (rebased to 1.0)")
    plt.ylabel("Equity")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio_risk_parity.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Risk-parity equity figure saved → {out_png}")

    plt.show()


def main():
    # 1) Load price data
    df = load_prices()

    # 2) Run the two strategies
    sma_res, brk_res = build_strategies(df, cost_bps=1.0)

    # 3) Compute risk-parity weights
    w_sma, w_brk = compute_risk_parity_weights(sma_res, brk_res)
    print("\n===== Risk-parity weights =====")
    print(f"SMA 20/100 weight     : {w_sma:.4f} ({w_sma:.1%})")
    print(f"Breakout 50d weight   : {w_brk:.4f} ({w_brk:.1%})")

    # 4) Build portfolio returns and metrics
    portfolio_ret = build_portfolio_returns(sma_res, brk_res, w_sma, w_brk)
    stats = summarize_portfolio(portfolio_ret)

    print("\n===== Portfolio metrics =====")
    for k, v in stats.items():
        print(f"{k:20s}: {v:.4f}")

    # 5) Plot equity curves
    plot_equity_curves(sma_res, brk_res, portfolio_ret, w_sma, w_brk)


if __name__ == "__main__":
    main()
