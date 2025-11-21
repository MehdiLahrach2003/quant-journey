# backtesting/report.py
"""
Reporting helpers for backtests:
- Drawdown and rolling Sharpe computation
- Tear sheet plot (equity, rolling Sharpe, drawdown)
- Trade log export to CSV
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class BacktestResult:
    """
    Container for everything returned by the backtest engine.
    """
    equity: pd.Series
    returns_under: pd.Series        # underlying simple returns
    strategy_ret: pd.Series         # strategy simple returns (net of costs)
    costs: pd.Series                # transaction costs time series
    positions: pd.Series            # position in {-1, 0, +1} (or size)
    trades: Optional[pd.DataFrame]  # trade log (one row per trade)
    metrics: Dict[str, Any]         # summary metrics (Sharpe, max DD, etc.)


# ---------------------------------------------------------------------------
# Risk / performance helpers
# ---------------------------------------------------------------------------

def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute drawdown series from an equity curve.

    Drawdown_t = equity_t / max_{s <= t}(equity_s) - 1
    """
    eq = equity.astype(float)
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    dd.name = "drawdown"
    return dd


def rolling_sharpe(returns: pd.Series, window: int = 126) -> pd.Series:
    """
    Compute rolling Sharpe ratio on a window of `window` observations.

    Here we assume returns are daily simple returns and use:
        Sharpe = mean(ret) / std(ret) * sqrt(252)
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)

    sharpe = roll_mean / roll_std * np.sqrt(252.0)
    sharpe.name = f"rolling_sharpe_{window}"
    return sharpe


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    ann_factor: int = 252,
) -> pd.Series:
    """
    Backward-compatible wrapper used by some scripts.

    It does the same thing as `rolling_sharpe`, but with an explicit
    annualization factor argument so older imports still work:
        from backtesting.report import compute_rolling_sharpe
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std(ddof=1)

    sharpe = roll_mean / roll_std * np.sqrt(float(ann_factor))
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sharpe.name = f"rolling_sharpe_{window}"
    return sharpe


# ---------------------------------------------------------------------------
# Tear sheet plotting
# ---------------------------------------------------------------------------

def make_tear_sheet(
    result: BacktestResult,
    price: pd.Series,
    out_png: Optional[str] = None,
) -> None:
    """
    Plot a simple tear sheet with:
    - Strategy equity vs buy & hold (both rebased to 1.0)
    - Rolling Sharpe (126 days)
    - Drawdown

    Parameters
    ----------
    result : BacktestResult
        Output of the backtest engine.
    price : pd.Series
        Underlying price series (indexed by date).
    out_png : str, optional
        If provided, save the figure to this path.
    """
    eq = result.equity.astype(float)
    eq_rebased = eq / eq.iloc[0]

    bh = price.astype(float) / price.iloc[0]
    bh.name = "buy_and_hold"

    dd = compute_drawdown(eq)
    roll_sh = rolling_sharpe(result.strategy_ret, window=126)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(12, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    # 1) Equity vs benchmark
    ax1.plot(eq_rebased.index, eq_rebased, label="Strategy", lw=1.5)
    ax1.plot(bh.index, bh, label="Buy & Hold", lw=1.2)
    ax1.set_ylabel("Index (× start)")
    ax1.set_title("Equity (rebased) vs Benchmark")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # 2) Rolling Sharpe
    ax2.plot(roll_sh.index, roll_sh, lw=1.2)
    ax2.axhline(0.0, color="black", lw=0.8)
    ax2.set_ylabel("Sharpe")
    ax2.set_title("Rolling Sharpe (126d)")
    ax2.grid(alpha=0.3)

    # 3) Drawdown
    ax3.fill_between(dd.index, dd, 0.0, color="steelblue", alpha=0.4)
    ax3.set_ylabel("Drawdown")
    ax3.set_title("Drawdown")
    ax3.grid(alpha=0.3)

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Tear sheet saved → {out_png}")

    plt.show()


# ---------------------------------------------------------------------------
# Trade log export
# ---------------------------------------------------------------------------

def export_trades_to_csv(result: BacktestResult, out_csv: str) -> None:
    """
    Export the trade log contained in `result.trades` to a CSV file.

    The function is safe to call even if there are no trades.
    """
    trades = result.trades

    if trades is None or len(trades) == 0:
        print("[WARN] No trades found in BacktestResult.trades, nothing to export.")
        return

    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    trades.to_csv(out_csv, index=True)
    print(f"[OK] Trades exported → {out_csv}")
