# backtesting/report.py
# Tools to compute performance statistics and plot a simple tear sheet.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- Core statistics helpers ----------

def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 126,
    periods_per_year: int = 252,
) -> pd.Series:
    """
    Compute rolling Sharpe ratio on a window of daily returns.

    Parameters
    ----------
    returns : pd.Series
        Strategy returns (daily or whatever frequency you use).
    window : int
        Rolling window size in bars (126 ~ 6 months of daily data).
    periods_per_year : int
        Number of periods per year (252 for daily).

    Returns
    -------
    pd.Series
        Rolling Sharpe ratio.
    """
    r = returns.astype(float)
    roll_mean = r.rolling(window).mean()
    roll_std = r.rolling(window).std()

    sharpe = np.sqrt(periods_per_year) * roll_mean / roll_std
    sharpe.name = "rolling_sharpe"
    return sharpe


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """
    Compute percentage drawdown series from an equity curve.

    Parameters
    ----------
    equity : pd.Series
        Equity curve (portfolio value over time).

    Returns
    -------
    pd.Series
        Drawdown in decimal form (e.g. -0.15 = -15%).
    """
    eq = equity.astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    dd.name = "drawdown"
    return dd


def summarize_equity(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 252,
) -> Dict[str, float]:
    """
    Build a small dict of summary statistics.

    This is mostly for printing in the terminal and saving to CSV.
    """
    r = returns.astype(float).dropna()
    eq = equity.astype(float)

    total_return = eq.iloc[-1] / eq.iloc[0] - 1.0
    avg_ret = r.mean()
    vol = r.std()
    sharpe = np.sqrt(periods_per_year) * avg_ret / vol if vol > 0 else np.nan

    dd = compute_drawdown(eq)
    max_dd = dd.min()

    stats = {
        "total_return": float(total_return),
        "annualized_return": float((1 + total_return) ** (periods_per_year / len(eq)) - 1)
        if len(eq) > 0
        else np.nan,
        "annualized_vol": float(vol * np.sqrt(periods_per_year)),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_dd),
    }
    return stats


# ---------- Plotting helpers ----------

def plot_tear_sheet(
    equity_rebased: pd.Series,
    benchmark_rebased: Optional[pd.Series] = None,
    rolling_sharpe: Optional[pd.Series] = None,
    drawdown: Optional[pd.Series] = None,
    title: str = "Equity (rebased) vs Benchmark",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a 3-panel tear sheet:
    1) Strategy equity vs buy & hold (both rebased to 1.0),
    2) Rolling Sharpe ratio,
    3) Drawdown.

    If save_path is provided, the figure is saved as PNG.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
    )

    ax_eq, ax_sharpe, ax_dd = axes

    # --- 1) Equity vs benchmark ---
    ax_eq.plot(equity_rebased.index, equity_rebased, label="Strategy", lw=1.4)
    if benchmark_rebased is not None:
        ax_eq.plot(
            benchmark_rebased.index,
            benchmark_rebased,
            label="Buy&Hold",
            lw=1.2,
        )

    ax_eq.set_ylabel("Index (x start)")
    ax_eq.set_title(title)
    ax_eq.legend()
    ax_eq.grid(alpha=0.3)

    # --- 2) Rolling Sharpe ---
    if rolling_sharpe is not None:
        ax_sharpe.plot(rolling_sharpe.index, rolling_sharpe, lw=1.0)
    ax_sharpe.axhline(0.0, color="black", lw=0.8)
    ax_sharpe.set_ylabel("Sharpe")
    ax_sharpe.set_title("Rolling Sharpe (126d)")
    ax_sharpe.grid(alpha=0.3)

    # --- 3) Drawdown ---
    if drawdown is not None:
        ax_dd.fill_between(
            drawdown.index,
            drawdown,
            0.0,
            step="pre",
            alpha=0.3,
        )
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_title("Drawdown")
    ax_dd.grid(alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"[OK] Tear sheet saved -> {save_path}")

    plt.show()
