# scripts/run_multiasset_corr.py
"""
Multi-asset correlation analysis for:
- raw asset returns
- SMA 20/100 strategy returns on each asset.

Outputs:
- Two correlation heatmaps saved to /data:
    * multiasset_price_corr.png
    * multiasset_strategy_corr.png
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make project root importable (so 'backtesting.*' and 'utils.*' work)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_multi_assets
from backtesting.ma_crossover import sma_crossover_positions
from backtesting.engine import run_backtest


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_asset_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns for each asset.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Columns = asset symbols, index = dates, values = prices.

    Returns
    -------
    pd.DataFrame
        Daily log returns for each asset (aligned index).
    """
    log_px = np.log(df_prices)
    returns = log_px.diff().dropna()
    return returns


def compute_sma_strategy_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    """
    For each asset, build a SMA 20/100 strategy, run a backtest,
    and collect the daily strategy returns in a single DataFrame.

    Parameters
    ----------
    df_prices : pd.DataFrame
        Columns = asset symbols, index = dates, values = prices.

    Returns
    -------
    pd.DataFrame
        Columns = asset symbols, values = SMA strategy daily returns.
    """
    strat_returns = {}

    for symbol in df_prices.columns:
        price = df_prices[symbol].dropna()

        # SMA crossover positions (+1 / 0 / -1)
        pos = sma_crossover_positions(price, short=20, long=100)

        # Run backtest for this asset/strategy
        res = run_backtest(price, pos, cost_bps=1.0, initial_capital=1.0)

        # Store net daily returns
        strat_returns[symbol] = res.returns

    # Align on common date index
    strat_ret_df = pd.concat(strat_returns, axis=1).dropna()
    strat_ret_df.columns = df_prices.columns  # ensure same order
    return strat_ret_df


def plot_corr_heatmap(corr_df: pd.DataFrame, title: str, out_name: str) -> None:
    """
    Plot and save a correlation heatmap from a correlation matrix.

    Parameters
    ----------
    corr_df : pd.DataFrame
        Correlation matrix.
    title : str
        Title of the figure.
    out_name : str
        File name (inside /data) for the PNG output.
    """
    plt.figure(figsize=(6, 5))
    im = plt.imshow(corr_df.values, vmin=-1.0, vmax=1.0, cmap="coolwarm")

    plt.title(title)
    plt.xticks(range(len(corr_df.columns)), corr_df.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr_df.index)), corr_df.index)

    # Add correlation values in each cell
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            val = corr_df.iloc[i, j]
            plt.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color="black" if abs(val) < 0.7 else "white",
                fontsize=9,
            )

    plt.colorbar(im, fraction=0.046, pad=0.04, label="Correlation")
    plt.tight_layout()

    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", out_name
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Correlation heatmap saved → {out_png}")

    plt.show()


# ---------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------
def main():
    # 1) Universe of assets
    symbols = ["MSFT", "BTCUSD", "SP500", "AAPL"]

    # 2) Load multi-asset prices (one column per symbol)
    df_prices = load_multi_assets(symbols)

    # Ensure no silly columns
    df_prices = df_prices[symbols].dropna(how="all")

    # 3) Raw asset log-returns
    asset_rets = compute_asset_returns(df_prices)
    asset_corr = asset_rets.corr()

    print("\n===== Correlation matrix – asset returns =====")
    print(asset_corr.round(3))

    plot_corr_heatmap(
        corr_df=asset_corr,
        title="Correlation of asset daily returns",
        out_name="multiasset_price_corr.png",
    )

    # 4) SMA 20/100 strategy returns per asset
    strat_rets = compute_sma_strategy_returns(df_prices)
    strat_corr = strat_rets.corr()

    print("\n===== Correlation matrix – SMA 20/100 strategy returns =====")
    print(strat_corr.round(3))

    plot_corr_heatmap(
        corr_df=strat_corr,
        title="Correlation of SMA 20/100 strategy returns",
        out_name="multiasset_strategy_corr.png",
    )


if __name__ == "__main__":
    main()
