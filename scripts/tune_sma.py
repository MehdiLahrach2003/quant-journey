# scripts/tune_sma.py
# SMA (short,long) grid search with CSV export and heatmap visualization.

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the project importable when using VS Code "Run" button
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from utils.param_search import evaluate_sma_grid, pick_best_params


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def plot_heatmap(pivot: pd.DataFrame, title: str, cmap: str = "viridis"):
    """
    Pure-matplotlib heatmap (no seaborn dependency).
    X-axis: short window; Y-axis: long window.
    """
    plt.figure(figsize=(8, 6))
    data = pivot.values.astype(float)
    im = plt.imshow(data, aspect="auto", origin="lower", cmap=cmap)

    # Axis ticks with actual window values
    plt.xticks(range(len(pivot.columns)), pivot.columns.tolist())
    plt.yticks(range(len(pivot.index)), pivot.index.tolist())

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label(pivot.columns.name or "value")

    plt.xlabel("short window")
    plt.ylabel("long window")
    plt.title(title)
    plt.tight_layout()

    # Save figure
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig_path = os.path.join(RESULTS_DIR, "sma_grid_heatmap.png")
    plt.savefig(fig_path, dpi=150)
    print(f"[OK] Heatmap saved to: {fig_path}")

    plt.show()


def main():
    # 1) Load prices
    df = load_prices()  # expects df['price']

    # 2) Define parameter grid
    shorts = list(range(5, 31, 5))     # 5,10,15,20,25,30
    longs  = list(range(40, 121, 10))  # 40,50,...,120

    # 3) Evaluate all combinations and create a pivot for heatmap
    total_bps = 10.0  # total trading costs (bps) per absolute position change
    results_df, pivot = evaluate_sma_grid(
        df, shorts, longs,
        total_bps=total_bps,
        initial_capital=1.0,
        criterion="sharpe",
    )

    # 4) Save raw results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "sma_grid_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[OK] Results saved to: {out_csv}")

    # 5) Pick best params (robust to NaN/±inf)
    try:
        best = pick_best_params(results_df, by="sharpe")
        by_used = "sharpe"
    except ValueError as e:
        print(f"[WARN] {e}")
        print("[INFO] Falling back to 'cumret' criterion.")
        best = pick_best_params(results_df, by="cumret")
        by_used = "cumret"

    print(f"\n=== Best params (by {by_used}) ===")
    print(best)

    # 6) Plot heatmap
    pivot.columns.name = "short"
    pivot.index.name = "long"
    plot_heatmap(pivot, title=f"SMA Grid Search — {by_used.capitalize()}")


if __name__ == "__main__":
    main()
