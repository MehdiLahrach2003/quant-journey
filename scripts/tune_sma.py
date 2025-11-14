# scripts/tune_sma.py
# Run a grid-search over SMA windows, save CSV, print best params, and (optionally) plot a heatmap.

import os
import sys
import matplotlib.pyplot as plt

# Make local packages importable when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from utils.param_search import evaluate_sma_grid


RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def main():
    df = load_prices()

    shorts = [5, 10, 20, 30, 40]
    longs  = [50, 80, 100, 150, 200]

    results_df, pivot = evaluate_sma_grid(
        df,
        shorts,
        longs,
        cost_bps=1.0,            # <- use 'cost_bps' (not total_bps / trans_cost_bps)
        criterion="sharpe",
        initial_capital=1.0,
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_csv = os.path.join(RESULTS_DIR, "sma_grid_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[OK] Results saved to: {out_csv}")

    if not pivot.empty:
        print("\n=== Best params (by Sharpe) ===")
        best_long, best_short = pivot.stack().idxmax()  # (long, short)
        best_val = pivot.max().max()
        print(f"short={best_short}, long={best_long}, sharpe={best_val:.4f}")

        # Simple heatmap
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", origin="lower")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("short window")
        ax.set_ylabel("long window")
        ax.set_title("SMA grid — Sharpe")
        plt.colorbar(im, ax=ax)
        out_png = os.path.join(RESULTS_DIR, "sma_grid_heatmap.png")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Heatmap saved to: {out_png}")
        plt.show()
    else:
        print("[WARN] Empty pivot — check grids or data length.")


if __name__ == "__main__":
    main()
