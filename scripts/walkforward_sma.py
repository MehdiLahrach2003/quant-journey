# scripts/walkforward_sma.py
# Entry-point you can "Run" in VS Code.
# It loads prices, runs walk-forward tuning/evaluation, prints a table,
# and plots the out-of-sample equity curve.

import sys
import os
import matplotlib.pyplot as plt

# Make repo importable when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.data_loader import load_prices
from utils.walkforward import walkforward_sma


def main():
    # 1) Load prices (utils/data_loader.py falls back to synthetic if file missing)
    df = load_prices()  # expects a 'price' column

    # 2) Run walk-forward
    wf = walkforward_sma(
        df,
        short_grid=(5, 10, 20, 30),
        long_grid=(50, 100, 150, 200),
        train_months=24,
        test_months=3,
        cost_bps=1.0,
    )

    # 3) Print top-level summary
    print("\n===== Walk-Forward summary =====")
    if not wf.params_table.empty:
        print(wf.params_table.round(4))
    else:
        print("No windows produced (not enough data?).")

    # 4) Plot OOS equity
    if not wf.equity_oos.empty:
        plt.figure(figsize=(11, 5))
        plt.plot(wf.equity_oos.index, wf.equity_oos, label="OOS Equity", lw=1.6)
        plt.title("Walk-Forward OOS Equity")
        plt.ylabel("Equity (normalized)")
        plt.grid(alpha=0.3)
        plt.legend()

        # Save plot next to your data folder
        out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "wf_equity.png")
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"\n[OK] Walk-forward equity plot saved to: {out_png}\n")

        plt.show()
    else:
        print("No equity to plot.")


if __name__ == "__main__":
    main()
