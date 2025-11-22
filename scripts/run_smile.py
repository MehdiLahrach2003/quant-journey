# scripts/run_smile.py
# Visualize an implied volatility smile using your Black–Scholes and implied vol modules.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# --- Make project root importable (so `pricing.*` works) ---
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import bs_call_price
from pricing.implied_vol import implied_vol_bs  # adapt name if your function is different


def true_sigma(K: float, S0: float = 100.0) -> float:
    """
    Simple 'smile' function for the true volatility:
    low vol near-the-money, higher vol in the wings.
    """
    moneyness = K / S0
    # Base 20% vol, plus curvature term to create a U-shape
    return 0.20 + 0.10 * (moneyness - 1.0) ** 2


def build_smile(
    S0: float = 100.0,
    r: float = 0.02,
    T: float = 1.0,
    k_min: float = 0.6,
    k_max: float = 1.4,
    n_strikes: int = 17,
):
    """
    Generate synthetic call prices with a 'true' smile,
    then back out implied vols.
    """
    strikes = np.linspace(k_min * S0, k_max * S0, n_strikes)

    true_vols = []
    call_prices = []
    implied_vols = []

    for K in strikes:
        sigma = true_sigma(K, S0)
        true_vols.append(sigma)

        # Theoretical call price under Black–Scholes
        price = bs_call_price(S0, K, r, sigma, T)
        call_prices.append(price)

        # Back out implied vol from the price
        iv = implied_vol_call(price, S0, K, r, T)
        implied_vols.append(iv)

    return strikes, np.array(true_vols), np.array(call_prices), np.array(implied_vols)


def plot_smile(strikes, true_vols, implied_vols, out_png: str | None = None):
    """
    Plot theoretical vs implied volatility smile.
    """
    plt.figure(figsize=(10, 6))

    plt.plot(strikes, true_vols, label="True vol (model)", lw=2, alpha=0.7)
    plt.scatter(strikes, implied_vols, label="Implied vol (from prices)", color="orange")

    plt.xlabel("Strike")
    plt.ylabel("Volatility")
    plt.title("Implied Volatility Smile (Call options)")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()

    if out_png is not None:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=150)
        print(f"[OK] Smile figure saved → {out_png}")

    plt.show()


def main():
    # 1) Build synthetic smile data
    S0 = 100.0
    r = 0.02
    T = 1.0

    strikes, true_vols, call_prices, implied_vols = build_smile(
        S0=S0,
        r=r,
        T=T,
        k_min=0.6,
        k_max=1.4,
        n_strikes=17,
    )

    # 2) Print a small table in the terminal
    print("\n===== Volatility Smile (Calls) =====")
    print("Strike    True vol   Implied vol")
    for K, sv, iv in zip(strikes, true_vols, implied_vols):
        print(f"{K:7.2f}   {sv:8.4f}   {iv:11.4f}")

    # 3) Plot smile and save PNG in data/
    project_root = os.path.dirname(os.path.dirname(__file__))
    out_png = os.path.join(project_root, "data", "smile_example.png")

    plot_smile(strikes, true_vols, implied_vols, out_png=out_png)


if __name__ == "__main__":
    main()
