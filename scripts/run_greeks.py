# scripts/run_greeks.py
# Visualise Black–Scholes Greeks as a function of the underlying price.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make the project root importable (so "pricing.*" works when you Run)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import (
    bs_call_price,
    bs_call_delta,
    bs_call_gamma,
    bs_call_vega,
    bs_call_theta,
)


def main():
    # --- Model parameters (you can tweak these) ---
    S0 = 100.0     # reference spot
    K = 100.0      # strike
    r = 0.01       # risk-free rate
    sigma = 0.20   # volatility
    T = 1.0        # time to maturity (in years)

    # Spot grid for plotting
    spots = np.linspace(50, 150, 200)

    # --- Compute price and Greeks on the grid ---
    call_prices = bs_call_price(spots, K, r, sigma, T)
    delta = bs_call_delta(spots, K, r, sigma, T)
    gamma = bs_call_gamma(spots, K, r, sigma, T)
    vega = bs_call_vega(spots, K, r, sigma, T)
    theta = bs_call_theta(spots, K, r, sigma, T)

    # --- Plot all of them ---
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # 1) Call price
    ax = axes[0]
    ax.plot(spots, call_prices, label="Call price")
    ax.axvline(K, color="grey", ls="--", label="Strike K")
    ax.set_ylabel("Price")
    ax.set_title("Black–Scholes call price and Greeks vs spot")
    ax.legend()
    ax.grid(alpha=0.3)

    # 2) Delta
    ax = axes[1]
    ax.plot(spots, delta)
    ax.axvline(K, color="grey", ls="--")
    ax.set_ylabel("Delta")
    ax.grid(alpha=0.3)

    # 3) Gamma
    ax = axes[2]
    ax.plot(spots, gamma)
    ax.axvline(K, color="grey", ls="--")
    ax.set_ylabel("Gamma")
    ax.grid(alpha=0.3)

    # 4) Vega (and optionally theta on secondary axis)
    ax = axes[3]
    ax.plot(spots, vega, label="Vega")
    ax.axvline(K, color="grey", ls="--")

    # Show theta on a second axis so scales are readable
    ax2 = ax.twinx()
    ax2.plot(spots, theta, color="tab:red", alpha=0.7, label="Theta")

    ax.set_xlabel("Spot S")
    ax.set_ylabel("Vega")
    ax2.set_ylabel("Theta")

    ax.grid(alpha=0.3)

    # Build a combined legend for Vega + Theta
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
