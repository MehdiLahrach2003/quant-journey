# pricing/black_scholes.py
"""
Black–Scholes pricing for European calls and puts, plus a small demo plot.

These functions are imported by other modules (delta hedging, smile, etc.).
The plotting part only runs when this file is executed as a script.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.stats import norm


def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> Tuple[float, float]:
    if sigma <= 0.0 or T <= 0.0:
        return float("inf"), float("inf")  # degenerate; handled separately

    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_call_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black–Scholes price for a European call.
    """
    if sigma <= 0.0 or T <= 0.0:
        return max(S - K, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Black–Scholes price for a European put.
    """
    if sigma <= 0.0 or T <= 0.0:
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------
# Demo: plot call & put prices vs underlying when run as a script
# ---------------------------------------------------------------------
def main():
    import matplotlib.pyplot as plt

    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0

    S_grid = np.linspace(50, 150, 200)
    calls = [bs_call_price(S, K, r, sigma, T) for S in S_grid]
    puts = [bs_put_price(S, K, r, sigma, T) for S in S_grid]

    plt.figure(figsize=(8, 5))
    plt.plot(S_grid, calls, label="Call price")
    plt.plot(S_grid, puts, label="Put price")
    plt.axvline(K, color="grey", ls="--", lw=1, label="Strike K")
    plt.title("Black–Scholes call/put prices vs underlying")
    plt.xlabel("Spot S")
    plt.ylabel("Option price")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
