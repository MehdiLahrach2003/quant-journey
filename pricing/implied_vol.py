# pricing/implied_vol.py
# Numerical implied volatility solver for Black–Scholes calls (Brent method)

from __future__ import annotations

import os
import sys
import numpy as np
from math import sqrt

# Make project importable when running this file directly
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# ABSOLUTE IMPORT — no relative import here
from pricing.black_scholes import bs_call_price


# --------------------------------------------------------------
# Implied volatility via Brent root finding
# --------------------------------------------------------------
def implied_vol_bs(
    price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    tol: float = 1e-8,
    max_iter: int = 100,
) -> float:
    """
    Find the Black–Scholes implied volatility using Brent's method.

    Parameters
    ----------
    price : float
        Observed market call price.
    S : float
        Spot price.
    K : float
        Strike price.
    r : float
        Continuous interest rate.
    T : float
        Time to maturity in years.
    tol : float
        Numerical tolerance for convergence.
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    float
        Implied volatility (annualized).
    """

    # Objective function: BS(sigma) - price
    def f(sig):
        return bs_call_price(S, K, r, sig, T) - price

    # Bounds for volatility search
    a, b = 1e-6, 5.0  # 500% volatility max allowed

    fa, fb = f(a), f(b)

    # If price is outside BS range, return NaN
    if fa * fb > 0:
        return np.nan

    # Brent’s method loop
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)

        if abs(fm) < tol:
            return m

        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm

    # Did not converge
    return np.nan


# --------------------------------------------------------------
# TEST (only executed if run directly)
# --------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    S0 = 100
    K = 100
    r = 0.00
    T = 1.0
    true_sigma = 0.20

    price = bs_call_price(S0, K, r, true_sigma, T)
    iv = implied_vol_bs(price, S0, K, r, T)

    print("True σ:", true_sigma)
    print("Implied σ:", iv)

    # Simple test plot
    plt.figure(figsize=(5,4))
    plt.title("Implied volatility test")
    plt.axhline(true_sigma, color="orange", label="true σ")
    plt.scatter([0],[iv], color="blue", label="implied σ")
    plt.legend()
    plt.tight_layout()
    plt.show()
