# pricing/black_scholes.py
"""
Black–Scholes pricing for European calls and puts, plus basic Greeks.

These functions are imported by other modules (delta hedging, smile, etc.).
The plotting demo only runs when this file is executed as a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------


def _d1_d2(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the standard Black–Scholes d1 and d2 terms.

    Parameters
    ----------
    S : float or np.ndarray
        Spot price of the underlying.
    K : float or np.ndarray
        Strike price.
    r : float
        Risk-free rate (continuously compounded).
    sigma : float
        Volatility (annualised).
    T : float
        Time to maturity in years.

    Returns
    -------
    d1, d2 : np.ndarray
        The usual Black–Scholes d1 and d2 terms.
    """
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError("sigma and T must be strictly positive.")

    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive.")

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


# ---------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------


def bs_call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Black–Scholes price for a European call.

    Works with scalars or NumPy arrays.
    """
    if sigma <= 0.0 or T <= 0.0:
        # Immediate expiry or zero vol → intrinsic value
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        return np.maximum(S_arr - K_arr, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Black–Scholes price for a European put.

    Works with scalars or NumPy arrays.
    """
    if sigma <= 0.0 or T <= 0.0:
        S_arr = np.asarray(S, dtype=float)
        K_arr = np.asarray(K, dtype=float)
        return np.maximum(K_arr - S_arr, 0.0)

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ---------------------------------------------------------------------
# Greeks (for the call; gamma/vega are same for the put)
# ---------------------------------------------------------------------


def bs_call_delta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Delta of a European call option.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm.cdf(d1)


def bs_call_gamma(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Gamma of a European call (and put).
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def bs_call_vega(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
    as_bp: bool = False,
) -> np.ndarray:
    """
    Vega of a European call (and put).

    If as_bp is True, returns sensitivity per 1bp of vol (i.e. divided by 10,000).
    """
    d1, _ = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    vega = S * norm.pdf(d1) * np.sqrt(T)
    if as_bp:
        vega = vega / 10_000.0
    return vega


def bs_call_theta(
    S: float | np.ndarray,
    K: float | np.ndarray,
    r: float,
    sigma: float,
    T: float,
) -> np.ndarray:
    """
    Theta of a European call (per year unit of T).

    If you want "per day" theta, divide the result by 252.
    """
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)

    term1 = -S * norm.pdf(d1) * sigma / (2.0 * np.sqrt(T))
    term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
    return term1 + term2


# ---------------------------------------------------------------------
# Demo plot when running this file as a script
# ---------------------------------------------------------------------


def main() -> None:
    """
    Simple visual sanity check: call/put prices vs underlying.
    """
    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0

    S_grid = np.linspace(50, 150, 200)
    calls = bs_call_price(S_grid, K, r, sigma, T)
    puts = bs_put_price(S_grid, K, r, sigma, T)

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
