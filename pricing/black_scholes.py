# pricing/black_scholes.py
"""
Black–Scholes pricing helpers for European options.
Vectorised with NumPy to work on scalars or arrays.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T):
    """
    Internal helper: compute Black–Scholes d1 and d2.
    Works for floats or NumPy arrays.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    r = float(r)
    sigma = float(sigma)
    T = float(T)

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return d1, d2


def bs_call_price(S, K, r, sigma, T):
    """
    Black–Scholes price of a European CALL.

    Parameters
    ----------
    S : float or array
        Spot price(s).
    K : float or array
        Strike(s).
    r : float
        Risk-free rate (annual, continuously compounded).
    sigma : float
        Volatility (annual).
    T : float
        Time to maturity (in years).

    Returns
    -------
    float or np.ndarray
        Call option price(s).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    d1, d2 = _d1_d2(S, K, r, sigma, T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, r, sigma, T):
    """
    Black–Scholes price of a European PUT.

    Put–call parity: P = C + K e^{-rT} − S
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    call = bs_call_price(S, K, r, sigma, T)
    return call + K * np.exp(-r * T) - S


# Optional aliases if some old code uses other names
bs_call = bs_call_price
bs_put = bs_put_price
