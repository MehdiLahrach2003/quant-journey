# pricing/delta_hedge_mc.py
"""
Monte Carlo study of Black–Scholes delta-hedging error.

- Simulate many GBM paths for the underlying.
- Delta-hedge a European call on each path with discrete rebalancing.
- Collect the hedging error (replicating portfolio - payoff at T).
- Plot the distribution of hedging PnL.

You can run this file directly from VS Code (Run button) or:
    python pricing/delta_hedge_mc.py
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Import Black–Scholes call price + delta
# ----------------------------------------------------------------------
try:
    # When pricing is used as a package (from scripts/)
    from .black_scholes import bs_call_price, bs_call_delta
except ImportError:  # When you run this file directly from VS Code
    from black_scholes import bs_call_price, bs_call_delta  # type: ignore


# ----------------------------------------------------------------------
# Core simulation: one delta-hedged path
# ----------------------------------------------------------------------
def simulate_delta_hedge_path(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Simulate one GBM path and delta-hedge a call option along the way.

    Parameters
    ----------
    S0 : float
        Initial spot price.
    K : float
        Strike of the call.
    r : float
        Risk-free rate (continuously compounded).
    sigma : float
        Volatility of the underlying.
    T : float
        Maturity in years.
    n_steps : int
        Number of rebalancing dates (time grid of size n_steps+1).
    rng : np.random.Generator
        NumPy random number generator.

    Returns
    -------
    times : np.ndarray
        Time grid from 0 to T (n_steps+1 points).
    spot : np.ndarray
        Simulated GBM path for the underlying.
    portfolio : np.ndarray
        Value of the replicating portfolio over time.
    hedging_error : float
        Final difference: portfolio_T - payoff_T.
    """
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    # --- 1) Simulate GBM path for S_t ---
    times = np.linspace(0.0, T, n_steps + 1)
    spot = np.empty(n_steps + 1, dtype=float)
    spot[0] = S0

    for i in range(1, n_steps + 1):
        z = rng.normal()
        spot[i] = spot[i - 1] * math.exp(
            (r - 0.5 * sigma * sigma) * dt + sigma * sqrt_dt * z
        )

    # --- 2) Delta-hedging strategy ---
    portfolio = np.empty(n_steps + 1, dtype=float)

    # Initial option price and delta at t = 0
    V0 = bs_call_price(S0, K, r, sigma, T)
    delta0 = bs_call_delta(S0, K, r, sigma, T)

    # We hold delta0 shares and finance the rest by borrowing/lending at rate r
    shares = delta0
    cash = V0 - shares * S0  # can be negative (borrowing)
    portfolio[0] = V0

    # Rebalance at each step
    for i in range(1, n_steps + 1):
        t_i = times[i]
        S_i = spot[i]

        # Cash grows at risk-free rate between rebalancings
        cash *= math.exp(r * dt)

        # Value of portfolio *just before* rebalancing
        portfolio[i] = shares * S_i + cash

        # If not at maturity, update delta and rebalance
        if i < n_steps:
            tau = T - t_i  # time to maturity
            new_delta = bs_call_delta(S_i, K, r, sigma, tau)

            # Buy/sell underlying to move from old delta to new delta
            d_shares = new_delta - shares
            cash -= d_shares * S_i  # funding the trade from the cash account
            shares = new_delta

    # --- 3) Payoff at maturity and hedging error ---
    payoff_T = max(spot[-1] - K, 0.0)
    portfolio_T = portfolio[-1]
    hedging_error = portfolio_T - payoff_T

    return times, spot, portfolio, hedging_error


# ----------------------------------------------------------------------
# Monte Carlo loop on many paths
# ----------------------------------------------------------------------
def simulate_hedging_errors_mc(
    S0: float = 100.0,
    K: float = 100.0,
    r: float = 0.02,
    sigma: float = 0.2,
    T: float = 1.0,
    n_steps: int = 52,   # weekly rebalancing on 1 year
    n_paths: int = 10_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Run a Monte Carlo on hedging error for a delta-hedged call.

    Returns
    -------
    errors : np.ndarray
        Vector of size n_paths with portfolio_T - payoff_T for each path.
    """
    rng = np.random.default_rng(seed)
    errors = np.empty(n_paths, dtype=float)

    for i in range(n_paths):
        _, _, _, err = simulate_delta_hedge_path(
            S0=S0,
            K=K,
            r=r,
            sigma=sigma,
            T=T,
            n_steps=n_steps,
            rng=rng,
        )
        errors[i] = err

    return errors


# ----------------------------------------------------------------------
# Demo when run as a script
# ----------------------------------------------------------------------
def main():
    # Parameters for the study
    S0 = 100.0
    K = 100.0
    r = 0.02
    sigma = 0.2
    T = 1.0
    n_steps = 52       # weekly
    n_paths = 10_000   # Monte Carlo size

    errors = simulate_hedging_errors_mc(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=123,
    )

    # Basic stats
    mean_err = float(errors.mean())
    std_err = float(errors.std(ddof=1))
    q05, q50, q95 = np.quantile(errors, [0.05, 0.5, 0.95])

    print("\n=== Delta-hedging error Monte Carlo ===")
    print(f"Paths           : {n_paths}")
    print(f"Rebalancing     : {n_steps} steps over T = {T}y")
    print(f"Mean error      : {mean_err:.6f}")
    print(f"Std of error    : {std_err:.6f}")
    print(f"5% / 50% / 95%  : {q05:.6f}  {q50:.6f}  {q95:.6f}")

    # Histogram of hedging PnL
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, density=True, alpha=0.7)
    plt.axvline(0.0, color="black", lw=1.2, ls="--", label="Perfect hedge (0)")
    plt.title("Distribution of delta-hedging error (portfolio_T - payoff_T)")
    plt.xlabel("Hedging error")
    plt.ylabel("Density")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
