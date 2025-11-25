# scripts/run_delta_hedge.py
# Simple delta-hedging simulation for a European call under Black–Scholes.

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# Make project root importable (so "pricing.*" works)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import bs_call_price, bs_call_delta


# --------------------------------------------------------------------
# 1) GBM price simulation
# --------------------------------------------------------------------
def simulate_gbm_path(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_steps: int,
    seed: int | None = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate a single geometric Brownian motion (GBM) path.

    dS_t = r S_t dt + sigma S_t dW_t

    Parameters
    ----------
    S0 : float
        Initial spot.
    r : float
        Risk-free rate (drift under risk-neutral).
    sigma : float
        Volatility.
    T : float
        Total time in years.
    n_steps : int
        Number of time steps.
    seed : int or None
        Random seed (for reproducibility).

    Returns
    -------
    t : np.ndarray
        Time grid from 0 to T (length n_steps + 1).
    S : np.ndarray
        Simulated spot path (same length as t).
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)
    S = np.empty(n_steps + 1, dtype=float)
    S[0] = S0

    for i in range(1, n_steps + 1):
        z = rng.standard_normal()
        # Exact GBM step (log-normal)
        S[i] = S[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    return t, S


# --------------------------------------------------------------------
# 2) Delta-hedging simulation
# --------------------------------------------------------------------
def delta_hedge_path(
    S: np.ndarray,
    t: np.ndarray,
    K: float,
    r: float,
    sigma: float,
    T: float,
) -> dict:
    """
    Simulate the P&L of a delta hedge for a short European call.

    Setup:
    - At t=0, we SELL 1 call at BS price C0 (short option).
    - We receive premium C0 and keep it as cash.
    - At each time step, we adjust our stock position to match Delta.
      (we ignore interest on cash for simplicity)

    At maturity:
    - Our portfolio value = cash + stock_position * S_T
    - We must pay the payoff of the short call: max(S_T - K, 0)
    - Hedging error = portfolio_value - payoff

    Returns
    -------
    dict containing:
        "option_values"   : np.ndarray of call prices along the path
        "deltas"          : np.ndarray of deltas
        "cash"            : np.ndarray of cash account
        "stock_pos"       : np.ndarray of stock positions
        "portfolio_value" : np.ndarray of replication portfolio value
        "payoff"          : float (option payoff at maturity)
        "hedge_error"     : float (final hedging error)
    """
    n_steps = len(S) - 1
    dt = T / n_steps

    option_values = np.zeros_like(S)
    deltas = np.zeros_like(S)
    cash = np.zeros_like(S)
    stock_pos = np.zeros_like(S)
    portfolio_value = np.zeros_like(S)

    # --- t = 0 : short 1 call, receive premium, no stock yet ---
    tau0 = T - t[0]
    C0 = bs_call_price(S[0], K, r, sigma, tau0)
    option_values[0] = C0

    # We are short the call: receive C0 as cash
    cash[0] = C0
    stock_pos[0] = 0.0

    # First delta hedge: buy Delta shares to hedge the short call
    deltas[0] = bs_call_delta(S[0], K, r, sigma, tau0)
    trade = deltas[0] - stock_pos[0]
    cash[0] -= trade * S[0]
    stock_pos[0] = deltas[0]

    # Portfolio value (replicating portfolio)
    portfolio_value[0] = cash[0] + stock_pos[0] * S[0]

    # --- Iterate over time steps ---
    for i in range(1, n_steps + 1):
        tau = max(T - t[i], 0.0)

        if tau > 0.0:
            # Call value and delta before maturity
            option_values[i] = bs_call_price(S[i], K, r, sigma, tau)
            deltas[i] = bs_call_delta(S[i], K, r, sigma, tau)
        else:
            # At maturity: option value = payoff
            option_values[i] = max(S[i] - K, 0.0)
            deltas[i] = 0.0  # after maturity there is no delta

        # Rebalance hedge (we ignore interest on cash for simplicity)
        trade = deltas[i] - stock_pos[i - 1]
        cash[i] = cash[i - 1] - trade * S[i]
        stock_pos[i] = stock_pos[i - 1] + trade

        # Portfolio value = cash + stock position
        portfolio_value[i] = cash[i] + stock_pos[i] * S[i]

    # --- Final payoff and hedging error ---
    payoff = max(S[-1] - K, 0.0)         # we are short the call
    hedge_error = portfolio_value[-1] - payoff

    return {
        "option_values": option_values,
        "deltas": deltas,
        "cash": cash,
        "stock_pos": stock_pos,
        "portfolio_value": portfolio_value,
        "payoff": payoff,
        "hedge_error": hedge_error,
    }


# --------------------------------------------------------------------
# 3) Main + plotting
# --------------------------------------------------------------------
def main():
    # Model and option parameters
    S0 = 100.0
    K = 100.0
    r = 0.01
    sigma = 0.20
    T = 1.0
    n_steps = 100

    # 1) Simulate one GBM path
    t, S = simulate_gbm_path(S0, r, sigma, T, n_steps, seed=42)

    # 2) Run delta hedge on this path
    res = delta_hedge_path(S, t, K, r, sigma, T)

    option_values = res["option_values"]
    portfolio_value = res["portfolio_value"]
    hedge_error = res["hedge_error"]
    payoff = res["payoff"]

    print("\n===== Delta hedge result (single path) =====")
    print(f"Final spot S_T      : {S[-1]:.4f}")
    print(f"Call payoff         : {payoff:.4f}")
    print(f"Final portfolio     : {portfolio_value[-1]:.4f}")
    print(f"Hedging error       : {hedge_error:.4f}")

    # 3) Plot S, option price, and hedging portfolio
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Spot
    ax = axes[0]
    ax.plot(t, S, label="Spot S_t")
    ax.axhline(K, color="grey", ls="--", lw=1, label="Strike K")
    ax.set_ylabel("Spot")
    ax.set_title("Delta-hedging simulation (single path)")
    ax.legend()
    ax.grid(alpha=0.3)

    # Option value
    ax = axes[1]
    ax.plot(t, option_values, label="Call value")
    ax.set_ylabel("Option price")
    ax.legend()
    ax.grid(alpha=0.3)

    # Portfolio vs payoff
    ax = axes[2]
    ax.plot(t, portfolio_value, label="Hedging portfolio value")
    ax.axhline(payoff, color="red", ls="--", lw=1.2, label="Call payoff at T")
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save plot to data/
    out_png = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "delta_hedge_single_path.png")
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Delta-hedge plot saved → {out_png}")

    plt.show()


if __name__ == "__main__":
    main()
