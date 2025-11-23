# scripts/run_smile.py
# Build and plot a Black–Scholes volatility smile from synthetic option prices.

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------
# Make the project root importable (so "pricing.*" works when you Run)
# --------------------------------------------------------------------
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pricing.black_scholes import bs_call_price   # your BS call pricer
from pricing.implied_vol import implied_vol_bs    # <-- IMPORTANT: function name


# --------------------------------------------------------------------
# Smile construction
# --------------------------------------------------------------------
def build_smile(
    S0: float,
    r: float,
    T: float,
    true_sigma: float,
    n_strikes: int = 17,
):
    """
    Build a simple volatility smile from synthetic Black–Scholes prices.

    Parameters
    ----------
    S0 : float
        Spot price of the underlying.
    r : float
        Risk-free rate (continuously compounded).
    T : float
        Time to maturity in years.
    true_sigma : float
        "Real" volatility used to generate option prices.
    n_strikes : int
        Number of strikes between 60% and 140% moneyness.

    Returns
    -------
    strikes : np.ndarray
    true_vols : np.ndarray
    call_prices : np.ndarray
    implied_vols : np.ndarray
    """
    # Strike grid between 60% and 140% of spot
    K_min, K_max = 0.6 * S0, 1.4 * S0
    strikes = np.linspace(K_min, K_max, n_strikes)

    true_vols = 0.20 + 0.3 * ((strikes / S0 - 1.0) ** 2)
    call_prices = np.zeros_like(strikes, dtype=float)
    implied_vols = np.zeros_like(strikes, dtype=float)

    for i, K in enumerate(strikes):
        # 1) Generate call price from BS model
        price = bs_call_price(S0, K, r, true_sigma, T)
        call_prices[i] = price

        # 2) Invert price -> implied volatility
        iv = implied_vol_bs(price, S0, K, r, T)
        implied_vols[i] = iv

    return strikes, true_vols, call_prices, implied_vols


# --------------------------------------------------------------------
# Plotting helpers
# --------------------------------------------------------------------
def plot_smile(strikes, true_vols, implied_vols, S0: float):
    """
    Plot implied vol vs strike and vs moneyness.
    """
    moneyness = strikes / S0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Implied vol vs strike ---
    ax = axes[0]
    ax.plot(strikes, implied_vols, "o-", label="Implied vol (from prices)")
    ax.plot(strikes, true_vols, "--", label="True vol (input)")
    ax.set_xlabel("Strike K")
    ax.set_ylabel("Volatility")
    ax.set_title("Volatility smile – σ(K)")
    ax.grid(alpha=0.3)
    ax.legend()

    # --- Implied vol vs moneyness ---
    ax2 = axes[1]
    ax2.plot(moneyness, implied_vols, "o-", label="Implied vol")
    ax2.axvline(1.0, color="grey", ls="--", lw=0.8)
    ax2.set_xlabel("Moneyness K / S0")
    ax2.set_ylabel("Volatility")
    ax2.set_title("Volatility smile – σ(K/S0)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    # Optional: save figure to data/
    out_png = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data",
        "vol_smile.png",
    )
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Smile plot saved → {out_png}")

    plt.show()


# --------------------------------------------------------------------
# Main script
# --------------------------------------------------------------------
def main():
    # Model parameters (you can tweak them)
    S0 = 100.0
    r = 0.01
    T = 1.0
    true_sigma = 0.20

    strikes, true_vols, call_prices, implied_vols = build_smile(
        S0=S0,
        r=r,
        T=T,
        true_sigma=true_sigma,
        n_strikes=17,
    )

    # Quick text summary in terminal
    print("\nStrike   CallPrice   ImpliedVol")
    for K, c, iv in zip(strikes, call_prices, implied_vols):
        print(f"{K:7.2f}   {c:9.4f}   {iv:10.4f}")

    # Plot the smile
    plot_smile(strikes, true_vols, implied_vols, S0=S0)


if __name__ == "__main__":
    main()