# pricing/implied_vol.py
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt


def bs_price(S0, K, T, r, sigma, option_type="call"):
    """Prix d'une option européenne (Black-Scholes analytique)."""
    if sigma <= 0 or T <= 0:
        return max(0.0, S0 - K) if option_type == "call" else max(0.0, K - S0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


def implied_vol_bs(price, S0, K, T, r, option_type="call", tol=1e-6):
    """
    Calcule la volatilité implicite via recherche de racine (Brent).
    Renvoie np.nan si le prix n'est pas dans le domaine valide.
    """
    # bornes de recherche
    def objective(sigma):
        return bs_price(S0, K, T, r, sigma, option_type) - price

    try:
        # bornes typiques : 1% à 500%
        vol = brentq(objective, 1e-4, 5.0, xtol=tol, maxiter=100)
    except ValueError:
        vol = np.nan
    return vol


def generate_smile(S0=100, T=1.0, r=0.03, true_sigma=0.2, option_type="call"):
    """
    Génère un smile de volatilité implicite à partir de prix Black-Scholes théoriques
    (avec une 'vraie' vol) et les reconvertit via implied_vol_bs.
    """
    Ks = np.linspace(60, 140, 15)
    prices = [bs_price(S0, K, T, r, true_sigma, option_type) for K in Ks]
    implied_vols = [implied_vol_bs(p, S0, K, T, r, option_type) for p, K in zip(prices, Ks)]

    plt.figure(figsize=(7,4))
    plt.plot(Ks / S0, implied_vols, marker='o', label='Vol implicite')
    plt.axhline(true_sigma, color='k', ls='--', lw=1, label='Vol réelle')
    plt.xlabel("Moneyness (K / S₀)")
    plt.ylabel("Volatilité implicite")
    plt.title("Smile de volatilité implicite (Black-Scholes)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return Ks, implied_vols


if __name__ == "__main__":
    S0, T, r, sigma_true = 100, 1.0, 0.03, 0.2
    generate_smile(S0, T, r, sigma_true)
