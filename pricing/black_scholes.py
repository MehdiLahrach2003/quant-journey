import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    """Calcule le prix d'une option européenne (call ou put) selon Black-Scholes."""
    if T <= 0 or sigma <= 0:
        raise ValueError("T et sigma doivent être positifs")

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type doit être 'call' ou 'put'")
    return price


def plot_payoff(S_range, K, option_type="call"):
    """Trace le payoff du call ou du put à maturité."""
    payoff = np.maximum(S_range - K, 0) if option_type == "call" else np.maximum(K - S_range, 0)
    plt.plot(S_range, payoff, label=f"Payoff {option_type}")
    plt.title(f"Payoff d'un {option_type.capitalize()} (K={K})")
    plt.xlabel("Prix à maturité $S_T$")
    plt.ylabel("Profit / Perte")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Paramètres
    S0 = 100     # prix spot
    K = 100      # strike
    T = 1.0      # 1 an
    r = 0.03     # taux sans risque
    sigma = 0.2  # volatilité

    # Prix call & put
    call = black_scholes_price(S0, K, T, r, sigma, "call")
    put = black_scholes_price(S0, K, T, r, sigma, "put")

    print(f"Prix call BS : {call:.2f}")
    print(f"Prix put BS  : {put:.2f}")

    # Payoff visuel
    S_range = np.linspace(50, 150, 200)
    plot_payoff(S_range, K, "call")
    plot_payoff(S_range, K, "put")