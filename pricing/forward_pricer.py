from math import *
import numpy as np
import matplotlib.pyplot as plt

def forward_price(S0, r, T):
    """Prix théorique du forward: F0 = S0 * e^{rT}"""
    return S0 * exp(r * T)

def plot_payoff(K, S_min, S_max, n):
    S_T = np.linspace(S_min, S_max, n)
    payoff = S_T - K
    plt.figure(figsize=(7, 4))
    plt.plot(S_T, payoff, label="Payoff long forward")
    plt.axhline(0, color="black", lw=1)
    plt.axvline(K, color="gray", ls="--", label="Strike")
    plt.title("Payoff Forward Contract")
    plt.xlabel("Prix à maturité (S_T)")
    plt.ylabel("Profit / Perte")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
S_min = 50
S_max = 150
n = 200
S0, r, T, K = 100, 0.03, 1.0, 100
F0 = forward_price(S0, r, T)
print(f"Prix théorique du forward : {F0:.2f}")
plot_payoff(K, S_min, S_max, n)