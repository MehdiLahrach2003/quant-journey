# pricing/asian_option_mc.py
import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42):
    """Simule des trajectoires GBM"""
    np.random.seed(seed)
    dt = T / n_steps
    Z = np.random.normal(size=(n_paths, n_steps))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_S = np.cumsum(increments, axis=1)
    S = S0 * np.exp(log_S)
    S = np.hstack([np.full((n_paths, 1), S0), S])  # ajout du S0 initial
    return S


def price_asian_option_mc(S0, K, r, sigma, T, n_steps, n_paths, option_type="call"):
    """Prix Monte Carlo d'une option asiatique (moyenne arithmétique)"""
    paths = simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths)
    S_avg = np.mean(paths[:, 1:], axis=1)  # moyenne sur la trajectoire (hors S0)
    
    if option_type == "call":
        payoff = np.maximum(S_avg - K, 0)
    else:
        payoff = np.maximum(K - S_avg, 0)

    discounted_payoff = np.exp(-r * T) * payoff
    price = discounted_payoff.mean()
    stderr = discounted_payoff.std(ddof=1) / np.sqrt(n_paths)
    return price, stderr


def plot_sample_paths(S, T, n_show=10):
    """Affiche quelques trajectoires simulées"""
    n_steps = S.shape[1] - 1
    t = np.linspace(0, T, n_steps + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(t, S[:n_show].T, lw=1)
    plt.title("Trajectoires simulées pour l'option asiatique")
    plt.xlabel("Temps (années)")
    plt.ylabel("Prix du sous-jacent")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Paramètres
    S0 = 100
    K = 100
    T = 1.0
    r = 0.03
    sigma = 0.2
    n_steps = 252
    n_paths = 10000

    # Simulation et pricing
    price, stderr = price_asian_option_mc(S0, K, r, sigma, T, n_steps, n_paths, "call")
    print(f"Prix estimé du call asiatique : {price:.4f} ± {1.96*stderr:.4f} (IC 95%)")

    # Quelques trajectoires pour visualiser
    S = simulate_gbm_paths(S0, r, sigma, T, n_steps, 50)
    plot_sample_paths(S, T)