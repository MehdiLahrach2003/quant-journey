# pricing/monte_carlo_gbm.py
# Monte Carlo GBM + pricing call/put avec variance antithétique et IC 95%

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple

@dataclass
class MCResult:
    price: float
    std_err: float
    ci_low: float
    ci_high: float
    n_paths: int

def simulate_gbm_paths(
    S0: float, r: float, sigma: float, T: float, steps: int, n_paths: int, seed: int | None = 42
) -> np.ndarray:
    """
    Simule des trajectoires GBM:
        dS_t = r S_t dt + sigma S_t dW_t
    Retourne un array (steps+1, n_paths) des prix.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dt = T / steps
    # Incréments brownien ~ N(0, dt)
    Z = rng.standard_normal(size=(steps, n_paths))
    # Schéma exact (Euler–Maruyama exact pour GBM via solution fermée)
    drift = (r - 0.5 * sigma**2) * dt
    diff = sigma * np.sqrt(dt) * Z
    log_increments = drift + diff  # shape (steps, n_paths)

    # S_t = S0 * exp( somme des increments log )
    log_S = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_increments, axis=0)])
    S = S0 * np.exp(log_S)
    return S  # (steps+1, n_paths)

def _discount(r: float, T: float) -> float:
    return np.exp(-r * T)

def price_european_call_mc(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_paths: int = 100_000, steps: int = 1,
    antithetic: bool = True, seed: int | None = 42
) -> MCResult:
    """
    Pricing MC d'un call européen par payoff max(S_T - K, 0).
    steps=1 suffit (on n'a besoin que de S_T), mais on laisse steps variable.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    if steps == 1:
        # On simule directement S_T sans stocker toute la trajectoire
        dt = T
        n_eff = n_paths
        if antithetic:
            # Variance antithétique: on double les chemins avec -Z
            Z = rng.standard_normal(n_paths // 2)
            Z = np.concatenate([Z, -Z])
            n_eff = Z.size
        else:
            Z = rng.standard_normal(n_paths)

        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    else:
        # On simule la trajectoire complète et on prend S_T
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]
        n_eff = n_paths

    payoff = np.maximum(ST - K, 0.0)
    disc = _discount(r, T)
    price = disc * payoff.mean()
    # Erreur-type et IC 95%
    std = payoff.std(ddof=1)
    std_err = disc * std / np.sqrt(n_eff)
    ci_low = price - 1.96 * std_err
    ci_high = price + 1.96 * std_err
    return MCResult(price, std_err, ci_low, ci_high, n_eff)

def price_european_put_mc(
    S0: float, K: float, r: float, sigma: float, T: float,
    n_paths: int = 100_000, steps: int = 1,
    antithetic: bool = True, seed: int | None = 42
) -> MCResult:
    """
    Pricing MC d'un put européen par payoff max(K - S_T, 0).
    """
    # Réutilise le simulateur du call pour ST
    res_call_like = price_european_call_mc(S0, K, r, sigma, T, n_paths, steps, antithetic, seed)
    # On ne peut pas déduire put du call ici sans S_T, donc on refait le payoff
    # (pour garder la structure simple, on recalcule ST rapidement)
    if steps == 1:
        rng = np.random.default_rng(seed)
        dt = T
        if antithetic:
            Z = rng.standard_normal(n_paths // 2)
            Z = np.concatenate([Z, -Z])
        else:
            Z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
        payoff = np.maximum(K - ST, 0.0)
        disc = _discount(r, T)
        price = disc * payoff.mean()
        std = payoff.std(ddof=1)
        std_err = disc * std / np.sqrt(ST.size)
        ci_low = price - 1.96 * std_err
        ci_high = price + 1.96 * std_err
        return MCResult(price, std_err, ci_low, ci_high, ST.size)
    else:
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]
        payoff = np.maximum(K - ST, 0.0)
        disc = _discount(r, T)
        price = disc * payoff.mean()
        std = payoff.std(ddof=1)
        std_err = disc * std / np.sqrt(n_paths)
        ci_low = price - 1.96 * std_err
        ci_high = price + 1.96 * std_err
        return MCResult(price, std_err, ci_low, ci_high, n_paths)

def demo_convergence(
    S0=100, K=100, r=0.03, sigma=0.2, T=1.0, steps=1, antithetic=True, seed=42
):
    """
    Trace la convergence du prix MC en fonction du nombre de chemins.
    """
    Ns = np.array([500, 1_000, 2_000, 5_000, 10_000, 20_000, 50_000])
    prices = []
    errs = []
    for n in Ns:
        res = price_european_call_mc(S0, K, r, sigma, T, n_paths=n, steps=steps, antithetic=antithetic, seed=seed)
        prices.append(res.price)
        errs.append(res.std_err)
    prices = np.array(prices)
    errs = np.array(errs)

    plt.figure(figsize=(7,4))
    plt.plot(Ns, prices, marker='o', label="Prix MC (call)")
    plt.fill_between(Ns, prices - 1.96*errs, prices + 1.96*errs, alpha=0.2, label="IC 95%")
    plt.xscale("log")
    plt.xlabel("Nombre de chemins (log)")
    plt.ylabel("Prix")
    plt.title("Convergence Monte Carlo (call européen)")
    plt.legend()
    plt.tight_layout()
    plt.show()

def demo_hist_terminal(
    S0=100, r=0.03, sigma=0.2, T=1.0, steps=1, n_paths=50_000, seed=42
):
    """
    Histogramme de S_T.
    """
    if steps == 1:
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal(n_paths)
        ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    else:
        S = simulate_gbm_paths(S0, r, sigma, T, steps, n_paths, seed)
        ST = S[-1, :]

    plt.figure(figsize=(7,4))
    plt.hist(ST, bins=60, density=True, alpha=0.7)
    plt.xlabel("S_T")
    plt.ylabel("Densité")
    plt.title("Distribution Monte Carlo de S_T (GBM)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Paramètres de test
    S0, K, r, sigma, T = 100.0, 100.0, 0.03, 0.2, 1.0

    call_res = price_european_call_mc(S0, K, r, sigma, T, n_paths=100_000, steps=1, antithetic=True, seed=42)
    put_res  = price_european_put_mc (S0, K, r, sigma, T, n_paths=100_000, steps=1, antithetic=True, seed=42)

    print(f"Call MC: {call_res.price:.4f}  ± {1.96*call_res.std_err:.4f} (IC95%)  n={call_res.n_paths}")
    print(f"Put  MC: {put_res.price:.4f}  ± {1.96*put_res.std_err:.4f} (IC95%)  n={put_res.n_paths}")

    # Décommente pour voir les figures
    # demo_convergence(S0, K, r, sigma, T, steps=1, antithetic=True, seed=42)
    # demo_hist_terminal(S0, r, sigma, T, steps=1, n_paths=50_000, seed=42)
