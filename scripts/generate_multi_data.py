# scripts/generate_multi_data.py
"""
Generate synthetic price series for several assets and save them
as CSV files in data/multi/.

Assets:
- AAPL
- MSFT
- SP500
- BTCUSD

Each CSV has:
    date, price

The paths are:
    data/multi/AAPL.csv
    data/multi/MSFT.csv
    data/multi/SP500.csv
    data/multi/BTCUSD.csv
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    n_days: int,
    start_date: str = "2015-01-01",
) -> pd.DataFrame:
    """
    Simple daily GBM simulation.

    dS = mu * S dt + sigma * S dW

    Parameters
    ----------
    S0 : float
        Initial price.
    mu : float
        Drift (annualised).
    sigma : float
        Volatility (annualised).
    n_days : int
        Number of calendar days.
    start_date : str
        Start date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame with columns:
        date, price
    """
    dt = 1.0 / 252.0  # daily step in years
    prices = np.empty(n_days, dtype=float)
    prices[0] = S0

    # Dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(days=i) for i in range(n_days)]

    # Simple GBM simulation (ignoring weekends/holidays)
    rng = np.random.default_rng(42)
    for t in range(1, n_days):
        z = rng.standard_normal()
        prices[t] = prices[t - 1] * np.exp(
            (mu - 0.5 * sigma * sigma) * dt + sigma * np.sqrt(dt) * z
        )

    df = pd.DataFrame({"date": dates, "price": prices})
    return df


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    out_dir = os.path.join(root, "data", "multi")
    os.makedirs(out_dir, exist_ok=True)

    n_days = 2000  # ~8 years de données

    specs = {
        "AAPL":  {"S0": 150.0, "mu": 0.12, "sigma": 0.25},
        "MSFT":  {"S0": 300.0, "mu": 0.10, "sigma": 0.20},
        "SP500": {"S0": 4000.0, "mu": 0.07, "sigma": 0.18},
        "BTCUSD": {"S0": 20000.0, "mu": 0.25, "sigma": 0.80},
    }

    for ticker, params in specs.items():
        df = simulate_gbm(
            S0=params["S0"],
            mu=params["mu"],
            sigma=params["sigma"],
            n_days=n_days,
            start_date="2015-01-01",
        )

        out_path = os.path.join(out_dir, f"{ticker}.csv")
        df.to_csv(out_path, index=False)
        print(f"[OK] Saved {ticker} to {out_path}")

    print("\nDone. You can now run scripts/run_multi_asset.py")


if __name__ == "__main__":
    main()
