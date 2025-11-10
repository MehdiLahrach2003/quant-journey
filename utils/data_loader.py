# utils/data_loader.py
# Loads prices from /data or generates synthetic data if missing.

import os
import numpy as np
import pandas as pd


def load_prices(filename: str = "prices.csv") -> pd.DataFrame:
    """
    Load a price DataFrame with a 'price' column indexed by dates.
    If /data/<filename> is missing, generate a synthetic GBM-like series.

    Returns
    -------
    pd.DataFrame
        Columns: ['price'], index: DatetimeIndex
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)

    if not os.path.exists(data_path):
        print("⚠️  File not found, generating synthetic prices...")
        # Synthetic GBM-like path (suitable length and volatility)
        rng = np.random.default_rng(42)
        n = 1200
        S0, mu, sigma = 100.0, 0.10, 0.20
        dt = 1.0 / 252.0
        z = rng.normal(size=n)
        r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        s = S0 * np.exp(np.cumsum(r))
        idx = pd.bdate_range("2020-01-01", periods=n)
        df = pd.DataFrame({"price": s}, index=idx)
        return df

    df = pd.read_csv(data_path)
    # Be tolerant to various column names
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", list(df.columns)[0])
    price_col = cols.get("close", list(df.columns)[-1])

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.rename(columns={price_col: "price"})[["price"]].astype(float)
    return df
