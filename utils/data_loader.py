# utils/data_loader.py
# Loads prices from /data or generates synthetic data if missing.

import os
from typing import Sequence, Dict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _load_or_simulate_price(
    csv_path: str,
    start_date: str = "2020-01-01",
    n_days: int = 1200,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load a single price CSV and normalize it to a DataFrame with a 'price'
    column indexed by dates. If the file does not exist, generate a
    synthetic GBM-like path.

    This is used by both `load_prices` and `load_multi_assets`.
    """
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found ({os.path.basename(csv_path)}), generating synthetic prices...")
        rng = np.random.default_rng(seed)
        S0, mu, sigma = 100.0, 0.10, 0.20
        dt = 1.0 / 252.0
        z = rng.normal(size=n_days)
        r = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        s = S0 * np.exp(np.cumsum(r))
        idx = pd.bdate_range(start_date, periods=n_days)
        df = pd.DataFrame({"price": s}, index=idx)
        return df

    df = pd.read_csv(csv_path)

    # Be tolerant to various column names
    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date", list(df.columns)[0])
    # common price columns: 'close', 'adj close', etc.
    price_col = (
        cols.get("close")
        or cols.get("adj close")
        or cols.get("adj_close")
        or cols.get("price")
        or list(df.columns)[-1]
    )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.rename(columns={price_col: "price"})[["price"]].astype(float)
    return df


# ---------------------------------------------------------------------
# Single-series loader (used by most scripts)
# ---------------------------------------------------------------------


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

    # Keep the original behaviour (2020 start, 1200 days, seed=42)
    df = _load_or_simulate_price(
        csv_path=data_path,
        start_date="2020-01-01",
        n_days=1200,
        seed=42,
    )
    return df


# ---------------------------------------------------------------------
# Multi-asset loader for portfolio / multi-asset scripts
# ---------------------------------------------------------------------


def load_multi_assets(symbols: Sequence[str]) -> pd.DataFrame:
    """
    Load multiple assets from /data as a single price DataFrame.

    For each symbol in `symbols`, this function expects a CSV file named
    '<SYMBOL>.csv' in the /data folder, with at least:
      - a date column (e.g. 'Date')
      - a price/close column (e.g. 'Close', 'Adj Close', 'price')

    If a file is missing, a synthetic GBM-like series is generated for
    that symbol so that the rest of the code still runs.

    Parameters
    ----------
    symbols : sequence of str
        Asset tickers, e.g. ["MSFT", "BTCUSD", "SP500", "AAPL"].

    Returns
    -------
    pd.DataFrame
        Columns: one column per symbol (MSFT, BTCUSD, ...),
        index: DatetimeIndex (business days), forward-filled and
        with any remaining NaNs dropped.
    """
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(base_path, "data")

    series_by_symbol: Dict[str, pd.Series] = {}

    # Use different seeds per symbol for synthetic generation
    base_seed = 1234

    for i, sym in enumerate(symbols):
        csv_path = os.path.join(data_dir, f"{sym}.csv")

        df_sym = _load_or_simulate_price(
            csv_path=csv_path,
            start_date="2015-01-01",  # longer history for multi-asset tests
            n_days=2000,
            seed=base_seed + i,
        )

        # Keep only the 'price' column and rename it to the symbol
        s = df_sym["price"].rename(sym)
        series_by_symbol[sym] = s

    # Align all symbols on a common date index
    df_all = pd.concat(
        [series_by_symbol[sym] for sym in symbols],
        axis=1,
    ).sort_index()

    # Forward-fill missing values (e.g. holidays) and drop any leading NaNs
    df_all = df_all.ffill().dropna()

    return df_all
