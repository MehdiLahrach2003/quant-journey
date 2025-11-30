# utils/load_multi.py
"""
Load multiple assets from data/multi/ and return a dict {ticker: price_series}.
"""

from __future__ import annotations
import os
import pandas as pd


def load_multi_assets() -> dict[str, pd.Series]:
    """
    Load multiple CSV files from data/multi/, each containing at least:
        date, price

    Returns
    -------
    dict: mapping ticker -> pd.Series of prices
    """
    root = os.path.dirname(os.path.dirname(__file__))
    multi_dir = os.path.join(root, "data", "multi")

    if not os.path.isdir(multi_dir):
        raise FileNotFoundError(f"Folder not found: {multi_dir}")

    assets = {}

    for fname in os.listdir(multi_dir):
        if not fname.endswith(".csv"):
            continue

        ticker = fname.replace(".csv", "")
        path = os.path.join(multi_dir, fname)

        df = pd.read_csv(path, parse_dates=["date"])
        df = df.sort_values("date").set_index("date")

        if "price" not in df:
            raise ValueError(f"CSV {fname} must contain a 'price' column.")

        assets[ticker] = df["price"].astype(float)

    return assets
