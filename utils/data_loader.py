import os
import pandas as pd
import numpy as np

def load_prices(filename="prices.csv"):
    base_path = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_path, "data", filename)

    if not os.path.exists(data_path):
        print("⚠️  Fichier introuvable, génération de données factices...")
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=1000, freq="B")
        prices = 100 + np.cumsum(np.random.randn(len(dates)))
        df = pd.DataFrame({"date": dates, "price": prices})
    else:
        df = pd.read_csv(data_path)
        df = df.rename(columns={"Date": "date", "Close": "price"})

    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    return df
