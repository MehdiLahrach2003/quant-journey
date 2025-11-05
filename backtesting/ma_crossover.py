# backtesting/ma_crossover.py
# Stratégie: SMA courte > SMA longue -> long; sinon flat.
# Lit un CSV si disponible, sinon simule un prix via GBM.
# Sorties: stats de perf, equity curve & drawdown.

from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --------- Utils data ---------------------------------------------------------

def maybe_load_csv(csv_path: str, price_col: str = "Close") -> pd.DataFrame:
    """
    Si csv_path existe, le charge (doit contenir une colonne `price_col`).
    Sinon, simule une série de prix via GBM pour ~2 ans (252*2 points).
    """
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # essaie d'inférer un index datetime
        for col in ["Date", "date", "Datetime", "datetime", "Time", "time"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                df = df.set_index(col).sort_index()
                break
        if price_col not in df.columns:
            # si pas de colonne Close, prend la 1ère colonne numérique
            num_cols = df.select_dtypes(include=[np.number]).columns
            if len(num_cols) == 0:
                raise ValueError("CSV sans colonne de prix numérique détectée.")
            df = df.rename(columns={num_cols[0]: price_col})
        return df[[price_col]].rename(columns={price_col: "price"})
    else:
        # simulation GBM
        n = 252 * 2
        S0, r, sigma, T = 100.0, 0.02, 0.20, 2.0
        dt = T / n
        rng = np.random.default_rng(42)
        z = rng.standard_normal(n)
        log_incr = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
        log_path = np.cumsum(log_incr)
        price = S0 * np.exp(log_path)
        idx = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({"price": price}, index=idx)

# --------- Strategy & Backtest ------------------------------------------------

@dataclass
class BTStats:
    total_return: float
    annual_return: float
    annual_vol: float
    sharpe: float
    max_dd: float

def compute_drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return dd

def run_ma_crossover(
    df: pd.DataFrame,
    short: int = 20,
    long: int = 100,
    tc_bps: float = 1.0,     # coût par aller (en bps) sur notionnel
    cash_rate: float = 0.0   # rendement cash simple (optionnel)
):
    """
    Stratégie: long 1x quand SMA_short > SMA_long, sinon 0 (flat).
    Retours calculés en pourcentage jour/jour.
    """
    assert long > short >= 1, "Paramètres MA invalides."
    px = df["price"].astype(float)

    # retours simples (close-to-close)
    rets = px.pct_change().fillna(0.0)

    # SMAs
    sma_s = px.rolling(short, min_periods=short).mean()
    sma_l = px.rolling(long,  min_periods=long).mean()

    # signal brut: 1 si SMA_short > SMA_long, sinon 0
    signal = (sma_s > sma_l).astype(float)
    signal = signal.reindex_like(px).fillna(0.0)

    # transitions -> coûts de transaction (|Δposition| * bps)
    pos = signal.shift(1).fillna(0.0)  # position appliquée au jour t (décalée)
    dpos = pos.diff().fillna(pos)      # taille du trade du jour
    tc = np.abs(dpos) * (tc_bps / 1e4) # coût proportionnel en bps

    # cash carry optionnel (si tu veux rémunérer le cash quand pos=0)
    carry = (1.0 - pos) * (cash_rate / 252.0)

    # P&L quotidien = position * retour - coûts - + carry
    pnl = pos * rets - tc + carry

    equity = (1.0 + pnl).cumprod()
    dd = compute_drawdown(equity)

    # stats annualisées
    n = len(pnl)
    if n == 0:
        raise ValueError("Série vide.")
    ann_ret = (equity.iloc[-1])**(252.0 / n) - 1.0
    ann_vol = pnl.std(ddof=1) * np.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = dd.min()

    stats = BTStats(
        total_return=float(equity.iloc[-1] - 1.0),
        annual_return=float(ann_ret),
        annual_vol=float(ann_vol),
        sharpe=float(sharpe),
        max_dd=float(max_dd),
    )
    out = pd.DataFrame({
        "price": px,
        "rets": rets,
        "sma_short": sma_s,
        "sma_long": sma_l,
        "pos": pos,
        "pnl": pnl,
        "equity": equity,
        "drawdown": dd,
    })
    return out, stats

# --------- Plots --------------------------------------------------------------

def plot_equity_and_dd(out: pd.DataFrame):
    fig = plt.figure(figsize=(9, 5))
    ax1 = plt.gca()
    out["equity"].plot(ax=ax1)
    ax1.set_title("Equity curve")
    ax1.set_ylabel("Equity (× initial)")
    plt.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(9, 3))
    ax2 = plt.gca()
    out["drawdown"].plot(ax=ax2)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Fraction")
    plt.tight_layout()
    plt.show()

# --------- Script -------------------------------------------------------------

if __name__ == "__main__":
    # 1) Charger données: mets un CSV dans data/ (ex: data/SPY.csv avec colonne Close)
    csv_path = "data/SPY.csv"  # change si tu as un autre fichier
    df = maybe_load_csv(csv_path, price_col="Close")

    # 2) Paramètres stratégie
    short, long = 20, 100
    tc_bps = 1.0
    cash_rate = 0.0

    # 3) Backtest
    out, stats = run_ma_crossover(df, short=short, long=long, tc_bps=tc_bps, cash_rate=cash_rate)

    # 4) Résumé
    print("\nSMA Crossover results")
    print(f"Total return : {stats.total_return: .2%}")
    print(f"Annual return: {stats.annual_return: .2%}")
    print(f"Annual vol   : {stats.annual_vol: .2%}")
    print(f"Sharpe       : {stats.sharpe: .2f}")
    print(f"Max drawdown : {stats.max_dd: .2%}")

    # 5) Graphiques
    plot_equity_and_dd(out)
