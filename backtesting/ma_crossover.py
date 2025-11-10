# backtesting/ma_crossover.py
# SMA crossover signal: returns positions based on short/long moving averages.

import pandas as pd


def sma_crossover_positions(prices: pd.Series, short: int = 20, long: int = 100) -> pd.Series:
    """
    Build a simple SMA crossover signal.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by dates.
    short : int
        Window length for the short SMA.
    long : int
        Window length for the long SMA.

    Returns
    -------
    pd.Series
        Positions in {-1, 0, +1}, shifted by 1 bar to avoid lookahead.
        +1 if SMA_short > SMA_long, -1 if SMA_short < SMA_long, 0 otherwise.
    """
    px = prices.astype(float)
    sma_s = px.rolling(short, min_periods=1).mean()
    sma_l = px.rolling(long,  min_periods=1).mean()

    raw = (sma_s > sma_l).astype(int) - (sma_s < sma_l).astype(int)
    pos = raw.shift(1).fillna(0.0)
    pos.name = "position"
    return pos
