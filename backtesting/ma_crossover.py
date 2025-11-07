import pandas as pd

def sma_crossover_positions(prices: pd.Series, short_window=20, long_window=100) -> pd.Series:
    """
    Renvoie une position dans {-1, 0, +1} :
    +1 si SMA_short > SMA_long, -1 sinon
    """
    px = prices.astype(float)
    sma_s = px.rolling(short_window, min_periods=1).mean()
    sma_l = px.rolling(long_window, min_periods=1).mean()

    pos = (sma_s > sma_l).astype(int) - (sma_s < sma_l).astype(int)
    pos = pos.shift(1).fillna(0)  # éviter le lookahead bias
    pos.name = "position"
    return pos