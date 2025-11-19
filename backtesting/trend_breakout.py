# backtesting/trend_breakout.py
# Simple breakout / trend-following signals based on rolling high/low.

from __future__ import annotations

import pandas as pd


def breakout_positions(
    prices: pd.Series,
    lookback: int = 50,
    hold_bars: int = 5,
) -> pd.Series:
    """
    Generate breakout-based positions in {-1, 0, +1}.

    Idea
    ----
    - Go long  (+1) when price breaks above the rolling max of the last `lookback` days.
    - Go short (-1) when price breaks below the rolling min of the last `lookback` days.
    - Otherwise: keep previous position (trend-following behaviour).
    - Positions are shifted by 1 bar to avoid look-ahead bias.

    Parameters
    ----------
    prices : pd.Series
        Close prices indexed by date.
    lookback : int
        Window length for rolling high/low.
    hold_bars : int
        Minimum number of bars to keep a new position before allowing a flip
        (reduces whipsaw / over-trading).

    Returns
    -------
    pd.Series
        Position series in {-1, 0, +1}, named "position".
    """
    if not isinstance(prices, pd.Series):
        raise TypeError("`prices` must be a pandas Series.")

    px = prices.astype(float)

    # Rolling extremes (excluding current bar by using shift)
    rolling_high = px.shift(1).rolling(lookback, min_periods=lookback // 2).max()
    rolling_low = px.shift(1).rolling(lookback, min_periods=lookback // 2).min()

    # Raw signals: +1 breakout up, -1 breakout down, 0 otherwise
    long_signal = px > rolling_high
    short_signal = px < rolling_low

    raw = (
        long_signal.astype(int)
        - short_signal.astype(int)
    )  # +1, 0, or -1

    # Now enforce a minimum holding period to avoid over-trading
    pos = pd.Series(0.0, index=px.index, name="position")
    current_pos = 0.0
    bars_since_change = 0

    for i, (idx, sig) in enumerate(raw.items()):
        bars_since_change += 1

        # If we have a new signal and we respected the minimum hold,
        # we allow a change of position.
        if sig != 0 and (bars_since_change >= hold_bars or current_pos == 0):
            if sig != current_pos:
                current_pos = sig
                bars_since_change = 0

        pos.iloc[i] = current_pos

    # One-bar shift to avoid entering on the same bar as the signal
    pos = pos.shift(1).fillna(0.0)
    pos.name = "position"
    return pos
