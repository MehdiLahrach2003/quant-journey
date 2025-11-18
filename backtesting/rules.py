# backtesting/rules.py
# Simple trading rules: stop-loss / take-profit on top of a base signal.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class StopConfig:
    """
    Configuration for stop-loss and take-profit rules.

    pct values are expressed in decimals:
    - 0.05  = 5% move against you (stop-loss)
    - 0.10  = 10% move in your favour (take-profit)
    """
    stop_loss_pct: Optional[float] = None    # e.g. 0.05 for -5%
    take_profit_pct: Optional[float] = None  # e.g. 0.10 for +10%


def apply_stop_loss_take_profit(
    prices: pd.Series,
    base_positions: pd.Series,
    config: StopConfig,
) -> pd.Series:
    """
    Apply stop-loss / take-profit rules on top of a base position series.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by dates (close-to-close).
    base_positions : pd.Series
        Raw trading signal (e.g. from SMA crossover), values in {-1, 0, +1}.
        We assume this is already shifted by 1 bar to avoid lookahead.
    config : StopConfig
        Stop-loss and take-profit configuration.

    Returns
    -------
    pd.Series
        New position series in {-1, 0, +1} where stops have been enforced.

    Notes
    -----
    - We work at end-of-day resolution: if a stop is hit on day t, we
      assume we exit at the close of day t (approximation).
    - Logic is deliberately simple and explicit (loop over time) so that
      it stays easy to understand and debug.
    """
    prices = prices.astype(float)
    pos_raw = base_positions.reindex(prices.index).fillna(0.0)

    stop_loss = config.stop_loss_pct
    take_profit = config.take_profit_pct

    pos = pos_raw.copy()
    pos.values[:] = 0.0  # we will rebuild positions bar by bar

    in_pos: float = 0.0         # current position: -1, 0, +1
    entry_price: Optional[float] = None

    for i, (dt, price) in enumerate(prices.items()):
        desired = float(pos_raw.iloc[i])

        # If we are flat, we can enter according to the raw signal
        if in_pos == 0.0:
            if desired != 0.0:
                in_pos = desired
                entry_price = float(price)
            pos.iloc[i] = in_pos
            continue

        # We are currently in a position
        assert entry_price is not None
        ret_since_entry = price / entry_price - 1.0

        hit_stop = (
            stop_loss is not None
            and ret_since_entry <= -abs(stop_loss)
        )
        hit_tp = (
            take_profit is not None
            and ret_since_entry >= abs(take_profit)
        )

        if hit_stop or hit_tp:
            # Exit the position because stop or take-profit is triggered
            in_pos = 0.0
            entry_price = None
            pos.iloc[i] = 0.0
            continue

        # No stop triggered → follow the base signal if it wants to close/reverse
        if desired == 0.0:
            # Signal says "flat": we exit
            in_pos = 0.0
            entry_price = None
        elif desired != in_pos:
            # Signal wants to reverse (from +1 to -1 or the opposite)
            in_pos = desired
            entry_price = float(price)

        pos.iloc[i] = in_pos

    pos.name = "position_with_stops"
    return pos
