# backtesting/regime.py
# Simple regime filters for trend-following strategies.

from dataclasses import dataclass
import pandas as pd


@dataclass
class RegimeConfig:
    """
    Configuration for the regime filter.

    Attributes
    ----------
    long_window : int
        Lookback window for the long-term moving average
        used to define the regime.
    """
    long_window: int = 200


def long_only_regime(prices: pd.Series, config: RegimeConfig) -> pd.Series:
    """
    Build a simple long-only regime filter based on a long-term SMA.

    Regime is:
      1.0 if price > SMA(long_window)
      0.0 otherwise

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by dates.
    config : RegimeConfig
        Configuration with long_window.

    Returns
    -------
    pd.Series
        Regime series in {0.0, 1.0}.
    """
    px = prices.astype(float)
    sma_long = px.rolling(config.long_window, min_periods=1).mean()

    regime = (px > sma_long).astype(float)
    regime.name = "regime"
    return regime
