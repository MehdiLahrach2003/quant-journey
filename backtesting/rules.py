# backtesting/rules.py
# Stop/Take-Profit overlay (supports long/short, fractional exposure, optional trailing)

from __future__ import annotations
import pandas as pd


def apply_stops(
    prices: pd.Series,
    positions: pd.Series,
    stop_loss_pct: float | None = 0.05,     # 5% loss
    take_profit_pct: float | None = 0.10,   # 10% gain
    trailing: bool = True,                  # trailing TP/SL when True
) -> pd.Series:
    """
    Apply stop-loss / take-profit to a raw position series.

    Rules (per continuous stream of exposure):
    - A "trade" starts when exposure crosses 0 -> non-zero.
    - While in a trade, compute PnL vs entry:
        * Long: r_t = price_t / entry_price - 1
        * Short: r_t = entry_price / price_t - 1
    - Hit stop-loss => flat (0) until the raw signal re-arms by crossing through 0 again.
    - Hit take-profit => same (flat) behavior.
    - trailing=True: the reference updates in favor of the trade:
        * Long: trail by highest close since entry (uses peak to compute drawdown).
        * Short: trail by lowest close since entry (mirror logic).
    - Fractional exposures are supported (e.g. vol-targeted 0.0..±3.0). Stop logic
      is driven by sign (long vs short). Magnitude is preserved until a stop flattens it.

    Parameters
    ----------
    prices : pd.Series
        Price series indexed by dates.
    positions : pd.Series
        Raw exposure (float allowed). Must be aligned to prices and already shifted.
    stop_loss_pct : float or None
        Stop-loss threshold (e.g., 0.05 => 5%). None disables SL.
    take_profit_pct : float or None
        Take-profit threshold (e.g., 0.10 => 10%). None disables TP.
    trailing : bool
        Use trailing reference (peak for longs, trough for shorts).

    Returns
    -------
    pd.Series
        Exposure after stops (0 when stopped, original exposure otherwise).
    """
    px = prices.astype(float).copy()
    pos = positions.astype(float).fillna(0.0).copy()
    pos_out = pos.copy()

    # State variables
    in_trade = False
    trade_side = 0            # +1 long, -1 short
    entry_price = None
    peak = None               # highest price since entry (long)
    trough = None             # lowest price since entry (short)
    armed = True              # requires a 0-crossing to re-arm after a stop

    idx = px.index

    for t in idx:
        raw = pos.loc[t]
        price = px.loc[t]
        sign = 0 if raw == 0 else (1 if raw > 0 else -1)

        # (Re-)arm logic: we only re-arm when the raw signal passes through 0
        if sign == 0:
            armed = True

        if not in_trade:
            # Can we open a trade?
            if armed and sign != 0:
                in_trade = True
                trade_side = sign
                entry_price = price
                peak = price
                trough = price
                pos_out.loc[t] = raw
            else:
                pos_out.loc[t] = 0.0
            continue

        # We are in a trade (trade_side set). If signal changed sign, treat as exit/flip.
        if sign == 0 or sign != trade_side:
            # Exit to 0 when raw says flat or flipped side (engine will account costs)
            in_trade = False
            trade_side = 0
            entry_price = None
            peak = None
            trough = None
            pos_out.loc[t] = 0.0
            # If flipped, do NOT auto-open on the same bar; a new bar will re-evaluate.
            if sign == 0:
                armed = True
            else:
                armed = False  # require a 0-cross to re-arm after flip
            continue

        # Still in same-side trade: update trail references
        if trailing:
            if trade_side > 0:
                peak = max(peak, price)
            else:
                trough = min(trough, price)

        # Compute running return vs entry (or vs trailing anchor if trailing)
        if trade_side > 0:
            ref = peak if trailing else entry_price
            # Drawdown from peak is (price/ref - 1)
            run_ret = price / (entry_price if not trailing else ref) - 1.0
            # For stop-loss on long with trailing we consider drawdown from peak:
            drawdown = price / peak - 1.0 if trailing else run_ret
        else:
            ref = trough if trailing else entry_price
            # For short: profit increases when price falls
            run_ret = (entry_price if not trailing else ref) / price - 1.0
            # For trailing SL on short we consider adverse move from trough:
            drawup = trough / price - 1.0 if trailing else run_ret  # negative when favorable

        hit_sl = False
        hit_tp = False

        if trade_side > 0:
            if stop_loss_pct is not None:
                # adverse move for long (drawdown if trailing, else run_ret < -SL)
                if (trailing and drawdown <= -stop_loss_pct) or (not trailing and run_ret <= -stop_loss_pct):
                    hit_sl = True
            if take_profit_pct is not None and run_ret >= take_profit_pct:
                hit_tp = True
        else:
            if stop_loss_pct is not None:
                # adverse move for short (price rising). If trailing: "drawup" >= SL
                if (trailing and drawup >= stop_loss_pct) or (not trailing and run_ret <= -stop_loss_pct):
                    hit_sl = True
            if take_profit_pct is not None and run_ret >= take_profit_pct:
                hit_tp = True

        if hit_sl or hit_tp:
            # Flat and disarm until raw crosses zero again
            in_trade = False
            trade_side = 0
            entry_price = None
            peak = None
            trough = None
            pos_out.loc[t] = 0.0
            armed = False
        else:
            # Keep raw exposure magnitude (can be fractional)
            pos_out.loc[t] = raw

    pos_out.name = "position_with_stops"
    return pos_out