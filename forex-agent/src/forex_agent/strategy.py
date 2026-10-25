from __future__ import annotations

import pandas as pd

from .config import Settings


def add_indicators(bars: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Build non-forward-looking daily indicators and a discrete regime signal."""
    out = bars.copy().sort_index()
    close = out["Close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            out["High"].astype(float) - out["Low"].astype(float),
            (out["High"].astype(float) - prev_close).abs(),
            (out["Low"].astype(float) - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["SMA"] = close.rolling(settings.sma_days, min_periods=settings.sma_days).mean()
    out["MOMENTUM"] = close / close.shift(settings.momentum_days) - 1.0
    # Simple rolling ATR is used for transparent first-version behavior.
    out["ATR"] = true_range.rolling(settings.atr_days, min_periods=settings.atr_days).mean()
    long_condition = (close > out["SMA"]) & (out["MOMENTUM"] > 0)
    short_condition = (close < out["SMA"]) & (out["MOMENTUM"] < 0)
    out["SIGNAL"] = 0
    out.loc[long_condition, "SIGNAL"] = 1
    out.loc[short_condition, "SIGNAL"] = -1
    out["VALID"] = out[["SMA", "MOMENTUM", "ATR"]].notna().all(axis=1)
    out.loc[~out["VALID"], "SIGNAL"] = 0
    return out
