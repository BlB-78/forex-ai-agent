from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from .config import Settings
from .strategy import add_indicators

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close")


def _normalise_download(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    out = raw.copy()
    if isinstance(out.columns, pd.MultiIndex):
        # For the one-ticker downloads used here, Yahoo can return a two-level column index.
        out.columns = out.columns.get_level_values(0)
    missing = set(REQUIRED_COLUMNS) - set(out.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")
    out = out.loc[:, list(REQUIRED_COLUMNS)].copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").dropna()
    out = out[(out["Open"] > 0) & (out["High"] > 0) & (out["Low"] > 0) & (out["Close"] > 0)]
    out = out[(out["High"] >= out[["Open", "Close", "Low"]].max(axis=1))]
    out = out[(out["Low"] <= out[["Open", "Close", "High"]].min(axis=1))]
    return out


def fetch_daily_data(
    settings: Settings, start: str | None = None, end: str | None = None, *, check_freshness: bool = True
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLC history using yfinance; not broker-specific bid/ask data."""
    settings.validate()
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError(
            "yfinance is not installed. Run `python -m pip install -e '.[dev]'` first."
        ) from exc
    begin = start or settings.start_date
    # yfinance end is exclusive. Tomorrow allows today's completed candle to be returned if available.
    end_exclusive = end or (datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat()
    result: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for pair, ticker in settings.ticker_symbols.items():
        try:
            raw = yf.download(
                ticker,
                start=begin,
                end=end_exclusive,
                interval="1d",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=False,
                group_by="column",
            )
            bars = _normalise_download(raw)
            if bars.empty:
                errors.append(f"{pair}: provider returned no usable bars")
                continue
            bars = add_indicators(bars, settings)
            result[pair] = bars
            print(f"{pair}: {len(bars)} daily bars from {bars.index.min().date()} to {bars.index.max().date()}")
        except Exception as exc:  # surface all pairs so the run can fail transparently
            errors.append(f"{pair}: {type(exc).__name__}: {exc}")

    if errors:
        details = "\n".join(f" - {message}" for message in errors)
        raise RuntimeError(
            "Market-data download was incomplete; refusing to run a partial portfolio.\n" + details
        )
    if not result:
        raise RuntimeError("No market data downloaded.")

    if check_freshness:
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        stale_pairs = []
        for pair, df in result.items():
            newest = pd.Timestamp(df.index.max())
            age_days = (today - pd.Timestamp(newest.date())).days
            if age_days > settings.max_data_age_calendar_days:
                stale_pairs.append(f"{pair}: latest candle {newest.date()} is {age_days} calendar days old")
        if stale_pairs:
            raise RuntimeError(
                "One or more currency pairs have stale data; refusing to process trades.\n"
                + "\n".join(f" - {item}" for item in stale_pairs)
            )
    return result
