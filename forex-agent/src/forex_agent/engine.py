from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from .config import Settings

STATE_SCHEMA_VERSION = 1


def new_state(settings: Settings) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "strategy_version": settings.strategy_version,
        "initial_equity_usd": settings.initial_equity_usd,
        "cash_usd": settings.initial_equity_usd,
        "peak_equity_usd": settings.initial_equity_usd,
        "last_processed_date": None,
        "last_close_by_pair": {},
        "positions": {},
        "trades": [],
        "equity_curve": [],
        "daily_period": None,
        "daily_start_equity_usd": settings.initial_equity_usd,
        "daily_locked": False,
        "weekly_period": None,
        "weekly_start_equity_usd": settings.initial_equity_usd,
        "weekly_locked": False,
        "locked_signal_dates": [],
        "locked_weeks": [],
        "system_paused": False,
        "pause_reason": None,
    }


def _pair_pip(pair: str) -> float:
    return 0.01 if pair.endswith("JPY") else 0.0001


def _quote_to_usd(pair: str, price: float) -> float:
    if pair.endswith("USD"):
        return 1.0
    # For USD/JPY, USD/CHF and USD/CAD, quote-currency P&L converted approximately at the current rate.
    return 1.0 / max(float(price), 1e-12)


def _half_spread_plus_slippage_price(pair: str, settings: Settings) -> float:
    pips = settings.spread_pips[pair] / 2.0 + settings.slippage_pips_per_side
    return pips * _pair_pip(pair)


def _commission_usd(units: float, settings: Settings) -> float:
    return abs(units) / 100_000.0 * settings.commission_usd_per_100k_per_side


def _fill_price(pair: str, reference_price: float, direction: int, is_entry: bool, settings: Settings) -> float:
    adjustment = _half_spread_plus_slippage_price(pair, settings)
    # Long entries buy at a higher executable price; long exits sell at a lower price.
    sign = direction if is_entry else -direction
    return float(reference_price + sign * adjustment)


def _open_risk(state: dict[str, Any]) -> float:
    # Uses original planned risk. This is conservative even after a stop has trailed closer.
    return float(sum(float(p["initial_risk_usd"]) for p in state["positions"].values()))


def _unrealized_pnl(state: dict[str, Any], settings: Settings) -> float:
    total = 0.0
    for pair, pos in state["positions"].items():
        close = float(state["last_close_by_pair"].get(pair, pos["entry_price"]))
        # Estimate immediate liquidation, including spread/slippage and exit commission.
        exit_estimate = _fill_price(pair, close, int(pos["direction"]), False, settings)
        gross = int(pos["direction"]) * float(pos["units"]) * (exit_estimate - float(pos["entry_price"]))
        total += gross * _quote_to_usd(pair, exit_estimate) - _commission_usd(float(pos["units"]), settings)
    return total


def equity_now(state: dict[str, Any], settings: Settings) -> float:
    return float(state["cash_usd"]) + _unrealized_pnl(state, settings)


def _week_key(value: pd.Timestamp | date) -> str:
    iso = pd.Timestamp(value).isocalendar()
    return f"{int(iso.year)}-W{int(iso.week):02d}"


def _get_prior_rows(df: pd.DataFrame, when: pd.Timestamp) -> tuple[pd.Series | None, pd.Series | None]:
    pos = int(df.index.searchsorted(when))
    if pos >= len(df) or df.index[pos] != when:
        return None, None
    previous = df.iloc[pos - 1] if pos >= 1 else None
    previous_previous = df.iloc[pos - 2] if pos >= 2 else None
    return previous, previous_previous


def _close_position(
    state: dict[str, Any], pair: str, reference_price: float, when: pd.Timestamp,
    reason: str, settings: Settings,
) -> None:
    pos = state["positions"].pop(pair)
    direction = int(pos["direction"])
    units = float(pos["units"])
    exit_price = _fill_price(pair, float(reference_price), direction, False, settings)
    gross_usd = direction * units * (exit_price - float(pos["entry_price"])) * _quote_to_usd(pair, exit_price)
    exit_commission = _commission_usd(units, settings)
    net_trade = gross_usd - float(pos["entry_commission_usd"]) - exit_commission
    # The entry commission was already deducted from cash when the position opened.
    state["cash_usd"] += gross_usd - exit_commission
    state["trades"].append({
        "pair": pair,
        "direction": "LONG" if direction == 1 else "SHORT",
        "entry_date": pos["entry_date"],
        "exit_date": str(pd.Timestamp(when).date()),
        "entry_price": float(pos["entry_price"]),
        "exit_price": float(exit_price),
        "units": units,
        "initial_risk_usd": float(pos["initial_risk_usd"]),
        "gross_pnl_usd": float(gross_usd),
        "entry_commission_usd": float(pos["entry_commission_usd"]),
        "exit_commission_usd": float(exit_commission),
        "net_pnl_usd": float(net_trade),
        "r_multiple_net": float(net_trade / pos["initial_risk_usd"]) if pos["initial_risk_usd"] > 0 else np.nan,
        "holding_calendar_days": int((pd.Timestamp(when) - pd.Timestamp(pos["entry_date"])).days),
        "exit_reason": reason,
    })


def _stop_reference(pos: dict[str, Any], row: pd.Series) -> tuple[float | None, str | None]:
    direction = int(pos["direction"])
    open_price, high, low = float(row["Open"]), float(row["High"]), float(row["Low"])
    stop = float(pos["stop_price"])
    if direction == 1:
        if open_price <= stop:
            return open_price, "gap_stop"
        if low <= stop:
            return stop, "stop"
    else:
        if open_price >= stop:
            return open_price, "gap_stop"
        if high >= stop:
            return stop, "stop"
    return None, None


def _update_trailing_stop(pos: dict[str, Any], row: pd.Series, settings: Settings) -> None:
    atr = float(row["ATR"]) if pd.notna(row.get("ATR")) else float(pos["entry_atr"])
    high, low = float(row["High"]), float(row["Low"])
    direction = int(pos["direction"])
    if direction == 1:
        pos["high_water"] = max(float(pos["high_water"]), high)
        pos["stop_price"] = max(float(pos["stop_price"]), float(pos["high_water"]) - settings.atr_stop_multiple * atr)
    else:
        pos["low_water"] = min(float(pos["low_water"]), low)
        pos["stop_price"] = min(float(pos["stop_price"]), float(pos["low_water"]) + settings.atr_stop_multiple * atr)


def _can_open_from_signal(
    state: dict[str, Any], signal_date: pd.Timestamp, current_date: pd.Timestamp,
) -> bool:
    signal_day = str(pd.Timestamp(signal_date).date())
    if signal_day in set(state["locked_signal_dates"]):
        return False
    if _week_key(signal_date) in set(state["locked_weeks"]):
        return False
    if state["system_paused"] or state["daily_locked"] or state["weekly_locked"]:
        return False
    return True


def _check_risk_limits(
    state: dict[str, Any], marked_equity: float, current_date: pd.Timestamp, settings: Settings
) -> tuple[float, float, float]:
    """Apply daily, weekly and peak drawdown brakes to the current account mark."""
    state["peak_equity_usd"] = max(float(state["peak_equity_usd"]), float(marked_equity))
    peak = float(state["peak_equity_usd"])
    drawdown = marked_equity / peak - 1.0 if peak > 0 else -1.0
    day_start = float(state["daily_start_equity_usd"])
    week_start = float(state["weekly_start_equity_usd"])
    daily_return = marked_equity / day_start - 1.0 if day_start > 0 else -1.0
    weekly_return = marked_equity / week_start - 1.0 if week_start > 0 else -1.0
    day_key = str(current_date.date())
    week_key = _week_key(current_date)

    if daily_return <= -settings.daily_loss_limit_pct:
        state["daily_locked"] = True
        if day_key not in state["locked_signal_dates"]:
            state["locked_signal_dates"].append(day_key)
    if weekly_return <= -settings.weekly_loss_limit_pct:
        state["weekly_locked"] = True
        if week_key not in state["locked_weeks"]:
            state["locked_weeks"].append(week_key)
    if drawdown <= -settings.peak_drawdown_pause_pct:
        state["system_paused"] = True
        state["pause_reason"] = f"peak drawdown reached {drawdown:.2%} on {day_key}; manual review required"
    return float(drawdown), float(daily_return), float(weekly_return)


def _open_position(
    state: dict[str, Any], pair: str, direction: int, open_reference: float,
    atr: float, when: pd.Timestamp, settings: Settings,
) -> bool:
    if atr <= 0 or not np.isfinite(atr):
        return False
    equity = equity_now(state, settings)
    if equity <= 0:
        state["system_paused"] = True
        state["pause_reason"] = "non-positive account equity"
        return False

    unused_combined_budget = equity * settings.max_combined_open_risk_pct - _open_risk(state)
    risk_budget = min(equity * settings.risk_per_trade_pct, max(0.0, unused_combined_budget))
    if risk_budget <= 0:
        return False

    stop_distance = settings.atr_stop_multiple * atr
    fx_rate = _quote_to_usd(pair, open_reference)
    pip = _pair_pip(pair)
    round_trip_cost_per_unit = (
        (settings.spread_pips[pair] + 2.0 * settings.slippage_pips_per_side) * pip * fx_rate
        + 2.0 * settings.commission_usd_per_100k_per_side / 100_000.0
    )
    risk_per_unit = stop_distance * fx_rate + round_trip_cost_per_unit
    if risk_per_unit <= 0:
        return False
    units = float(math_floor_units(risk_budget / risk_per_unit))
    if units <= 0:
        return False

    entry_price = _fill_price(pair, open_reference, direction, True, settings)
    entry_commission = _commission_usd(units, settings)
    actual_budget = units * risk_per_unit
    stop_price = entry_price - direction * stop_distance
    state["cash_usd"] -= entry_commission
    state["positions"][pair] = {
        "direction": int(direction),
        "units": units,
        "entry_date": str(pd.Timestamp(when).date()),
        "entry_price": float(entry_price),
        "stop_price": float(stop_price),
        "entry_atr": float(atr),
        "high_water": float(entry_price),
        "low_water": float(entry_price),
        "initial_risk_usd": float(actual_budget),
        "entry_commission_usd": float(entry_commission),
    }
    return True


def math_floor_units(value: float) -> int:
    # FX base-currency units rather than broker lots; execution adapters must round to broker constraints.
    return max(0, int(np.floor(value)))


def advance_state(
    data: dict[str, pd.DataFrame],
    state: dict[str, Any],
    settings: Settings,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    seed_initial_entries: bool = False,
) -> dict[str, Any]:
    """Advance a paper/backtest state through new daily bars. No live orders are sent."""
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported state schema version; migrate or reset the state intentionally")
    if state.get("strategy_version") != settings.strategy_version:
        raise ValueError(
            "Saved state was created by a different strategy version. Keep it separate or deliberately reset paper state."
        )
    if not data:
        raise ValueError("No market data supplied")

    all_dates = sorted(set().union(*(set(df.index) for df in data.values())))
    last_processed = state.get("last_processed_date")
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    process_dates: list[pd.Timestamp] = []
    for item in all_dates:
        current = pd.Timestamp(item)
        if last_processed and current <= pd.Timestamp(last_processed):
            continue
        if start_ts is not None and current < start_ts:
            continue
        if end_ts is not None and current >= end_ts:
            continue
        process_dates.append(current)

    if not process_dates:
        return state

    first_process_date = process_dates[0]
    seed_for_first_bar = bool(seed_initial_entries)

    for current_date in process_dates:
        # Period start equity is the previous known mark; day/week controls are account-level.
        mark_before = equity_now(state, settings)
        day_key = str(current_date.date())
        week_key = _week_key(current_date)
        if state.get("daily_period") != day_key:
            state["daily_period"] = day_key
            state["daily_start_equity_usd"] = mark_before
            state["daily_locked"] = False
        if state.get("weekly_period") != week_key:
            state["weekly_period"] = week_key
            state["weekly_start_equity_usd"] = mark_before
            state["weekly_locked"] = week_key in set(state["locked_weeks"])

        rows_for_date: dict[str, pd.Series] = {
            pair: df.loc[current_date] for pair, df in data.items() if current_date in df.index
        }

        # 1) Manage existing positions at this bar. A stop has priority over trend exit.
        for pair in list(state["positions"].keys()):
            row = rows_for_date.get(pair)
            if row is None:
                continue
            pos = state["positions"][pair]
            prev_row, _ = _get_prior_rows(data[pair], current_date)
            prev_signal = int(prev_row["SIGNAL"]) if prev_row is not None and bool(prev_row["VALID"]) else 0
            direction = int(pos["direction"])
            opening_price = float(row["Open"])
            stop_price = float(pos["stop_price"])
            gap_stop = (direction == 1 and opening_price <= stop_price) or (direction == -1 and opening_price >= stop_price)

            if gap_stop:
                _close_position(state, pair, opening_price, current_date, "gap_stop", settings)
            elif prev_signal != direction:
                # The exit condition was known at the prior close, so exit at this open.
                _close_position(state, pair, opening_price, current_date, "trend_exit", settings)
            else:
                stop_reference, stop_reason = _stop_reference(pos, row)
                if stop_reference is not None:
                    _close_position(state, pair, float(stop_reference), current_date, str(stop_reason), settings)
                else:
                    _update_trailing_stop(pos, row, settings)

        # Mark existing positions at today's open before sizing any new positions.
        # OHLC daily bars still cannot reconstruct the exact intraday equity path.
        for pair, row in rows_for_date.items():
            state["last_close_by_pair"][pair] = float(row["Open"])
        # Enforce the loss limits before considering new entries, based on realized exits and opening gaps.
        _check_risk_limits(state, equity_now(state, settings), current_date, settings)

        # 2) Entries are based only on the signal from the preceding completed candle.
        # The first OOS bar may seed a position if the regime is already active.
        for pair in sorted(rows_for_date):
            if pair in state["positions"]:
                continue
            if state["system_paused"] or state["daily_locked"] or state["weekly_locked"]:
                continue
            df = data[pair]
            prev_row, prevprev_row = _get_prior_rows(df, current_date)
            if prev_row is None or not bool(prev_row["VALID"]):
                continue
            prior_signal = int(prev_row["SIGNAL"])
            if prior_signal == 0 or not np.isfinite(float(prev_row["ATR"])):
                continue
            is_initial_bar = seed_for_first_bar and current_date == first_process_date
            signal_transition = prevprev_row is None or (not bool(prevprev_row["VALID"])) or int(prevprev_row["SIGNAL"]) != prior_signal
            if not (is_initial_bar or signal_transition):
                continue
            signal_date = pd.Timestamp(prev_row.name)
            if not _can_open_from_signal(state, signal_date, current_date):
                continue
            opened = _open_position(
                state, pair, prior_signal, float(rows_for_date[pair]["Open"]),
                float(prev_row["ATR"]), current_date, settings,
            )
            if not opened:
                continue
            # Intraday stop check on the entry candle; no trailing-stop update is used to test the same bar.
            pos = state["positions"].get(pair)
            if pos:
                stop_reference, stop_reason = _stop_reference(pos, rows_for_date[pair])
                if stop_reference is not None:
                    _close_position(state, pair, float(stop_reference), current_date, f"entry_day_{stop_reason}", settings)
                else:
                    _update_trailing_stop(pos, rows_for_date[pair], settings)
                _check_risk_limits(state, equity_now(state, settings), current_date, settings)

        # 3) Save latest closes after evaluating stop levels for the current candle.
        for pair, row in rows_for_date.items():
            state["last_close_by_pair"][pair] = float(row["Close"])

        marked_equity = equity_now(state, settings)
        drawdown, daily_return, weekly_return = _check_risk_limits(state, marked_equity, current_date, settings)

        state["equity_curve"].append({
            "date": day_key,
            "cash_usd": float(state["cash_usd"]),
            "unrealized_pnl_usd": float(marked_equity - float(state["cash_usd"])),
            "equity_usd": float(marked_equity),
            "drawdown_pct": float(drawdown),
            "daily_return_pct": float(daily_return),
            "weekly_return_pct": float(weekly_return),
            "open_positions": int(len(state["positions"])),
            "daily_locked": bool(state["daily_locked"]),
            "weekly_locked": bool(state["weekly_locked"]),
            "system_paused": bool(state["system_paused"]),
        })
        state["last_processed_date"] = day_key

    return state


def initialise_paper_state(data: dict[str, pd.DataFrame], settings: Settings) -> dict[str, Any]:
    """Start a fresh paper account at the latest available bar; do not replay old history as paper trades."""
    state = new_state(settings)
    latest_dates = {pair: pd.Timestamp(df.index.max()) for pair, df in data.items() if not df.empty}
    if not latest_dates:
        raise ValueError("Cannot initialize paper state without data")
    latest = max(latest_dates.values())
    state["last_processed_date"] = str(latest.date())
    state["last_close_by_pair"] = {
        pair: float(df.loc[df.index <= latest, "Close"].iloc[-1])
        for pair, df in data.items() if not df.loc[df.index <= latest].empty
    }
    state["daily_period"] = str(latest.date())
    state["weekly_period"] = _week_key(latest)
    state["daily_start_equity_usd"] = equity_now(state, settings)
    state["weekly_start_equity_usd"] = equity_now(state, settings)
    state["equity_curve"].append({
        "date": str(latest.date()),
        "cash_usd": float(state["cash_usd"]),
        "unrealized_pnl_usd": 0.0,
        "equity_usd": float(state["cash_usd"]),
        "drawdown_pct": 0.0,
        "daily_return_pct": 0.0,
        "weekly_return_pct": 0.0,
        "open_positions": 0,
        "daily_locked": False,
        "weekly_locked": False,
        "system_paused": False,
    })
    return state


def current_signals(data: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for pair, df in sorted(data.items()):
        valid_rows = df[df["VALID"]]
        if valid_rows.empty:
            results.append({"pair": pair, "date": str(df.index[-1].date()) if len(df) else "", "signal": "WARMUP", "close": None})
            continue
        row = valid_rows.iloc[-1]
        signal = int(row["SIGNAL"])
        results.append({
            "pair": pair,
            "date": str(valid_rows.index[-1].date()),
            "signal": {1: "LONG", -1: "SHORT", 0: "FLAT"}[signal],
            "close": float(row["Close"]),
            "sma": float(row["SMA"]),
            "momentum": float(row["MOMENTUM"]),
            "atr": float(row["ATR"]),
        })
    return results
