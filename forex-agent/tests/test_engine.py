from copy import deepcopy

import numpy as np
import pandas as pd

from forex_agent.config import Settings
from forex_agent.engine import advance_state, new_state
from forex_agent.strategy import add_indicators


def synthetic_data(settings, periods=500):
    index = pd.bdate_range("2010-01-01", periods=periods)
    # Deterministic uptrend with periodic pullbacks so the engine receives valid signals.
    t = np.arange(periods)
    close = 1.05 + 0.00045 * t + 0.004 * np.sin(t / 13.0)
    open_price = close - 0.00015 * np.cos(t / 4.0)
    high = np.maximum(open_price, close) + 0.0015
    low = np.minimum(open_price, close) - 0.0015
    bars = pd.DataFrame({"Open": open_price, "High": high, "Low": low, "Close": close}, index=index)
    return {"EURUSD": add_indicators(bars, settings)}


def test_state_advances_only_once_per_date():
    settings = Settings.load()
    data = synthetic_data(settings)
    first_date = str(data["EURUSD"].index[270].date())
    state = new_state(settings)
    state["last_processed_date"] = str((pd.Timestamp(first_date) - pd.Timedelta(days=1)).date())
    state["equity_curve"].append({
        "date": state["last_processed_date"],
        "cash_usd": settings.initial_equity_usd,
        "unrealized_pnl_usd": 0.0,
        "equity_usd": settings.initial_equity_usd,
        "drawdown_pct": 0.0,
        "daily_return_pct": 0.0,
        "weekly_return_pct": 0.0,
        "open_positions": 0,
        "daily_locked": False,
        "weekly_locked": False,
        "system_paused": False,
    })
    state = advance_state(data, state, settings, start_date=first_date, seed_initial_entries=True)
    n_curve = len(state["equity_curve"])
    n_trades = len(state["trades"])
    last_date = state["last_processed_date"]
    state_again = advance_state(data, deepcopy(state), settings)
    assert len(state_again["equity_curve"]) == n_curve
    assert len(state_again["trades"]) == n_trades
    assert state_again["last_processed_date"] == last_date


def test_engine_keeps_risk_limits_in_state():
    settings = Settings.load()
    data = synthetic_data(settings)
    state = new_state(settings)
    start = str(data["EURUSD"].index[270].date())
    state["last_processed_date"] = str((pd.Timestamp(start) - pd.Timedelta(days=1)).date())
    state["equity_curve"].append({
        "date": state["last_processed_date"], "cash_usd": settings.initial_equity_usd,
        "unrealized_pnl_usd": 0.0, "equity_usd": settings.initial_equity_usd,
        "drawdown_pct": 0.0, "daily_return_pct": 0.0, "weekly_return_pct": 0.0,
        "open_positions": 0, "daily_locked": False, "weekly_locked": False, "system_paused": False,
    })
    advanced = advance_state(data, state, settings, start_date=start, seed_initial_entries=True)
    assert "daily_locked" in advanced
    assert "weekly_locked" in advanced
    assert "system_paused" in advanced
    assert advanced["peak_equity_usd"] > 0


def test_position_risk_budget_is_capped_and_state_is_json_serializable():
    import json

    settings = Settings.load()
    data = synthetic_data(settings)
    start = str(data["EURUSD"].index[270].date())
    state = new_state(settings)
    state["last_processed_date"] = str((pd.Timestamp(start) - pd.Timedelta(days=1)).date())
    state["equity_curve"].append({
        "date": state["last_processed_date"], "cash_usd": settings.initial_equity_usd,
        "unrealized_pnl_usd": 0.0, "equity_usd": settings.initial_equity_usd,
        "drawdown_pct": 0.0, "daily_return_pct": 0.0, "weekly_return_pct": 0.0,
        "open_positions": 0, "daily_locked": False, "weekly_locked": False, "system_paused": False,
    })
    state = advance_state(data, state, settings, start_date=start, seed_initial_entries=True)
    assert all(p["initial_risk_usd"] <= settings.initial_equity_usd * settings.risk_per_trade_pct + 1e-6
               for p in state["positions"].values())
    json.dumps(state, allow_nan=False)


def test_paper_state_starts_now_without_replaying_history():
    from forex_agent.engine import initialise_paper_state

    settings = Settings.load()
    data = synthetic_data(settings, periods=500)
    state = initialise_paper_state(data, settings)
    latest = max(df.index.max() for df in data.values()).date().isoformat()
    assert state["last_processed_date"] == latest
    assert state["positions"] == {}
    assert state["trades"] == []
    assert len(state["equity_curve"]) == 1
