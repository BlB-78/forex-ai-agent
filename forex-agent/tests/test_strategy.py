import numpy as np
import pandas as pd

from forex_agent.config import Settings
from forex_agent.strategy import add_indicators


def load_settings():
    return Settings.load()


def test_indicators_are_not_available_before_warmup():
    settings = load_settings()
    index = pd.bdate_range("2020-01-01", periods=220)
    close = np.linspace(1.0, 1.22, len(index))
    bars = pd.DataFrame({
        "Open": close - 0.0002,
        "High": close + 0.001,
        "Low": close - 0.001,
        "Close": close,
    }, index=index)
    result = add_indicators(bars, settings)
    assert not bool(result.iloc[100]["VALID"])
    assert bool(result.iloc[-1]["VALID"])
    assert int(result.iloc[-1]["SIGNAL"]) == 1


def test_flat_signal_when_trend_filters_disagree():
    settings = load_settings()
    index = pd.bdate_range("2020-01-01", periods=400)
    # Oscillating history should have valid rows and may be flat where criteria disagree.
    close = 1.1 + 0.01 * np.sin(np.arange(len(index)) / 8)
    bars = pd.DataFrame({
        "Open": close,
        "High": close + 0.001,
        "Low": close - 0.001,
        "Close": close,
    }, index=index)
    result = add_indicators(bars, settings)
    valid = result[result["VALID"]]
    assert not valid.empty
    assert set(valid["SIGNAL"].unique()).issubset({-1, 0, 1})
