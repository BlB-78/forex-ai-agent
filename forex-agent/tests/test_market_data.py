import pandas as pd

from forex_agent.market_data import _normalise_download


def test_normalise_download_accepts_yahoo_multiindex_ohlc():
    index = pd.date_range("2024-01-01", periods=2, tz="UTC")
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close"], ["EURUSD=X"]]
    )
    values = [
        [1.10, 1.12, 1.09, 1.11],
        [1.11, 1.13, 1.10, 1.12],
    ]
    raw = pd.DataFrame(values, index=index, columns=columns)
    result = _normalise_download(raw)
    assert list(result.columns) == ["Open", "High", "Low", "Close"]
    assert result.index.tz is None
    assert len(result) == 2
