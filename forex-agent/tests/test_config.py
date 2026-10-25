from forex_agent.config import Settings


def test_live_trading_is_disabled_by_default():
    settings = Settings.load()
    settings.validate()
    assert settings.live_trading_enabled is False
    assert settings.risk_per_trade_pct == 0.005
    assert settings.max_combined_open_risk_pct == 0.015
