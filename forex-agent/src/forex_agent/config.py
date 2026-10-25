from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "settings.json"


@dataclass(frozen=True)
class Settings:
    project_name: str
    strategy_version: str
    initial_equity_usd: float
    start_date: str
    ticker_symbols: dict[str, str]
    sma_days: int
    momentum_days: int
    atr_days: int
    atr_stop_multiple: float
    risk_per_trade_pct: float
    max_combined_open_risk_pct: float
    daily_loss_limit_pct: float
    weekly_loss_limit_pct: float
    peak_drawdown_pause_pct: float
    spread_pips: dict[str, float]
    slippage_pips_per_side: float
    commission_usd_per_100k_per_side: float
    include_historical_swap: bool
    live_trading_enabled: bool
    max_data_age_calendar_days: int

    @classmethod
    def load(cls, path: Path | str = DEFAULT_CONFIG_PATH) -> "Settings":
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as handle:
            raw: dict[str, Any] = json.load(handle)
        return cls(
            project_name=raw["project_name"],
            strategy_version=raw["strategy_version"],
            initial_equity_usd=float(raw["account"]["initial_equity_usd"]),
            start_date=str(raw["market_data"]["start_date"]),
            ticker_symbols={k: str(v) for k, v in raw["market_data"]["ticker_symbols"].items()},
            sma_days=int(raw["strategy"]["sma_days"]),
            momentum_days=int(raw["strategy"]["momentum_days"]),
            atr_days=int(raw["strategy"]["atr_days"]),
            atr_stop_multiple=float(raw["strategy"]["atr_stop_multiple"]),
            risk_per_trade_pct=float(raw["risk"]["risk_per_trade_pct"]),
            max_combined_open_risk_pct=float(raw["risk"]["max_combined_open_risk_pct"]),
            daily_loss_limit_pct=float(raw["risk"]["daily_loss_limit_pct"]),
            weekly_loss_limit_pct=float(raw["risk"]["weekly_loss_limit_pct"]),
            peak_drawdown_pause_pct=float(raw["risk"]["peak_drawdown_pause_pct"]),
            spread_pips={k: float(v) for k, v in raw["costs"]["spread_pips"].items()},
            slippage_pips_per_side=float(raw["costs"]["slippage_pips_per_side"]),
            commission_usd_per_100k_per_side=float(raw["costs"]["commission_usd_per_100k_per_side"]),
            include_historical_swap=bool(raw["costs"]["include_historical_swap"]),
            live_trading_enabled=bool(raw["safety"]["live_trading_enabled"]),
            max_data_age_calendar_days=int(raw["safety"]["max_data_age_calendar_days"]),
        )

    def validate(self) -> None:
        if self.initial_equity_usd <= 0:
            raise ValueError("initial_equity_usd must be positive")
        if not 0 < self.risk_per_trade_pct <= 0.01:
            raise ValueError("risk_per_trade_pct must be in (0, 1%] for this starter")
        if not 0 < self.max_combined_open_risk_pct <= 0.03:
            raise ValueError("max_combined_open_risk_pct must be in (0, 3%] for this starter")
        for name, value in {
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "weekly_loss_limit_pct": self.weekly_loss_limit_pct,
            "peak_drawdown_pause_pct": self.peak_drawdown_pause_pct,
        }.items():
            if not 0 < value < 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.sma_days < 2 or self.momentum_days < 1 or self.atr_days < 2:
            raise ValueError("Indicator periods are invalid")
        missing_spreads = set(self.ticker_symbols) - set(self.spread_pips)
        if missing_spreads:
            raise ValueError(f"Missing spread assumptions for: {sorted(missing_spreads)}")
        if any(value < 0 for value in self.spread_pips.values()):
            raise ValueError("Spread assumptions cannot be negative")
        if self.slippage_pips_per_side < 0 or self.commission_usd_per_100k_per_side < 0:
            raise ValueError("Slippage and commission assumptions cannot be negative")
        # Live trading is intentionally not implemented in this starter.
        if self.live_trading_enabled:
            raise ValueError("Live trading is not implemented; keep live_trading_enabled=false")
