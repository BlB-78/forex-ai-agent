from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG_PATH, Settings
from .engine import advance_state, current_signals, equity_now, initialise_paper_state, new_state
from .market_data import fetch_daily_data
from .strategy import add_indicators


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    temp.replace(path)


def _csv_from_records(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def _curve_dataframe(state: dict[str, Any]) -> pd.DataFrame:
    curve = pd.DataFrame(state.get("equity_curve", []))
    if not curve.empty:
        curve["date"] = pd.to_datetime(curve["date"])
        curve = curve.sort_values("date").drop_duplicates("date", keep="last")
    return curve


def _summary(state: dict[str, Any], initial_equity: float) -> dict[str, Any]:
    curve = _curve_dataframe(state)
    trades = pd.DataFrame(state.get("trades", []))
    if curve.empty:
        return {"error": "No equity records available"}

    equities = curve["equity_usd"].astype(float)
    # Include initial equity as a prior point so drawdown calculations include day-one losses.
    equity_for_dd = pd.concat([pd.Series([initial_equity]), equities.reset_index(drop=True)], ignore_index=True)
    peaks = equity_for_dd.cummax()
    max_drawdown = float((equity_for_dd / peaks - 1.0).min())
    returns = equities.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    day_span = max((curve["date"].iloc[-1] - curve["date"].iloc[0]).days, 1)
    ending_equity = float(equities.iloc[-1])
    cagr = (ending_equity / initial_equity) ** (365.25 / day_span) - 1.0 if ending_equity > 0 else -1.0
    volatility = float(returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 else float("nan")
    sharpe = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252)) if len(returns) > 1 and returns.std(ddof=1) > 0 else float("nan")

    if not trades.empty and "net_pnl_usd" in trades:
        pnl = pd.to_numeric(trades["net_pnl_usd"], errors="coerce").dropna()
        wins, losses = pnl[pnl > 0], pnl[pnl < 0]
        gross_profit = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))
        win_rate = float((pnl > 0).mean()) if len(pnl) else float("nan")
        expectancy = float(pnl.mean()) if len(pnl) else float("nan")
        trade_count = int(len(pnl))
        net_closed_pnl = float(pnl.sum())
    else:
        profit_factor, win_rate, expectancy, trade_count, net_closed_pnl = float("nan"), float("nan"), float("nan"), 0, 0.0

    return {
        "period_start": str(curve["date"].iloc[0].date()),
        "period_end": str(curve["date"].iloc[-1].date()),
        "initial_equity_usd": float(initial_equity),
        "final_equity_usd": ending_equity,
        "total_return_pct": (ending_equity / initial_equity - 1.0) * 100.0,
        "cagr_pct_approx": cagr * 100.0,
        "max_drawdown_pct": max_drawdown * 100.0,
        "annualized_volatility_pct_approx": volatility * 100.0 if np.isfinite(volatility) else None,
        "daily_sharpe_0rf_approx": sharpe if np.isfinite(sharpe) else None,
        "closed_trades": trade_count,
        "win_rate_pct": win_rate * 100.0 if np.isfinite(win_rate) else None,
        "profit_factor": profit_factor if np.isfinite(profit_factor) else None,
        "average_net_trade_usd": expectancy if np.isfinite(expectancy) else None,
        "net_closed_trade_pnl_usd": net_closed_pnl,
        "open_positions_at_end": len(state.get("positions", {})),
        "system_paused": bool(state.get("system_paused", False)),
        "pause_reason": state.get("pause_reason"),
        "note": "Research output only. Swap/financing is not included; data and cost assumptions are not broker-verified.",
    }


def _force_close_all(state: dict[str, Any], data: dict[str, pd.DataFrame], settings: Settings) -> None:
    # Only backtests are force-closed at period end. Paper positions persist between scheduled runs.
    from .engine import _close_position

    if not state.get("equity_curve"):
        return
    end_date = pd.Timestamp(state["last_processed_date"])
    for pair in list(state["positions"].keys()):
        df = data[pair]
        available = df.loc[df.index <= end_date]
        if available.empty:
            continue
        row = available.iloc[-1]
        _close_position(state, pair, float(row["Close"]), end_date, "end_of_test", settings)
    if state["equity_curve"]:
        last = state["equity_curve"][-1]
        last["cash_usd"] = float(state["cash_usd"])
        last["unrealized_pnl_usd"] = 0.0
        last["equity_usd"] = float(state["cash_usd"])
        last["open_positions"] = 0
        state["peak_equity_usd"] = max(float(state["peak_equity_usd"]), float(state["cash_usd"]))
        last["drawdown_pct"] = float(state["cash_usd"]) / float(state["peak_equity_usd"]) - 1.0


def _create_backtest_state(settings: Settings, start_date: str) -> dict[str, Any]:
    state = new_state(settings)
    base_date = pd.Timestamp(start_date) - pd.Timedelta(days=1)
    state["last_processed_date"] = str(base_date.date())
    state["equity_curve"].append({
        "date": str(base_date.date()),
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
    return state


def run_backtest(settings: Settings, args: argparse.Namespace) -> None:
    data = fetch_daily_data(settings, start=settings.start_date, end=args.end, check_freshness=False)
    split = pd.Timestamp(args.split_date)
    start = pd.Timestamp(args.start)
    if split <= start:
        raise ValueError("--split-date must be after --start")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    periods = [
        ("in_sample", str(start.date()), str(split.date())),
        ("out_of_sample", str(split.date()), args.end),
    ]
    summaries: dict[str, Any] = {}
    for name, period_start, period_end in periods:
        print(f"\nRunning {name}: {period_start} to before {period_end}")
        state = _create_backtest_state(settings, period_start)
        state = advance_state(
            data, state, settings,
            start_date=period_start, end_date=period_end,
            seed_initial_entries=True,
        )
        _force_close_all(state, data, settings)
        summary = _summary(state, settings.initial_equity_usd)
        summaries[name] = summary
        _atomic_json(output / f"{name}_summary.json", summary)
        _csv_from_records(state.get("trades", []), output / f"{name}_trades.csv")
        _csv_from_records(state.get("equity_curve", []), output / f"{name}_equity.csv")
        print(json.dumps(summary, indent=2))

    _atomic_json(output / "comparison_summary.json", summaries)
    print("\nNo profitability conclusion is made automatically. Review data quality, financing and costs.")


def run_learning(settings: Settings, args: argparse.Namespace) -> None:
    """Compare a small, pre-declared candidate grid using development-period data only.

    This reports experiments; it never rewrites settings.json or changes the paper strategy.
    """
    development_end = pd.Timestamp(args.development_end)
    development_start = pd.Timestamp(args.start)
    if development_end <= development_start:
        raise ValueError("--development-end must be after --start")
    # The data request ends before the development cutoff; later bars are not downloaded for this job.
    base_data = fetch_daily_data(
        settings, start=settings.start_date, end=str(development_end.date()), check_freshness=False
    )
    candidates = [
        (150, 50, 1.5), (150, 100, 2.0), (150, 150, 2.5),
        (200, 50, 2.0), (200, 100, 1.5), (200, 100, 2.0),
        (200, 100, 2.5), (200, 150, 2.0), (250, 100, 2.0),
    ]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for sma_days, momentum_days, atr_multiple in candidates:
        candidate_settings = replace(
            settings,
            strategy_version=f"research-sma{sma_days}-mom{momentum_days}-atr{atr_multiple}",
            sma_days=sma_days,
            momentum_days=momentum_days,
            atr_stop_multiple=atr_multiple,
        )
        candidate_settings.validate()
        candidate_data = {pair: add_indicators(df, candidate_settings) for pair, df in base_data.items()}
        state = _create_backtest_state(candidate_settings, str(development_start.date()))
        state = advance_state(
            candidate_data, state, candidate_settings,
            start_date=str(development_start.date()), end_date=str(development_end.date()),
            seed_initial_entries=True,
        )
        _force_close_all(state, candidate_data, candidate_settings)
        metrics = _summary(state, candidate_settings.initial_equity_usd)
        rows.append({
            "candidate": candidate_settings.strategy_version,
            "sma_days": sma_days,
            "momentum_days": momentum_days,
            "atr_stop_multiple": atr_multiple,
            **metrics,
        })
        print(f"Completed research candidate: {candidate_settings.strategy_version}")

    results = pd.DataFrame(rows)
    # Sort only for convenient inspection; ranking is not automatic promotion.
    if "total_return_pct" in results:
        results = results.sort_values("total_return_pct", ascending=False, na_position="last")
    results.to_csv(output / "candidate_experiments.csv", index=False)
    _atomic_json(output / "candidate_experiments.json", rows)
    (output / "LEARNING_NOTES.txt").write_text(
        "These candidates were evaluated only on the configured development period.\n"
        "The highest historical return is not automatically selected or promoted.\n"
        "Review costs, drawdown, trade count, robustness and selection bias.\n"
        "Do not use a period for tuning after treating it as a final out-of-sample test.\n"
        "No live trading is implemented.\n",
        encoding="utf-8",
    )
    print(results[["candidate", "total_return_pct", "max_drawdown_pct", "closed_trades", "profit_factor"]].to_string(index=False))
    print("Candidate experiment results saved. No settings were changed and no candidate was promoted.")


def run_paper(settings: Settings, args: argparse.Namespace) -> None:
    state_path = Path(args.state_path)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    data = fetch_daily_data(settings, start=settings.start_date, end=args.end)

    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        if state.get("strategy_version") != settings.strategy_version:
            raise RuntimeError(
                "The persisted paper account uses a different strategy_version. "
                "Do not silently mix strategy versions; review and intentionally reset/migrate state."
            )
        old_last_date = state.get("last_processed_date")
        state = advance_state(data, state, settings)
        new_last_date = state.get("last_processed_date")
        print(f"Paper state advanced from {old_last_date} to {new_last_date} (latest downloaded date: {max(df.index.max() for df in data.values()).date()}).")
    else:
        state = initialise_paper_state(data, settings)
        print("Initialized an empty paper account at the latest downloaded candle. No historical trades were replayed.")
        print("The first eligible simulated entry requires a subsequent daily candle.")

    _atomic_json(state_path, state)
    _csv_from_records(current_signals(data), report_dir / "latest_signals.csv")
    _csv_from_records(state.get("trades", []), report_dir / "paper_trades.csv")
    _csv_from_records(state.get("equity_curve", []), report_dir / "paper_equity.csv")
    summary = _summary(state, settings.initial_equity_usd)
    _atomic_json(report_dir / "paper_summary.json", summary)
    print("Latest signals:")
    print(pd.DataFrame(current_signals(data)).to_string(index=False))
    print("Paper trading only: no broker API or real-money order execution is implemented.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Forex research and paper-trading starter (live trading disabled)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to settings JSON")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser("backtest", help="Run chronological in-sample and out-of-sample backtests")
    backtest.add_argument("--start", default="2008-01-01", help="In-sample start date (YYYY-MM-DD)")
    backtest.add_argument("--split-date", default="2018-01-01", help="Out-of-sample start date (YYYY-MM-DD)")
    backtest.add_argument("--end", default=(datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(), help="Exclusive end date YYYY-MM-DD")
    backtest.add_argument("--output-dir", default="reports/backtest", help="Output directory")

    paper = subparsers.add_parser("paper", help="Advance the simulated paper account using new daily bars")
    paper.add_argument("--state-path", default="runtime/paper_state.json", help="Persistent paper account JSON")
    paper.add_argument("--report-dir", default="reports/paper", help="Output directory")
    paper.add_argument("--end", default=(datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(), help="Exclusive end date YYYY-MM-DD")

    learn = subparsers.add_parser("learn", help="Compare predeclared strategy candidates using development data only")
    learn.add_argument("--start", default="2008-01-01", help="Development start date YYYY-MM-DD")
    learn.add_argument("--development-end", default="2018-01-01", help="Exclusive development cutoff YYYY-MM-DD")
    learn.add_argument("--output-dir", default="reports/learning", help="Output directory")

    signals = subparsers.add_parser("signals", help="Print latest trend signals without creating positions")
    signals.add_argument("--end", default=(datetime.now(timezone.utc).date() + timedelta(days=1)).isoformat(), help="Exclusive end date YYYY-MM-DD")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings.load(args.config)
    settings.validate()
    if settings.live_trading_enabled:
        raise RuntimeError("Live trading must remain disabled in this starter")

    if args.command == "backtest":
        run_backtest(settings, args)
    elif args.command == "paper":
        run_paper(settings, args)
    elif args.command == "learn":
        run_learning(settings, args)
    elif args.command == "signals":
        data = fetch_daily_data(settings, start=settings.start_date, end=args.end)
        print(pd.DataFrame(current_signals(data)).to_string(index=False))
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()
