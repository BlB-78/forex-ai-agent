# Forex AI Agent — public GitHub Actions starter

This repository is the first automation scaffold for a daily Forex trend-following **research and paper-trading** agent. It does not place real-money orders and has no broker execution adapter.

## What is included

- Python strategy and risk-management engine shared by historical backtests and the paper simulator.
- Daily D1 signals for EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD and USD/CAD.
- Entry rule: on a new non-flat signal (or at the beginning of a historical test if the regime is already active), simulate entry at the next daily bar's open.
- Long when close > 200-day SMA and 100-day momentum > 0; short when close < SMA and momentum < 0; otherwise flat.
- Initial stop at 2 × 20-day ATR; stop is trailed using completed-bar highs/lows and ATR.
- Default planned risk 0.5% of equity per trade, combined planned open-risk cap 1.5%, daily loss circuit breaker 1.5%, weekly loss circuit breaker 3%, and an 8% peak-drawdown pause.
- Estimated spread, slippage and configurable per-side commission.
- Historical in-sample and out-of-sample backtest reports.
- Scheduled daily paper-simulator workflow, with state and journals persisted to a public `agent-state` branch.
- A manual, bounded candidate-parameter experiment workflow. This is controlled research, not online/reinforcement learning; it never promotes a candidate automatically.
- Automated unit tests on pushes and pull requests.

## Very important limitations

1. **Not live trading.** Live order execution is intentionally not implemented and `live_trading_enabled` must remain `false`. The paper simulator cannot access a broker account or place orders.
2. **Data limitations.** The default data source is Yahoo Finance through `yfinance`. These OHLC candles are not broker-specific bid/ask quotes and availability/completeness can vary. The workflow fails if any configured pair has no usable data, but successful downloads still require inspection.
3. **Costs are placeholders.** Default spreads and slippage are illustrative assumptions, not verified quotes for a particular account. Configure them for the intended broker before evaluating results.
4. **Swap/financing omitted.** Historical overnight financing is not included yet. A Forex swing strategy's performance cannot be judged responsibly until financing costs have been modeled.
5. **Daily OHLC limitation.** A daily candle cannot reconstruct exact intraday equity paths. Daily/weekly circuit breakers are evaluated using daily marks, so they are not a guarantee of a maximum loss. The stop simulation uses conservative assumptions for gap-through stops, but it is still an approximation.
6. **Not a validated profitability claim.** The strategy, risk settings and costs have not been demonstrated profitable. Use out-of-sample results, cost sensitivity, more detailed data and paper trading before even considering real money.
7. **Public state.** The `agent-state` branch will contain the simulated paper account state, trade journal, equity curve and summary. These files are public. Do not store account details, broker credentials, personal data or real portfolio information in this repository or its workflows.
8. **Persistence is not a database.** The state branch is a lightweight prototype. Concurrent runs are serialized, but a production deployment should use a proper database with backups and integrity checks.
9. **Scheduled workflows can be delayed.** GitHub Actions is a scheduler, not a continuously running trading server. This is acceptable for research and daily simulation, not as the only safety mechanism for live positions.

## Strategy outline

Indicators use data from completed daily bars:

- SMA: 200 days.
- Momentum: current close divided by close 100 bars earlier, minus one.
- ATR: simple 20-day rolling average of True Range (not Wilder's ATR).
- Signal: long if close is above SMA and momentum is positive; short if below SMA and momentum is negative; otherwise flat.
- Entry: next daily bar's open after a transition into long/short signal.
- Exit: stop/gap stop or next-open exit when the previous completed bar no longer supports the position direction.
- Trailing stop: 2 × ATR from the running extreme, updated only after that day's stop checks.

The rules are a research hypothesis, not a recommendation or claim of an edge. The candidate experiment job compares a fixed list of alternatives on the development window only. It does not train a predictive ML model, learn autonomously from live trades, or edit the approved settings. A person must review evidence and deliberately update the config/version after separate validation.

## Quick start on Windows

Install Python 3.12 or newer from [python.org](https://www.python.org/downloads/windows/). During installation, select **Add Python to PATH**. Extract or clone this repository, open Command Prompt in the project directory, then run:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -e ".[dev]"
py -m pytest -q
```

Run a historical backtest (downloads OHLC history when run):

```bat
py -m forex_agent.cli backtest --start 2008-01-01 --split-date 2018-01-01 --output-dir reports/backtest
```

Run bounded candidate experiments on the development period only (this does not promote any strategy):

```bat
py -m forex_agent.cli learn --start 2008-01-01 --development-end 2018-01-01 --output-dir reports/learning
```

Run the signal report without creating a paper account:

```bat
py -m forex_agent.cli signals
```

Run the paper simulator locally:

```bat
py -m forex_agent.cli paper --state-path runtime/paper_state.json --report-dir reports/paper
```

The first paper run starts with a clean paper account at the latest available candle; it does **not** replay old historical bars as paper trades. A later run processes newly available daily bars and persists the simulated account in the JSON state.

## Upload to GitHub

1. Create a **public** repository named `forex-ai-agent` (or use another name).
2. Upload the contents of this folder, including hidden folders such as `.github/workflows/`.
3. In the repository, open **Actions** and enable workflows if GitHub asks.
4. Run **Tests and safety checks** from the Actions tab or push a commit to the default branch.
5. Run **Historical backtest** manually. Download the artifact after the job finishes.
6. Run **Strategy learning experiments** manually. It compares a small fixed candidate grid using data before the development cutoff; it does not change the deployed settings. Download its artifact.
7. Run **Daily paper-trading cycle** manually once. The workflow should create an `agent-state` branch containing public simulated state; later runs restore and update it.
8. After that, the paper workflow is scheduled for 23:55 UTC, Monday through Friday. GitHub scheduled workflow starts may be delayed. Change the schedule if data availability indicates a different time is appropriate.

### Required GitHub workflow permission

The daily paper workflow writes only to the `agent-state` branch. In repository Settings → Actions → General, ensure workflows are allowed to have **read and write permissions** for `GITHUB_TOKEN`. The workflow itself scopes the token to `contents: write`. If the policy doesn't permit this, the paper workflow can run but cannot persist state.

## Outputs

The backtest workflow uploads:

- `in_sample_summary.json`, `in_sample_trades.csv`, `in_sample_equity.csv`
- `out_of_sample_summary.json`, `out_of_sample_trades.csv`, `out_of_sample_equity.csv`
- `comparison_summary.json`

The daily workflow uploads:

- `latest_signals.csv`
- `paper_summary.json`
- `paper_trades.csv`
- `paper_equity.csv`

Public repository workflow artifacts are downloadable by people with repository read access and expire according to GitHub's artifact retention policy. They are reports, not a secret store.

## How to evaluate results

Do not judge the system on total return alone. Review trade count, net returns after modeled costs, maximum drawdown, profit factor, trade expectancy, per-pair results and the untouched out-of-sample period. Repeat testing under higher spread/slippage assumptions. Add historical financing before treating the backtest as a serious economic evaluation. Never tune using the holdout period and continue calling that same period out-of-sample.

## Public repository security

- Never commit secrets into source, config files, workflow YAML or the public state branch.
- Never add real broker tokens as plain text.
- The current workflows do not need any broker credentials.
- The `GITHUB_TOKEN` is given `contents: read` in test/backtest workflows, and `contents: write` only in the paper workflow for the state branch.
- Do not enable live trading by changing a config flag; an execution adapter has not been implemented.

## Official docs

- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
- [GitHub Actions artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
- [GitHub Actions security](https://docs.github.com/en/actions/reference/security/secure-use)
- [yfinance documentation](https://ranaroussi.github.io/yfinance/)
