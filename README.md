# alpha-engine-backtester

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-189_passing-brightgreen.svg)]()

> Signal quality analysis, evaluation framework (component grades + P/R/F1), portfolio simulation, and autonomous parameter optimization. The system's learning mechanism — validates whether signals predict outperformance and feeds optimized parameters back to upstream modules.

**Part of the [Nous Ergon](https://nousergon.ai) autonomous trading system.**
See the [system overview](https://github.com/cipher813/alpha-engine#readme) for how all modules connect, or the [full documentation index](https://github.com/cipher813/alpha-engine-docs#readme).

## Table of Contents

- [Role in the System](#role-in-the-system)
- [Quick Start](#quick-start)
- [Modes](#modes)
- [How It Works](#how-it-works)
- [Optimization Architecture](#optimization-architecture)
- [Configuration Reference](#configuration-reference)
- [Key Files](#key-files)
- [Deployment](#deployment)
- [S3 Contract](#s3-contract)
- [Testing](#testing)
- [Related Modules](#related-modules)

---

## Role in the System

The Backtester closes the feedback loop. It reads historical signals and prices, measures accuracy, identifies which sub-scores are most predictive, and writes four auto-optimization config files to S3 that upstream modules pick up on their next cold-start:

| S3 Key | Read By | Controls |
|--------|---------|----------|
| `config/scoring_weights.json` | Research | Sub-score composite weights (news/research balance) |
| `config/executor_params.json` | Executor | Risk parameters and position sizing |
| `config/predictor_params.json` | Predictor | Veto confidence threshold |
| `config/research_params.json` | Research | Signal boost parameters (short interest, institutional) |

Each config file has a `_previous.json` backup and a dated archive in `config/{type}_history/{date}.json` for rollback and stability tracking.

---

## Quick Start

### Prerequisites

- Python 3.11+
- AWS credentials with S3 read/write and SES send permission
- `research.db` in S3 (written by Research after each run)
- `signals/{date}/signals.json` in S3 (at least a few trading days)

### Setup

```bash
git clone https://github.com/cipher813/alpha-engine-backtester.git
cd alpha-engine-backtester
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp config.yaml.example config.yaml
# Edit config.yaml — set S3 bucket names, paths, email settings

python evaluate.py --mode diagnostics --freeze   # evaluation only (signal quality, diagnostics)
python backtest.py --mode simulate               # simulation only
```

> **Note:** This repo has two entry points. `backtest.py` runs simulation (param sweep, predictor backtest). `evaluate.py` runs evaluation (signal quality, diagnostics, config optimization). See `evaluate.py --help` for all options.

---

## Modes

### Signal Quality (`evaluate.py --mode diagnostics`)

Reads `score_performance` from `research.db` and computes:
- % of BUY signals that beat SPY at 5d/10d/30d horizons (Wilson CI, BH FDR correction)
- Accuracy by score bucket (60-70, 70-80, 80-90, 90+) with exploratory flags for small samples
- Accuracy by market regime (bull / neutral / bear / caution)
- **Accuracy by sector** (joined from universe_returns)
- **Precision/recall/F1** at every decision boundary (scanner, teams, CIO, predictor, executor)
- Sub-score attribution (quant vs qual correlation with outperformance)
- **Unified scorecard** — A-F grades for every component (scanner, 6 sector teams, CIO, macro, predictor, veto, triggers, risk guard, exits, sizing, portfolio)
- **Predictor confusion matrix** — 3x3 UP/FLAT/DOWN with per-direction precision/recall/F1
- **Scoring weight recommendation** — applied to S3 automatically if guardrails pass
- **Veto threshold analysis** — sweeps confidence thresholds with precision/recall/cost tradeoff, per-sector breakdown
- **Predictor rolling metrics** — backfills outcomes, pushes 30-day hit rate + IC to S3
- **Grade history** — weekly grades appended to S3 for trend tracking (52-week rolling)

### Portfolio Simulation (`--mode simulate`)

Replays historical signal dates through the executor's `run(simulate=True)`, converts orders to a VectorBT portfolio, and produces Sharpe ratio, max drawdown, Calmar ratio, win rate, and alpha vs SPY. Supports configurable slippage (10bps default) and next-day fill simulation.

### Parameter Sweep (`--mode param-sweep`)

Runs portfolio simulation across 60 random trials from a grid of 6 core risk parameters (1,728 total combinations). 60 trials gives 95% confidence of finding a top-5% combination (Bergstra & Bengio 2012). Price matrix is built once and reused across all trials.

### Predictor Backtest (`--mode predictor-backtest`)

The **primary** param sweep source. Generates synthetic signals by running the GBM model on up to 10 years of OHLCV data from the full S3 price cache, then replays through the full executor pipeline. This provides ~2,500 trading days for statistically robust parameter optimization — far more data than the live signal history.

### All Modes (`--mode all`)

Runs signal quality, simulation, param sweep, and predictor backtest sequentially. This is what the weekly spot instance runs.

---

## How It Works

```
                        ┌─────────────────────────────────┐
                        │  S3: research.db, signals.json, │
                        │  predictor/price_cache (10y)    │
                        └────────────────┬────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    │                    │                    │
                    ▼                    ▼                    ▼
         Signal Quality          Predictor Backtest    Portfolio Simulation
         ├─ Accuracy 10d/30d     ├─ Load 10y OHLCV     ├─ Replay live signals
         ├─ Score buckets        ├─ Compute features    └─ VectorBT portfolio
         ├─ Regime analysis      ├─ GBM inference
         ├─ Attribution (BH)     ├─ Synthetic signals
         └─ Veto analysis        └─ Param sweep (60 trials)
                    │                    │                    │
                    ▼                    ▼                    ▼
         Weight Optimizer        Executor Optimizer     Veto Optimizer
         └─ news/research        └─ 6 core params       └─ confidence threshold
            weights                 (holdout validated)
                    │                    │                    │
                    └────────────────────┼────────────────────┘
                                         ▼
                              S3 Config Files + Report
                              ├─ config/scoring_weights.json
                              ├─ config/executor_params.json
                              ├─ config/predictor_params.json
                              ├─ config/research_params.json
                              ├─ backtest/{date}/report.md
                              ├─ backtest/{date}/grading.json
                              ├─ backtest/{date}/confusion_matrix.json
                              ├─ backtest/{date}/trigger_scorecard.json
                              ├─ backtest/{date}/shadow_book.json
                              ├─ backtest/{date}/exit_timing.json
                              ├─ backtest/{date}/e2e_lift.json
                              ├─ backtest/{date}/veto_analysis.json
                              └─ backtest/grade_history.json
```

---

## Optimization Architecture

The backtester runs four optimizers. Three run weekly; the research optimizer is deferred until 200+ samples accumulate (~6 months of live data). All optimizers include conservative guardrails and S3 rollback support.

### Weight Optimizer → `config/scoring_weights.json`

Tunes the balance between Research sub-scores (news vs research) in the composite attractiveness score. Effectively 1 degree of freedom (weights sum to 1.0).

**Approach:** Correlation-based optimization with conservative blending.

1. Split `score_performance` data 70/30 by date (train/test)
2. Correlate each sub-score with `beat_spy_10d` and `beat_spy_30d` on train set
3. Blend horizons 50/50 (configurable)
4. Progressive blend factor: 20% data-driven at low sample sizes → 50% at 500+ samples
5. Validate on test set: OOS degradation must be < 20%
6. Stability check: no direction reversals in prior 3 weeks

**Guardrails:** max 10% single change, min 3% meaningful, medium+ confidence (100+ samples), OOS validation, weights normalize to 1.0.

### Executor Optimizer → `config/executor_params.json`

Recommends the best executor parameter combination from the parameter sweep, ranked by risk-adjusted return.

**Approach:** Random search (Bergstra & Bengio) — 60 trials gives 95% confidence of finding a top-5% combination from 1,728 total.

**Ranking:** `combined_score = Sharpe ratio - 0.5 × max_drawdown`

**Improvement gate:** Best combo must exceed baseline by ≥ 10% Sharpe improvement. Holdout validation (last 30% of dates) must achieve ≥ 50% of training Sharpe.

**Core parameter sweep (6 parameters, 1,728 combinations):**

| Parameter | Grid values | Impact |
|-----------|-------------|--------|
| `min_score` | 65, 70, 75, 80 | Gates every entry — most impactful |
| `max_position_pct` | 0.05, 0.10, 0.15 | Caps loss on any single position |
| `atr_multiplier` | 2.0, 2.5, 3.0, 4.0 | Stop distance — affects every position |
| `time_decay_reduce_days` | 5, 7, 10 | When to start trimming aging positions |
| `time_decay_exit_days` | 10, 15, 20 | When to force-exit stale positions |
| `profit_take_pct` | 0.15, 0.20, 0.25, 0.30 | When to take profits on winners |

These 6 are regime-invariant risk/exit rules that fire on every trade. ATR-based stops are volatility-scaled by design — an ATR multiplier of 2.5 means "2.5× recent volatility range" regardless of market conditions, making 10 years of diverse market data (2017 low-vol, 2020 crash, 2022 bear, 2024 rally) more robust than 2 years of a single regime.

### Veto Analyzer → `config/predictor_params.json`

Calibrates the predictor's veto confidence threshold.

**Thresholds swept:** 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80

**Scoring:** `score = precision - 0.30 × (missed_alpha / max_missed_alpha)`
- Wilson CI for precision estimates on small samples
- Lift gate: must exceed base rate by 5%+
- Cost sensitivity sweep across [0.15, 0.30, 0.50, 0.70] penalty weights

**Gates:** Min 30 predictions, min 10 vetoes per threshold, 5% lift, 0.10 threshold change.

### Research Optimizer → `config/research_params.json` (deferred)

Tunes signal boost parameters (short interest thresholds, institutional boost, consistency scoring). Currently gated behind 200+ samples (~6 months of live data) because boost correlations are too noisy at smaller sample sizes. See [Deferred Opportunities](#deferred-opportunities) below.

### Portfolio Simulation (VectorBT)

All parameter sweep trials and standalone simulation use VectorBT:

- Starting NAV: $1M (configurable)
- Transaction fees: 10bps round-trip
- Slippage: 10bps per side (configurable)
- Fill model: next-day close (conservative, configurable)
- Alpha = portfolio return - SPY return over the simulation period

---

## Usage

```bash
# Signal quality report + weight optimizer + veto analysis
python backtest.py --mode signal-quality

# Portfolio simulation (live signals)
python backtest.py --mode simulate

# Parameter sweep (live signals)
python backtest.py --mode param-sweep

# Predictor-only backtest (10y synthetic signals, primary sweep source)
python backtest.py --mode predictor-backtest

# All modes (what the weekly spot instance runs)
python backtest.py --mode all

# Upload results to S3 and send email
python backtest.py --mode all --upload

# Override research.db path (local development)
python backtest.py --mode signal-quality --db ~/path/to/research.db

# Rollback all S3 configs to previous versions
python backtest.py --rollback
```

---

## Configuration Reference

`config.yaml` is gitignored — copy from `config.yaml.example`:

| Section | Controls |
|---------|----------|
| `signals_bucket` / `output_bucket` | S3 bucket names |
| `simulation` | Slippage (bps), next-day fill toggle |
| `param_sweep` | Core 6-parameter grid values |
| `param_sweep_settings` | Random/grid mode, trial count, seed |
| `predictor_backtest` | Trading day range, full vs slim cache |
| `weight_optimizer` | Blend factor ramp, guardrails, horizon blend |
| `executor_optimizer` | Min Sharpe improvement, drawdown penalty |
| `veto_analysis` | Confidence thresholds, cost penalty weight |
| `email` | Sender, recipients |

---

## Key Files

```
alpha-engine-backtester/
├── backtest.py                     # CLI entry point (5 modes + rollback)
├── loaders/
│   ├── signal_loader.py            # S3 signals loader
│   └── price_loader.py             # S3 → yfinance → IBKR fallback + gap/freshness detection
├── analysis/
│   ├── signal_quality.py           # Accuracy metrics (Wilson CI, BH FDR)
│   ├── regime_analysis.py          # Accuracy by market regime
│   ├── score_analysis.py           # Accuracy by score range + optimal threshold
│   ├── attribution.py              # Sub-score correlation (BH FDR, predictor IC)
│   ├── param_sweep.py              # Random search over core 6 risk parameters
│   ├── veto_analysis.py            # Veto threshold sweep (precision/cost/lift)
│   └── stats_utils.py              # Benjamini-Hochberg FDR correction
├── optimizer/
│   ├── weight_optimizer.py         # Scoring weights → S3 (stability tracking)
│   ├── executor_optimizer.py       # Executor params → S3 (holdout validation)
│   ├── research_optimizer.py       # Research boosts → S3 (deferred, 200+ samples)
│   └── rollback.py                 # S3 config backup + restore
├── synthetic/
│   ├── predictor_backtest.py       # 10y GBM-based synthetic signal pipeline
│   └── signal_generator.py         # Technical scoring + GBM enrichment
├── vectorbt_bridge.py              # Orders → vbt.Portfolio (slippage + next-day fill)
├── reporter.py                     # Markdown + CSV + metrics.json + S3 upload
├── emailer.py                      # SES email delivery
├── config.yaml.example             # Template — copy to config.yaml
└── infrastructure/
    ├── spot_backtest.sh            # Launch spot instance for weekly backtest
    ├── add-cron.sh                 # Idempotent cron registration (spot launcher)
    └── setup-ec2.sh               # Post-clone EC2 setup (for always-on instance)
```

---

## Deployment

### Spot Instance (Weekly Backtest)

The backtester runs weekly on a c5.large spot instance (~$0.03/hr) launched from the always-on EC2. The spot script handles the full lifecycle: launch → clone repos → install deps → copy config → run backtest → upload to S3 → self-terminate. Total cost: ~$0.01/week.

```bash
# Launch from local machine (or always-on EC2 via cron)
bash infrastructure/spot_backtest.sh

# Smoke test only
bash infrastructure/spot_backtest.sh --smoke-only

# Override instance type for heavier workloads
bash infrastructure/spot_backtest.sh --instance-type c5.xlarge
```

| Step | Time (UTC) | Action |
|------|------------|--------|
| Cron fires | Monday 08:00 | `spot_backtest.sh` launches c5.large spot |
| Bootstrap | ~08:02 | Clone repos, install deps, copy config |
| Backtest | ~08:05 | Signal quality + predictor backtest (10y, 60 trials) |
| Complete | ~08:15 | Results to S3, email sent, instance terminated |

### Always-On EC2

The always-on instance (t3.micro) hosts nginx (nousergon.ai), the Streamlit dashboard, IB Gateway, and the executor/daemon. It launches the spot instance for backtesting via cron.

```bash
# Deploy latest code
git push origin main && ae "cd ~/alpha-engine-backtester && git pull"

# Register cron (launches spot weekly)
ae "cd ~/alpha-engine-backtester && bash infrastructure/add-cron.sh"

# View logs
ae "tail -50 /var/log/backtester.log"
```

---

## Testing

```bash
pytest tests/ -v  # 189 tests
```

---

## S3 Contract

### Reads
| Path | Source | Content |
|------|--------|---------|
| `signals/{date}/signals.json` | Research | Historical signals for accuracy analysis |
| `research.db` | Research | score_performance, predictor_outcomes, universe_returns, scanner/team/CIO evaluations |
| `trades.db` | Executor | Trade history, shadow book, EOD P&L (via SCP from trading EC2) |
| ArcticDB `universe` | Data | 10y OHLCV for synthetic predictor backtest |

### Writes
| Path | Content |
|------|---------|
| `config/scoring_weights.json` | Auto-optimized quant/qual weights → Research |
| `config/executor_params.json` | Auto-optimized risk params → Executor |
| `config/predictor_params.json` | Auto-optimized veto threshold → Predictor |
| `config/research_params.json` | Signal boost params → Research (deferred) |
| `backtest/{date}/report.md` | Weekly markdown report |
| `backtest/{date}/grading.json` | Component grades (A-F scorecard) |
| `backtest/{date}/confusion_matrix.json` | Predictor 3x3 confusion matrix |
| `backtest/{date}/trigger_scorecard.json` | Entry trigger effectiveness |
| `backtest/{date}/shadow_book.json` | Risk guard analysis |
| `backtest/{date}/exit_timing.json` | MFE/MAE exit analysis |
| `backtest/{date}/e2e_lift.json` | Pipeline decision boundary lift |
| `backtest/{date}/veto_analysis.json` | Veto threshold sweep |
| `backtest/grade_history.json` | 52-week rolling component grades |

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor + system overview
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — Meta-model predictor
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard
- [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) — Centralized data collection and ArcticDB
- [`alpha-engine-docs`](https://github.com/cipher813/alpha-engine-docs) — Documentation index

## License

MIT — see [LICENSE](LICENSE).
