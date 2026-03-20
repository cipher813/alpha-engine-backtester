# Alpha Engine Backtester

Signal quality analysis, portfolio simulation, and autonomous parameter optimization for the Alpha Engine trading system. The system's learning mechanism — validates whether signals predict outperformance and feeds optimized parameters back to upstream modules.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

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

python backtest.py --mode signal-quality
```

---

## Modes

### Signal Quality (`--mode signal-quality`)

Reads `score_performance` from `research.db` and computes:
- % of BUY signals that beat SPY at 10d and 30d horizons (Wilson CI, BH FDR correction)
- Accuracy by score bucket (60-70, 70-80, 80-90, 90+) with exploratory flags for small samples
- Accuracy by market regime (bull / neutral / bear / caution)
- Sub-score attribution (news vs research correlation with outperformance)
- **Scoring weight recommendation** — applied to S3 automatically if guardrails pass
- **Veto threshold analysis** — sweeps predictor confidence thresholds with precision/cost tradeoff
- **Predictor rolling metrics** — backfills outcomes, pushes 30-day hit rate + IC to S3

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
                              └─ backtest/{date}/report.md
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
pytest tests/ -v
```

---

## Deferred Opportunities

These items are documented for future implementation as the system matures and data accumulates.

### Expand to Extended Grid (at 6+ months of live data)

The current core grid has 6 parameters. An `EXTENDED_GRID` with 16 parameters (adding reduce_fraction, confidence_sizing, staleness, earnings, momentum, correlation) is defined in `param_sweep.py` and can be activated via `config.yaml`. Requires sufficient data to avoid overfitting — at 6+ months (~120 signal dates), the holdout validation becomes robust enough to support more parameters.

### Research Boost Optimizer (at 200+ samples)

`optimizer/research_optimizer.py` tunes 10 signal boost parameters (short interest thresholds, institutional boost, consistency scoring). Currently gated behind 200 samples minimum because boost correlations are too noisy with fewer observations. The heuristic approach (±15% nudges based on correlation direction) should be upgraded to a proper grid search when data supports it.

### Volume-Based Fill Simulation

Currently assumes 100% fill rate. A fill-rate model based on order size vs average daily volume would reject simulated orders representing >5% of daily volume, improving simulation realism for small-cap positions.

### Walk-Forward Cross-Validation

The current 70/30 holdout split is a single evaluation. Walk-forward validation (rolling 70/30 windows advancing by 20% each) would provide more robust OOS estimates and reduce the risk of a lucky holdout split.

### Executor Parameter Stability Tracking

Like the weight optimizer's 3-week direction reversal check, the executor optimizer could track whether optimal parameters flip-flop week over week (e.g., `atr_multiplier` alternating between 2.0 and 4.0). This would flag when the optimization is fitting noise rather than finding stable structure.

### 10-Year Rolling-Window Analysis

The current 10y backtest uses the full window as one block. Rolling-window analysis (e.g., 3-year train / 1-year test, advancing by 6 months) would reveal how parameter stability varies across market regimes and identify regime-dependent parameters that should be excluded from auto-tuning.

### Downsize Always-On EC2

With bursty compute (backtester, predictor training) moved to spot instances, the always-on EC2 only needs nginx + Streamlit + IB Gateway. A t3.micro ($7.57/month) may be sufficient, saving ~$10-30/month vs a larger instance.

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution, intraday daemon, system overview)
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
