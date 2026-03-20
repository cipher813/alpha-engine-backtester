# Alpha Engine Backtester

Signal quality analysis, portfolio simulation, and autonomous parameter optimization for the Alpha Engine trading system. The system's learning mechanism — it validates whether signals actually predict outperformance and feeds optimized parameters back to upstream modules.

> Part of [Nous Ergon: Alpha Engine](https://github.com/cipher813/alpha-engine).

---

## Role in the System

The Backtester closes the feedback loop. It reads historical signals and prices, measures accuracy, identifies which sub-scores are most predictive, and writes three auto-optimization config files to S3 that upstream modules pick up on their next cold-start:

| S3 Key | Read By | Controls |
|--------|---------|----------|
| `config/scoring_weights.json` | Research | Sub-score composite weights |
| `config/executor_params.json` | Executor | Risk parameters and sizing |
| `config/predictor_params.json` | Predictor | Veto confidence threshold |

Without this module, signal generation operates blind to whether its predictions are working.

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
- % of BUY signals that beat SPY at configurable horizons
- Accuracy by score bucket and by market regime
- Sub-score attribution (which scores best predict outperformance)
- **Scoring weight recommendation** — applied to S3 automatically if guardrails pass

### Portfolio Simulation (`--mode simulate`)

Replays historical signal dates through the executor's simulation mode, converts orders to a VectorBT portfolio, and produces Sharpe ratio, max drawdown, Calmar ratio, and win rate.

### Parameter Sweep (`--mode param-sweep`)

Runs portfolio simulation across a grid of risk parameters to find the combination with the best Sharpe ratio. Price matrix is built once and reused across all combinations.

### All Modes (`--mode all`)

Runs signal quality, simulation, and parameter sweep sequentially. This is what the weekly cron job runs.

---

## How It Works

```
research.db (S3)
signals/{date}/signals.json (S3)
         │
         ▼
Signal Quality Analysis
  ├── Accuracy at 10d / 30d horizons
  ├── Score bucket analysis (by score range)
  ├── Regime analysis (bull / neutral / bear / caution)
  └── Sub-score attribution (technical vs news vs research)
         │
         ▼
Weight Optimizer
  └── Data-driven weight recommendations (conservative guardrails)
         │
         ▼
Portfolio Simulation (VectorBT)
  └── Historical replay → Sharpe, drawdown, Calmar
         │
         ▼
Parameter Sweep
  └── Grid search over risk params → optimal Sharpe
         │
         ▼
S3 Output
  ├── config/scoring_weights.json    → Research
  ├── config/executor_params.json    → Executor
  ├── config/predictor_params.json   → Predictor
  ├── backtest/{date}/report.md
  └── backtest/{date}/metrics.json
```

---

## Optimization Architecture

The backtester runs three independent optimizers, each writing an S3 config file that the target module reads on cold-start. All optimizers include conservative guardrails to prevent harmful parameter swings.

### Weight Optimizer → `config/scoring_weights.json`

Tunes the balance between Research sub-scores (news vs research) in the composite attractiveness score.

**Approach:** Correlation-based optimization with conservative blending.

1. Split `score_performance` data 70/30 by date (train/test)
2. Correlate each sub-score with `beat_spy_10d` and `beat_spy_30d` on train set
3. Blend horizons 50/50 (configurable)
4. Progressive blend factor: 20% data-driven at low sample sizes → 50% at 500+ samples
5. Validate on test set: OOS degradation must be < 20%
6. Stability check: no direction reversals in prior 3 weeks

**Guardrails:**
- Max single weight change: 10%
- Min meaningful change: 3%
- Minimum confidence: "medium" (50+ samples per tier)
- OOS validation must pass
- Weights always normalize to sum = 1.0

### Executor Optimizer → `config/executor_params.json`

Recommends the best executor parameter combination from the parameter sweep, ranked by risk-adjusted return.

**Approach:** Random search (Bergstra & Bengio sampling) over a configurable grid.

**Search strategy:**
- Trial count = 25% of grid size, clamped to [50, 400]
- 95% confidence of finding a top-5% configuration with ~60 trials
- Each trial runs a full portfolio simulation via VectorBT

**Ranking function:** `combined_score = Sharpe ratio - 0.5 × max_drawdown`

**Improvement gate:** Best combo must exceed baseline by ≥ 10% Sharpe improvement. Holdout validation (last 30% of dates) must achieve ≥ 50% of training Sharpe.

**Parameter sweep space (16 parameters):**

| Parameter | Grid values | Controls |
|-----------|-------------|----------|
| `min_score` | 65, 70, 75, 80 | Minimum research score to enter |
| `max_position_pct` | 0.05, 0.10, 0.15 | Max single position (% NAV) |
| `drawdown_circuit_breaker` | 0.10, 0.15, 0.20 | Halt threshold (excluded from auto-tune) |
| `atr_multiplier` | 2.0, 3.0, 4.0 | ATR trailing stop multiplier |
| `time_decay_reduce_days` | 5, 7, 10 | Days before position reduction |
| `time_decay_exit_days` | 10, 15, 20 | Days before full exit |
| `reduce_fraction` | 0.25, 0.33, 0.50 | Fraction to sell on REDUCE |
| `atr_sizing_target_risk` | 0.01, 0.02, 0.03 | Target risk per trade (ATR sizing) |
| `confidence_sizing_min` | 0.6, 0.7, 0.8 | Min p_up for confidence sizing |
| `confidence_sizing_range` | 0.4, 0.6, 0.8 | p_up range for linear scaling |
| `staleness_decay_per_day` | 0.02, 0.03, 0.05 | Score decay per day of signal age |
| `earnings_sizing_reduction` | 0.30, 0.50, 0.70 | Sizing reduction near earnings |
| `earnings_proximity_days` | 3, 5, 7 | Days before earnings to reduce |
| `momentum_gate_threshold` | -10.0, -5.0, -2.0 | Min 5d momentum to enter |
| `correlation_block_threshold` | 0.70, 0.75, 0.80, 0.85 | Max pairwise correlation allowed |
| `profit_take_pct` | 0.15, 0.20, 0.25, 0.30 | Gain % to trigger profit-taking |

**Safe-to-auto-tune** (all except `drawdown_circuit_breaker`, `max_sector_pct`, `max_equity_pct` which are too dangerous to auto-adjust).

### Veto Analyzer → `config/predictor_params.json`

Calibrates the predictor's veto confidence threshold — the minimum GBM confidence required to override a BUY signal with a DOWN prediction.

**Approach:** Precision-cost tradeoff optimization.

**Thresholds swept:** 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80

**Scoring function:** `score = precision - 0.30 × (missed_alpha / max_missed_alpha)`
- Precision = % of vetoed signals that actually underperformed (true negatives)
- Cost = alpha left on the table by false vetoes
- Cost sensitivity analysis: sweeps penalty weights [0.15, 0.30, 0.50, 0.70]

**Gates to pass:**
- Min 30 predictions total
- Min 10 veto decisions per threshold
- Lift > 5% over base rate (% of BUY signals that already beat SPY)
- Threshold change ≥ 0.10 from current setting

### Portfolio Simulation (VectorBT)

All parameter sweep trials and the standalone simulation mode use VectorBT:

- `init_cash`: Starting NAV (default $1M)
- `fees`: Transaction cost (default 10bps round-trip)
- `slippage_bps`: Additional slippage per side (configurable)
- `assume_next_day_fill`: Shift ENTER orders to next trading day (conservative)
- Alpha computed as portfolio return minus SPY return over the simulation period

---

## Usage

```bash
# Signal quality report + weight optimizer
python backtest.py --mode signal-quality

# Portfolio simulation
python backtest.py --mode simulate

# Parameter sweep
python backtest.py --mode param-sweep

# All modes
python backtest.py --mode all

# Upload results to S3 and send email
python backtest.py --mode signal-quality --upload

# Override research.db path (local development)
python backtest.py --mode signal-quality --db ~/path/to/research.db
```

---

## Configuration Reference

`config.yaml` is gitignored — copy from `config.yaml.example`:

| Section | Controls |
|---------|----------|
| `s3` | Bucket names, signal paths, output paths |
| `analysis` | Accuracy windows, score bucket ranges, regime definitions |
| `optimizer` | Weight change guardrails, minimum data requirements |
| `param_sweep` | Risk parameter grid ranges, optimization target |
| `email` | Sender, recipients, SES region |

---

## Key Files

```
alpha-engine-backtester/
├── backtest.py                  # CLI entry point
├── loaders/
│   ├── signal_loader.py         # S3 signals loader
│   └── price_loader.py          # S3 → yfinance → IBKR fallback chain
├── analysis/
│   ├── signal_quality.py        # Accuracy metrics
│   ├── regime_analysis.py       # Accuracy by market regime
│   ├── score_analysis.py        # Accuracy by score range
│   ├── attribution.py           # Sub-score correlation with outperformance
│   └── param_sweep.py           # Grid search over risk parameters
├── optimizer/
│   └── weight_optimizer.py      # Autonomous scoring weight updates
├── vectorbt_bridge.py           # Orders → vbt.Portfolio
├── reporter.py                  # Markdown + CSV + S3 upload + email
├── emailer.py                   # SES email delivery
├── config.yaml.example          # Template — copy to config.yaml
└── infrastructure/
    ├── setup-ec2.sh             # Post-clone EC2 setup
    └── add-cron.sh              # Idempotent cron registration
```

---

## Deployment (EC2)

The backtester runs weekly on EC2 via cron. The EC2 instance runs 24/7 (hosts nousergon.ai and the private dashboard), so no start/stop orchestration is needed.

| Step | Time (UTC) | Time (ET) | Action |
|------|------------|-----------|--------|
| Cron | Monday 08:00 | Monday 3:00 AM | Runs `backtest.py --mode all --upload` |
| Completion | ~08:30 | ~3:30 AM | Results uploaded to S3, email sent |

Results are uploaded to S3 and emailed. Scoring weights and parameter configs are updated automatically if the data supports a change.

```bash
# Deploy latest code to EC2
git push origin main && ae "cd ~/alpha-engine-backtester && git pull"

# View logs
ae "tail -50 /var/log/backtester.log"
```

---

## Testing

```bash
pytest tests/ -v
```

---

## Opportunities for Improvement

### Statistical Rigor

- **No multiple testing correction** — across 20-30 test statistics per run, no Bonferroni or Benjamini-Hochberg correction. Expected false positive rate: 30-50%. Plan: apply BH FDR correction at alpha=0.05 across all correlation tests in a single run.
- **MIN_SAMPLES = 10 is too low** — 10 samples is too small for reliable statistical conclusions. Plan: raise to 30 and flag buckets with <20 samples as "exploratory" in reports.
- **Score bucket analysis reports small buckets without caveat** — small buckets are unreliable but displayed as definitive. Plan: add sample size badges and suppress buckets below a configurable floor from weight optimization inputs.

### Simulation Realism

- **No slippage simulation** — orders fill at close price on signal date; real execution fills at next-day open with market impact. Plan: add configurable slippage model (fixed bps or volume-based impact function) to VectorBT bridge.
- **No fill simulation** — assumes 100% fill rate at exact price. Plan: add fill-rate model based on order size vs average daily volume, rejecting simulated orders that would represent >5% of daily volume.

### Optimization Integrity

- **No recommendation stability tracking** — each week's recommendation is independent. System can flip-flop between contradictory recommendations. Plan: maintain a 4-week rolling history of recommendations in S3 and flag as "unstable" if direction reverses (e.g., news weight goes 0.55 → 0.45 → 0.55).
- **Aggressive blending masks strong signals** — blend factor is 20% data-driven, 80% current. With 200+ samples and strong signal, this is overly conservative. Plan: scale blend factor with sample size (e.g., 20% at n=50, 40% at n=200, 60% at n=500).

### Veto Analysis

- **Precision metric noisy on small samples** — precision=100% on n=1 is meaningless. Plan: require minimum n per threshold before including in recommendation.
- **No base rate accounting** — if 80% of BUY signals beat SPY anyway, a veto with 30% precision barely beats random. Plan: compare veto precision against the base accuracy rate.
- **Cost penalty weight is arbitrary** — 0.30 has no justification. Plan: run sensitivity analysis across [0.10, 0.30, 0.50, 0.70] and report recommendation stability.

### Data Quality

- **Forward/backward fill masks price gaps** — if a ticker has a 20-day gap (delisted temporarily), ffill repeats stale prices. Plan: detect gaps >5 days and mark affected rows as invalid rather than filling.
- **No data freshness validation** — prices from S3 not checked for trading day validity. Plan: validate that price dates align with NYSE trading calendar before simulation.

---

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
