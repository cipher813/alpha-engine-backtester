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

The backtester runs weekly on EC2 via cron:

| Step | Time | Action |
|------|------|--------|
| EventBridge | Monday ~07:45 UTC | Starts EC2 instance |
| Cron | Monday 08:00 UTC | Runs `backtest.py --mode all --upload --stop-instance` |
| Completion | ~08:30 UTC | Instance stops itself |

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

## Related Modules

- [`alpha-engine`](https://github.com/cipher813/alpha-engine) — Executor (trade execution + system overview)
- [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) — Autonomous LLM research pipeline
- [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) — GBM predictor (5-day alpha predictions)
- [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) — Streamlit monitoring dashboard

---

## License

MIT — see [LICENSE](LICENSE).
