# alpha-engine-backtester

Backtesting, signal quality analysis, and autonomous scoring optimization for the [alpha-engine](https://github.com/cipher813/alpha-engine) trading system.

Answers four questions the live system cannot:

1. **Do the signals work?** — What % of BUY-rated stocks outperform SPY over 10d and 30d windows?
2. **Are the risk parameters right?** — Would different `min_score`, `max_position_pct`, or `drawdown_circuit_breaker` values produce better risk-adjusted returns?
3. **Is signal quality improving or degrading?** — As the research pipeline evolves, are signals getting sharper or noisier?
4. **Are the scoring weights optimal?** — Which sub-score (technical / news / research) best predicts outperformance, and should the weights be rebalanced?

---

## Architecture

```
alpha-engine-research (Lambda)
  └── s3://your-bucket/signals/{date}/signals.json
  └── s3://your-bucket/prices/{date}/prices.json
  └── s3://your-bucket/research.db

alpha-engine (executor, EC2)
  └── executor/ibkr.py        ← SimulatedIBKRClient
  └── executor/main.py        ← run(simulate=True)

              ↓ reads all of the above ↓

   alpha-engine-backtester
     ├── backtest.py           CLI entry point
     ├── loaders/
     │   ├── signal_loader.py  S3 signals
     │   └── price_loader.py   S3 prices → yfinance → IBKR fallback chain
     ├── vectorbt_bridge.py    orders → vbt.Portfolio
     ├── analysis/
     │   ├── signal_quality.py
     │   ├── regime_analysis.py
     │   ├── score_analysis.py
     │   ├── attribution.py
     │   └── param_sweep.py
     ├── optimizer/
     │   └── weight_optimizer.py   ← autonomous scoring weight updates
     ├── reporter.py           markdown + CSV + S3 upload + SES email
     └── config.yaml

              ↓ outputs ↓

   s3://your-bucket/backtest/{date}/report.md
   results/{date}/report.md
   results/{date}/signal_quality.csv
   results/{date}/param_sweep.csv
   results/{date}/metrics.json

              ↓ writes back ↓

   s3://your-bucket/config/scoring_weights.json   ← picked up by Lambda on next cold-start
```

The backtester writes one upstream artifact: `config/scoring_weights.json`, updated autonomously by the weight optimizer when the data supports a change.

---

## Modes

### Mode 1 — Signal quality

Reads `score_performance` from `research.db` and computes:
- % of BUY signals that beat SPY at 10d and 30d
- Accuracy by score bucket (60–70, 70–80, 80–90, 90+)
- Accuracy by market regime (bull / neutral / bear / caution)
- Sub-score attribution (technical vs. news vs. research)
- **Scoring weight recommendation** — computed automatically; applied to S3 if guardrails pass

### Mode 2 — Portfolio simulation

Replays all historical signal dates through `executor.main.run(simulate=True)`, converts orders to a `vbt.Portfolio`, and produces Sharpe ratio, max drawdown, Calmar ratio, and win rate.

### Mode: param-sweep

Runs Mode 2 across a grid of `min_score`, `max_position_pct`, and `drawdown_circuit_breaker` values to find the risk parameter combination with the best Sharpe ratio. Price matrix is built once and reused across all combinations.

See [DOCS.md](DOCS.md) for full details on all modes.

---

## Quick start

```bash
git clone https://github.com/cipher813/alpha-engine-backtester.git
cd alpha-engine-backtester
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml   # edit bucket names, paths, and email
python backtest.py --mode signal-quality
```

AWS credentials must be configured (`aws configure` or IAM role). The S3 bucket must contain `research.db` and at least some `signals/{date}/signals.json` files.

---

## Usage

```bash
# Signal quality report + weight optimizer (Mode 1)
python backtest.py --mode signal-quality

# Portfolio simulation (Mode 2)
python backtest.py --mode simulate

# Param sweep — grid search over risk parameters
python backtest.py --mode param-sweep

# All modes
python backtest.py --mode all

# Upload results to S3 and send email report
python backtest.py --mode signal-quality --upload

# Override research.db path (local development)
python backtest.py --mode signal-quality --db ~/path/to/research.db

# Date label for output directory
python backtest.py --mode signal-quality --date 2026-03-09

# Stop the EC2 instance after completion (used by the Sunday cron job)
python backtest.py --mode signal-quality --upload --stop-instance
```

---

## Requirements

- Python 3.11+
- AWS credentials with access to the research S3 bucket
- `research.db` in S3 (written by alpha-engine-research after each pipeline run)
- `signals/{date}/signals.json` in S3 (one file per trading day)
- `alpha-engine` repo cloned locally (required for Mode 2 / param-sweep)
- `alpha-engine-research` repo cloned locally (used to read current scoring weights)

See [DOCS.md](DOCS.md) for full setup, EC2 deployment, IAM policy, and vectorbt metric reference.

---

## EC2 Schedule

The backtester runs automatically every Sunday via EventBridge + cron on the `alpha-engine-executor` EC2 instance.

### EventBridge (starts the instance)

| Schedule | Time | Action |
|----------|------|--------|
| `alpha-engine-sunday-start` | 9:45 AM ET Sunday | Starts EC2 instance |

The instance is normally stopped outside market hours. This schedule boots it in time for the 10:00 AM cron.

### Crontab (runs the backtester)

```
0 14 * * 0  cd /home/ec2-user/alpha-engine-backtester && .venv/bin/python backtest.py --mode signal-quality --upload --stop-instance >> /var/log/backtester.log 2>&1
```

14:00 UTC = 10:00 AM ET. The `--stop-instance` flag stops the EC2 instance automatically when the run completes — no fixed shutdown time needed.

### Sunday flow

1. **9:45 AM ET** — EventBridge starts the instance
2. **10:00 AM ET** — cron fires, backtester runs
3. **When done** — `--stop-instance` stops the instance

Results are uploaded to S3 and emailed. Scoring weights in S3 are updated automatically if the data supports a change.

---

## License

MIT
