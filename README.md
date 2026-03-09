# alpha-engine-backtester

Backtesting and signal quality analysis for the [alpha-engine](https://github.com/cipher813/alpha-engine) trading system.

Answers three questions the live system cannot:

1. **Do the signals work?** — What % of BUY-rated stocks outperform SPY over 10d and 30d windows?
2. **Are the risk parameters right?** — Would different `min_score`, `max_position_pct`, or `drawdown_circuit_breaker` values produce better risk-adjusted returns?
3. **Is signal quality improving or degrading?** — As the research pipeline evolves, are signals getting sharper or noisier?

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
     ├── reporter.py           markdown + CSV + S3 upload + SES email
     └── config.yaml

              ↓ outputs ↓

   s3://your-bucket/backtest/{date}/report.md
   results/{date}/report.md
   results/{date}/signal_quality.csv
   results/{date}/metrics.json
```

The backtester is **read-only** with respect to all upstream systems.

---

## Two modes

### Mode 1 — Signal quality (available now)

Reads `score_performance` from `research.db` and computes:
- % of BUY signals that beat SPY at 10d and 30d
- Accuracy by score bucket (60–70, 70–80, 80–90, 90+)
- Accuracy by market regime (bull / neutral / bear / caution)
- Sub-score attribution (technical vs. news vs. research)

### Mode 2 — Portfolio simulation (available Week 4+)

Replays all historical signal dates through `executor.main.run(simulate=True)`, converts orders to a `vbt.Portfolio`, and produces Sharpe ratio, max drawdown, alpha vs SPY, and win rate. See [DOCS.md](DOCS.md#mode-2--portfolio-simulation) for details.

---

## Quick start

```bash
git clone https://github.com/cipher813/alpha-engine-backtester.git
cd alpha-engine-backtester
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.yaml.example config.yaml   # edit bucket name and email
python backtest.py --mode signal-quality
```

AWS credentials must be configured (`aws configure` or IAM role). The S3 bucket must contain `research.db` and at least some `signals/{date}/signals.json` files.

---

## Usage

```bash
# Signal quality report (Mode 1)
python backtest.py --mode signal-quality

# Portfolio simulation (Mode 2 — requires 20+ signal dates)
python backtest.py --mode simulate

# Both modes
python backtest.py --mode all

# Upload results to S3 and send email report
python backtest.py --mode signal-quality --upload

# Override research.db path (local development)
python backtest.py --mode signal-quality --db ~/path/to/research.db

# Date label for output directory
python backtest.py --mode signal-quality --date 2026-03-09
```

---

## Requirements

- Python 3.11+
- AWS credentials with access to the research S3 bucket
- `research.db` in S3 (written by alpha-engine-research after each pipeline run)
- `signals/{date}/signals.json` in S3 (one file per trading day)

See [DOCS.md](DOCS.md) for full setup, EC2 deployment, IAM policy, and vectorbt metric reference.

---

## Cron schedule (EC2)

```
0 14 * * 0   cd /home/ec2-user/alpha-engine-backtester && \
             .venv/bin/python backtest.py --mode signal-quality --upload \
             >> /var/log/backtester.log 2>&1
```

Runs every Sunday at 14:00 UTC (9am ET / 6am PT). Results uploaded to S3 and emailed.

---

## License

MIT
