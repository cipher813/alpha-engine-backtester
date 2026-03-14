# Alpha Engine Backtester

## What this repo is

Signal quality analysis and portfolio simulation backtester for the alpha-engine trading system. Reads signals and prices from S3, score_performance from research.db (pulled from S3 at runtime), and outputs markdown reports, CSVs, and email summaries.

## Stack

- Python 3.11, venv at `.venv/`
- vectorbt for portfolio analytics
- AWS: S3 (signals in, backtest results out), SES (weekly email report)
- Deployed on EC2, runs via cron every Sunday at 14:00 UTC

## Key files

```
backtest.py                  # CLI entry point (--mode signal-quality | simulate | all)
loaders/signal_loader.py     # S3 signals loader
loaders/price_loader.py      # S3 prices → yfinance → IBKR fallback chain
vectorbt_bridge.py           # orders list → vbt.Portfolio
analysis/signal_quality.py   # score_performance from research.db
analysis/regime_analysis.py  # accuracy by market regime
analysis/score_analysis.py   # accuracy vs. score threshold
analysis/attribution.py      # sub-score correlation with beat_spy
analysis/param_sweep.py      # grid search scaffold
reporter.py                  # markdown + CSV + metrics.json + S3 upload
emailer.py                   # SES email delivery
config.yaml                  # GITIGNORED — local config with bucket names and email
infrastructure/setup-ec2.sh  # post-clone EC2 setup
infrastructure/add-cron.sh   # idempotent cron registration
```

## Config

`config.yaml` is gitignored and never committed. Copy `config.yaml.example` to `config.yaml` to configure a new environment.

## Rules

- **Never commit design docs.** Design docs (e.g. `*-design-*.md`) are gitignored. Keep them local only.
- `config.yaml` is gitignored — contains bucket names, email addresses, and credentials.
- The backtester is read-only with respect to all upstream systems (S3 signals, research.db).

## Common commands

```bash
# Activate venv
source .venv/bin/activate

# Signal quality report (Mode 1)
python backtest.py --mode signal-quality

# Upload results to S3 and send email
python backtest.py --mode signal-quality --upload

# Portfolio simulation (Mode 2 — requires 20+ signal dates)
python backtest.py --mode simulate
```

## EC2 deployment

```bash
# Deploy latest code
git push origin main && ae "cd ~/alpha-engine-backtester && git pull"

# View logs
ae "tail -50 /var/log/backtester.log"
```

## EC2 cron schedule

```
0 14 * * 0   cd /home/ec2-user/alpha-engine-backtester && \
             .venv/bin/python backtest.py --mode all --upload \
             >> /var/log/backtester.log 2>&1
```

Runs every Sunday at 14:00 UTC (9am ET / 6am PT).
