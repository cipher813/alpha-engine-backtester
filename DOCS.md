# alpha-engine-backtester — Full Documentation

---

## Contents

1. [Setup](#1-setup)
2. [Configuration](#2-configuration)
3. [Data sources](#3-data-sources)
4. [Mode 1 — Signal quality](#4-mode-1--signal-quality)
5. [Mode 2 — Portfolio simulation](#5-mode-2--portfolio-simulation)
6. [vectorbt metrics reference](#6-vectorbt-metrics-reference)
7. [Reporter output](#7-reporter-output)
8. [EC2 deployment](#8-ec2-deployment)
9. [IAM policy](#9-iam-policy)
10. [Development workflow](#10-development-workflow)

---

## 1. Setup

### Prerequisites

- Python 3.11+
- AWS credentials with access to the research S3 bucket (see [§9 IAM policy](#9-iam-policy))
- The [alpha-engine-research](https://github.com/cipher813/alpha-engine-research) pipeline running and writing to S3
- The [alpha-engine](https://github.com/cipher813/alpha-engine) executor repo cloned alongside this one (required for Mode 2)

### Install

```bash
git clone https://github.com/cipher813/alpha-engine-backtester.git
cd alpha-engine-backtester
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Executor dependency (Mode 2 only)

Mode 2 imports `executor.main.run(simulate=True)` and `executor.ibkr.SimulatedIBKRClient` from the alpha-engine repo. The executor must be on `sys.path`:

```bash
# Option A — clone executor next to backtester
git clone https://github.com/cipher813/alpha-engine.git ../alpha-engine
export PYTHONPATH=../alpha-engine:$PYTHONPATH

# Option B — add to config.yaml
executor_path: /home/ec2-user/alpha-engine
```

---

## 2. Configuration

All settings live in `config.yaml`:

```yaml
# S3 bucket written by alpha-engine-research
signals_bucket: alpha-engine-research

# S3 bucket and prefix for backtest output
output_bucket: alpha-engine-research
output_prefix: backtest

# Local results directory
results_dir: results

# Starting portfolio value for Mode 2 simulation
init_cash: 1_000_000.0

# Score thresholds for Mode 1 accuracy-vs-threshold table
score_thresholds: [60, 65, 70, 75, 80, 85, 90]

# Minimum samples required before a bucket appears in the report
min_samples: 5

# Email (AWS SES)
email_sender: "you@example.com"
email_recipients:
  - "you@example.com"

# research.db is pulled from S3 automatically — no path needed.
# Override with --db flag for local development.

# Param sweep grid (Mode 2, Phase 5)
param_sweep:
  min_score: [65, 70, 75, 80]
  max_position_pct: [0.05, 0.10, 0.15]
  drawdown_circuit_breaker: [0.10, 0.15, 0.20]
```

### research.db

`research.db` is a SQLite database maintained by the alpha-engine-research Lambda. At the start of every backtester run, `backtest.py` pulls a fresh copy from `s3://{signals_bucket}/research.db` into a temp file. It is read-only — the backtester never writes to it.

Override with `--db /path/to/research.db` to use a local copy during development:

```bash
# Pull a local copy for development
aws s3 cp s3://alpha-engine-research/research.db ./research.db
python backtest.py --mode signal-quality --db ./research.db
```

---

## 3. Data sources

### Signal files

Written daily by alpha-engine-research at `s3://{bucket}/signals/{date}/signals.json`:

```json
{
  "date": "2026-03-09",
  "signals": [
    {
      "symbol": "PLTR",
      "rating": "BUY",
      "score": 82,
      "conviction": "rising",
      "market_regime": "bull",
      "sub_scores": {"technical": 85, "news": 78, "research": 83}
    }
  ]
}
```

### Price files

Written daily by alpha-engine-research at `s3://{bucket}/prices/{date}/prices.json`:

```json
{
  "date": "2026-03-09",
  "prices": {
    "PLTR": {"open": 84.12, "close": 85.47, "high": 86.10, "low": 83.90}
  }
}
```

### Price fallback chain

`price_loader.py` resolves prices in this order for any date:

1. **S3** `prices/{date}/prices.json` — canonical source
2. **yfinance** — tickers resolved automatically from the corresponding `signals.json`
3. **IBKR `reqHistoricalData`** — optional; pass `ibkr_client=` to `build_matrix()` for gap-filling

This means price data is available for all historical signal dates even before the research pipeline started writing `prices.json` files.

### research.db schema (relevant tables)

```sql
-- One row per BUY signal. beat_spy_10d/30d populated ~10/30 trading days later.
score_performance (
    symbol, score_date, score, price_on_date,
    price_10d, price_30d,
    spy_10d_return, spy_30d_return,
    return_10d, return_30d,
    beat_spy_10d, beat_spy_30d,   -- NULL until evaluation date passes
    eval_date_10d, eval_date_30d
)

-- Daily macro snapshot written by research pipeline
macro_snapshots (
    date, market_regime,          -- "bull" | "neutral" | "bear" | "caution"
    fed_funds_rate, treasury_10yr, yield_curve_slope,
    vix, sp500_close, sp500_30d_return, ...
)

-- Full investment thesis per stock per day
investment_thesis (
    symbol, date, rating, score,
    technical_score, news_score, research_score,
    conviction, signal, ...
)
```

---

## 4. Mode 1 — Signal quality

Reads `score_performance` from `research.db` and aggregates accuracy metrics.

### Run

```bash
python backtest.py --mode signal-quality
python backtest.py --mode signal-quality --upload    # + S3 upload + email
```

### What it computes

| Metric | Description |
|--------|-------------|
| `accuracy_10d` | % of BUY signals where `beat_spy_10d = True` |
| `accuracy_30d` | % of BUY signals where `beat_spy_30d = True` |
| `avg_alpha_10d` | Mean of `return_10d - spy_10d_return` across all signals |
| `avg_alpha_30d` | Mean of `return_30d - spy_30d_return` |

Slices computed: overall, by score bucket (60–70, 70–80, 80–90, 90+), by conviction, by market regime.

### Interpretation

| accuracy_10d | Interpretation |
|---|---|
| < 50% | Signals are subtracting value |
| ~50% | Random — no edge |
| 55–60% | Meaningful edge |
| > 60% | Strong edge |

50 samples is the minimum for meaningful accuracy estimates. Results before Week 4 (~200 `score_performance` rows with 10d returns) will return `insufficient_data` and are expected.

### Score threshold analysis

`analysis/score_analysis.py` computes accuracy for every threshold in `score_thresholds`. Use this to find the optimal `min_score` cutoff — the point where raising the bar improves accuracy faster than it shrinks sample size.

### Attribution

`analysis/attribution.py` computes correlation between each sub-score (technical, news, research) and `beat_spy_10d`. Use this quarterly to identify which signal component is driving returns and adjust scoring weights in the research pipeline accordingly.

---

## 5. Mode 2 — Portfolio simulation

Replays all historical signal dates through the executor's logic and builds a vectorbt portfolio.

### Status

Mode 2 is data-gated. It requires 20+ trading days of `signals.json` in S3. The `run_simulate()` function in `backtest.py` raises `NotImplementedError` until it is wired to the executor. See `backtest.py:run_simulate()` for the wiring instructions.

### How it works

For each historical date:

```python
from executor.main import run as executor_run
from executor.ibkr import SimulatedIBKRClient

signals = signal_loader.load(bucket, date)
prices  = price_loader.load(bucket, date, tickers=all_tickers)
client  = SimulatedIBKRClient(prices["prices"], nav=current_nav)
orders  = executor_run(simulate=True, ibkr_client=client, signals_override=signals)
```

The same executor logic — risk guard, position sizer, conviction check — runs unchanged. No separate copy of trading rules exists in the backtester.

Orders are then passed to `vectorbt_bridge.orders_to_portfolio()` which builds a `vbt.Portfolio` for analysis.

---

## 6. vectorbt metrics reference

### Building the portfolio

```python
from loaders import price_loader, signal_loader
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

# 1. Get all available signal dates
dates = signal_loader.list_dates(bucket="alpha-engine-research")

# 2. Build price matrix (S3 → yfinance fallback)
prices = price_loader.build_matrix(dates, bucket="alpha-engine-research")
# DataFrame: rows = datetime index, columns = ticker symbols

# 3. Build order list (from executor simulation — see Mode 2)
# orders = [{"date": "2026-03-09", "ticker": "PLTR",
#            "action": "ENTER", "shares": 100, "price_at_order": 84.12}, ...]

# 4. Build vectorbt Portfolio
pf = orders_to_portfolio(orders, prices, init_cash=1_000_000.0)
```

### Key metrics

```python
# ── Returns ──────────────────────────────────────────────────────────────────
pf.total_return()           # float — total return over full period
pf.annualized_return()      # float — annualized
pf.daily_returns()          # pd.Series — daily P&L %

# ── Risk-adjusted ─────────────────────────────────────────────────────────────
pf.sharpe_ratio()           # float — higher is better; >1 is good, >2 is excellent
pf.sortino_ratio()          # float — like Sharpe but penalises downside only
pf.calmar_ratio()           # float — annualized return / max drawdown
pf.omega_ratio()            # float — probability-weighted return ratio

# ── Drawdown ──────────────────────────────────────────────────────────────────
pf.max_drawdown()           # float — worst peak-to-trough decline (negative)
pf.drawdown()               # pd.Series — rolling drawdown over time

# ── Trades ────────────────────────────────────────────────────────────────────
pf.trades.count()           # int — total number of closed trades
pf.trades.win_rate()        # float — % of trades that were profitable
pf.trades.avg_pnl()         # float — average P&L per trade
pf.trades.records_readable  # DataFrame — one row per trade, fully annotated

# ── Portfolio value ───────────────────────────────────────────────────────────
pf.value()                  # pd.Series — portfolio NAV over time
pf.cash()                   # pd.Series — uninvested cash over time

# ── Benchmark comparison (SPY) ────────────────────────────────────────────────
import vectorbt as vbt
import yfinance as yf

spy = yf.download("SPY", start=dates[0], end=dates[-1], auto_adjust=True)
spy_pf = vbt.Portfolio.from_holding(spy["Close"], init_cash=1_000_000.0)

print(f"Portfolio Sharpe: {pf.sharpe_ratio():.2f}")
print(f"SPY Sharpe:       {spy_pf.sharpe_ratio():.2f}")
print(f"Portfolio return: {pf.total_return()*100:.1f}%")
print(f"SPY return:       {spy_pf.total_return()*100:.1f}%")
```

### Interactive plots

```python
# Full portfolio stats dashboard
pf.plot().show()

# NAV vs SPY over time
pf.value().vbt.plot(trace_kwargs=dict(name="Portfolio")).show()

# Drawdown chart
pf.drawdown().vbt.plot().show()

# Individual trade waterfall
pf.trades.plot().show()
```

### Summary dict (for metrics.json)

```python
from vectorbt_bridge import portfolio_stats
stats = portfolio_stats(pf)
# {
#   "total_return": 0.142,
#   "sharpe_ratio": 1.38,
#   "max_drawdown": -0.067,
#   "calmar_ratio": 2.11,
#   "total_trades": 34,
#   "win_rate": 0.59
# }
```

### Param sweep

`analysis/param_sweep.py` runs `orders_to_portfolio()` across a grid of executor risk parameters to find the combination with the best Sharpe ratio:

```python
from analysis.param_sweep import sweep, best_params

results_df = sweep(
    grid=config["param_sweep"],
    run_simulation_fn=my_run_fn,   # wraps backtest.run_simulate()
    base_config=config,
)
print(results_df.head(10))         # sorted by sharpe_ratio
print(best_params(results_df))     # {"min_score": 75, "max_position_pct": 0.10, ...}
```

---

## 7. Reporter output

Every run produces three files in `results/{date}/`:

| File | Contents |
|------|----------|
| `report.md` | Full markdown report — signal quality, regime, attribution, portfolio stats |
| `signal_quality.csv` | Accuracy by score threshold — one row per threshold |
| `metrics.json` | Overall summary — status, accuracy_10d/30d, avg_alpha, portfolio stats |

With `--upload`, all three are also written to `s3://{output_bucket}/{output_prefix}/{date}/`.

With `email_sender` configured, an HTML-formatted email is sent via SES. Subject line format:

```
Alpha Engine Backtester | 2026-03-09 | results ready
Alpha Engine Backtester | 2026-03-09 | insufficient data (accumulating)
Alpha Engine Backtester | 2026-03-09 | ERROR
```

---

## 8. EC2 deployment

The backtester runs on the same EC2 instance as the executor (`i-xxxx`, t3.small, Amazon Linux 2023).

### First-time setup

```bash
# 1. SSH in and configure GitHub credentials (one-time)
ae
cat >> ~/.netrc << 'EOF'
machine github.com
login your-github-username
password your-pat
EOF
chmod 600 ~/.netrc

# 2. Clone and set up
git clone https://github.com/cipher813/alpha-engine-backtester.git
bash ~/alpha-engine-backtester/infrastructure/setup-ec2.sh
```

`setup-ec2.sh` creates the virtualenv, installs dependencies, creates `/var/log/backtester.log`, and registers the Sunday cron job.

### Deploying updates

```bash
# From local machine (alpha-engine-backtester repo)
git push origin main && ae "cd ~/alpha-engine-backtester && git pull"
```

### Cron job

```
0 14 * * 0   cd /home/ec2-user/alpha-engine-backtester && \
             .venv/bin/python backtest.py --mode signal-quality --upload \
             >> /var/log/backtester.log 2>&1
```

Sundays, 14:00 UTC = 9:00am ET = 6:00am PT.

### Logs

```bash
ae "tail -50 /var/log/backtester.log"
```

---

## 9. IAM policy

The EC2 instance role (`alpha-engine-executor-role`) requires these S3 permissions:

```json
{
  "Statement": [
    {
      "Sid": "ReadResearchSignals",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::alpha-engine-research",
        "arn:aws:s3:::alpha-engine-research/signals/*"
      ]
    },
    {
      "Sid": "ReadResearchDb",
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": ["arn:aws:s3:::alpha-engine-research/research.db"]
    },
    {
      "Sid": "WriteBacktestResults",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": ["arn:aws:s3:::alpha-engine-research/backtest/*"]
    }
  ]
}
```

`ses:SendEmail` is required for email reports. See `alpha-engine/alpha-engine/infrastructure/s3-policy.json` for the full policy file used to apply these permissions.

---

## 10. Development workflow

### Run locally against a pulled research.db

```bash
# Pull latest research.db from S3
aws s3 cp s3://alpha-engine-research/research.db ./research.db

# Run Mode 1 locally
python backtest.py --mode signal-quality --db ./research.db

# Run with debug logging
python backtest.py --mode signal-quality --db ./research.db --log-level DEBUG
```

### Inspect vectorbt output interactively

```python
# In a Python REPL or notebook after Mode 2 is wired up:
import yaml
from backtest import load_config, run_simulate
from vectorbt_bridge import orders_to_portfolio, portfolio_stats

config = load_config("config.yaml")
# pf = orders_to_portfolio(orders, prices)  # once Mode 2 is wired
# pf.plot().show()
# print(portfolio_stats(pf))
```

### Check score_performance directly

```python
import sqlite3
import pandas as pd

conn = sqlite3.connect("research.db")

# How many rows are populated?
df = pd.read_sql("SELECT * FROM score_performance ORDER BY score_date", conn)
print(f"Total rows: {len(df)}")
print(f"beat_spy_10d populated: {df['beat_spy_10d'].notna().sum()}")
print(f"beat_spy_30d populated: {df['beat_spy_30d'].notna().sum()}")

# Overall accuracy (once populated)
populated = df[df["beat_spy_10d"].notna()]
print(f"10d accuracy: {populated['beat_spy_10d'].mean():.1%}")
```

### Data availability timeline

| Milestone | Date | What becomes available |
|-----------|------|------------------------|
| Pipeline start | 2026-03-05 | signals.json, investment_thesis |
| Week 4 (~2026-03-20) | +10 trading days | First `beat_spy_10d` values |
| Week 8 (~2026-05-01) | +30 trading days | First `beat_spy_30d` values; Mode 2 meaningful |
| Month 6+ | +~120 trading days | Attribution reliable; weight optimizer viable |
