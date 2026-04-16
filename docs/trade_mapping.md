# Trade Mapping — Backtester vs `trades.db` Parity Spec

**Purpose:** Define the field mapping and tolerance contract used by
`tests/test_parity_replay.py` to diff backtester-generated orders against
the live executor's `trades.db` over a historical window.

This is Phase 1 of `backtester-audit-260415.md`: the replay parity test is
the operational gate that every production change must pass. Without a
shared schema for "same trade on same date," divergence goes undetected.

---

## Goal

For the last N live-traded dates, replay those dates through the backtester
and assert that the resulting order stream matches `trades.db` within
documented tolerances. Any larger divergence is a regression — either the
backtester has drifted from the executor, or the executor has a bug the
backtester catches.

Phase 0 (ArcticDB cutover) was the hard prerequisite: both sides now read
the same canonical price source, so any remaining divergence is real
logic drift, not data drift.

---

## Data sources

| Side | Source | Populated by |
|---|---|---|
| **Live** | `s3://alpha-engine-research/trades/trades_latest.db` → `trades` table | `alpha-engine/executor/trade_logger.py` after each fill |
| **Replay** | Output of `executor.main.run(simulate=True, signals_override=...)` captured in `all_orders` list | `alpha-engine-backtester/backtest.py::_run_simulation_loop` |
| **Prices (both)** | ArcticDB universe library (Phase 0) | `alpha-engine-data/builders/` |

---

## Field mapping

The backtester's `all_orders` list is a list of dicts produced by the
executor's simulate path. Column on the left is what `trades.db` logs; the
middle column is what the backtester emits; the right column is the
comparison rule.

| `trades.db` column | Backtester order field | Comparison rule |
|---|---|---|
| `date` | `order["date"]` | Exact match |
| `ticker` | `order["ticker"]` | Exact match |
| `action` | `order["action"]` ∈ {`ENTER`, `EXIT`, `REDUCE`} | Exact match |
| `shares` | `order["shares"]` | Within ±1 share (rounding at fill) |
| `fill_price` | `order["fill_price"]` | Within ±0.1% relative, ±$0.01 absolute |
| `price_at_order` | `order["price_at_order"]` | Within ±0.1% relative |
| `trigger_type` | `order["trigger_type"]` ∈ {`pullback`, `vwap`, `support`, `expiry`, `market_open`, None} | Exact match when both non-null; divergence reported separately when one side logs null |
| `trigger_price` | `order["trigger_price"]` | Within ±0.2% relative (triggers measure intraday bands) |
| `signal_price` | `order["signal_price"]` | Within ±0.1% |
| `research_score` | `order["research_score"]` | Within ±0.5 (score is integer-like, tolerance accounts for upstream rounding in config propagation) |
| `predicted_direction` | `order["predicted_direction"]` | Exact match |
| `prediction_confidence` | `order["prediction_confidence"]` | Within ±0.02 |
| `position_pct` | `order["position_pct"]` | Within ±0.005 (0.5 bps of NAV) |
| `realized_return_pct` | `order["realized_return_pct"]` (EXITs only) | Within ±0.1% — closes roundtrip |
| `days_held` | `order["days_held"]` (EXITs only) | Exact match |
| `entry_trade_id` | via `signal_price` matching | EXIT rows: confirm both sides pair to the same entry |

**Not compared** (live-only, no backtester analogue):
- `trade_id`, `created_at`, `fill_time`, `ib_order_id`, `execution_latency_ms`
- `spy_price_at_order`, `spy_return_during_hold` — caller-side enrichment
  that doesn't affect parity (drift here is a telemetry issue, not a trade
  correctness issue)
- `rationale_json`, `thesis_summary` — prose, not structured

**Not compared** (backtester-only):
- Internal simulator timestamps, intermediate risk-guard debug fields

---

## Divergence categories

The parity test reports at three escalating granularities. Each has its
own threshold; any category crossing threshold fails the test.

### 1. Trade count per day

```
|N_backtester_trades(d) - N_live_trades(d)| / max(N_live_trades(d), 1) > 0.05
```

Threshold: **5% per-day divergence**. Captures gross logic drift (e.g.,
risk guard blocking entries the live executor took, or vice versa).

Reports: `{date: {live: N, backtester: M, diff: M-N, pct: ...}}`.

### 2. Ticker set divergence

For each date, compute symmetric difference between live tickers and
backtester tickers (separately for ENTER, EXIT, REDUCE).

Threshold: **>1 ticker divergence on any single day**, OR cumulative
divergence >5% across the N-date window.

Reports: `{date: {only_live: [...], only_backtester: [...]}}`.

### 3. Price / field divergence (per-trade)

For trades that appear on both sides (matched by `(date, ticker, action)`),
diff the fields per the tolerance table above.

Threshold: **any trade with >1 field outside tolerance** fails the test.

Reports: `{trade_id: {field: {live: X, backtester: Y, delta: ..., threshold: ...}}}`.

---

## Example diff output

```json
{
  "status": "fail",
  "run_date": "2026-04-16",
  "window": ["2026-04-02", "2026-04-15"],
  "n_live_trades": 47,
  "n_backtester_trades": 45,
  "trade_count_divergence": {
    "2026-04-09": {"live": 4, "backtester": 2, "diff": -2, "pct": 0.50}
  },
  "ticker_set_divergence": {
    "2026-04-09": {
      "only_live":        ["PLTR", "NVDA"],
      "only_backtester":  []
    }
  },
  "field_divergence": [
    {
      "date": "2026-04-10",
      "ticker": "AAPL",
      "action": "ENTER",
      "fields": {
        "fill_price":    {"live": 172.34, "backtester": 172.11, "delta_rel": 0.00133, "threshold": 0.001}
      }
    }
  ],
  "assessment": "2 trades missing from backtester on 2026-04-09; 1 fill_price outside tolerance. Investigate executor diff since 2026-04-09."
}
```

---

## Execution contract

```
pytest tests/test_parity_replay.py -m parity
```

**Inputs:**
- `TRADES_DB_PATH` env var OR `s3://alpha-engine-research/trades/trades_latest.db` downloaded to a tmpfile
- `SIGNALS_BUCKET` env var (defaults to `alpha-engine-research`)
- Optional: `PARITY_WINDOW_DAYS` (default 10)

**Outputs:**
- Stdout: summary table
- `backtest/{run_date}/parity_report.json` (uploaded to S3 on spot-instance runs)
- Exit 0 on pass; exit 1 on any category breach

**Opt-in via pytest marker:** the test is gated by `@pytest.mark.parity` and
excluded from the default collection so CI on feature branches doesn't try
to reach S3. Spot instance runs it explicitly via
`infrastructure/spot_backtest.sh` post-backtest.

**Integration into `spot_backtest.sh`** (Phase 1.4, follow-up commit):
```bash
python backtest.py --mode all --upload && \
python -m pytest tests/test_parity_replay.py -m parity -v
# Non-zero exit → send parity divergence email; zero exit → report green
```

---

## What this does not cover

- **Forward-looking parity** for changes not yet merged. The test compares
  historical backtester output to historical live trades. A not-yet-merged
  change needs a staging run against a synthetic baseline, not this test.
- **Intraday micro-timing** differences caused by IB's 15-min delayed data
  vs. ArcticDB's daily-close bar. VWAP/support trigger prices will never
  match exactly because the backtester doesn't have intraday ticks.
  Tolerance of ±0.2% absorbs the usual mismatch; larger drifts flag a
  real logic divergence.
- **Drawdown circuit-breaker state**. If live hit a circuit-breaker day
  that the backtester doesn't reproduce (e.g., NAV floor config drift),
  expect trade-count divergence. The test reports this but does not
  distinguish "circuit-breaker fired" from "logic bug" automatically.
  Manual triage required.

---

## Revision protocol

When a parity divergence is legitimately resolved (e.g., a planned
executor change that the backtester should follow on the next run), the
engineer adding the PR should:
1. Update `tests/test_parity_replay.py` tolerance or expected delta with
   a dated comment explaining the cause
2. Bump `CHANGELOG.md` in this repo
3. Re-run Phase 1 against the prior N-day window to confirm zero drift
   before merging
