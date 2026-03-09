# Alpha Engine — Backtester Design
_Drafted: 2026-03-06_

---

## Current State
_Updated: 2026-03-09_

### What's built

The `alpha-engine-backtester` repo is scaffolded with all modules from the design:

| File | Status | Notes |
|------|--------|-------|
| `backtest.py` | ✅ Built | CLI entry point; Mode 1 runs; Mode 2 raises `NotImplementedError` until 20+ days of signal history available |
| `loaders/signal_loader.py` | ✅ Built | Lists + loads `signals/{date}/signals.json` from S3 |
| `loaders/price_loader.py` | ✅ Built | S3 → yfinance → IBKR fallback chain; tickers auto-resolved from signals.json when prices.json missing |
| `vectorbt_bridge.py` | ✅ Built | `orders_to_portfolio()` + `portfolio_stats()` |
| `analysis/signal_quality.py` | ✅ Built | Reads `score_performance`; returns `insufficient_data` until Week 4+ |
| `analysis/regime_analysis.py` | ✅ Built | Joins `score_performance` + `macro_snapshots`; deferred until Week 4+ |
| `analysis/score_analysis.py` | ✅ Built | Accuracy vs. score threshold; deferred until Week 4+ |
| `analysis/attribution.py` | ✅ Built | Sub-score correlation; deferred until Week 8+ |
| `analysis/param_sweep.py` | ✅ Built | Grid search scaffold; `run_simulation_fn` must be wired up in `backtest.py` (Week 8+) |
| `reporter.py` | ✅ Built | Builds markdown + CSV + `metrics.json`; uploads to S3 |
| `config.yaml` | ✅ Built | Score thresholds, sweep grid, S3 bucket names, `research_db` path |
| `requirements.txt` | ✅ Built | vectorbt, pandas, boto3, pyyaml, yfinance |

### Open items

#### Phase 0 — Upstream changes ✅ Complete

All three upstream changes have been deployed:

| ID | Repo | Change | Status |
|----|------|--------|--------|
| **0a** | `alpha-engine` | Add `SimulatedIBKRClient` to `executor/ibkr.py` | ✅ Done |
| **0b** | `alpha-engine` | Add `simulate=` mode to `executor/main.py:run()` | ✅ Done |
| **0c** | `alpha-engine-research` | Write `prices/{date}/prices.json` to S3 at end of pipeline run | ✅ Done |

#### Data availability — time-gated items

| Item | Unblocks | Available |
|------|----------|-----------|
| `score_performance` rows with `beat_spy_10d` populated | Mode 1 signal quality report | **Week 4** (~10 trading days after first signals = ~2026-03-20) |
| `score_performance` rows with `beat_spy_30d` populated | Full accuracy + attribution | **Week 8** (~2026-05-01) |
| 20+ days of `signals.json` in S3 | Mode 2 simulation smoke test | **Week 4** (prices for all historical dates back-fillable via yfinance now) |
| 40+ days of signals | Param sweep (Phase 5) | **Week 8** |
| 500+ rows of `score_performance` | Automated weight optimizer (`optimizer/weight_optimizer.py`) | **Month 6+** |

#### Features not yet built

| Item | Phase | Notes |
|------|-------|-------|
| `optimizer/weight_optimizer.py` — automated scoring weight rebalancer | Phase 2 (Month 6+) | Design in §"Phase 2 — Automated rebalancing". Guardrails: min 200 samples, ±10% max change, walk-forward validation. |
| EC2 cron for weekly Sunday runs | ✅ Done | `infrastructure/setup-ec2.sh` (one-time) + `infrastructure/add-cron.sh`. Runs Sundays 14:00 UTC (9am ET). Will produce `insufficient_data` reports until Week 4 — that's fine. |
| `backtest.py` wiring for Mode 2 | Week 4 (data-gated) | Replace `NotImplementedError` in `run_simulate()` with executor import + loop once 20+ signal dates exist in S3 |
| `analysis/param_sweep.py` wiring | Phase 5 (Week 8+) | Pass `run_simulation_fn` callback into `sweep()` from `backtest.py` |

#### Config to verify

- `signals_bucket` / `output_bucket` set to `alpha-engine-research` — confirm bucket name

---

## Purpose

The backtester answers three questions the rest of the system cannot:

1. **Do the signals actually work?** — What % of BUY-rated stocks (score ≥ 70) outperform SPY over 10d and 30d windows?
2. **Are the risk parameters right?** — Would different values of `min_score`, `max_position_pct`, or `drawdown_circuit_breaker` produce better risk-adjusted returns?
3. **Is signal quality improving or degrading?** — As the research pipeline evolves, are the signals getting sharper or noisier?

Without the backtester, the executor is flying blind — placing orders on signals whose quality has never been measured. The backtester is the feedback loop that connects research output to real-world outcomes.

---

## Architecture position

```
alpha-engine-research
  └── research.db (investment_thesis, score_performance, macro_snapshots)
  └── s3://alpha-engine-research/signals/{date}/signals.json
  └── s3://alpha-engine-research/prices/{date}/prices.json   ← NEW (see below)

alpha-engine (executor)
  └── s3://alpha-engine-executor/trades/trades_{date}.db
  └── executor/ibkr.py        ← adds SimulatedIBKRClient
  └── executor/main.py        ← adds simulate= mode

                    ↓ reads all of the above ↓

         alpha-engine-backtester
           ├── backtest.py          → main loop — calls executor.run(simulate=True) per date
           ├── loaders/
           │   ├── signal_loader.py → reads signals/{date}/signals.json from S3
           │   └── price_loader.py  → reads prices/{date}/prices.json from S3
           ├── vectorbt_bridge.py   → orders_to_matrices() — ~30 lines, pure reshape
           ├── analysis/
           │   ├── signal_quality.py
           │   ├── regime_analysis.py
           │   ├── score_analysis.py
           │   ├── attribution.py
           │   └── param_sweep.py
           └── reporter.py          → markdown + CSV output

                    ↓ outputs ↓

         s3://alpha-engine-research/backtest/{date}/report.md
         local results/ directory
```

The backtester is **read-only** with respect to all upstream systems. It never writes to research.db, signals, or trades.db. It only reads them and produces analysis artifacts.

---

## Two distinct backtest modes

### Mode 1 — Signal quality backtest

**Question:** Given a BUY signal on date D with score S, does the stock outperform SPY over the next 10 and 30 trading days?

**Data source:** `research.db → investment_thesis` (already has 223 records from 3 days of pipeline runs). This table grows daily as the research pipeline runs.

**Mechanism:** For each BUY-rated row in `investment_thesis`, fetch forward prices via yfinance and compute:
- 10d forward return vs. SPY 10d return → `beat_spy_10d`
- 30d forward return vs. SPY 30d return → `beat_spy_30d`

**Key insight:** The research pipeline already built `score_performance` in `research.db` to track exactly this. The schema:
```
score_performance(
    symbol, score_date, score, price_on_date,
    price_10d, price_30d,
    spy_10d_return, spy_30d_return,
    return_10d, return_30d,
    beat_spy_10d, beat_spy_30d,
    eval_date_10d, eval_date_30d
)
```
As of 2026-03-06 this table has 9 rows with `beat_spy_10d = NULL` because 10 trading days haven't elapsed yet. **The backtester doesn't need to re-implement this tracking** — it just needs to read `score_performance` once rows are populated and aggregate them into reports.

### Mode 2 — Executor portfolio simulation (vectorbt)

**Question:** If we had run the executor with the current risk parameters against all historical signals, what would the portfolio look like today?

**Data source:** Historical `signals.json` + `prices.json` files from S3.

**Mechanism:** For each historical date, call `executor.main.run(simulate=True)` — the same executor logic, but using `SimulatedIBKRClient` (reads prices from S3 instead of connecting to IB Gateway) and returning an order list instead of placing trades. The backtester collects orders across all dates, converts to vectorbt price/signal matrices via `vectorbt_bridge.py`, and runs `vbt.Portfolio.from_signals()`.

This approach means **the executor is the single source of truth for trading logic** — the backtester has no separate copy of position sizing or risk rules. Any change to `risk_guard.py` or `position_sizer.py` is automatically reflected in simulation.

This mode requires at least 20 trading days of signal history to produce meaningful results. As of 2026-03-06 we have 2 days. **Mode 2 is deferred until Week 4 of paper trading.**

---

## Required changes to upstream modules

Three targeted changes are needed before the backtester can be built. These are small, self-contained additions — they do not change any existing behaviour.

### Change 1 — `executor/ibkr.py`: add `SimulatedIBKRClient`

Add a second class alongside `IBKRClient` that implements the same interface but reads from a prices dict instead of connecting to IB Gateway:

```python
class SimulatedIBKRClient:
    """Drop-in replacement for IBKRClient used in backtesting.
    Reads prices from a pre-loaded dict; never connects to IB Gateway."""

    def __init__(self, prices: dict[str, float], nav: float = 1_000_000.0):
        self._prices = prices      # {ticker: price}
        self._nav = nav
        self._positions: dict = {}

    def get_portfolio_nav(self) -> float:
        return self._nav

    def get_positions(self) -> dict:
        return self._positions

    def get_peak_nav(self, conn) -> float:
        return self._nav

    def get_current_price(self, ticker: str) -> float | None:
        return self._prices.get(ticker)

    def place_market_order(self, ticker: str, action: str, shares: int) -> dict:
        # Record fill for portfolio tracking; return stub order ID
        price = self._prices.get(ticker, 0)
        if action == "BUY":
            self._positions[ticker] = {"shares": shares, "avg_cost": price}
            self._nav -= shares * price
        elif action == "SELL":
            self._positions.pop(ticker, None)
            self._nav += shares * price
        return {"ib_order_id": None}

    def disconnect(self):
        pass
```

This is a pure addition — `IBKRClient` is unchanged.

### Change 2 — `executor/main.py`: add `simulate=` mode

Add an optional `simulate` parameter to `run()`. When `True`, the function accepts an injected IBKR client and returns the order list rather than connecting to IB Gateway:

```python
def run(
    dry_run: bool = False,
    simulate: bool = False,
    ibkr_client=None,          # injected by backtester when simulate=True
    signals_override: dict = None,  # injected signals dict (skips S3 read)
) -> list[dict] | None:
    """
    Returns list of order dicts when simulate=True, else None.
    All other behaviour (risk guard, position sizer, trade logger) is unchanged.
    """
    orders = []
    ...
    # replace: ibkr = IBKRClient(...)
    # with:
    if simulate:
        ibkr = ibkr_client
    else:
        ibkr = IBKRClient(host=..., port=..., client_id=...)
    ...
    # replace: ibkr.place_market_order(...)
    # with:
    if simulate:
        orders.append({"date": run_date, "ticker": ticker, "action": "ENTER", ...})
    else:
        order_result = ibkr.place_market_order(...)
    ...
    if simulate:
        return orders
```

The backtester calls this as:
```python
from executor.main import run as executor_run
from executor.ibkr import SimulatedIBKRClient

for date, signals in signals_by_date.items():
    prices = price_loader.load(date)          # from S3 prices/{date}/prices.json
    client = SimulatedIBKRClient(prices, nav=portfolio_nav)
    orders = executor_run(simulate=True, ibkr_client=client, signals_override=signals)
    all_orders.extend(orders)
```

### Change 3 — `alpha-engine-research`: write daily price snapshot to S3

At the end of each research pipeline run, write a JSON price snapshot alongside `signals.json`:

```
s3://alpha-engine-research/prices/{date}/prices.json
```

Format:
```json
{
  "date": "2026-03-06",
  "prices": {
    "PLTR": {"open": 84.12, "close": 85.47, "high": 86.10, "low": 83.90},
    "NVDA": {"open": 118.30, "close": 119.55, "high": 120.00, "low": 117.80}
  }
}
```

Cover all tickers in `universe` + `buy_candidates`. The research pipeline already fetches these prices for scoring — this is a single `boto3.put_object` call at the end of the run.

**Why is this not redundant with `signals.json`?** `signals.json` contains ratings, scores, conviction levels, and thesis text — it has no OHLCV data. The two files are complementary: signals tell the simulator *which* trades to consider; prices tell it *what they cost*.

**Can IBKR's API serve prices instead?** Yes — `reqHistoricalData` on `IBKRClient` can retrieve daily OHLCV bars for any ticker. Tradeoffs:

| | S3 price snapshot | IBKR reqHistoricalData |
|---|---|---|
| Requires live IB Gateway | No | Yes |
| Rate limits | None | Yes (~50 req/10s, manageable) |
| Historical depth | Only from pipeline start date | Up to 1 year (paper) / longer (live) |
| Price source | yfinance (good enough) | IBKR direct (authoritative) |
| Backtester can run standalone | Yes | No |

The S3 snapshot wins on simplicity: the backtester runs standalone without IB Gateway, and the research pipeline already has the prices in memory at write time. IBKR historical data is a viable fallback if a date is missing from S3.

---

## Data availability timeline

| What | Current (2026-03-06) | Week 4 | Week 8 |
|---|---|---|---|
| `investment_thesis` records | 223 (3 days) | ~1,500 (20 days) | ~3,000 (40 days) |
| `score_performance` with 10d returns | 0 (too early) | ~200 | ~500 |
| `score_performance` with 30d returns | 0 (too early) | 0 (too early) | ~200 |
| `signals.json` files in S3 | 2 | ~20 | ~40 |
| `prices.json` files in S3 | 0 (not yet built) | ~20 | ~40 |
| `trades.db` live trades | 0 (pre-go-live) | 20-50 | 50-150 |

**Implication:** The backtester infrastructure should be built now, but Mode 1 results won't be statistically meaningful until Week 4, and Mode 2 until Week 8.

---

## Module structure

```
alpha-engine-backtester/
├── backtest.py              # CLI entry point
├── loaders/
│   ├── signal_loader.py    # read signals/{date}/signals.json from S3
│   └── price_loader.py     # read prices/{date}/prices.json from S3 (+ yfinance fallback)
├── vectorbt_bridge.py       # orders_to_matrices() — reshape order list for vbt.Portfolio
├── analysis/
│   ├── signal_quality.py   # Mode 1: aggregate score_performance, compute accuracy metrics
│   ├── regime_analysis.py  # split accuracy metrics by market_regime
│   ├── score_analysis.py   # accuracy vs. score threshold (optimal cutoff?)
│   ├── attribution.py      # which sub-score drives returns: technical vs news vs research?
│   └── param_sweep.py      # grid search over risk.yaml params for Mode 2
├── reporter.py              # build markdown + CSV output files
├── config.yaml              # backtest parameters (lookback window, score thresholds, etc.)
└── requirements.txt
```

The module is intentionally thin. The heavy lifting is done by:
- `executor.main.run(simulate=True)` — all trading logic
- `vectorbt` — all portfolio math (Sharpe, drawdown, alpha)
- `research.db → score_performance` — all signal quality tracking

---

## Core components

### `vectorbt_bridge.py` — orders to matrices

The only custom code needed to bridge the executor's order list into vectorbt:

```python
import vectorbt as vbt
import pandas as pd

def orders_to_portfolio(
    orders: list[dict],
    prices: pd.DataFrame,   # indexed by date, columns by ticker
    init_cash: float = 1_000_000.0,
) -> vbt.Portfolio:
    """
    Convert executor order list to a vectorbt Portfolio.

    orders: [{"date": "2026-03-06", "ticker": "PLTR", "action": "ENTER",
               "shares": 100, "price_at_order": 84.12}, ...]

    Returns a vbt.Portfolio with full Sharpe, drawdown, and alpha analytics.
    """
    tickers = prices.columns.tolist()
    dates = prices.index

    entries = pd.DataFrame(False, index=dates, columns=tickers)
    exits   = pd.DataFrame(False, index=dates, columns=tickers)
    sizes   = pd.DataFrame(0.0,   index=dates, columns=tickers)  # in shares

    for order in orders:
        d = order["date"]
        t = order["ticker"]
        if t not in tickers or d not in entries.index:
            continue
        if order["action"] == "ENTER":
            entries.loc[d, t] = True
            sizes.loc[d, t]   = order["shares"]
        elif order["action"] in ("EXIT", "REDUCE"):
            exits.loc[d, t] = True

    return vbt.Portfolio.from_signals(
        close=prices,
        entries=entries,
        exits=exits,
        size=sizes,
        size_type="shares",
        init_cash=init_cash,
        fees=0.0,
        freq="D",
    )
```

That's the entire bridge. `vbt.Portfolio` provides `.sharpe_ratio()`, `.max_drawdown()`, `.total_return()`, `.plot()`, and benchmark comparison with SPY out of the box.

### `backtest.py` — main loop (Mode 2)

```python
def run_simulation(config: dict) -> vbt.Portfolio:
    signal_dates = signal_loader.list_dates(config["signals_bucket"])
    all_orders = []

    for date in signal_dates:
        signals = signal_loader.load(config["signals_bucket"], date)
        prices  = price_loader.load(config["signals_bucket"], date)
        nav     = portfolio_tracker.nav  # tracks across dates

        client = SimulatedIBKRClient(prices["prices"], nav=nav)
        orders = executor_run(
            simulate=True,
            ibkr_client=client,
            signals_override=signals,
        )
        portfolio_tracker.apply_orders(orders, prices)
        all_orders.extend(orders)

    price_matrix = price_loader.build_matrix(signal_dates, config["signals_bucket"])
    return orders_to_portfolio(all_orders, price_matrix, init_cash=config["init_cash"])
```

### `analysis/signal_quality.py` — Mode 1

Reads `score_performance` (once populated) and computes accuracy metrics:

```python
def compute_accuracy(df: pd.DataFrame) -> dict:
    """
    Given score_performance rows with beat_spy_10d/30d populated:
    - Overall BUY accuracy at 10d and 30d
    - Accuracy by score bucket (60-70, 70-80, 80-90, 90+)
    - Accuracy by conviction (rising, stable, declining)
    - Accuracy by market_regime (join with macro_snapshots)
    - Average alpha (stock return - SPY return) by bucket
    - Sample size at each slice
    """
```

---

## Step-by-step build plan

### Phase 0 — Upstream changes (before building backtester)

| Step | Repo | Change | Effort |
|---|---|---|---|
| 0a | `alpha-engine` | Add `SimulatedIBKRClient` to `executor/ibkr.py` | ~50 lines |
| 0b | `alpha-engine` | Add `simulate=` mode to `executor/main.py:run()` | ~20 lines |
| 0c | `alpha-engine-research` | Write `prices/{date}/prices.json` to S3 at end of pipeline run | ~15 lines |

These are the only changes needed in the existing repos. Everything else lives in `alpha-engine-backtester`.

### Phase 1 — Repo + loaders (Week 1)

- Repo setup, `requirements.txt` (vectorbt, pandas, boto3, pyyaml)
- `loaders/signal_loader.py` — list and load `signals.json` from S3
- `loaders/price_loader.py` — load `prices.json` from S3; fall back to yfinance if missing

### Phase 2 — Bridge + simulation (Week 2)

- `vectorbt_bridge.py` — `orders_to_portfolio()` (~30 lines)
- `backtest.py` — main loop calling `executor_run(simulate=True)` per date
- Smoke test with 2 available signal dates

### Phase 3 — Signal quality analysis (Week 4, once `score_performance` populated)

- `analysis/signal_quality.py` — aggregate `score_performance`, compute accuracy
- `analysis/regime_analysis.py` — split by `market_regime`
- `analysis/score_analysis.py` — accuracy vs. score threshold

### Phase 4 — Reporter + S3 output (Week 4)

- `reporter.py` — markdown report + CSVs
- Upload to `s3://alpha-engine-research/backtest/{date}/`

### Phase 5 — Attribution + param sweep (Week 8)

- `analysis/attribution.py` — which sub-score (technical/news/research) predicts beat-SPY?
- `analysis/param_sweep.py` — grid search over `risk.yaml` params using Mode 2

---

## Cadence

| Frequency | Mode | Trigger | Meaningful after |
|---|---|---|---|
| **Daily** | Score performance tracking | Already running in research pipeline — `score_performance` table fills automatically | Immediately (but needs 10d to see first 10d returns) |
| **Weekly (Sunday)** | Mode 1 signal quality report | Cron on EC2 or run locally | Week 4 (20+ trading days of theses) |
| **Weekly (Sunday)** | Mode 2 portfolio simulation | Same cron run | Week 8 (40+ days, enough for Sharpe to stabilise) |
| **On-demand** | Parameter sweep | Manual, after major research pipeline changes | Week 8 |
| **Quarterly** | Deep attribution + sweep | Manual | Week 12+ |

---

## Problems it solves

### 1. Validates signal quality before scaling
Before moving from paper to live trading (real money), signal quality needs a track record. The backtester provides an objective go/no-go metric: if BUY accuracy at 10d vs. SPY is consistently above 55% over 30+ trading days, the signals are adding value. Below 50% = random.

### 2. Prevents parameter tuning on gut feel
Every value in `config/risk.yaml` is currently a reasonable assumption. The param sweep replaces assumptions with data. A parameter set that looks conservative (`min_score: 70`) may actually perform worse than a tighter threshold (`min_score: 75`) if noise is high at lower scores.

### 3. Regime validation
The research pipeline classifies market regime (bull/neutral/bear/caution) and the executor adjusts position sizes accordingly. The backtester verifies that this classification is actually correlated with forward returns — i.e., that "bear" signals precede underperformance and "bull" signals precede outperformance. If not, the regime adjustments are adding noise, not value.

### 4. Research pipeline regression testing
When the research pipeline changes (new agent added, scoring weights adjusted, prompt changes), the backtester can be run on historical data to verify that signal quality held or improved. It prevents silent regressions where a "fix" accidentally degrades alpha.

### 5. Feeds the dashboard (future)
The daily NAV series, signal accuracy, and metrics CSVs written by the reporter are the data layer for `alpha-engine-dashboard`. The dashboard doesn't compute anything — it visualises what the backtester produces.

---

## Why vectorbt over pandas

vectorbt is the right tool for this use case:

- **Speed**: vectorbt operations are C-accelerated via NumPy. A pandas simulation loop over 40 dates × 50 tickers takes seconds; vectorbt does it in milliseconds. This matters for the param sweep, which runs hundreds of combinations.
- **Analytics out of the box**: `.sharpe_ratio()`, `.max_drawdown()`, `.calmar_ratio()`, `.plot()`, SPY benchmark comparison — all single method calls. Pandas requires implementing each metric manually.
- **Correctness**: vectorbt handles lookahead bias, fill-at-next-bar logic, and position tracking consistently. Rolling your own portfolio simulator introduces subtle bugs.
- **Iteration speed**: because the backtester will be run hundreds of times as the pipeline evolves, the tooling needs to be fast and reliable. vectorbt is built for exactly this.

The thin `vectorbt_bridge.py` (one function, ~30 lines) is all the glue needed. The executor handles the trading logic; vectorbt handles the analytics.

---

## Feedback loop: backtester → research pipeline

The backtester's `attribution.py` output is the primary mechanism for improving the research pipeline's scoring weights over time. The loop is:

```
backtester attribution output
        ↓
  human reviews (quarterly)
        ↓
  manual weight change in alpha-engine-research scoring config
        ↓
  research pipeline runs with new weights next trading day
        ↓
  backtester validates change didn't degrade signal quality (regression test)
```

### Phase 1 — Manual (now through Month 6)

Attribution results will be noisy with fewer than ~200 populated `score_performance` rows. Automated rebalancing on this little data will overfit. Until Month 6, the process is:

1. Run backtester attribution weekly (Sunday cron)
2. Review quarterly: which sub-score (technical / news / research) shows the highest correlation with `beat_spy_10d`?
3. If one sub-score consistently underperforms, manually adjust weights in `alpha-engine-research` scoring config
4. Re-run backtester on historical data to confirm signal quality held or improved

### Phase 2 — Automated rebalancing (Month 6+)

Once 6+ months of `score_performance` data is available (~500+ rows with 10d returns populated), automated weight optimization becomes viable.

**Planned module:** `alpha-engine-backtester/optimizer/weight_optimizer.py`

```python
def optimize_weights(
    score_performance: pd.DataFrame,
    current_weights: dict,          # {"technical": 0.4, "news": 0.3, "research": 0.3}
    min_samples: int = 200,         # refuse to optimize on less than this
    max_weight_change: float = 0.10, # max ±10% shift per sub-score per cycle
    min_weight: float = 0.15,       # no sub-score below 15%
) -> dict:
    """
    Walk-forward weight optimization:
    1. Split score_performance into train (first 2/3) and validation (last 1/3)
    2. Grid search weights on train set to maximize beat_spy_10d accuracy
    3. Validate best weights on held-out set — only accept if validation accuracy >= train accuracy * 0.9
    4. Apply bounds: no single change > max_weight_change, no weight < min_weight
    5. Write new weights to alpha-engine-research/config/scoring_weights.yaml
    6. Log change to weights_history.json for audit trail
    Returns proposed new weights (not applied until confirmed).
    """
```

**Guardrails:**
- Minimum 200 samples before any optimization runs
- Walk-forward validation — train on first 2/3, validate on last 1/3 before applying
- Bounded changes: ±10% per sub-score per quarterly cycle
- Floor: no sub-score weight below 15% (prevents a signal source being effectively zeroed)
- Changes written to `weights_history.json` — full audit trail, easy rollback
- Optimizer proposes changes; a manual confirmation step applies them (initially — can be made fully automatic once trust is established)

**Output written to:** `alpha-engine-research/config/scoring_weights.yaml` — the research pipeline reads this on each Lambda invocation, so new weights take effect the next trading day without a redeploy.

---

## What it does NOT do

- **Live trading decisions** — the backtester is analysis only. The executor makes all trading decisions.
- **Options or short selling** — equity long-only, matching the executor scope.
- **Intraday simulation** — daily granularity only. The executor places market orders at open; the backtester uses open or close prices as proxies.
- **Transaction cost modelling** — paper trading has no commissions. Add this before simulating live trading.
- **Walk-forward optimisation** — the param sweep uses the full history in-sample. Walk-forward (train on first half, test on second half) is a Phase 2 enhancement once data is abundant.

---

## Relationship to other modules

| Module | Relationship |
|---|---|
| `alpha-engine-research` | Provides signals and price snapshots being evaluated. The backtester is a quality check on this module's output. |
| `alpha-engine` (executor) | Backtester imports `executor.main.run(simulate=True)` and `executor.ibkr.SimulatedIBKRClient` — same logic, historical data. |
| `alpha-engine-dashboard` | Consumes backtester output (CSVs, metrics.json) for visualisation. |
| `alpha-engine-backtester` | Standalone repo — minimal shared code (executor imports above). |
