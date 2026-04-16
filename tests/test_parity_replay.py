"""
tests/test_parity_replay.py — replay parity test (Phase 1.1).

Diffs backtester output against the live trades.db over a historical window.
See docs/trade_mapping.md for the field mapping + tolerance contract.

Opt-in via the `parity` pytest marker so CI on feature branches doesn't
try to reach S3. Spot instance runs it explicitly via spot_backtest.sh
after the weekly backtest completes.

Status: SCAFFOLD (2026-04-16).
  * Diff logic (pure functions) is complete and unit-tested below.
  * `_run_backtester_for_dates()` is a placeholder — the integration test
    skips until the wiring lands in a follow-up commit (Phase 1.1b).

Usage:
    # Unit tests (diff logic only) — always run
    pytest tests/test_parity_replay.py

    # Full parity replay — opt-in, requires trades.db + S3 ArcticDB access
    pytest tests/test_parity_replay.py -m parity -v

Environment (for parity run):
    TRADES_DB_PATH       override path to trades.db (else download from S3)
    TRADES_DB_S3_URI     e.g. s3://alpha-engine-research/trades/trades_latest.db
    SIGNALS_BUCKET       default "alpha-engine-research"
    PARITY_WINDOW_DAYS   default 10
"""

from __future__ import annotations

import os
import sqlite3
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# arcticdb is heavy C-ext; stub for local test runs (real calls mocked per test).
sys.modules.setdefault("arcticdb", MagicMock())


# ── Tolerance contract (see docs/trade_mapping.md for rationale) ────────────

@dataclass(frozen=True)
class Tolerance:
    rel: float          # relative tolerance (fraction, e.g. 0.001 = 0.1%)
    abs_: float = 0.0   # absolute tolerance (same units as the field)


FIELD_TOLERANCES: dict[str, Tolerance] = {
    "fill_price":             Tolerance(rel=0.001, abs_=0.01),
    "price_at_order":         Tolerance(rel=0.001),
    "trigger_price":          Tolerance(rel=0.002),
    "signal_price":           Tolerance(rel=0.001),
    "research_score":         Tolerance(rel=0.0, abs_=0.5),
    "prediction_confidence":  Tolerance(rel=0.0, abs_=0.02),
    "position_pct":           Tolerance(rel=0.0, abs_=0.005),
    "realized_return_pct":    Tolerance(rel=0.001),
}

# Shares have rounding at fill time; allow ±1 share
SHARES_ABS_TOLERANCE = 1

# Exact-match fields (any deviation = divergence)
EXACT_FIELDS = {"date", "ticker", "action", "trigger_type", "predicted_direction", "days_held"}

# Per-day divergence thresholds (see docs/trade_mapping.md)
TRADE_COUNT_PCT_THRESHOLD = 0.05     # 5%
TICKER_SET_PER_DAY_MAX = 1           # >1 ticker differ per day fails
TICKER_SET_CUMULATIVE_PCT = 0.05     # OR >5% cumulative across window


# ── Pure diff helpers (unit-tested below) ───────────────────────────────────

def within_tolerance(live: float | None, replay: float | None, tol: Tolerance) -> bool:
    """Return True if `live` and `replay` agree within the tolerance."""
    if live is None and replay is None:
        return True
    if live is None or replay is None:
        return False
    diff = abs(float(live) - float(replay))
    if tol.abs_ > 0 and diff <= tol.abs_:
        return True
    if tol.rel > 0 and abs(float(live)) > 1e-9:
        if diff / abs(float(live)) <= tol.rel:
            return True
    # Both thresholds failed (or rel undefined and abs failed)
    return tol.rel == 0 and tol.abs_ == 0  # both-zero means require exact match (handled by exact-match branch)


def diff_trade_count(live_by_date: dict[str, int], replay_by_date: dict[str, int]) -> dict[str, dict]:
    """Per-day trade count divergence above TRADE_COUNT_PCT_THRESHOLD."""
    out: dict[str, dict] = {}
    all_dates = set(live_by_date) | set(replay_by_date)
    for d in sorted(all_dates):
        n_live = live_by_date.get(d, 0)
        n_replay = replay_by_date.get(d, 0)
        denom = max(n_live, 1)
        pct = abs(n_replay - n_live) / denom
        if pct > TRADE_COUNT_PCT_THRESHOLD:
            out[d] = {"live": n_live, "backtester": n_replay,
                      "diff": n_replay - n_live, "pct": round(pct, 4)}
    return out


def diff_ticker_sets(live_by_date: dict[str, set[str]],
                     replay_by_date: dict[str, set[str]]) -> dict[str, dict]:
    """Per-day ticker set symmetric difference above TICKER_SET_PER_DAY_MAX."""
    out: dict[str, dict] = {}
    all_dates = set(live_by_date) | set(replay_by_date)
    for d in sorted(all_dates):
        live_t = live_by_date.get(d, set())
        replay_t = replay_by_date.get(d, set())
        only_live = sorted(live_t - replay_t)
        only_replay = sorted(replay_t - live_t)
        if len(only_live) + len(only_replay) > TICKER_SET_PER_DAY_MAX:
            out[d] = {"only_live": only_live, "only_backtester": only_replay}
    return out


def diff_fields(live_trade: dict, replay_trade: dict) -> dict[str, dict]:
    """Per-field comparison for a single matched trade. Returns {field: {live, replay, ...}} for violations."""
    violations: dict[str, dict] = {}

    for field in EXACT_FIELDS:
        lv, rv = live_trade.get(field), replay_trade.get(field)
        if lv != rv:
            violations[field] = {"live": lv, "backtester": rv, "match_rule": "exact"}

    # Shares special case (integer, ±1 tolerance)
    lv_shares, rv_shares = live_trade.get("shares"), replay_trade.get("shares")
    if lv_shares is not None and rv_shares is not None:
        if abs(int(lv_shares) - int(rv_shares)) > SHARES_ABS_TOLERANCE:
            violations["shares"] = {"live": lv_shares, "backtester": rv_shares,
                                    "threshold_abs": SHARES_ABS_TOLERANCE}

    for field, tol in FIELD_TOLERANCES.items():
        lv, rv = live_trade.get(field), replay_trade.get(field)
        if not within_tolerance(lv, rv, tol):
            violations[field] = {"live": lv, "backtester": rv,
                                 "threshold_rel": tol.rel, "threshold_abs": tol.abs_}

    return violations


# ── Placeholder: backtester invocation wiring (Phase 1.1b) ──────────────────

def _run_backtester_for_dates(dates: list[str], bucket: str) -> list[dict]:
    """Replay the backtester for each date, return the order list.

    PLACEHOLDER — wiring lands in the next commit. Extracts the single-date
    call from backtest.py::_run_simulation_loop into a reusable helper that
    yields the order stream without side effects (no S3 report upload).
    """
    raise NotImplementedError(
        "Phase 1.1b wiring not yet landed — _run_backtester_for_dates is a "
        "scaffold. Extract backtest.py::_run_simulation_loop into a "
        "single-date-replay helper in a follow-up commit."
    )


# ── trades.db access ────────────────────────────────────────────────────────

def _load_trades_from_db(db_path: str, since_date: str | None = None) -> pd.DataFrame:
    """Read the `trades` table. Returns a DataFrame (empty if table missing)."""
    conn = sqlite3.connect(db_path)
    try:
        q = "SELECT * FROM trades"
        params: tuple = ()
        if since_date:
            q += " WHERE date >= ?"
            params = (since_date,)
        q += " ORDER BY date, ticker"
        return pd.read_sql_query(q, conn, params=params)
    except pd.errors.DatabaseError as exc:
        # Graceful: empty DataFrame if table missing (first boot)
        if "no such table" in str(exc).lower():
            return pd.DataFrame()
        raise
    finally:
        conn.close()


def _last_n_trading_dates(trades_df: pd.DataFrame, n: int) -> list[str]:
    if trades_df.empty:
        return []
    return sorted(trades_df["date"].dropna().unique())[-n:]


# ── Unit tests (pure diff logic) — always run ───────────────────────────────

class TestWithinTolerance:
    def test_both_none_matches(self):
        assert within_tolerance(None, None, Tolerance(rel=0.001, abs_=0.01))

    def test_one_none_fails(self):
        assert not within_tolerance(None, 100.0, Tolerance(rel=0.001))
        assert not within_tolerance(100.0, None, Tolerance(rel=0.001))

    def test_within_rel_passes(self):
        # 0.08% delta under 0.1% threshold
        assert within_tolerance(100.0, 100.08, Tolerance(rel=0.001))

    def test_within_abs_passes(self):
        # $0.005 delta under $0.01 threshold
        assert within_tolerance(100.000, 100.005, Tolerance(rel=0.0, abs_=0.01))

    def test_outside_rel_and_abs_fails(self):
        # 0.5% delta, exceeds both
        assert not within_tolerance(100.0, 100.5, Tolerance(rel=0.001, abs_=0.01))

    def test_rel_with_abs_fallback_wins(self):
        # Very small absolute OK even though relative exceeds
        assert within_tolerance(100.0, 100.005, Tolerance(rel=0.00001, abs_=0.01))


class TestDiffTradeCount:
    def test_below_threshold_returns_empty(self):
        # 4% diff, under 5% threshold
        live = {"2026-04-10": 100}
        replay = {"2026-04-10": 96}
        assert diff_trade_count(live, replay) == {}

    def test_above_threshold_reported(self):
        # 10% diff
        live = {"2026-04-10": 100}
        replay = {"2026-04-10": 90}
        out = diff_trade_count(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["diff"] == -10

    def test_date_only_on_one_side(self):
        # 100% divergence — always exceeds threshold
        live = {"2026-04-10": 5}
        replay = {}
        out = diff_trade_count(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["backtester"] == 0


class TestDiffTickerSets:
    def test_zero_diff_returns_empty(self):
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "MSFT"}}
        assert diff_ticker_sets(live, replay) == {}

    def test_single_diff_under_threshold(self):
        # 1 ticker differ — at threshold, not exceeding
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "MSFT", "NVDA"}}
        assert diff_ticker_sets(live, replay) == {}

    def test_two_diffs_reported(self):
        live = {"2026-04-10": {"AAPL", "MSFT"}}
        replay = {"2026-04-10": {"AAPL", "NVDA", "PLTR"}}
        out = diff_ticker_sets(live, replay)
        assert "2026-04-10" in out
        assert out["2026-04-10"]["only_live"] == ["MSFT"]
        assert out["2026-04-10"]["only_backtester"] == ["NVDA", "PLTR"]


class TestDiffFields:
    def _base_trade(self, **kwargs):
        base = {
            "date": "2026-04-10", "ticker": "AAPL", "action": "ENTER",
            "shares": 100, "fill_price": 172.34, "price_at_order": 172.30,
            "trigger_type": "pullback", "trigger_price": 172.00,
            "signal_price": 172.50, "research_score": 78.0,
            "predicted_direction": "UP", "prediction_confidence": 0.65,
            "position_pct": 0.05, "realized_return_pct": None, "days_held": None,
        }
        base.update(kwargs)
        return base

    def test_identical_trades_no_violations(self):
        a = self._base_trade()
        b = self._base_trade()
        assert diff_fields(a, b) == {}

    def test_fill_price_within_rel(self):
        a = self._base_trade(fill_price=172.34)
        b = self._base_trade(fill_price=172.42)  # 0.046% — under 0.1%
        assert diff_fields(a, b) == {}

    def test_fill_price_outside_rel(self):
        a = self._base_trade(fill_price=172.34)
        b = self._base_trade(fill_price=173.00)  # 0.38% — exceeds 0.1%
        v = diff_fields(a, b)
        assert "fill_price" in v

    def test_action_exact_mismatch(self):
        a = self._base_trade(action="ENTER")
        b = self._base_trade(action="EXIT")
        v = diff_fields(a, b)
        assert "action" in v and v["action"]["match_rule"] == "exact"

    def test_shares_within_one(self):
        a = self._base_trade(shares=100)
        b = self._base_trade(shares=101)
        assert diff_fields(a, b) == {}

    def test_shares_beyond_one(self):
        a = self._base_trade(shares=100)
        b = self._base_trade(shares=102)
        v = diff_fields(a, b)
        assert "shares" in v

    def test_trigger_type_null_vs_value(self):
        a = self._base_trade(trigger_type=None)
        b = self._base_trade(trigger_type="vwap")
        v = diff_fields(a, b)
        assert "trigger_type" in v


class TestLoadTradesFromDB:
    def _write_trades(self, path, rows):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (date TEXT, ticker TEXT, action TEXT, shares INTEGER, fill_price REAL)")
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?)", rows)
        conn.commit()
        conn.close()

    def test_empty_table(self, tmp_path):
        db = tmp_path / "trades.db"
        self._write_trades(str(db), [])
        df = _load_trades_from_db(str(db))
        assert df.empty

    def test_missing_table_returns_empty(self, tmp_path):
        db = tmp_path / "empty.db"
        sqlite3.connect(str(db)).close()
        df = _load_trades_from_db(str(db))
        assert df.empty

    def test_since_date_filter(self, tmp_path):
        db = tmp_path / "trades.db"
        self._write_trades(str(db), [
            ("2026-04-01", "AAPL", "ENTER", 100, 170.0),
            ("2026-04-10", "MSFT", "ENTER", 50, 400.0),
        ])
        df = _load_trades_from_db(str(db), since_date="2026-04-05")
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "MSFT"


class TestLastNTradingDates:
    def test_empty(self):
        assert _last_n_trading_dates(pd.DataFrame(), 5) == []

    def test_takes_latest_n(self):
        df = pd.DataFrame({"date": ["2026-04-01", "2026-04-02", "2026-04-03",
                                     "2026-04-04", "2026-04-05"]})
        assert _last_n_trading_dates(df, 3) == ["2026-04-03", "2026-04-04", "2026-04-05"]


# ── Integration test (opt-in) ───────────────────────────────────────────────

@pytest.mark.parity
def test_parity_replay_end_to_end():
    """Full parity test — replays backtester over last N live trade dates.

    Opt-in via `pytest -m parity`. Requires:
      * trades.db reachable (TRADES_DB_PATH or S3 download)
      * ArcticDB live (SIGNALS_BUCKET in AWS)
      * Backtester-invocation helper wired (Phase 1.1b)
    """
    bucket = os.environ.get("SIGNALS_BUCKET", "alpha-engine-research")
    window_days = int(os.environ.get("PARITY_WINDOW_DAYS", "10"))

    # Resolve trades.db
    db_path = os.environ.get("TRADES_DB_PATH")
    if not db_path:
        pytest.skip("TRADES_DB_PATH not set — S3 download not yet wired (Phase 1.1b)")

    if not Path(db_path).exists():
        pytest.skip(f"trades.db not found at {db_path}")

    trades_df = _load_trades_from_db(db_path)
    if trades_df.empty:
        pytest.skip("trades.db is empty — no history to parity-check against")

    dates = _last_n_trading_dates(trades_df, window_days)
    if len(dates) < 3:
        pytest.skip(f"Need >=3 dates with trades; have {len(dates)}")

    # Run backtester (placeholder — raises NotImplementedError until 1.1b)
    replay_orders = _run_backtester_for_dates(dates, bucket)

    # Group both sides by date
    window = trades_df[trades_df["date"].isin(dates)]
    live_by_date: dict[str, int] = window.groupby("date").size().to_dict()
    replay_by_date_count: dict[str, int] = {}
    replay_tickers_by_date: dict[str, set[str]] = {}
    for o in replay_orders:
        d = o["date"]
        replay_by_date_count[d] = replay_by_date_count.get(d, 0) + 1
        replay_tickers_by_date.setdefault(d, set()).add(o["ticker"])

    live_tickers_by_date = {
        d: set(window[window["date"] == d]["ticker"].tolist()) for d in dates
    }

    count_violations = diff_trade_count(live_by_date, replay_by_date_count)
    ticker_violations = diff_ticker_sets(live_tickers_by_date, replay_tickers_by_date)

    # Field-level diffs on matched trades
    field_violations: list[dict] = []
    for _, row in window.iterrows():
        key = (row["date"], row["ticker"], row["action"])
        match = next((o for o in replay_orders
                      if (o["date"], o["ticker"], o["action"]) == key), None)
        if match is None:
            continue
        vs = diff_fields(row.to_dict(), match)
        if vs:
            field_violations.append({"date": row["date"], "ticker": row["ticker"],
                                     "action": row["action"], "fields": vs})

    report = {
        "status": "fail" if (count_violations or ticker_violations or field_violations) else "pass",
        "window": [dates[0], dates[-1]],
        "n_live_trades": int(len(window)),
        "n_backtester_trades": len(replay_orders),
        "trade_count_divergence": count_violations,
        "ticker_set_divergence": ticker_violations,
        "field_divergence": field_violations,
    }

    # Write report for the spot-instance post-run hook
    report_dir = Path(os.environ.get("PARITY_REPORT_DIR", tempfile.gettempdir()))
    report_dir.mkdir(parents=True, exist_ok=True)
    import json
    (report_dir / "parity_report.json").write_text(json.dumps(report, indent=2, default=str))

    assert report["status"] == "pass", (
        f"Parity divergence: {len(count_violations)} trade-count, "
        f"{len(ticker_violations)} ticker-set, {len(field_violations)} field-level. "
        f"See {report_dir / 'parity_report.json'}"
    )
