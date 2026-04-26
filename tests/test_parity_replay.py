"""
tests/test_parity_replay.py — replay parity test (Phase 1.1).

Diffs backtester output against the live trades.db over a historical window.
See docs/trade_mapping.md for the field mapping + tolerance contract.

Opt-in via the `parity` pytest marker so CI on feature branches doesn't
try to reach S3. Spot instance runs it explicitly via spot_backtest.sh
after the weekly backtest completes.

Status: WIRING LANDED (2026-04-16, Phase 1.1b).
  * Diff logic (pure functions) is complete and unit-tested below.
  * `_run_backtester_for_dates()` delegates to `backtest.replay_for_dates()`
    which exercises the live executor (`simulate=True` path) for each date.
  * Remaining: Phase 1.4 — invoke this test from infrastructure/spot_backtest.sh
    post-backtest; upload parity_report.json to S3; email on divergence.

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

# arcticdb stubbing lives in tests/conftest.py — unit tests get a MagicMock
# by default, integration tests (this file's @pytest.mark.parity case) opt
# in to the real module by setting USE_REAL_ARCTICDB=1 before pytest starts
# (spot_backtest.sh's parity stage does this).


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
}

# Shares have rounding at fill time; allow ±1 share
SHARES_ABS_TOLERANCE = 1

# Exact-match fields (any deviation = divergence)
EXACT_FIELDS = {"ticker", "action", "trigger_type", "predicted_direction"}

# Lifecycle fields populated post-trade by the live executor (forward-looking
# from the trade's perspective) that the backtester sim cannot reproduce —
# its replay is a single-shot decision per signal_date with no time advance.
# These are skipped in diff_fields rather than treated as divergence. See
# DATE_CONVENTIONS.md migration notes + ROADMAP "Backtester ↔ executor parity
# divergence" P1.
LIFECYCLE_SKIP_FIELDS = frozenset({
    "days_held",
    "realized_return_pct",
    "realized_pnl",
    "realized_alpha_pct",
    "spy_return_during_hold",
    "fill_time",
    "created_at",
    "ib_order_id",
    "slippage_vs_signal",
    "execution_latency_ms",
    # The legacy `date` column is the calendar fill date (live) vs signal_date
    # (backtester) — they semantically differ until the broader date-convention
    # migration is fully rolled out. Cohort matching uses signal_trading_day,
    # so the per-row `date` field check would always trip on the legacy column.
    "date",
    # Daemon-stage fields. Live populates these when the intraday daemon's
    # trigger (VWAP / pullback / support / time-expiry / graduated_entry)
    # fires and IB returns a fill: trigger_type names the rule that fired,
    # trigger_price is the gate, signal_price is the daemon's snapshot at
    # decision time, fill_price is what IB filled at. Backtester sim runs
    # the morning planner only — none of these exist at planner stage,
    # so they're null in replay output. Including them as required-match
    # fields produced spurious divergence on every cohort-matched ENTER
    # (observed 2026-04-26 ROST 4/12 trade). Phase 2 (entry_triggers.py
    # daily-bar port) will eventually populate them in sim, but until
    # then they're a pure noise floor. ROADMAP P1 "Backtester ↔ executor
    # parity divergence" root cause #3.
    "trigger_type",
    "trigger_price",
    "signal_price",
    "fill_price",
    # Live writes ``prediction_confidence=NaN`` when GBM coverage is
    # missing for the ticker; backtester serializes the same absent
    # value as ``null``. Float comparison treats NaN != null even though
    # they encode the same "no prediction" semantics. Skip until the
    # serialization gap is closed at the writer side.
    "prediction_confidence",
})

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
    """Per-field comparison for a single matched trade. Returns
    ``{field: {live, replay, ...}}`` for violations.

    Fields in ``LIFECYCLE_SKIP_FIELDS`` are excluded from the comparison —
    they're populated post-trade by the live executor (e.g. ``days_held``,
    ``realized_return_pct``) and the backtester sim cannot reproduce them.
    Including them would generate noise on every matched ENTER trade.
    """
    violations: dict[str, dict] = {}

    for field in EXACT_FIELDS:
        if field in LIFECYCLE_SKIP_FIELDS:
            continue
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
        if field in LIFECYCLE_SKIP_FIELDS:
            continue
        lv, rv = live_trade.get(field), replay_trade.get(field)
        if not within_tolerance(lv, rv, tol):
            violations[field] = {"live": lv, "backtester": rv,
                                 "threshold_rel": tol.rel, "threshold_abs": tol.abs_}

    return violations


# ── Backtester invocation (Phase 1.1b) ──────────────────────────────────────

def _run_backtester_for_dates(dates: list[str], bucket: str,
                              config_path: str | None = None,
                              trades_db_path: str | None = None) -> list[dict]:
    """Replay the backtester for each date, return the aggregated order list.

    Thin wrapper over ``backtest.replay_for_dates`` — loads config, overrides
    the signals_bucket for the caller's convenience, delegates to the helper
    that factors the per-date orchestration out of ``_run_simulation_loop``.

    Requires ``executor_paths`` in config.yaml to point to a live
    ``alpha-engine`` checkout — the backtester imports the executor directly
    rather than reimplementing it, so ``simulate=True`` actually exercises
    live executor code.

    ``trades_db_path``: when provided, sim bootstraps initial positions/cash
    from ``eod_pnl``'s most recent snapshot strictly before the parity
    window. Replaces the cold-start warmup (Option A long-term parity
    strategy — see ``_load_initial_state_from_eod_pnl`` docstring in
    backtest.py).
    """
    from pipeline_common import load_config
    import backtest as _bt

    cfg_path = config_path or os.environ.get("BACKTESTER_CONFIG", "config.yaml")
    config = load_config(cfg_path)
    if bucket:
        config["signals_bucket"] = bucket
    if trades_db_path:
        config["trades_db_path"] = trades_db_path

    return _bt.replay_for_dates(sorted(dates), config)


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
    """Return the last n unique signal_trading_day values from the trades
    DataFrame — the parity cohort key per DATE_CONVENTIONS.md.

    Falls back to the legacy ``date`` column when ``signal_trading_day`` is
    missing or empty (pre-migration DBs). Operators running the parity test
    against a pre-PR-2 trades.db will see zero matchable rows downstream and
    the integration test will skip with a clear message.
    """
    if trades_df.empty:
        return []
    if "signal_trading_day" in trades_df.columns:
        col_values = trades_df["signal_trading_day"].dropna()
        if not col_values.empty:
            return sorted(col_values.unique())[-n:]
    # Fallback for pre-migration DBs — legacy `date` column. Test will
    # filter to ENTERs with non-null signal_trading_day downstream and skip
    # if nothing matches.
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

    def test_price_at_order_within_rel(self):
        # price_at_order is the planner-stage price both sides should have;
        # fill_price moved to LIFECYCLE_SKIP_FIELDS so it's no longer the
        # canonical "rel-tolerance compared field" in the test surface.
        a = self._base_trade(price_at_order=172.30)
        b = self._base_trade(price_at_order=172.39)  # 0.052% — under 0.1%
        assert diff_fields(a, b) == {}

    def test_price_at_order_outside_rel(self):
        a = self._base_trade(price_at_order=172.30)
        b = self._base_trade(price_at_order=173.00)  # 0.41% — exceeds 0.1%
        v = diff_fields(a, b)
        assert "price_at_order" in v

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

    def test_predicted_direction_exact_mismatch(self):
        # Replaces the prior test_trigger_type_null_vs_value — trigger_type
        # is now in LIFECYCLE_SKIP_FIELDS. predicted_direction stays in
        # EXACT_FIELDS as the remaining backtester-vs-live exact-match
        # field that's not skipped, so it stands in for the same code path.
        a = self._base_trade(predicted_direction="UP")
        b = self._base_trade(predicted_direction="DOWN")
        v = diff_fields(a, b)
        assert "predicted_direction" in v and v["predicted_direction"]["match_rule"] == "exact"


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

    def test_prefers_signal_trading_day_when_present(self):
        """Post-PR-2 DBs have signal_trading_day; cohort matching uses it."""
        df = pd.DataFrame({
            "date": ["2026-04-13", "2026-04-14", "2026-04-15", "2026-04-16", "2026-04-17"],
            "signal_trading_day": ["2026-04-10", "2026-04-10", "2026-04-10", "2026-04-17", "2026-04-17"],
        })
        # Should return unique signal_trading_days, not unique fill dates.
        result = _last_n_trading_dates(df, 5)
        assert result == ["2026-04-10", "2026-04-17"]

    def test_falls_back_to_date_when_signal_trading_day_all_null(self):
        """Pre-backfill DB might have the column but with all NULLs.
        Falls back to the legacy `date` column so the test can still
        function (and the integration path skips later if no matchable
        rows remain after the ENTER + non-null filter)."""
        df = pd.DataFrame({
            "date": ["2026-04-13", "2026-04-14", "2026-04-15"],
            "signal_trading_day": [None, None, None],
        })
        assert _last_n_trading_dates(df, 5) == ["2026-04-13", "2026-04-14", "2026-04-15"]


class TestLifecycleSkipFields:
    """Lifecycle fields populated post-trade by the live executor are
    excluded from diff comparison — they can never match backtester sim."""

    def _trade(self, **kwargs):
        base = {"date": "2026-04-13", "ticker": "AAPL", "action": "ENTER", "shares": 100}
        base.update(kwargs)
        return base

    def test_days_held_difference_not_flagged(self):
        live = self._trade(days_held=5)
        replay = self._trade(days_held=None)
        assert "days_held" not in diff_fields(live, replay)

    def test_realized_return_pct_difference_not_flagged(self):
        live = self._trade(realized_return_pct=2.5)
        replay = self._trade(realized_return_pct=None)
        assert "realized_return_pct" not in diff_fields(live, replay)

    def test_fill_time_difference_not_flagged(self):
        live = self._trade(fill_time="2026-04-13T14:30:00+00:00")
        replay = self._trade(fill_time=None)
        assert "fill_time" not in diff_fields(live, replay)

    def test_legacy_date_difference_not_flagged(self):
        # Live's `date` is the calendar fill day; backtester's is signal_date.
        # Cohort matching uses signal_trading_day, so the per-row `date`
        # column is irrelevant to the diff and is in LIFECYCLE_SKIP_FIELDS.
        live = self._trade(date="2026-04-20")
        replay = self._trade(date="2026-04-17")
        assert "date" not in diff_fields(live, replay)

    def test_non_lifecycle_field_still_compared(self):
        # ticker is exact-match; this confirms the skip-set doesn't
        # accidentally swallow real comparisons.
        live = self._trade(ticker="AAPL")
        replay = self._trade(ticker="MSFT")
        v = diff_fields(live, replay)
        assert "ticker" in v

    def test_trigger_type_difference_not_flagged(self):
        # Daemon-stage field — backtester sim cannot produce a trigger_type
        # at planner stage. ROST 4/12 incident 2026-04-26.
        live = self._trade(trigger_type="graduated_entry (+0.0% vs morning)")
        replay = self._trade(trigger_type=None)
        assert "trigger_type" not in diff_fields(live, replay)

    def test_trigger_price_difference_not_flagged(self):
        live = self._trade(trigger_price=220.59)
        replay = self._trade(trigger_price=None)
        assert "trigger_price" not in diff_fields(live, replay)

    def test_signal_price_difference_not_flagged(self):
        live = self._trade(signal_price=220.56)
        replay = self._trade(signal_price=None)
        assert "signal_price" not in diff_fields(live, replay)

    def test_fill_price_difference_not_flagged(self):
        # Live fill_price comes from IB at fill time; backtester sim has
        # no fill stage. Skip — same posture as the other daemon-stage
        # fields; replaces the prior FIELD_TOLERANCES entry.
        live = self._trade(fill_price=220.53)
        replay = self._trade(fill_price=None)
        assert "fill_price" not in diff_fields(live, replay)

    def test_prediction_confidence_nan_vs_null_not_flagged(self):
        # Live writes NaN when GBM coverage missing; backtester serializes
        # the same absent value as None. Float-compare treats them as
        # different even though the semantics match.
        live = self._trade(prediction_confidence=float("nan"))
        replay = self._trade(prediction_confidence=None)
        assert "prediction_confidence" not in diff_fields(live, replay)


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

    if "signal_trading_day" not in trades_df.columns:
        pytest.skip(
            "trades.db missing `signal_trading_day` column — run "
            "alpha-engine PR 2 migration (init_db on next executor start "
            "applies the schema; then run scripts/backfill_trading_day.py)."
        )

    # Cohort matching: filter live trades to ENTERs with a populated
    # signal_trading_day. Exits leave signal_trading_day NULL by design (PR 2);
    # pre-backfill rows would also be NULL. Those rows are excluded from
    # cohort-level comparison — they're not signal-driven decisions and
    # shouldn't be expected to round-trip against backtester output.
    matchable = trades_df[
        (trades_df["action"] == "ENTER")
        & trades_df["signal_trading_day"].notna()
    ]
    n_excluded = len(trades_df) - len(matchable)

    if matchable.empty:
        pytest.skip(
            f"trades.db has {len(trades_df)} rows but 0 are matchable ENTER "
            f"+ signal_trading_day populated rows. Run "
            f"scripts/backfill_trading_day.py to populate the column on "
            f"historical rows."
        )

    dates = _last_n_trading_dates(matchable, window_days)
    if len(dates) < 3:
        pytest.skip(
            f"Need >=3 signal_trading_days with matchable trades; have "
            f"{len(dates)} (matchable rows: {len(matchable)})"
        )

    # Run backtester replay over the same signal_trading_day cohorts.
    # backtest.replay_for_dates iterates the input dates as signal_dates,
    # producing orders tagged with `o["date"] = signal_date` — which equals
    # signal_trading_day on the live side post-backfill. The cohort key
    # matches across both sides without further translation.
    replay_orders = _run_backtester_for_dates(dates, bucket, trades_db_path=db_path)

    # Filter both sides to ENTERs for cohort matching. Exits don't have
    # signal_trading_day on the live side (NULL by design), so cohort
    # comparison is ENTER-only. Backtester also produces exits but those
    # are tagged with the same signal_date and could be cohort-matched
    # in a future expansion.
    matchable_enters = matchable
    replay_enters = [o for o in replay_orders if o.get("action") == "ENTER"]

    window = matchable_enters[matchable_enters["signal_trading_day"].isin(dates)]
    live_by_date: dict[str, int] = window.groupby("signal_trading_day").size().to_dict()
    replay_by_date_count: dict[str, int] = {}
    replay_tickers_by_date: dict[str, set[str]] = {}
    for o in replay_enters:
        d = o["date"]
        replay_by_date_count[d] = replay_by_date_count.get(d, 0) + 1
        replay_tickers_by_date.setdefault(d, set()).add(o["ticker"])

    live_tickers_by_date = {
        d: set(window[window["signal_trading_day"] == d]["ticker"].tolist())
        for d in dates
    }

    count_violations = diff_trade_count(live_by_date, replay_by_date_count)
    ticker_violations = diff_ticker_sets(live_tickers_by_date, replay_tickers_by_date)

    # Field-level diffs on matched trades — keyed on
    # (signal_trading_day, ticker, action). Lifecycle fields (days_held,
    # realized_return_pct, etc.) are excluded by LIFECYCLE_SKIP_FIELDS in
    # diff_fields — they're populated post-trade by the live executor and
    # the backtester sim cannot reproduce them.
    field_violations: list[dict] = []
    for _, row in window.iterrows():
        key = (row["signal_trading_day"], row["ticker"], row["action"])
        match = next(
            (
                o for o in replay_enters
                if (o.get("date"), o.get("ticker"), o.get("action")) == key
            ),
            None,
        )
        if match is None:
            continue
        vs = diff_fields(row.to_dict(), match)
        if vs:
            field_violations.append({
                "signal_trading_day": row["signal_trading_day"],
                "ticker": row["ticker"],
                "action": row["action"],
                "fields": vs,
            })

    report = {
        "status": "fail" if (count_violations or ticker_violations or field_violations) else "pass",
        "match_key": "(signal_trading_day, ticker, action)",
        "window_signal_trading_days": [dates[0], dates[-1]],
        "n_live_trades_total": int(len(trades_df)),
        "n_live_enters_matchable": int(len(matchable)),
        "n_live_excluded_no_signal_day": int(n_excluded),
        "n_backtester_orders_total": len(replay_orders),
        "n_backtester_enters": len(replay_enters),
        "lifecycle_fields_skipped": sorted(LIFECYCLE_SKIP_FIELDS),
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
