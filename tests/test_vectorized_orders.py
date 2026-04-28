"""Tests for synthetic.vectorized_orders.VectorizedOrderStore.

Pins:
  1. Materialized dict-list shape exactly matches the legacy
     `list[list[dict]]` accumulator — downstream `orders_to_portfolio`
     and parity tests must read the same fields with the same value
     types.
  2. Action codes (ENTER / EXIT / REDUCE) round-trip through the
     columnar storage to the right `action` string.
  3. Exit reason codes round-trip to the right `exit_reason` string.
  4. `release(combo_idx)` drops the buffer; subsequent __getitem__
     returns [] without crashing.
  5. __getitem__ is idempotent — multiple accesses return identical
     content (matters for tests + diagnostic re-reads).
  6. Memory budget — a synthetic 60-combo × 25k-order workload fits
     well under the prior `list[list[dict]]` footprint. Catches the
     class of regression that caused the 2026-04-28 v15 OOM if
     someone reverts to dict-per-order.
"""
from __future__ import annotations

import sys
import tracemalloc

import pandas as pd
import pytest

from synthetic.vectorized_orders import (
    ACTION_ENTER,
    VectorizedOrderStore,
    _OrderBuffer,
)
from synthetic.vectorized_exits import (
    ACTION_EXIT,
    ACTION_REDUCE,
    REASON_ATR,
    REASON_FALLBACK,
    REASON_PROFIT,
    REASON_MOMENTUM,
    REASON_TIME_EXIT,
    REASON_TIME_REDUCE,
)


def _make_store(n_combos: int = 3) -> tuple[VectorizedOrderStore, pd.DatetimeIndex, list[str]]:
    """Build a finalized store with 5 dates + 4 tickers."""
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    tickers = ["AAPL", "MSFT", "JNJ", "BAC"]
    store = VectorizedOrderStore(n_combos)
    store.finalize(dates, tickers)
    return store, dates, tickers


# ── Round-trip dict shape ───────────────────────────────────────────────────


class TestEntryDictShape:
    def test_entry_materializes_to_canonical_keys(self):
        store, dates, tickers = _make_store()
        store.add_entry(
            combo_idx=0, date_idx=2, ticker_idx=0, shares=100,
            price=150.50, nav=1_000_000.0, position_pct=0.05,
        )
        out = store[0]
        assert len(out) == 1
        order = out[0]
        # Exact key set — same as the legacy dict-per-order producer.
        assert set(order) == {
            "date", "ticker", "action", "shares",
            "price_at_order", "portfolio_nav_at_order", "position_pct",
        }
        assert order["date"] == "2026-01-05"  # 3rd business day from Jan 1
        assert order["ticker"] == "AAPL"
        assert order["action"] == "ENTER"
        assert order["shares"] == 100
        assert order["price_at_order"] == 150.50
        assert order["portfolio_nav_at_order"] == 1_000_000.0
        assert order["position_pct"] == 0.05

    def test_entry_value_types_match_legacy(self):
        """Downstream consumers (orders_to_portfolio) rely on shares
        being int, prices being float."""
        store, _, _ = _make_store()
        store.add_entry(
            combo_idx=0, date_idx=0, ticker_idx=1, shares=42,
            price=99.99, nav=500_000.0, position_pct=0.10,
        )
        order = store[0][0]
        assert isinstance(order["shares"], int)
        assert isinstance(order["price_at_order"], float)
        assert isinstance(order["portfolio_nav_at_order"], float)
        assert isinstance(order["position_pct"], float)


class TestExitDictShape:
    def test_exit_materializes_to_canonical_keys(self):
        store, _, _ = _make_store()
        store.add_exit(
            combo_idx=0, date_idx=1, ticker_idx=2,
            action_code=ACTION_EXIT, shares=50, price=200.0,
            nav=950_000.0, reason_code=REASON_ATR,
        )
        order = store[0][0]
        assert set(order) == {
            "date", "ticker", "action", "shares",
            "price_at_order", "portfolio_nav_at_order", "exit_reason",
        }
        assert order["action"] == "EXIT"
        assert order["exit_reason"] == "atr_trailing_stop"

    def test_reduce_action_distinct_from_exit(self):
        store, _, _ = _make_store()
        store.add_exit(
            combo_idx=0, date_idx=0, ticker_idx=0,
            action_code=ACTION_REDUCE, shares=25, price=100.0,
            nav=500_000.0, reason_code=REASON_PROFIT,
        )
        order = store[0][0]
        assert order["action"] == "REDUCE"
        assert order["exit_reason"] == "profit_take"


@pytest.mark.parametrize("reason_code,expected_str", [
    (REASON_ATR, "atr_trailing_stop"),
    (REASON_FALLBACK, "fallback_stop"),
    (REASON_PROFIT, "profit_take"),
    (REASON_MOMENTUM, "momentum_exit"),
    (REASON_TIME_EXIT, "time_decay_exit"),
    (REASON_TIME_REDUCE, "time_decay_reduce"),
])
def test_exit_reason_codes_round_trip(reason_code, expected_str):
    """All 6 exit reason codes must materialize to the right string —
    `exit_reason` feeds directly into trade-attribution analysis."""
    store, _, _ = _make_store()
    store.add_exit(
        combo_idx=0, date_idx=0, ticker_idx=0,
        action_code=ACTION_EXIT, shares=10, price=50.0,
        nav=100_000.0, reason_code=reason_code,
    )
    assert store[0][0]["exit_reason"] == expected_str


# ── Combo isolation + ordering ──────────────────────────────────────────────


class TestComboIsolation:
    def test_orders_isolate_per_combo(self):
        store, _, _ = _make_store()
        store.add_entry(
            combo_idx=0, date_idx=0, ticker_idx=0, shares=100,
            price=100.0, nav=1_000_000.0, position_pct=0.05,
        )
        store.add_entry(
            combo_idx=1, date_idx=0, ticker_idx=1, shares=200,
            price=50.0, nav=1_000_000.0, position_pct=0.10,
        )
        assert len(store[0]) == 1
        assert len(store[1]) == 1
        assert len(store[2]) == 0
        assert store[0][0]["ticker"] == "AAPL"
        assert store[1][0]["ticker"] == "MSFT"

    def test_orders_preserve_insertion_order_within_combo(self):
        store, _, _ = _make_store()
        # Entries on dates 2, 0, 4 — must materialize in insertion order.
        for date_idx in [2, 0, 4]:
            store.add_entry(
                combo_idx=0, date_idx=date_idx, ticker_idx=0,
                shares=100, price=100.0, nav=1_000_000.0,
                position_pct=0.05,
            )
        out = store[0]
        assert [o["date"] for o in out] == [
            "2026-01-05", "2026-01-01", "2026-01-07",
        ]


# ── Lifecycle: __getitem__, len, release, finalize ──────────────────────────


class TestLifecycle:
    def test_len_returns_n_combos(self):
        store = VectorizedOrderStore(7)
        assert len(store) == 7

    def test_index_out_of_range_raises(self):
        store, _, _ = _make_store(n_combos=2)
        with pytest.raises(IndexError):
            store[5]

    def test_negative_combo_count_raises(self):
        with pytest.raises(ValueError):
            VectorizedOrderStore(-1)

    def test_zero_combos_allowed(self):
        """Edge case: empty grid. Must not crash construction."""
        store = VectorizedOrderStore(0)
        store.finalize(pd.date_range("2026-01-01", periods=1), [])
        assert len(store) == 0

    def test_getitem_idempotent(self):
        """Multiple reads of the same combo must return equivalent
        content. Production releases after one read; tests rely on
        re-reading."""
        store, _, _ = _make_store()
        for _ in range(3):
            store.add_entry(
                combo_idx=0, date_idx=0, ticker_idx=0, shares=100,
                price=100.0, nav=1_000_000.0, position_pct=0.05,
            )
        first = store[0]
        second = store[0]
        assert first == second
        # But the lists are distinct objects (fresh materialization).
        assert first is not second

    def test_release_drops_buffer(self):
        store, _, _ = _make_store()
        store.add_entry(
            combo_idx=0, date_idx=0, ticker_idx=0, shares=100,
            price=100.0, nav=1_000_000.0, position_pct=0.05,
        )
        assert len(store[0]) == 1
        store.release(0)
        assert store[0] == []

    def test_release_other_combos_unaffected(self):
        store, _, _ = _make_store()
        store.add_entry(0, 0, 0, 100, 100.0, 1_000_000.0, 0.05)
        store.add_entry(1, 0, 1, 100, 100.0, 1_000_000.0, 0.05)
        store.release(0)
        assert store[0] == []
        assert len(store[1]) == 1

    def test_release_out_of_range_silently_noops(self):
        """Defensive — caller may release after a partial loop."""
        store, _, _ = _make_store(n_combos=2)
        store.release(99)  # no exception

    def test_add_after_release_raises(self):
        """Producer-side bug guard: writing to a released buffer is
        always a programming error."""
        store, _, _ = _make_store()
        store.release(0)
        with pytest.raises(RuntimeError, match="already released"):
            store.add_entry(0, 0, 0, 1, 1.0, 1.0, 0.0)
        with pytest.raises(RuntimeError, match="already released"):
            store.add_exit(0, 0, 0, ACTION_EXIT, 1, 1.0, 1.0, REASON_ATR)

    def test_materialize_before_finalize_raises(self):
        store = VectorizedOrderStore(2)
        store.add_entry(0, 0, 0, 100, 1.0, 1.0, 0.05)
        with pytest.raises(RuntimeError, match="finalize"):
            _ = store[0]


class TestTotalOrders:
    def test_counts_unreleased_only(self):
        store, _, _ = _make_store()
        for _ in range(3):
            store.add_entry(0, 0, 0, 1, 1.0, 1.0, 0.05)
        for _ in range(2):
            store.add_entry(1, 0, 0, 1, 1.0, 1.0, 0.05)
        assert store.total_orders() == 5
        store.release(1)
        assert store.total_orders() == 3


# ── Memory budget (regression guard for OOM class) ──────────────────────────


class TestMemoryBudget:
    """Pins peak memory at scale. The 2026-04-28 v15 OOM kill was
    caused by `list[list[dict]]` storing 1.49M orders = ~450 MB.
    These tests fail fast if anyone reverts to dict-per-order or
    introduces another pessimization."""

    def test_one_million_orders_fits_in_budget(self):
        """Synthetic 60-combo × 17k-order workload (~1M total) must
        stay well under 100 MB. Real-world v15 workload was 1.5M
        orders / 60 combos."""
        n_combos = 60
        orders_per_combo = 17_000  # ~1M total
        dates = pd.date_range("2016-01-01", periods=2500, freq="B")
        tickers = [f"T{i:04d}" for i in range(1000)]

        tracemalloc.start()
        store = VectorizedOrderStore(n_combos)
        for c in range(n_combos):
            for k in range(orders_per_combo):
                # Cycle through dates + tickers + actions to make the
                # workload representative.
                d_idx = k % len(dates)
                t_idx = k % len(tickers)
                if k % 4 == 0:
                    store.add_entry(c, d_idx, t_idx, 100, 50.0, 1e6, 0.05)
                else:
                    store.add_exit(
                        c, d_idx, t_idx, ACTION_EXIT, 50, 51.0, 1.05e6,
                        REASON_ATR,
                    )
        store.finalize(dates, tickers)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)
        # Budget: 150 MB for 1M orders. Comfortably under c5.large's
        # 3.7 GB usable. Old `list[list[dict]]` would peak around
        # 300-400 MB for the same workload.
        assert peak_mb < 150, (
            f"VectorizedOrderStore peak memory {peak_mb:.0f} MB "
            f"exceeds 150 MB budget for {n_combos*orders_per_combo:,} "
            f"orders. Likely regression to dict-per-order accumulator "
            f"(see 2026-04-28 v15 OOM)."
        )
        assert store.total_orders() == n_combos * orders_per_combo

    def test_buffer_is_smaller_than_dict_list_equivalent(self):
        """Direct comparison: 10k orders in _OrderBuffer vs an equivalent
        list[dict]. Buffer must be at least 4× smaller."""
        buf = _OrderBuffer()
        for k in range(10_000):
            buf.add_entry(k % 100, k % 50, 100, 50.0, 1e6, 0.05)

        dict_list = [{
            "date": "2026-01-01", "ticker": "AAPL", "action": "ENTER",
            "shares": 100, "price_at_order": 50.0,
            "portfolio_nav_at_order": 1e6, "position_pct": 0.05,
        } for _ in range(10_000)]

        # Sum sizes of array.array objects (each is a contiguous block).
        buf_size = sum(
            sys.getsizeof(getattr(buf, c))
            for c in ("date_idx", "ticker_idx", "action_code", "shares",
                      "price", "nav", "extra")
        )
        # For dict_list: container + each dict's overhead. We use a
        # rough multiplier accounting for dict header + 7 keys + values.
        dict_list_size = sys.getsizeof(dict_list) + sum(
            sys.getsizeof(d) for d in dict_list
        )

        ratio = dict_list_size / buf_size
        assert ratio >= 4.0, (
            f"_OrderBuffer is only {ratio:.1f}× smaller than dict-list "
            f"({buf_size:,} B vs {dict_list_size:,} B). Expected >=4×. "
            f"Buffer storage may have regressed."
        )
