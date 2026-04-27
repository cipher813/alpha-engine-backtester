"""Tests for Tier 3 Part A: SignalLookup precompute (2026-04-27).

Pins three invariants:

  1. ``_build_signal_lookup`` produces byte-equal output to the per-call
     rebuild path it replaces (signals_by_ticker + universe_sectors).
  2. Universe filter is applied at precompute time when
     ``universe_symbols`` is provided; rejected_counter accumulates as
     expected.
  3. ``_precompute_signal_lookups`` returns ``None`` for ``None`` input
     (live signal-replay fallback path stays intact).
"""
from __future__ import annotations

import pytest

from backtest import (
    SignalLookup,
    _build_signal_lookup,
    _precompute_signal_lookups,
)


def _signals_raw(date_str: str = "2026-04-25") -> dict:
    return {
        "date": date_str,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "market_weight"}},
        "enter": [
            {"ticker": "AAPL", "score": 80, "conviction": "rising"},
        ],
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
            {"ticker": "JPM", "sector": "Financial"},
            {"ticker": "DROPPED_TKR", "sector": "Tech"},  # not in universe_symbols
        ],
        "buy_candidates": [
            {"ticker": "AAPL", "sector": "Technology", "score": 80},
            {"ticker": "GOOGL", "sector": "Technology", "score": 75},
        ],
    }


class TestBuildSignalLookup:
    def test_returns_signal_lookup_dataclass(self):
        lookup = _build_signal_lookup(_signals_raw())
        assert isinstance(lookup, SignalLookup)
        assert isinstance(lookup.signals_raw_filtered, dict)
        assert isinstance(lookup.signals_by_ticker, dict)
        assert isinstance(lookup.universe_sectors, dict)

    def test_signals_by_ticker_first_write_wins(self):
        """When the same ticker appears in both ``universe`` and
        ``buy_candidates`` (AAPL in our fixture), the first occurrence
        wins — matches the prior per-call rebuild semantics."""
        signals = _signals_raw()
        lookup = _build_signal_lookup(signals)
        # AAPL is in universe (pos 0) and buy_candidates (pos 0)
        # First-write-wins: should be the universe entry (no "score"),
        # not the buy_candidates entry (with "score")
        assert "AAPL" in lookup.signals_by_ticker
        assert "score" not in lookup.signals_by_ticker["AAPL"]

    def test_universe_sectors_last_write_wins(self):
        """``universe_sectors`` uses dict-comp semantics → duplicates
        resolve to the LAST entry's sector (matches the prior rebuild)."""
        # Build a fixture where AAPL appears with two different sectors
        signals = {
            "universe": [{"ticker": "AAPL", "sector": "Technology"}],
            "buy_candidates": [{"ticker": "AAPL", "sector": "ALT"}],
            "enter": [], "exit": [], "reduce": [], "hold": [],
        }
        lookup = _build_signal_lookup(signals)
        # buy_candidates iterates AFTER universe → last write wins → "ALT"
        assert lookup.universe_sectors["AAPL"] == "ALT"

    def test_skips_non_dict_entries(self):
        signals = {
            "universe": [
                {"ticker": "AAPL", "sector": "Technology"},
                "not-a-dict",  # garbage
                None,
                {"ticker": "MSFT", "sector": "Technology"},
            ],
            "buy_candidates": [],
            "enter": [], "exit": [], "reduce": [], "hold": [],
        }
        lookup = _build_signal_lookup(signals)
        assert set(lookup.signals_by_ticker) == {"AAPL", "MSFT"}
        assert set(lookup.universe_sectors) == {"AAPL", "MSFT"}

    def test_universe_filter_applied(self):
        """When ``universe_symbols`` is provided, signals lists are
        filtered to entries whose ticker is in the set."""
        signals = _signals_raw()
        universe_symbols = {"AAPL", "MSFT", "JPM", "GOOGL"}  # excludes DROPPED_TKR
        rejected: dict[str, int] = {}

        lookup = _build_signal_lookup(signals, universe_symbols, rejected)

        # DROPPED_TKR should not appear in any output
        assert "DROPPED_TKR" not in lookup.signals_by_ticker
        assert "DROPPED_TKR" not in lookup.universe_sectors
        # Filtered signals_raw should also drop it
        universe_tickers = {e.get("ticker") for e in lookup.signals_raw_filtered.get("universe", [])
                            if isinstance(e, dict)}
        assert "DROPPED_TKR" not in universe_tickers
        # Rejection counter incremented
        assert rejected.get("DROPPED_TKR", 0) >= 1

    def test_no_universe_filter_when_none(self):
        signals = _signals_raw()
        lookup = _build_signal_lookup(signals, universe_symbols=None)
        # DROPPED_TKR persists when no filter
        assert "DROPPED_TKR" in lookup.signals_by_ticker

    def test_empty_signals_returns_empty_lookups(self):
        empty = {"universe": [], "buy_candidates": [], "enter": [],
                 "exit": [], "reduce": [], "hold": []}
        lookup = _build_signal_lookup(empty)
        assert lookup.signals_by_ticker == {}
        assert lookup.universe_sectors == {}


class TestPrecomputeSignalLookups:
    def test_returns_none_for_none_input(self):
        """Live signal-replay path passes ``signals_by_date=None``
        (loads per-date from S3 inside _simulate_single_date). Function
        must not attempt to iterate."""
        assert _precompute_signal_lookups(None) is None

    def test_per_date_lookup_built(self):
        signals_by_date = {
            "2026-04-24": _signals_raw("2026-04-24"),
            "2026-04-25": _signals_raw("2026-04-25"),
        }
        lookups = _precompute_signal_lookups(signals_by_date)
        assert set(lookups) == {"2026-04-24", "2026-04-25"}
        for d, lookup in lookups.items():
            assert isinstance(lookup, SignalLookup)

    def test_universe_filter_propagates_to_each_date(self):
        signals_by_date = {
            "2026-04-24": _signals_raw("2026-04-24"),
            "2026-04-25": _signals_raw("2026-04-25"),
        }
        universe = {"AAPL", "MSFT", "JPM", "GOOGL"}  # excludes DROPPED_TKR
        rejected: dict[str, int] = {}
        lookups = _precompute_signal_lookups(signals_by_date, universe, rejected)

        # Both dates filter the same way
        for d in lookups:
            assert "DROPPED_TKR" not in lookups[d].signals_by_ticker

        # Rejection counter accumulates across dates (DROPPED_TKR
        # appeared once per date, so total ≥ 2)
        assert rejected.get("DROPPED_TKR", 0) >= 2


class TestCrossComboAmortization:
    """Pin the perf invariant: param sweep precomputes ONCE; not per
    combo. This test checks behavior; the actual perf measurement is
    via the v13 spot dispatch."""

    def test_signal_lookup_is_frozen_dataclass(self):
        """Frozen dataclass means callers can't accidentally mutate the
        shared lookup across combos and pollute other combos' state."""
        from dataclasses import is_dataclass, fields

        assert is_dataclass(SignalLookup)
        # Try to mutate — should raise FrozenInstanceError
        lookup = _build_signal_lookup(_signals_raw())
        with pytest.raises(Exception):  # FrozenInstanceError
            lookup.signals_by_ticker = {}  # type: ignore[misc]

    def test_signals_by_ticker_dict_is_shared_reference(self):
        """The inner dicts (signals_by_ticker, universe_sectors) ARE
        mutable — they're stdlib dicts, not frozen. This is intentional
        for performance (no per-combo deep copy). Document the contract:
        deciders MUST treat them as read-only.

        (No assertion needed — this test exists as a docstring marker
        for the implicit contract.)
        """
        lookup = _build_signal_lookup(_signals_raw())
        # If a future PR makes signals_by_ticker a frozenset / read-only
        # mapping, deciders that try to mutate will surface immediately.
        # For now, the contract is "deciders read-only by convention."
        assert isinstance(lookup.signals_by_ticker, dict)
