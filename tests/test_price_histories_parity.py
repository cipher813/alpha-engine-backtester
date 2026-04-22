"""Equivalence guard: the bisect+slice path in ``_simulate_single_date``
must produce identical ``price_histories`` to the prior scalar
``[b for b in bars if b["date"] <= signal_date]`` list comprehension.

The pre-2026-04-22 implementation rebuilt ``price_histories`` per call
via that O(N_bars) dict-lookup+compare loop. On a 60-combo × 2000-date
× 900-ticker × 2500-bar predictor param sweep, the inner op count
worked out to ~270B Python ops, which pushed the Saturday SF dry-run
past its 2hr SSM ceiling (2026-04-22 07:16 → 09:16 PT, TIMED_OUT with
zero S3 artifacts written).

The vectorized rewrite precomputes ``ohlcv_dates_index`` — a parallel
list of sorted ISO8601 date strings per ticker — once at the top of
``_run_simulation_loop`` (or passed in from the caller), then uses
``bisect.bisect_right`` for the cut index per date and slices
``bars[:cut]``. Since date strings are ISO8601, lexicographic
comparison is chronological, so bisect semantics match the original
``<=`` filter exactly.

These tests lock the byte-identical output on adversarial date
scenarios — signal_date exactly matching a stored date (inclusive vs
exclusive boundary), signal_date before all stored dates (empty
slice), signal_date after all stored dates (full slice), and
multi-ticker with heterogeneous histories.
"""

from __future__ import annotations

from bisect import bisect_right

import numpy as np
import pytest

from backtest import _build_ohlcv_date_index


def _scalar_filter(ohlcv_by_ticker: dict, signal_date: str) -> dict:
    """Reference implementation — the pre-vectorization behavior."""
    return {
        ticker: [b for b in bars if b["date"] <= signal_date]
        for ticker, bars in ohlcv_by_ticker.items()
    }


def _vectorized_filter(
    ohlcv_by_ticker: dict,
    ohlcv_dates_index: dict,
    signal_date: str,
) -> dict:
    """Production implementation."""
    return {
        ticker: bars[:bisect_right(ohlcv_dates_index[ticker], signal_date)]
        for ticker, bars in ohlcv_by_ticker.items()
    }


def _synthetic_bars(dates: list[str]) -> list[dict]:
    """Deterministic bars for a given date list."""
    return [
        {"date": d, "open": 100.0 + i, "high": 101.0 + i,
         "low": 99.0 + i, "close": 100.5 + i}
        for i, d in enumerate(dates)
    ]


@pytest.fixture
def ohlcv_multi_ticker() -> dict:
    """Three tickers with different history lengths + start dates, mirroring
    production shape (short-history tickers alongside full-history ones)."""
    return {
        "AAPL": _synthetic_bars([
            "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05",
            "2024-01-08", "2024-01-09", "2024-01-10",
        ]),
        # Short-history (e.g. SNDK post-spinoff)
        "NEWIPO": _synthetic_bars(["2024-01-08", "2024-01-09", "2024-01-10"]),
        # Ends earlier (delisted-like)
        "STALE": _synthetic_bars(["2024-01-02", "2024-01-03", "2024-01-04"]),
    }


class TestParityAcrossDates:
    """Every date must produce byte-identical output from both paths."""

    def test_date_before_any_history(self, ohlcv_multi_ticker):
        """signal_date before every stored date → empty slice for every ticker."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        scalar = _scalar_filter(ohlcv_multi_ticker, "2023-12-31")
        vec = _vectorized_filter(ohlcv_multi_ticker, idx, "2023-12-31")
        assert scalar == vec
        assert all(v == [] for v in vec.values())

    def test_date_exactly_matches_first_bar(self, ohlcv_multi_ticker):
        """Inclusive boundary: signal_date == first stored date → slice of 1."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        scalar = _scalar_filter(ohlcv_multi_ticker, "2024-01-02")
        vec = _vectorized_filter(ohlcv_multi_ticker, idx, "2024-01-02")
        assert scalar == vec
        assert len(vec["AAPL"]) == 1
        assert vec["AAPL"][0]["date"] == "2024-01-02"
        # NEWIPO starts later → empty
        assert vec["NEWIPO"] == []

    def test_date_between_bars(self, ohlcv_multi_ticker):
        """signal_date falls between stored dates (e.g. weekend)."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        scalar = _scalar_filter(ohlcv_multi_ticker, "2024-01-06")  # Saturday
        vec = _vectorized_filter(ohlcv_multi_ticker, idx, "2024-01-06")
        assert scalar == vec

    def test_date_exactly_matches_last_bar(self, ohlcv_multi_ticker):
        """Inclusive boundary: signal_date == last stored date → full slice."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        scalar = _scalar_filter(ohlcv_multi_ticker, "2024-01-10")
        vec = _vectorized_filter(ohlcv_multi_ticker, idx, "2024-01-10")
        assert scalar == vec
        assert len(vec["AAPL"]) == 7

    def test_date_after_all_history(self, ohlcv_multi_ticker):
        """signal_date after every stored date → full slice for every ticker."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        scalar = _scalar_filter(ohlcv_multi_ticker, "2025-01-01")
        vec = _vectorized_filter(ohlcv_multi_ticker, idx, "2025-01-01")
        assert scalar == vec
        assert len(vec["AAPL"]) == 7
        assert len(vec["NEWIPO"]) == 3
        assert len(vec["STALE"]) == 3

    def test_every_date_in_history(self, ohlcv_multi_ticker):
        """Iterate every date across every ticker — parity must hold on every
        call. Mirrors the simulate loop's actual access pattern."""
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        all_dates = sorted({b["date"]
                            for bars in ohlcv_multi_ticker.values()
                            for b in bars})
        for d in all_dates:
            scalar = _scalar_filter(ohlcv_multi_ticker, d)
            vec = _vectorized_filter(ohlcv_multi_ticker, idx, d)
            assert scalar == vec, f"divergence at signal_date={d}"


class TestBuildIndex:
    def test_returns_sorted_dates_per_ticker(self, ohlcv_multi_ticker):
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        for ticker, dates in idx.items():
            assert dates == sorted(dates), (
                f"{ticker} date index not sorted — bisect requires sorted input"
            )
            # Parallel to bars
            assert len(dates) == len(ohlcv_multi_ticker[ticker])
            assert dates == [b["date"] for b in ohlcv_multi_ticker[ticker]]

    def test_empty_input(self):
        assert _build_ohlcv_date_index({}) == {}

    def test_none_input(self):
        """Defensive: None collapses to empty dict (no NoneType crash)."""
        assert _build_ohlcv_date_index(None) == {}

    def test_single_ticker_single_bar(self):
        data = {"AAPL": [{"date": "2024-01-02", "close": 100.0}]}
        assert _build_ohlcv_date_index(data) == {"AAPL": ["2024-01-02"]}


class TestShareSemantics:
    """Under heavy param-sweep load the sliced list refs are shared across
    many callers. Assert that slicing does NOT copy the underlying dict
    objects — only pointers. A regression that deep-copies each bar would
    balloon memory and undo the vectorization win."""

    def test_sliced_bars_share_underlying_dicts(self, ohlcv_multi_ticker):
        idx = _build_ohlcv_date_index(ohlcv_multi_ticker)
        sliced = _vectorized_filter(ohlcv_multi_ticker, idx, "2024-01-10")

        # Same dict object (id-equal), not a deep copy.
        for ticker, bars in ohlcv_multi_ticker.items():
            for i, bar in enumerate(sliced[ticker]):
                assert bar is bars[i], (
                    f"{ticker}[{i}] was copied — slice must share refs"
                )


@pytest.mark.parametrize("n_tickers,n_bars_per_ticker", [
    (50, 100),
    (200, 500),
])
def test_parity_on_larger_synthetic(n_tickers, n_bars_per_ticker):
    """Larger synthetic case — still runs fast (well under 1s) but gives
    a higher-confidence equivalence signal."""
    rng = np.random.default_rng(42)
    dates = [f"2020-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}"
             for i in range(n_bars_per_ticker)]
    ohlcv = {}
    for ti in range(n_tickers):
        # Variable-length history so short-history tickers are in the mix
        cut = int(rng.integers(10, n_bars_per_ticker + 1))
        ohlcv[f"T{ti}"] = _synthetic_bars(dates[:cut])

    idx = _build_ohlcv_date_index(ohlcv)
    for d in dates[::23]:  # sample 1 in 23 dates across the range
        scalar = _scalar_filter(ohlcv, d)
        vec = _vectorized_filter(ohlcv, idx, d)
        assert scalar == vec, f"divergence at date={d}"
