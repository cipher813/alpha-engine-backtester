"""tests/test_price_histories_filter.py — _build_filtered_price_histories
unit tests.

Locks the queried-ticker invariant introduced 2026-04-26 v8: per-date
``price_histories`` is materialized ONLY for tickers the executor will
actually access (held + signals + sector ETFs), not the full
911-ticker universe. The 18× per-date slicing speedup brought
predictor_single_run from 53 min to ~32 min on c5.large.

If the executor ever adds a code path that iterates
``price_histories`` keys/values/items, or queries a ticker outside
the held + signals + sector-ETF set, this test catches the regression
because the filter no longer covers the new lookup.

2026-04-27: output shape flipped from ``dict[str, list[dict]]`` to
``dict[str, pd.DataFrame]`` (alpha-engine PR #108). The slice itself
is ``df.loc[:ts]`` — pandas binary search on the DatetimeIndex, no
per-row materialization. Each ``out[ticker]`` is a DataFrame view.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest import (
    _SECTOR_ETF_TICKERS,
    _build_filtered_price_histories,
)


def _df(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    """Tiny OHLCV DataFrame for a single ticker."""
    return pd.DataFrame(
        {
            "open":  [base + i * 0.1 for i in range(n_bars)],
            "high":  [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low":   [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2026-01-01", periods=n_bars),
    )


class TestBuildFilteredPriceHistories:
    def test_held_ticker_included(self):
        ohlcv = {"AAPL": _df(), "MSFT": _df(base=200.0)}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers={"AAPL"},
        )
        assert "AAPL" in out
        assert "MSFT" not in out  # not held + not in signals + not sector ETF
        # Output is a DataFrame view, not a list-of-dicts
        assert isinstance(out["AAPL"], pd.DataFrame)
        assert list(out["AAPL"].columns) == ["open", "high", "low", "close"]

    def test_signals_buy_candidates_included(self):
        ohlcv = {"NVDA": _df(), "TSLA": _df(base=300.0)}
        signals = {
            "buy_candidates": [{"ticker": "NVDA", "signal": "ENTER"}],
            "universe": [],
        }
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw=signals,
            held_tickers=set(),
        )
        assert "NVDA" in out
        assert "TSLA" not in out

    def test_signals_universe_entries_included(self):
        ohlcv = {"COST": _df(), "PEP": _df(base=170.0)}
        signals = {
            "buy_candidates": [],
            "universe": [
                {"ticker": "COST", "signal": "EXIT"},
                {"ticker": "PEP", "signal": "HOLD"},
            ],
        }
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw=signals,
            held_tickers=set(),
        )
        assert "COST" in out
        assert "PEP" in out

    def test_sector_etfs_always_included(self):
        """All 11 sector ETFs + SPY are always materialized — the
        executor's sector_relative_veto references them every call
        regardless of signals or holdings."""
        ohlcv = {etf: _df(base=100 + i) for i, etf in enumerate(_SECTOR_ETF_TICKERS)}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers=set(),
        )
        assert set(out.keys()) == set(_SECTOR_ETF_TICKERS)

    def test_combined_set_overlaps_dedup(self):
        """A ticker that's both held + in signals + a sector ETF
        should appear exactly once (set semantics)."""
        ohlcv = {"SPY": _df(base=400.0)}  # SPY: in sector set
        signals = {
            "buy_candidates": [{"ticker": "SPY"}],
            "universe": [{"ticker": "SPY"}],
        }
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw=signals,
            held_tickers={"SPY"},
        )
        assert "SPY" in out
        # Set dedup ensures one entry, not three
        assert len(out) == 1

    def test_filter_skips_full_universe(self):
        """Verify the speedup: a 911-ticker universe with only 3 in
        signals + 0 held should produce ~14 entries (3 signals + 11
        sector ETFs + SPY) rather than 911."""
        full_universe = {
            f"TKR{i:04d}": _df(base=50 + i % 200) for i in range(911)
        }
        # Add the sector ETFs to the universe so the filter can find them
        for etf in _SECTOR_ETF_TICKERS:
            full_universe[etf] = _df(base=100.0)
        # Three signals tickers
        for t in ("TKR0001", "TKR0500", "TKR0900"):
            assert t in full_universe

        signals = {
            "buy_candidates": [{"ticker": "TKR0001"}, {"ticker": "TKR0500"}],
            "universe": [{"ticker": "TKR0900"}],
        }
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=full_universe,
            signal_date="2026-02-01",
            signals_raw=signals,
            held_tickers=set(),
        )
        # 3 signals tickers + 12 sector ETFs (SPY is in the set) = 15 total
        assert len(out) == 3 + len(_SECTOR_ETF_TICKERS)
        # The 908 untouched tickers are NOT in the result
        assert "TKR0002" not in out
        assert "TKR0050" not in out

    def test_missing_ohlcv_for_queried_ticker_skipped(self):
        """If a queried ticker isn't in ohlcv_by_ticker (e.g. a held
        position whose ticker fell out of the universe), the filter
        skips it — no KeyError. The executor's .get(ticker) then sees
        it as absent and falls back to no-history semantics."""
        ohlcv = {"AAPL": _df()}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers={"AAPL", "MISSING"},
        )
        assert "AAPL" in out
        assert "MISSING" not in out

    def test_empty_inputs_produces_only_sector_etfs(self):
        """Edge case: no signals, no held positions. Result still
        contains sector ETF entries because they're always-on."""
        ohlcv = {etf: _df() for etf in _SECTOR_ETF_TICKERS}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers=set(),
        )
        assert set(out.keys()) == set(_SECTOR_ETF_TICKERS)

    def test_empty_dataframe_skipped(self):
        """A ticker present in ohlcv_by_ticker but with an empty
        DataFrame is skipped (consistent with the legacy behavior of
        the per-ticker comprehension)."""
        ohlcv = {"AAPL": _df(), "EMPTY": pd.DataFrame()}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers={"AAPL", "EMPTY"},
        )
        assert "AAPL" in out
        assert "EMPTY" not in out

    def test_signal_date_truncates_slice(self):
        """The slice is ``df.loc[:signal_date]`` — bars after signal_date
        are excluded. Locks the no-lookahead invariant."""
        df = _df(n_bars=100)  # spans 2026-01-01 through ~2026-05-21
        ohlcv = {"AAPL": df}
        out = _build_filtered_price_histories(
            ohlcv_by_ticker=ohlcv,
            signal_date="2026-02-01",
            signals_raw={"buy_candidates": [], "universe": []},
            held_tickers={"AAPL"},
        )
        sliced = out["AAPL"]
        assert sliced.index[-1] <= pd.Timestamp("2026-02-01")
        # And the slice should contain at least the bars up to 2026-02-01
        assert sliced.index[-1] >= pd.Timestamp("2026-01-30")
