"""tests/test_df_slice_to_bars.py — _df_slice_to_bars unit tests.

Locks the trailing-window cap introduced 2026-04-27: the simulate-path
``price_histories`` materialization is bounded to the last 400 bars,
regardless of how much history precedes ``until_date``. This flattens
the per-date cost growth that dominated v9's rate decay
(0.80s/date → 2.04s/date over 1500 dates).

If a future executor consumer needs more than 400 bars of history (e.g.
extending correlation lookback past 60 days, or a new long-window
feature), this test catches the regression because the cap would silently
truncate the input.
"""
from __future__ import annotations

import pandas as pd
import pytest

from synthetic.predictor_backtest import _MAX_BARS_FOR_EXECUTOR, _df_slice_to_bars


def _df(n_bars: int, base: float = 100.0) -> pd.DataFrame:
    """OHLCV DataFrame with a DatetimeIndex spanning n_bars business days."""
    return pd.DataFrame(
        {
            "open":  [base + i * 0.1 for i in range(n_bars)],
            "high":  [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low":   [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2018-01-02", periods=n_bars),
    )


class TestDfSliceToBars:
    def test_short_history_returned_in_full(self):
        # 100 bars total, ask for everything up to the last bar — nothing
        # to cap, full slice returned.
        df = _df(n_bars=100)
        until = df.index[-1].strftime("%Y-%m-%d")
        bars = _df_slice_to_bars(df, until)
        assert len(bars) == 100

    def test_at_cap_returned_in_full(self):
        df = _df(n_bars=_MAX_BARS_FOR_EXECUTOR)
        until = df.index[-1].strftime("%Y-%m-%d")
        bars = _df_slice_to_bars(df, until)
        assert len(bars) == _MAX_BARS_FOR_EXECUTOR

    def test_long_history_capped_to_max(self):
        # 2500 bars (~10y) — must cap to last 400.
        df = _df(n_bars=2500)
        until = df.index[-1].strftime("%Y-%m-%d")
        bars = _df_slice_to_bars(df, until)
        assert len(bars) == _MAX_BARS_FOR_EXECUTOR

    def test_cap_keeps_most_recent_bars(self):
        # The cap keeps the *last* 400 bars, not the first.
        df = _df(n_bars=2500)
        until = df.index[-1].strftime("%Y-%m-%d")
        bars = _df_slice_to_bars(df, until)
        assert bars[-1]["date"] == df.index[-1].strftime("%Y-%m-%d")
        # The kept first bar is at index (n_bars - cap).
        expected_first = df.index[2500 - _MAX_BARS_FOR_EXECUTOR].strftime("%Y-%m-%d")
        assert bars[0]["date"] == expected_first

    def test_until_date_mid_history_caps_relative_to_until(self):
        # Slice up to bar 1500 (mid-history) — cap relative to that point,
        # not relative to df end.
        df = _df(n_bars=2500)
        until = df.index[1500].strftime("%Y-%m-%d")
        bars = _df_slice_to_bars(df, until)
        assert len(bars) == _MAX_BARS_FOR_EXECUTOR
        assert bars[-1]["date"] == until
        # First bar should be 400 business days before until.
        expected_first = df.index[1500 - _MAX_BARS_FOR_EXECUTOR + 1].strftime("%Y-%m-%d")
        assert bars[0]["date"] == expected_first

    def test_empty_df_returns_empty_list(self):
        df = pd.DataFrame(
            {"open": [], "high": [], "low": [], "close": []},
            index=pd.DatetimeIndex([], name=None),
        )
        bars = _df_slice_to_bars(df, "2024-01-01")
        assert bars == []

    def test_until_date_before_history_start_returns_empty(self):
        df = _df(n_bars=100)
        bars = _df_slice_to_bars(df, "2017-01-01")
        assert bars == []

    def test_max_bars_constant_at_expected_value(self):
        # Pin the constant — if a future PR changes it, the test author
        # must intentionally update this assertion (and re-justify against
        # the docstring's enumeration of consumer lookback windows).
        assert _MAX_BARS_FOR_EXECUTOR == 400
