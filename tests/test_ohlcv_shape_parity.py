"""Parity tests for the ohlcv_by_ticker pandas/numpy refactor.

Backtester pandas refactor plan (2026-04-23): ohlcv_by_ticker's shape
changes from ``dict[str, list[dict]]`` to ``dict[str, pd.DataFrame]`` to
recover ~1 GB of peak memory in predictor_data_prep. These tests lock
behavioral equivalence at the boundaries where the two shapes coexist
during the migration:

1. Producer parity — ``build_ohlcv_df_by_ticker`` (new) piped through
   ``_df_to_bars`` must produce byte-identical bars to
   ``build_ohlcv_by_ticker`` (old).
2. Slice parity — ``_df_slice_to_bars(df, until_date)`` must return the
   same list-of-dicts as filtering the old form by ``<= until_date``.
3. Artifact round-trip — ``save_dict_of_dataframes`` /
   ``load_dict_of_dataframes`` survive byte-identical on the new shape.
4. Macro + sector filter — both producers drop the same ticker set.

Full-simulate parity (executor-boundary order equivalence) lives in a
follow-up test added alongside ``_simulate_single_date``'s shape
detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phase_artifacts import (
    load_dict_of_dataframes,
    save_dict_of_dataframes,
)
from synthetic.predictor_backtest import (
    _df_slice_to_bars,
    _df_to_bars,
    build_ohlcv_by_ticker,
    build_ohlcv_df_by_ticker,
)
from tests.test_phase_registry import _FakeS3


@pytest.fixture
def small_price_data() -> dict[str, pd.DataFrame]:
    """Deterministic ArcticDB-style price_data fixture: 5 tickers over
    50 business days, capitalized OHLC columns, mixed full- and
    short-history. Mirrors the production shape the producer sees
    coming out of ``load_universe_from_arctic``."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=50)
    data: dict[str, pd.DataFrame] = {}
    for ticker, n_bars, starts_late in [
        ("AAPL",  50, False),
        ("MSFT",  50, False),
        ("GOOGL", 50, False),
        # Short-history spin-off-like (starts late)
        ("NEWIPO", 20, True),
        # Delisted-like (ends early)
        ("STALE", 15, False),
    ]:
        closes = 100 + rng.normal(0, 1, n_bars).cumsum()
        idx = dates[-n_bars:] if starts_late else dates[:n_bars]
        df = pd.DataFrame(
            {
                "Open":  closes - 0.5,
                "High":  closes + 1.0,
                "Low":   closes - 1.0,
                "Close": closes,
                "Volume": (rng.uniform(1e6, 5e6, n_bars)).astype(int),
            },
            index=idx,
        )
        data[ticker] = df
    return data


def _canon(bars: list[dict]) -> list[dict]:
    """Canonicalize a bar list so numpy-scalar vs. Python-float
    differences between producers don't cause false divergence."""
    return [
        {
            "date":  b["date"],
            "open":  float(b["open"]),
            "high":  float(b["high"]),
            "low":   float(b["low"]),
            "close": float(b["close"]),
        }
        for b in bars
    ]


class TestProducerParity:
    def test_producer_keys_match(self, small_price_data):
        old = build_ohlcv_by_ticker(small_price_data)
        new = build_ohlcv_df_by_ticker(small_price_data)
        assert set(old.keys()) == set(new.keys())

    def test_producer_bar_content_byte_identical(self, small_price_data):
        """``build_ohlcv_by_ticker`` (list-of-dicts) must produce
        byte-identical bars to ``_df_to_bars(build_ohlcv_df_by_ticker[t])``
        for every ticker and every bar."""
        old = build_ohlcv_by_ticker(small_price_data)
        new_df = build_ohlcv_df_by_ticker(small_price_data)
        for ticker in old:
            via_new = _df_to_bars(new_df[ticker])
            assert _canon(old[ticker]) == _canon(via_new), (
                f"producer divergence for {ticker}"
            )

    def test_producer_skips_macro_and_sector_etfs(self, small_price_data):
        """Both producers drop the same ``_MACRO_TICKERS | _SECTOR_ETFS``
        set — parity here ensures the skip-list stays a single source
        of truth in ``synthetic/predictor_backtest.py``."""
        fixture = dict(small_price_data)
        # Inject a macro + sector ETF from real _MACRO_TICKERS / _SECTOR_ETFS
        fixture["SPY"] = small_price_data["AAPL"]
        fixture["XLK"] = small_price_data["MSFT"]

        old = build_ohlcv_by_ticker(fixture)
        new_df = build_ohlcv_df_by_ticker(fixture)

        for skip in ("SPY", "XLK"):
            assert skip not in old
            assert skip not in new_df

    def test_producer_drops_empty_dataframe(self):
        """None / empty DataFrame inputs don't land in the output dict."""
        data = {
            "AAPL": pd.DataFrame(
                {"Close": [100.0, 101.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
            ),
            "EMPTY": pd.DataFrame(),
        }
        new_df = build_ohlcv_df_by_ticker(data)
        assert "EMPTY" not in new_df
        assert "AAPL" in new_df

    def test_producer_normalizes_to_lowercase_columns(self, small_price_data):
        """Fixture uses capitalized columns; new producer emits lowercase."""
        new_df = build_ohlcv_df_by_ticker(small_price_data)
        for ticker, df in new_df.items():
            assert list(df.columns) == ["open", "high", "low", "close"], (
                f"{ticker} columns: {list(df.columns)}"
            )

    def test_producer_fallback_to_close_when_ohl_missing(self):
        """Ticker with only Close populated: O/H/L fall back to Close —
        matches the list-of-dicts producer's _df_to_bars behavior."""
        data = {
            "CLOSEONLY": pd.DataFrame(
                {"Close": [100.0, 101.0, 102.0]},
                index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            ),
        }
        new_df = build_ohlcv_df_by_ticker(data)
        frame = new_df["CLOSEONLY"]
        for col in ("open", "high", "low"):
            pd.testing.assert_series_equal(
                frame[col], frame["close"], check_names=False,
            )


class TestSliceParity:
    @pytest.mark.parametrize("until_date", [
        "2023-12-31",  # before any bar
        "2024-01-02",  # inclusive of first bar
        "2024-01-20",  # weekend (Saturday)
        "2024-02-15",  # mid-range
        "2024-03-12",  # last bar in fixture
        "2099-01-01",  # after all bars
    ])
    def test_slice_matches_scalar_filter(self, small_price_data, until_date):
        """``_df_slice_to_bars`` must match the scalar ``<=`` filter on
        every slice boundary we care about in production."""
        old = build_ohlcv_by_ticker(small_price_data)
        new = build_ohlcv_df_by_ticker(small_price_data)
        for ticker in new:
            scalar = [b for b in old[ticker] if b["date"] <= until_date]
            vectorized = _df_slice_to_bars(new[ticker], until_date)
            assert _canon(scalar) == _canon(vectorized), (
                f"slice divergence for {ticker} at {until_date}"
            )

    def test_slice_inclusive_on_exact_boundary(self, small_price_data):
        """until_date exactly matches a bar's date — that bar must be
        included in the slice (inclusive semantic)."""
        new = build_ohlcv_df_by_ticker(small_price_data)
        # AAPL spans 2024-01-02..2024-03-12
        bars = _df_slice_to_bars(new["AAPL"], "2024-01-05")
        assert bars[-1]["date"] == "2024-01-05"
        assert len(bars) >= 1


class TestArtifactRoundTrip:
    def test_save_load_dict_of_dataframes_preserves_frames(self, small_price_data):
        """``save_dict_of_dataframes`` → ``load_dict_of_dataframes`` must
        preserve each ticker's DataFrame exactly when data is in the new
        shape. Uses the existing ``_FakeS3`` fake for in-memory I/O."""
        new = build_ohlcv_df_by_ticker(small_price_data)
        s3 = _FakeS3()

        key = save_dict_of_dataframes(
            bucket="b",
            date="2024-03-12",
            phase="predictor_data_prep",
            name="ohlcv_by_ticker",
            data=new,
            s3_client=s3,
        )
        loaded = load_dict_of_dataframes("b", key, s3_client=s3)

        assert set(loaded.keys()) == set(new.keys())
        for ticker, df in new.items():
            loaded_df = loaded[ticker]
            # Normalize column order (groupby round-trip can re-order)
            loaded_df = loaded_df[list(df.columns)]
            pd.testing.assert_frame_equal(
                loaded_df, df,
                check_names=False,
                check_freq=False,
                check_column_type=False,
            )
