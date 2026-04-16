"""Unit tests for loaders.price_loader — post-Phase-0 ArcticDB-only path.

Covers build_matrix() behavior when the underlying ArcticDB read is mocked:
  * attrs populated (price_gap_warnings, unfilled_gaps, staleness_warning,
    stale_circuit_break, no_data_dates)
  * ffill limit of 5 days — tickers with larger gaps dropped
  * Empty dates list → empty DataFrame with attrs intact
  * Signal-ticker resolution filters ArcticDB output to the requested universe
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# arcticdb is a heavy C-extension dep only available on the spot instance;
# stub it in sys.modules so the module-level `import arcticdb as adb` in
# store.arctic_reader resolves during local test runs. Real calls are
# patched per test via mock.patch.
sys.modules.setdefault("arcticdb", MagicMock())

# Force submodule attribute registration so @patch("store.arctic_reader...") +
# @patch("loaders.price_loader...") dotted-path resolution succeeds (store/ and
# loaders/ are namespace packages — submodules aren't auto-exposed as attrs).
import loaders.price_loader  # noqa: E402, F401
import store.arctic_reader  # noqa: E402, F401


def _make_arctic_mock(ticker_series_map: dict[str, pd.Series], field: str = "Close") -> tuple:
    """Build a (price_data, features_by_ticker) tuple matching ArcticDB's shape.

    Each series becomes a DataFrame with Open/High/Low/Close/Volume columns.
    """
    price_data: dict[str, pd.DataFrame] = {}
    for ticker, series in ticker_series_map.items():
        df = pd.DataFrame({
            "Open": series,
            "High": series,
            "Low": series,
            field: series,
            "Volume": pd.Series(1_000_000, index=series.index, dtype="int64"),
        })
        price_data[ticker] = df
    return price_data, {}


class TestBuildMatrixArcticDB:

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_returns_df_with_required_attrs(self, mock_arctic, mock_signals):
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]
        mock_signals.return_value = ["AAPL", "MSFT"]

        idx = pd.to_datetime(dates)
        mock_arctic.return_value = _make_arctic_mock({
            "AAPL": pd.Series(np.arange(100, 110, dtype=float), index=idx),
            "MSFT": pd.Series(np.arange(200, 210, dtype=float), index=idx),
        })

        df = build_matrix(dates, bucket="test")

        for key in ("price_gap_warnings", "unfilled_gaps", "staleness_warning",
                    "stale_circuit_break", "no_data_dates"):
            assert key in df.attrs, f"missing attr: {key}"

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_ffill_limit_drops_tickers_with_wide_gaps(self, mock_arctic, mock_signals):
        """Tickers with gaps > 5 days get dropped from the matrix."""
        from loaders.price_loader import build_matrix

        dates = [f"2026-03-{d:02d}" for d in range(2, 12)]
        mock_signals.return_value = ["AAPL", "MSFT"]

        full_idx = pd.to_datetime(dates)
        sparse_idx = pd.to_datetime(["2026-03-02"])

        price_data, _ = _make_arctic_mock({
            "AAPL": pd.Series([100.0], index=sparse_idx),     # 1 date of 10
            "MSFT": pd.Series(np.arange(200, 210, dtype=float), index=full_idx),
        })
        mock_arctic.return_value = (price_data, {})

        df = build_matrix(dates, bucket="test")
        assert "AAPL" not in df.columns, "AAPL has wide gap — must be dropped"
        assert "AAPL" in df.attrs["unfilled_gaps"]
        assert "MSFT" in df.columns

    @patch("loaders.price_loader._tickers_from_signals")
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_empty_dates_returns_empty_df(self, mock_arctic, mock_signals):
        from loaders.price_loader import build_matrix

        df = build_matrix([], bucket="test")
        assert df.empty
        assert df.attrs["no_data_dates"] == []
        # ArcticDB read should not have happened for zero signal tickers
        mock_arctic.assert_not_called()

    @patch("loaders.price_loader._tickers_from_signals", return_value=[])
    @patch("loaders.price_loader.load_universe_from_arctic")
    def test_no_signals_returns_empty_df(self, mock_arctic, mock_signals):
        """When signals.json resolves zero tickers, skip the ArcticDB read."""
        from loaders.price_loader import build_matrix

        df = build_matrix(["2026-03-10"], bucket="test")
        assert df.empty
        assert df.attrs["no_data_dates"] == ["2026-03-10"]
        mock_arctic.assert_not_called()


class TestArcticFreshnessGate:
    """_verify_arctic_fresh guards against stale/missing SPY in ArcticDB."""

    def _macro_lib(self, last_date=None, raise_exc=None):
        from unittest.mock import MagicMock
        lib = MagicMock()
        if raise_exc is not None:
            lib.read.side_effect = raise_exc
            return lib
        if last_date is None:
            lib.read.return_value.data = pd.DataFrame(columns=["Close"])
        else:
            idx = pd.DatetimeIndex([pd.Timestamp(last_date)])
            lib.read.return_value.data = pd.DataFrame({"Close": [500.0]}, index=idx)
        return lib

    @patch("store.arctic_reader._get_arctic")
    def test_missing_spy_raises(self, mock_get_arctic):
        import pytest

        from store.arctic_reader import _verify_arctic_fresh

        macro = self._macro_lib(raise_exc=Exception("SymbolNotFound"))
        arctic = mock_get_arctic.return_value
        arctic.get_library.return_value = macro

        with pytest.raises(RuntimeError, match="unreadable"):
            _verify_arctic_fresh(bucket="test", min_date="2026-04-16")

    @patch("store.arctic_reader._get_arctic")
    def test_fresh_spy_passes(self, mock_get_arctic):
        from store.arctic_reader import _verify_arctic_fresh

        macro = self._macro_lib(last_date="2026-04-16")
        arctic = mock_get_arctic.return_value
        arctic.get_library.return_value = macro

        _verify_arctic_fresh(bucket="test", min_date="2026-04-16")  # should not raise

    @patch("store.arctic_reader._get_arctic")
    def test_stale_spy_raises(self, mock_get_arctic):
        import pytest

        from store.arctic_reader import _verify_arctic_fresh

        macro = self._macro_lib(last_date="2026-04-15")
        arctic = mock_get_arctic.return_value
        arctic.get_library.return_value = macro

        with pytest.raises(RuntimeError, match="stale"):
            _verify_arctic_fresh(bucket="test", min_date="2026-04-16")
