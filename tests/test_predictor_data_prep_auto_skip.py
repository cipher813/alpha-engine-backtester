"""
tests/test_predictor_data_prep_auto_skip.py — round-trip test for the
predictor-side artifact persistence (PR 3/3).

Covers the biggest artifact surface: predictor_data_prep, which persists
8 files (7 JSON/parquet scalars + features_by_ticker stacked parquet).
Verifies save→load reconstructs the same dict shape that
`synthetic.predictor_backtest.run(keep_features=True)` produces.
"""

from __future__ import annotations

import pandas as pd
import pytest

from pipeline_common import PhaseRegistry
import backtest as bt
from tests.test_phase_registry import _FakeS3


@pytest.fixture
def s3():
    return _FakeS3()


def _sample_result() -> dict:
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    price_matrix = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
        index=dates,
    )
    ohlcv = {
        # DataFrame shape per Option A step 9 (DatetimeIndex + lowercase OHLC)
        "AAPL": pd.DataFrame(
            {"open": [100.0, 100.5], "high": [101.0, 102.0],
             "low": [99.0, 100.0], "close": [100.5, 101.5]},
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        ),
    }
    spy = pd.Series([400.0, 401.0, 402.0], index=dates, name="SPY")
    features = {
        "AAPL": pd.DataFrame(
            {"feat1": [0.1, 0.2, 0.3], "feat2": [1.1, 1.2, 1.3]}, index=dates,
        ),
        "MSFT": pd.DataFrame(
            {"feat1": [0.5, 0.6], "feat2": [1.5, 1.6]}, index=dates[:2],
        ),
    }
    return {
        "status": "ok",
        "signals_by_date": {
            "2026-01-01": {"buy_candidates": [{"ticker": "AAPL", "score": 80}]},
            "2026-01-02": {"buy_candidates": [{"ticker": "MSFT", "score": 75}]},
        },
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv,
        "spy_prices": spy,
        "metadata": {"n_tickers": 2, "n_dates": 3, "top_n_per_day": 20},
        "sector_map": {"AAPL": "XLK", "MSFT": "XLK"},
        "trading_dates": ["2026-01-01", "2026-01-02", "2026-01-03"],
        "predictions_by_date": {
            "2026-01-01": {"AAPL": 0.02, "MSFT": -0.01},
            "2026-01-02": {"AAPL": 0.01, "MSFT": 0.03},
        },
        "features_by_ticker": features,
    }


def test_predictor_data_prep_roundtrip(s3):
    """Save on run 1, load on run 2 reconstructs the full result dict."""
    original = _sample_result()

    registry1 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    with registry1.phase("predictor_data_prep", supports_auto_skip=True) as ctx:
        assert ctx.skipped is False
        bt._save_predictor_data_prep(
            ctx, "b", "2026-04-23", original, s3_client=registry1.s3_client,
        )

    # Retry: fresh registry, same S3
    registry2 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    run, reason = registry2.should_run("predictor_data_prep", supports_auto_skip=True)
    assert run is False
    assert reason == "auto_skip_marker_ok"

    reloaded = bt._load_predictor_data_prep("b", registry2)

    assert reloaded["status"] == "ok"
    assert reloaded["signals_by_date"] == original["signals_by_date"]
    assert reloaded["metadata"] == original["metadata"]
    assert reloaded["sector_map"] == original["sector_map"]
    assert reloaded["trading_dates"] == original["trading_dates"]
    assert reloaded["predictions_by_date"] == original["predictions_by_date"]

    pd.testing.assert_frame_equal(
        reloaded["price_matrix"], original["price_matrix"], check_freq=False,
    )

    # OHLCV round-trip (per-ticker, per-bar)
    assert set(reloaded["ohlcv_by_ticker"].keys()) == {"AAPL"}
    assert len(reloaded["ohlcv_by_ticker"]["AAPL"]) == 2

    # SPY Series
    pd.testing.assert_series_equal(
        reloaded["spy_prices"].astype(float),
        original["spy_prices"].astype(float),
        check_freq=False, check_names=False,
    )

    # features_by_ticker — values preserved per column
    assert set(reloaded["features_by_ticker"].keys()) == {"AAPL", "MSFT"}
    for ticker in ("AAPL", "MSFT"):
        orig = original["features_by_ticker"][ticker]
        got = reloaded["features_by_ticker"][ticker]
        assert set(got.columns) == set(orig.columns)
        for col in orig.columns:
            assert list(got[col].astype(float)) == list(orig[col].astype(float))


def test_save_skips_when_status_not_ok(s3):
    """A failed data_prep shouldn't persist a partial snapshot."""
    registry = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    with registry.phase("predictor_data_prep", supports_auto_skip=True) as ctx:
        bt._save_predictor_data_prep(
            ctx, "b", "2026-04-23", {"status": "error"}, s3_client=registry.s3_client,
        )

    marker = registry.load_marker("predictor_data_prep")
    assert marker is not None
    assert marker["status"] == "ok"
    assert marker["artifact_keys"] == []


def test_predictor_feature_maps_roundtrip(s3):
    atr = {"AAPL": 0.0123, "MSFT": 0.0234}
    coverage = {"AAPL": 0.98, "MSFT": 0.91}
    vwap = {
        "AAPL": pd.Series([100.0, 101.0], index=["2026-01-01", "2026-01-02"]),
        "MSFT": pd.Series([200.0, 201.0], index=["2026-01-01", "2026-01-02"]),
    }

    registry1 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    with registry1.phase("predictor_feature_maps_bulk_load", supports_auto_skip=True) as ctx:
        bt._save_predictor_feature_maps(
            ctx, "b", "2026-04-23", atr, vwap, coverage,
            s3_client=registry1.s3_client,
        )

    registry2 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    reloaded_atr, reloaded_vwap, reloaded_cov = bt._load_predictor_feature_maps("b", registry2)

    assert reloaded_atr == atr
    assert reloaded_cov == coverage
    assert set(reloaded_vwap.keys()) == {"AAPL", "MSFT"}
    # Values preserved (index is retained via the `idx` column)
    assert list(reloaded_vwap["AAPL"].astype(float)) == [100.0, 101.0]
    assert list(reloaded_vwap["MSFT"].astype(float)) == [200.0, 201.0]
