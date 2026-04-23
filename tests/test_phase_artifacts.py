"""
tests/test_phase_artifacts.py — round-trip + S3 key contract for
phase_artifacts save/load helpers.

Reuses the minimal in-memory S3 fake from test_phase_registry.
"""

from __future__ import annotations

import pandas as pd
import pytest

from phase_artifacts import (
    artifact_key,
    load_dataframe,
    load_json,
    load_ohlcv_by_ticker,
    save_dataframe,
    save_json,
    save_ohlcv_by_ticker,
)
from tests.test_phase_registry import _FakeS3


@pytest.fixture
def s3():
    return _FakeS3()


def test_artifact_key_layout():
    k = artifact_key("2026-04-23", "simulate", "portfolio_stats", "json")
    assert k == "backtest/2026-04-23/.phases/simulate/portfolio_stats.json"


# ── JSON round-trips ─────────────────────────────────────────────────────────


def test_save_load_json_dict_roundtrip(s3):
    obj = {"status": "ok", "sharpe_ratio": 1.23, "total_trades": 42}
    key = save_json("b", "2026-04-23", "simulate", "portfolio_stats", obj, s3_client=s3)
    loaded = load_json("b", key, s3_client=s3)
    assert loaded == obj


def test_save_json_handles_non_native_types(s3):
    """numpy / datetime scalars must not crash the encoder."""
    import numpy as np
    from datetime import datetime
    obj = {"np_scalar": np.float64(3.14), "dt": datetime(2026, 4, 23)}
    key = save_json("b", "2026-04-23", "simulate", "stats", obj, s3_client=s3)
    loaded = load_json("b", key, s3_client=s3)
    # default=str coerces both; round-trip produces strings
    assert loaded["np_scalar"] == 3.14  # JSON float
    assert "2026-04-23" in loaded["dt"]


def test_save_json_writes_to_expected_s3_key(s3):
    save_json("my-bucket", "2026-04-23", "simulate", "portfolio_stats", {"ok": True}, s3_client=s3)
    assert ("my-bucket", "backtest/2026-04-23/.phases/simulate/portfolio_stats.json") in s3.store


# ── DataFrame round-trips ────────────────────────────────────────────────────


def test_save_load_dataframe_preserves_index(s3):
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    df = pd.DataFrame({"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
                      index=dates)
    key = save_dataframe("b", "2026-04-23", "simulation_setup", "price_matrix",
                         df, s3_client=s3)
    loaded = load_dataframe("b", key, s3_client=s3)
    pd.testing.assert_frame_equal(df, loaded, check_freq=False)


def test_save_dataframe_without_index(s3):
    df = pd.DataFrame({"sharpe": [1.2, 1.3], "total_alpha": [0.5, 0.6]})
    key = save_dataframe("b", "2026-04-23", "param_sweep", "sweep_df",
                         df, s3_client=s3, preserve_index=False)
    loaded = load_dataframe("b", key, s3_client=s3)
    # Index is default RangeIndex on both sides
    pd.testing.assert_frame_equal(df, loaded)


# ── OHLCV round-trips ────────────────────────────────────────────────────────


def _sample_ohlcv() -> dict[str, list[dict]]:
    return {
        "AAPL": [
            {"date": "2026-01-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"date": "2026-01-02", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5},
        ],
        "MSFT": [
            {"date": "2026-01-01", "open": 200.0, "high": 201.0, "low": 199.0, "close": 200.5},
            {"date": "2026-01-02", "open": 200.5, "high": 202.0, "low": 200.0, "close": 201.5},
            {"date": "2026-01-03", "open": 201.5, "high": 203.0, "low": 201.0, "close": 202.5},
        ],
    }


def test_save_load_ohlcv_by_ticker_roundtrip(s3):
    original = _sample_ohlcv()
    key = save_ohlcv_by_ticker("b", "2026-04-23", "simulation_setup", "ohlcv",
                               original, s3_client=s3)
    loaded = load_ohlcv_by_ticker("b", key, s3_client=s3)

    assert set(loaded.keys()) == set(original.keys())
    for ticker, bars in original.items():
        assert len(loaded[ticker]) == len(bars), f"ticker {ticker}"
        for orig_bar, loaded_bar in zip(bars, loaded[ticker]):
            for field in ("date", "open", "high", "low", "close"):
                assert loaded_bar[field] == orig_bar[field], f"{ticker} field {field}"


def test_load_ohlcv_rejects_missing_ticker_column(s3):
    """Guardrail: a parquet uploaded without a 'ticker' column must not
    silently become an empty-dict OHLCV — fail loud so the pipeline
    doesn't run a zero-ticker simulation."""
    df = pd.DataFrame({"date": ["2026-01-01"], "close": [100.0]})
    key = save_dataframe("b", "2026-04-23", "simulation_setup", "ohlcv",
                         df, s3_client=s3, preserve_index=False)
    with pytest.raises(ValueError, match="missing 'ticker' column"):
        load_ohlcv_by_ticker("b", key, s3_client=s3)


def test_ohlcv_preserves_date_order_per_ticker(s3):
    """Regression guard: simulate relies on OHLCV bars being in date order."""
    ohlcv = {
        "AAPL": [
            {"date": "2026-01-03", "close": 103.0},
            {"date": "2026-01-01", "close": 101.0},
            {"date": "2026-01-02", "close": 102.0},
        ],
    }
    key = save_ohlcv_by_ticker("b", "2026-04-23", "simulation_setup", "ohlcv",
                               ohlcv, s3_client=s3)
    loaded = load_ohlcv_by_ticker("b", key, s3_client=s3)
    # Preserve original enumeration order — callers that want sorted
    # dates should sort themselves (matches current in-process behavior).
    assert [b["date"] for b in loaded["AAPL"]] == ["2026-01-03", "2026-01-01", "2026-01-02"]
