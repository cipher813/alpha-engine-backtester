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
    load_dict_of_dataframes,
    load_dict_of_series,
    load_json,
    load_ohlcv_by_ticker,
    load_series,
    save_dataframe,
    save_dict_of_dataframes,
    save_dict_of_series,
    save_json,
    save_ohlcv_by_ticker,
    save_series,
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


def _sample_ohlcv() -> dict[str, pd.DataFrame]:
    """DataFrame-shape fixture per Option A step 9. DatetimeIndex +
    lowercase OHLCV columns (matches ``build_ohlcv_df_by_ticker``)."""
    aapl = pd.DataFrame(
        {"open": [100.0, 100.5], "high": [101.0, 102.0],
         "low": [99.0, 100.0], "close": [100.5, 101.5]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    msft = pd.DataFrame(
        {"open": [200.0, 200.5, 201.5], "high": [201.0, 202.0, 203.0],
         "low": [199.0, 200.0, 201.0], "close": [200.5, 201.5, 202.5]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02", "2026-01-03"]),
    )
    return {"AAPL": aapl, "MSFT": msft}


def test_save_load_ohlcv_by_ticker_roundtrip(s3):
    original = _sample_ohlcv()
    key = save_ohlcv_by_ticker("b", "2026-04-23", "simulation_setup", "ohlcv",
                               original, s3_client=s3)
    loaded = load_ohlcv_by_ticker("b", key, s3_client=s3)

    assert set(loaded.keys()) == set(original.keys())
    for ticker, df in original.items():
        loaded_df = loaded[ticker]
        # Index round-trips as DatetimeIndex
        assert list(pd.DatetimeIndex(loaded_df.index)) == list(df.index), f"{ticker} index"
        # Columns round-trip with values
        for col in ("open", "high", "low", "close"):
            assert list(loaded_df[col]) == list(df[col]), f"{ticker} col {col}"


def test_load_ohlcv_rejects_missing_ticker_column(s3):
    """Guardrail: a parquet uploaded without a 'ticker' column must not
    silently become an empty-dict OHLCV — fail loud so the pipeline
    doesn't run a zero-ticker simulation."""
    df = pd.DataFrame({"date": ["2026-01-01"], "close": [100.0]})
    key = save_dataframe("b", "2026-04-23", "simulation_setup", "ohlcv",
                         df, s3_client=s3, preserve_index=False)
    with pytest.raises(ValueError, match="missing 'ticker' column"):
        load_ohlcv_by_ticker("b", key, s3_client=s3)


def test_load_ohlcv_rejects_legacy_list_of_dicts_artifact(s3):
    """Step 9 invariant: a legacy artifact (no __idx__ column) must be
    rejected hard so an auto-skip can't silently feed list-of-dicts
    into the now-DataFrame-only consumers. Operator response is to
    re-run upstream predictor_data_prep to regenerate."""
    df = pd.DataFrame({
        "ticker": ["AAPL", "AAPL"],
        "date": ["2026-01-01", "2026-01-02"],
        "close": [100.0, 101.0],
    })
    key = save_dataframe("b", "2026-04-23", "simulation_setup", "ohlcv",
                         df, s3_client=s3, preserve_index=False)
    with pytest.raises(ValueError, match="legacy list-of-dicts schema"):
        load_ohlcv_by_ticker("b", key, s3_client=s3)


def test_save_ohlcv_rejects_legacy_list_of_dicts_input(s3):
    """Producer-side guardrail: passing list-of-dicts to save_ohlcv_by_ticker
    raises immediately rather than silently writing the legacy schema."""
    legacy = {"AAPL": [{"date": "2026-01-01", "close": 100.0}]}
    with pytest.raises(TypeError, match="dict\\[str, pd.DataFrame\\]"):
        save_ohlcv_by_ticker("b", "2026-04-23", "simulation_setup", "ohlcv",
                             legacy, s3_client=s3)


def test_ohlcv_preserves_date_order_per_ticker(s3):
    """Regression guard: simulate relies on OHLCV bars being in date order.
    DataFrame producer (``build_ohlcv_df_by_ticker``) sorts on
    construction, so unsorted input would only happen via direct
    construction. This test asserts whatever index order the caller
    provides round-trips intact."""
    df = pd.DataFrame(
        {"close": [103.0, 101.0, 102.0]},
        index=pd.DatetimeIndex(["2026-01-03", "2026-01-01", "2026-01-02"]),
    )
    ohlcv = {"AAPL": df}
    key = save_ohlcv_by_ticker("b", "2026-04-23", "simulation_setup", "ohlcv",
                               ohlcv, s3_client=s3)
    loaded = load_ohlcv_by_ticker("b", key, s3_client=s3)
    assert [str(d.date()) for d in pd.DatetimeIndex(loaded["AAPL"].index)] == [
        "2026-01-03", "2026-01-01", "2026-01-02",
    ]


# ── Series round-trip (spy_prices) ───────────────────────────────────────────


def test_series_roundtrip(s3):
    dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    s = pd.Series([100.0, 101.0, 102.0], index=dates, name="SPY")
    key = save_series("b", "2026-04-23", "predictor_data_prep", "spy_prices",
                      s, s3_client=s3)
    loaded = load_series("b", key, s3_client=s3)
    pd.testing.assert_series_equal(loaded, s, check_freq=False)


def test_series_preserves_unnamed_as_value_column(s3):
    """Unnamed Series survive round-trip with a 'value' column."""
    s = pd.Series([1.0, 2.0, 3.0])
    key = save_series("b", "2026-04-23", "p", "n", s, s3_client=s3)
    loaded = load_series("b", key, s3_client=s3)
    assert list(loaded) == [1.0, 2.0, 3.0]


def test_load_series_rejects_multicolumn(s3):
    """Loading a multi-col parquet as a Series must fail loud."""
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    key = save_dataframe("b", "2026-04-23", "p", "n", df,
                         s3_client=s3, preserve_index=False)
    with pytest.raises(ValueError, match="expected 1"):
        load_series("b", key, s3_client=s3)


# ── Dict-of-Series round-trip (vwap_series_by_ticker) ────────────────────────


def test_dict_of_series_roundtrip(s3):
    data = {
        "AAPL": pd.Series([100.0, 101.0], index=["2026-01-01", "2026-01-02"]),
        "MSFT": pd.Series([200.0, 201.0, 202.0],
                          index=["2026-01-01", "2026-01-02", "2026-01-03"]),
    }
    key = save_dict_of_series("b", "2026-04-23", "predictor_feature_maps",
                              "vwap_series", data, s3_client=s3)
    loaded = load_dict_of_series("b", key, s3_client=s3)
    assert set(loaded.keys()) == set(data.keys())
    for ticker in data:
        pd.testing.assert_series_equal(
            loaded[ticker].astype(float).reset_index(drop=True),
            data[ticker].astype(float).reset_index(drop=True),
            check_names=False,
        )
        assert list(loaded[ticker].index) == list(data[ticker].index)


def test_dict_of_series_empty_dict(s3):
    key = save_dict_of_series("b", "2026-04-23", "p", "n", {}, s3_client=s3)
    loaded = load_dict_of_series("b", key, s3_client=s3)
    assert loaded == {}


def test_dict_of_series_skips_none_values(s3):
    """A None Series for a ticker must not crash the stacker."""
    data = {"AAPL": pd.Series([100.0], index=["2026-01-01"]), "MSFT": None}
    key = save_dict_of_series("b", "2026-04-23", "p", "n", data, s3_client=s3)
    loaded = load_dict_of_series("b", key, s3_client=s3)
    assert "AAPL" in loaded
    assert "MSFT" not in loaded


# ── Dict-of-DataFrames round-trip (features_by_ticker) ──────────────────────


def test_dict_of_dataframes_roundtrip(s3):
    data = {
        "AAPL": pd.DataFrame(
            {"feat1": [1.0, 2.0, 3.0], "feat2": [4.0, 5.0, 6.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        ),
        "MSFT": pd.DataFrame(
            {"feat1": [10.0, 20.0], "feat2": [40.0, 50.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02"]),
        ),
    }
    key = save_dict_of_dataframes("b", "2026-04-23", "predictor_data_prep",
                                  "features", data, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)

    assert set(loaded.keys()) == set(data.keys())
    for ticker in data:
        orig = data[ticker]
        got = loaded[ticker]
        # Columns preserved (plus no extras)
        assert set(got.columns) == set(orig.columns)
        # Row count preserved
        assert len(got) == len(orig)
        # Values preserved (column-wise; index may be materialized as dt64)
        for col in orig.columns:
            assert list(got[col].astype(float)) == list(orig[col].astype(float))


def test_dict_of_dataframes_empty_dict(s3):
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", {}, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)
    assert loaded == {}


def test_dict_of_dataframes_drops_empty_frames(s3):
    """A ticker with an empty DataFrame is dropped (avoids writing NaN rows)."""
    data = {
        "AAPL": pd.DataFrame({"f1": [1.0]}, index=[0]),
        "MSFT": pd.DataFrame(),
    }
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", data, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)
    assert "AAPL" in loaded
    assert "MSFT" not in loaded


def test_dict_of_dataframes_promotes_int_float_drift(s3):
    """Regression guard for the 2026-04-26 ``post_inference`` failure:
    ``predictor_data_prep`` aborted with ``Unable to merge: Field
    Volume has incompatible types: int64 vs double`` because 804
    tickers had int64 Volume and 107 had float64 (the float ones are
    recent listings whose early bars had NaN volumes that pandas
    auto-cast to float64).

    ``pa.unify_schemas`` doesn't auto-promote int→float; it raises.
    Save path now pre-casts on int-vs-float drift in any column,
    matching the wider float type. This test reproduces the drift
    pattern and asserts the save+load round-trips clean."""
    int_volume = pd.DataFrame(
        {"Close": [100.0, 101.0], "Volume": [1000, 2000]},  # int64
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    float_volume = pd.DataFrame(
        {"Close": [50.0, 51.0], "Volume": [3000.0, 4000.0]},  # float64
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    assert str(int_volume["Volume"].dtype) == "int64"
    assert str(float_volume["Volume"].dtype) == "float64"

    data = {"AAA": int_volume, "BBB": float_volume}
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", data, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)

    # Both tickers round-trip; both Volume columns are float64 on load
    # (the int instances were promoted on save).
    assert set(loaded.keys()) == {"AAA", "BBB"}
    assert str(loaded["AAA"]["Volume"].dtype) == "float64"
    assert str(loaded["BBB"]["Volume"].dtype) == "float64"
    # Values preserved (1000, 2000 → 1000.0, 2000.0)
    assert list(loaded["AAA"]["Volume"]) == [1000.0, 2000.0]
    assert list(loaded["BBB"]["Volume"]) == [3000.0, 4000.0]


def test_dict_of_dataframes_homogeneous_int_unchanged(s3):
    """When all frames have the same int dtype, no promotion happens —
    the column stays int on save+load. Avoids unnecessary widening
    when there's no drift."""
    int_only = {
        "AAA": pd.DataFrame(
            {"Volume": [1000, 2000]},
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        ),
        "BBB": pd.DataFrame(
            {"Volume": [3000, 4000]},
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        ),
    }
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", int_only, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)
    # Stays int — pyarrow round-trips int64 cleanly when all tickers
    # agree on the type.
    assert str(loaded["AAA"]["Volume"].dtype).startswith("int")
    assert str(loaded["BBB"]["Volume"].dtype).startswith("int")


def test_dict_of_dataframes_promotes_float32_float64_drift(s3):
    """Second-pass regression: 2026-04-26 v2 validation surfaced
    ``rsi_14: float vs double`` after the int64-vs-float64 fix. Same
    underlying problem (numeric-width drift), different column. The
    helper now handles all numeric drift, not just int↔float."""
    import numpy as np
    f32_frame = pd.DataFrame(
        {"rsi_14": np.array([50.0, 51.5], dtype=np.float32)},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    f64_frame = pd.DataFrame(
        {"rsi_14": np.array([60.0, 61.5], dtype=np.float64)},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    assert str(f32_frame["rsi_14"].dtype) == "float32"
    assert str(f64_frame["rsi_14"].dtype) == "float64"

    data = {"AAA": f32_frame, "BBB": f64_frame}
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", data, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)

    assert set(loaded.keys()) == {"AAA", "BBB"}
    assert str(loaded["AAA"]["rsi_14"].dtype) == "float64"
    assert str(loaded["BBB"]["rsi_14"].dtype) == "float64"
    # Float32→float64 introduces tiny rep error; tolerance check:
    import numpy as np
    np.testing.assert_allclose(loaded["AAA"]["rsi_14"], [50.0, 51.5], rtol=1e-6)
    np.testing.assert_allclose(loaded["BBB"]["rsi_14"], [60.0, 61.5], rtol=1e-9)


def test_dict_of_dataframes_column_order_drift_handled(s3):
    """Third regression in the c5.large validation arc: with dtype
    drift fixed, the 2026-04-26 v4 spot then hit ``Target schema's
    field names are not matching the table's field names`` because
    different ticker DataFrames had the same column SET but different
    ORDER. ``pa.Table.cast(schema)`` requires order match; reindex
    pass before pyarrow conversion locks every frame to a stable
    union ordering.

    Test: two tickers with identical columns in different orders +
    one ticker missing a column. After save+load, all three should
    round-trip clean with NaN-fill on the missing column."""
    aaa = pd.DataFrame(
        {"momentum_5d": [0.01, 0.02], "rsi_14": [50.0, 51.5],
         "rel_volume_ratio": [1.0, 1.1]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    bbb = pd.DataFrame(
        # Same columns, different order
        {"rsi_14": [60.0, 61.5], "rel_volume_ratio": [0.9, 1.0],
         "momentum_5d": [0.03, 0.04]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    ccc = pd.DataFrame(
        # Subset: missing rel_volume_ratio
        {"momentum_5d": [0.05, 0.06], "rsi_14": [70.0, 71.5]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    data = {"AAA": aaa, "BBB": bbb, "CCC": ccc}
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", data, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)

    assert set(loaded.keys()) == {"AAA", "BBB", "CCC"}
    # All three frames have all three columns now (CCC's missing one
    # comes back as NaN-filled)
    for ticker in ("AAA", "BBB", "CCC"):
        assert set(loaded[ticker].columns) == {
            "momentum_5d", "rsi_14", "rel_volume_ratio",
        }, f"{ticker} columns: {list(loaded[ticker].columns)}"

    # Values preserved on populated cols; NaN on the missing one
    assert list(loaded["AAA"]["momentum_5d"]) == [0.01, 0.02]
    assert list(loaded["BBB"]["rsi_14"]) == [60.0, 61.5]
    assert list(loaded["CCC"]["momentum_5d"]) == [0.05, 0.06]
    assert loaded["CCC"]["rel_volume_ratio"].isna().all()


def test_dict_of_dataframes_homogeneous_columns_no_reindex(s3):
    """No-op fast path: when every frame has identical column order,
    skip the reindex (no unnecessary allocations)."""
    same_order = {
        "AAA": pd.DataFrame(
            {"a": [1.0, 2.0], "b": [3.0, 4.0]},
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        ),
        "BBB": pd.DataFrame(
            {"a": [5.0, 6.0], "b": [7.0, 8.0]},  # same column order
            index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
        ),
    }
    key = save_dict_of_dataframes("b", "2026-04-23", "p", "n", same_order, s3_client=s3)
    loaded = load_dict_of_dataframes("b", key, s3_client=s3)
    assert list(loaded["AAA"]["a"]) == [1.0, 2.0]
    assert list(loaded["BBB"]["b"]) == [7.0, 8.0]


def test_dict_of_dataframes_object_vs_numeric_still_raises(s3):
    """Non-numeric drift is intentionally left alone — string-vs-int
    or datetime-vs-object are real semantic divergences that should
    still surface as a hard error (not silent coercion).

    Post 2026-04-26 v5 stream-write: the error surface shifted from
    pa.unify_schemas's ``Unable to merge`` to pa.Table.from_pandas's
    ``Conversion failed for column ... with type ...`` because we
    no longer materialize all tables before unifying schemas. Either
    error means the operator must investigate; the test asserts the
    failure happens, not the specific message."""
    str_frame = pd.DataFrame(
        {"col": ["a", "b"]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    int_frame = pd.DataFrame(
        {"col": [1, 2]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"]),
    )
    data = {"AAA": str_frame, "BBB": int_frame}
    with pytest.raises(Exception, match="(?i)Unable to merge|incompatible|Conversion failed|expected.*dtype"):
        save_dict_of_dataframes("b", "2026-04-23", "p", "n", data, s3_client=s3)
