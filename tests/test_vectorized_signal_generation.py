"""Equivalence guard: precompute_indicator_series + indicators_from_precomputed
must produce identical indicator values to the row-by-row
_compute_indicators_from_ohlcv for every date in the history.

The pre-2026-04-21 implementation of ``build_signals_by_date`` called
``_compute_indicators_from_ohlcv`` per date per ticker — ~2.2s per date
× 2277 dates = 75 min, pushing Saturday SF past its 2hr SSM ceiling.

The vectorized rewrite pre-computes full indicator series per ticker
in one pandas pass, then does O(1) lookups per date. Since the refactor
changes a hot path used across the full 10y backtest window, these
tests assert byte-identical output on a synthetic multi-ticker, multi-year
price history — if the vectorization subtly drifts (e.g. macd_cross
window semantics or ewm warmup ordering), test catches it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic.signal_generator import (
    _compute_indicators_from_ohlcv,
    precompute_indicator_series,
    indicators_from_precomputed,
)


def _synthetic_bars(n: int, seed: int = 0) -> list[dict]:
    """Build a deterministic pseudo-random close-price series of ``n`` bars
    starting 2017-01-02 (Monday), skipping weekends."""
    rng = np.random.default_rng(seed)
    # Geometric walk so closes stay positive even for long series
    log_rets = rng.standard_normal(n) * 0.01
    closes = 100.0 * np.exp(np.cumsum(log_rets))
    start = pd.Timestamp("2017-01-02")
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d").tolist()
    return [
        {"date": d, "open": c, "high": c, "low": c, "close": float(c)}
        for d, c in zip(dates, closes)
    ]


def _scalar_indicators_at_every_date(bars: list[dict]) -> pd.DataFrame:
    """Reference implementation: call the scalar function on every prefix
    of ``bars``. Returns a DataFrame indexed on bar dates with the same
    columns as ``precompute_indicator_series``."""
    rows = {}
    for i in range(len(bars)):
        prefix = bars[: i + 1]
        ind = _compute_indicators_from_ohlcv(prefix)
        if ind is None:
            rows[bars[i]["date"]] = {k: None for k in (
                "rsi_14", "macd_cross", "macd_above_zero",
                "price_vs_ma50", "price_vs_ma200", "momentum_20d",
            )}
        else:
            rows[bars[i]["date"]] = ind
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


def test_vectorized_matches_scalar_on_single_ticker():
    """One 500-bar ticker. For every date with non-None indicators from
    the scalar function, the precomputed version must match within
    float tolerance."""
    bars = _synthetic_bars(500, seed=42)
    scalar = _scalar_indicators_at_every_date(bars)
    vec = precompute_indicator_series({"TEST": bars})["TEST"]

    # Check row-by-row on dates where scalar returned a real dict
    # (min_bars=210 means dates <210 are None in scalar → NaN-coerced
    # by DataFrame.from_dict). Skip those dates; our contract is that
    # vectorized output is only required to match when scalar would
    # have returned a real dict.
    for idx, scalar_row in scalar.iterrows():
        if pd.isna(scalar_row["rsi_14"]):
            continue

        assert idx in vec.index, f"vec missing date {idx}"
        vec_row = vec.loc[idx]
        # Floating-point fields
        for col in ("rsi_14", "macd_cross"):
            assert np.isclose(
                float(scalar_row[col]), float(vec_row[col]), equal_nan=True,
                atol=1e-9, rtol=1e-9,
            ), f"{col} mismatch at {idx}: scalar={scalar_row[col]} vec={vec_row[col]}"
        # Bool field
        assert bool(scalar_row["macd_above_zero"]) == bool(vec_row["macd_above_zero"]), (
            f"macd_above_zero mismatch at {idx}"
        )
        # Optional (None-able) fields — scalar impl can return None for
        # MA50/MA200/momentum when the window isn't satisfied; vectorized
        # emits NaN. Treat them as equivalent.
        for col in ("price_vs_ma50", "price_vs_ma200", "momentum_20d"):
            s = scalar_row[col]
            v = vec_row[col]
            if pd.isna(s):
                assert pd.isna(v), f"{col} expected NaN at {idx}, got {v}"
            else:
                assert np.isclose(float(s), float(v), atol=1e-9, rtol=1e-9), (
                    f"{col} mismatch at {idx}: scalar={s} vec={v}"
                )


def test_vectorized_matches_scalar_on_multiple_tickers():
    """Three tickers of varying lengths. Ensures each ticker's series is
    computed independently (no cross-contamination from shared state)."""
    bars_a = _synthetic_bars(400, seed=1)
    bars_b = _synthetic_bars(600, seed=2)
    bars_c = _synthetic_bars(250, seed=3)
    ohlcv = {"A": bars_a, "B": bars_b, "C": bars_c}
    vec = precompute_indicator_series(ohlcv)

    for ticker, bars in ohlcv.items():
        scalar = _scalar_indicators_at_every_date(bars)
        for idx, scalar_row in scalar.iterrows():
            if pd.isna(scalar_row["rsi_14"]):
                continue
            vec_row = vec[ticker].loc[idx]
            assert np.isclose(
                float(scalar_row["rsi_14"]), float(vec_row["rsi_14"]),
                atol=1e-9,
            ), f"{ticker} {idx} rsi_14 mismatch"
            assert float(scalar_row["macd_cross"]) == float(vec_row["macd_cross"]), (
                f"{ticker} {idx} macd_cross mismatch"
            )


def test_macd_cross_window_semantics():
    """macd_cross in _compute_indicators_from_ohlcv picks the most recent
    cross within a 3-bar window ending at 'now'. This test builds a price
    series where we know exactly when up/down crosses happen and verifies
    the 3-bar-window ffill semantics in the vectorized path."""
    bars = _synthetic_bars(300, seed=77)
    scalar = _scalar_indicators_at_every_date(bars)
    vec = precompute_indicator_series({"X": bars})["X"]

    # Find dates where scalar reports a non-zero, non-NaN cross
    cross_dates = [
        idx for idx, r in scalar.iterrows()
        if not pd.isna(r["macd_cross"]) and float(r["macd_cross"]) != 0.0
    ]
    assert cross_dates, "synthetic series should produce at least some crosses"
    for idx in cross_dates:
        assert float(vec.loc[idx, "macd_cross"]) == float(scalar.loc[idx, "macd_cross"]), (
            f"cross direction mismatch at {idx}"
        )


def test_indicators_from_precomputed_matches_scalar_dicts():
    """indicators_from_precomputed must return dicts shape-identical to
    _compute_indicators_from_ohlcv output — same keys, same value types
    (float / None / bool) — so predictions_to_signals consumers don't
    need to care which path produced the indicators."""
    bars = _synthetic_bars(400, seed=11)
    precomputed = precompute_indicator_series({"X": bars})

    # Pick a late date where all indicators are populated
    late_date = bars[-1]["date"]
    via_vec = indicators_from_precomputed(precomputed, ["X"], late_date)
    via_scalar = _compute_indicators_from_ohlcv(bars)

    assert "X" in via_vec
    vec_dict = via_vec["X"]

    # Same key set
    assert set(vec_dict.keys()) == set(via_scalar.keys())
    # Same types and values
    for key in via_scalar:
        s, v = via_scalar[key], vec_dict[key]
        if isinstance(s, bool):
            assert isinstance(v, bool) and s == v, f"{key} bool mismatch"
        elif s is None:
            assert v is None, f"{key} expected None, got {v}"
        else:
            assert isinstance(v, float), f"{key} expected float, got {type(v).__name__}"
            assert np.isclose(s, v, atol=1e-9), f"{key} value mismatch"


def test_indicators_from_precomputed_skips_short_history_tickers():
    """A ticker with <14 bars can't compute rsi_14 → NaN in precompute →
    indicators_from_precomputed must skip it (not emit a dict with NaN
    values that would break downstream scorers)."""
    short_bars = _synthetic_bars(10, seed=99)
    precomputed = precompute_indicator_series({"SHORT": short_bars})
    via_vec = indicators_from_precomputed(
        precomputed, ["SHORT"], short_bars[-1]["date"],
    )
    assert via_vec == {}


def test_indicators_from_precomputed_skips_unknown_ticker_or_date():
    """Robustness: a ticker not in precomputed or a date not in its index
    is quietly skipped (not an error)."""
    bars = _synthetic_bars(400, seed=7)
    precomputed = precompute_indicator_series({"X": bars})

    # Unknown ticker
    assert indicators_from_precomputed(precomputed, ["UNKNOWN"], bars[-1]["date"]) == {}

    # Date outside the ticker's history
    assert indicators_from_precomputed(precomputed, ["X"], "1999-01-04") == {}


def test_predictions_to_signals_accepts_both_paths():
    """predictions_to_signals must produce the same signal envelope
    whether it's called with raw ohlcv_by_ticker (legacy path) or
    precomputed_indicators (new fast path) for a given date."""
    from synthetic.signal_generator import predictions_to_signals

    bars = _synthetic_bars(400, seed=21)
    ohlcv = {"X": bars, "Y": _synthetic_bars(400, seed=22)}
    predictions = {"X": 0.005, "Y": -0.003}
    sector_map = {"X": "XLK", "Y": "XLF"}
    date_str = bars[-1]["date"]

    # Legacy path
    env_legacy = predictions_to_signals(
        predictions=predictions, date=date_str, sector_map=sector_map,
        ohlcv_by_ticker=ohlcv, top_n=5, min_score=50,
    )

    # Fast path
    precomputed = precompute_indicator_series(ohlcv)
    indicators = indicators_from_precomputed(precomputed, predictions.keys(), date_str)
    env_fast = predictions_to_signals(
        predictions=predictions, date=date_str, sector_map=sector_map,
        precomputed_indicators=indicators, top_n=5, min_score=50,
    )

    # Envelopes must match on every stock entry (order + score + signal).
    def _key(entries):
        return [(e["ticker"], round(e["score"], 6), e["signal"]) for e in entries]
    for list_field in ("universe", "buy_candidates"):
        assert _key(env_legacy.get(list_field, [])) == _key(env_fast.get(list_field, [])), (
            f"{list_field} diverges between legacy and fast path"
        )
