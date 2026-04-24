"""
tests/test_inference_vectorization_parity.py — byte-for-byte parity
between the legacy dict-of-dicts inference path and the new tensor
path (PR "perf/vectorize-inference-hotloops").

Two inner loops were collapsed into a single shared tensor builder:
  - synthetic/predictor_backtest.py::run_inference
  - optimizer/predictor_optimizer.py::_run_variant_inference

Both used a 911-entry dict-of-dicts with a ~2.28M-inner-tick per-date
loop that Python iterated serially. The tensor path pre-builds a
(n_dates, n_tickers, n_features) float32 array once, then slices per
date (O(n_dates) vectorized work, no inner Python loop).

The legacy `_zero_out_features` helper was removed; its semantics are
now expressed via ``run_inference(zero_features=...)``. Parity coverage:
  - Same predictions across shapes
  - zero_features kwarg reproduces legacy _zero_out_features semantics
  - NaN-in-zeroed-column edge case: legacy → 0.0 re-admits the row;
    tensor path must do the same

These tests use a mock scorer so they run offline without GBM weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from synthetic.predictor_backtest import (
    build_inference_tensor,
    _predict_from_tensor,
)


class _MockScorer:
    """Deterministic row-sum scorer. Per-ticker prediction is the sum
    of that ticker's feature vector — stable, sensitive to every input,
    and trivially verifiable by hand."""

    def __init__(self, feature_names: list[str]):
        self.feature_names = feature_names

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError("use constructor directly in tests")

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert X.ndim == 2, f"expected 2D, got {X.shape}"
        return X.sum(axis=1)


# ── Reference implementations (legacy paths, inlined) ───────────────────────


def _legacy_run_inference(
    features_by_ticker: dict[str, pd.DataFrame],
    feature_names: list[str],
    scorer: _MockScorer,
    trading_dates: list[str],
) -> dict[str, dict[str, float]]:
    """Re-impl of the pre-refactor inference path — dict-of-dicts + per-
    date Python loop over all tickers. Used as the parity reference."""
    feature_arrays: dict[str, dict[str, np.ndarray]] = {}
    for ticker, featured_df in features_by_ticker.items():
        try:
            arr = featured_df[feature_names].to_numpy(dtype=np.float32)
            dates = featured_df.index.strftime("%Y-%m-%d")
            feature_arrays[ticker] = dict(zip(dates, arr))
        except (KeyError, ValueError):
            continue

    predictions_by_date: dict[str, dict[str, float]] = {}
    for date_str in trading_dates:
        tickers_batch, vectors_batch = [], []
        for ticker, date_to_vec in feature_arrays.items():
            vec = date_to_vec.get(date_str)
            if vec is not None and not np.any(np.isnan(vec)):
                tickers_batch.append(ticker)
                vectors_batch.append(vec)
        if not vectors_batch:
            continue
        X = np.stack(vectors_batch)
        alphas = scorer.predict(X)
        predictions_by_date[date_str] = {
            ticker: float(alpha)
            for ticker, alpha in zip(tickers_batch, alphas)
        }
    return predictions_by_date


def _legacy_zero_out_features(
    features_by_ticker: dict[str, pd.DataFrame],
    noise_features: list[str],
) -> dict[str, pd.DataFrame]:
    """Re-impl of the removed ``_zero_out_features`` helper."""
    result = {}
    for ticker, df in features_by_ticker.items():
        df_copy = df.copy()
        for feat in noise_features:
            if feat in df_copy.columns:
                df_copy[feat] = 0.0
        result[ticker] = df_copy
    return result


# ── Fixtures ─────────────────────────────────────────────────────────────────


FEATURES = ["rsi_14", "momentum_20d", "atr_14"]


def _build_fixture(rng_seed: int = 42) -> dict[str, pd.DataFrame]:
    """3 tickers × 10 dates × 3 features, deterministic values."""
    rng = np.random.default_rng(rng_seed)
    dates = pd.to_datetime(
        [f"2026-04-{d:02d}" for d in range(1, 11)]
    )
    tickers = ["AAPL", "MSFT", "GOOGL"]
    return {
        t: pd.DataFrame(
            rng.standard_normal((len(dates), len(FEATURES))).astype(np.float32),
            index=dates,
            columns=FEATURES,
        )
        for t in tickers
    }


def _run_new_path(
    features_by_ticker: dict[str, pd.DataFrame],
    feature_names: list[str],
    scorer: _MockScorer,
    trading_dates: list[str],
    zero_features: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Exercise the tensor path the way ``run_inference`` does."""
    tensor, tickers, date_to_idx = build_inference_tensor(
        features_by_ticker, feature_names,
    )
    if zero_features:
        zero_idx = [
            feature_names.index(f) for f in zero_features if f in feature_names
        ]
        if zero_idx:
            tensor[:, :, zero_idx] = 0.0
    return _predict_from_tensor(
        tensor, tickers, date_to_idx, trading_dates,
        scorer=scorer, heartbeat_every=10_000, log_label="parity_test",
    )


# ── Parity tests ─────────────────────────────────────────────────────────────


def _assert_predictions_equal(
    actual: dict[str, dict[str, float]],
    expected: dict[str, dict[str, float]],
    tol: float = 1e-6,
) -> None:
    assert set(actual.keys()) == set(expected.keys()), (
        f"date key mismatch: {set(actual)} vs {set(expected)}"
    )
    for d in expected:
        assert set(actual[d].keys()) == set(expected[d].keys()), (
            f"ticker set mismatch on {d}: {set(actual[d])} vs {set(expected[d])}"
        )
        for t in expected[d]:
            assert abs(actual[d][t] - expected[d][t]) <= tol, (
                f"value mismatch on {d}/{t}: {actual[d][t]} vs {expected[d][t]}"
            )


def test_parity_clean_features_no_nan():
    features = _build_fixture()
    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)


def test_parity_with_nan_excludes_same_rows():
    features = _build_fixture()
    # Inject NaN on a specific (date, ticker) — both paths must exclude it
    features["MSFT"].iloc[2, 0] = np.nan  # 2026-04-03 / MSFT / rsi_14
    features["GOOGL"].iloc[5, 2] = np.nan  # 2026-04-06 / GOOGL / atr_14

    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)

    # Sanity: the NaN'd (date, ticker) should be missing from both
    assert "MSFT" not in new.get("2026-04-03", {})
    assert "GOOGL" not in new.get("2026-04-06", {})
    # Other tickers on those dates still present
    assert "AAPL" in new["2026-04-03"]
    assert "AAPL" in new["2026-04-06"]


def test_parity_all_nan_date_dropped():
    features = _build_fixture()
    # NaN out all tickers on 2026-04-05
    for t in features:
        features[t].iloc[4, :] = np.nan

    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)
    assert "2026-04-05" not in new
    assert "2026-04-05" not in legacy


def test_parity_ticker_missing_feature_column_is_skipped():
    features = _build_fixture()
    # AAPL drops a feature column — should be excluded entirely in both paths
    features["AAPL"] = features["AAPL"].drop(columns=["atr_14"])

    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)
    for d in new:
        assert "AAPL" not in new[d]


def test_parity_date_outside_tensor_range_is_skipped():
    features = _build_fixture()
    # Trading date 2026-04-15 has no data — both paths skip it
    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)] + ["2026-04-15"]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)
    assert "2026-04-15" not in new
    assert "2026-04-15" not in legacy


def test_parity_duplicate_dates_keep_last():
    """Legacy used dict(zip(dates, arr)) which keeps the last value on
    duplicate keys. Tensor builder must do the same via
    ``df.index.duplicated(keep='last')``."""
    features = _build_fixture()
    # Inject a duplicate date for MSFT with a distinct feature vector
    dup_row = pd.DataFrame(
        [[999.0, 999.0, 999.0]], columns=FEATURES,
        index=pd.to_datetime(["2026-04-03"]),
    )
    features["MSFT"] = pd.concat([features["MSFT"], dup_row])

    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    legacy = _legacy_run_inference(features, FEATURES, scorer, trading_dates)
    new = _run_new_path(features, FEATURES, scorer, trading_dates)

    _assert_predictions_equal(new, legacy)
    # The last (999.0) vector wins on 2026-04-03
    assert new["2026-04-03"]["MSFT"] == pytest.approx(999.0 * 3)


# ── zero_features parity ─────────────────────────────────────────────────────


def test_zero_features_matches_legacy_zero_out_features():
    features = _build_fixture()
    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)
    noise = ["rsi_14", "not_a_real_feature"]  # mixed valid + invalid

    # Legacy path: _zero_out_features → run_inference
    zeroed = _legacy_zero_out_features(features, noise)
    legacy = _legacy_run_inference(zeroed, FEATURES, scorer, trading_dates)

    # New path: run_inference(zero_features=noise)
    new = _run_new_path(
        features, FEATURES, scorer, trading_dates,
        zero_features=noise,
    )

    _assert_predictions_equal(new, legacy)

    # Sanity: sums reflect zeroed rsi_14
    for date, tpreds in new.items():
        for ticker, pred in tpreds.items():
            expected = features[ticker].loc[date, ["momentum_20d", "atr_14"]].sum()
            assert pred == pytest.approx(float(expected), abs=1e-5)


def test_zero_features_readmits_nan_row_same_as_legacy():
    """Edge case: if a row has NaN ONLY in a zeroed column, the legacy
    path's _zero_out_features turns the NaN to 0 → row becomes valid →
    included in predictions. The tensor path must match."""
    features = _build_fixture()
    # Inject NaN in rsi_14 only (zeroable column) on MSFT/2026-04-04
    features["MSFT"].iloc[3, 0] = np.nan

    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)
    noise = ["rsi_14"]

    # Legacy
    zeroed = _legacy_zero_out_features(features, noise)
    legacy = _legacy_run_inference(zeroed, FEATURES, scorer, trading_dates)

    # New
    new = _run_new_path(
        features, FEATURES, scorer, trading_dates,
        zero_features=noise,
    )

    _assert_predictions_equal(new, legacy)
    # MSFT/2026-04-04 should be admitted (the zeroed NaN is now 0.0)
    assert "MSFT" in new["2026-04-04"]


def test_zero_features_empty_list_is_noop():
    features = _build_fixture()
    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    baseline = _run_new_path(features, FEATURES, scorer, trading_dates)
    with_empty = _run_new_path(
        features, FEATURES, scorer, trading_dates,
        zero_features=[],
    )
    _assert_predictions_equal(with_empty, baseline)


def test_zero_features_all_unknown_is_noop():
    features = _build_fixture()
    trading_dates = [f"2026-04-{d:02d}" for d in range(1, 11)]
    scorer = _MockScorer(FEATURES)

    baseline = _run_new_path(features, FEATURES, scorer, trading_dates)
    with_unknown = _run_new_path(
        features, FEATURES, scorer, trading_dates,
        zero_features=["not_real", "also_not_real"],
    )
    _assert_predictions_equal(with_unknown, baseline)


# ── build_inference_tensor direct tests ──────────────────────────────────────


def test_builder_returns_correct_shape_and_ordering():
    features = _build_fixture()
    tensor, tickers, date_to_idx = build_inference_tensor(features, FEATURES)

    assert tensor.shape == (10, 3, 3)
    assert tensor.dtype == np.float32
    assert tickers == ["AAPL", "MSFT", "GOOGL"]
    assert date_to_idx == {f"2026-04-{d:02d}": i for i, d in enumerate(range(1, 11))}


def test_builder_empty_input_returns_empty_tensor():
    tensor, tickers, date_to_idx = build_inference_tensor({}, FEATURES)
    assert tensor.shape == (0, 0, 3)
    assert tickers == []
    assert date_to_idx == {}


def test_builder_all_tickers_missing_features_returns_empty():
    df = pd.DataFrame({"unrelated_col": [1.0, 2.0]}, index=pd.to_datetime(["2026-04-01", "2026-04-02"]))
    tensor, tickers, date_to_idx = build_inference_tensor({"AAPL": df}, FEATURES)
    assert tensor.shape == (0, 0, 3)
    assert tickers == []
