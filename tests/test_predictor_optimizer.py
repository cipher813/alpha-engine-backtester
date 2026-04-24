"""Tests for optimizer/predictor_optimizer.py — Phase 4 predictor feedback."""

import json
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import pytest

from optimizer.predictor_optimizer import (
    _discover_model_variants,
    _pick_best_mode,
    _load_noise_candidates,
    _filter_predictions_by_alpha,
    apply_recommendations,
    _MIN_SHARPE_IMPROVEMENT,
    _MIN_TRADING_DAYS_5Y,
)


# ── _pick_best_mode tests ───────────────────────────────────────────────────

def test_pick_best_mode_baseline_wins():
    baseline = {"sharpe_ratio": 1.5}
    variants = {"mse": {"sharpe_ratio": 1.4}, "rank": {"sharpe_ratio": 1.3}}
    result = _pick_best_mode(baseline, variants, has_sufficient_data=True)
    assert result["recommended_mode"] is None
    assert "baseline" in result["recommendation_reason"].lower()


def test_pick_best_mode_variant_wins_with_sufficient_improvement():
    baseline = {"sharpe_ratio": 1.0}
    variants = {"mse": {"sharpe_ratio": 1.2}}  # 20% improvement > 10% threshold
    result = _pick_best_mode(baseline, variants, has_sufficient_data=True)
    assert result["recommended_mode"] == "mse"
    assert "20" in result["recommendation_reason"]


def test_pick_best_mode_variant_below_threshold():
    baseline = {"sharpe_ratio": 1.0}
    variants = {"mse": {"sharpe_ratio": 1.05}}  # 5% improvement < 10% threshold
    result = _pick_best_mode(baseline, variants, has_sufficient_data=True)
    assert result["recommended_mode"] is None
    assert "below" in result["recommendation_reason"].lower()


def test_pick_best_mode_insufficient_data():
    baseline = {"sharpe_ratio": 1.0}
    variants = {"mse": {"sharpe_ratio": 1.5}}  # clearly better
    result = _pick_best_mode(baseline, variants, has_sufficient_data=False)
    assert result["recommended_mode"] is None
    assert "insufficient" in result["recommendation_reason"].lower()


def test_pick_best_mode_picks_best_among_variants():
    baseline = {"sharpe_ratio": 1.0}
    variants = {
        "mse": {"sharpe_ratio": 1.15},
        "rank": {"sharpe_ratio": 1.25},
        "catboost": {"sharpe_ratio": 1.20},
    }
    result = _pick_best_mode(baseline, variants, has_sufficient_data=True)
    assert result["recommended_mode"] == "rank"


def test_pick_best_mode_skips_error_variants():
    baseline = {"sharpe_ratio": 1.0}
    variants = {
        "mse": {"status": "error", "error": "download failed"},
        "rank": {"sharpe_ratio": 1.3},
    }
    result = _pick_best_mode(baseline, variants, has_sufficient_data=True)
    assert result["recommended_mode"] == "rank"


# ── _discover_model_variants tests ──────────────────────────────────────────

@patch("optimizer.predictor_optimizer.boto3")
def test_discover_model_variants_finds_mse_and_rank(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3

    # mse exists, rank exists, catboost doesn't
    def head_object(Bucket, Key):
        if "catboost" in Key:
            raise Exception("Not found")
    s3.head_object.side_effect = head_object

    available = _discover_model_variants("bucket")
    assert "mse" in available
    assert "rank" in available
    assert "catboost" not in available


@patch("optimizer.predictor_optimizer.boto3")
def test_discover_model_variants_none_found(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    s3.head_object.side_effect = Exception("Not found")

    available = _discover_model_variants("bucket")
    assert available == {}


# ── _load_noise_candidates tests ─────────────────────────────────────────────

@patch("optimizer.predictor_optimizer.boto3")
def test_load_noise_candidates(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3

    summary = {"noise_candidates": ["rsi_7", "vol_20d"], "feature_ics": {}}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(summary).encode())}

    result = _load_noise_candidates("bucket")
    assert result == ["rsi_7", "vol_20d"]


@patch("optimizer.predictor_optimizer.boto3")
def test_load_noise_candidates_empty(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3

    summary = {"noise_candidates": []}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(summary).encode())}

    result = _load_noise_candidates("bucket")
    assert result == []


@patch("optimizer.predictor_optimizer.boto3")
def test_load_noise_candidates_missing_key(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    s3.get_object.side_effect = Exception("Not found")

    result = _load_noise_candidates("bucket")
    assert result == []


# ── apply_recommendations tests ──────────────────────────────────────────────

@patch("optimizer.predictor_optimizer.boto3")
def test_apply_recommendations_ensemble_only(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3

    # No existing params
    s3.get_object.side_effect = Exception("Not found")

    ensemble_result = {
        "recommended_mode": "mse",
        "date": "2026-04-07",
        "recommendation_reason": "better Sharpe",
    }

    with patch("optimizer.rollback.save_previous"):
        result = apply_recommendations(ensemble_result, None, "bucket")

    assert result["applied"] is True
    assert "preferred_ensemble_mode" in result["updates"]

    # Verify the written payload
    put_calls = [c for c in s3.put_object.call_args_list if "predictor_params.json" in str(c)]
    assert len(put_calls) >= 1
    written = json.loads(put_calls[0].kwargs["Body"])
    assert written["preferred_ensemble_mode"] == "mse"


@patch("optimizer.predictor_optimizer.boto3")
def test_apply_recommendations_merges_with_existing(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3

    # Existing veto threshold
    existing = {"veto_confidence": 0.75, "updated_at": "2026-04-06"}
    s3.get_object.return_value = {"Body": MagicMock(read=lambda: json.dumps(existing).encode())}

    pruning_result = {
        "recommend_pruning": True,
        "prune_features": ["rsi_7"],
        "date": "2026-04-07",
        "recommendation_reason": "Sharpe held",
    }

    with patch("optimizer.rollback.save_previous"):
        result = apply_recommendations(None, pruning_result, "bucket")

    assert result["applied"] is True
    put_calls = [c for c in s3.put_object.call_args_list if "predictor_params.json" in str(c)]
    written = json.loads(put_calls[0].kwargs["Body"])
    # Should preserve existing veto_confidence
    assert written["veto_confidence"] == 0.75
    assert written["prune_features"] == ["rsi_7"]


@patch("optimizer.predictor_optimizer.boto3")
def test_apply_recommendations_no_recommendations(mock_boto):
    result = apply_recommendations(
        {"recommended_mode": None},
        {"recommend_pruning": False},
        "bucket",
    )
    assert result["applied"] is False
    assert "no_recommendations" in result["reason"]


@patch("optimizer.predictor_optimizer.boto3")
def test_apply_recommendations_none_inputs(mock_boto):
    result = apply_recommendations(None, None, "bucket")
    assert result["applied"] is False


@patch("optimizer.predictor_optimizer.boto3")
def test_apply_recommendations_signal_threshold(mock_boto):
    s3 = MagicMock()
    mock_boto.client.return_value = s3
    s3.get_object.side_effect = Exception("Not found")

    threshold_result = {
        "recommended_signal_threshold": 0.015,
        "date": "2026-04-07",
        "recommendation_reason": "better Sharpe",
    }

    with patch("optimizer.rollback.save_previous"):
        result = apply_recommendations(None, None, "bucket", threshold_result=threshold_result)

    assert result["applied"] is True
    assert "recommended_signal_threshold" in result["updates"]
    put_calls = [c for c in s3.put_object.call_args_list if "predictor_params.json" in str(c)]
    written = json.loads(put_calls[0].kwargs["Body"])
    assert written["recommended_signal_threshold"] == 0.015


# ── _filter_predictions_by_alpha tests ───────────────────────────────────────

def test_filter_predictions_no_filtering_at_zero():
    preds = {"2026-04-01": {"AAPL": 0.03, "MSFT": -0.01}}
    result = _filter_predictions_by_alpha(preds, 0.0)
    assert result is preds  # no copy, same object


def test_filter_predictions_removes_below_threshold():
    preds = {
        "2026-04-01": {"AAPL": 0.03, "MSFT": 0.005, "GOOGL": -0.01},
        "2026-04-02": {"AAPL": 0.01, "MSFT": 0.02},
    }
    result = _filter_predictions_by_alpha(preds, 0.015)

    # Day 1: only AAPL survives (0.03 >= 0.015)
    assert "AAPL" in result["2026-04-01"]
    assert "MSFT" not in result["2026-04-01"]
    assert "GOOGL" not in result["2026-04-01"]

    # Day 2: only MSFT survives (0.02 >= 0.015)
    assert "MSFT" in result["2026-04-02"]
    assert "AAPL" not in result["2026-04-02"]


def test_filter_predictions_drops_empty_dates():
    preds = {
        "2026-04-01": {"AAPL": 0.001, "MSFT": 0.002},  # all below
        "2026-04-02": {"AAPL": 0.05},  # survives
    }
    result = _filter_predictions_by_alpha(preds, 0.01)

    assert "2026-04-01" not in result
    assert "2026-04-02" in result


def test_filter_predictions_exact_threshold():
    preds = {"2026-04-01": {"AAPL": 0.02}}
    result = _filter_predictions_by_alpha(preds, 0.02)
    assert "AAPL" in result["2026-04-01"]


def test_filter_predictions_all_filtered_out():
    preds = {"2026-04-01": {"AAPL": 0.01, "MSFT": 0.005}}
    result = _filter_predictions_by_alpha(preds, 0.05)
    assert result == {}
