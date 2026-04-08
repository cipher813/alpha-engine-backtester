"""Tests for analysis/classification_metrics.py."""

import pytest

from analysis.classification_metrics import compute_binary_metrics, compute_from_boolean_arrays


class TestComputeBinaryMetrics:
    def test_perfect_classifier(self):
        result = compute_binary_metrics(tp=50, fp=0, fn=0, tn=50)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["accuracy"] == 1.0

    def test_coin_flip(self):
        result = compute_binary_metrics(tp=25, fp=25, fn=25, tn=25)
        assert result["precision"] == 0.5
        assert result["recall"] == 0.5
        assert result["f1"] == 0.5
        assert result["accuracy"] == 0.5

    def test_no_positives_predicted(self):
        result = compute_binary_metrics(tp=0, fp=0, fn=30, tn=70)
        assert result["precision"] is None
        assert result["recall"] == 0.0
        assert result["f1"] is None

    def test_no_actual_positives(self):
        result = compute_binary_metrics(tp=0, fp=20, fn=0, tn=80)
        assert result["precision"] == 0.0
        assert result["recall"] is None

    def test_high_precision_low_recall(self):
        result = compute_binary_metrics(tp=5, fp=0, fn=45, tn=50)
        assert result["precision"] == 1.0
        assert result["recall"] == 0.1

    def test_counts(self):
        result = compute_binary_metrics(tp=10, fp=5, fn=3, tn=82)
        assert result["tp"] == 10
        assert result["fp"] == 5
        assert result["fn"] == 3
        assert result["tn"] == 82
        assert result["n"] == 100

    def test_f1_harmonic_mean(self):
        result = compute_binary_metrics(tp=10, fp=10, fn=5, tn=75)
        p = 10 / 20
        r = 10 / 15
        expected_f1 = 2 * p * r / (p + r)
        assert result["f1"] == pytest.approx(round(expected_f1, 4))


class TestComputeFromBooleanArrays:
    def test_simple(self):
        selected = [True, True, False, False]
        positive = [True, False, True, False]
        result = compute_from_boolean_arrays(selected, positive)
        assert result["tp"] == 1
        assert result["fp"] == 1
        assert result["fn"] == 1
        assert result["tn"] == 1

    def test_all_true(self):
        selected = [True, True, True]
        positive = [True, True, True]
        result = compute_from_boolean_arrays(selected, positive)
        assert result["tp"] == 3
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            compute_from_boolean_arrays([True, False], [True])
