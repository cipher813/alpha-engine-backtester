"""Unit tests for optimizer.regression_monitor — metrics extraction, regression detection."""
import pytest
from unittest.mock import patch, MagicMock

from optimizer.regression_monitor import extract_metrics, check_regression


# ── extract_metrics ──────────────────────────────────────────────────────────


class TestExtractMetrics:

    def test_extracts_portfolio_fields(self):
        """Should extract sharpe_ratio, total_alpha, max_drawdown, win_rate."""
        stats = {
            "sharpe_ratio": 1.5,
            "total_alpha": 0.08,
            "max_drawdown": -0.12,
            "win_rate": 0.55,
            "irrelevant_key": 42,
        }
        metrics = extract_metrics(stats, None)
        assert metrics["sharpe_ratio"] == 1.5
        assert metrics["total_alpha"] == 0.08
        assert metrics["max_drawdown"] == -0.12
        assert metrics["win_rate"] == 0.55
        assert "irrelevant_key" not in metrics

    def test_extracts_signal_quality_fields(self):
        """Should extract accuracy_10d and accuracy_30d from overall dict."""
        sq = {
            "status": "ok",
            "overall": {
                "accuracy_10d": 0.62,
                "accuracy_30d": 0.58,
            },
        }
        metrics = extract_metrics(None, sq)
        assert metrics["accuracy_10d"] == 0.62
        assert metrics["accuracy_30d"] == 0.58

    def test_combines_both_sources(self):
        """Should merge fields from both portfolio stats and signal quality."""
        stats = {"sharpe_ratio": 1.2, "total_alpha": 0.05}
        sq = {"overall": {"accuracy_10d": 0.60}}
        metrics = extract_metrics(stats, sq)
        assert "sharpe_ratio" in metrics
        assert "accuracy_10d" in metrics

    def test_none_inputs_return_empty(self):
        """Both None inputs should return empty dict."""
        assert extract_metrics(None, None) == {}

    def test_empty_dicts_return_empty(self):
        """Empty dicts should return empty metrics."""
        assert extract_metrics({}, {}) == {}

    def test_missing_overall_key(self):
        """Signal quality without 'overall' key should not crash."""
        metrics = extract_metrics(None, {"status": "ok"})
        assert metrics == {}


# ── check_regression ─────────────────────────────────────────────────────────


class TestCheckRegression:

    @patch("optimizer.regression_monitor._load_baseline")
    def test_no_baseline_skips_check(self, mock_load):
        """No baseline → checked=False, no regression."""
        mock_load.return_value = None
        result = check_regression("test-bucket", {"sharpe_ratio": 1.0})
        assert result["checked"] is False
        assert "no baseline" in result.get("reason", "")

    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_positive_sharpe_detects_large_drop(self, mock_load, mock_rollback):
        """Sharpe dropping >20% from positive baseline should trigger regression."""
        mock_load.return_value = {
            "sharpe_ratio": 2.0,
            "accuracy_10d": 0.60,
        }
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.0, "accuracy_10d": 0.58},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is True
        assert result["details"]["sharpe_drop_pct"] == pytest.approx(0.5, abs=0.01)

    @patch("optimizer.regression_monitor._load_baseline")
    def test_positive_sharpe_no_regression_when_stable(self, mock_load):
        """Sharpe within 20% of baseline should NOT trigger regression."""
        mock_load.return_value = {"sharpe_ratio": 2.0}
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.8},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor._load_baseline")
    def test_negative_sharpe_baseline_skips_sharpe_check(self, mock_load):
        """Negative baseline Sharpe should skip the Sharpe regression check."""
        mock_load.return_value = {
            "sharpe_ratio": -0.5,
            "accuracy_10d": 0.60,
        }
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": -1.0, "accuracy_10d": 0.58},
        )
        assert result["checked"] is True
        # Sharpe check skipped (base_sharpe <= 0), so no sharpe_drop_pct
        assert "sharpe_drop_pct" not in result["details"]
        # Accuracy drop is only 2pp (< 5pp threshold), so no regression
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_accuracy_drop_triggers_regression(self, mock_load, mock_rollback):
        """Accuracy dropping >5pp should trigger regression."""
        mock_load.return_value = {
            "accuracy_10d": 0.65,
        }
        result = check_regression(
            "test-bucket",
            {"accuracy_10d": 0.55},
        )
        assert result["checked"] is True
        assert result["regression_detected"] is True
        assert result["details"]["accuracy_drop"] == pytest.approx(10.0, abs=0.1)

    @patch("optimizer.regression_monitor._load_baseline")
    def test_same_metrics_no_regression(self, mock_load):
        """Identical metrics should not trigger regression."""
        baseline = {"sharpe_ratio": 1.5, "accuracy_10d": 0.60}
        mock_load.return_value = baseline
        result = check_regression("test-bucket", baseline.copy())
        assert result["checked"] is True
        assert result["regression_detected"] is False

    @patch("optimizer.regression_monitor.rollback_all", return_value=[])
    @patch("optimizer.regression_monitor._load_baseline")
    def test_custom_thresholds(self, mock_load, mock_rollback):
        """Custom config thresholds should be respected."""
        mock_load.return_value = {"sharpe_ratio": 2.0}
        # 15% drop with a strict 10% threshold → should trigger
        result = check_regression(
            "test-bucket",
            {"sharpe_ratio": 1.7},
            config={"regression_monitor": {"sharpe_drop_threshold_pct": 0.10}},
        )
        assert result["regression_detected"] is True
