"""Tests for analysis/grading.py — unified component scorecard."""

import pytest

from analysis.grading import (
    _clamp,
    _ic_to_grade,
    _letter,
    _lift_to_grade,
    _pct_to_grade,
    _ratio_to_grade,
    _weighted_avg,
    compute_scorecard,
)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestLetter:
    def test_a(self):
        assert _letter(92) == "A"

    def test_b_plus(self):
        assert _letter(75) == "B+"

    def test_f(self):
        assert _letter(10) == "F"

    def test_none(self):
        assert _letter(None) == "N/A"

    def test_clamped_above_100(self):
        assert _letter(105) == "A"

    def test_zero(self):
        assert _letter(0) == "F"


class TestPctToGrade:
    def test_baseline_maps_to_30(self):
        g = _pct_to_grade(0.50, baseline=0.50, ceiling=0.80)
        assert g == pytest.approx(30.0)

    def test_ceiling_maps_to_95(self):
        g = _pct_to_grade(0.80, baseline=0.50, ceiling=0.80)
        assert g == pytest.approx(95.0)

    def test_none_returns_none(self):
        assert _pct_to_grade(None) is None

    def test_below_baseline_clamps(self):
        g = _pct_to_grade(0.20, baseline=0.50, ceiling=0.80)
        assert g >= 0.0


class TestLiftToGrade:
    def test_zero_lift_maps_to_40(self):
        g = _lift_to_grade(0.0)
        assert g == pytest.approx(40.0)

    def test_positive_lift(self):
        g = _lift_to_grade(1.5, floor=-2.0, ceiling=3.0)
        assert 40.0 < g < 100.0

    def test_negative_lift(self):
        g = _lift_to_grade(-1.0, floor=-2.0, ceiling=3.0)
        assert 0.0 < g < 40.0

    def test_none_returns_none(self):
        assert _lift_to_grade(None) is None


class TestIcToGrade:
    def test_zero_ic(self):
        g = _ic_to_grade(0.0)
        assert g == pytest.approx(20.0)

    def test_good_ic(self):
        g = _ic_to_grade(0.05)
        assert g == pytest.approx(55.0)

    def test_great_ic(self):
        g = _ic_to_grade(0.10)
        assert g == pytest.approx(90.0)

    def test_none(self):
        assert _ic_to_grade(None) is None


class TestWeightedAvg:
    def test_simple(self):
        result = _weighted_avg([(1.0, 80.0), (1.0, 60.0)])
        assert result == pytest.approx(70.0)

    def test_skips_none(self):
        result = _weighted_avg([(1.0, 80.0), (1.0, None), (1.0, 60.0)])
        assert result == pytest.approx(70.0)

    def test_all_none(self):
        assert _weighted_avg([(1.0, None), (1.0, None)]) is None

    def test_weighted(self):
        result = _weighted_avg([(3.0, 90.0), (1.0, 50.0)])
        assert result == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# Scorecard integration tests
# ---------------------------------------------------------------------------


class TestComputeScorecard:
    def test_empty_returns_insufficient(self):
        result = compute_scorecard()
        assert result["status"] == "insufficient_data"
        assert result["overall"]["grade"] is None
        assert result["research"]["grade"] is None
        assert result["predictor"]["grade"] is None
        assert result["executor"]["grade"] is None

    def test_partial_data(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_10d": 0.58, "avg_alpha_10d": 1.5, "n_10d": 50},
                "by_score_bucket": [{"bucket": "90+", "accuracy_10d": 0.71}],
            },
        )
        # Only portfolio sub-component has data → executor partial, research/predictor empty
        assert result["status"] in ("partial", "insufficient_data")

    def test_full_data_produces_grades(self):
        result = compute_scorecard(
            signal_quality={
                "status": "ok",
                "overall": {"accuracy_10d": 0.58, "avg_alpha_10d": 1.5, "n_10d": 50},
                "by_score_bucket": [{"bucket": "90+", "accuracy_10d": 0.71}],
            },
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.2, "n_passing": 55, "n_universe": 900},
                "team_lift": [
                    {"team_id": "technology", "lift": 2.5, "lift_vs_quant": 1.1, "n_picks": 12},
                ],
                "cio_lift": {
                    "lift": 1.5, "advance_avg": 2.1, "reject_avg": -0.3,
                    "n_advance": 15, "n_reject": 20,
                },
                "cio_vs_ranking": {
                    "lift": 0.8, "cio_beats_ranking": True,
                    "n_dates": 8, "n_picks": 15, "avg_overlap": 0.6,
                    "cio_avg": 2.1, "ranking_avg": 1.3,
                },
            },
            predictor_sizing={
                "status": "ok", "overall_rank_ic": 0.06,
                "recent_positive_weeks": 6, "recent_total_weeks": 8,
                "sizing_lift": 0.3, "n_samples": 100,
            },
            veto_result={
                "status": "ok", "current_threshold": 0.65, "base_rate": 0.55,
                "thresholds": [{
                    "confidence": 0.65, "precision": 0.68, "lift": 13.0,
                    "n_vetoes": 25, "true_negatives": 17, "false_negatives": 8,
                    "missed_alpha": 2.1,
                }],
                "recommended_threshold": 0.65,
            },
            veto_value={"net_value": 420.0},
            trigger_scorecard={
                "status": "ok",
                "triggers": [{"trigger": "pullback", "n_trades": 20, "avg_slippage_vs_signal": -0.3, "win_rate_vs_spy": 0.55}],
                "summary": {"total_entries": 35, "avg_slippage_vs_signal": -0.4, "win_rate_vs_spy": 0.57, "avg_realized_alpha": 1.2},
            },
            shadow_book={
                "status": "ok", "n_blocked": 12, "n_traded": 35,
                "guard_lift": 1.5, "blocked_beat_spy_pct": 0.33, "assessment": "appropriate",
            },
            exit_timing={
                "status": "ok", "n_roundtrips": 28,
                "summary": {"avg_capture_ratio": 0.62, "avg_realized_return": 1.8},
                "diagnosis": "exits_could_improve",
            },
        )

        assert result["status"] == "ok"
        assert result["overall"]["grade"] is not None
        assert 0 <= result["overall"]["grade"] <= 100
        assert result["overall"]["letter"] != "N/A"

        # All modules should have grades
        assert result["research"]["grade"] is not None
        assert result["predictor"]["grade"] is not None
        assert result["executor"]["grade"] is not None

    def test_team_grades_ordered(self):
        """A team with higher lift should get a higher grade."""
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "good_team", "lift": 3.0, "lift_vs_quant": 2.0, "n_picks": 15},
                    {"team_id": "bad_team", "lift": -1.5, "lift_vs_quant": -1.0, "n_picks": 10},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5, "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 1.5, "ranking_avg": 1.0},
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        good = next(t for t in teams if t["team_id"] == "good_team")
        bad = next(t for t in teams if t["team_id"] == "bad_team")
        assert good["grade"] > bad["grade"]

    def test_insufficient_team_picks(self):
        """Teams with fewer than 3 picks get N/A."""
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": [
                    {"team_id": "tiny_team", "lift": 5.0, "lift_vs_quant": 3.0, "n_picks": 2},
                ],
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5, "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 8, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 1.5, "ranking_avg": 1.0},
            },
        )
        teams = result["research"]["components"]["sector_teams"]
        tiny = next(t for t in teams if t["team_id"] == "tiny_team")
        assert tiny["grade"] is None
        assert tiny["letter"] == "N/A"

    def test_scorecard_structure(self):
        """Verify the scorecard has the expected structure."""
        result = compute_scorecard()
        assert "status" in result
        assert "overall" in result
        assert "research" in result
        assert "predictor" in result
        assert "executor" in result
        assert "grade" in result["overall"]
        assert "letter" in result["overall"]
        assert "components" in result["research"]
        assert "components" in result["predictor"]
        assert "components" in result["executor"]

    def test_classification_metrics_in_grading(self):
        """When e2e_lift includes classification dicts, grading uses them."""
        clf = {"precision": 0.65, "recall": 0.30, "f1": 0.41, "tp": 20, "fp": 11, "fn": 47, "tn": 22, "n": 100}
        result = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900, "classification": clf},
                "team_lift": [
                    {"team_id": "tech", "lift": 2.0, "lift_vs_quant": 1.0, "n_picks": 10,
                     "classification": {"precision": 0.70, "recall": 0.35, "f1": 0.47, "tp": 7, "fp": 3, "fn": 13, "tn": 17, "n": 40}},
                ],
                "cio_lift": {
                    "lift": 1.0, "advance_avg": 2.0, "reject_avg": -0.5,
                    "n_advance": 10, "n_reject": 8,
                    "classification": {"precision": 0.60, "recall": 0.50, "f1": 0.55, "tp": 6, "fp": 4, "fn": 6, "tn": 4, "n": 20},
                },
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True, "n_dates": 4, "n_picks": 10, "avg_overlap": 0.5, "cio_avg": 2.0, "ranking_avg": 1.5},
            },
            shadow_book={
                "status": "ok", "n_blocked": 10, "n_traded": 30,
                "guard_lift": 1.0, "assessment": "appropriate",
                "classification": {"precision": 0.70, "recall": 0.20, "f1": 0.31, "tp": 7, "fp": 3, "fn": 28, "tn": 12, "n": 50},
            },
            veto_result={
                "status": "ok", "current_threshold": 0.65, "base_rate": 0.55,
                "thresholds": [{
                    "confidence": 0.65, "precision": 0.68, "recall": 0.40, "f1": 0.50,
                    "lift": 13.0, "n_vetoes": 25, "true_negatives": 17, "false_negatives": 8, "missed_alpha": 2.0,
                }],
                "recommended_threshold": 0.65,
            },
        )

        # Scanner should show P/R in detail
        scanner = result["research"]["components"]["scanner"]
        assert "precision" in scanner["detail"]
        assert "recall" in scanner["detail"]
        assert "f1" in scanner["detail"]

        # Team should show P/R in detail
        teams = result["research"]["components"]["sector_teams"]
        assert "precision" in teams[0]["detail"]
        assert "recall" in teams[0]["detail"]

        # CIO should show P/R in detail
        cio = result["research"]["components"]["cio"]
        assert "precision" in cio["detail"]
        assert "recall" in cio["detail"]

        # Risk guard should show P/R in detail
        guard = result["executor"]["components"]["risk_guard"]
        assert "precision" in guard["detail"]
        assert "recall" in guard["detail"]

        # Veto gate should show recall
        veto = result["predictor"]["components"]["veto_gate"]
        assert "recall" in veto["detail"]
        assert "f1" in veto["detail"]
