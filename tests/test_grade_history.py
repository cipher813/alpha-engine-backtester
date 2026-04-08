"""Tests for analysis/grade_history.py."""

from analysis.grade_history import _extract_component_grades


class TestExtractComponentGrades:
    def test_full_grading(self):
        grading = {
            "status": "ok",
            "overall": {"grade": 72, "letter": "B"},
            "research": {
                "grade": 68,
                "letter": "B",
                "components": {
                    "scanner": {"grade": 75, "letter": "B+"},
                    "cio": {"grade": 70, "letter": "B"},
                    "sector_teams": [
                        {"team_id": "technology", "grade": 80, "letter": "B+"},
                        {"team_id": "healthcare", "grade": 60, "letter": "B-"},
                    ],
                },
            },
            "predictor": {
                "grade": 71,
                "letter": "B",
                "components": {
                    "gbm_model": {"grade": 65, "letter": "B"},
                    "veto_gate": {"grade": 77, "letter": "B+"},
                },
            },
            "executor": {
                "grade": 66,
                "letter": "B",
                "components": {
                    "entry_triggers": {"grade": 60, "letter": "B-"},
                    "risk_guard": {"grade": 72, "letter": "B"},
                },
            },
        }
        result = _extract_component_grades(grading)

        assert result["research"] == 68
        assert result["predictor"] == 71
        assert result["executor"] == 66
        assert result["research.scanner"] == 75
        assert result["research.cio"] == 70
        assert result["research.team.technology"] == 80
        assert result["research.team.healthcare"] == 60
        assert result["predictor.gbm_model"] == 65
        assert result["predictor.veto_gate"] == 77
        assert result["executor.entry_triggers"] == 60
        assert result["executor.risk_guard"] == 72

    def test_empty_grading(self):
        result = _extract_component_grades({})
        # Module keys are always iterated, but grades will be None
        for v in result.values():
            assert v is None

    def test_none_grades(self):
        grading = {
            "research": {"grade": None, "components": {"scanner": {"grade": None}}},
            "predictor": {"grade": 50, "components": {}},
            "executor": {"grade": None, "components": {}},
        }
        result = _extract_component_grades(grading)
        assert result["research"] is None
        assert result["research.scanner"] is None
        assert result["predictor"] == 50
        assert result["executor"] is None
