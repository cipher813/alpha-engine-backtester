"""Unit tests for reporter.py — pipeline health section and report structure."""
import pytest

from reporter import _section_pipeline_health, build_report


# ── _section_pipeline_health ─────────────────────────────────────────────────


class TestSectionPipelineHealth:

    def test_healthy_pipeline(self):
        """All systems OK should produce a clean section."""
        health = {
            "db_pull_status": "ok",
            "coverage": 1.0,
            "dates_simulated": 12,
            "dates_expected": 12,
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "## Pipeline Health" in text
        assert "Research DB: Loaded" in text
        assert "12/12 dates (100%)" in text

    def test_missing_db(self):
        """Failed DB pull should show MISSING warning."""
        health = {"db_pull_status": "failed"}
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "MISSING" in text
        assert "signal quality analysis skipped" in text

    def test_staleness_warning_shown(self):
        """Staleness warning should appear as blockquote."""
        health = {
            "db_pull_status": "ok",
            "staleness_warning": "STALE price data: last date 2026-03-20",
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "> STALE price data" in text

    def test_coverage_with_skip_reasons(self):
        """Low coverage with skip reasons should show breakdown."""
        health = {
            "db_pull_status": "ok",
            "coverage": 0.5,
            "dates_simulated": 6,
            "dates_expected": 12,
            "skip_reasons": {"no_price_index": 4, "no_signals": 2},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "6/12 dates (50%)" in text
        assert "no_price_index" in text

    def test_price_gaps_shown(self):
        """Price gap and unfilled gap counts should appear."""
        health = {
            "db_pull_status": "ok",
            "price_gap_warnings": {"AAPL": 8, "TSLA": 12},
            "unfilled_gaps": {"TSLA": 7},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "Price gaps (>5 days): 2 tickers" in text
        assert "Unfilled gaps after ffill: 1 tickers" in text

    def test_predictor_feature_skips(self):
        """Predictor feature skip reasons should appear."""
        health = {
            "db_pull_status": "ok",
            "feature_skip_reasons": {"too_short": 50, "computation_error": 3},
        }
        lines = _section_pipeline_health(health)
        text = "\n".join(lines)
        assert "too_short" in text

    def test_empty_health_dict(self):
        """Empty health dict should still produce a valid section."""
        lines = _section_pipeline_health({})
        text = "\n".join(lines)
        assert "## Pipeline Health" in text
        assert "unknown" in text  # db_pull_status defaults to "unknown"


# ── build_report() ───────────────────────────────────────────────────────────


class TestBuildReport:

    def test_returns_string_with_header(self):
        """Report should be a string starting with the title."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "ok", "overall": {}},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert isinstance(md, str)
        assert "# Alpha Engine Backtest Report" in md
        assert "2026-03-29" in md

    def test_pipeline_health_included_when_provided(self):
        """Pipeline health section should appear when health dict is passed."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "ok", "overall": {}},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            pipeline_health={"db_pull_status": "ok", "coverage": 0.9,
                             "dates_simulated": 9, "dates_expected": 10},
        )
        assert "## Pipeline Health" in md
        assert "9/10 dates" in md

    def test_pipeline_health_absent_when_none(self):
        """No pipeline health section when not provided."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert "## Pipeline Health" not in md

    def test_skipped_mode_produces_valid_report(self):
        """All-skipped inputs should still produce a valid markdown report."""
        md = build_report(
            run_date="2026-03-29",
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
        )
        assert isinstance(md, str)
        assert len(md) > 50
