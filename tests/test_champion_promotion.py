"""champion_promotion — weekly winner-take-all champion/challenger gate
(alpha-engine-config-I2518 / epic I2515, 2026-07-14 ruling; supersedes the
config#2364/#2367 HAC-significance/hysteresis/cooldown engine).

Pins: (1) the seat swap — VALID_CHAMPIONS is now
(scanner_predictor_direct, thinktank_coverage), "agentic" is read-tolerated
on pointer/audit READ paths (WARN + normalize) but write-forbidden; (2)
weekly winner-take-all — whichever arm's realized score is higher this
week wins, no significance/hysteresis/cooldown; (3) validity guards —
either arm's score being unavailable (missing leaderboard, thinktank_coverage
not yet in the leaderboard, no resolved outcomes) is a NO-CONTEST, never a
default win; (4) --freeze suppresses the pointer write but the audit record
is still written every week (the liveness proxy, config#2054); (5)
frozen-schema (v2) conformance against contracts/producer_champion.schema.json
and contracts/producer_champion_audit.schema.json; (6) alpha-engine-config
-I2544 (2026-07-14, same-session follow-up) — thinktank_coverage's evidence
is read as the LATEST research/producer_leaderboard/{date}.json available
<= run_date (list+parse over the prefix, not an exact-key read), honestly
bounded to LEADERBOARD_STALENESS_DAYS (8) calendar days, with the date
actually used recorded on every outcome via leaderboard_date_used.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

from nousergon_lib.contracts import load_schema as _load_contract_schema

from optimizer.champion_promotion import (
    ARM_FEED_DEPENDENCIES,
    CONFIDENCE_INSUFFICIENT,
    CONFIDENCE_NOT_LEADERBOARD_SCORED,
    MIN_CYCLES_FOR_INFERENCE,
    CONFIDENCE_OK,
    CONFIDENCE_THIN,
    CONFIDENCE_UNKNOWN,
    _BLOCKED_BY_SLUGS,
    CONFIDENCE_UNRECOGNISED,
    GATE_HORIZON_DAYS,
    LEADERBOARD_STALENESS_DAYS,
    OUTCOMES,
    VALID_CHAMPIONS,
    build_champion_audit,
    build_leaderboard_artifact,
    build_weekly_arm_scores,
    check_feed_dependencies_live,
    evaluate_gates,
    find_latest_research_producer_leaderboard_date,
    hac_significance,
    leaderboard_entry_from_e2e_lift,
    leaderboard_gate_inputs,
    leaderboard_row_confidence,
    read_champion_pointer,
    read_latest_research_producer_leaderboard,
    read_prior_leaderboard_history,
    read_research_producer_leaderboard,
    run_weekly_evaluation,
    write_champion_pointer,
    write_leaderboard,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
POINTER_SCHEMA_PATH = REPO_ROOT / "contracts" / "producer_champion.schema.json"
# Moved to nousergon-lib (alpha-engine-config-I7605): the audit contract now
# reaches both producer (here) and consumer (crucible-dashboard) via the SAME
# published resource, rather than crucible-dashboard walking this repo's
# working tree via a sibling checkout.
AUDIT_SCHEMA = _load_contract_schema("producer_champion_audit")


# ── HAC/Newey-West significance (retained utility, not gate-connected) ─────


class TestHacSignificance:
    """hac_significance is retained as an independently-tested utility (see
    module docstring) even though the winner-take-all gate no longer calls
    it — these pins guard against accidental regression of code that other
    future work may still depend on."""

    def test_insufficient_data_below_two_points(self):
        assert hac_significance([])["status"] == "insufficient_data"
        assert hac_significance([0.01])["status"] == "insufficient_data"

    def test_significant_positive_series_flagged(self):
        series = [0.02, 0.021, 0.019, 0.022, 0.02, 0.018, 0.021]
        result = hac_significance(series)
        assert result["status"] == "ok"
        assert result["significant"] is True
        assert result["mean"] > 0

    def test_noisy_near_zero_series_not_significant(self):
        series = [0.01, -0.015, 0.008, -0.006, 0.002, -0.009, 0.004]
        result = hac_significance(series)
        assert result["status"] == "ok"
        assert result["significant"] is False

    def test_lag_selection_is_horizon_over_cadence(self):
        series = [0.01] * 10
        result = hac_significance(series)
        assert result["lags"] == 3


# ── Weekly arm-score sourcing ────────────────────────────────────────────────


def _e2e_lift_ok(sn_lift=0.02, n_cycles=6):
    """alpha-engine-config-I2998: the gate's score source is
    ``sector_neutral_mean_alpha_21d`` (this arm's own SPY-relative realized
    alpha, NOT ``sn_lift_vs_agentic_cio``) -- the ``sn_lift`` param sets
    that field directly (kept as ``sn_lift`` to minimize churn across the
    many existing call sites); the retired ``sn_lift_vs_agentic_cio``
    observability field is set to the same value for fixture realism."""
    return {
        "scanner_then_predictor_counterfactual": {
            "status": "ok",
            "n_cycles": n_cycles,
            "methods": {
                "scanner_then_predictor_topN": {
                    "mean_alpha_21d": 0.03,
                    "sector_neutral_mean_alpha_21d": sn_lift,
                    "n_picks": 40,
                    "sn_lift_vs_agentic_cio": sn_lift,
                },
                "agentic_cio_advance": {
                    "mean_alpha_21d": 0.01,
                    "sector_neutral_mean_alpha_21d": 0.005,
                    "n_picks": 40,
                },
            },
        },
    }


# The producer's own evidence floor (crucible-research
# scoring/leaderboard_scoring.py::MIN_DATES_FOR_INFERENCE, declared per slot in
# LEADERBOARD_SLOTS and stamped onto the artifact as min_dates_for_inference).
# Mirrored here ONLY to build realistic fixtures — the module under test reads
# it off the artifact and never hardcodes it (alpha-engine-config-I7542/I7549).
FIXTURE_MIN_DATES_FOR_INFERENCE = 5

# Distinguishes "caller did not override confidence" from an explicit
# ``confidence=None`` (= build a PRE-I7542 artifact with no confidence key).
_MISSING = object()


def _confidence_for(n_dates_scored):
    """The producer's own confidence rule, replicated for fixture realism
    (crucible-research scoring/leaderboard_scoring.py::confidence_for)."""
    n = int(n_dates_scored or 0)
    if n <= 0:
        return CONFIDENCE_INSUFFICIENT
    if n < FIXTURE_MIN_DATES_FOR_INFERENCE:
        return CONFIDENCE_THIN
    return CONFIDENCE_OK


def _tt_leaderboard_ok(
    run_date="2026-07-18", mean=0.015, n_dates_scored=5, *, confidence=_MISSING,
):
    """Mimics the REAL crucible-research schema
    (scoring/leaderboard_producers.py::build_producer_leaderboard /
    scoring/leaderboard_scoring.py::score_leaderboard) verified against the
    crucible-research checkout 2026-07-20 (alpha-engine-config-I2998:
    champion is optional, ``topn_alpha_vs_benchmark`` added -- the gate's
    actual score source; ``topn_alpha_vs_champion`` kept for schema realism
    but no longer read by ``_score_thinktank_coverage``), re-verified against
    ``origin/main`` 2026-08-17 after crucible-research PR643 /
    alpha-engine-config-I7542 added the per-spec ``confidence`` field and
    I7540 added the artifact-level ``min_dates_for_inference`` /
    ``horizons_days`` / ``horizons`` multi-horizon block. ``mean`` sets
    ``topn_alpha_vs_benchmark.mean`` for the thinktank_coverage row.

    ``confidence`` overrides the thinktank_coverage row's verdict (pass
    ``None`` to build a PRE-I7542 artifact whose row carries no confidence key
    at all); by default it is derived from ``n_dates_scored`` exactly as the
    producer would."""
    tt_confidence = (
        _confidence_for(n_dates_scored) if confidence is _MISSING else confidence
    )
    tt_row = {
        "name": "thinktank_coverage", "kind": "challenger",
        "realized_rank_ic": {"mean": 0.04, "se": 0.02, "t_stat": 2.0, "n_dates": n_dates_scored},
        "topn_alpha_vs_champion": {"mean": mean, "se": 0.01, "t_stat": 1.5, "n_dates": n_dates_scored},
        "topn_alpha_vs_benchmark": {"mean": mean, "se": 0.01, "t_stat": 1.5, "n_dates": n_dates_scored},
        "n_dates_scored": n_dates_scored,
    }
    if tt_confidence is not None:
        tt_row["confidence"] = tt_confidence
    specs = [
        {
            "name": "agentic_sector_teams", "kind": "champion",
            "realized_rank_ic": {"mean": 0.05, "se": 0.02, "t_stat": 2.5, "n_dates": 12},
            "topn_alpha_vs_champion": None,
            "topn_alpha_vs_benchmark": {"mean": 0.01, "se": 0.01, "t_stat": 1.0, "n_dates": 12},
            "n_dates_scored": 12,
            "confidence": CONFIDENCE_OK,
        },
        {
            "name": "no_agent_quant", "kind": "challenger",
            "realized_rank_ic": {"mean": 0.03, "se": 0.02, "t_stat": 1.5, "n_dates": 12},
            "topn_alpha_vs_champion": {"mean": 0.008, "se": 0.01, "t_stat": 0.8, "n_dates": 12},
            "topn_alpha_vs_benchmark": {"mean": 0.006, "se": 0.01, "t_stat": 0.6, "n_dates": 12},
            "n_dates_scored": 12,
            "confidence": CONFIDENCE_OK,
        },
        tt_row,
    ]
    return {
        "champion": "agentic_sector_teams",
        "horizon_days": 21,
        "top_n": 50,
        "benchmark_ticker": "SPY",
        "n_dates": 12,
        "date": run_date,
        "leaderboard_id": "producer",
        "specs": specs,
        # alpha-engine-config-I7540 multi-horizon block. The 21-session block
        # is ALSO spread across the top level above (§3 continuity), which is
        # what the gate reads; 126/252 are immature at rollout.
        "horizons_days": [21, 126, 252],
        "min_dates_for_inference": FIXTURE_MIN_DATES_FOR_INFERENCE,
        "horizons": [
            {"horizon_days": 21, "status": "ok", "reason": None,
             "n_dates": 12, "specs": specs},
            {"horizon_days": 126, "status": "immature", "reason": "cohort not matured",
             "n_dates": 0, "specs": []},
            {"horizon_days": 252, "status": "immature", "reason": "cohort not matured",
             "n_dates": 0, "specs": []},
        ],
    }


class TestBuildWeeklyArmScores:
    def test_both_sides_valid(self):
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.02), _tt_leaderboard_ok(mean=0.015), run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] == 0.02
        assert result["scores"]["thinktank_coverage"] == 0.015
        assert result["unavailable_reasons"] == {}
        assert result["leaderboard_date_used"] == "2026-07-18"

    def test_missing_e2e_lift(self):
        result = build_weekly_arm_scores(
            None, _tt_leaderboard_ok(), run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] is None
        assert result["unavailable_reasons"]["scanner_predictor_direct"] == (
            "scanner_predictor_direct_counterfactual_unavailable"
        )

    def test_missing_leaderboard(self):
        """No leaderboard <= run_date was found at all (find_latest_...
        returned None) -- leaderboard_date_used is also None, distinct from
        the stale-but-found case below."""
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), None, run_date="2026-07-18", leaderboard_date_used=None,
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_unavailable"
        assert result["leaderboard_date_used"] is None

    def test_stale_beyond_8_days_is_no_contest(self):
        """alpha-engine-config-I2544: the latest leaderboard available <=
        run_date was found, but it's more than LEADERBOARD_STALENESS_DAYS
        (8) calendar days older than run_date -- an honest no-contest, not
        a fabricated score against evidence this old."""
        stale = _tt_leaderboard_ok(run_date="2026-07-09")
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), stale, run_date="2026-07-18", leaderboard_date_used="2026-07-09",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_stale_gt_8d"
        assert result["leaderboard_date_used"] == "2026-07-09"

    def test_exactly_8_days_stale_is_still_scored(self):
        """The boundary is inclusive: "more than 8 days" fails, exactly 8
        days does not -- age_days > LEADERBOARD_STALENESS_DAYS, not >=."""
        assert LEADERBOARD_STALENESS_DAYS == 8
        lb = _tt_leaderboard_ok(run_date="2026-07-10", mean=0.02)
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-10",
        )
        assert result["scores"]["thinktank_coverage"] == 0.02
        assert "thinktank_coverage" not in result["unavailable_reasons"]

    def test_negative_age_is_leaderboard_unavailable(self):
        """Defensive: a leaderboard_date_used somehow after run_date (never
        produced by find_latest_research_producer_leaderboard_date itself,
        but a caller could pass this directly) must never be trusted as
        this week's evidence."""
        lb = _tt_leaderboard_ok(run_date="2026-07-20", mean=0.02)
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-20",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_unavailable"

    def test_thinktank_coverage_not_registered_in_leaderboard(self):
        """The KNOWN, TRACKED GAP (module docstring / alpha-engine-config
        -I2519): thinktank_coverage isn't yet registered in crucible
        -research's producers/registry.py, so its row is simply absent from
        specs -- must be an honest no-contest reason, not a crash."""
        lb = _tt_leaderboard_ok()
        lb["specs"] = [s for s in lb["specs"] if s["name"] != "thinktank_coverage"]
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_not_in_leaderboard"
        )

    def test_thinktank_coverage_zero_dates_scored(self):
        lb = _tt_leaderboard_ok(n_dates_scored=0)
        for s in lb["specs"]:
            if s["name"] == "thinktank_coverage":
                s["topn_alpha_vs_champion"] = None
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_no_resolved_outcomes"
        )

    def test_malformed_specs_is_leaderboard_unavailable(self):
        lb = {"date": "2026-07-18", "specs": "not-a-list"}
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == "leaderboard_unavailable"

    def test_thinktank_coverage_scored_when_no_champion_registered(self):
        """alpha-engine-config-I2998: config-I2993 retired agentic_sector_teams
        with no successor champion registered -- crucible-research's
        score_leaderboard now writes champion=None and scores every
        challenger champion-free. This arm's score must still resolve from
        topn_alpha_vs_benchmark, independent of the champion field."""
        lb = _tt_leaderboard_ok(mean=0.021)
        lb["champion"] = None
        for spec in lb["specs"]:
            if spec["name"] == "agentic_sector_teams":
                spec["kind"] = "retired"
            spec["topn_alpha_vs_champion"] = None
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] == pytest.approx(0.021)
        assert "thinktank_coverage" not in result["unavailable_reasons"]


# ── Evidence-confidence gate (alpha-engine-config-I7549) ───────────────────
#
# RED-FIRST EVIDENCE (champion-challenger-policy.md §7.4). Every test in this
# class was run against the PRE-FIX optimizer/champion_promotion.py (the
# `if not row.get("n_dates_scored")` truthiness check) and observed to FAIL
# before the fix landed -- see the PR body for the recorded run. The
# load-bearing one is test_thin_row_is_not_evidence: at n_dates_scored=1 the
# pre-fix gate returned a SCORE of -0.060751 and no unavailable reason, i.e.
# it would have compared a one-date mean carrying a null SE and a null t-stat
# against the live champion.


class TestEvidenceConfidenceGate:
    """A thin arm is not evidence. The gate refuses to promote OR demote on a
    non-``ok`` confidence row, and says which evidence declined to decide."""

    def test_thin_row_is_not_evidence(self):
        """The live 2026-08-14 shape: n_dates_scored=1, mean present, se and
        t_stat null. Pre-fix this SCORED (n=1 is truthy) and could have moved
        config/producer_champion.json."""
        lb = _tt_leaderboard_ok(n_dates_scored=1, mean=-0.060751)
        for s in lb["specs"]:
            if s["name"] == "thinktank_coverage":
                assert s["confidence"] == CONFIDENCE_THIN
                s["topn_alpha_vs_benchmark"]["se"] = None
                s["topn_alpha_vs_benchmark"]["t_stat"] = None
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_thin_evidence"
        )
        assert result["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_THIN

    def test_thin_slug_is_distinct_from_absent_slug(self):
        """I7549: `thin` (wait for the cohort) and `insufficient` (go look at
        the producer) are different operator responses, so they must never
        share a slug."""
        thin = build_weekly_arm_scores(
            _e2e_lift_ok(), _tt_leaderboard_ok(n_dates_scored=2),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        absent = build_weekly_arm_scores(
            _e2e_lift_ok(), _tt_leaderboard_ok(n_dates_scored=0),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert thin["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_thin_evidence"
        )
        assert absent["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_no_resolved_outcomes"
        )
        assert (
            thin["unavailable_reasons"]["thinktank_coverage"]
            != absent["unavailable_reasons"]["thinktank_coverage"]
        )
        assert thin["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_THIN
        assert absent["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_INSUFFICIENT

    def test_ok_row_scores_exactly_as_before(self):
        """No behaviour change on the healthy path -- the whole point of
        gating on the producer's verdict rather than on a new local rule."""
        lb = _tt_leaderboard_ok(mean=0.0173, n_dates_scored=9)
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] == pytest.approx(0.0173)
        assert "thinktank_coverage" not in result["unavailable_reasons"]
        assert result["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_OK

    def test_thin_row_cannot_demote_the_sitting_champion_either(self):
        """§5.2 is not weakened: a thin row holds the pointer in BOTH
        directions. With thinktank_coverage as the sitting champion and a thin
        row, the gate must no-contest -- never hand the seat to
        scanner_predictor_direct on non-evidence."""
        arm_scores = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.05), _tt_leaderboard_ok(n_dates_scored=1),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        record = evaluate_gates(
            champion_before="thinktank_coverage", arm_scores=arm_scores, freeze=False,
        )
        assert record["outcome"] == "no_contest"
        assert record["champion_after"] == "thinktank_coverage"
        assert "thinktank_coverage_thin_evidence" in record["blocked_by"]

    def test_thin_row_cannot_promote_the_challenger_either(self):
        """The mirror case: a thin row with a great mean must not take the
        seat from the incumbent."""
        arm_scores = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.001),
            _tt_leaderboard_ok(n_dates_scored=1, mean=0.99),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        record = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores,
            freeze=False,
        )
        assert record["outcome"] == "no_contest"
        assert record["champion_after"] == "scanner_predictor_direct"
        assert record["challenger_score"] is None

    def test_legacy_artifact_without_confidence_falls_back_to_declared_floor(self):
        """A pre-I7542 artifact carries no `confidence`, but DOES carry the
        artifact-level floor -- derive the verdict from it rather than
        trusting the row."""
        thin = _tt_leaderboard_ok(n_dates_scored=1, confidence=None)
        assert "confidence" not in [
            k for s in thin["specs"] if s["name"] == "thinktank_coverage" for k in s
        ]
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), thin, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_thin_evidence"
        )

        healthy = _tt_leaderboard_ok(n_dates_scored=11, mean=0.02, confidence=None)
        ok_result = build_weekly_arm_scores(
            _e2e_lift_ok(), healthy, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert ok_result["scores"]["thinktank_coverage"] == pytest.approx(0.02)

    def test_legacy_artifact_with_neither_field_is_unavailable_not_ok(self):
        """The defect being fixed, repeated one layer down: an artifact that
        can say NOTHING about how much evidence stands behind its row must
        never be read as confident."""
        lb = _tt_leaderboard_ok(n_dates_scored=1, confidence=None)
        lb.pop("min_dates_for_inference")
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_confidence_unknown"
        )
        assert result["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_UNKNOWN

    def test_unrecognised_confidence_value_is_refused_not_guessed(self):
        lb = _tt_leaderboard_ok(n_dates_scored=12, confidence="excellent")
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_confidence_unknown"
        )
        assert result["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_UNRECOGNISED

    def test_confidence_ok_but_metric_absent_is_still_unavailable(self):
        """An internally inconsistent row -- confidence says ok, the primary
        metric is missing. Fail static, keep the claimed confidence so the
        audit shows the contradiction."""
        lb = _tt_leaderboard_ok(n_dates_scored=12)
        for s in lb["specs"]:
            if s["name"] == "thinktank_coverage":
                s["topn_alpha_vs_benchmark"] = None
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_no_resolved_outcomes"
        )
        assert result["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_OK

    def test_the_floor_is_read_off_the_artifact_never_hardcoded(self):
        """champion-challenger-policy.md §10: the evidence floor is a per-slot
        fact owned by the producer's registry. A row at n=6 is `ok` under a
        floor of 5 and `thin` under a floor of 8 -- this module must follow
        the artifact, not a literal of its own."""
        lb = _tt_leaderboard_ok(n_dates_scored=6, mean=0.03, confidence=None)
        lb["min_dates_for_inference"] = 8
        strict = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert strict["unavailable_reasons"]["thinktank_coverage"] == (
            "thinktank_coverage_thin_evidence"
        )
        lb["min_dates_for_inference"] = 5
        lenient = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert lenient["scores"]["thinktank_coverage"] == pytest.approx(0.03)

    def test_scanner_predictor_direct_confidence_comes_from_its_own_floor(self):
        """That arm is scored from this repo's own counterfactual, so its
        verdict is measured against this repo's own floor — never inherited
        from, and never left absent because of, the leaderboard.

        Superseded ``not_leaderboard_scored`` (alpha-engine-config-I7549
        champion-side half): naming the SOURCE of the evidence where the field
        states how MUCH of it there is left the arm the gate compares against
        with no verdict at all. The constant stays defined for read-tolerance
        of records written in that window; nothing emits it.
        """
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), _tt_leaderboard_ok(), run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_OK
        assert result["arm_confidence"]["scanner_predictor_direct"] != (
            CONFIDENCE_NOT_LEADERBOARD_SCORED
        )
        # A leaderboard verdict must never leak onto this arm: the thin
        # leaderboard row below does not make the counterfactual thin.
        thin_lb = build_weekly_arm_scores(
            _e2e_lift_ok(), _tt_leaderboard_ok(n_dates_scored=1),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert thin_lb["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_THIN
        assert thin_lb["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_OK

    def test_confidence_reaches_the_audit_record(self):
        """I7549 deliverable 4 / §7.2: failing to decide must be VISIBLE, and
        the record must name WHICH evidence declined."""
        arm_scores = build_weekly_arm_scores(
            _e2e_lift_ok(), _tt_leaderboard_ok(n_dates_scored=1),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        record = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores,
            freeze=False,
        )
        audit = build_champion_audit("2026-07-18", record, freeze=False)
        assert audit["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_THIN
        assert audit["blocked_by"] == ["thinktank_coverage_thin_evidence"]
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        jsonschema.validate(
            instance=audit, schema=AUDIT_SCHEMA,
        )

    def test_declining_to_decide_is_logged_not_silent(self, caplog):
        """§7.2: fail-soft must never extend to silence. The no-op has to be
        readable on the run's own surface, not only in S3."""
        import logging
        with caplog.at_level(logging.WARNING, logger="optimizer.champion_promotion"):
            build_weekly_arm_scores(
                _e2e_lift_ok(), _tt_leaderboard_ok(n_dates_scored=1),
                run_date="2026-07-18", leaderboard_date_used="2026-07-18",
            )
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "thinktank_coverage NOT scored" in joined
        assert "thin" in joined
        assert "thinktank_coverage_thin_evidence" in joined


class TestGateHorizon:
    """Which horizon the champion gate decides on, asserted rather than
    inherited (alpha-engine-config-I7540/I7549)."""

    def test_gate_decides_at_the_primary_21_session_horizon(self):
        assert GATE_HORIZON_DAYS == 21

    def test_non_primary_horizon_blocks_do_not_move_the_gate(self):
        """The artifact now carries 126/252-session blocks under `horizons`.
        A spectacular 252-session thinktank_coverage number must not touch
        this gate: scanner_predictor_direct has no 252-session score, so
        ranking them would violate §4's same-horizon requirement."""
        lb = _tt_leaderboard_ok(mean=0.015, n_dates_scored=9)
        lb["horizons"][2] = {
            "horizon_days": 252, "status": "ok", "reason": None, "n_dates": 40,
            "specs": [{
                "name": "thinktank_coverage", "kind": "challenger",
                "realized_rank_ic": None,
                "topn_alpha_vs_champion": None,
                "topn_alpha_vs_benchmark": {
                    "mean": 0.91, "se": 0.02, "t_stat": 9.0, "n_dates": 40,
                },
                "n_dates_scored": 40, "confidence": CONFIDENCE_OK,
            }],
        }
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] == pytest.approx(0.015)

    def test_upstream_primary_horizon_change_is_a_loud_no_contest(self):
        """If the producer ever moves its primary block off 21 sessions, this
        gate must stop, not silently rescore the live champion on a horizon
        the other arm is not measured at."""
        lb = _tt_leaderboard_ok(mean=0.9, n_dates_scored=40)
        lb["horizon_days"] = 252
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] is None
        assert result["unavailable_reasons"]["thinktank_coverage"] == (
            "leaderboard_horizon_mismatch"
        )

    def test_pre_multi_horizon_artifact_without_horizon_days_still_scores(self):
        """Tolerant of the field's ABSENCE (it always meant 21), intolerant of
        it disagreeing."""
        lb = _tt_leaderboard_ok(mean=0.02, n_dates_scored=9)
        lb.pop("horizon_days")
        result = build_weekly_arm_scores(
            _e2e_lift_ok(), lb, run_date="2026-07-18",
            leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["thinktank_coverage"] == pytest.approx(0.02)


class TestLeaderboardRowConfidence:
    """The shared, arm-agnostic verdict helper (I7549 deliverable 2): any
    future arm scored off this artifact routes through here rather than
    growing a second thinness test."""

    @pytest.mark.parametrize("declared", [CONFIDENCE_OK, CONFIDENCE_THIN, CONFIDENCE_INSUFFICIENT])
    def test_producers_verdict_wins_over_any_local_derivation(self, declared):
        # n_dates_scored deliberately contradicts the declared verdict: the
        # producer owns the floor, this module does not second-guess it.
        assert leaderboard_row_confidence(
            {"name": "x", "confidence": declared, "n_dates_scored": 12},
            {"min_dates_for_inference": 5},
        ) == declared

    def test_never_infers_ok_without_a_declared_floor(self):
        assert leaderboard_row_confidence(
            {"name": "x", "n_dates_scored": 999}, {},
        ) == CONFIDENCE_UNKNOWN

    @pytest.mark.parametrize("floor", [None, 0, -1, "5", True, 1.5])
    def test_malformed_floor_is_unknown_not_ok(self, floor):
        assert leaderboard_row_confidence(
            {"name": "x", "n_dates_scored": 7}, {"min_dates_for_inference": floor},
        ) == CONFIDENCE_UNKNOWN

    @pytest.mark.parametrize("n_scored", [None, "7", -1, True, 1.5])
    def test_malformed_n_dates_scored_is_unknown_not_ok(self, n_scored):
        assert leaderboard_row_confidence(
            {"name": "x", "n_dates_scored": n_scored}, {"min_dates_for_inference": 5},
        ) == CONFIDENCE_UNKNOWN


# ── Gate engine (pure function, weekly winner-take-all) ────────────────────


class TestEvaluateGates:
    def test_seat_swap_valid_champions(self):
        assert VALID_CHAMPIONS == ("scanner_predictor_direct", "thinktank_coverage")
        assert "agentic" not in VALID_CHAMPIONS

    def test_challenger_wins_promotes(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "thinktank_coverage"
        assert result["champion_score"] == 0.01
        assert result["challenger_score"] == 0.03
        assert result["blocked_by"] is None

    def test_champion_still_winning_stays_champion(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert result["champion_after"] == "scanner_predictor_direct"
        assert result["blocked_by"] is None

    def test_exact_tie_favors_incumbent(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.02, "thinktank_coverage": 0.02},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_bidirectional_thinktank_coverage_as_champion(self):
        """Same rule set, reversed seats: thinktank_coverage is champion,
        scanner_predictor_direct challenges and wins -> promotes."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.04, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="thinktank_coverage", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "promoted"
        assert result["challenger"] == "scanner_predictor_direct"
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_no_contest_champion_score_missing(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": None, "thinktank_coverage": 0.02},
            "unavailable_reasons": {"scanner_predictor_direct": "scanner_predictor_direct_counterfactual_unavailable"},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["scanner_predictor_direct_counterfactual_unavailable"]
        assert result["champion_after"] == "scanner_predictor_direct"

    def test_no_contest_challenger_score_missing(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.02, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["thinktank_coverage_not_in_leaderboard"]

    def test_no_contest_both_sides_missing(self):
        arm_scores = {"scores": {"scanner_predictor_direct": None, "thinktank_coverage": None},
                       "unavailable_reasons": {}}
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["arm_score_unavailable", "arm_score_unavailable"]

    def test_freeze_suppresses_pointer_move_but_reports_would_be_outcome(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=True,
        )
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        assert result["champion_after"] == "scanner_predictor_direct"  # NOT advanced under freeze


# ── Pointer writer (single writer, dual promotion_source caller) ───────────


class TestWriteChampionPointer:
    def test_writes_expected_schema(self):
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", "scanner_predictor_direct",
            promotion_source="gate_engine", upload=True, s3_client=s3,
        )
        assert pointer["schema_version"] == 1
        assert pointer["champion"] == "scanner_predictor_direct"
        assert pointer["promotion_source"] == "gate_engine"
        assert "promoted_at" in pointer
        s3.put_object.assert_called_once()
        call = s3.put_object.call_args
        assert call.kwargs["Key"] == "config/producer_champion.json"
        body = json.loads(call.kwargs["Body"])
        assert body == pointer

    def test_writes_thinktank_coverage(self):
        s3 = MagicMock()
        pointer = write_champion_pointer(
            "bucket", "thinktank_coverage",
            promotion_source="gate_engine", upload=True, s3_client=s3,
        )
        assert pointer["champion"] == "thinktank_coverage"

    def test_upload_false_skips_s3(self):
        s3 = MagicMock()
        write_champion_pointer(
            "bucket", "scanner_predictor_direct", promotion_source="gate_engine",
            upload=False, s3_client=s3,
        )
        s3.put_object.assert_not_called()

    def test_rejects_unknown_champion(self):
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "not_a_real_arm", promotion_source="gate_engine",
                upload=False,
            )

    def test_rejects_retired_agentic_seat(self):
        """Write-forbidden half of the read-tolerated/write-forbidden
        posture: 'agentic' is no longer in VALID_CHAMPIONS, so any attempt
        to write it must raise -- belt-and-braces against ever re-writing
        the retired seat."""
        with pytest.raises(ValueError):
            write_champion_pointer(
                "bucket", "agentic", promotion_source="gate_engine", upload=False,
            )


class TestReadChampionPointer:
    def test_missing_key_returns_none(self):
        from botocore.exceptions import ClientError
        s3 = MagicMock()
        s3.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject",
        )
        assert read_champion_pointer("bucket", s3_client=s3) is None

    def test_reads_legacy_agentic_pointer_without_crashing(self):
        """A historical pointer object (or a defensive/manual read) carrying
        'agentic' must be readable without error -- normalization to the
        base-case arm happens in run_weekly_evaluation, not here (this
        function is a raw, faithful reader)."""
        s3 = MagicMock()
        body = json.dumps({"schema_version": 1, "champion": "agentic",
                            "promoted_at": "2026-07-01T00:00:00Z",
                            "promotion_source": "operator_bootstrap"}).encode()
        s3.get_object.return_value = {"Body": MagicMock(read=MagicMock(return_value=body))}
        pointer = read_champion_pointer("bucket", s3_client=s3)
        assert pointer["champion"] == "agentic"


# ── Leaderboard artifact (research/producer_leaderboard_champion_gate/{date}.json) ───────
# Retained for observability / config#2452 continuity — see module docstring.


def test_leaderboard_key_distinct_from_research_producer_leaderboard():
    """config#2452 regression guard: this module's OWN observability key
    must never collide with crucible-research's
    scoring/leaderboard_producers.py key (research/producer_leaderboard/
    {date}.json) -- a prior version of this module shared that exact key
    with an incompatible schema."""
    from optimizer.champion_promotion import LEADERBOARD_KEY_TMPL
    assert LEADERBOARD_KEY_TMPL != "research/producer_leaderboard/{date}.json"
    assert LEADERBOARD_KEY_TMPL.format(date="2026-07-13") != "research/producer_leaderboard/2026-07-13.json"


class _FakeS3:
    """Minimal in-memory S3 stand-in supporting exactly the get/put calls
    champion_promotion.py issues, keyed by (Bucket, Key)."""

    def __init__(self):
        self.store: dict[str, bytes] = {}

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        full = f"{Bucket}/{Key}"
        if full not in self.store:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": MagicMock(read=MagicMock(return_value=self.store[full]))}

    def put_object(self, Bucket, Key, Body, ContentType=None):
        full = f"{Bucket}/{Key}"
        self.store[full] = Body if isinstance(Body, bytes) else Body.encode()

    def list_objects_v2(self, Bucket, Prefix):
        bucket_prefix = f"{Bucket}/{Prefix}"
        contents = [
            {"Key": full[len(f"{Bucket}/"):]}
            for full in self.store
            if full.startswith(bucket_prefix)
        ]
        return {"Contents": contents}


def _put_research_free_backfill_parquet(s3: _FakeS3, bucket: str, *, newest_prediction_date: str) -> None:
    """Write a synthetic ``research_free_backfill`` parquet artifact into
    ``s3.store`` at the real key
    (``analysis.scanner_predictor_research_free_backfill.ARTIFACT_KEY``) with
    a single row dated ``newest_prediction_date`` — exactly the shape
    ``assert_champion_feed_fresh`` reads (see
    ``tests/test_scanner_predictor_research_free_backfill.py``'s identical
    fixture pattern). Used to synthesize both a live/fresh feed and a
    dead/stale-orphaned one for the alpha-engine-config-I3165 promotion-time
    feed-liveness gate tests below."""
    import io as _io

    import pandas as _pd

    from analysis.scanner_predictor_research_free_backfill import ARTIFACT_KEY

    df = _pd.DataFrame(
        [("AAPL", newest_prediction_date, 0.01, 0)],
        columns=["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"],
    )
    buf = _io.BytesIO()
    df.to_parquet(buf, index=False)
    s3.store[f"{bucket}/{ARTIFACT_KEY}"] = buf.getvalue()


class TestLeaderboardObservability:
    def test_entry_extraction_from_e2e_lift(self):
        entry = leaderboard_entry_from_e2e_lift(_e2e_lift_ok(sn_lift=0.017))
        assert entry["sector_neutral_mean_alpha_21d"] == 0.017
        assert entry["sn_lift_vs_agentic_cio"] == 0.017
        assert entry["n_cycles"] == 6

    def test_entry_extraction_gates_on_sector_neutral_alpha_not_agentic_lift(self):
        """alpha-engine-config-I2998: the entry (and hence the gate score)
        must remain usable even when the retired agentic-comparator field
        is unavailable -- gating is on sector_neutral_mean_alpha_21d only."""
        e2e = _e2e_lift_ok(sn_lift=0.019)
        e2e["scanner_then_predictor_counterfactual"]["methods"][
            "scanner_then_predictor_topN"
        ]["sn_lift_vs_agentic_cio"] = None
        entry = leaderboard_entry_from_e2e_lift(e2e)
        assert entry is not None
        assert entry["sector_neutral_mean_alpha_21d"] == 0.019
        assert entry["sn_lift_vs_agentic_cio"] is None

    def test_entry_extraction_handles_missing_or_skipped(self):
        assert leaderboard_entry_from_e2e_lift(None) is None
        assert leaderboard_entry_from_e2e_lift({}) is None
        skipped = {"scanner_then_predictor_counterfactual": {"status": "skipped", "reason": "x"}}
        assert leaderboard_entry_from_e2e_lift(skipped) is None

    def test_build_leaderboard_appends_and_dedupes_by_date(self):
        history = [{"date": "2026-07-04", "sn_lift_vs_agentic_cio": 0.01}]
        entry = leaderboard_entry_from_e2e_lift(_e2e_lift_ok(sn_lift=0.02))
        artifact = build_leaderboard_artifact("2026-07-11", history, entry)
        dates = [p["date"] for p in artifact["weekly_points"]]
        assert dates == ["2026-07-04", "2026-07-11"]

    def test_history_scan_anchors_on_run_date_not_wall_clock(self):
        s3 = _FakeS3()
        prior_artifact = {
            "schema_version": 1, "as_of": "2020-01-03",
            "weekly_points": [{"date": "2020-01-03", "sn_lift_vs_agentic_cio": 0.011}],
        }
        write_leaderboard("bucket", "2020-01-03", prior_artifact, s3_client=s3)
        history = read_prior_leaderboard_history("bucket", "2020-01-10", s3_client=s3)
        assert history == [{"date": "2020-01-03", "sn_lift_vs_agentic_cio": 0.011}]

    def test_history_scan_cold_start_returns_empty(self):
        s3 = _FakeS3()
        history = read_prior_leaderboard_history("bucket", "2026-07-11", s3_client=s3)
        assert history == []

    def test_gate_inputs_reduction(self):
        artifact = {
            "weekly_points": [
                {"date": "2026-06-27", "sn_lift_vs_agentic_cio": 0.01},
                {"date": "2026-07-04", "sn_lift_vs_agentic_cio": 0.02},
                {"date": "2026-07-11", "sn_lift_vs_agentic_cio": None},
            ],
        }
        gi = leaderboard_gate_inputs(artifact)
        assert gi["challenger_matured_cohorts"] == 2
        assert gi["challenger_weekly_sn_lift"] == [0.01, 0.02]


class TestReadResearchProducerLeaderboard:
    """The NEW (I2518) read of crucible-research's real champion/challenger
    producer leaderboard — thinktank_coverage's evidence source."""

    def test_missing_key_returns_none(self):
        s3 = _FakeS3()
        assert read_research_producer_leaderboard("bucket", "2026-07-18", s3_client=s3) is None

    def test_reads_existing_artifact(self):
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-18.json",
                       Body=json.dumps(lb).encode())
        result = read_research_producer_leaderboard("bucket", "2026-07-18", s3_client=s3)
        assert result["date"] == "2026-07-18"
        names = [s["name"] for s in result["specs"]]
        assert "thinktank_coverage" in names

    def test_key_matches_crucible_research_producer(self):
        from optimizer.champion_promotion import RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL
        assert RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL == "research/producer_leaderboard/{date}.json"


class TestFindLatestResearchProducerLeaderboardDate:
    """alpha-engine-config-I2544 (2026-07-14): the async advisory child SF
    writing research/producer_leaderboard/{date}.json may lag or fail, so
    this gate lists the prefix and picks the latest date <= run_date rather
    than assuming an exact same-day key exists."""

    PREFIX = "research/producer_leaderboard"

    def _seed(self, s3, bucket, *dates):
        for d in dates:
            s3.put_object(Bucket=bucket, Key=f"{self.PREFIX}/{d}.json", Body=b"{}")

    def test_exact_date_present_is_selected(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-11", "2026-07-18")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-18"

    def test_falls_back_to_older_date_when_exact_missing(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-04", "2026-07-11")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_picks_max_not_first_among_multiple_older_dates(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-06-20", "2026-07-11", "2026-06-27")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_nothing_at_or_before_run_date_returns_none(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-25")  # only a date AFTER run_date
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None

    def test_empty_prefix_returns_none(self):
        s3 = _FakeS3()
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None

    def test_future_dated_key_never_selected_over_valid_past_date(self):
        s3 = _FakeS3()
        self._seed(s3, "bucket", "2026-07-11", "2026-07-25")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_malformed_keys_under_prefix_are_skipped_not_crashed(self):
        s3 = _FakeS3()
        s3.put_object(Bucket="bucket", Key=f"{self.PREFIX}/latest.json", Body=b"{}")
        s3.put_object(Bucket="bucket", Key=f"{self.PREFIX}/README.md", Body=b"x")
        self._seed(s3, "bucket", "2026-07-11")
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) == "2026-07-11"

    def test_list_failure_returns_none_not_raise(self):
        s3 = MagicMock()
        from botocore.exceptions import ClientError
        s3.list_objects_v2.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied"}}, "ListObjectsV2",
        )
        assert find_latest_research_producer_leaderboard_date(
            "bucket", "2026-07-18", s3_client=s3,
        ) is None


class TestReadLatestResearchProducerLeaderboard:
    """The combined list-then-read production entry point
    (alpha-engine-config-I2544) wired into evaluate.py."""

    def test_returns_leaderboard_and_date_used_for_exact_match(self):
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-18.json",
                       Body=json.dumps(lb).encode())
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert date_used == "2026-07-18"
        assert leaderboard["date"] == "2026-07-18"

    def test_falls_back_to_older_leaderboard_and_reports_its_date(self):
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-11")
        s3.put_object(Bucket="bucket", Key="research/producer_leaderboard/2026-07-11.json",
                       Body=json.dumps(lb).encode())
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert date_used == "2026-07-11"
        assert leaderboard["date"] == "2026-07-11"

    def test_nothing_available_returns_none_none(self):
        s3 = _FakeS3()
        leaderboard, date_used = read_latest_research_producer_leaderboard(
            "bucket", "2026-07-18", s3_client=s3,
        )
        assert leaderboard is None
        assert date_used is None


# ── Weekly audit record (config/apply_audit/producer_champion/{date}.json) ─


class TestBuildChampionAudit:
    def test_error_path_conforms_and_reports_unavailable_leaderboard(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="leaderboard missing")
        assert audit["outcome"] == "error"
        assert audit["blocked_by"] == ["leaderboard_unavailable"]
        assert audit["champion_before"] is None
        assert audit["schema_version"] == 2
        assert audit["leaderboard_date_used"] is None

    def test_no_contest_path_records_zero_pointer_movement(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
            "leaderboard_date_used": "2026-07-18",
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["outcome"] == "no_contest"
        assert audit["champion_before"] == audit["champion_after"] == "scanner_predictor_direct"
        assert audit["leaderboard_date_used"] == "2026-07-18"

    def test_promoted_path_records_leaderboard_date_used(self):
        """leaderboard_date_used must be recorded on the PROMOTED path too
        (not only no_contest) -- the audit trail must show which week's
        evidence decided a flip."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
            "leaderboard_date_used": "2026-07-11",
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["outcome"] == "promoted"
        assert audit["leaderboard_date_used"] == "2026-07-11"

    @pytest.mark.parametrize("outcome", OUTCOMES)
    def test_all_outcomes_are_in_frozen_vocabulary(self, outcome):
        assert outcome in (
            "promoted", "no_contest", "unchanged_winner_already_champion", "error",
        )


# ── Synthetic end-to-end transitions via run_weekly_evaluation ──────────────


class TestRunWeeklyEvaluation:
    BUCKET = "test-bucket"

    def test_challenger_wins_promotes_and_writes_pointer(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.03),
            tt_leaderboard_date_used="2026-07-18",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "thinktank_coverage"
        assert result["leaderboard_date_used"] == "2026-07-18"
        pointer = json.loads(s3.store[f"{self.BUCKET}/config/producer_champion.json"])
        assert pointer["champion"] == "thinktank_coverage"
        assert pointer["promotion_source"] == "gate_engine"
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/latest.json" in s3.store
        audit = json.loads(s3.store[f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json"])
        assert audit["leaderboard_date_used"] == "2026-07-18"

    def test_promotion_from_prior_weeks_leaderboard_records_that_date(self):
        """alpha-engine-config-I2544: a promotion can be decided using a
        leaderboard dated earlier than run_date (the latest available <=
        run_date, e.g. because this week's async child SF write hasn't
        landed yet) -- leaderboard_date_used must record the OLDER date
        actually consulted, not run_date."""
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-11", mean=0.03),
            tt_leaderboard_date_used="2026-07-11",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["leaderboard_date_used"] == "2026-07-11"
        audit = json.loads(s3.store[f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json"])
        assert audit["leaderboard_date_used"] == "2026-07-11"

    def test_champion_defends_no_pointer_write(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.01),
            tt_leaderboard_date_used="2026-07-18",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "unchanged_winner_already_champion"
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_no_contest_missing_tt_week_no_pointer_write(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=None,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["leaderboard_unavailable"]
        assert result["leaderboard_date_used"] is None
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        audit = json.loads(s3.store[f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json"])
        assert audit["leaderboard_date_used"] is None

    def test_no_contest_thinktank_not_yet_registered(self):
        """Mirrors the CURRENT real-world state (2026-07-14): thinktank
        _coverage's row is absent from the real leaderboard until
        crucible-research registers it -- must be an honest no-contest."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        lb["specs"] = [s for s in lb["specs"] if s["name"] != "thinktank_coverage"]
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=lb,
            tt_leaderboard_date_used="2026-07-18",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["thinktank_coverage_not_in_leaderboard"]
        assert result["leaderboard_date_used"] == "2026-07-18"

    def test_no_contest_leaderboard_stale_beyond_8_days(self):
        """alpha-engine-config-I2544: the latest leaderboard found is more
        than 8 calendar days older than run_date -- honest no-contest with
        the new slug, leaderboard_date_used still recorded (the audit
        trail shows what was found and rejected, not silence)."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-09", mean=0.03)
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=lb,
            tt_leaderboard_date_used="2026-07-09",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["leaderboard_stale_gt_8d"]
        assert result["leaderboard_date_used"] == "2026-07-09"
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_freeze_suppresses_pointer_write_but_audit_always_written(self):
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.03),
            tt_leaderboard_date_used="2026-07-18",
            freeze=True, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["blocked_by"] == ["frozen"]
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store

    def test_legacy_agentic_pointer_normalizes_without_crashing(self):
        """Belt-and-braces: a pointer object carrying the retired 'agentic'
        value must be readable/normalized without raising -- treated as
        scanner_predictor_direct for gate purposes. Not expected in
        practice (the live pointer has been scanner_predictor_direct since
        2026-07-13), but must be safe if it ever occurs."""
        s3 = _FakeS3()
        s3.put_object(
            Bucket=self.BUCKET, Key="config/producer_champion.json",
            Body=json.dumps({
                "schema_version": 1, "champion": "agentic",
                "promoted_at": "2026-07-01T00:00:00Z",
                "promotion_source": "operator_bootstrap",
            }).encode(),
        )
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=_tt_leaderboard_ok(run_date="2026-07-18", mean=0.01),
            tt_leaderboard_date_used="2026-07-18",
            freeze=False, upload=True, s3_client=s3,
        )
        # champion_before normalized to scanner_predictor_direct, which wins
        # this synthetic week (0.03 > 0.01) -> unchanged (already champion).
        assert result["champion_before"] == "scanner_predictor_direct"
        assert result["outcome"] == "unchanged_winner_already_champion"

    def test_scoring_exception_is_error_outcome_but_still_audited(self):
        """A malformed thinktank_coverage row (topn_alpha_vs_benchmark.mean
        is non-numeric) raises inside _score_thinktank_coverage's float()
        call -- run_weekly_evaluation's own try/except must catch it,
        record outcome='error', and STILL write the audit record (the
        liveness proxy, config#2054) rather than propagating the crash."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        for spec in lb["specs"]:
            if spec["name"] == "thinktank_coverage":
                spec["topn_alpha_vs_benchmark"] = {"mean": "not-a-number"}
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date="2026-07-18",
            e2e_lift=_e2e_lift_ok(sn_lift=0.03),
            tt_leaderboard=lb,
            tt_leaderboard_date_used="2026-07-18",
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "error"
        assert result["champion_before"] is None
        assert result["champion_after"] is None
        assert result["leaderboard_date_used"] is None
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store
        assert f"{self.BUCKET}/config/producer_champion.json" not in s3.store

    def test_scoring_exception_publishes_active_alert(self):
        """config#2884: an outcome='error' week must fire an active alert,
        not just the passive audit-JSON write -- the only prior liveness
        signal (ARTIFACT_REGISTRY file-presence SLA) is satisfied by a
        routine error write, so a persistently-erroring gate could freeze
        the champion pointer for an unbounded number of weeks with nobody
        paged. Same malformed-leaderboard trigger as the test above."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        for spec in lb["specs"]:
            if spec["name"] == "thinktank_coverage":
                spec["topn_alpha_vs_benchmark"] = {"mean": "not-a-number"}
        with mock.patch("ops_alerts.publish_ops_alert") as mock_publish:
            result = run_weekly_evaluation(
                bucket=self.BUCKET, run_date="2026-07-18",
                e2e_lift=_e2e_lift_ok(sn_lift=0.03),
                tt_leaderboard=lb,
                tt_leaderboard_date_used="2026-07-18",
                freeze=False, upload=True, s3_client=s3,
            )
        assert result["outcome"] == "error"
        mock_publish.assert_called_once()
        _, kwargs = mock_publish.call_args
        assert kwargs["severity"] == "error"
        assert kwargs["dedup_key"] == "champion_promotion_gate_error_2026-07-18"
        assert "champion_promotion.py::run_weekly_evaluation" in kwargs["source"]

    def test_alert_publish_failure_does_not_propagate(self):
        """The alert channel itself failing (e.g. SNS unreachable) must not
        crash the already-erroring evaluate run -- best-effort, swallowed,
        same posture as the audit-JSON write's own failure handling."""
        s3 = _FakeS3()
        lb = _tt_leaderboard_ok(run_date="2026-07-18")
        for spec in lb["specs"]:
            if spec["name"] == "thinktank_coverage":
                spec["topn_alpha_vs_benchmark"] = {"mean": "not-a-number"}
        with mock.patch(
            "ops_alerts.publish_ops_alert", side_effect=RuntimeError("sns down"),
        ):
            result = run_weekly_evaluation(
                bucket=self.BUCKET, run_date="2026-07-18",
                e2e_lift=_e2e_lift_ok(sn_lift=0.03),
                tt_leaderboard=lb,
                tt_leaderboard_date_used="2026-07-18",
                freeze=False, upload=True, s3_client=s3,
            )
        assert result["outcome"] == "error"
        assert f"{self.BUCKET}/config/apply_audit/producer_champion/2026-07-18.json" in s3.store


# ── Frozen-schema conformance ────────────────────────────────────────────────


class TestSchemaConformance:
    def _validate(self, schema_path_or_dict, instance):
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        schema = (
            schema_path_or_dict
            if isinstance(schema_path_or_dict, dict)
            else json.loads(schema_path_or_dict.read_text())
        )
        jsonschema.validate(instance=instance, schema=schema)

    def test_pointer_conforms(self):
        pointer = write_champion_pointer(
            "bucket", "scanner_predictor_direct",
            promotion_source="gate_engine", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_pointer_conforms_thinktank_coverage(self):
        pointer = write_champion_pointer(
            "bucket", "thinktank_coverage", promotion_source="gate_engine", upload=False,
        )
        self._validate(POINTER_SCHEMA_PATH, pointer)

    def test_legacy_agentic_pointer_shape_is_schema_valid(self):
        """Read-tolerance: a historical pointer-shaped object with
        champion='agentic' must still validate against the schema (the
        schema enum keeps 'agentic' for exactly this reason) even though
        write_champion_pointer itself refuses to produce one."""
        legacy = {
            "schema_version": 1, "champion": "agentic",
            "promoted_at": "2026-07-01T00:00:00Z",
            "promotion_source": "operator_bootstrap",
        }
        self._validate(POINTER_SCHEMA_PATH, legacy)

    def test_promoted_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA, audit)

    def test_no_contest_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "thinktank_coverage_not_in_leaderboard"},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA, audit)

    def test_unchanged_winner_audit_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        self._validate(AUDIT_SCHEMA, audit)

    def test_error_audit_conforms(self):
        audit = build_champion_audit("2026-07-18", None, freeze=False, error="boom")
        self._validate(AUDIT_SCHEMA, audit)

    def test_legacy_v1_audit_shape_still_schema_valid_by_git_history(self):
        """v1 historical records are NOT expected to validate against the
        v2 schema (schema_version const changed, fields renamed) -- this is
        by design (module/schema docstrings): v1 documents remain valid
        under the FROZEN v1 shape recoverable via git history, and this
        repo's tests only ever validate newly-built (v2) records. This test
        simply pins that expectation so a future reader doesn't mistake the
        absence of v1 conformance testing for an oversight."""
        legacy_v1 = {
            "schema_version": 1, "date": "2026-07-13", "outcome": "promoted",
            "champion_before": "agentic", "champion_after": "scanner_predictor_direct",
            "challenger_matured_cohorts": 0, "sn_lift_vs_champion": None,
            "consecutive_wins": 0, "cooldown_until": "2026-07-27", "blocked_by": None,
        }
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        schema = AUDIT_SCHEMA
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(instance=legacy_v1, schema=schema)

    def test_valid_champions_subset_of_schema_enum(self):
        """The schema enum is a SUPERSET of VALID_CHAMPIONS (it additionally
        read-tolerates the retired 'agentic' seat) -- not an exact-set
        match like the pre-I2518 engine had, by design."""
        schema = json.loads(POINTER_SCHEMA_PATH.read_text())
        enum = set(schema["properties"]["champion"]["enum"])
        assert set(VALID_CHAMPIONS).issubset(enum)
        assert "agentic" in enum

    def test_stale_no_contest_audit_conforms_and_carries_date_used(self):
        """alpha-engine-config-I2544: the leaderboard_stale_gt_8d no-contest
        record must both conform to the (additive) v2 schema and carry the
        leaderboard_date_used field naming what was found and rejected."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": None},
            "unavailable_reasons": {"thinktank_coverage": "leaderboard_stale_gt_8d"},
            "leaderboard_date_used": "2026-07-09",
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["blocked_by"] == ["leaderboard_stale_gt_8d"]
        assert audit["leaderboard_date_used"] == "2026-07-09"
        self._validate(AUDIT_SCHEMA, audit)

    def test_promoted_audit_carries_leaderboard_date_used_and_conforms(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
            "leaderboard_date_used": "2026-07-11",
        }
        gate_result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
        )
        audit = build_champion_audit("2026-07-18", gate_result, freeze=False)
        assert audit["leaderboard_date_used"] == "2026-07-11"
        self._validate(AUDIT_SCHEMA, audit)

    def test_leaderboard_stale_gt_8d_slug_in_schema_enum(self):
        schema = AUDIT_SCHEMA
        slugs = set(schema["properties"]["blocked_by"]["oneOf"][1]["items"]["enum"])
        assert "leaderboard_stale_gt_8d" in slugs

    def test_feed_producer_dead_slug_in_schema_enum(self):
        schema = AUDIT_SCHEMA
        slugs = set(schema["properties"]["blocked_by"]["oneOf"][1]["items"]["enum"])
        assert "feed_producer_dead" in slugs

    def test_feed_dependencies_field_declared_in_schema(self):
        schema = AUDIT_SCHEMA
        assert "feed_dependencies" in schema["properties"]
        assert "feed_dependencies" not in schema["required"]  # additive, optional

    def test_promoted_audit_with_feed_dependencies_conforms(self):
        audit = build_champion_audit(
            "2026-07-18",
            evaluate_gates(
                champion_before="thinktank_coverage",
                arm_scores={
                    "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
                    "unavailable_reasons": {},
                },
                freeze=False,
            ),
            freeze=False,
        )
        assert audit["outcome"] == "promoted"
        assert audit["champion_after"] == "scanner_predictor_direct"
        assert audit["feed_dependencies"] == ["research_free_backfill"]
        self._validate(AUDIT_SCHEMA, audit)


# ── Promotion-time feed-dependency liveness gate (alpha-engine-config-I3165)
# ────────────────────────────────────────────────────────────────────────────
#
# config#3053's root cause, restated as this gate's closes-when bar: a
# promotion record must NAME the promoted arm's upstream feed dependency
# (ARM_FEED_DEPENDENCIES / build_champion_audit's feed_dependencies field,
# covered by TestSchemaConformance above and TestArmFeedDependencies below),
# and a synthetic test must demonstrate the gate BLOCKING a promotion whose
# declared feed has no live producer (TestCheckFeedDependenciesLive /
# TestEvaluateGatesFeedLiveness / TestRunWeeklyEvaluationFeedLiveness below)
# -- as well as the mirror-image case, a promotion proceeding normally when
# the declared feed IS live, so the new gate cannot be mistaken for an
# always-block regression.


class TestArmFeedDependencies:
    def test_scanner_predictor_direct_declares_research_free_backfill(self):
        assert ARM_FEED_DEPENDENCIES["scanner_predictor_direct"] == ["research_free_backfill"]

    def test_thinktank_coverage_declares_no_feed_dependency(self):
        """thinktank_coverage's evidence chain is the producer leaderboard,
        already gated by leaderboard_date_used/leaderboard_stale_gt_8d -- it
        names no live-trade feed artifact of its own."""
        assert ARM_FEED_DEPENDENCIES.get("thinktank_coverage") in (None, [])


class TestCheckFeedDependenciesLive:
    BUCKET = "test-bucket"

    def test_no_declared_dependency_is_always_live(self):
        """An arm with no ARM_FEED_DEPENDENCIES entry (thinktank_coverage)
        is trivially never blocked by this gate -- it has nothing to
        probe."""
        s3 = _FakeS3()
        assert check_feed_dependencies_live(
            "thinktank_coverage", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) is None

    def test_live_fresh_feed_passes(self):
        """(a) The declared feed's producer IS live/fresh -- the gate
        returns None (not blocked)."""
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-17")
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) is None

    def test_stale_feed_is_blocked(self):
        """(b) The declared feed's newest prediction_date is stale beyond
        the freshness window (config#3053's exact incident shape: the
        producer silently stopped refreshing) -- blocked with the new
        slug."""
        s3 = _FakeS3()
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-01")
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) == "feed_producer_dead"

    def test_missing_feed_artifact_is_blocked(self):
        """The declared feed artifact does not exist at all (orphaned
        producer, never wrote anything) -- blocked, not a crash."""
        s3 = _FakeS3()  # nothing uploaded
        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
        ) == "feed_producer_dead"

    def test_probe_exception_is_blocked_not_raised(self):
        """Belt-and-braces: even an UNEXPECTED exception from the
        registered prober (not just the StaleChampionFeedError it's
        designed to raise) must degrade to feed_producer_dead, never
        propagate -- the module's binding config#2884 lesson applies to
        this gate exactly as much as to the rest of evaluate_gates."""
        s3 = _FakeS3()

        class _ExplodingS3:
            def get_object(self, Bucket, Key):
                raise RuntimeError("boom - unexpected probe failure")

        assert check_feed_dependencies_live(
            "scanner_predictor_direct", bucket=self.BUCKET, run_date="2026-07-20",
            s3_client=_ExplodingS3(),
        ) == "feed_producer_dead"

    def test_unregistered_feed_dependency_fails_open_without_crashing(self):
        """An arm declaring a feed id with no registered prober in
        _FEED_LIVENESS_PROBES must not crash this gate -- it's simply not
        checked (logged), never a silent block or a crash. Verified via a
        monkeypatched ARM_FEED_DEPENDENCIES entry rather than mutating the
        real one."""
        import optimizer.champion_promotion as cp

        original = dict(cp.ARM_FEED_DEPENDENCIES)
        cp.ARM_FEED_DEPENDENCIES["thinktank_coverage"] = ["some_unregistered_feed"]
        try:
            s3 = _FakeS3()
            result = check_feed_dependencies_live(
                "thinktank_coverage", bucket=self.BUCKET, run_date="2026-07-20", s3_client=s3,
            )
            assert result is None
        finally:
            cp.ARM_FEED_DEPENDENCIES.clear()
            cp.ARM_FEED_DEPENDENCIES.update(original)


class TestEvaluateGatesFeedLiveness:
    def test_feed_blocked_slug_degrades_would_be_promotion_to_no_contest(self):
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
            feed_blocked_slug=None,
        )
        assert result["outcome"] == "promoted"  # sanity: unblocked path still promotes

        blocked = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
            feed_blocked_slug="feed_producer_dead",
        )
        assert blocked["outcome"] == "no_contest"
        assert blocked["blocked_by"] == ["feed_producer_dead"]
        assert blocked["champion_after"] == "scanner_predictor_direct"  # pointer never moves

    def test_feed_blocked_slug_irrelevant_when_challenger_does_not_win(self):
        """The feed check only matters on the WIN path -- an incumbent that
        defends its title, or a no-contest week, must not be affected by
        the challenger's feed liveness (nothing would move regardless)."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.03, "thinktank_coverage": 0.01},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=False,
            feed_blocked_slug="feed_producer_dead",
        )
        assert result["outcome"] == "unchanged_winner_already_champion"

    def test_feed_blocked_slug_takes_priority_over_freeze(self):
        """A dead feed must degrade to no_contest even under --freeze --
        the audit trail should show the TRUE validity-guard reason, not a
        suppression that implies the promotion was otherwise valid."""
        arm_scores = {
            "scores": {"scanner_predictor_direct": 0.01, "thinktank_coverage": 0.03},
            "unavailable_reasons": {},
        }
        result = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores, freeze=True,
            feed_blocked_slug="feed_producer_dead",
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["feed_producer_dead"]


class TestRunWeeklyEvaluationFeedLiveness:
    """End-to-end via run_weekly_evaluation -- the actual wiring evaluate.py
    calls. Demonstrates both halves of the issue's closes-when bar: (a) a
    promotion proceeds normally when the declared feed's producer is
    live/fresh, and (b) the gate blocks (degrades to no_contest) when the
    declared feed's producer looks dead/orphaned."""

    BUCKET = "test-bucket"
    RUN_DATE = "2026-07-20"  # the real config#3053 incident date

    def test_promotion_proceeds_when_challenger_feed_is_live(self):
        """(a) THE mirror-image synthetic test: same exact setup as
        ``test_promotion_onto_dead_feed_degrades_to_no_contest`` below
        (thinktank_coverage is champion_before, scanner_predictor_direct is
        the challenger and wins this week on score) except its declared
        feed (research_free_backfill) IS live/fresh -- the promotion must
        proceed normally and the pointer must move, exactly as it would
        have before this gate existed. Demonstrates the new gate is not an
        always-block regression."""
        s3 = _FakeS3()
        s3.put_object(
            Bucket=self.BUCKET, Key="config/producer_champion.json",
            Body=json.dumps({
                "schema_version": 1, "champion": "thinktank_coverage",
                "promoted_at": "2026-07-13T00:00:00Z",
                "promotion_source": "gate_engine",
            }).encode(),
        )
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-17")
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date=self.RUN_DATE,
            e2e_lift=_e2e_lift_ok(sn_lift=0.05),          # scanner_predictor_direct's score
            tt_leaderboard=_tt_leaderboard_ok(run_date=self.RUN_DATE, mean=0.01),  # thinktank_coverage's score
            tt_leaderboard_date_used=self.RUN_DATE,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "promoted"
        assert result["champion_before"] == "thinktank_coverage"
        assert result["champion_after"] == "scanner_predictor_direct"
        assert result["blocked_by"] is None
        pointer_key = f"{self.BUCKET}/config/producer_champion.json"
        assert json.loads(s3.store[pointer_key])["champion"] == "scanner_predictor_direct"
        audit = json.loads(s3.store[f"{self.BUCKET}/config/apply_audit/producer_champion/{self.RUN_DATE}.json"])
        assert audit["outcome"] == "promoted"
        assert audit["blocked_by"] is None
        assert audit["feed_dependencies"] == ["research_free_backfill"]

    def test_promotion_onto_dead_feed_degrades_to_no_contest(self):
        """(b) THE synthetic test the issue's closes-when bar asks for:
        scanner_predictor_direct would win this week on score alone
        (thinktank_coverage incumbent, scanner_predictor_direct challenger,
        higher score) but its declared feed_dependencies
        (research_free_backfill) has no live producer -- the champion
        pointer must NOT move, and the audit record must show
        blocked_by=['feed_producer_dead'], not a fabricated
        unchanged/promoted outcome and not a crash."""
        s3 = _FakeS3()
        # Seed the pointer so thinktank_coverage is champion_before and
        # scanner_predictor_direct is genuinely the winning CHALLENGER.
        s3.put_object(
            Bucket=self.BUCKET, Key="config/producer_champion.json",
            Body=json.dumps({
                "schema_version": 1, "champion": "thinktank_coverage",
                "promoted_at": "2026-07-13T00:00:00Z",
                "promotion_source": "gate_engine",
            }).encode(),
        )
        # Deliberately do NOT write a research_free_backfill parquet at all
        # -- the config#3053 shape: the producer's ultimate upstream was
        # orphaned and nothing was ever written this cycle.
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date=self.RUN_DATE,
            e2e_lift=_e2e_lift_ok(sn_lift=0.05),          # scanner_predictor_direct's score
            tt_leaderboard=_tt_leaderboard_ok(run_date=self.RUN_DATE, mean=0.01),  # thinktank_coverage's score
            tt_leaderboard_date_used=self.RUN_DATE,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["feed_producer_dead"]
        assert result["champion_before"] == "thinktank_coverage"
        assert result["champion_after"] == "thinktank_coverage"  # pointer never moved
        pointer_key = f"{self.BUCKET}/config/producer_champion.json"
        # Pointer object is untouched -- still the seeded thinktank_coverage
        # pointer, never overwritten with scanner_predictor_direct.
        assert json.loads(s3.store[pointer_key])["champion"] == "thinktank_coverage"
        audit = json.loads(s3.store[f"{self.BUCKET}/config/apply_audit/producer_champion/{self.RUN_DATE}.json"])
        assert audit["outcome"] == "no_contest"
        assert audit["blocked_by"] == ["feed_producer_dead"]
        # feed_dependencies still names what champion_after (unchanged)
        # would need if it had a declared dependency -- thinktank_coverage
        # has none, so this is None, not a stale scanner_predictor_direct
        # value left over from the blocked would-be promotion.
        assert audit["feed_dependencies"] is None

    def test_stale_feed_also_blocks_promotion(self):
        """Same closes-when scenario, but the feed artifact EXISTS and is
        readable -- just stale (the producer stopped refreshing rather
        than never having run at all). Must block identically."""
        s3 = _FakeS3()
        s3.put_object(
            Bucket=self.BUCKET, Key="config/producer_champion.json",
            Body=json.dumps({
                "schema_version": 1, "champion": "thinktank_coverage",
                "promoted_at": "2026-07-13T00:00:00Z",
                "promotion_source": "gate_engine",
            }).encode(),
        )
        _put_research_free_backfill_parquet(s3, self.BUCKET, newest_prediction_date="2026-07-01")
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date=self.RUN_DATE,
            e2e_lift=_e2e_lift_ok(sn_lift=0.05),
            tt_leaderboard=_tt_leaderboard_ok(run_date=self.RUN_DATE, mean=0.01),
            tt_leaderboard_date_used=self.RUN_DATE,
            freeze=False, upload=True, s3_client=s3,
        )
        assert result["outcome"] == "no_contest"
        assert result["blocked_by"] == ["feed_producer_dead"]
        pointer_key = f"{self.BUCKET}/config/producer_champion.json"
        assert json.loads(s3.store[pointer_key])["champion"] == "thinktank_coverage"

    def test_champion_defending_own_seat_is_unaffected_by_challenger_feed_liveness(self):
        """A no-op week (incumbent defends, or the challenger loses on
        score) must not be perturbed by this gate at all -- feed liveness
        of a NON-winning challenger is irrelevant since the pointer would
        not move either way. No research_free_backfill artifact is written
        (feed looks dead) but scanner_predictor_direct challenges and LOSES
        on score, so the outcome must be the ordinary unchanged path, not
        a feed-liveness no_contest."""
        s3 = _FakeS3()
        result = run_weekly_evaluation(
            bucket=self.BUCKET, run_date=self.RUN_DATE,
            e2e_lift=_e2e_lift_ok(sn_lift=0.01),           # scanner_predictor_direct loses
            tt_leaderboard=_tt_leaderboard_ok(run_date=self.RUN_DATE, mean=0.05),  # thinktank_coverage's score N/A (it's champion_before here... )
            tt_leaderboard_date_used=self.RUN_DATE,
            freeze=False, upload=True, s3_client=s3,
        )
        # champion_before defaults to scanner_predictor_direct (pre
        # -bootstrap base case) since no pointer was seeded; challenger is
        # thinktank_coverage, which wins here (0.05 > 0.01) -- a genuine
        # promotion onto thinktank_coverage, which declares NO feed
        # dependency, so the missing research_free_backfill artifact must
        # not block it.
        assert result["outcome"] == "promoted"
        assert result["champion_after"] == "thinktank_coverage"
        assert result["blocked_by"] is None


# ── Champion-side evidence floor (alpha-engine-config-I7549, second half) ──


class TestChampionSideEvidenceFloor:
    """#688 floored the arm scored off the producer leaderboard. This class
    pins the OTHER arm of the same two-arm gate.

    ``analysis/end_to_end.py::_scanner_then_predictor_topN`` returns a result
    at ``if n_cycles < 1``, so ``scanner_predictor_direct`` could enter the
    comparison as a single observation while ``thinktank_coverage`` was held to
    five date clusters. The gate acts on the DIFFERENCE between the two: a thin
    champion makes that difference noise exactly as a thin challenger does, and
    the audit record reads the same either way.

    Verified RED against #688's code (champion-challenger-policy.md §7.4):
    ``_score_scanner_predictor_direct`` there returns a 2-tuple with no floor,
    so ``test_one_cycle_champion_cannot_defend_or_flip`` promotes on a
    one-cycle mean.
    """

    def test_one_cycle_champion_cannot_defend_or_flip(self):
        """A one-cycle champion score is not evidence in either direction: it
        cannot defend the incumbency and it cannot lose it."""
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.09, n_cycles=1),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] is None
        assert result["unavailable_reasons"]["scanner_predictor_direct"] == (
            "scanner_predictor_direct_thin_evidence"
        )
        assert result["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_THIN
        # ... and the well-evidenced challenger does not win by default either.
        gate = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=result, freeze=False,
        )
        assert gate["outcome"] == "no_contest"
        assert gate["champion_after"] == "scanner_predictor_direct"
        assert gate["blocked_by"] == ["scanner_predictor_direct_thin_evidence"]

    def test_thin_champion_cannot_be_flipped_off_by_a_challenger(self):
        """Reversed seats — the same protection, in the direction that moves
        the pointer. thinktank_coverage is champion; a one-cycle
        scanner_predictor_direct number must not promote."""
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.5, n_cycles=1),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        gate = evaluate_gates(
            champion_before="thinktank_coverage", arm_scores=result, freeze=False,
        )
        assert gate["outcome"] == "no_contest"
        assert gate["champion_after"] == "thinktank_coverage"

    def test_at_the_floor_scores_exactly_as_before(self):
        assert MIN_CYCLES_FOR_INFERENCE == 5
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.09, n_cycles=MIN_CYCLES_FOR_INFERENCE),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] == 0.09
        assert result["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_OK
        assert result["unavailable_reasons"] == {}

    def test_live_cycle_count_is_above_the_floor(self):
        """The floor bounds the degenerate case; it must not freeze the live
        gate. Measured 2026-08-17 from
        research/producer_leaderboard_champion_gate/2026-08-14.json:
        n_cycles=15, n_picks=119."""
        result = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=-0.00203, n_cycles=15),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] == -0.00203
        assert result["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_OK

    def test_missing_n_cycles_is_refused_not_trusted(self):
        """Same posture leaderboard_row_confidence takes toward an artifact
        that cannot supply a usable verdict: refuse, never guess."""
        lift = _e2e_lift_ok(sn_lift=0.09)
        del lift["scanner_then_predictor_counterfactual"]["n_cycles"]
        result = build_weekly_arm_scores(
            lift, _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["scores"]["scanner_predictor_direct"] is None
        assert result["unavailable_reasons"]["scanner_predictor_direct"] == (
            "scanner_predictor_direct_confidence_unknown"
        )
        assert result["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_UNKNOWN

    def test_absent_counterfactual_keeps_its_existing_slug(self):
        """No behaviour change where the arm produced nothing at all — that
        was never the defect, and the slug other consumers already match on
        must not move."""
        result = build_weekly_arm_scores(
            None, _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        assert result["unavailable_reasons"]["scanner_predictor_direct"] == (
            "scanner_predictor_direct_counterfactual_unavailable"
        )
        assert result["arm_confidence"]["scanner_predictor_direct"] == (
            CONFIDENCE_INSUFFICIENT
        )

    def test_champion_side_verdict_reaches_the_audit_record(self):
        arm_scores = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.09, n_cycles=2),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=9),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        record = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores,
            freeze=False,
        )
        audit = build_champion_audit("2026-07-18", record, freeze=False)
        assert audit["arm_confidence"]["scanner_predictor_direct"] == CONFIDENCE_THIN
        assert audit["arm_confidence"]["thinktank_coverage"] == CONFIDENCE_OK
        assert audit["blocked_by"] == ["scanner_predictor_direct_thin_evidence"]
        jsonschema = pytest.importorskip("jsonschema", reason="jsonschema not installed")
        jsonschema.validate(
            instance=audit, schema=AUDIT_SCHEMA,
        )

    def test_both_arms_thin_names_both(self):
        arm_scores = build_weekly_arm_scores(
            _e2e_lift_ok(sn_lift=0.09, n_cycles=2),
            _tt_leaderboard_ok(mean=0.01, n_dates_scored=1),
            run_date="2026-07-18", leaderboard_date_used="2026-07-18",
        )
        record = evaluate_gates(
            champion_before="scanner_predictor_direct", arm_scores=arm_scores,
            freeze=False,
        )
        assert record["blocked_by"] == [
            "scanner_predictor_direct_thin_evidence",
            "thinktank_coverage_thin_evidence",
        ]

    def test_champion_side_slugs_are_registered_in_the_schema_enum(self):
        schema = AUDIT_SCHEMA
        enum = set(schema["properties"]["blocked_by"]["oneOf"][1]["items"]["enum"])
        for slug in (
            "scanner_predictor_direct_thin_evidence",
            "scanner_predictor_direct_confidence_unknown",
        ):
            assert slug in _BLOCKED_BY_SLUGS
            assert slug in enum
