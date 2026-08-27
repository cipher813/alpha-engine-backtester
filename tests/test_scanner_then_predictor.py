"""Tests for the scanner -> research-free predictor direct counterfactual
(config#1405, arm 4 of the agentic-ablation ladder).

The fixture makes realized 21d alpha selection-driven (winners = low ``i``),
has the research-free predictor's ``predicted_alpha`` favour the winners, and
has the live agentic CIO ADVANCE the losers — so the count-matched
``scanner_then_predictor_topN`` must beat BOTH the actual scanner pass pool and
the agentic CIO selection. Mirrors ``test_scanner_factor_counterfactual.py``.

The meta-ensemble backfill that populates ``predictor_outcomes_research_free``
runs only on the Saturday spot box (ArcticDB-gated); this exercises the
analysis/consumer layer with a synthetic fixture, no ArcticDB needed.
"""

from __future__ import annotations

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.end_to_end import _scanner_then_predictor_topN  # noqa: E402


def _db(tmp_path, *, cio_decision_for=lambda i: "ADVANCE" if i >= 7 else "REJECT"):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute(
        "CREATE TABLE predictor_outcomes_research_free (ticker TEXT, prediction_date TEXT, "
        "predicted_alpha REAL, n_research_features_missing INTEGER)"
    )
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT)")
    dates = ["2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"]
    for d in dates:
        for i in range(10):
            alpha = 0.03 if i < 5 else -0.03  # winners = low i
            # the live scanner passes the WHOLE pool (10/cycle) -> the predictor re-ranks it
            conn.execute("INSERT INTO scanner_evaluations VALUES (?,?,?)", (f"T{i}", d, 1))
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)", (f"T{i}", d, "Tech", alpha, 0.0)
            )
            # research-free predicted_alpha favours the winners (low i -> high score);
            # 4 research meta-features omitted -> n_research_features_missing = 4
            conn.execute(
                "INSERT INTO predictor_outcomes_research_free VALUES (?,?,?,?)",
                (f"T{i}", d, float(10 - i), 4),
            )
            # the live agentic CIO advances the LOSERS (high i) -> agentic underperforms
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?)", (f"T{i}", d, cio_decision_for(i))
            )
    conn.commit()
    return conn


def test_predictor_beats_agentic_and_scanner(tmp_path):
    conn = _db(tmp_path)
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4, r
    m = r["methods"]
    # research-free predictor picks the winners -> positive
    assert m["scanner_then_predictor_topN"]["mean_alpha_21d"] > 0, r
    # the live agentic CIO advanced the losers -> negative (the path being replaced)
    assert m["agentic_cio_advance"]["mean_alpha_21d"] < 0, r
    # the scanner pass pool is a 50/50 mix -> ~0
    assert abs(m["actual_scanner_pass"]["mean_alpha_21d"]) < 1e-6, r
    # both lifts positive
    assert m["scanner_then_predictor_topN"]["lift_vs_actual_scanner"] > 0, r
    assert m["scanner_then_predictor_topN"]["lift_vs_agentic_cio"] > 0, r
    assert m["scanner_then_predictor_topN"]["sn_lift_vs_actual_scanner"] is not None, r
    assert r["predictor_beats_agentic_cio"] is True, r
    assert r["predictor_beats_actual_scanner"] is True, r
    # count-match: 3 advance/cycle x 4 = 12 predictor picks; agentic 12; scanner pool 40
    assert m["scanner_then_predictor_topN"]["n_picks"] == 12, r
    assert m["agentic_cio_advance"]["n_picks"] == 12, r
    assert m["actual_scanner_pass"]["n_picks"] == 40, r
    # research-free guard: every prediction omitted the 4 research meta-features
    assert r["research_features_missing_mode"] == 4, r


def test_advance_forced_counts_as_agentic(tmp_path):
    """``ADVANCE_FORCED`` (the force-fill path) is part of the agentic selection."""
    conn = _db(tmp_path, cio_decision_for=lambda i: "ADVANCE_FORCED" if i >= 7 else "REJECT")
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "ok", r
    assert r["methods"]["agentic_cio_advance"]["n_picks"] == 12, r
    assert r["predictor_beats_agentic_cio"] is True, r


def test_skipped_without_predictions_table(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    conn.execute(
        "CREATE TABLE scanner_evaluations (ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute("CREATE TABLE cio_evaluations (ticker TEXT, eval_date TEXT, cio_decision TEXT)")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r
    assert "predictor_outcomes_research_free" in r["reason"], r


def test_skipped_without_predicted_alpha_column(tmp_path):
    conn = _db(tmp_path)
    conn.execute("DROP TABLE predictor_outcomes_research_free")
    conn.execute("CREATE TABLE predictor_outcomes_research_free (ticker TEXT, prediction_date TEXT)")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r
    assert "predicted_alpha" in r["reason"], r


def test_skipped_when_no_predictions_match(tmp_path):
    """Table exists but the backfill hasn't populated rows yet -> honest skip."""
    conn = _db(tmp_path)
    conn.execute("DELETE FROM predictor_outcomes_research_free")
    conn.commit()
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "skipped", r


def test_insufficient_without_cio_advance(tmp_path):
    """No agentic ADVANCE anywhere -> no count-match basis -> insufficient_data."""
    conn = _db(tmp_path, cio_decision_for=lambda i: "REJECT")
    r = _scanner_then_predictor_topN(conn)
    assert r["status"] == "insufficient_data", r


# ── source staleness (alpha-engine-config-I8757) ─────────────────────────
#
# Both tables this counterfactual joins — scanner_evaluations and
# cio_evaluations — were written by the six-team Research LangGraph, retired
# 2026-07-12. The reads never failed; the data stopped changing. Measured
# 2026-08-27 on the live research.db: universe_returns ran to 2026-08-14 while
# scanner_evaluations stopped at 2026-07-17 and cio_evaluations at 2026-07-10,
# and this function returned status="ok" with a number computed entirely from
# cohorts on or before 2026-07-10. The champion gate consumed that identical
# number (sector_neutral_mean_alpha_21d=-0.00203, n_cycles=15, n_picks=119) on
# 2026-08-13, 2026-08-14 AND 2026-08-21.
#
# Every test below is RED without the guard: the pre-change function computes
# and returns status="ok" regardless of how far behind its sources are.


def _age_the_retired_tables(conn, newest_return_date: str):
    """Advance universe_returns past the retired-writer tables, reproducing the
    live shape: fresh returns, frozen sources."""
    conn.execute(
        "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
        ("T0", newest_return_date, "Tech", 0.01, 0.0),
    )
    conn.commit()


def test_stale_sources_are_unmeasurable_not_ok(tmp_path):
    conn = _db(tmp_path)
    # Sources end 2026-01-23; returns run to 2026-03-01 — 37 days, past the 14d bound.
    _age_the_retired_tables(conn, "2026-03-01")

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "unmeasurable", r
    assert "scanner_evaluations" in r["reason"]
    assert "cio_evaluations" in r["reason"]
    assert r["reference_date"] == "2026-03-01"
    stale = {row["table"]: row for row in r["source_freshness"]}
    assert stale["cio_evaluations"]["stale"] is True
    assert stale["cio_evaluations"]["newest_date"] == "2026-01-23"


def test_an_unmeasurable_result_yields_no_leaderboard_point(tmp_path):
    """The whole consequence chain: unmeasurable -> no point -> the champion
    gate has no score -> no_contest -> the pointer is held. Which is what the
    gate would have done all along had the frozen number not been available to
    mislead it."""
    from optimizer.champion_promotion import leaderboard_entry_from_e2e_lift

    conn = _db(tmp_path)
    _age_the_retired_tables(conn, "2026-03-01")

    r = _scanner_then_predictor_topN(conn)

    assert leaderboard_entry_from_e2e_lift(
        {"scanner_then_predictor_counterfactual": r}
    ) is None


def test_one_missed_cycle_still_measures(tmp_path):
    """The bound is 14d against a 7d cadence. A single late weekly run must not
    blank the arm's score — refusing on a transient is its own failure mode."""
    conn = _db(tmp_path)
    _age_the_retired_tables(conn, "2026-02-04")  # 12 days past 2026-01-23

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "ok", r


def test_a_healthy_run_states_its_source_ages(tmp_path):
    """Emitted on the healthy path too — an absent field is unmeasured, not
    fine, and a frozen source was indistinguishable from a fresh one for seven
    weeks precisely because nothing said how old it was."""
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "ok"
    assert r["reference_date"] == "2026-01-23"
    ages = {row["table"]: row["age_days"] for row in r["source_freshness"]}
    assert ages == {"scanner_evaluations": 0, "cio_evaluations": 0}


def test_reference_date_is_data_derived_so_an_archived_db_still_reads_true(tmp_path):
    """Re-running this over an archived research.db must report what was true
    THEN, not fail for being old. The reference is MAX(universe_returns.eval_date),
    never wall-clock."""
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn)

    assert r["reference_date"] == "2026-01-23"
    assert r["status"] == "ok"


def test_an_explicit_reference_date_overrides_the_derived_one(tmp_path):
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn, reference_date="2026-06-01")

    assert r["status"] == "unmeasurable", r
    assert r["reference_date"] == "2026-06-01"
