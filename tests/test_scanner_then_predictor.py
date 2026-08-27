"""``_scanner_then_predictor_topN`` — the champion arm's realized 21d lift.

REWRITTEN 2026-08-27 (alpha-engine-config-I8757). The previous contract joined
``scanner_evaluations`` for its pool and count-matched each cycle to the agentic
CIO's ADVANCE count, skipping any cycle with none. Both tables lost their writer
on 2026-07-12, so no cohort after 2026-07-10 could be scored and the function
returned an identical number on three consecutive weekly gate runs while
reporting ``status: ok``.

Brian, 2026-08-27: "the cio no longer exists, that was deprecated long ago."

The tests for the retired behaviour are deleted with it — a test that pins a
dead component's semantics is how the semantics survive the component.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.end_to_end import _scanner_then_predictor_topN  # noqa: E402
from optimizer.champion_promotion import leaderboard_entry_from_e2e_lift  # noqa: E402


def _db(tmp_path, *, dates=("2026-01-02", "2026-01-09", "2026-01-16", "2026-01-23"),
        matured=True, n=10):
    """A research.db carrying ONLY the two tables this function now reads.

    `scanner_evaluations` and `cio_evaluations` are deliberately ABSENT: their
    absence must not change the result, which is the property the rewrite buys.
    """
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute(
        "CREATE TABLE predictor_outcomes_research_free (ticker TEXT, prediction_date TEXT, "
        "predicted_alpha REAL, n_research_features_missing INTEGER)"
    )
    for d in dates:
        for i in range(n):
            alpha = 0.03 if i < n // 2 else -0.03  # winners = low i
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (f"T{i}", d, "Tech", alpha if matured else None, 0.0),
            )
            # research-free predicted_alpha favours the winners (low i)
            conn.execute(
                "INSERT INTO predictor_outcomes_research_free VALUES (?,?,?,?)",
                (f"T{i}", d, float(n - i), 4),
            )
    conn.commit()
    return conn


def test_it_scores_without_scanner_evaluations_or_cio_evaluations(tmp_path):
    """The load-bearing property. Neither table exists in this connection."""
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "ok", r
    assert r["n_cycles"] == 4, r
    tabs = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "scanner_evaluations" not in tabs and "cio_evaluations" not in tabs


def test_ranking_the_pool_beats_holding_the_pool(tmp_path):
    """A SELECTION stage is graded against the population it drew from, not
    against SPY (alpha-engine-config-I7552).

    The pool must be WIDER than the selection or the comparison is vacuous:
    top-10 of 10 is the pool, and both legs read identically. 20 names, top 10.
    """
    conn = _db(tmp_path, n=20)

    r = _scanner_then_predictor_topN(conn)

    pred = r["methods"]["scanner_then_predictor_topN"]
    pool = r["methods"]["actual_research_free_pool"]
    assert pred["mean_alpha_21d"] > pool["mean_alpha_21d"], r
    assert pred["sn_lift_vs_pool"] > 0
    assert r["predictor_beats_actual_pool"] is True


def test_the_count_basis_is_a_fixed_N_not_another_components_output(tmp_path):
    from analysis.arm_realized_lift import DEFAULT_SELECTION_N

    conn = _db(tmp_path, n=30)

    r = _scanner_then_predictor_topN(conn)

    assert r["methods"]["scanner_then_predictor_topN"]["selection_n"] == DEFAULT_SELECTION_N
    # 10 picks x 4 cycles
    assert r["methods"]["scanner_then_predictor_topN"]["n_picks"] == 40
    assert "champion_top_n_default" in r["selection_count_basis"]


def test_the_retired_agentic_method_is_gone(tmp_path):
    """`agentic_cio_advance` described a component that no longer exists.
    Leaving it as a null-valued key would read as 'measured, and empty'."""
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn)

    assert set(r["methods"]) == {"scanner_then_predictor_topN", "actual_research_free_pool"}


def test_a_wholly_unmatured_window_is_insufficient_not_ok(tmp_path):
    conn = _db(tmp_path, matured=False)

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "insufficient_data", r
    assert leaderboard_entry_from_e2e_lift(
        {"scanner_then_predictor_counterfactual": r}
    ) is None


def test_cohort_dates_with_no_matured_overlap_are_insufficient_not_ok(tmp_path):
    """The exact shape the new arm is in today: scored dates exist, but none of
    them has a matured 21d return yet. That is 'no point this week', never a
    number."""
    conn = _db(tmp_path, dates=("2026-01-02",))
    conn.execute("DELETE FROM universe_returns")
    conn.execute(
        "INSERT INTO universe_returns VALUES ('T0','2025-12-01','Tech',0.05,0.0)"
    )
    conn.execute(
        "INSERT INTO universe_returns VALUES ('T1','2025-12-01','Tech',0.01,0.0)"
    )
    conn.commit()

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "insufficient_data", r
    assert r["n_cycles"] == 0
    assert "2025-12-01" in r["reason"]


def test_missing_tables_are_skipped_not_errored(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "empty.db"))

    r = _scanner_then_predictor_topN(conn)

    assert r["status"] == "skipped"
    assert "universe_returns" in r["reason"]


def test_a_healthy_run_states_the_ages_of_the_sources_it_ACTUALLY_reads(tmp_path):
    """It used to report the freshness of `scanner_evaluations` and
    `cio_evaluations`. Reporting the age of a table you no longer read is worse
    than reporting nothing."""
    conn = _db(tmp_path)

    r = _scanner_then_predictor_topN(conn)

    fresh = r["source_freshness"]
    assert set(fresh) == {
        "universe_returns_max_eval_date", "research_free_max_prediction_date",
    }
    assert fresh["universe_returns_max_eval_date"] == "2026-01-23"
    assert fresh["research_free_max_prediction_date"] == "2026-01-23"


def test_a_healthy_result_yields_a_leaderboard_point(tmp_path):
    conn = _db(tmp_path)

    entry = leaderboard_entry_from_e2e_lift(
        {"scanner_then_predictor_counterfactual": _scanner_then_predictor_topN(conn)}
    )

    assert entry is not None
    assert entry["n_cycles"] == 4
    assert entry["sector_neutral_mean_alpha_21d"] is not None
