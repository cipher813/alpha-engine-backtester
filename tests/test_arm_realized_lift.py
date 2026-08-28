"""alpha-engine-config-I8757 — scoring an entry-selection arm without the CIO.

The scorer this replaces count-matched each cycle to the number of names the
agentic CIO stage marked ADVANCE, and `continue`d any cycle with none. The CIO
was deprecated and its table stopped at 2026-07-10, so every later cycle was
skipped and the champion's weekly score froze at -0.00203 across three
consecutive gate runs — carried with confidence "ok".

Brian, 2026-08-27: "the cio no longer exists, that was deprecated long ago."
"""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from analysis.arm_realized_lift import (
    DEFAULT_SELECTION_N,
    MIN_RANKED_FOR_SELECTION,
    load_realized_alpha,
    predictor_cut_ranked_picks,
    research_free_ranked_picks,
    score_arm,
)


def _realized(rows):
    """rows: (ticker, eval_date, sector, alpha21)"""
    return pd.DataFrame(rows, columns=["ticker", "eval_date", "sector", "alpha21"])


# ── score_arm ────────────────────────────────────────────────────────────


def test_every_arm_is_cut_at_the_SAME_selection_size():
    """Count-matching (champion-challenger-policy.md §4) is enforced HERE, not
    by trusting each arm to truncate itself. An arm handing over 30 ranked
    names and one handing over 10 are scored on their top-N either way."""
    realized = _realized([(f"T{i}", "2026-05-01", "Tech", 0.10 - i * 0.01) for i in range(30)])
    wide = {"2026-05-01": [f"T{i}" for i in range(30)]}
    narrow = {"2026-05-01": [f"T{i}" for i in range(10)]}

    a = score_arm("wide", wide, realized, selection_n=5)
    b = score_arm("narrow", narrow, realized, selection_n=5)

    assert a.n_picks == b.n_picks == 5
    assert a.selection_n == b.selection_n == 5


def test_the_default_selection_size_is_what_the_arms_actually_trade():
    """10 = crucible-executor's `champion_top_n_default`. The size the arms are
    compared at is the size they trade at — which is what makes the basis
    independent of any other component being alive."""
    assert DEFAULT_SELECTION_N == 10


def test_ranking_order_decides_which_names_are_scored():
    realized = _realized([
        ("WIN", "2026-05-01", "Tech", 0.10),
        ("LOSE", "2026-05-01", "Tech", -0.10),
    ])

    good = score_arm("good", {"2026-05-01": ["WIN", "LOSE"]}, realized, selection_n=1)
    bad = score_arm("bad", {"2026-05-01": ["LOSE", "WIN"]}, realized, selection_n=1)

    assert good.mean_alpha_21d == pytest.approx(0.10)
    assert bad.mean_alpha_21d == pytest.approx(-0.10)


def test_no_cycle_is_skipped_for_a_missing_third_party():
    """The whole defect: the old scorer dropped any cycle with no CIO ADVANCE
    row. Nothing outside the arm's own picks and the realized returns can
    remove a cycle now."""
    realized = _realized([
        ("A", d, "Tech", 0.05) for d in ("2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01")
    ] + [
        ("B", d, "Tech", 0.01) for d in ("2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01")
    ])
    picks = {d: ["A", "B"] for d in ("2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01")}

    out = score_arm("arm", picks, realized)

    assert out.n_cycles == 4
    assert out.first_date == "2026-05-01"
    assert out.last_date == "2026-08-01"


def test_a_thin_cycle_is_counted_not_silently_dropped():
    """The top-10 of a 1-name list measures the pool, not the rule. An arm thin
    for half the window and an arm that was whole are different arms, and a
    bare mean cannot tell them apart."""
    realized = _realized([
        ("A", "2026-05-01", "Tech", 0.05), ("B", "2026-05-01", "Tech", 0.03),
        ("A", "2026-06-01", "Tech", 0.05),
    ])
    picks = {"2026-05-01": ["A", "B"], "2026-06-01": ["A"]}

    out = score_arm("arm", picks, realized)

    assert MIN_RANKED_FOR_SELECTION == 2
    assert out.n_cycles == 1
    assert out.n_cycles_skipped_thin == 1


def test_sector_neutral_demeans_within_the_cycle():
    realized = _realized([
        ("A", "2026-05-01", "Tech", 0.10),
        ("B", "2026-05-01", "Tech", 0.00),
        ("C", "2026-05-01", "Health", 0.04),
        ("D", "2026-05-01", "Health", 0.00),
    ])

    out = score_arm("arm", {"2026-05-01": ["A", "C"]}, realized, selection_n=2)

    # Tech mean 0.05 -> A = +0.05 ; Health mean 0.02 -> C = +0.02
    assert out.mean_alpha_21d == pytest.approx(0.07)
    assert out.sector_neutral_mean_alpha_21d == pytest.approx(0.035)


def test_a_cycle_with_no_sector_falls_back_to_raw_alpha():
    """A residual demeaned against nothing is not more neutral than the raw
    number, it is the same number with a false claim attached."""
    realized = _realized([("A", "2026-05-01", None, 0.10), ("B", "2026-05-01", None, 0.02)])

    out = score_arm("arm", {"2026-05-01": ["A", "B"]}, realized)

    assert out.sector_neutral_mean_alpha_21d == out.mean_alpha_21d


def test_an_unmatured_cycle_contributes_nothing_but_does_not_crash():
    realized = _realized([("A", "2026-05-01", "Tech", 0.05), ("B", "2026-05-01", "Tech", 0.01)])

    out = score_arm("arm", {"2026-05-01": ["A", "B"], "2026-08-27": ["A", "B"]}, realized)

    assert out.n_cycles == 1


def test_an_arm_with_no_matured_cycles_reports_None_not_zero():
    """None is 'not measured'; 0.0 is 'measured, and flat'. A gate reading the
    second as evidence would promote on an arm that has never been scored."""
    out = score_arm("arm", {"2026-08-27": ["A", "B"]}, _realized([]))

    assert out.n_cycles == 0
    assert out.mean_alpha_21d is None
    assert out.sector_neutral_mean_alpha_21d is None


# ── research_free_ranked_picks ───────────────────────────────────────────


def test_research_free_picks_rank_by_predicted_alpha_and_need_no_scanner_table():
    """The parquet contains only scanner-passing names by construction, so the
    old join against `scanner_evaluations` re-filtered an already-filtered set
    — and in doing so made a live measurement depend on a dead table."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE predictor_outcomes_research_free "
        "(ticker TEXT, prediction_date TEXT, predicted_alpha REAL)"
    )
    conn.executemany(
        "INSERT INTO predictor_outcomes_research_free VALUES (?,?,?)",
        [("LOW", "2026-05-01", 0.01), ("HIGH", "2026-05-01", 0.09),
         ("MID", "2026-05-01", 0.05), ("X", "2026-06-01", 0.02)],
    )
    conn.commit()

    picks = research_free_ranked_picks(conn)

    assert picks["2026-05-01"] == ["HIGH", "MID", "LOW"]
    assert picks["2026-06-01"] == ["X"]
    # No scanner_evaluations table exists in this connection at all.
    tabs = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "scanner_evaluations" not in tabs


# ── predictor_cut_ranked_picks ───────────────────────────────────────────


def _row(ticker, alpha, source):
    return {"ticker": ticker, "predicted_alpha": alpha, "watchlist_source": source}


def test_cut_picks_rank_by_the_predictors_alpha():
    picks = predictor_cut_ranked_picks({
        "2026-08-21": [
            _row("LOW", 0.01, "attractiveness_top_20"),
            _row("HIGH", 0.09, "attractiveness_top_20"),
        ]
    })

    assert picks["2026-08-21"] == ["HIGH", "LOW"]


def test_a_held_only_name_is_never_a_candidate():
    """The predictor unions holdings into its scoring universe so EXITS can be
    decided. This arm proposes ENTRIES — crediting it for a held name would
    credit it for a position it never chose."""
    picks = predictor_cut_ranked_picks({
        "2026-08-21": [
            _row("INCUT", 0.01, "attractiveness_top_20"),
            _row("HELDX", 0.99, "held"),
        ]
    })

    assert picks["2026-08-21"] == ["INCUT"]


def test_in_cut_AND_held_is_a_candidate():
    picks = predictor_cut_ranked_picks({
        "2026-08-21": [
            _row("INCUT", 0.01, "attractiveness_top_20"),
            _row("BOTHX", 0.05, "both"),
        ]
    })

    assert picks["2026-08-21"] == ["BOTHX", "INCUT"]


def test_a_date_whose_cut_is_not_this_arms_cut_is_REFUSED():
    """`predictor_universe_cut` was `scanner_candidates` (60 names) until
    ~2026-07-30. Scoring those dates would score 'top-10 of whatever the cut
    happened to be that month' — a different rule per era wearing this arm's
    name, and a near-clone of the incumbent on exactly the dates both drew from
    the scanner's 60.

    Measured 2026-08-27 on the live artifacts: without this refusal the arm
    scored 51 cycles back to 2026-05-01, 47 of them pre-cut.
    """
    picks = predictor_cut_ranked_picks({
        "2026-07-10": [_row("A", 0.05, "both"), _row("B", 0.01, "population")],
        "2026-08-21": [_row("C", 0.05, "attractiveness_top_20")],
    })

    assert "2026-07-10" not in picks
    assert sorted(picks) == ["2026-08-21"]


def test_the_arms_cut_is_a_parameter_not_a_literal():
    picks = predictor_cut_ranked_picks(
        {"2026-08-21": [_row("A", 0.05, "attractiveness_top_25")]},
        cut_names=("attractiveness_top_25",),
    )

    assert picks["2026-08-21"] == ["A"]


def test_spy_and_alphaless_rows_are_dropped():
    picks = predictor_cut_ranked_picks({
        "2026-08-21": [
            _row("SPY", 0.0, "attractiveness_top_20"),
            _row("NOALPHA", None, "attractiveness_top_20"),
            _row("REAL", 0.03, "attractiveness_top_20"),
        ]
    })

    assert picks["2026-08-21"] == ["REAL"]


# ── load_realized_alpha ──────────────────────────────────────────────────


def test_realized_alpha_is_spy_relative_at_the_source_and_skips_unmatured():
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, sector TEXT, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.executemany(
        "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
        [("A", "2026-05-01", "Tech", 0.08, 0.03),
         ("B", "2026-08-27", "Tech", None, 0.01)],
    )
    conn.commit()

    out = load_realized_alpha(conn)

    assert len(out) == 1
    assert out.iloc[0]["alpha21"] == pytest.approx(0.05)
