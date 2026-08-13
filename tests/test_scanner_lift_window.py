"""Tests for the scanner_lift window restriction (alpha-engine-config-I1458).

Between crucible-research PR#344 (merged 2026-06-30) and PR#383 (merged
2026-07-05), ``graph/research_graph.py`` wrote ``quant_filter_pass=0`` for
100% of rows on every eval_date in that window (an always-empty process-local
stash). Grading those dates alongside dates where the gate genuinely ran
manufactures false negatives and collapses measured recall. ``_scanner_lift``
now excludes any eval_date whose scanner_evaluations rows sum to zero passes,
and emits window-provenance keys (n_dates, n_dates_with_passes,
first_eval_date, last_eval_date, window_rule) plus an additive ``unfiltered``
sub-block carrying the pre-restriction figures.
"""

import sqlite3

import pandas as pd

from analysis.end_to_end import _scanner_lift

GOOD_DATE = "2026-07-10"
ZERO_DATE = "2026-06-30"


def _build_db(tmp_path, *, include_zero_pass_date: bool):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE universe_returns ("
        "ticker TEXT, eval_date TEXT, sector TEXT, "
        "return_5d REAL, spy_return_5d REAL, beat_spy_5d INTEGER, "
        "return_21d REAL, spy_return_21d REAL, beat_spy_21d INTEGER, "
        "log_return_21d REAL, log_spy_return_21d REAL)"
    )
    conn.execute(
        "CREATE TABLE scanner_evaluations "
        "(ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )

    dates = [GOOD_DATE]
    if include_zero_pass_date:
        dates.append(ZERO_DATE)

    for d in dates:
        for i in range(10):
            t = f"T{i:02d}"
            # On GOOD_DATE the gate genuinely records passes for the first 5
            # names; on ZERO_DATE (the empty-stash bug) every row is 0.
            selected = (d == GOOD_DATE) and (i < 5)
            beat_5d = 1 if selected else 0
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (t, d, "Tech", 1.0 if selected else 0.5, 0.5, beat_5d,
                 2.0, 1.0, beat_5d, 0.05 if selected else -0.02, 0.0),
            )
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?)",
                (t, d, 1 if selected else 0),
            )
    conn.commit()
    conn.close()
    return str(db)


def _run(db_path):
    conn = sqlite3.connect(db_path)
    ur = pd.read_sql_query(
        "SELECT * FROM universe_returns ORDER BY eval_date, ticker", conn
    )
    out = _scanner_lift(conn, ur, "", [])
    conn.close()
    return out


def test_zero_pass_dates_excluded_from_window(tmp_path):
    db = _build_db(tmp_path, include_zero_pass_date=True)
    sl = _run(db)

    # Only GOOD_DATE contributed a recorded pass; ZERO_DATE is excluded.
    assert sl["n_dates_with_passes"] == 1
    assert sl["n_dates"] == 1
    assert sl["n_dates_excluded_zero_pass"] == 1
    assert sl["first_eval_date"] == GOOD_DATE
    assert sl["last_eval_date"] == GOOD_DATE
    assert sl["window_rule"] == "recorded_zero_pass_dates_excluded"

    # The windowed n_universe/n_passing only reflect GOOD_DATE's 10 rows / 5 passes.
    assert sl["n_universe"] == 10
    assert sl["n_passing"] == 5

    # The unfiltered sub-block retains the full, unrestricted history: both
    # dates, 20 rows total, still only 5 recorded passes (ZERO_DATE contributes 0).
    unfiltered = sl["unfiltered"]
    assert unfiltered["n_dates"] == 2
    assert unfiltered["n_universe"] == 20
    assert unfiltered["n_passing"] == 5
    assert unfiltered["first_eval_date"] == ZERO_DATE
    assert unfiltered["last_eval_date"] == GOOD_DATE


def test_window_is_a_noop_when_every_date_has_passes(tmp_path):
    db = _build_db(tmp_path, include_zero_pass_date=False)
    sl = _run(db)

    assert sl["n_dates_with_passes"] == 1
    assert sl["n_dates"] == 1
    assert sl["n_dates_excluded_zero_pass"] == 0
    assert sl["n_universe"] == sl["unfiltered"]["n_universe"] == 10
    assert sl["n_passing"] == sl["unfiltered"]["n_passing"] == 5
    assert sl["lift"] == sl["unfiltered"]["lift"]


def test_additive_window_provenance_keys_present(tmp_path):
    db = _build_db(tmp_path, include_zero_pass_date=True)
    sl = _run(db)

    for key in (
        "n_dates", "n_dates_with_passes", "n_dates_excluded_zero_pass",
        "first_eval_date", "last_eval_date", "window_rule", "unfiltered",
    ):
        assert key in sl

    # Pre-existing contract keys are untouched (still present, still the
    # windowed values — no rename/removal).
    for key in ("universe_avg", "passing_avg", "lift", "n_universe",
                "n_passing", "classification", "classification_21d",
                "lift_21d_log", "arm"):
        assert key in sl

    # `arm` label and units (raw fraction, never multiplied by 100) unchanged.
    assert sl["arm"] == "tech_score_baseline (retired from live feed 2026-06-29)"
    assert -1.0 < sl["lift"] < 1.0
