"""alpha-engine-config-I8757 — a measurement built on a frozen source is not a
measurement.

The defect these cover, measured 2026-08-27 against
``s3://alpha-engine-research/research.db``:

    cio_evaluations       315 rows, MAX(eval_date) = 2026-07-10
    scanner_evaluations 16252 rows, MAX(eval_date) = 2026-07-17
    universe_returns  2145202 rows, MAX(eval_date) = 2026-08-14

Both retired-writer tables were 4-5 weeks behind the returns table they are
joined against, and ``_scanner_then_predictor_topN`` kept returning
``status="ok"`` with a number computed entirely from cohorts on or before
2026-07-10. The champion gate consumed that number three weeks running.
"""
from __future__ import annotations

import sqlite3

import pytest

from analysis.retired_table_guard import (
    DEFAULT_MAX_AGE_DAYS,
    RETIRED_WRITER_TABLES,
    StaleResearchTableError,
    assert_sources_current,
    survey_retired_tables,
    table_max_date,
)


def _db(rows: dict[str, list[str]]) -> sqlite3.Connection:
    """``{table: [eval_date, ...]}`` — a table with an empty list is created
    but left empty, which is a different finding from an absent one."""
    conn = sqlite3.connect(":memory:")
    for table, dates in rows.items():
        conn.execute(f"CREATE TABLE {table} (ticker TEXT, eval_date TEXT)")
        conn.executemany(
            f"INSERT INTO {table} VALUES (?,?)", [("T", d) for d in dates]
        )
    conn.commit()
    return conn


# ── the registry itself ──────────────────────────────────────────────────


def test_registry_covers_both_retired_research_graph_tables():
    """Liveness is a queryable fact, not prose a reader has to interpret
    (champion-challenger-policy.md §6). Both tables lost the same writer on the
    same date; if one is registered and the other is not, the guard covers half
    a join and reports the other half as healthy."""
    by_name = {t.table: t for t in RETIRED_WRITER_TABLES}
    assert set(by_name) == {"cio_evaluations", "scanner_evaluations"}
    for spec in RETIRED_WRITER_TABLES:
        assert spec.retired_date == "2026-07-12"
        assert spec.date_column == "eval_date"
        assert spec.writer and spec.reference


# ── table_max_date ───────────────────────────────────────────────────────


def test_table_max_date_reads_the_newest_row():
    conn = _db({"cio_evaluations": ["2026-07-01", "2026-07-10", "2026-06-30"]})
    assert table_max_date(conn, "cio_evaluations", "eval_date") == "2026-07-10"


def test_table_max_date_is_none_for_absent_or_empty():
    conn = _db({"cio_evaluations": []})
    assert table_max_date(conn, "cio_evaluations", "eval_date") is None
    assert table_max_date(conn, "not_a_table", "eval_date") is None


# ── survey ───────────────────────────────────────────────────────────────


def test_survey_reproduces_the_live_defect():
    """The exact live shape. Both sources stale against the returns table."""
    conn = _db(
        {
            "cio_evaluations": ["2026-07-10"],
            "scanner_evaluations": ["2026-07-17"],
        }
    )

    rows = {r["table"]: r for r in survey_retired_tables(conn, "2026-08-14")}

    assert rows["cio_evaluations"]["newest_date"] == "2026-07-10"
    assert rows["cio_evaluations"]["age_days"] == 35
    assert rows["cio_evaluations"]["stale"] is True
    assert rows["scanner_evaluations"]["age_days"] == 28
    assert rows["scanner_evaluations"]["stale"] is True


def test_survey_reports_a_fresh_table_as_fresh():
    conn = _db({"cio_evaluations": ["2026-08-10"], "scanner_evaluations": ["2026-08-12"]})

    rows = {r["table"]: r for r in survey_retired_tables(conn, "2026-08-14")}

    assert all(not r["stale"] for r in rows.values())
    assert rows["cio_evaluations"]["age_days"] == 4


def test_survey_treats_an_absent_table_as_stale_not_fresh():
    """No current evidence either way — a measurement built on it is not a
    measurement, so it must never fall through as healthy."""
    conn = _db({"cio_evaluations": ["2026-08-13"]})

    rows = {r["table"]: r for r in survey_retired_tables(conn, "2026-08-14")}

    assert rows["scanner_evaluations"]["present"] is False
    assert rows["scanner_evaluations"]["stale"] is True


def test_survey_never_raises_and_ignores_unregistered_names():
    conn = _db({})
    assert survey_retired_tables(conn, "2026-08-14", tables=("no_such_registry_entry",)) == []


def test_one_missed_weekly_cycle_is_tolerated():
    """The bound is 14d against a 7d cadence — one fully missed cycle is slack,
    two is a dead producer."""
    assert DEFAULT_MAX_AGE_DAYS == 14
    conn = _db({"cio_evaluations": ["2026-08-01"], "scanner_evaluations": ["2026-08-01"]})

    rows = survey_retired_tables(conn, "2026-08-14")  # 13 days

    assert all(not r["stale"] for r in rows)


# ── assert_sources_current ───────────────────────────────────────────────


def test_assert_raises_and_names_every_stale_source():
    conn = _db(
        {"cio_evaluations": ["2026-07-10"], "scanner_evaluations": ["2026-07-17"]}
    )

    with pytest.raises(StaleResearchTableError) as exc:
        assert_sources_current(
            conn,
            "2026-08-14",
            ("scanner_evaluations", "cio_evaluations"),
            measurement="scanner_then_predictor_topN",
        )

    msg = str(exc.value)
    assert "scanner_then_predictor_topN" in msg
    assert "cio_evaluations" in msg and "scanner_evaluations" in msg
    # The operator needs the WRITER and the reference to act on it, not just
    # the table name.
    assert "2026-07-12" in msg
    assert "I2515" in msg


def test_assert_returns_the_survey_when_sources_are_current():
    """A healthy run states its sources' ages rather than staying silent."""
    conn = _db({"cio_evaluations": ["2026-08-13"], "scanner_evaluations": ["2026-08-13"]})

    survey = assert_sources_current(
        conn,
        "2026-08-14",
        ("scanner_evaluations", "cio_evaluations"),
        measurement="scanner_then_predictor_topN",
    )

    assert len(survey) == 2
    assert all(r["age_days"] == 1 for r in survey)
