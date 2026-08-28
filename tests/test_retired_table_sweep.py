"""alpha-engine-config-I8757, deliverable 3 — the CLASS, not the instance.

`crucible-backtester-PR746` made the champion gate's counterfactual refuse a
frozen source, and `-PR748` repointed it. Neither touched the other consumers
of the same two dead tables, and there are eleven of them. This module covers
the sweep: every remaining consumer either refuses, declares itself
historical-only, or rides an explicit staleness verdict on its own result.

The production facts these tests encode (measured 2026-08-27 against
`s3://alpha-engine-research/research.db`, and re-measured 2026-08-28):

    scanner_evaluations   16252 rows   MAX(eval_date) = 2026-07-17   18 cycles
    cio_evaluations         315 rows   MAX(eval_date) = 2026-07-10   19 cycles
    universe_returns    2145202 rows   MAX(eval_date) = 2026-08-14  178 dates

Two of the consumers ACT — `optimizer/scanner_optimizer.py` and
`optimizer/tech_weight_ablation.py` both feed an `apply()` that WRITES live S3
config. They were re-deriving the same recommendation from the same frozen rows
every weekly run and presenting it as this week's finding: the champion gate's
failure with a different pointer on the end of it.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from analysis import cio_rule_tag_precision, macro_eval, retired_table_guard
from analysis.retired_table_guard import STALE_SOURCES_STATUS
from optimizer import scanner_optimizer, tech_weight_ablation


def _iso(days_ago: int) -> str:
    return (date.today() - timedelta(days=days_ago)).isoformat()


# ── the shared verdict helpers ─────────────────────────────────────────────


def _conn_with(table: str, dates: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(f"CREATE TABLE {table} (ticker TEXT, eval_date TEXT)")
    conn.executemany(
        f"INSERT INTO {table} VALUES (?,?)", [("T", d) for d in dates]
    )
    conn.commit()
    return conn


def test_source_freshness_never_raises_and_rides_on_healthy_runs():
    """A verdict emitted only on failure is a field nobody learns to read, and
    a frozen source looked exactly like a fresh one for seven weeks precisely
    because nothing said how old it was."""
    conn = _conn_with("scanner_evaluations", [_iso(3)])
    block = retired_table_guard.source_freshness(
        conn, ("scanner_evaluations",), measurement="unit",
    )
    assert block["stale"] is False
    assert block["reason"] is None
    assert block["stale_tables"] == []
    assert block["sources"][0]["age_days"] == 3
    assert block["sources"][0]["writer"]          # provenance, not just an age
    assert block["sources"][0]["retired_date"] == "2026-07-12"


def test_source_freshness_flags_a_table_past_the_bound():
    conn = _conn_with("cio_evaluations", [_iso(40)])
    block = retired_table_guard.source_freshness(
        conn, ("cio_evaluations",), measurement="unit",
    )
    assert block["stale"] is True
    assert block["stale_tables"] == ["cio_evaluations"]
    assert "cio_evaluations" in block["reason"]
    assert "2026-07-12" in block["reason"]        # names the retirement


def test_source_freshness_treats_an_absent_table_as_stale():
    """Absent and frozen are different findings but the same verdict: no
    current evidence exists, so nothing built on it is a measurement."""
    conn = sqlite3.connect(":memory:")
    block = retired_table_guard.source_freshness(
        conn, ("scanner_evaluations",), measurement="unit",
    )
    assert block["stale"] is True
    assert block["sources"][0]["present"] is False


def test_refuse_carries_the_whole_verdict_not_a_bare_skip():
    conn = _conn_with("cio_evaluations", [_iso(40)])
    block = retired_table_guard.source_freshness(
        conn, ("cio_evaluations",), measurement="unit",
    )
    result = retired_table_guard.refuse(block, run_date="2026-08-28")
    assert result["status"] == STALE_SOURCES_STATUS
    assert result["source_freshness"] == block
    assert result["run_date"] == "2026-08-28"
    assert "cio_evaluations" in result["reason"]


def test_stale_status_is_neither_error_nor_insufficient_data():
    """Nothing failed and the data is abundant — it is just old. Collapsing
    this into `error` loses the diagnosis, and into `insufficient_data` sends
    a reader looking for more rows that will never come."""
    assert STALE_SOURCES_STATUS == "stale_sources"
    assert STALE_SOURCES_STATUS not in ("ok", "error", "insufficient_data")


# ── the two consumers that ACT ─────────────────────────────────────────────


def _scanner_opt_db(tmp_path: Path, dates: list[str]) -> str:
    path = tmp_path / "research.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE scanner_evaluations (
            ticker TEXT, eval_date TEXT, tech_score REAL,
            quant_filter_pass INTEGER, liquidity_pass INTEGER,
            volatility_pass INTEGER, balance_sheet_pass INTEGER
        );
        CREATE TABLE universe_returns (
            ticker TEXT, eval_date TEXT, return_5d REAL, beat_spy_5d INTEGER
        );
        """
    )
    for d in dates:
        for i in range(20):
            conn.execute(
                "INSERT INTO scanner_evaluations VALUES (?,?,?,?,?,?,?)",
                (f"T{i}", d, 50 + i, 1 if i < 10 else 0, 1, 1, 1),
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?)",
                (f"T{i}", d, 0.01 * (i - 10), 1 if i >= 10 else 0),
            )
    conn.commit()
    conn.close()
    return str(path)


def test_scanner_optimizer_refuses_a_frozen_evidence_table(tmp_path):
    """The production shape: ten weekly cycles, all of them old. The 8-week
    min-data gate is satisfied ENTIRELY by frozen rows, so without this check
    the optimizer keeps recommending — and applying — params derived from a
    universe that stopped being observed in July."""
    dates = [_iso(60 + 7 * i) for i in range(10)]
    result = scanner_optimizer.analyze(_scanner_opt_db(tmp_path, dates))

    assert result["status"] == STALE_SOURCES_STATUS
    assert result["source_freshness"]["stale_tables"] == ["scanner_evaluations"]
    # ...and the refusal propagates all the way to the S3 write.
    recommendation = scanner_optimizer.recommend(result, {})
    assert recommendation["status"] != "ok"
    assert scanner_optimizer.apply(recommendation, "bucket")["applied"] is False


def test_scanner_optimizer_still_analyses_a_live_table(tmp_path):
    """The guard must be able to PASS, or it is indistinguishable from
    deleting the module (champion-challenger-policy.md §7.4)."""
    dates = [_iso(2 + 7 * i) for i in range(10)]
    result = scanner_optimizer.analyze(_scanner_opt_db(tmp_path, dates))

    assert result["status"] != STALE_SOURCES_STATUS


def _ablation_conn(dates: list[str]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(
        """
        CREATE TABLE scanner_evaluations (
            id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT, sector TEXT,
            quant_filter_pass INTEGER DEFAULT 1,
            rsi_sub_score REAL, macd_sub_score REAL, ma50_sub_score REAL,
            ma200_sub_score REAL, momentum_sub_score REAL
        );
        CREATE TABLE universe_returns (
            id INTEGER PRIMARY KEY, ticker TEXT, eval_date TEXT,
            return_5d REAL, beat_spy_5d INTEGER
        );
        """
    )
    for d in dates:
        for i in range(5):
            conn.execute(
                "INSERT INTO scanner_evaluations "
                "(ticker, eval_date, sector, rsi_sub_score, macd_sub_score, "
                "ma50_sub_score, ma200_sub_score, momentum_sub_score) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (f"T{d}{i}", d, "technology", 90 - i * 10, 50, 50, 50, i * 10),
            )
            conn.execute(
                "INSERT INTO universe_returns (ticker, eval_date, return_5d) "
                "VALUES (?,?,?)", (f"T{d}{i}", d, 0.05 - i * 0.02),
            )
    conn.commit()
    return conn


def test_tech_weight_ablation_refuses_a_frozen_evidence_table():
    """Same class, second pointer: apply() writes
    `config/scoring_weights_per_sector.json`, and its 4-week reproduction gate
    is SATISFIED by a frozen table — the recommendation reproduces every week
    because the evidence never changes."""
    dates = [_iso(40 + 7 * i) for i in range(8)]
    result = tech_weight_ablation.compute_tech_weight_ablation(
        db_conn=_ablation_conn(dates), run_date=date.today().isoformat(),
    )

    assert result["status"] == STALE_SOURCES_STATUS
    assert "scanner_evaluations" in result["reason"]
    assert tech_weight_ablation.apply(result, "bucket")["applied"] is False


def test_tech_weight_ablation_still_runs_on_a_live_table():
    dates = [_iso(3 + 7 * i) for i in range(8)]
    result = tech_weight_ablation.compute_tech_weight_ablation(
        db_conn=_ablation_conn(dates), run_date=date.today().isoformat(),
    )

    assert result["status"] != STALE_SOURCES_STATUS


# ── the consumers whose EVERY input is dead ────────────────────────────────


def _cio_db(tmp_path: Path, dates: list[str], *, macro: bool = False) -> str:
    path = tmp_path / "research.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE cio_evaluations (
            ticker TEXT, eval_date TEXT, cio_decision TEXT, rule_tags TEXT,
            combined_score REAL, macro_shift REAL, final_score REAL
        );
        CREATE TABLE universe_returns (
            ticker TEXT, eval_date TEXT, return_5d REAL,
            spy_return_5d REAL, beat_spy_5d INTEGER
        );
        """
    )
    for d in dates:
        for i in range(20):
            conn.execute(
                "INSERT INTO cio_evaluations VALUES (?,?,?,?,?,?,?)",
                (f"T{i}", d, "ADVANCE" if i < 10 else "REJECT",
                 '["liquidity"]', 70.0, 2.0, 72.0),
            )
            conn.execute(
                "INSERT INTO universe_returns VALUES (?,?,?,?,?)",
                (f"T{i}", d, 0.02, 0.01, 1 if i % 2 else 0),
            )
    conn.commit()
    conn.close()
    return str(path)


def test_macro_eval_refuses_when_its_only_source_is_dead(tmp_path):
    """Both legs of the A/B (combined_score, final_score) are cio_evaluations
    columns. With that table frozen the verdict is a July finding, and it was
    being rendered as this week's."""
    result = macro_eval.compute_macro_evaluation(
        _cio_db(tmp_path, [_iso(40 + 7 * i) for i in range(6)])
    )

    assert result["status"] == STALE_SOURCES_STATUS
    assert result["source_freshness"]["stale_tables"] == ["cio_evaluations"]
    assert "macro_lift" not in result          # no number is emitted at all


def test_macro_eval_still_computes_on_live_rows(tmp_path):
    result = macro_eval.compute_macro_evaluation(
        _cio_db(tmp_path, [_iso(2 + 7 * i) for i in range(6)])
    )

    assert result["status"] != STALE_SOURCES_STATUS


def test_cio_rule_tag_precision_refuses_when_its_only_source_is_dead(tmp_path):
    """Its rolling 8-week window slid off the end of the data: the window
    kept moving, the rows did not, and the precision numbers stopped changing
    while still rendering as a current measurement."""
    result = cio_rule_tag_precision.compute_cio_rule_tag_precision(
        db_path=_cio_db(tmp_path, [_iso(40 + 7 * i) for i in range(4)]),
        run_date=date.today().isoformat(),
        emit_metrics=False,
    )

    assert result["status"] == STALE_SOURCES_STATUS
    assert "overall_advance_precision" not in result


def test_cio_rule_tag_precision_still_computes_on_live_rows(tmp_path):
    result = cio_rule_tag_precision.compute_cio_rule_tag_precision(
        db_path=_cio_db(tmp_path, [_iso(2 + 7 * i) for i in range(4)]),
        run_date=date.today().isoformat(),
        emit_metrics=False,
    )

    assert result["status"] != STALE_SOURCES_STATUS


# ── the consumers that keep measuring, but say how old their cohort is ─────


def test_attractiveness_counterfactual_states_its_cohort_age(tmp_path):
    """NOT refused — the attractiveness legs beside it are computed from live
    history and refusing would discard them. But `live_gate.mean_alpha` is the
    figure that renders largest on the report card and it was the one basket
    published with no age on it."""
    from analysis import attractiveness_eval

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE scanner_evaluations "
        "(ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.executemany(
        "INSERT INTO scanner_evaluations VALUES (?,?,?)",
        [("T", "2026-07-17", 1)],
    )
    conn.commit()
    import pandas as pd

    block = attractiveness_eval._counterfactual(
        conn, pd.DataFrame(columns=["eval_date", "ticker", "attractiveness_score"]),
        pd.DataFrame(columns=["eval_date", "ticker", "alpha", "sector"]),
        as_of="2026-08-28",
    )

    assert block["source_freshness"]["stale"] is True
    assert block["source_freshness"]["sources"][0]["age_days"] == 42


def test_scanner_lift_states_its_cohort_age(tmp_path):
    """The arm the report card grades. Its lift has not moved since July and
    the artifact said so only implicitly, through `last_eval_date`."""
    import pandas as pd

    from analysis import end_to_end

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE scanner_evaluations "
        "(ticker TEXT, eval_date TEXT, quant_filter_pass INTEGER)"
    )
    conn.executemany(
        "INSERT INTO scanner_evaluations VALUES (?,?,?)",
        [("A", _iso(40), 1), ("B", _iso(40), 0)],
    )
    conn.commit()
    ur = pd.DataFrame(
        [
            {"ticker": "A", "eval_date": _iso(40), "return_5d": 0.02,
             "beat_spy_5d": 1, "beat_spy_21d": 1},
            {"ticker": "B", "eval_date": _iso(40), "return_5d": -0.01,
             "beat_spy_5d": 0, "beat_spy_21d": 0},
        ]
    )

    block = end_to_end._scanner_lift(conn, ur, "", [])

    assert block["source_freshness"]["stale"] is True
    assert block["source_freshness"]["stale_tables"] == ["scanner_evaluations"]
    # the numbers are still emitted — the attestation battery drives this path
    assert "lift" in block


# ── the report must SAY it, not silently drop the section ─────────────────


def _stale_result(measurement: str) -> dict:
    conn = _conn_with("scanner_evaluations", ["2026-07-17"])
    block = retired_table_guard.source_freshness(
        conn, ("scanner_evaluations",), measurement=measurement,
        run_date="2026-08-28",
    )
    return retired_table_guard.refuse(block)


def test_scanner_opt_section_refuses_rather_than_printing_a_zero():
    """Without a branch, a `stale_sources` result falls through to the
    `analysis.get('leakage_rate', 0)` default and the report prints
    **Filter leakage: 0.0%** — a fabricated number for a computation that
    never ran."""
    import reporter

    lines = reporter._section_scanner_opt(_stale_result("scanner_optimizer"))
    text = "\n".join(lines)

    assert "sources stale" in text
    assert "scanner_evaluations" in text
    assert "0.0%" not in text


def test_tech_weight_ablation_section_names_the_dead_table():
    import reporter

    text = "\n".join(
        reporter._section_tech_weight_ablation(_stale_result("tech_weight_ablation"))
    )

    assert "sources stale" in text
    assert "unknown" not in text          # the old catch-all branch


def test_macro_eval_refusal_is_rendered_not_omitted():
    """An omitted section reads as "nothing to report". The refusal is the
    report's only chance to say the diagnostic has had no input since July."""
    import reporter

    text = "\n".join(reporter._stale_sources_lines(_stale_result("macro_eval")))

    assert "cio" not in text.lower() or "scanner_evaluations" in text
    assert "alpha-engine-config-I8757" in text


# ── the class backstop ─────────────────────────────────────────────────────


#: Every module under analysis/ and optimizer/ that names a retired table,
#: with the disposition it has been given. A module may not read one of these
#: tables without appearing here — the point of the sweep is that the NEXT
#: consumer cannot be added blind, which is how nine of them accumulated.
#:
#: guarded            — refuses, or rides a verdict on its result
#: historical-only    — retired from the live path; emits a retired marker
#: mentions-only      — the name appears in prose/fixtures, no live read
_DISPOSITIONS = {
    "analysis/retired_table_guard.py": "guarded",
    # PR748 removed both dead-table READS from these two; what remains is
    # docstring text explaining the removal. They are the champion path.
    "analysis/arm_realized_lift.py": "mentions-only",
    "analysis/attractiveness_eval.py": "guarded",
    "analysis/cio_rule_tag_precision.py": "guarded",
    "analysis/macro_eval.py": "guarded",
    "analysis/wide_feed_counterfactual_review.py": "guarded",
    "analysis/contribution_lift/groups/research_composite.py": "guarded",
    "optimizer/scanner_optimizer.py": "guarded",
    "optimizer/tech_weight_ablation.py": "guarded",
    "optimizer/champion_promotion.py": "mentions-only",
    # `_scanner_lift` is STAMPED (its result states its cohort age) rather
    # than refused, because `analysis/attestation.py` drives it over a frozen
    # 2024 fixture on purpose and a refusal would break that battery.
    # `_cio_lift`, `_cio_consolidation_counterfactual`, `_cio_selection_skill`
    # and `_cio_layer_attribution` are RETIRED from the live
    # `compute_lift_metrics` path (config#1580 / I2993 / I3000): the live path
    # emits a retired marker and never calls them.
    "analysis/end_to_end.py": "guarded",
    "analysis/attestation.py": "mentions-only",
    "analysis/contribution_lift/inputs.py": "mentions-only",
    "analysis/scanner_predictor_research_free_backfill.py": "mentions-only",
    "evaluate.py": "historical-only",
    "reporter.py": "mentions-only",
}

_REPO = Path(__file__).resolve().parent.parent


def test_every_consumer_of_a_retired_table_has_a_declared_disposition():
    """A guard that cannot fail is indistinguishable from no guard
    (champion-challenger-policy.md §7.4). This is the half that fails: add a
    module that reads `scanner_evaluations` or `cio_evaluations` and this test
    goes red until someone says, in writing, what that module does about the
    fact that nothing has written those tables since 2026-07-12."""
    found = set()
    for path in list(_REPO.glob("analysis/**/*.py")) + \
            list(_REPO.glob("optimizer/**/*.py")) + \
            [_REPO / "evaluate.py", _REPO / "reporter.py"]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "scanner_evaluations" in text or "cio_evaluations" in text:
            found.add(str(path.relative_to(_REPO)))

    undeclared = sorted(found - set(_DISPOSITIONS))
    assert not undeclared, (
        "these modules read a retired-writer table with no declared "
        f"disposition: {undeclared} — add them to _DISPOSITIONS in this file "
        "AND give them a guard or a historical-only declaration "
        "(alpha-engine-config-I8757)"
    )

    gone = sorted(set(_DISPOSITIONS) - found)
    assert not gone, (
        f"_DISPOSITIONS names modules that no longer read a retired table: "
        f"{gone} — a stale allowlist hides the next real consumer"
    )


@pytest.mark.parametrize(
    "module", [m for m, d in _DISPOSITIONS.items() if d == "guarded"]
)
def test_every_guarded_module_actually_reaches_the_guard(module):
    """The disposition is a claim; this checks it. A module declared `guarded`
    that never imports the guard is a comment, not a control."""
    text = (_REPO / module).read_text(encoding="utf-8")
    assert "retired_table_guard" in text, (
        f"{module} is declared `guarded` but never reaches "
        "analysis/retired_table_guard.py"
    )
