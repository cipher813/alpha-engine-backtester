"""Staleness guard for research.db tables whose WRITERS have been retired.

alpha-engine-config-I8757.

## The failure this exists for

`research.db` carries tables written by components that no longer run. The
reads never fail — the tables are still there, still well-formed, still
joinable. The data simply stopped changing. Every consumer keeps producing a
well-formed number, and that number is a historical artifact presented as this
week's measurement.

Measured 2026-08-27 against `s3://alpha-engine-research/research.db`:

    cio_evaluations       315 rows, MAX(eval_date) = 2026-07-10   (7 weeks dead)
    scanner_evaluations 16252 rows, MAX(eval_date) = 2026-07-17   (6 weeks dead)

Both writers were the six-team Research LangGraph, removed from the weekly SF
by the 2026-07-14 config#1580 restructure and formally retired 2026-07-12
(`crucible-research/producers/registry.py::agentic_sector_teams`,
`retired_date="2026-07-12"`).

The consequence reached a LIVE POINTER. `optimizer/champion_promotion.py`
scores `scanner_predictor_direct` from `analysis/end_to_end.py::
_scanner_then_predictor_topN`, which joins `scanner_evaluations` and
count-matches against `cio_evaluations` ADVANCE rows, skipping any cycle with
no ADVANCE row. With both tables frozen, the counterfactual can only ever score
cohort dates on or before 2026-07-10 — so the champion's weekly score has been
frozen at **-0.00203 with n_cycles=15 and n_picks=119 for 2026-08-13, -08-14
and -08-21**, three consecutive weekly gate runs reporting an identical number
as if it were new evidence.

This is the fleet's dominant bug class — a record asserting an action that never
happened — and `champion-challenger-policy.md` §7.2 names the remedy: an
unmeasurable result must fail LOUD, never render as an empty (or in this case a
stale) success.

## The general lesson, already learned once

`crucible-predictor/inference/stages/load_universe.py` carries it verbatim, from
an identical incident where the predictor scored a frozen 25-name list for three
weekly cycles while every daily run looked green:

    a universe read must fail on STALENESS, not just on absence

This module is that rule, applied to the sqlite side, in one place instead of
nine.

## Posture

Two layers, deliberately different, because the cost of being wrong differs.

* **Detection is universal and cheap.** `survey_retired_tables` reports the
  freshness of every registered table and is safe to call from anywhere; it
  never raises. A consumer that only wants to say "my sources are N days old"
  uses this.
* **Enforcement is targeted.** `assert_sources_current` raises, and is wired
  only where a stale number ACTS — today, the champion gate's counterfactual,
  which moves `config/producer_champion.json`. Wiring the raise into every
  analytic at once would red an entire evaluate run over diagnostics nobody
  trades on, which trades one silent failure for a loud irrelevant one.

Adding a table here is the whole cost of covering a new retired writer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)


class StaleResearchTableError(RuntimeError):
    """A source table's newest row is older than the measurement allows.

    Raised only by ``assert_sources_current`` — see the module docstring's
    posture note on why detection is universal and enforcement is not.
    """


@dataclass(frozen=True)
class RetiredTable:
    """One research.db table whose writer no longer runs."""

    table: str
    date_column: str
    writer: str
    retired_date: str
    reference: str


# The registry. A table belongs here the moment its writer stops running —
# `champion-challenger-policy.md` §6 requires liveness to be a queryable FACT
# rather than an inference from prose, and this is the consumer-side half of
# that: `producers/registry.py` records that the PRODUCER retired, and this
# records which stored artifacts went cold with it.
RETIRED_WRITER_TABLES: tuple[RetiredTable, ...] = (
    RetiredTable(
        table="cio_evaluations",
        date_column="eval_date",
        writer="crucible-research graph/research_graph.py (six-team LangGraph, CIO pass)",
        retired_date="2026-07-12",
        reference="alpha-engine-config-I2515 / config#1580",
    ),
    RetiredTable(
        table="scanner_evaluations",
        date_column="eval_date",
        writer="crucible-research graph/research_graph.py::write_scanner_evaluations",
        retired_date="2026-07-12",
        reference="alpha-engine-config-I2515 / config#1580 (producer side repointed to "
        "candidates/{run_date}/candidates.json::scanner_eval_log by config#3053; "
        "these consumers were not)",
    ),
)

_BY_NAME = {t.table: t for t in RETIRED_WRITER_TABLES}

# How stale a source may be before a measurement built on it stops being a
# measurement. The weekly cadence is 7 days; 14 allows one fully missed cycle
# before a number is refused, which is the same one-cycle slack
# `crucible-predictor`'s MEMBERSHIP_MAX_AGE_DAYS grants for the same reason.
DEFAULT_MAX_AGE_DAYS = 14


def _table_exists(conn, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def table_max_date(conn, table: str, date_column: str) -> str | None:
    """Newest ``date_column`` value in ``table``, or None if absent/empty.

    Never raises on a missing table or column — an absent table is a different
    finding from a stale one, and the caller distinguishes them.
    """
    if not _table_exists(conn, table):
        return None
    try:
        row = conn.execute(f"SELECT MAX({date_column}) FROM {table}").fetchone()
    except Exception as exc:  # noqa: BLE001 — a malformed table is reported, not raised
        logger.warning(
            "retired_table_guard: cannot read MAX(%s) from %s: %s",
            date_column, table, exc,
        )
        return None
    return row[0] if row and row[0] else None


def age_days(newest: str, run_date: str) -> int | None:
    """Whole days between ``newest`` and ``run_date``; None if either is unparseable.

    Public because a consumer whose staleness bound is the DATE RANGE it is
    replaying — rather than today — needs the same arithmetic
    (``contribution_lift.groups.research_composite``).
    """
    try:
        return (date.fromisoformat(run_date) - date.fromisoformat(str(newest)[:10])).days
    except (TypeError, ValueError):
        return None


def survey_retired_tables(
    conn,
    run_date: str,
    *,
    tables: tuple[str, ...] | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> list[dict]:
    """Freshness of each registered retired-writer table. NEVER raises.

    One dict per table: ``table``, ``newest_date``, ``age_days``, ``present``,
    ``stale`` (bool), ``writer``, ``retired_date``, ``reference``,
    ``max_age_days``.

    Emitted on every run, healthy included — an absent field is unmeasured, not
    fine, and the whole point of this module is that a frozen source looked
    exactly like a fresh one for seven weeks.
    """
    names = tables if tables is not None else tuple(_BY_NAME)
    out: list[dict] = []
    for name in names:
        spec = _BY_NAME.get(name)
        if spec is None:
            logger.warning(
                "retired_table_guard: %r is not in RETIRED_WRITER_TABLES — "
                "surveying it is a no-op; register it if its writer is retired",
                name,
            )
            continue
        newest = table_max_date(conn, spec.table, spec.date_column)
        age = age_days(newest, run_date) if newest else None
        out.append(
            {
                "table": spec.table,
                "present": newest is not None,
                "newest_date": newest,
                "age_days": age,
                # An absent/unreadable table is stale for this purpose: either
                # way no current evidence exists, and a measurement built on it
                # is not a measurement.
                "stale": age is None or age > max_age_days,
                "max_age_days": max_age_days,
                "writer": spec.writer,
                "retired_date": spec.retired_date,
                "reference": spec.reference,
            }
        )
    return out


def assert_sources_current(
    conn,
    run_date: str,
    tables: tuple[str, ...],
    *,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
    measurement: str,
) -> list[dict]:
    """Raise :class:`StaleResearchTableError` if any of ``tables`` is stale.

    Returns the survey rows on success so a caller can record them alongside
    its result — a healthy run states its sources' ages rather than staying
    silent about them.

    Wire this only where a stale number ACTS. See the module docstring.
    """
    survey = survey_retired_tables(
        conn, run_date, tables=tables, max_age_days=max_age_days
    )
    stale = [row for row in survey if row["stale"]]
    if stale:
        raise StaleResearchTableError(
            _stale_reason(stale, measurement, run_date, max_age_days)
        )
    return survey


# --------------------------------------------------------------------------
# The sweep half (alpha-engine-config-I8757, deliverable 3).
#
# `assert_sources_current` above is the enforcement layer, wired only where a
# stale number ACTS. But nine further modules read these same dead tables and
# emit a number from them every weekly run, and the champion gate proved that
# a well-formed number from a frozen source is indistinguishable from a
# measurement. So every such consumer now carries a VERDICT on its own result:
#
#   * `source_freshness(...)` builds that verdict and never raises — safe in
#     any analytic, including ones nobody trades on.
#   * `stamp(result, block)` rides it on the result, healthy runs included. A
#     field present only on failure is a field nobody learns to read.
#   * `refuse(block, ...)` is the shape a consumer returns INSTEAD of a number
#     when its sources are dead. `status` is the fleet-standard
#     "stale_sources" so the existing `status != "ok"` guards in
#     `optimizer/*.apply()` and `reporter._section_*` route it without any of
#     them needing to learn a new concept.
#
# Which consumers refuse and which only stamp is decided by ONE question:
# would the number act, or is every input of it dead? Both answers refuse.
# A block that mixes live and dead inputs stamps, because refusing it would
# discard the live half.
# --------------------------------------------------------------------------

#: `status` value a consumer returns when its sources are past the bound.
#: Deliberately not "error" (nothing failed) and not "insufficient_data"
#: (the data is abundant — it is just old).
STALE_SOURCES_STATUS = "stale_sources"


def _stale_reason(stale: list[dict], measurement: str, run_date: str,
                  max_age_days: int) -> str:
    detail = "; ".join(
        f"{r['table']} newest={r['newest_date']} age={r['age_days']}d "
        f"(writer {r['writer']} retired {r['retired_date']}, {r['reference']})"
        for r in stale
    )
    return (
        f"{measurement} cannot be measured on {run_date}: "
        f"{len(stale)} source table(s) past the {max_age_days}d bound — {detail}. "
        "Refusing to emit a number computed from a frozen source: it would be "
        "a historical artifact presented as this week's evidence."
    )


def source_freshness(
    conn,
    tables: tuple[str, ...],
    *,
    measurement: str,
    run_date: str | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> dict:
    """Build a staleness verdict for ``tables``. NEVER raises.

    ``run_date`` defaults to today (UTC-naive local date) so a consumer that
    has no run_date of its own can still say how old its inputs are — an age
    against today is the honest reading when no as-of was supplied.

    Returns ``{measurement, run_date, stale, stale_tables, max_age_days,
    sources, reason}``. ``reason`` is None when nothing is stale.
    """
    run_date = run_date or date.today().isoformat()
    survey = survey_retired_tables(
        conn, run_date, tables=tables, max_age_days=max_age_days
    )
    stale = [r for r in survey if r["stale"]]
    return {
        "measurement": measurement,
        "run_date": run_date,
        "stale": bool(stale),
        "stale_tables": [r["table"] for r in stale],
        "max_age_days": max_age_days,
        "sources": survey,
        "reason": (
            _stale_reason(stale, measurement, run_date, max_age_days)
            if stale else None
        ),
    }


def stamp(result: dict, block: dict) -> dict:
    """Attach ``block`` to ``result`` under ``source_freshness``. In place.

    Emitted on healthy runs too: the champion gate read a frozen table for
    seven weeks precisely because nothing on the result said how old it was.
    """
    result["source_freshness"] = block
    return result


def refuse(block: dict, **extra) -> dict:
    """The result a consumer returns instead of a number, when sources are dead.

    Carries the full verdict, so the artifact says WHICH table died, WHEN, to
    which retirement, rather than a bare "skipped".
    """
    return {
        "status": STALE_SOURCES_STATUS,
        "reason": block.get("reason") or f"{block.get('measurement')}: sources stale",
        "source_freshness": block,
        **extra,
    }
