"""producer_arena.py — the selection-producer slot's arena wiring (alpha-engine-config-I9318).

**This module owns the pointer DECISION for the selection-producer slot**
(``signals/{date}/signals.json``). It does not implement any of it: every
rule in ``champion-challenger-policy.md`` §§3–6 — the score ladder, the
longest-common-window pairing, the anytime-valid confidence sequence, the
Copeland ranking, the pointer rule and the cap-with-grace retirement rule —
comes from ``nousergon_lib.arena``, the fleet's single implementation
(`shared-code-policy.md`; a slot re-implementing §§3–6 is a defect, §10).
What lives here is the four things the engine deliberately does NOT do:

1. **The register** — which arms exist, when each was first observed, and
   which are retired. Derived from crucible-research's producer registry as
   projected onto ``research/producer_leaderboard/{date}.json``; persisted as
   a durable append-only artifact (:data:`REGISTER_PATH`), never recomputed
   from scratch per cycle.
2. **The benchmark and the series** — the per-date, POPULATION-relative
   score each arm is graded on.
3. **The serving preconditions** — shadow-only arms, feed liveness, and
   whether the executor can actually serve an arm — passed in as results.
4. **The artifact** — ``arena/producer/{date}.json``, schema-valid
   against ``nousergon_lib.contracts``'s ``arena_cycle``, emitted EVERY
   cycle whatever the outcome (§11).

**Which contract is authoritative.** From this change the ``arena_cycle``
artifact is the AUTHORITATIVE record of the slot's decision: the pointer,
every pairwise verdict with the window it rests on, the confidence-sequence
bound, and every retirement verdict including the non-retirements.
``config/apply_audit/producer_champion/{date}.json``
(``producer_champion_audit`` v2) is retained as a NARROWED, derived view for
its existing consumers — crucible-dashboard ``views/46_Experiments.py``,
crucible-evaluator ``grading/attestation.py`` /
``grading/tiles/backtester.py`` / ``director/report_card_digest.py`` — and
its ``outcome``/``champion_after``/``blocked_by`` fields are now PROJECTIONS
of the arena decision rather than an independent computation. It is
deliberately not dual-WRITTEN in the sense §10 forbids: there is exactly one
decision, taken once, in :func:`run_arena_cycle`, and two renderings of it.
``producer_champion_audit`` RETIRES when every consumer above reads
``arena_cycle`` instead, which is option (B) of
``alpha-engine-config-I9406`` — so "when" is a tracked issue rather than a
mood. The same issue records the narrowed view's live cost: its four
enum-typed arm fields cannot NAME ``no_agent_quant`` or
``single_agent_quant``, so the projection nulls them and logs, and the open
``arm_scores`` map is what keeps their measurement from being lost.

**Why the benchmark may not be SPY.** ``ArenaConfig`` REFUSES a
selection-stage slot graded against anything but the population it selected
from, and the refusal is load-bearing rather than stylistic: on 2026-08-17
SPY trailed the drawn-from population by 140bp at 21d, which inverts wins
and losses outright. The producer leaderboard's own
``LEADERBOARD_SLOTS["producer"].primary_metric`` is
``topn_alpha_vs_benchmark`` against SPY; this module therefore reads
``topn_alpha_vs_population`` and NEVER the SPY figure, and it refuses to
score rather than substitute one for the other.

**Fail loud, never fill in.** An arm the register knows about that the board
cannot supply a series for is recorded as an explicit, named
:class:`SeriesGap` and reaches the artifact as an unmeasurable comparison —
never as a zero, never as an omission, never as a silently-narrowed roster
(§7.2: the fleet's dominant bug class is a well-formed artifact containing
nothing). A cycle whose decision status is ``unmeasurable`` or
``unservable`` publishes an ops alert (§11).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3

from nousergon_lib.arena import (
    ArenaConfig,
    ArenaCycle,
    ArmRegister,
    ArmSeries,
    ServingPrecondition,
    run_cycle,
)
from nousergon_lib.contracts import validate as validate_contract

logger = logging.getLogger(__name__)

__all__ = [
    "ARENA_CONFIG",
    "ARENA_CYCLE_PREFIX",
    "POINTER_CONTRACT_PATH",
    "POPULATION_SERIES_FIELD",
    "REGISTER_PATH",
    "SLOT",
    "UNBOARDED_ARMS",
    "SLOT_KIND",
    "WRITE_FORBIDDEN_ARMS",
    "SeriesGap",
    "arm_id_for",
    "build_series",
    "load_register",
    "pointer_admissible_arms",
    "promotion_eligible_arm_names",
    "register_events_from_boards",
    "roster_disagreement",
    "run_arena_cycle",
    "write_arena_cycle",
]

SLOT = "producer"

#: ``ArenaConfig`` REFUSES ``benchmark != "population"`` for this value. That
#: refusal is the mechanism, not a convention — see the module docstring.
SLOT_KIND = "selection_producer"

#: The durable, append-only arm register for this slot. A COMMITTED artifact,
#: not something recomputed per cycle: ``created_date`` drives the four-week
#: grace period, so a value that could silently change between cycles would
#: make retirement non-reproducible. Backfilled once by
#: ``scripts/backfill_producer_arena_register.py`` from the real
#: ``research/producer_leaderboard/`` history; extended (never rewritten) when
#: a new arm first appears on the board.
REGISTER_PATH = Path(__file__).resolve().parent / "arena" / "producer_register.json"

#: ``arena/producer/{date}.json`` + ``latest.json`` — the §11 artifact.
#:
#: MIRRORS the S-slot's layout (``optimizer/strategy_arena.py``'s
#: ``arena/strategy/{date}.json``, alpha-engine-config-I9320) rather than
#: inventing a second prefix family under ``research/``. One ``arena/``
#: namespace means a console adapter, a freshness row or a backfill can
#: enumerate every slot's cycles with one prefix — the alternative is a
#: per-slot convention that has to be discovered slot by slot.
ARENA_CYCLE_PREFIX = "arena/producer"

#: The per-date, POPULATION-relative score series this slot grades on, read
#: off each ``specs[]`` row of ``research/producer_leaderboard/{date}.json``.
#:
#: crucible-research COMPUTES this series today — ``scoring/
#: leaderboard_scoring.py::_topn_alpha_vs_population_metric`` builds the
#: per-date list and hands it to ``date_clustered_stats`` — and then publishes
#: only the aggregate (``topn_alpha_vs_population``: mean/se/t_stat/n_dates).
#: The per-date values are discarded before the artifact is written.
#:
#: A confidence sequence is a statement about a SEQUENCE of paired per-date
#: differences; it cannot be formed from two cumulative means, and feeding it
#: one would be a false statement about what the interval covers. So this
#: module reads the per-date field and, when the board does not carry it,
#: records a named :class:`SeriesGap` and lets the cycle come out
#: ``unmeasurable`` — LOUDLY, on the artifact and on an ops alert — rather
#: than substituting a number that would decide the live pointer on a
#: statistic nobody can defend.
#:
#: Emitting it is a one-field additive change in the repo that OWNS the
#: measurement (crucible-research), which is the correct architectural layer;
#: tracked as alpha-engine-config-I9405.
POPULATION_SERIES_FIELD = "topn_alpha_vs_population_by_date"

#: The frozen cross-repo pointer contract THIS repo owns and the executor
#: reads. Its ``champion`` enum is the authoritative statement of which values
#: ``config/producer_champion.json`` may carry.
POINTER_CONTRACT_PATH = (
    Path(__file__).resolve().parents[1] / "contracts" / "producer_champion.schema.json"
)

#: Enum values that are READ-TOLERATED but WRITE-FORBIDDEN. ``agentic`` names
#: the retired multi-agent pipeline; the enum keeps it so a historical pointer
#: object still validates, and the pointer writer has refused to emit it since
#: the 2026-07-14 seat swap.
WRITE_FORBIDDEN_ARMS: frozenset[str] = frozenset({"agentic"})


def pointer_admissible_arms(path: Path | None = None) -> frozenset[str]:
    """Arms the live pointer may be moved ONTO, read off the frozen contract.

    DERIVED, not typed — and deliberately derived from the POINTER CONTRACT
    rather than from a mirror of ``crucible-executor/executor/champion.py``.
    A mirror of another repo's literal is what this whole change exists to
    delete: the previous version of this constant was written on 2026-08-29
    as a copy of that tuple and was stale within hours, because
    ``alpha-engine-config-I9299`` landed the same day and made
    ``no_agent_quant`` and ``single_agent_quant`` servable. A copy cannot
    detect that it has gone stale; a contract read from disk cannot go stale
    without the file changing.

    It is the right boundary as well as the safer one: the pointer's
    ``champion`` enum is the promise this repo makes to the executor, the
    executor fail-louds on a value outside it (``ChampionPointerError``), and
    the enum is additive-only — so widening it is the deliberate act that
    admits a new arm, in the same repo as the writer.
    """
    schema = json.loads((path or POINTER_CONTRACT_PATH).read_text())
    return frozenset(schema["properties"]["champion"]["enum"]) - WRITE_FORBIDDEN_ARMS


#: Evaluated once at import: a contract that cannot be read is a reason to
#: refuse to load, not to guess a permissive default.
POINTER_ADMISSIBLE_ARMS: frozenset[str] = pointer_admissible_arms()

#: Arms this repo knows are real but that NO producer leaderboard has ever
#: listed, with the earliest date each is documented to have started producing
#: and where that date comes from.
#:
#: ``scanner_top20_predictor`` is the live example and the reason this exists:
#: Brian's 2026-08-27 ruling names it, ``config/producer_champion.json``'s
#: schema admits it as a pointer value, and crucible-executor can serve it —
#: yet it appears on no board, because until alpha-engine-config-I9307 it was
#: scored ONLY as a crucible-backtester end-to-end counterfactual, on a
#: different source and a different cohort from every other arm. Leaving it
#: out of the register because the board is silent about it would reproduce
#: exactly the silent-omission defect this change closes; registering it
#: makes its silence visible as a named :class:`SeriesGap` every cycle
#: instead.
#:
#: A board that later lists one of these arms WINS: the derivation takes the
#: minimum, so a real first-observed date always supersedes the seed.
UNBOARDED_ARMS: dict[str, tuple[str, str]] = {
    "scanner_top20_predictor": (
        "2026-07-30",
        "the top-20 cut began 2026-07-30 — recorded in this repo at "
        "optimizer/champion_promotion.py::_score_scanner_top20_predictor and in "
        "crucible-research producers/registry.py. No producer leaderboard has "
        "ever carried a row for this arm (verified across every artifact under "
        "research/producer_leaderboard/, 2026-08-03 .. 2026-08-28).",
    ),
}


# ── The slot's ArenaConfig (champion-challenger-policy.md §10) ─────────────
#
# §10 requires every slot to name its metric, benchmark and every ArenaConfig
# parameter in the registry that owns it, deliberately NOT in the policy —
# they are per-slot facts CI can check against code. This IS that registry row
# for the selection-producer slot.
#
#   metric                  topn_alpha_vs_population, per cohort date, 21
#                           trading sessions forward, top-N equal weight
#   benchmark               population (the scanner candidate set the arm
#                           narrowed) — SPY is REFUSED by the engine here
#   count-matching width    top_n = 50, held constant across arms by
#                           crucible-research's producer board
#
# ``diff_clip`` — declared bound on a per-date score DIFFERENCE between two
# arms, in the score's own units (21-day excess return over the drawn-from
# population, as a fraction). Justified from the observed range rather than
# picked: across every producer leaderboard that carries the population
# metric, the per-arm cumulative means run from -0.070646 (thinktank_coverage,
# 2026-08-28) to +0.018975 (no_agent_quant, 2026-08-21) — a widest observed
# CROSS-ARM gap of 0.0861 (no_agent_quant vs thinktank_coverage, 2026-08-21)
# and 0.0752 on 2026-08-28. 0.10 sits just above the widest observed gap, so
# the clip bounds the sub-Gaussian scale without truncating a difference the
# slot has actually produced. Clipping tighter would bias every comparison
# toward the incumbent by shrinking real leads; the count of clipped
# observations is reported on the artifact (`n_clipped`) so the choice stays
# reviewable rather than assumed.
#
# Consequence, stated rather than discovered later: with
# ``variance_mode="declared"`` the sub-Gaussian scale IS 0.10, so the interval
# is wide and a promotion needs a sustained lead rather than one good cohort.
# That is the intended trade — the alternative is a bar this slot's history
# (2026-07-13 pointer, never moved on evidence) shows it cannot honestly
# clear. If the slot proves unable to promote within ``opt_n`` cycles on a
# real lead, the declared next step is ``variance_mode="empirical"``, which is
# tighter and is what most practitioners use, at the cost of an interval that
# is no longer checkable from configuration alone.
ARENA_CONFIG = ArenaConfig(
    slot=SLOT,
    slot_kind=SLOT_KIND,
    benchmark="population",
    alpha=0.05,
    diff_clip=0.10,
    variance_mode="declared",
    opt_n=26,
    # Well-formedness only: one paired date is the least from which any
    # statistic can be formed. NOT an evidence bar — the confidence sequence
    # is the evidence bar, and every `thin_evidence` / minimum-cohort /
    # minimum-week floor on this slot's decision path is deleted by this
    # change (issue deliverable 6).
    min_paired_dates=1,
    cap=5,
    grace_weeks=4,
    min_active_arms=3,
    # Matches crucible-research's RETIRED_TRAILING_WINDOW_CYCLES = 8, so the
    # two repos score a retired arm over the same window.
    retired_trailing_cycles=8,
    retire_evidence="point",
    max_ladder_weeks=26,
)


@dataclass(frozen=True)
class SeriesGap:
    """An arm the register knows about that this cycle could not score.

    Carried onto the cycle's own record and into the ops alert. Deliberately
    a first-class value rather than a log line: an arm that silently drops
    out of the roster is the ``thinktank_coverage`` defect, and the whole
    point of §7.2 is that absence must be as legible as presence.
    """

    arm_name: str
    arm_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {"arm_name": self.arm_name, "arm_id": self.arm_id, "reason": self.reason}


# ── The register ──────────────────────────────────────────────────────────


def _spec_for(name: str) -> dict[str, str]:
    """The immutable identity a producer arm's ``arm_id`` hashes.

    Deliberately minimal and STABLE. The arm's real recipe — features,
    prompt, top-N width, refit cadence — lives in crucible-research's
    ``producers/registry.py`` and is not published on the leaderboard
    artifact, so hashing anything the board DOES carry (``kind``,
    ``promotion_eligible``) would mint a brand-new arm, and destroy the
    track record, the first time an arm was retired or its eligibility
    changed. Identity is the arm's NAME in the producer registry; the
    register's ``notes`` record where the recipe itself lives.
    """
    return {"producer_name": name, "registry": "crucible-research/producers/registry.py"}


def arm_id_for(name: str) -> str:
    from nousergon_lib.arena import derive_arm_id

    return derive_arm_id(SLOT, name, _spec_for(name))


def load_register(path: Path | None = None) -> ArmRegister:
    """Load the durable append-only register. RAISES if it is missing.

    No fallback to an empty register: an empty one would silently reset every
    arm's ``created_date`` to today and disable the four-week grace period
    across the whole pool, which is exactly the shape of failure this slot
    already suffered once.
    """
    target = path or REGISTER_PATH
    payload = json.loads(target.read_text())
    return ArmRegister.from_dicts(payload["events"])


def register_events_from_boards(boards: list[dict]) -> list[dict]:
    """DERIVE the register from a chronological list of producer leaderboards.

    This is the backfill, expressed as a pure function so the committed
    artifact is reproducible and testable rather than a hand-typed fixture —
    three fixtures in this fleet rotted this week by restating a registry as
    a literal.

    ``created_date`` is the earliest date on which an arm is OBSERVED: the
    minimum of its own ``dates_scored`` (when the board publishes them) and
    the date of the earliest board that lists it. That is a lower bound, and
    the register says so in ``notes`` — an arm whose true registration
    predates the artifact history gets the honest "first observed on this
    board" value rather than a guessed one. It never moves once written.

    An arm the board reports as ``kind == "retired"`` gets a retirement event
    dated to the first board on which it appeared retired, for the same
    reason.
    """
    seen: dict[str, str] = {}
    seeded: dict[str, str] = {}
    retired: dict[str, str] = {}
    order: list[str] = []
    for board in sorted(boards, key=lambda b: b.get("date") or ""):
        board_date = board["date"]
        rows: list[dict] = list(board.get("arms") or board.get("specs") or [])
        for row in rows:
            name = row.get("name")
            if not name:
                continue
            dates = [d for d in (row.get("dates_scored") or []) if isinstance(d, str)]
            first_seen = min([board_date, *dates])
            if name not in seen:
                seen[name] = first_seen
                order.append(name)
            elif first_seen < seen[name]:
                seen[name] = first_seen
            if row.get("kind") == "retired" and name not in retired:
                retired[name] = board_date

    for name, (seed_date, provenance) in UNBOARDED_ARMS.items():
        if name not in seen:
            seen[name] = seed_date
            order.append(name)
            seeded[name] = provenance
        elif seed_date < seen[name]:
            seen[name] = seed_date
            seeded[name] = provenance

    events: list[dict] = []
    for name in sorted(order, key=lambda n: (seen[n], n)):
        arm_id = arm_id_for(name)
        events.append(
            {
                "kind": "registered",
                "arm_id": arm_id,
                "date": seen[name],
                "reason": "",
                "record": {
                    "arm_id": arm_id,
                    "slot": SLOT,
                    "name": name,
                    "spec_hash": arm_id.rsplit(":", 1)[-1],
                    "created_date": seen[name],
                    "supersedes": None,
                    "bootstrap": False,
                    "notes": seeded.get(
                        name,
                        "created_date is FIRST OBSERVED on research/producer_leaderboard/; "
                        "the arm's true registration may predate the artifact history. "
                        "Recipe: crucible-research/producers/registry.py::RESEARCH_PRODUCERS.",
                    ),
                },
            }
        )
    for name, retired_date in sorted(retired.items()):
        events.append(
            {
                "kind": "retired",
                "arm_id": arm_id_for(name),
                "date": retired_date,
                "reason": (
                    "kind=='retired' on research/producer_leaderboard/ from this date; "
                    "scored for the champion-challenger-policy.md §3 trailing window"
                ),
            }
        )
    return events


def promotion_eligible_arm_names(register: ArmRegister | None = None) -> tuple[str, ...]:
    """Every ACTIVE arm's producer name, in register order.

    THE resolution of the arm roster for this repo. ``champion_promotion.
    VALID_CHAMPIONS`` is this, and is no longer a hand-typed tuple: the tuple
    was a second, independent register that silently omitted ``no_agent_quant``
    and ``single_agent_quant`` — the two arms with the most evidence — with
    nothing anywhere recording the omission or its reason.
    """
    reg = register or load_register()
    return tuple(reg.state(a).record.name for a in reg.active_arms())


def roster_disagreement(register: ArmRegister, leaderboard: dict | None) -> list[str]:
    """Arms the board scores that the register does not know about.

    The class defect this closes is four hand-maintained rosters drifting
    apart. One is now derived; this is the guard that the DERIVED one has not
    fallen behind its source. Returns names, never raises — the caller decides
    whether the disagreement is fatal for the cycle it is running.
    """
    if not isinstance(leaderboard, dict):
        return []
    known = {register.state(a).record.name for a in register.all_arms()}
    rows = list(leaderboard.get("arms") or leaderboard.get("specs") or [])
    return sorted({r["name"] for r in rows if isinstance(r, dict) and r.get("name")} - known)


# ── The series ────────────────────────────────────────────────────────────


def build_series(
    register: ArmRegister, leaderboard: dict | None, as_of: str,
) -> tuple[dict[str, ArmSeries], list[SeriesGap]]:
    """Per-date POPULATION-relative scores for every arm the cycle must score.

    Covers exactly ``register.scored_arms(as_of, retired_trailing_cycles)`` —
    active arms plus retired arms inside their §3 trailing window — because
    ``run_cycle`` RAISES on a missing series and RAISES on a series for an
    unregistered arm. Both raises are the point: the first stops an arm
    quietly dropping out of the contest, the second is the
    ``thinktank_coverage`` defect (output written with no register row, data
    rotting unnoticed).

    Every arm routes through THIS one path — the champion included. There is
    no per-arm scorer, no arm scored from a different source on a different
    cohort, and therefore no way for one arm's silence to render as another
    arm's thinness. That asymmetry is what hid the champion's two-date cohort
    behind two challengers' six.
    """
    rows_by_name: dict[str, dict] = {}
    if isinstance(leaderboard, dict):
        for row in leaderboard.get("specs") or []:
            if isinstance(row, dict) and row.get("name"):
                rows_by_name[row["name"]] = row

    series: dict[str, ArmSeries] = {}
    gaps: list[SeriesGap] = []
    for arm_id in register.scored_arms(as_of, ARENA_CONFIG.retired_trailing_cycles):
        name = register.state(arm_id).record.name
        row = rows_by_name.get(name)
        if row is None:
            gaps.append(
                SeriesGap(
                    name,
                    arm_id,
                    "not present in research/producer_leaderboard/ specs — the arm is "
                    "registered but the board built no history for it (no "
                    "signals_shadow/ writer, or the board predates its registration)",
                )
            )
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=frozenset())
            continue
        by_date = row.get(POPULATION_SERIES_FIELD)
        dates_scored = frozenset(
            d for d in (row.get("dates_scored") or []) if isinstance(d, str)
        )
        if not isinstance(by_date, dict) or not by_date:
            gaps.append(
                SeriesGap(
                    name,
                    arm_id,
                    f"research/producer_leaderboard/ carries no {POPULATION_SERIES_FIELD!r} "
                    "for this arm; it publishes only the aggregate "
                    "topn_alpha_vs_population. A confidence sequence cannot be formed "
                    "from cumulative means, and the SPY-relative "
                    "topn_alpha_vs_benchmark is REFUSED for a selection-stage slot "
                    "(SPY trailed the drawn-from population by 140bp at 21d on "
                    "2026-08-17, which inverts wins and losses). Upstream one-field "
                    "emission tracked as alpha-engine-config-I9405",
                )
            )
            series[arm_id] = ArmSeries(arm_id=arm_id, scores={}, misses=dates_scored)
            continue
        scores = {d: float(v) for d, v in by_date.items() if v is not None}
        series[arm_id] = ArmSeries(
            arm_id=arm_id,
            scores=scores,
            misses=frozenset(dates_scored - set(scores)),
        )
    return series, gaps


# ── Serving preconditions ─────────────────────────────────────────────────


def build_preconditions(
    register: ArmRegister,
    series_by_arm: dict[str, ArmSeries],
    *,
    shadow_only_names: frozenset[str],
    feed_blocked_names: dict[str, str] | None = None,
) -> dict[str, tuple[ServingPrecondition, ...]]:
    """The slot's hard gates on SERVING, evaluated here and passed in.

    Three, each a per-arm FACT recorded with its reason on the artifact
    rather than a hidden veto: the arm is declared shadow-only (measured,
    never served); the arm's live-trade feed producer looks dead; the frozen
    pointer contract does not admit the arm as a ``champion`` value
    (:data:`POINTER_ADMISSIBLE_ARMS`).

    The third one passes for every arm registered today, and that is a
    measurement rather than a design flaw — every arm on the board is in the
    enum as of ``alpha-engine-config-I9299``. It is reachable and it fires:
    an arm that appears on the producer board before this repo widens the
    enum is held off the pointer and named, which is exactly the sequence
    I9299 was filed for after the executor was found to have no handler for
    two arms Brian had already ruled eligible.

    These are the ONLY things that stop an arm taking the pointer. There is
    no hysteresis margin, no cooldown, and no evidence floor here — the
    confidence sequence is the evidence bar and the pointer moves freely in
    both directions (Brian ruling 2026-08-29, policy §5.2).
    """
    blocked = feed_blocked_names or {}
    out: dict[str, tuple[ServingPrecondition, ...]] = {}
    for arm_id in series_by_arm:
        name = register.state(arm_id).record.name
        out[arm_id] = (
            ServingPrecondition(
                name="not_shadow_only",
                passed=name not in shadow_only_names,
                reason=(
                    "" if name not in shadow_only_names
                    else f"{name} is declared shadow-only: measured every cycle, never served"
                ),
            ),
            ServingPrecondition(
                name="feed_producer_live",
                passed=name not in blocked,
                reason=blocked.get(name, ""),
            ),
            ServingPrecondition(
                name="pointer_contract_admits",
                passed=name in POINTER_ADMISSIBLE_ARMS,
                reason=(
                    "" if name in POINTER_ADMISSIBLE_ARMS
                    else (
                        f"contracts/producer_champion.schema.json does not admit {name!r} as a "
                        f"`champion` value (admitted: {sorted(POINTER_ADMISSIBLE_ARMS)}). The "
                        "executor fail-louds on an unrecognized pointer and refuses to start "
                        "a planning cycle, so moving the pointer here would halt trading. "
                        "Widening the enum is the deliberate act that admits the arm"
                    )
                ),
            ),
        )
    return out


# ── The cycle ─────────────────────────────────────────────────────────────


def run_arena_cycle(
    *,
    as_of: str,
    leaderboard: dict | None,
    incumbent_name: str | None,
    shadow_only_names: frozenset[str],
    feed_blocked_names: dict[str, str] | None = None,
    register: ArmRegister | None = None,
) -> tuple[ArenaCycle, list[SeriesGap], ArmRegister]:
    """One evaluation cycle of the selection-producer slot.

    ``training=None``: this slot's arms are selection RECIPES, not fitted
    models — there is no per-arm fit for the slot to vouch for, and asserting
    a training status it cannot observe would be provenance that is not true
    by construction (§7.5).
    """
    reg = register or load_register()
    unknown = roster_disagreement(reg, leaderboard)
    if unknown:
        # Fail LOUD on a producer. A board scoring an arm this repo has never
        # heard of means the derived roster has fallen behind its source, and
        # every comparison this cycle would be taken over an incomplete pool.
        raise ValueError(
            f"research/producer_leaderboard/ scores arm(s) {unknown} that "
            f"{REGISTER_PATH.name} does not register. The register is DERIVED from that "
            "board and has fallen behind it; re-run "
            "scripts/backfill_producer_arena_register.py and commit the result. "
            "Deciding the live pointer over an incomplete pool is the defect "
            "champion-challenger-policy.md §3 exists to prevent."
        )

    series_by_arm, gaps = build_series(reg, leaderboard, as_of)
    incumbent_id = arm_id_for(incumbent_name) if incumbent_name else None
    if incumbent_id is not None and incumbent_id not in series_by_arm:
        # A pointer sitting on an arm no longer in the scored set (retired
        # past its trailing window, or removed upstream). Not silently
        # bootstrapped away: `decide_pointer` handles `incumbent not in
        # series` as the §9.1 bootstrap path and says so on the artifact.
        logger.warning(
            "[producer_arena] incumbent %r is not in this cycle's scored set — the "
            "engine will take the §9.1 bootstrap path and the artifact will say so",
            incumbent_name,
        )
    cycle = run_cycle(
        config=ARENA_CONFIG,
        as_of=as_of,
        register=reg,
        series_by_arm=series_by_arm,
        incumbent=incumbent_id,
        preconditions=build_preconditions(
            reg,
            series_by_arm,
            shadow_only_names=shadow_only_names,
            feed_blocked_names=feed_blocked_names,
        ),
        training=None,
    )
    return cycle, gaps, reg


def cycle_document(cycle: ArenaCycle, gaps: list[SeriesGap]) -> dict[str, Any]:
    """The durable artifact body, validated against the ``arena_cycle`` contract.

    Validation happens HERE — on the producer side, before the write — so a
    contract break surfaces at the earliest call site rather than in a
    consumer weeks later (M0 contract discipline).
    """
    doc = cycle.to_dict()
    validate_contract("arena_cycle", doc)
    # Additive, after validation so it can never be the reason a valid cycle
    # is refused: the arms this cycle could not score, by name and reason.
    # `scored_arms` alone cannot express "present in the roster, no series".
    doc["series_gaps"] = [g.to_dict() for g in gaps]
    return doc


def write_arena_cycle(
    bucket: str, as_of: str, doc: dict[str, Any], *, upload: bool, s3_client=None,
) -> str | None:
    """Write ``arena/producer/{date}.json`` + ``latest.json``.

    RAISES on failure. §11: a slot that emits nothing is not healthy, it is
    unobserved — a swallowed write here would make an unrun cycle and a
    silent one indistinguishable, which is the exact class this artifact
    exists to retire.
    """
    dated_key = f"{ARENA_CYCLE_PREFIX}/{as_of}.json"
    if not upload:
        logger.info("[producer_arena] arena_cycle write skipped (upload=False): %s", dated_key)
        return None
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(doc, indent=2, allow_nan=False).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(
        Bucket=bucket, Key=f"{ARENA_CYCLE_PREFIX}/latest.json", Body=body,
        ContentType="application/json",
    )
    logger.info("[producer_arena] arena_cycle written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


#: Decision statuses that mean the slot could not decide on evidence. §11:
#: both are first-class, both carry a reason, and both ALARM.
ALARMING_STATUSES = ("unmeasurable", "unservable")


def publish_cycle_alert(cycle: ArenaCycle, gaps: list[SeriesGap]) -> None:
    """Alarm on an ``unmeasurable`` / ``unservable`` cycle (§11).

    Best-effort: an alerting failure must not red the weekly pipeline it
    reports on. It cannot become a second silence — the status and every gap
    are already durable on the ``arena_cycle`` artifact before this runs, and
    the failure is logged with a traceback.
    """
    if cycle.decision.status not in ALARMING_STATUSES:
        return
    detail = "; ".join(f"{g.arm_name}: {g.reason}" for g in gaps) or "no per-arm gap recorded"
    message = (
        f"producer arena cycle {cycle.as_of} is {cycle.decision.status}: "
        f"{cycle.decision.reason}. The live config/producer_champion.json pointer is "
        f"HELD at {cycle.decision.champion!r}. Per-arm gaps: {detail}. "
        f"See s3 {ARENA_CYCLE_PREFIX}/{cycle.as_of}.json."
    )
    try:
        from ops_alerts import publish_ops_alert

        publish_ops_alert(
            message,
            severity="error",
            # `alpha-engine-backtester/`, not `crucible-backtester/`: the
            # canonical alert-source prefix the overseer's `alert_classes`
            # registry keys on (alpha-engine-config-I3302 completed that
            # rename). A second prefix for the same repo would land the class
            # in the registry twice under two names.
            source="alpha-engine-backtester/optimizer/producer_arena.py::run_arena_cycle",
            dedup_key=f"producer_arena_{cycle.decision.status}_{cycle.as_of}",
            dedup_window_min=720,
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the weekly run
        logger.exception(
            "[producer_arena] %s alert publish failed (best-effort); the status and "
            "every gap remain durable on the arena_cycle artifact",
            cycle.decision.status,
        )
