"""The strategy (S) slot, wired to the shared arena engine (alpha-engine-config-I9320).

Normative source: ``nous-ergon-ops/policies/champion-challenger-policy.md``.
The decision machinery — score ladder, longest-common-window pairing, the
anytime-valid confidence sequence, the immutable arm register, pairwise-wins
ranking and cap-with-grace retirement — is ``nousergon_lib.arena`` and is NOT
re-implemented here. A slot that re-implements §§3-6 is a defect
(``shared-code-policy.md``). This module supplies only the four things the
engine cannot know: what an S-slot arm IS, what its per-date score IS, which
serving preconditions apply, and where the cycle artifact goes.

WHAT THIS SLOT DECIDES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------------
The S slot's champion is a **strategy recipe** — an exit/risk rule chain plus
the parameter bundle it runs under. It is the only slot whose output IS a
market position, which is why §4's population benchmark does not apply here
and market-relative grading against SPY stays correct (:data:`BENCHMARK`).

**This module is PROPOSE-ONLY and that is a hard boundary, not a phase.**
``overseer-policy.md`` §8's first bullet is untouched by the 2026-08-29
champion/challenger carve-out:

    "Order-path changes — executor risk guard, position sizer, daemon, entry
    triggers, intraday exit manager, strategies, order book, EOD reconcile,
    broker adapter. Propose-only, always, regardless of tier or kill-switch
    state."

The carve-out permits automation to move a champion pointer *between
registered arms*; it permits no automated edit to a strategy's code or to any
live order-path config. So this module writes its verdict to
:data:`PROPOSAL_KEY` — a proposal nothing on the order path reads — and
:func:`assert_no_order_path_write` refuses, loudly, any attempt to route a
write from here to a key in :data:`ORDER_PATH_KEYS`.
:func:`assert_outside_market_hours` refuses to run the cycle at all inside a
regular NYSE session, because "any remediation against the live trading path
during market hours" is separately never-autonomous, whatever the verdict says.

The two guards are deliberately independent. A key check alone would pass a
cycle that ran at 11:00 ET and wrote only a proposal — harmless today, and
exactly the shape that becomes harmful the moment someone wires a consumer to
the proposal. A clock check alone would pass a Saturday run that wrote
``config/executor_params.json``. Neither guard subsumes the other.

FINDING RECORDED HERE RATHER THAN SILENTLY FIXED (alpha-engine-config-I9399)
----------------------------------------------------------------------------
``optimizer/executor_optimizer.py::apply`` writes ``config/executor_params.json``
— a live order-path config — from a weekly automated grid search, with no human
in the loop. That is an autonomous order-path change, and it is not the
carve-out: the carve-out lets automation choose among arms a human already
approved, and a grid search *invents* the parameter bundle it applies. This
module does not change that behaviour, because narrowing a live production
authority is a ruling and not a side effect of wiring a slot. It is filed for
Brian, and :func:`assert_no_order_path_write` documents the boundary this
slot holds in the meantime.

THE SLOT HAS ONE ARM TODAY, AND THAT IS REPORTED, NOT PAPERED OVER
-------------------------------------------------------------------
``crucible-executor/executor/strategies/contract.py::stock_registry`` declares
exactly one rule chain, and no competing strategy recipe has ever had a shadow
write path. Under §9.2 a slot with one plausible rule does not need a
challenger — but it still needs its champion scored, "so that 'we never
checked' is never the answer to how it is performing". :func:`run_cycle` is
therefore called with whatever arms genuinely have a series, and a slot below
``min_active_arms`` renders as :data:`STATUS_SINGLE_ARM` on the artifact and
alarms. It never renders as green, and it never renders as a promotion.
``principles.md`` §2.7: a component emitting nothing is not healthy, it is
unobserved.

NOT YET SCHEDULED, AND WHY (alpha-engine-config-I9401)
------------------------------------------------------
This module is complete and tested; it has no weekly call site yet, because
two inputs cannot be obtained honestly today.

First, an S-slot recipe includes the ORDERED exit-rule chain, and that chain
lives in ``crucible-executor``, which is not a dependency of this repo. Naming
it here would mean a hardcoded literal of another repo's registry, guarded by
nothing — the exact shape that silently rotted three fixtures on 2026-08-29.
:func:`strategy_recipe` therefore takes the chain from its caller, and the
production caller waits on the executor publishing its own chain identity.

Second, §10 requires an arm's registry row, shadow write path and scoring
wiring to land together. ``simulate/portfolio_daily_returns.parquet`` carries
a per-date series for exactly ONE configuration — whichever was live that run
— so no competing recipe has ever had a series. :data:`SHADOW_KEY` declares
where that series will live; the producer is I9401's deliverable 2.

Both are tracked. What is deliberately NOT done is inventing a second arm, or
scheduling a cycle that would emit a well-formed artifact describing a
comparison that never happened — §7.2's dominant bug class.

MEASURABILITY
-------------
The number that says this is working is ``decision.status`` on
``arena/strategy/{date}.json``, together with ``len(active_arms)`` against
``cap``. Its ABSENCE is a missing or stale ``arena/strategy/latest.json`` —
a freshness-registry row — which pages. No data is never rendered as green.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timezone

from nousergon_lib.arena.arms import ArmRegister, spec_hash
from nousergon_lib.arena.engine import (
    ArenaConfig,
    ArenaCycle,
    ServingPrecondition,
    run_cycle,
)
from nousergon_lib.arena.window import ArmSeries
from nousergon_lib.contracts import conformance_errors

logger = logging.getLogger(__name__)

__all__ = [
    "SLOT_ID",
    "SLOT_KIND",
    "BENCHMARK",
    "STRATEGY_ARENA_CONFIG",
    "ARENA_DATED_KEY",
    "ARENA_LATEST_KEY",
    "PROPOSAL_KEY",
    "SHADOW_KEY",
    "ORDER_PATH_KEYS",
    "STATUS_SINGLE_ARM",
    "StrategyArenaError",
    "OrderPathWriteRefused",
    "MarketHoursRefused",
    "strategy_recipe",
    "build_arm_register",
    "market_relative_series",
    "arm_series_from_shadow",
    "input_completeness_precondition",
    "assert_no_order_path_write",
    "assert_outside_market_hours",
    "build_strategy_cycle",
    "write_strategy_cycle",
    "run_strategy_arena",
]

SLOT_ID = "strategy"
SLOT_KIND = "strategy"

#: Market-relative is legitimate HERE and only here. §4: "The
#: market-relative canonical-alpha framework remains correct for slots whose
#: output IS a market position (S), and stays the fleet default there." The
#: engine's ``SELECTION_SLOT_KINDS`` refusal does not apply to ``strategy``,
#: so this value is a declaration rather than something the engine enforces —
#: which is why it is asserted below and named on every artifact.
BENCHMARK = "SPY"

ARENA_DATED_KEY = "arena/strategy/{date}.json"
ARENA_LATEST_KEY = "arena/strategy/latest.json"

#: The verdict. Deliberately NOT an order-path key and deliberately NOT named
#: like one: nothing in crucible-executor reads it, and a test asserts that.
PROPOSAL_KEY = "config/strategy_champion_proposal.json"

#: One arm's per-date market-relative return series — the S slot's shadow
#: write path. §10: an arm registered without a scoring path is the
#: ``thinktank_coverage`` defect repeating.
SHADOW_KEY = "strategy_shadow/{arm_id}/{date}.json"

#: Keys this module may never write, at any time, for any reason. Not a
#: denylist of everything dangerous in the fleet — a denylist of the order-path
#: config THIS slot could plausibly be wired to by a later change. The guard's
#: value is that adding such a wire fails a test rather than reaching a broker.
ORDER_PATH_KEYS: frozenset[str] = frozenset(
    {
        "config/executor_params.json",
        "config/risk.yaml",
        "config/strategy_config.json",
        "config/predictor_params.json",
    }
)

#: A slot holding fewer than ``min_active_arms`` scored arms. §9.2 makes this
#: legitimate; §7.2 makes it loud. It is a first-class status carrying a
#: reason, never an empty pass.
STATUS_SINGLE_ARM = "single_arm"

_MARKET_OPEN_ET = time(9, 30)
_MARKET_CLOSE_ET = time(16, 0)


class StrategyArenaError(RuntimeError):
    """The S slot cannot produce a trustworthy cycle."""


class OrderPathWriteRefused(StrategyArenaError):
    """A write from this module targeted a live order-path key.

    Raised rather than logged: ``overseer-policy.md`` §8 makes order-path
    changes propose-only ALWAYS, and a swallowed refusal is indistinguishable
    from a permitted write in every artifact anyone would later read.
    """


class MarketHoursRefused(StrategyArenaError):
    """The cycle was invoked inside a regular NYSE session."""


# ── the slot registry row (champion-challenger-policy.md §10) ────────────────


@dataclass(frozen=True)
class StrategySlotSpec:
    """The measurement contract for the S slot, per §10.

    §10 requires every slot to name, in its registry, the metric, benchmark,
    count-matching width and its ``ArenaConfig`` parameters, "per-slot facts
    that CI can check against code". This is that row; the tests read it
    rather than restating its values.
    """

    slot_id: str
    primary_metric: str
    benchmark: str
    horizon: str
    #: Count-matching (§4) is vacuous for this slot and saying so is the
    #: point: arms are graded on the SAME book of entries, differing only in
    #: exit/risk rules, so there is no width to match. Recorded rather than
    #: omitted, because an absent field reads as an oversight.
    count_matched_width: str


STRATEGY_SLOT = StrategySlotSpec(
    slot_id=SLOT_ID,
    primary_metric="daily_market_relative_log_return",
    benchmark=BENCHMARK,
    horizon="daily",
    count_matched_width="not_applicable_same_entry_book",
)


#: ``diff_clip`` is a declared bound on the per-date score DIFFERENCE between
#: two arms, in the score's own units (§5.0: "The scale must be declared, not
#: discovered", so the sequence's validity is checkable from configuration
#: alone). The score here is one day's portfolio log return minus SPY's log
#: return. Two arms in this slot trade the SAME entry book and differ only in
#: exit and risk rules, so their daily difference is bounded by how much
#: exiting differently can move one day's book — not by how much the market
#: can move. 0.02 (200bp of daily divergence) is already an extreme day for
#: that quantity: a full liquidation by one arm against a full hold by the
#: other, on a day the held names moved 2%. Clipping is conservative in the
#: right direction — a clip too TIGHT widens the interval and delays a
#: promotion; a clip too LOOSE would understate it, so erring small is erring
#: safe.
_DIFF_CLIP = 0.02

STRATEGY_ARENA_CONFIG = ArenaConfig(
    slot=SLOT_ID,
    slot_kind=SLOT_KIND,
    benchmark=BENCHMARK,
    diff_clip=_DIFF_CLIP,
    # Fleet defaults, restated here because §10 requires the slot to NAME them
    # rather than inherit them silently — a parameter nobody wrote down is a
    # parameter nobody can check against code.
    cap=5,
    grace_weeks=4,
    min_active_arms=3,
    retired_trailing_cycles=8,
)

if STRATEGY_ARENA_CONFIG.benchmark != STRATEGY_SLOT.benchmark:
    raise StrategyArenaError(
        "the slot registry row and the ArenaConfig disagree about the benchmark "
        f"({STRATEGY_SLOT.benchmark!r} vs {STRATEGY_ARENA_CONFIG.benchmark!r}); two "
        "declarations of one fact is the drift §10 exists to prevent"
    )


# ── arms ────────────────────────────────────────────────────────────────────


def strategy_recipe(
    params: Mapping[str, object], rule_chain: Sequence[str]
) -> dict[str, object]:
    """One S-slot RECIPE: the exit-rule chain plus the parameters it runs under.

    §3.1: an arm fixes its features, hyperparameters, training-window rule and
    refit cadence at registration, and the arm id encodes the hash of its own
    spec so a changed recipe cannot reuse an id. A strategy has no fitted
    weights and no refit — it is deterministic given its parameters — so the
    recipe is the whole of it, and every S-slot arm is immutable by
    construction rather than by discipline.

    ``rule_chain`` is ORDERED and the order is part of the recipe: the exit
    registry short-circuits on the first rule that decides, so reordering the
    chain is a different strategy that happens to contain the same rules.
    """
    if not rule_chain:
        raise StrategyArenaError(
            "a strategy recipe with an empty rule chain decides nothing; refusing to "
            "register an arm that cannot produce a position"
        )
    return {
        "rule_chain": list(rule_chain),
        "params": {k: params[k] for k in sorted(params)},
    }


def build_arm_register(
    recipes_by_first_seen: Mapping[str, dict[str, object]],
    *,
    retired: Mapping[str, tuple[str, str]] | None = None,
) -> ArmRegister:
    """Fold the S slot's recipe history into an append-only :class:`ArmRegister`.

    ``recipes_by_first_seen`` maps an ISO date to the recipe first observed on
    that date — derived from the applied/shadow parameter history, NEVER
    restated as a literal. A registry restated as a literal in a fixture rots
    silently the first time the real registry moves, and the tests here derive
    the roster from this function's own output for exactly that reason.

    ``retired`` maps an arm NAME to ``(retired_date, reason)``. Retirement
    appends an event; nothing is overwritten (§3.1).
    """
    register = ArmRegister()
    names: dict[str, str] = {}
    for created_date in sorted(recipes_by_first_seen):
        recipe = recipes_by_first_seen[created_date]
        name = _recipe_name(recipe)
        if name in names:
            raise StrategyArenaError(
                f"recipe {name!r} first seen on both {names[name]} and {created_date}; a "
                "recipe has exactly one birth date and a second one means the "
                "history was folded twice. A birth date is what the grace window "
                "is measured from, so two of them is not a cosmetic problem."
            )
        names[name] = created_date
        register, _record = register.register(
            slot=SLOT_ID,
            name=name,
            spec=recipe,
            created_date=created_date,
            notes="derived from the executor parameter history",
        )
    for name, (retired_date, reason) in sorted((retired or {}).items()):
        arm_id = _arm_id_for(register, name)
        register = register.retire(arm_id, retired_date, reason)
    return register


def _recipe_name(recipe: Mapping[str, object]) -> str:
    """A stable, human-legible, UNIQUE name for a recipe.

    Uniqueness is not decoration here. An earlier cut of this function named a
    recipe after the SHAPE of its chain and parameter set — ``chain7_p2`` —
    which is stable and readable and collides the moment two recipes tune the
    same two knobs to different values, which is the ordinary case for this
    slot. Its own test caught it: three genuinely distinct recipes folded to
    one name and the register refused the history as double-folded.

    So the name carries the engine's own ``spec_hash`` prefix. The full hash
    remains the IDENTITY — ``ArmRegister.register`` derives the arm id from
    ``(slot, name, spec)`` — and this is only the label a human reads, but a
    label that can collide is a label that will eventually mislabel.
    """
    chain = recipe.get("rule_chain") or []
    return "strategy_chain{}_{}".format(len(chain), spec_hash(recipe)[:10])


def _arm_id_for(register: ArmRegister, name: str) -> str:
    matches = [
        arm for arm in register.all_arms() if register.state(arm).record.name == name
    ]
    if len(matches) != 1:
        raise StrategyArenaError(
            f"expected exactly one registered arm named {name!r}, found {len(matches)}"
        )
    return matches[0]


# ── scoring ─────────────────────────────────────────────────────────────────


def market_relative_series(
    portfolio_log_returns: Mapping[str, float],
    benchmark_log_returns: Mapping[str, float],
) -> dict[str, float]:
    """Per-date portfolio log return MINUS the benchmark's, on shared dates.

    The engine never applies a benchmark itself — the correct benchmark is a
    per-slot fact — so an :class:`ArmSeries` must arrive already
    benchmark-relative. This is where that happens for the S slot.

    A date the benchmark does not cover is DROPPED rather than treated as a
    zero benchmark return. A missing benchmark is not a flat market, and
    scoring an arm against an assumed-zero SPY on days SPY moved is the
    140bp-inversion failure of 2026-08-17 in miniature. The caller records
    such dates as misses if the arm genuinely produced on them.
    """
    out: dict[str, float] = {}
    for iso_date, value in portfolio_log_returns.items():
        bench = benchmark_log_returns.get(iso_date)
        if bench is None:
            continue
        if value != value or bench != bench:  # NaN
            raise StrategyArenaError(
                f"NaN return on {iso_date} (portfolio={value}, benchmark={bench}); a "
                "missing return is a MISS, not a NaN "
                "(champion-challenger-policy.md §3)"
            )
        out[iso_date] = float(value) - float(bench)
    return out


def arm_series_from_shadow(
    arm_id: str,
    portfolio_log_returns: Mapping[str, float],
    benchmark_log_returns: Mapping[str, float],
    *,
    expected_dates: Sequence[str] = (),
) -> ArmSeries:
    """One arm's :class:`ArmSeries`, with genuine absences recorded as misses.

    ``expected_dates`` are the dates this arm was expected to produce on. Any
    of them without a score becomes a MISS — §3 requires that silent absence
    and a genuine zero never render identically, and a zero market-relative
    return is a real and unremarkable outcome for a strategy.
    """
    scores = market_relative_series(portfolio_log_returns, benchmark_log_returns)
    misses = frozenset(d for d in expected_dates if d not in scores)
    return ArmSeries(arm_id=arm_id, scores=scores, misses=misses)


def input_completeness_precondition(
    arm_id: str,
    *,
    n_scored_dates: int,
    n_expected_dates: int,
) -> ServingPrecondition:
    """§5.3's second hard serving precondition, for one arm.

    "An arm scored on partial inputs may rank first and still be unfit to
    trade." This is a gate on SERVING and never a ranking input: the engine
    excludes a failing arm from consideration however far ahead it is, and a
    failing INCUMBENT forces the pointer to move.

    Deliberately NOT an evidence bar. It does not ask whether the arm has
    accumulated enough history to be judged — the confidence sequence is the
    only thing that asks that (§5.0) — it asks whether the dates the arm DID
    cover are the dates it was supposed to cover. An arm with three complete
    dates passes; an arm with three of a hundred does not.
    """
    if n_expected_dates <= 0:
        return ServingPrecondition(
            name="input_completeness",
            passed=False,
            reason=(
                f"{arm_id}: no dates were expected, so completeness is undefined; an "
                "undefined precondition is a failure, never a pass "
                "(champion-challenger-policy.md §5.1)"
            ),
        )
    complete = n_scored_dates == n_expected_dates
    return ServingPrecondition(
        name="input_completeness",
        passed=complete,
        reason=(
            ""
            if complete
            else f"{arm_id}: scored {n_scored_dates} of {n_expected_dates} expected dates"
        ),
    )


# ── the two guards (overseer-policy.md §8) ──────────────────────────────────


def assert_no_order_path_write(key: str) -> None:
    """Refuse, loudly, a write from this slot to a live order-path key."""
    if key in ORDER_PATH_KEYS:
        raise OrderPathWriteRefused(
            f"refusing to write {key!r} from the strategy arena: order-path changes are "
            "PROPOSE-ONLY, always, regardless of tier or kill-switch state "
            "(overseer-policy.md §8). The 2026-08-29 champion/challenger carve-out "
            "permits moving a pointer between registered arms; it permits no "
            f"automated edit to live order-path config. The verdict belongs in "
            f"{PROPOSAL_KEY!r}."
        )


def assert_outside_market_hours(now_utc: datetime | None = None) -> None:
    """Refuse to run the cycle inside a regular NYSE session.

    "Any remediation against the live trading path during market hours, of any
    kind" is never autonomous (``overseer-policy.md`` §8), and I9320 makes the
    consequence explicit: if wiring this slot would let automation change live
    order behaviour intraday, STOP rather than proceed.

    The trading calendar comes from ``krepis.dates`` rather than a second
    implementation here (``shared-code-policy.md``); only the intraday window
    is local, and that window IS the thing this guard is about.

    A naive datetime RAISES rather than being assumed to be UTC. A guard whose
    correctness depends on guessing a timezone is not a guard.
    """
    from krepis.dates import is_trading_day

    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise StrategyArenaError(
            "assert_outside_market_hours received a naive datetime; refusing to guess a "
            "timezone for a market-hours guard"
        )
    eastern = _to_eastern(now)
    if not is_trading_day(eastern.date()):
        return
    if _MARKET_OPEN_ET <= eastern.time() < _MARKET_CLOSE_ET:
        raise MarketHoursRefused(
            f"refusing to run the strategy arena cycle at {eastern.isoformat()} ET: this is "
            "a regular NYSE session, and any automated action against the live "
            "trading path during market hours is never autonomous "
            "(overseer-policy.md §8). Run it off-session."
        )


def _to_eastern(moment: datetime) -> datetime:
    from zoneinfo import ZoneInfo

    return moment.astimezone(ZoneInfo("America/New_York"))


# ── the cycle ───────────────────────────────────────────────────────────────


def build_strategy_cycle(
    *,
    as_of: str,
    register: ArmRegister,
    series_by_arm: Mapping[str, ArmSeries],
    incumbent: str | None,
    preconditions: Mapping[str, Sequence[ServingPrecondition]] | None = None,
    now_utc: datetime | None = None,
) -> ArenaCycle:
    """One S-slot arena cycle. Pure apart from the clock guard — no S3.

    ``training`` is deliberately not passed to :func:`run_cycle`: a strategy
    recipe is deterministic given its parameters and is never fitted, so there
    is no training substrate to vouch for. §3's training-integrity failure is
    a fitted-slot concern and inventing a vacuous always-OK status here would
    be a gate that cannot fail — §7.4's "indistinguishable from no guard, and
    worse, because it reads as coverage".
    """
    assert_outside_market_hours(now_utc)
    return run_cycle(
        config=STRATEGY_ARENA_CONFIG,
        as_of=as_of,
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        preconditions=preconditions,
        training=None,
    )


def cycle_document(cycle: ArenaCycle) -> dict:
    """The validated ``arena_cycle`` document, plus this slot's own status note.

    Validation happens at the PRODUCER (M0 contract discipline): a
    non-conforming artifact is never written, because a consumer discovering
    the violation later has already been given a document it cannot trust.
    """
    doc = cycle.to_dict()
    errors = conformance_errors("arena_cycle", doc)
    if errors:
        raise StrategyArenaError(
            "the strategy arena cycle does not conform to the arena_cycle contract and "
            f"will not be written: {errors}"
        )
    n_active = len(cycle.active_arms)
    if n_active < STRATEGY_ARENA_CONFIG.min_active_arms:
        doc["slot_status"] = STATUS_SINGLE_ARM
        doc["slot_status_reason"] = (
            f"{n_active} active arm(s) against min_active_arms="
            f"{STRATEGY_ARENA_CONFIG.min_active_arms}: no promotion is possible and none "
            "should be inferred. §9.2 permits a slot with one plausible rule; §3 still "
            "requires its champion be scored, so that 'we never checked' is never the "
            "answer to how it is performing."
        )
    return doc


def write_strategy_cycle(
    doc: Mapping[str, object],
    bucket: str,
    as_of: str,
    *,
    s3_client=None,
) -> list[str]:
    """Write the cycle artifact — dated first, then the latest mirror.

    Dated FIRST, deliberately: if the second write fails, the immutable record
    exists and the mirror is stale, which a freshness probe catches. The
    reverse order would leave a mirror pointing at a cycle that was never
    persisted.
    """
    import boto3

    s3 = s3_client or boto3.client("s3")
    keys = [ARENA_DATED_KEY.format(date=as_of), ARENA_LATEST_KEY]
    body = json.dumps(doc, indent=2, sort_keys=True, default=str).encode()
    for key in keys:
        assert_no_order_path_write(key)
        s3.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )
    return keys


def write_champion_proposal(
    doc: Mapping[str, object],
    bucket: str,
    *,
    s3_client=None,
) -> str:
    """Write the PROPOSAL. Never the live order-path config.

    §8 again: this is the whole output of the slot's decision as far as
    production is concerned. Nothing in crucible-executor reads
    :data:`PROPOSAL_KEY`, and a test asserts that no consumer appears.
    """
    import boto3

    assert_no_order_path_write(PROPOSAL_KEY)
    s3 = s3_client or boto3.client("s3")
    payload = {
        "schema_version": 1,
        "slot": SLOT_ID,
        "as_of": doc.get("as_of"),
        "proposed_champion": (doc.get("decision") or {}).get("champion"),
        "incumbent": (doc.get("decision") or {}).get("incumbent"),
        "status": (doc.get("decision") or {}).get("status"),
        "reason": (doc.get("decision") or {}).get("reason"),
        "slot_status": doc.get("slot_status"),
        "advisory": (
            "PROPOSAL ONLY. Order-path changes are propose-only, always "
            "(overseer-policy.md §8). Applying this requires a human."
        ),
    }
    s3.put_object(
        Bucket=bucket,
        Key=PROPOSAL_KEY,
        Body=json.dumps(payload, indent=2, sort_keys=True, default=str).encode(),
        ContentType="application/json",
    )
    return PROPOSAL_KEY


def run_strategy_arena(
    *,
    bucket: str,
    as_of: str,
    register: ArmRegister,
    series_by_arm: Mapping[str, ArmSeries],
    incumbent: str | None,
    preconditions: Mapping[str, Sequence[ServingPrecondition]] | None = None,
    s3_client=None,
    now_utc: datetime | None = None,
) -> dict:
    """Score, decide, validate, write. The slot's whole cycle.

    Fail-loud throughout: nothing here degrades gracefully, because every
    artifact this writes is read as a statement that the slot was evaluated.
    §7.2's dominant bug class is a well-formed artifact containing nothing.
    """
    cycle = build_strategy_cycle(
        as_of=as_of,
        register=register,
        series_by_arm=series_by_arm,
        incumbent=incumbent,
        preconditions=preconditions,
        now_utc=now_utc,
    )
    doc = cycle_document(cycle)
    keys = write_strategy_cycle(doc, bucket, as_of, s3_client=s3_client)
    proposal_key = write_champion_proposal(doc, bucket, s3_client=s3_client)
    logger.info(
        "strategy arena cycle %s: status=%s champion=%s active_arms=%d keys=%s",
        as_of,
        doc.get("decision", {}).get("status"),
        doc.get("decision", {}).get("champion"),
        len(doc.get("active_arms") or []),
        keys + [proposal_key],
    )
    return {"document": doc, "keys": keys, "proposal_key": proposal_key}
