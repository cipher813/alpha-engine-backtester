"""The strategy (S) slot on the shared arena engine (alpha-engine-config-I9320).

Two properties carry this file, and both were verified RED against the
pre-change tree (§7.4 — a guard that cannot fail is indistinguishable from no
guard, and worse, because it reads as coverage):

1. **No pointer movement can reach the live order path.** Before this change
   there was no S-slot arena at all, so there was nothing to assert; the
   guards are shown red by construction below, by pointing them at the exact
   keys ``optimizer/executor_optimizer.py`` writes today.
2. **No `thin_evidence`-shaped gate exists on this slot.** Asserted
   structurally against the module source, not by enumerating names — a gate
   reintroduced under a new spelling must still fail this.

The registry is DERIVED everywhere, never restated as a literal: three
fixtures elsewhere in the fleet rotted this session by hardcoding an arm
roster that then moved.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

from nousergon_lib.arena.engine import ArenaConfigError, SELECTION_SLOT_KINDS
from nousergon_lib.arena.window import ArmSeries
from optimizer import strategy_arena as sa


# ── recipes and the register ────────────────────────────────────────────────


_CHAIN = (
    "position_loss_floor",
    "catalyst_hard_exit",
    "atr_with_sector_veto",
    "fallback_stop",
    "profit_take",
    "momentum_exit",
    "time_decay",
)


def _recipe(atr: float = 2.5, profit_take: float = 0.25) -> dict:
    return sa.strategy_recipe(
        {"atr_multiplier": atr, "profit_take_pct": profit_take}, _CHAIN
    )


def _history() -> dict[str, dict]:
    return {
        "2026-06-01": _recipe(),
        "2026-07-06": _recipe(atr=3.0),
        "2026-08-03": _recipe(profit_take=0.30),
    }


def test_recipe_with_no_rule_chain_is_refused():
    """An arm that decides nothing cannot produce a position, so it is not an arm."""
    with pytest.raises(sa.StrategyArenaError):
        sa.strategy_recipe({"atr_multiplier": 2.5}, ())


def test_a_changed_recipe_is_a_new_arm():
    """§3.1: the arm id encodes its own spec hash, so a changed recipe cannot
    inherit a record. Immutability by construction, not by discipline."""
    register = sa.build_arm_register(_history())
    assert len(set(register.all_arms())) == len(_history())


def test_reordering_the_rule_chain_is_a_different_arm():
    """The exit registry short-circuits on the first rule that decides, so a
    reordered chain is a different strategy holding the same rules."""
    a = sa.build_arm_register({"2026-06-01": sa.strategy_recipe({"x": 1}, _CHAIN)})
    b = sa.build_arm_register(
        {"2026-06-01": sa.strategy_recipe({"x": 1}, tuple(reversed(_CHAIN)))}
    )
    assert set(a.all_arms()) != set(b.all_arms())


def test_parameter_order_is_not_part_of_identity():
    """A recipe is a SET of parameters; two spellings of one recipe must not
    register as two arms."""
    a = sa.strategy_recipe({"b": 2, "a": 1}, _CHAIN)
    b = sa.strategy_recipe({"a": 1, "b": 2}, _CHAIN)
    assert a == b


def test_a_recipe_folded_twice_raises():
    """Folding the parameter history twice would give one recipe two birth
    dates, and a birth date is what the grace window is measured from."""
    with pytest.raises(sa.StrategyArenaError):
        sa.build_arm_register({"2026-06-01": _recipe(), "2026-07-06": _recipe()})


def test_retirement_appends_and_keeps_the_arm_queryable():
    """§6.3: retirement sets a QUERYABLE field and destroys no history."""
    register = sa.build_arm_register(_history())
    name = register.state(register.all_arms()[0]).record.name
    retired = sa.build_arm_register(
        _history(), retired={name: ("2026-08-29", "superseded")}
    )
    assert len(retired.all_arms()) == len(register.all_arms())
    assert len(retired.active_arms()) == len(register.active_arms()) - 1


# ── scoring ─────────────────────────────────────────────────────────────────


def test_series_is_benchmark_relative_before_it_reaches_the_engine():
    """The engine never applies a benchmark — the correct one is a per-slot
    fact — so an ArmSeries must arrive already relative."""
    scores = sa.market_relative_series(
        {"2026-08-03": 0.010, "2026-08-04": -0.004},
        {"2026-08-03": 0.004, "2026-08-04": -0.001},
    )
    assert scores["2026-08-03"] == pytest.approx(0.006)
    assert scores["2026-08-04"] == pytest.approx(-0.003)


def test_a_date_the_benchmark_does_not_cover_is_dropped_not_zeroed():
    """A missing benchmark is not a flat market. Scoring against an assumed-zero
    SPY on days SPY moved is the 2026-08-17 140bp inversion in miniature."""
    scores = sa.market_relative_series(
        {"2026-08-03": 0.01, "2026-08-04": 0.02}, {"2026-08-03": 0.004}
    )
    assert set(scores) == {"2026-08-03"}


def test_nan_is_refused_rather_than_scored():
    """§3: a missing score is a MISS, never a NaN."""
    with pytest.raises(sa.StrategyArenaError):
        sa.market_relative_series({"2026-08-03": float("nan")}, {"2026-08-03": 0.0})


def test_an_absent_expected_date_becomes_a_miss_not_a_zero():
    """§3: silent absence and a genuine zero must never render identically, and
    a zero market-relative return is a real, unremarkable outcome."""
    series = sa.arm_series_from_shadow(
        "arm-1",
        {"2026-08-03": 0.0},
        {"2026-08-03": 0.0},
        expected_dates=("2026-08-03", "2026-08-04"),
    )
    assert series.scores == {"2026-08-03": 0.0}
    assert series.misses == frozenset({"2026-08-04"})


# ── serving preconditions (§5.3) ────────────────────────────────────────────


def test_input_completeness_is_not_an_evidence_bar():
    """It asks whether the dates covered are the dates owed, never whether
    enough history has accrued — the confidence sequence is the only thing
    that asks that (§5.0). Three complete dates PASS."""
    assert sa.input_completeness_precondition(
        "arm-1", n_scored_dates=3, n_expected_dates=3
    ).passed


def test_input_completeness_fails_on_a_partial_arm():
    assert not sa.input_completeness_precondition(
        "arm-1", n_scored_dates=3, n_expected_dates=100
    ).passed


def test_an_undefined_precondition_is_a_failure_never_a_pass():
    """§5.1: you cannot gate on a statistic you did not measure, and an
    uncomputed gate reported as a PASS is the defect the rule prevents."""
    assert not sa.input_completeness_precondition(
        "arm-1", n_scored_dates=0, n_expected_dates=0
    ).passed


# ── the two order-path guards (overseer-policy.md §8) ───────────────────────


def test_every_live_order_path_key_this_repo_writes_is_refused():
    """RED before the change: nothing refused these, because no guard existed.

    The keys are read from the module that actually writes them today rather
    than restated, so a new order-path key added there is not silently outside
    the guard.
    """
    from optimizer import executor_optimizer as eo

    assert eo.S3_PARAMS_KEY in sa.ORDER_PATH_KEYS, (
        "the live executor params key this repo writes weekly is not in the "
        "strategy arena's order-path denylist"
    )
    for key in sorted(sa.ORDER_PATH_KEYS):
        with pytest.raises(sa.OrderPathWriteRefused):
            sa.assert_no_order_path_write(key)


def test_the_slots_own_output_keys_are_not_order_path_keys():
    """A guard that refused the slot's own artifact would be a guard nobody
    could ship; assert the boundary sits where it is meant to."""
    for key in (
        sa.ARENA_LATEST_KEY,
        sa.ARENA_DATED_KEY.format(date="2026-08-29"),
        sa.PROPOSAL_KEY,
    ):
        sa.assert_no_order_path_write(key)


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 8, 27, 14, 0, tzinfo=timezone.utc),  # 10:00 ET, Thursday
        datetime(2026, 8, 27, 19, 59, tzinfo=timezone.utc),  # 15:59 ET
        datetime(2026, 8, 27, 13, 30, tzinfo=timezone.utc),  # 09:30 ET, the open
    ],
)
def test_the_cycle_refuses_to_run_during_a_regular_session(moment):
    """"Any remediation against the live trading path during market hours, of
    any kind" is never autonomous. I9320: STOP rather than proceed."""
    with pytest.raises(sa.MarketHoursRefused):
        sa.assert_outside_market_hours(moment)


@pytest.mark.parametrize(
    "moment",
    [
        datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),  # Saturday
        datetime(2026, 8, 27, 21, 0, tzinfo=timezone.utc),  # 17:00 ET, after close
        datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc),  # 08:00 ET, pre-open
    ],
)
def test_the_cycle_runs_outside_a_regular_session(moment):
    sa.assert_outside_market_hours(moment)


def test_a_naive_timestamp_is_refused_rather_than_assumed_utc():
    """A guard whose correctness depends on guessing a timezone is not a guard."""
    with pytest.raises(sa.StrategyArenaError):
        sa.assert_outside_market_hours(datetime(2026, 8, 27, 14, 0))


def test_build_strategy_cycle_is_itself_market_hours_gated():
    """The clock guard sits on the CYCLE, not only on the writers: a verdict
    computed intraday is already an action if anything later consumes it."""
    register = sa.build_arm_register(_history())
    with pytest.raises(sa.MarketHoursRefused):
        sa.build_strategy_cycle(
            as_of="2026-08-27",
            register=register,
            series_by_arm={
                arm: ArmSeries(arm_id=arm, scores={}) for arm in register.all_arms()
            },
            incumbent=None,
            now_utc=datetime(2026, 8, 27, 14, 0, tzinfo=timezone.utc),
        )


# ── slot configuration (§10) ────────────────────────────────────────────────


def test_the_slot_registry_row_and_the_arena_config_agree():
    """§10: one fact, one declaration. Two spellings of the benchmark is the
    drift the registry exists to prevent — asserted at import time too."""
    assert sa.STRATEGY_SLOT.benchmark == sa.STRATEGY_ARENA_CONFIG.benchmark
    assert sa.STRATEGY_SLOT.slot_id == sa.STRATEGY_ARENA_CONFIG.slot


def test_market_relative_grading_is_legitimate_only_because_this_is_slot_s():
    """§4: the population benchmark is for the SELECTION stages. S is the only
    slot whose output IS a market position. Assert we did not simply get away
    with SPY — assert the engine would have refused it for a selection slot,
    which is what makes this declaration meaningful rather than incidental."""
    assert sa.SLOT_KIND not in SELECTION_SLOT_KINDS
    with pytest.raises(ArenaConfigError):
        type(sa.STRATEGY_ARENA_CONFIG)(
            slot="would_be_refused",
            slot_kind=SELECTION_SLOT_KINDS[0],
            benchmark=sa.BENCHMARK,
        )


def test_the_cap_is_a_retirement_criterion_not_an_admission_gate():
    """§6.1: creating an arm is never blocked, and the pool may exceed cap
    while a new recipe is inside its grace window."""
    cfg = sa.STRATEGY_ARENA_CONFIG
    assert cfg.grace_weeks >= 1
    assert 2 <= cfg.min_active_arms <= cfg.cap


def test_diff_clip_is_declared_in_the_units_of_the_per_date_score():
    """§5.0: the sub-Gaussian scale is DERIVED from a declared per-date clip,
    so the interval's validity is checkable from configuration alone. The
    score is a daily log-return difference, so a clip at or above 1.0 would be
    a clip in the wrong units — it would bound nothing."""
    assert 0 < sa.STRATEGY_ARENA_CONFIG.diff_clip < 0.1


# ── no thin-evidence gate survives on this slot (§5.0) ──────────────────────


def test_no_minimum_evidence_gate_is_reintroduced_under_any_name():
    """§5.0 removes thin_evidence gates, minimum-week bars and minimum-cohort
    counts from every slot. Asserted structurally rather than by name, so a
    gate reintroduced under a new spelling still fails.

    ``min_paired_dates`` is deliberately exempt where it is the engine's own
    well-formedness check — this slot never sets it, and this test is what
    stops it being set as an evidence bar later.
    """
    source = inspect.getsource(sa)
    forbidden = ("thin_evidence", "MIN_WEEKS", "min_weeks", "MIN_COHORT", "min_cohort")
    offenders = [token for token in forbidden if token in source]
    assert not offenders, f"minimum-evidence gate reintroduced on the S slot: {offenders}"
    assert sa.STRATEGY_ARENA_CONFIG.min_paired_dates == 1, (
        "min_paired_dates above 1 on this slot would be an evidence bar wearing a "
        "well-formedness check's name (champion-challenger-policy.md §5.0)"
    )


def test_no_hysteresis_margin_or_cooldown_survives():
    """§5.2: the pointer moves freely in both directions. The engine exposes no
    margin or cooldown at all, so the assertion is that this slot did not
    invent one."""
    source = inspect.getsource(sa)
    for token in ("promotion_margin", "cooldown", "hysteresis"):
        assert token not in source, f"S slot reintroduced {token!r} (§5.2)"


# ── the cycle artifact (§11) ────────────────────────────────────────────────


def _series_for(register, dates, value):
    return {
        arm: ArmSeries(arm_id=arm, scores={d: value for d in dates})
        for arm in register.all_arms()
    }


def test_the_cycle_conforms_to_the_arena_cycle_contract():
    """M0: validation happens at the PRODUCER. A consumer discovering the
    violation later has already been handed a document it cannot trust."""
    register = sa.build_arm_register(_history())
    dates = ["2026-08-0{}".format(i) for i in range(3, 8)]
    cycle = sa.build_strategy_cycle(
        as_of="2026-08-29",
        register=register,
        series_by_arm=_series_for(register, dates, 0.001),
        incumbent=register.all_arms()[0],
        now_utc=datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),
    )
    doc = sa.cycle_document(cycle)
    assert doc["slot"] == sa.SLOT_ID
    assert doc["benchmark"] == sa.BENCHMARK
    assert doc["schema_version"] == 1


def test_a_slot_below_the_arm_floor_says_so_and_is_never_green():
    """§9.2 permits a one-rule slot; §7.2 makes it loud. The 2026-08-21 and
    -08-28 `no_promotable_challenger` writes are what an unreported version of
    this state looks like."""
    history = {"2026-06-01": _recipe()}
    register = sa.build_arm_register(history)
    cycle = sa.build_strategy_cycle(
        as_of="2026-08-29",
        register=register,
        series_by_arm=_series_for(register, ["2026-08-03"], 0.001),
        incumbent=register.all_arms()[0],
        now_utc=datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),
    )
    doc = sa.cycle_document(cycle)
    assert doc["slot_status"] == sa.STATUS_SINGLE_ARM
    assert doc["slot_status_reason"]


def test_a_healthy_pool_carries_no_single_arm_status():
    """The status must not be a permanent decoration — it has to clear."""
    register = sa.build_arm_register(_history())
    cycle = sa.build_strategy_cycle(
        as_of="2026-08-29",
        register=register,
        series_by_arm=_series_for(register, ["2026-08-03", "2026-08-04"], 0.001),
        incumbent=register.all_arms()[0],
        now_utc=datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),
    )
    assert "slot_status" not in sa.cycle_document(cycle)


class _RecordingS3:
    def __init__(self):
        self.puts: list[tuple[str, str]] = []

    def put_object(self, *, Bucket, Key, Body, ContentType):  # noqa: N803
        self.puts.append((Bucket, Key))


def test_the_cycle_writes_the_dated_key_before_the_latest_mirror():
    """If the second write fails the immutable record exists and the mirror is
    stale, which a freshness probe catches. The reverse order would leave a
    mirror pointing at a cycle that was never persisted."""
    register = sa.build_arm_register(_history())
    cycle = sa.build_strategy_cycle(
        as_of="2026-08-29",
        register=register,
        series_by_arm=_series_for(register, ["2026-08-03", "2026-08-04"], 0.001),
        incumbent=register.all_arms()[0],
        now_utc=datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),
    )
    s3 = _RecordingS3()
    sa.write_strategy_cycle(sa.cycle_document(cycle), "bkt", "2026-08-29", s3_client=s3)
    assert [k for _, k in s3.puts] == [
        sa.ARENA_DATED_KEY.format(date="2026-08-29"),
        sa.ARENA_LATEST_KEY,
    ]


def test_a_full_run_writes_a_proposal_and_never_an_order_path_key():
    """The load-bearing assertion of this whole file: the slot's entire
    production output is a proposal, and no write reaches the order path."""
    register = sa.build_arm_register(_history())
    dates = ["2026-08-0{}".format(i) for i in range(3, 8)]
    s3 = _RecordingS3()
    result = sa.run_strategy_arena(
        bucket="bkt",
        as_of="2026-08-29",
        register=register,
        series_by_arm=_series_for(register, dates, 0.001),
        incumbent=register.all_arms()[0],
        s3_client=s3,
        now_utc=datetime(2026, 8, 29, 14, 0, tzinfo=timezone.utc),
    )
    written = {k for _, k in s3.puts}
    assert sa.PROPOSAL_KEY in written
    assert not (written & sa.ORDER_PATH_KEYS)
    assert result["proposal_key"] == sa.PROPOSAL_KEY


def test_the_proposal_says_it_is_a_proposal():
    """A record that does not say what it is gets read as what the reader
    expects. §7.5: provenance must be true by construction."""
    source = inspect.getsource(sa.write_champion_proposal)
    assert "PROPOSAL ONLY" in source
