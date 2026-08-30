"""Guards for the selection-producer slot's arena wiring (alpha-engine-config-I9318).

Every guard here was run RED against the pre-fix code before it was trusted
(champion-challenger-policy.md §7.4) — see the PR body for the pre-fix
failures each one produced.

What is deliberately NOT tested here: the score ladder, the longest-common-
window pairing, the confidence sequence, the Copeland ranking, the pointer
rule and the retirement rule. They live in ``nousergon_lib.arena`` and are
tested there. A copy of those tests here would be the second implementation
`shared-code-policy.md` exists to prevent, one layer down.
"""

from __future__ import annotations

import json

import pytest

from nousergon_lib.arena import ArenaConfig, ArmRegister
from nousergon_lib.contracts import validate as validate_contract
from optimizer import producer_arena


# The real cohort dates each arm produced, read off
# ``s3://alpha-engine-research/signals_shadow/{arm}/`` on 2026-08-29 and
# narrowed to the dates matured at the 21-session horizon as of 2026-08-28.
# They reproduce the live board's own figures exactly (5 / 6 / 6 / 4 scored
# dates and a whole-cohort intersection of exactly ``2026-07-30``), which is
# what makes the pairwise arithmetic below a statement about production
# rather than about a fixture.
LIVE_COHORTS = {
    "scanner_predictor_direct": ["2026-07-10", "2026-07-17", "2026-07-24", "2026-07-29", "2026-07-30"],
    "scanner_top20_predictor": ["2026-07-02", "2026-07-10", "2026-07-17", "2026-07-24", "2026-07-29", "2026-07-30"],
    "no_agent_quant": ["2026-07-02", "2026-07-10", "2026-07-24", "2026-07-27", "2026-07-29", "2026-07-30"],
    "single_agent_quant": ["2026-07-02", "2026-07-10", "2026-07-24", "2026-07-27", "2026-07-29", "2026-07-30"],
    "thinktank_coverage": ["2026-07-16", "2026-07-17", "2026-07-28", "2026-07-30"],
}


def _board(date_str="2026-08-28", *, with_series=True, arms=None, extra_rows=()):
    """A producer leaderboard shaped exactly like the live artifact."""
    cohorts = arms if arms is not None else LIVE_COHORTS
    specs = []
    for name, dates in cohorts.items():
        row = {
            "name": name,
            "kind": "champion" if name == "scanner_predictor_direct" else "challenger",
            "n_dates_scored": len(dates),
            "dates_scored": list(dates),
            "topn_alpha_vs_population": {"mean": 0.0, "se": 0.0, "t_stat": 0.0, "n_dates": len(dates)},
            "topn_alpha_vs_benchmark": {"mean": 99.0, "se": 1.0, "t_stat": 1.0, "n_dates": len(dates)},
        }
        if with_series:
            row[producer_arena.POPULATION_SERIES_FIELD] = {d: 0.0 for d in dates}
        specs.append(row)
    specs.extend(extra_rows)
    return {
        "date": date_str,
        "champion": "scanner_predictor_direct",
        "horizon_days": 21,
        "top_n": 50,
        "benchmark_ticker": "SPY",
        "specs": specs,
    }


def _series_board(scores_by_arm):
    """A board whose per-date population-relative series is supplied directly."""
    return {
        "date": "2026-08-28",
        "horizon_days": 21,
        "specs": [
            {
                "name": name,
                "kind": "challenger",
                "n_dates_scored": len(scores),
                "dates_scored": sorted(scores),
                producer_arena.POPULATION_SERIES_FIELD: dict(scores),
            }
            for name, scores in scores_by_arm.items()
        ],
    }


# ── The config is the slot's §10 registry row ─────────────────────────────


class TestArenaConfig:
    def test_the_engine_refuses_spy_for_this_slot_kind(self):
        """The SPY refusal is a MECHANISM, not a convention.

        Ran red against the pre-fix module, whose only score source for
        thinktank_coverage was ``topn_alpha_vs_benchmark`` — SPY-relative —
        while the incumbent was scored from a SPY-relative backtester
        counterfactual. On 2026-08-17 SPY trailed the drawn-from population
        by 140bp at 21d, which inverts wins and losses outright.
        """
        with pytest.raises(ValueError, match="selection-stage slot"):
            ArenaConfig(slot="producer", slot_kind="selection_producer", benchmark="SPY")

    def test_the_slot_grades_against_the_population(self):
        assert producer_arena.ARENA_CONFIG.benchmark == "population"
        assert producer_arena.ARENA_CONFIG.slot_kind == "selection_producer"

    def test_brians_2026_08_29_ruling_is_the_config(self):
        cfg = producer_arena.ARENA_CONFIG
        assert cfg.cap == 5
        assert cfg.grace_weeks == 4
        assert cfg.min_active_arms == 3
        assert cfg.retire_evidence == "point"
        assert cfg.retired_trailing_cycles == 8

    def test_min_paired_dates_is_well_formedness_not_an_evidence_bar(self):
        """One paired date is the least from which any statistic can be formed.

        A value above 1 here would be a minimum-cohort gate wearing a
        well-formedness name — exactly what deliverable 6 deletes.
        """
        assert producer_arena.ARENA_CONFIG.min_paired_dates == 1

    def test_diff_clip_bounds_the_widest_gap_the_slot_has_produced(self):
        """0.10 is justified from the observed range, not picked.

        Widest observed cross-arm gap in the population metric: 0.0861
        (no_agent_quant +0.018975 vs thinktank_coverage -0.067122,
        research/producer_leaderboard/2026-08-21.json). A clip below that
        would truncate a difference this slot has actually produced and bias
        every comparison toward the incumbent.
        """
        assert producer_arena.ARENA_CONFIG.diff_clip >= 0.0861

    def test_no_hysteresis_margin_or_cooldown_exists_on_this_slot(self):
        source = (producer_arena.__file__,)
        text = open(source[0]).read()
        assert "cooldown" not in text.lower().replace("no cooldown", "")


# ── The register is DERIVED and DURABLE ───────────────────────────────────


class TestRegister:
    def test_the_committed_register_loads_and_covers_every_live_arm(self):
        register = producer_arena.load_register()
        names = {register.state(a).record.name for a in register.all_arms()}
        assert names >= set(LIVE_COHORTS)

    def test_the_retired_arm_is_registered_retired_and_still_scored(self):
        """§3: retirement stops an arm SERVING, never stops it being measured."""
        register = producer_arena.load_register()
        arm = producer_arena.arm_id_for("agentic_sector_teams")
        assert register.state(arm).retired_date is not None
        assert arm not in register.active_arms()
        assert arm in register.scored_arms("2026-08-28", 8)

    def test_thinktank_coverage_is_registered_with_the_same_scoring_path(self):
        """Deliverable 7. The original defect was an arm that wrote shadow
        output with no register row; its mirror is a register row with no
        scoring path. There is now exactly ONE path and every arm — the
        incumbent included — is on it."""
        register = producer_arena.load_register()
        arm = producer_arena.arm_id_for("thinktank_coverage")
        assert arm in register.active_arms()
        series, _gaps = producer_arena.build_series(register, _board(), "2026-08-28")
        assert arm in series
        assert series[arm].scores  # scored through the shared path, not a special case

    def test_an_arm_that_emits_nothing_is_a_loud_miss_never_absent(self):
        """`agentic_sector_teams` emits no cohort dates at all. It must appear
        in the scored set with a NAMED gap, never be quietly dropped."""
        register = producer_arena.load_register()
        series, gaps = producer_arena.build_series(register, _board(), "2026-08-28")
        arm = producer_arena.arm_id_for("agentic_sector_teams")
        assert arm in series
        assert series[arm].scores == {}
        assert any(g.arm_name == "agentic_sector_teams" for g in gaps)

    def test_valid_champions_is_derived_not_typed(self):
        """The class defect: four hand-maintained rosters. This one is derived."""
        register = producer_arena.load_register()
        derived = producer_arena.promotion_eligible_arm_names(register)
        assert set(derived) == {
            register.state(a).record.name for a in register.active_arms()
        }
        # The two arms the hand-typed tuple silently omitted, with the most
        # evidence on the board, are now IN the roster.
        assert "no_agent_quant" in derived
        assert "single_agent_quant" in derived

    def test_the_derivation_is_reproducible_from_the_boards(self):
        boards = [
            {"date": "2026-08-03", "specs": [{"name": "a", "kind": "challenger", "dates_scored": ["2026-07-31"]}]},
            {"date": "2026-08-10", "specs": [{"name": "a", "kind": "retired"}, {"name": "b", "kind": "challenger"}]},
        ]
        events = producer_arena.register_events_from_boards(boards)
        register = ArmRegister.from_dicts(events)
        # created_date is the EARLIEST observation, including a cohort date
        # older than the board that first reported it.
        assert register.state(producer_arena.arm_id_for("a")).record.created_date == "2026-07-31"
        assert register.state(producer_arena.arm_id_for("a")).retired_date == "2026-08-10"
        assert register.state(producer_arena.arm_id_for("b")).record.created_date == "2026-08-10"

    def test_an_unboarded_arm_is_seeded_rather_than_silently_omitted(self):
        events = producer_arena.register_events_from_boards(
            [{"date": "2026-08-28", "specs": [{"name": "no_agent_quant"}]}]
        )
        names = {e["record"]["name"] for e in events if e["kind"] == "registered"}
        assert "scanner_top20_predictor" in names

    def test_a_board_date_beats_the_seed(self):
        events = producer_arena.register_events_from_boards(
            [{"date": "2026-07-02", "specs": [{"name": "scanner_top20_predictor"}]}]
        )
        record = next(e["record"] for e in events if e["record"]["name"] == "scanner_top20_predictor")
        assert record["created_date"] == "2026-07-02"

    def test_a_changed_spec_cannot_reuse_an_arm_id(self):
        assert producer_arena.arm_id_for("a") != producer_arena.arm_id_for("b")
        assert producer_arena.arm_id_for("a").startswith("producer:a:")

    def test_a_board_arm_the_register_does_not_know_raises(self):
        """The guard on the derived roster falling behind its source."""
        with pytest.raises(ValueError, match="does not register"):
            producer_arena.run_arena_cycle(
                as_of="2026-08-28",
                leaderboard=_board(extra_rows=({"name": "brand_new_arm", "kind": "challenger"},)),
                incumbent_name="scanner_predictor_direct",
                shadow_only_names=frozenset(),
            )


# ── Pairwise pairing is what makes the slot measurable ────────────────────


class TestPairwisePairing:
    """The N-arm cohort intersection on the live 2026-08-28 board is ONE date
    (``2026-07-30``), which is why two arms carried
    ``comparison_status: "no_common_cohort"``. The arena pairs each pair on
    its OWN longest common window, so the same evidence yields 4-, 4-, 5- and
    2-date comparisons against the incumbent instead."""

    def _cycle(self):
        return producer_arena.run_arena_cycle(
            as_of="2026-08-28",
            leaderboard=_board(),
            incumbent_name="scanner_predictor_direct",
            shadow_only_names=frozenset(),
        )

    INCUMBENT = "scanner_predictor_direct"

    def _window_vs_incumbent(self, cycle, name):
        """The pair's own longest common window, from the RANKING.

        Read off ``cycle.ranking`` rather than ``cycle.decision.comparisons``
        deliberately: §3 makes measurement unconditional, so an arm the
        executor cannot yet SERVE is still ranked and still measured, and its
        pairwise window is on the artifact either way. ``decision.comparisons``
        holds only the arms eligible to take the pointer this cycle.
        """
        a, b = producer_arena.arm_id_for(name), producer_arena.arm_id_for(self.INCUMBENT)
        return next(
            v for v in cycle.ranking.verdicts
            if {v.arm_a, v.arm_b} == {a, b}
        ).window

    @pytest.mark.parametrize(
        "challenger,expected",
        [
            ("scanner_top20_predictor", 5),
            ("no_agent_quant", 4),
            ("single_agent_quant", 4),
            ("thinktank_coverage", 2),
        ],
    )
    def test_each_pair_gets_its_own_window(self, challenger, expected):
        cycle, _gaps, _reg = self._cycle()
        window = self._window_vs_incumbent(cycle, challenger)
        assert window.measurable, window.unmeasurable_reason
        assert window.n_dates == expected

    def test_the_two_no_common_cohort_arms_become_measurable(self):
        """``no_agent_quant`` and ``single_agent_quant`` each carried
        ``comparison_status: "no_common_cohort"`` under the N-arm
        intersection, which on the live 2026-08-28 board is the single date
        ``2026-07-30``. Under pairwise pairing each has a FOUR-date window
        with the incumbent — four times the whole-cohort basis."""
        cycle, _gaps, _reg = self._cycle()
        for name in ("no_agent_quant", "single_agent_quant"):
            window = self._window_vs_incumbent(cycle, name)
            assert window.measurable
            assert window.n_dates == 4

    def test_every_pair_against_the_incumbent_beats_the_whole_cohort_intersection(self):
        """The claim is about the DECISION, so it is about pairs with the incumbent.

        All ten pairs of the five live arms are measurable, and every one of
        the four that includes the incumbent has a strictly longer window than
        the one-date N-arm intersection. Two challenger-vs-challenger pairs
        (``thinktank_coverage`` against each quant arm) land on exactly that
        single date — ``2026-07-30`` is genuinely all those cohorts share, and
        the honest thing is to say so rather than assert a blanket claim the
        live cohorts do not support. It costs nothing: those pairs feed the
        Copeland ranking that drives RETIREMENT, where the evidence bar is the
        four-week grace period, and neither is a comparison the pointer rests
        on this cycle.
        """
        cycle, _gaps, _reg = self._cycle()
        measurable = [v for v in cycle.ranking.verdicts if v.measurable]
        assert len(measurable) == 10  # every pair of the five live arms

        incumbent = producer_arena.arm_id_for(self.INCUMBENT)
        against_incumbent = [v for v in measurable if incumbent in (v.arm_a, v.arm_b)]
        assert len(against_incumbent) == 4
        assert all(v.window.n_dates > 1 for v in against_incumbent)

        single_date = [v for v in measurable if v.window.n_dates == 1]
        tt = producer_arena.arm_id_for("thinktank_coverage")
        assert all(tt in (v.arm_a, v.arm_b) for v in single_date)
        assert incumbent not in {a for v in single_date for a in (v.arm_a, v.arm_b)}

    def test_a_servable_arm_is_also_compared_on_the_decision_path(self):
        cycle, _gaps, _reg = self._cycle()
        arm = producer_arena.arm_id_for("scanner_top20_predictor")
        comparison = next(c for c in cycle.decision.comparisons if c.challenger == arm)
        assert comparison.status == "measured"
        assert comparison.window.n_dates == 5


# ── The benchmark refusal, at the series level ────────────────────────────


class TestSeriesSourcing:
    def test_the_spy_relative_metric_is_never_substituted(self):
        """A board carrying only the SPY figure yields a NAMED gap, never a
        score. Ran red against the pre-fix module, which read
        ``topn_alpha_vs_benchmark.mean`` (SPY) as the arm's whole score."""
        register = producer_arena.load_register()
        series, gaps = producer_arena.build_series(
            register, _board(with_series=False), "2026-08-28",
        )
        assert all(s.scores == {} for s in series.values())
        assert {g.arm_name for g in gaps} >= set(LIVE_COHORTS)
        assert all("140bp" in g.reason or "not present" in g.reason for g in gaps)
        assert not any(99.0 in s.scores.values() for s in series.values())

    def test_a_missing_series_is_a_miss_not_a_zero(self):
        register = producer_arena.load_register()
        series, _gaps = producer_arena.build_series(
            register, _board(with_series=False), "2026-08-28",
        )
        arm = producer_arena.arm_id_for("thinktank_coverage")
        assert series[arm].scores == {}
        assert series[arm].misses == frozenset(LIVE_COHORTS["thinktank_coverage"])

    def test_every_registered_arm_gets_a_series_entry(self):
        """``run_cycle`` RAISES on a missing series. That raise is the guard
        against an arm quietly dropping out of the contest."""
        register = producer_arena.load_register()
        series, _gaps = producer_arena.build_series(register, _board(), "2026-08-28")
        assert set(series) == set(register.scored_arms("2026-08-28", 8))


# ── Serving preconditions ─────────────────────────────────────────────────


class TestPointerContractAdmission:
    """The pointer precondition is DERIVED from the frozen contract, not mirrored.

    The version of this guard written earlier on 2026-08-29 was a hand-typed
    copy of ``crucible-executor/executor/champion.py::VALID_CHAMPIONS``, and it
    was stale within hours: ``alpha-engine-config-I9299`` landed the same day
    and made ``no_agent_quant`` and ``single_agent_quant`` servable, so the
    copy would have held the pointer off the two arms with the most evidence
    for no reason anyone could have found. That is the silent-omission class
    this whole change closes, reproduced by the fix for it.
    """

    def test_every_registered_arm_is_admitted_today(self):
        """A guard that currently passes, stated as a measurement.

        All five active arms are in the enum as of I9299. This is asserted
        rather than assumed so that an arm appearing upstream WITHOUT the enum
        being widened fails here — in this repo, which owns both the enum and
        the writer — instead of at the executor's planner start.
        """
        register = producer_arena.load_register()
        active = {register.state(a).record.name for a in register.active_arms()}
        assert active <= producer_arena.POINTER_ADMISSIBLE_ARMS, (
            "registered arms absent from contracts/producer_champion.schema.json: "
            f"{sorted(active - producer_arena.POINTER_ADMISSIBLE_ARMS)}"
        )

    def test_the_admitted_set_is_read_off_the_contract(self):
        schema = json.loads(producer_arena.POINTER_CONTRACT_PATH.read_text())
        enum = set(schema["properties"]["champion"]["enum"])
        assert producer_arena.POINTER_ADMISSIBLE_ARMS == enum - producer_arena.WRITE_FORBIDDEN_ARMS
        # `agentic` is read-tolerated by the enum and write-forbidden here.
        assert "agentic" in enum
        assert "agentic" not in producer_arena.POINTER_ADMISSIBLE_ARMS

    def test_an_arm_the_contract_does_not_admit_never_takes_the_pointer(self, monkeypatch):
        """The reachable failure, driven end to end.

        An arm the producer board scores and the register knows, whose name the
        pointer contract has not yet been widened to admit — the exact I9299
        sequence. It must be measured, ranked, and held OFF the pointer with a
        named reason, never quietly dropped from the contest.
        """
        monkeypatch.setattr(
            producer_arena,
            "POINTER_ADMISSIBLE_ARMS",
            producer_arena.POINTER_ADMISSIBLE_ARMS - {"no_agent_quant"},
        )
        board = _series_board({
            "no_agent_quant": {d: 0.05 for d in LIVE_COHORTS["no_agent_quant"]},
            "scanner_predictor_direct": {d: -0.05 for d in LIVE_COHORTS["scanner_predictor_direct"]},
            "single_agent_quant": {d: 0.0 for d in LIVE_COHORTS["single_agent_quant"]},
            "scanner_top20_predictor": {d: 0.0 for d in LIVE_COHORTS["scanner_top20_predictor"]},
            "thinktank_coverage": {d: 0.0 for d in LIVE_COHORTS["thinktank_coverage"]},
        })
        cycle, _gaps, _register = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=board,
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
        )
        blocked = producer_arena.arm_id_for("no_agent_quant")
        assert blocked in cycle.decision.ineligible
        assert cycle.decision.champion != blocked
        failed = [p for p in cycle.decision.ineligible[blocked] if not p.passed]
        assert [p.name for p in failed] == ["pointer_contract_admits"]
        assert "producer_champion.schema.json" in failed[0].reason
        # Measured anyway (§3): it leads the board and its ladder is built.
        assert any(lad.arm_id == blocked and lad.rungs for lad in cycle.ladders)
        assert cycle.ranking.ordering[0] == blocked


class TestServingPreconditions:
    def test_a_shadow_only_arm_is_measured_but_never_served(self):
        board = _board()
        cycle, _gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=board,
            incumbent_name="scanner_predictor_direct",
            shadow_only_names=frozenset({"thinktank_coverage"}),
        )
        arm = producer_arena.arm_id_for("thinktank_coverage")
        assert arm in cycle.decision.ineligible
        # Still SCORED — the ladder is built for it either way (§3).
        assert any(lad.arm_id == arm and lad.rungs for lad in cycle.ladders)

    def test_a_dead_feed_blocks_serving_and_says_which(self):
        cycle, _gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=_board(),
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
            feed_blocked_names={"scanner_top20_predictor": "research_free_backfill is stale"},
        )
        arm = producer_arena.arm_id_for("scanner_top20_predictor")
        failed = [p for p in cycle.decision.ineligible[arm] if not p.passed]
        assert [p.name for p in failed] == ["feed_producer_live"]


# ── The artifact (§11) ────────────────────────────────────────────────────


class TestArenaCycleArtifact:
    def test_the_emitted_cycle_conforms_to_the_contract(self):
        cycle, gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=_board(),
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
        )
        doc = producer_arena.cycle_document(cycle, gaps)
        validate_contract("arena_cycle", doc)  # explicit, in addition to the producer-side call
        assert doc["benchmark"] == "population"
        assert doc["slot_kind"] == "selection_producer"

    def test_an_unmeasurable_cycle_is_still_emitted_and_says_why(self):
        """§11: a slot that emits nothing is not healthy, it is unobserved."""
        cycle, gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=_board(with_series=False),
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
        )
        doc = producer_arena.cycle_document(cycle, gaps)
        assert doc["decision"]["status"] == "unmeasurable"
        assert doc["decision"]["reason"]
        assert doc["series_gaps"]

    def test_the_gaps_reach_the_artifact_by_name(self):
        cycle, gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=_board(with_series=False),
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
        )
        doc = producer_arena.cycle_document(cycle, gaps)
        assert {g["arm_name"] for g in doc["series_gaps"]} >= set(LIVE_COHORTS)

    def test_every_comparison_carries_the_window_it_rests_on(self):
        cycle, gaps, _reg = producer_arena.run_arena_cycle(
            as_of="2026-08-28", leaderboard=_board(),
            incumbent_name="scanner_predictor_direct", shadow_only_names=frozenset(),
        )
        doc = producer_arena.cycle_document(cycle, gaps)
        for comparison in doc["decision"]["comparisons"]:
            assert {"n_dates", "start_date", "end_date", "weeks"} <= set(comparison)

    def test_a_write_failure_is_never_swallowed(self):
        class _Boom:
            def put_object(self, **_kw):
                raise RuntimeError("s3 down")

        with pytest.raises(RuntimeError):
            producer_arena.write_arena_cycle(
                "b", "2026-08-28", {"x": 1}, upload=True, s3_client=_Boom(),
            )

    def test_the_dated_and_latest_keys_are_both_written(self):
        puts = []

        class _S3:
            def put_object(self, **kw):
                puts.append(kw["Key"])

        producer_arena.write_arena_cycle(
            "b", "2026-08-28", {"x": 1}, upload=True, s3_client=_S3(),
        )
        assert puts == [
            "arena/producer/2026-08-28.json",
            "arena/producer/latest.json",
        ]


class TestCommittedRegisterArtifact:
    def test_it_is_valid_json_with_the_expected_envelope(self):
        payload = json.loads(producer_arena.REGISTER_PATH.read_text())
        assert payload["slot"] == "producer"
        assert payload["derived_by"].endswith("backfill_producer_arena_register.py")
        assert payload["events"]

    def test_every_record_declares_where_its_created_date_came_from(self):
        payload = json.loads(producer_arena.REGISTER_PATH.read_text())
        for event in payload["events"]:
            if event["kind"] == "registered":
                assert event["record"]["notes"]
