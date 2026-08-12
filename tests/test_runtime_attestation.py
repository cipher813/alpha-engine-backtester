"""Tests for the RUNTIME numeric attestation (`analysis/attestation.py`).

Legs (a)–(c) of the L4593 correctness battery (null calibration, the independent
oracle, the golden benchmarks) all run in **CI**, against whatever wheels the CI
runner resolved. The weekly Saturday backtest runs on a **spot instance** that
`pip install -r requirements.txt`s fresh: `vectorbt~=0.28.5` is a compatible-release
specifier, so the deployed engine can be a different build from the one CI proved
correct, and `numpy`/`pandas`/`numba` resolve independently there too. Nothing in
the pipeline asserts the deployed engine still computes the right number.

Per `sf-pipeline-policy.md` §2.3a a correctness verdict must (1) be produced where
the number is produced, (2) reach every consumer whose output depends on it, and
(3) propagate as UNKNOWN — never as a pass — when it is missing. This module tests
the producer half: the verdict is computed in-process on the deployed engine, the
checks have teeth, and every failure mode lands on a non-PASS verdict rather than
an exception that strands the artifact.
"""
from __future__ import annotations

import json

import pytest

from analysis import attestation


class TestVerdictOnAHealthyEngine:
    def test_passes_and_declares_its_schema(self):
        result = attestation.run_attestation(run_date="2026-08-15")
        assert result["schema"] == attestation.SCHEMA
        assert result["run_date"] == "2026-08-15"
        assert result["verdict"] == attestation.PASS, result.get("checks")
        assert result["status"] == "ok"

    def test_every_check_reports_its_own_outcome(self):
        result = attestation.run_attestation()
        names = {c["name"] for c in result["checks"]}
        # The battery must cover each accounting axis a systematic bug rides on.
        assert {
            "pnl_no_fees",
            "fee_charged_both_sides",
            "drawdown_peak_to_trough",
            "alpha_over_active_window",
            "oracle_nav_agreement",
        } <= names
        for c in result["checks"]:
            assert c["passed"] is True, c
            assert c["expected"] is not None
            assert c["observed"] is not None

    def test_records_the_engine_it_attested(self):
        # A verdict that does not name the engine build cannot be used to
        # explain a later divergence.
        engine = attestation.run_attestation()["engine"]
        assert engine["vectorbt"] and engine["numpy"] and engine["pandas"]
        assert engine["python"]

    def test_serializes_to_json(self):
        json.dumps(attestation.run_attestation())

    def test_cheap_enough_to_run_every_cycle(self):
        result = attestation.run_attestation()
        assert result["wall_clock_seconds"] < 60.0


class TestTeeth:
    """A check that cannot fail is decorative. Perturb the expectation and the
    verdict must flip to FAIL — proving the comparison is load-bearing."""

    def test_a_wrong_expectation_fails_the_verdict(self, monkeypatch):
        original = attestation._SCENARIOS

        def _poisoned():
            scenarios = list(original())
            first = scenarios[0]
            scenarios[0] = first._replace(expected=first.expected + 1.0)
            return scenarios

        monkeypatch.setattr(attestation, "_SCENARIOS", _poisoned)
        result = attestation.run_attestation()
        assert result["verdict"] == attestation.FAIL
        assert any(c["passed"] is False for c in result["checks"])

    def test_tolerances_are_tight_enough_to_catch_a_basis_point(self, monkeypatch):
        """A 1bp systematic bias on a check must not slip through its tolerance."""
        original = attestation._SCENARIOS

        def _poisoned():
            scenarios = list(original())
            out = []
            for s in scenarios:
                if s.expected is not None and s.expected != 0.0:
                    out.append(s._replace(expected=s.expected * 1.0001))
                else:
                    out.append(s)
            return out

        monkeypatch.setattr(attestation, "_SCENARIOS", _poisoned)
        assert attestation.run_attestation()["verdict"] == attestation.FAIL


class TestUnknownNeverReadsAsPass:
    def test_engine_import_failure_yields_unknown_not_an_exception(self, monkeypatch):
        def _boom():
            raise RuntimeError("vectorbt exploded on import")

        monkeypatch.setattr(attestation, "_SCENARIOS", _boom)
        result = attestation.run_attestation()
        assert result["verdict"] == attestation.UNKNOWN
        assert result["status"] == "error"
        assert result["error_class"] == "RuntimeError"
        assert "vectorbt exploded" in result["error_msg"]

    def test_a_check_that_cannot_run_is_unknown_not_fail(self, monkeypatch):
        """Absence of evidence is not evidence of a wrong number. An engine that
        cannot be imported must not read as 'the arithmetic regressed' — both
        withhold the guarantee, only one accuses the engine."""
        original = attestation._SCENARIOS

        def _boom():
            raise ModuleNotFoundError("no module named 'vectorbt'")

        def _exploding_scenarios():
            scenarios = list(original())
            return [scenarios[0]._replace(compute=_boom)] + scenarios[1:]

        monkeypatch.setattr(attestation, "_SCENARIOS", _exploding_scenarios)
        result = attestation.run_attestation()
        assert result["verdict"] == attestation.UNKNOWN
        assert result["n_errored"] == 1
        assert result["n_failed"] == 0
        assert result["checks"][0]["errored"] is True

    def test_a_disagreement_outranks_an_error(self, monkeypatch):
        original = attestation._SCENARIOS

        def _boom():
            raise ModuleNotFoundError("gone")

        def _mixed():
            scenarios = list(original())
            scenarios[0] = scenarios[0]._replace(compute=_boom)
            scenarios[1] = scenarios[1]._replace(expected=scenarios[1].expected + 1.0)
            return scenarios

        monkeypatch.setattr(attestation, "_SCENARIOS", _mixed)
        result = attestation.run_attestation()
        assert result["verdict"] == attestation.FAIL

    def test_unknown_is_not_pass(self):
        assert attestation.UNKNOWN != attestation.PASS
        assert attestation.verdict_is_pass(attestation.UNKNOWN) is False
        assert attestation.verdict_is_pass(attestation.FAIL) is False
        assert attestation.verdict_is_pass(attestation.PASS) is True
        assert attestation.verdict_is_pass(None) is False
        assert attestation.verdict_is_pass("ok") is False


class TestVersionedSchemaContract:
    """M0 contract discipline: every new cross-repo artifact gets a versioned
    schema + producer/consumer contract test AT BIRTH. The consumer half lives in
    `crucible-evaluator tests/test_attestation.py`."""

    def _schema(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        return json.loads((root / "contracts" / "backtest_attestation.schema.json").read_text())

    def _validate(self, doc):
        import jsonschema

        jsonschema.validate(doc, self._schema())

    def test_a_passing_verdict_validates(self):
        self._validate(attestation.run_attestation(run_date="2026-08-15"))

    def test_an_unknown_verdict_validates(self, monkeypatch):
        def _boom():
            raise RuntimeError("engine gone")

        monkeypatch.setattr(attestation, "_SCENARIOS", _boom)
        self._validate(attestation.run_attestation(run_date="2026-08-15"))

    def test_the_schema_constant_matches_the_module(self):
        assert self._schema()["properties"]["schema"]["const"] == attestation.SCHEMA

    def test_the_verdict_vocabulary_is_closed(self):
        """A verdict enum that admits new values is not a verdict — the consumer
        maps anything outside this set to UNKNOWN."""
        assert set(self._schema()["properties"]["verdict"]["enum"]) == {
            attestation.PASS, attestation.FAIL, attestation.UNKNOWN,
        }


class TestArtifactContract:
    def test_reporter_always_emits_the_attestation_artifact(self):
        """§2.3a rule 2: absence must mean 'the producer never ran', so the
        artifact is on the ALWAYS-EMIT list, not the OK-ONLY one."""
        import inspect

        import reporter

        src = inspect.getsource(reporter.save)
        always, _, ok_only = src.partition("# OK-ONLY contract")
        assert '("attestation.json", attestation)' in always
        assert "attestation.json" not in ok_only

    def test_reporter_computes_the_attestation_itself(self):
        """It must not be an optional caller-supplied kwarg — a verdict a caller
        can forget to pass is a verdict that silently becomes UNKNOWN forever."""
        import inspect

        import reporter

        assert "attestation" not in inspect.signature(reporter.save).parameters
        assert "run_attestation" in inspect.getsource(reporter.save)


class TestClassificationCountLayer:
    """config-I6975 — the counting layer, one level above the arithmetic.

    Everything else in this module attests numbers. These attest *what is being
    counted*: the tp/fp/fn/tn in ``e2e_lift.json`` that crucible-evaluator's
    research tile turns into precision, edge-over-base-rate and a Wilson CI.
    A mislabelled selection or a bad outcome join produces internally consistent
    and entirely wrong precision, and no arithmetic check can see it.
    """

    def test_the_frozen_pair_produces_the_hand_derived_confusion_matrix(self):
        counts = attestation._observed_counts()
        assert {k: counts[k] for k in ("tp", "fp", "fn", "tn")} == {
            "tp": 2, "fp": 2, "fn": 2, "tn": 3,
        }
        # The denominator is the join+resolution result, not the fixture size:
        # 13 universe rows in, 9 counted.
        assert counts["n"] == 9
        assert len(attestation._FROZEN_UNIVERSE) == 13

    def test_every_confusion_cell_is_exercised(self):
        """A fixture that produces no fn (or no tn) certifies nothing about the
        cell it never fills — the config-I6975 gotcha, made a test."""
        counts = attestation._observed_counts()
        assert all(counts[cell] > 0 for cell in ("tp", "fp", "fn", "tn"))

    def test_the_checks_are_registered_in_the_live_battery(self):
        """§2.3a propagation: registered in ``_SCENARIOS`` means the outcome
        rides the existing verdict → artifact → tile → Director digest path
        with no new wiring."""
        names = {s.name for s in attestation._SCENARIOS()}
        assert {f"classification_counts_{c}" for c in ("tp", "fp", "fn", "tn")} <= names

    def test_counts_are_compared_exactly(self):
        """Integer counts admit no tolerance. A band would let an off-by-one on
        a large cohort pass."""
        for s in attestation._SCENARIOS():
            if s.name.startswith("classification_counts_"):
                assert s.rtol == 0.0 and s.atol == 0.0

    def test_the_21d_block_is_not_the_5d_block(self):
        """The fixture inverts ``beat_spy_5d`` against ``beat_spy_21d`` on every
        resolved row, so a producer that graded the canonical block off the
        diagnostic column lands on a different matrix and FAILs."""
        resolved = [r for r in attestation._FROZEN_UNIVERSE if r[3] is not None]
        assert resolved
        assert all(r[4] == 1 - r[3] for r in resolved)

    def test_horizon_policy_is_asserted_not_assumed(self, monkeypatch):
        """A policy whose primary horizon moved must ERROR (→ UNKNOWN), never
        silently attest a stale column."""
        import nousergon_lib.quant.horizons as horizons

        class _Moved:
            primary_horizon = 63

            @staticmethod
            def outcome_columns(_h):
                class _C:
                    beat_spy = "beat_spy_63d"
                return _C()

        monkeypatch.setattr(horizons, "DEFAULT_POLICY", _Moved)
        with pytest.raises(ValueError, match="beat_spy_63d"):
            attestation._assert_frozen_grid_resolves()


class TestClassificationCountTeeth:
    """Mirror of ``TestTeeth`` for the count layer: perturb the expectation and
    the verdict must flip to FAIL."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        attestation._OBSERVED_COUNTS_CACHE = None
        yield
        attestation._OBSERVED_COUNTS_CACHE = None

    @pytest.mark.parametrize("cell", ["tp", "fp", "fn", "tn"])
    def test_a_perturbed_count_expectation_flips_the_verdict(self, monkeypatch, cell):
        poisoned = dict(attestation._EXPECTED_COUNTS)
        poisoned[cell] += 1
        monkeypatch.setattr(attestation, "_EXPECTED_COUNTS", poisoned)

        result = attestation.run_attestation(run_date="2026-08-12")
        assert result["verdict"] == attestation.FAIL, (
            f"a wrong expectation for {cell} did not fail the verdict — the "
            "comparison is decorative"
        )
        bad = [c for c in result["checks"]
               if c["name"] == f"classification_counts_{cell}"]
        assert bad and bad[0]["passed"] is False
        assert bad[0]["observed"] == attestation._EXPECTED_COUNTS[cell] - 1

    def test_an_off_by_one_count_is_not_absorbed_by_tolerance(self, monkeypatch):
        monkeypatch.setattr(
            attestation, "_EXPECTED_COUNTS",
            {k: v + 1 for k, v in attestation._EXPECTED_COUNTS.items()},
        )
        result = attestation.run_attestation(run_date="2026-08-12")
        assert result["n_failed"] >= 4

    def test_a_producer_emitting_no_classification_block_is_unknown_not_pass(
        self, monkeypatch,
    ):
        """§2.3a rule 2 at the count layer: a path that produced no counts is
        absence of evidence — it must ERROR into UNKNOWN, never quietly pass."""
        import analysis.end_to_end as e2e

        monkeypatch.setattr(
            e2e, "_scanner_lift",
            lambda *a, **k: {"status": "skipped", "reason": "scanner_evaluations empty"},
        )
        with pytest.raises(ValueError, match="no classification_21d block"):
            attestation._observed_counts()


class TestFrozenFixtureExclusions:
    """Each excluded row is a distinct miscounting bug. Removing the exclusion
    must change the matrix — otherwise the row is decoration."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        attestation._OBSERVED_COUNTS_CACHE = None
        yield
        attestation._OBSERVED_COUNTS_CACHE = None

    def _counts_with_universe(self, monkeypatch, rows):
        monkeypatch.setattr(attestation, "_FROZEN_UNIVERSE", tuple(rows))
        return attestation._observed_counts()

    def test_an_unresolved_outcome_counted_as_a_miss_would_inflate_fp(self, monkeypatch):
        rows = [
            (t, d, r, (0 if b is None else b), (1 if s is None else s))
            for (t, d, r, b, s) in attestation._FROZEN_UNIVERSE
        ]
        counts = self._counts_with_universe(monkeypatch, rows)
        assert counts["fp"] == 3, "the unresolved row is not actually excluded today"

    def test_admitting_a_row_without_return_5d_would_inflate_tp(self, monkeypatch):
        rows = [
            (t, d, (0.01 if r is None else r), b, s)
            for (t, d, r, b, s) in attestation._FROZEN_UNIVERSE
        ]
        counts = self._counts_with_universe(monkeypatch, rows)
        assert counts["tp"] == 3, "the return_5d admission rule is not load-bearing"

    def test_a_ticker_only_join_would_inflate_tp(self, monkeypatch):
        """NNN is scanner-evaluated on a different date than its universe row.
        Only an (ticker, eval_date) pair join keeps it out."""
        nnn_u = [r for r in attestation._FROZEN_UNIVERSE if r[0] == "NNN"]
        nnn_s = [r for r in attestation._FROZEN_SCANNER if r[0] == "NNN"]
        assert nnn_u and nnn_s
        assert nnn_u[0][1] != nnn_s[0][1], "the date-mismatch trap no longer mismatches"

        # Realign NNN's scanner row onto its universe date: the join now pairs
        # them and NNN (a winner the scanner passed) becomes a third TP. That
        # delta is what proves the pair join — not the ticker alone — is what
        # keeps the real fixture at tp=2.
        realigned = tuple(
            (t, nnn_u[0][1], p) if t == "NNN" else (t, d, p)
            for (t, d, p) in attestation._FROZEN_SCANNER
        )
        monkeypatch.setattr(attestation, "_FROZEN_SCANNER", realigned)
        assert attestation._observed_counts()["tp"] == 3

    def test_a_scanner_only_ticker_never_enters_the_matrix(self, monkeypatch):
        """LLL has a scanner evaluation and no universe row. An outer join would
        admit it with a null outcome and inflate a cell."""
        assert any(r[0] == "LLL" for r in attestation._FROZEN_SCANNER)
        assert not any(r[0] == "LLL" for r in attestation._FROZEN_UNIVERSE)
        assert sum(attestation._observed_counts()[c]
                   for c in ("tp", "fp", "fn", "tn")) == 9
