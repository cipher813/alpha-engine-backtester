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
