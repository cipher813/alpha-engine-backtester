"""The Evaluator stage's §2.3a correctness verdict, and its fail-closed gate.

Three things are proven here, in order of how badly their absence would hurt:

1. **Absence is never a pass.** Every path by which the upstream engine verdict
   can fail to arrive — no object, unparseable body, a body stamped with another
   cycle's run_date, a verdict string that is not the literal "PASS" — resolves
   to UNKNOWN, and a non-PASS combined verdict forces ``--freeze`` so nothing is
   promoted. This is the defect the issue names as most likely to be written:
   a consumer treating a missing verdict as "older artifact, proceed".

2. **The known-answer expectations are analytic.** Each is asserted here against
   the value written out from the metric's definition — independently of the
   battery — so a future edit that "fixes" an expectation to make the battery
   pass has to change the number in two places that disagree by construction.

3. **The emitted artifact conforms to the frozen cross-repo contract**
   (``contracts/evaluator_attestation.schema.json``), which crucible-evaluator
   consumes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis import evaluator_attestation as ea
from analysis.attestation import FAIL, PASS, UNKNOWN, read_attestation, worst

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "contracts" / "evaluator_attestation.schema.json"

RUN_DATE = "2026-08-07"


# ════════════════════════════════════════════════════════════════════════════
# 1. Fail-closed — absence, staleness and unrecognised states are never a pass
# ════════════════════════════════════════════════════════════════════════════

class _FakeS3:
    """Minimal S3 stub. ``body=None`` means the object does not exist."""

    def __init__(self, body=None, raw: bytes | None = None, last_modified=None):
        self._body = body
        self._raw = raw
        self._last_modified = last_modified
        self.puts: list[dict] = []

    def get_object(self, Bucket, Key):  # noqa: N803 — boto3 kwarg casing
        if self._body is None and self._raw is None:
            raise FileNotFoundError(f"NoSuchKey: {Key}")
        payload = self._raw if self._raw is not None else json.dumps(self._body).encode()

        class _Stream:
            def __init__(self, data):
                self._data = data

            def read(self):
                return self._data

        out = {"Body": _Stream(payload)}
        if self._last_modified is not None:
            out["LastModified"] = self._last_modified
        return out

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}


def _engine_verdict(verdict=PASS, run_date=RUN_DATE):
    return {"schema": "backtest_attestation-1.0.0", "component": "backtester",
            "run_date": run_date, "verdict": verdict, "checks": [], "n_checks": 5,
            "n_failed": 0}


class TestUpstreamReadIsFailClosed:
    def test_absent_object_is_unknown_not_pass(self):
        got = read_attestation("b", RUN_DATE, s3_client=_FakeS3(body=None))
        assert got["verdict"] == UNKNOWN
        assert got["verdict"] != PASS
        assert "could not read" in got["reason"]

    def test_unparseable_body_is_unknown(self):
        got = read_attestation("b", RUN_DATE, s3_client=_FakeS3(raw=b"{not json"))
        assert got["verdict"] == UNKNOWN

    def test_non_object_body_is_unknown(self):
        got = read_attestation("b", RUN_DATE, s3_client=_FakeS3(raw=b"[1, 2, 3]"))
        assert got["verdict"] == UNKNOWN
        assert "not a JSON object" in got["reason"]

    def test_verdict_from_another_cycle_is_never_inherited(self):
        stale = _engine_verdict(verdict=PASS, run_date="2026-07-31")
        got = read_attestation("b", RUN_DATE, s3_client=_FakeS3(body=stale))
        assert got["verdict"] == UNKNOWN
        assert "another cycle" in got["reason"]

    @pytest.mark.parametrize("raw", ["ok", "pass", "PASSED", "true", "", None, 1])
    def test_only_the_literal_PASS_is_a_pass(self, raw):
        body = _engine_verdict()
        body["verdict"] = raw
        got = read_attestation("b", RUN_DATE, s3_client=_FakeS3(body=body))
        assert got["verdict"] == UNKNOWN, f"{raw!r} must not grant the guarantee"

    def test_explicit_pass_is_a_pass_and_carries_as_of(self):
        from datetime import datetime, timezone

        stamp = datetime(2026, 8, 8, 6, 13, 29, tzinfo=timezone.utc)
        got = read_attestation(
            "b", RUN_DATE,
            s3_client=_FakeS3(body=_engine_verdict(), last_modified=stamp),
        )
        assert got["verdict"] == PASS
        assert got["as_of"] == "2026-08-08T06:13:29Z"

    def test_explicit_fail_propagates_as_fail_not_unknown(self):
        got = read_attestation(
            "b", RUN_DATE, s3_client=_FakeS3(body=_engine_verdict(verdict=FAIL)),
        )
        # A stage that DISAGREED is evidence the numbers are wrong; a stage that
        # could not run is absence of evidence. The consumer must not flatten
        # the first into the second.
        assert got["verdict"] == FAIL


class TestWorstCombine:
    @pytest.mark.parametrize("a,b,expected", [
        (PASS, PASS, PASS),
        (PASS, UNKNOWN, UNKNOWN),
        (UNKNOWN, PASS, UNKNOWN),
        (PASS, FAIL, FAIL),
        (UNKNOWN, FAIL, FAIL),
        (FAIL, FAIL, FAIL),
        (PASS, None, UNKNOWN),
        (None, None, UNKNOWN),
        (PASS, "ok", UNKNOWN),
    ])
    def test_combine(self, a, b, expected):
        assert worst(a, b) == expected

    def test_empty_combine_is_unknown(self):
        assert worst() == UNKNOWN


class _Args:
    def __init__(self, freeze=False, upload=True):
        self.freeze = freeze
        self.upload = upload


class TestPromotionGate:
    """The teeth: a non-PASS verdict must stop this stage promoting anything."""

    def test_missing_upstream_verdict_forces_freeze(self):
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=_FakeS3(body=None))
        assert body["upstream"]["verdict"] == UNKNOWN
        assert body["verdict"] == UNKNOWN
        assert body["promotion_withheld"] is True

        args = _Args(freeze=False)
        assert ea.apply_promotion_gate(body, args) is True
        assert args.freeze is True, (
            "an absent correctness verdict must withhold promotion — reading it "
            "as 'older artifact, proceed' is the defect §2.3a exists to remove"
        )

    def test_failed_upstream_verdict_forces_freeze(self):
        s3 = _FakeS3(body=_engine_verdict(verdict=FAIL))
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=s3)
        assert body["verdict"] == FAIL
        args = _Args(freeze=False)
        assert ea.apply_promotion_gate(body, args) is True
        assert args.freeze is True

    def test_stale_dated_upstream_verdict_forces_freeze(self):
        s3 = _FakeS3(body=_engine_verdict(run_date="2026-07-31"))
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=s3)
        args = _Args(freeze=False)
        assert ea.apply_promotion_gate(body, args) is True
        assert args.freeze is True

    def test_pass_on_both_halves_permits_promotion(self):
        s3 = _FakeS3(body=_engine_verdict(verdict=PASS))
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=s3)
        assert body["own"]["verdict"] == PASS, body["own"]
        assert body["verdict"] == PASS
        assert body["promotion_withheld"] is False
        args = _Args(freeze=False)
        assert ea.apply_promotion_gate(body, args) is False
        assert args.freeze is False

    def test_gate_never_unfreezes_an_explicitly_frozen_run(self):
        s3 = _FakeS3(body=_engine_verdict(verdict=PASS))
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=s3)
        args = _Args(freeze=True)
        ea.apply_promotion_gate(body, args)
        assert args.freeze is True


# ════════════════════════════════════════════════════════════════════════════
# 2. The known-answer expectations are analytic
# ════════════════════════════════════════════════════════════════════════════

class TestExpectationsAreAnalytic:
    """Each expectation restated from the metric's definition, not from the battery.

    These deliberately duplicate the literals in ``analysis/evaluator_attestation.py``.
    The duplication is the point: an expectation edited to make a failing battery
    go green has to be edited here too, and the reason for the number lives in
    both places.
    """

    def _by_name(self):
        return {s.name: s for s in ea._SCENARIOS()}

    def test_perfect_rank_agreement_is_one(self):
        assert self._by_name()["ic_perfect_monotone"].expected == 1.0

    def test_perfect_rank_reversal_is_minus_one(self):
        assert self._by_name()["ic_perfect_inverse"].expected == -1.0

    def test_single_adjacent_transposition_matches_the_no_ties_identity(self):
        # rho = 1 - 6*sum(d^2)/(n(n^2-1)); one adjacent swap => sum(d^2) = 2.
        n = 30
        expected = 1.0 - 6.0 * 2.0 / (n * (n * n - 1))
        assert self._by_name()["ic_single_adjacent_transposition"].expected == pytest.approx(
            expected, rel=1e-15,
        )
        # ...and it is not a degenerate +-1, which is what makes it the check
        # that can catch a wrong denominator.
        assert 0.99 < expected < 1.0

    def test_accuracy_is_the_beat_count_over_the_resolved_count(self):
        assert self._by_name()["hit_rate_accuracy_21d"].expected == pytest.approx(30 / 50)

    def test_unresolved_rows_stay_out_of_the_denominator(self):
        # 50 resolved + 12 unresolved rows; the denominator must be 50, not 62.
        assert self._by_name()["hit_rate_excludes_unresolved"].expected == 50.0

    def test_mean_alpha_is_the_constant_per_row_difference(self):
        assert self._by_name()["hit_rate_avg_alpha_21d"].expected == 0.02

    def test_perfectly_calibrated_set_has_zero_ece(self):
        assert self._by_name()["calibration_ece_perfect"].expected == 0.0

    def test_brier_score_of_the_perfect_set(self):
        # sum_bins[hits*(1-p)^2 + misses*p^2] / n_total
        total = sum(
            hits * (1 - p) ** 2 + (n - hits) * p ** 2
            for p, n, hits in ea._CAL_PERFECT
        )
        n_total = sum(n for _, n, _ in ea._CAL_PERFECT)
        assert total / n_total == pytest.approx(0.17, abs=1e-12)
        assert self._by_name()["calibration_brier_perfect"].expected == 0.17

    def test_miscalibrated_ece_is_the_sample_weighted_gap(self):
        # One bin of 20/100 realizes 0.9 against a predicted 0.5 -> 20*0.4/100.
        assert self._by_name()["calibration_ece_miscalibrated"].expected == pytest.approx(
            20 * 0.4 / 100, abs=1e-12,
        )


class TestBatteryRunsThroughTheProductionPath:
    def test_battery_passes_on_this_build(self):
        result = ea.run_evaluator_attestation(RUN_DATE)
        failed = [c for c in result["checks"] if not c["passed"]]
        assert result["verdict"] == PASS, failed
        assert result["n_checks"] == 11
        assert result["n_failed"] == 0

    def test_every_check_agrees_to_near_machine_precision(self):
        # The tolerances are not doing the work: real agreement is ~1e-16.
        for check in ea.run_evaluator_attestation(RUN_DATE)["checks"]:
            assert check["abs_error"] is not None
            assert check["abs_error"] < 1e-12, check

    def test_checks_call_the_production_functions_not_a_local_copy(self):
        """A parallel implementation inside the battery would attest the battery.

        Each compute helper must reach the module the Evaluator's own diagnostics
        import — asserted by patching the production symbol and observing the
        battery break.
        """
        import analysis.information_coefficient as prod_ic

        original = prod_ic.compute_ic
        try:
            prod_ic.compute_ic = lambda *a, **k: {"status": "ok", "ic": 0.123, "n": 30}
            result = ea.run_evaluator_attestation(RUN_DATE)
            ic_checks = [c for c in result["checks"] if c["name"].startswith("ic_")
                         and c["name"] != "ic_no_variance_is_not_zero_ic"]
            assert ic_checks
            assert all(not c["passed"] for c in ic_checks), (
                "the IC checks did not go through analysis.information_coefficient — "
                "they are attesting something other than the production path"
            )
            assert result["verdict"] == FAIL
        finally:
            prod_ic.compute_ic = original

    def test_a_disagreeing_check_is_FAIL_and_a_broken_one_is_UNKNOWN(self):
        """The taxonomy that keeps a harness fault from reading as a defect."""
        real = ea._SCENARIOS

        def disagreeing():
            scenarios = real()
            return [scenarios[0]._replace(compute=lambda: 0.5)]

        def broken():
            scenarios = real()
            def _raise():
                raise RuntimeError("numpy import blew up")
            return [scenarios[0]._replace(compute=_raise)]

        try:
            ea._SCENARIOS = disagreeing
            assert ea.run_evaluator_attestation(RUN_DATE)["verdict"] == FAIL
            ea._SCENARIOS = broken
            unknown = ea.run_evaluator_attestation(RUN_DATE)
            assert unknown["verdict"] == UNKNOWN
            assert unknown["n_failed"] == 0
            assert unknown["n_errored"] == 1
        finally:
            ea._SCENARIOS = real

    def test_battery_construction_failure_is_unknown_not_pass(self):
        real = ea._SCENARIOS
        try:
            def _boom():
                raise ImportError("no pandas on this box")
            ea._SCENARIOS = _boom
            result = ea.run_evaluator_attestation(RUN_DATE)
            assert result["verdict"] == UNKNOWN
            assert result["status"] == "error"
        finally:
            ea._SCENARIOS = real


# ════════════════════════════════════════════════════════════════════════════
# 3. Producer/consumer contract
# ════════════════════════════════════════════════════════════════════════════

class TestArtifactContract:
    def _schema(self):
        return json.loads(SCHEMA_PATH.read_text())

    def test_schema_file_is_valid_json_schema(self):
        jsonschema = pytest.importorskip("jsonschema")
        jsonschema.Draft202012Validator.check_schema(self._schema())

    @pytest.mark.parametrize("upstream_body,expected_verdict", [
        (_engine_verdict(verdict=PASS), PASS),
        (_engine_verdict(verdict=FAIL), FAIL),
        (None, UNKNOWN),
    ])
    def test_emitted_artifact_conforms(self, upstream_body, expected_verdict):
        jsonschema = pytest.importorskip("jsonschema")
        body = ea.build_stage_attestation(
            "b", RUN_DATE, s3_client=_FakeS3(body=upstream_body),
        )
        assert body["verdict"] == expected_verdict
        # Round-trip through JSON: the artifact is what S3 will hold, not the
        # in-memory dict.
        jsonschema.Draft202012Validator(self._schema()).validate(
            json.loads(json.dumps(body, default=str)),
        )

    def test_key_is_beside_the_engine_verdict(self):
        assert ea.attestation_key(RUN_DATE) == f"backtest/{RUN_DATE}/evaluator_attestation.json"

    def test_write_persists_under_that_key(self):
        s3 = _FakeS3(body=_engine_verdict())
        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=s3)
        key = ea.write_attestation("b", RUN_DATE, body, s3_client=s3)
        assert key == ea.attestation_key(RUN_DATE)
        assert s3.puts and s3.puts[0]["Key"] == key
        assert json.loads(s3.puts[0]["Body"])["verdict"] == PASS

    def test_write_failure_raises_rather_than_silently_dropping_the_evidence(self):
        class _Dead:
            def get_object(self, **kwargs):
                raise FileNotFoundError("nope")

            def put_object(self, **kwargs):
                raise RuntimeError("AccessDenied")

        body = ea.build_stage_attestation("b", RUN_DATE, s3_client=_Dead())
        with pytest.raises(RuntimeError):
            ea.write_attestation("b", RUN_DATE, body, s3_client=_Dead())
