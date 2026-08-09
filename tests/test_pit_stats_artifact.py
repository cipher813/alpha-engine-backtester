"""Producer contract tests for the per-pass pit_stats artifact
(alpha-engine-config#6030, M0 rule: versioned schema + producer/consumer
contract tests at birth).

Producer side lives HERE (crucible-backtester owns both passes and the
compare); the SF-level consumer contract (S3 key wiring + schema reference)
lives in nousergon-data (tests/test_pit_stats_consumer_contract.py there).
The key template and schema id asserted below are the shared contract
surface — change them in lockstep or not at all.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

import analysis.pit_stats_artifact as psa
from analysis import pit_parity as pp

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "contracts" / "pit_stats_pass.schema.json"


def _schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _validate(artifact: dict) -> None:
    import jsonschema

    jsonschema.validate(artifact, _schema())


def _stats(sortino, psr, cvar, mdd, dlr, alpha) -> dict:
    return {
        "sortino_ratio": sortino,
        "psr": psr,
        "cvar_95": cvar,
        "max_drawdown": mdd,
        "daily_log_returns": dlr,
        "total_alpha": alpha,
        "total_return": 0.1,
    }


class _FakeS3:
    """Minimal in-memory S3 double for the compare/publish paths."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.puts: list[dict] = []

    def put_object(self, *, Bucket, Key, Body, **kw):
        body = Body if isinstance(Body, bytes) else Body.encode()
        self.objects[Key] = body
        self.puts.append({"Bucket": Bucket, "Key": Key, "Body": body})

    def get_object(self, *, Bucket, Key):
        from botocore.exceptions import ClientError

        if Key not in self.objects:
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": Key}}, "GetObject"
            )
        return {"Body": types.SimpleNamespace(read=lambda: self.objects[Key])}


@pytest.fixture()
def fake_s3(monkeypatch):
    s3 = _FakeS3()
    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: s3
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    return s3


@pytest.fixture(autouse=True)
def _mute_alerts(monkeypatch):
    """Never let a test page. ``_alert`` imports nousergon_lib.alerts inside
    the function — stub the module."""
    sent: list[str] = []
    fake_alerts = types.ModuleType("nousergon_lib.alerts")
    fake_alerts.publish = lambda msg, **kw: sent.append(msg)
    monkeypatch.setitem(sys.modules, "nousergon_lib.alerts", fake_alerts)
    return sent


# ---------------------------------------------------------------------------
# Contract surface — key template + schema id (lockstep with nousergon-data)
# ---------------------------------------------------------------------------


def test_pass_artifact_key_template():
    assert psa.pass_artifact_key("2026-08-08", "lookahead") == (
        "parity/2026-08-08/pit_stats_lookahead.json"
    )
    assert psa.pass_artifact_key("2026-08-08", "walkforward") == (
        "parity/2026-08-08/pit_stats_walkforward.json"
    )
    with pytest.raises(ValueError):
        psa.pass_artifact_key("2026-08-08", "sideways")


def test_schema_id_pinned():
    assert psa.PASS_SCHEMA == "pit_stats_pass-1.0.0"
    assert _schema()["properties"]["schema"]["const"] == psa.PASS_SCHEMA


# ---------------------------------------------------------------------------
# Producer contract — every artifact shape the producer can emit validates
# ---------------------------------------------------------------------------


def test_ok_artifact_from_numpy_stats_validates_and_is_strict_json():
    rng = np.random.default_rng(7)
    stats = _stats(
        np.float64(1.2), np.float32(0.9), np.float64(-0.03), -0.15,
        np.asarray(list(rng.normal(0.001, 0.01, 50)) + [np.nan, np.inf]),
        np.float64(0.03),
    )
    stats["predictor_metadata"] = {
        "walk_forward": {"n_folds": np.int64(12), "n_cold_start_excluded": 2},
    }
    stats["_cscv_block_matrix"] = np.asarray(rng.normal(0, 0.1, size=(8, 6)))
    stats["_cscv_spec_ids"] = list(range(6))
    stats["_cscv_n_trials"] = np.int64(6)

    artifact = psa.build_pass_artifact(stats, "walkforward", "2026-08-08",
                                       wall_clock_seconds=612.7)
    _validate(artifact)
    # Strict JSON: no NaN/Infinity literals may survive serialization.
    text = json.dumps(artifact, allow_nan=False)
    round_trip = json.loads(text)
    assert round_trip["stats"]["daily_log_returns"][-2:] == [None, None]
    assert round_trip["status"] == "ok"
    assert round_trip["pass"] == "walkforward"
    assert round_trip["wall_clock_seconds"] == pytest.approx(612.7)
    assert round_trip["stats"]["_cscv_n_trials"] == 6


def test_non_ok_status_dict_produces_validating_non_ok_artifact():
    artifact = psa.build_pass_artifact(
        {"status": "insufficient_data"}, "lookahead", "2026-08-08",
    )
    _validate(artifact)
    assert artifact["status"] == "insufficient_data"
    assert "stats" not in artifact


def test_failure_artifact_validates():
    artifact = psa.build_failure_pass_artifact(
        "lookahead", "2026-08-08", RuntimeError("child exploded"),
    )
    _validate(artifact)
    assert artifact["status"] == "failed"
    assert artifact["error_class"] == "RuntimeError"


def test_producer_validation_rejects_contract_break():
    import jsonschema

    bad = psa.build_pass_artifact(_stats(1, 1, -0.1, -0.1, [0.01], 0.0),
                                  "lookahead", "2026-08-08")
    bad["pass"] = "sideways"
    with pytest.raises(jsonschema.ValidationError):
        psa.validate_pass_artifact(bad)


# ---------------------------------------------------------------------------
# Round-trip fidelity — the S3 hop must not change the report the compare
# builds relative to the retired in-process path
# ---------------------------------------------------------------------------


def test_round_trip_preserves_contamination_delta():
    rng = np.random.default_rng(4)
    n = 120
    cur = _stats(1.0, 0.9, -0.03, -0.15, list(rng.normal(0.001, 0.01, n)), 0.03)
    pit = _stats(0.6, 0.9, -0.03, -0.15, list(rng.normal(0.0005, 0.01, n)), 0.02)
    pit["predictor_metadata"] = {"walk_forward": {"n_folds": 12}}

    direct = pp.build_contamination_report(cur, pit, run_date="2026-08-08")

    cur_rt = psa._stats_from_artifact(
        json.loads(json.dumps(
            psa.build_pass_artifact(cur, "lookahead", "2026-08-08")))
    )
    pit_rt = psa._stats_from_artifact(
        json.loads(json.dumps(
            psa.build_pass_artifact(pit, "walkforward", "2026-08-08")))
    )
    via_s3 = pp.build_contamination_report(cur_rt, pit_rt, run_date="2026-08-08")

    for k, v in direct["delta_pit_minus_current"].items():
        assert via_s3["delta_pit_minus_current"][k] == pytest.approx(v), k
    assert via_s3["materiality"]["basis"] == direct["materiality"]["basis"]


# ---------------------------------------------------------------------------
# load_pass_artifact — the four §2.3a input states
# ---------------------------------------------------------------------------


def test_load_pass_artifact_states():
    ok_doc = psa.build_pass_artifact(
        _stats(1, 1, -0.1, -0.1, [0.01, 0.0], 0.0), "lookahead", "2026-08-08")
    s3 = _FakeS3({
        "parity/2026-08-08/pit_stats_lookahead.json":
            json.dumps(ok_doc).encode(),
        "parity/2026-08-07/pit_stats_lookahead.json": b"{not json",
        "parity/2026-08-06/pit_stats_lookahead.json":
            json.dumps({"schema": psa.PASS_SCHEMA, "status": "failed"}).encode(),
    })
    assert psa.load_pass_artifact("b", "2026-08-08", "lookahead", s3)[0] == "ok"
    assert psa.load_pass_artifact("b", "2026-08-07", "lookahead", s3)[0] == "unparseable"
    assert psa.load_pass_artifact("b", "2026-08-06", "lookahead", s3)[0] == "failed"
    assert psa.load_pass_artifact("b", "2026-08-05", "lookahead", s3)[0] == "missing"


def test_load_pass_artifact_non_404_raises():
    from botocore.exceptions import ClientError

    class _Denied:
        def get_object(self, **kw):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject"
            )

    with pytest.raises(ClientError):
        psa.load_pass_artifact("b", "2026-08-08", "lookahead", _Denied())


# ---------------------------------------------------------------------------
# run_compare_and_publish — §2.3a join semantics
# ---------------------------------------------------------------------------


def _cfg():
    return {"signals_bucket": "b", "_run_date": "2026-08-08"}


def test_compare_missing_pass_emits_unknown_never_pass(fake_s3, _mute_alerts):
    report = psa.run_compare_and_publish(_cfg())
    assert report["status"] == "unknown"
    assert report["verdict"] == "UNKNOWN"
    assert report["pass_availability"] == {
        "lookahead": "missing", "walkforward": "missing",
    }
    # The report is still WRITTEN (always-emit) to the canonical consumer key.
    assert "backtest/2026-08-08/pit_parity.json" in fake_s3.objects
    written = json.loads(fake_s3.objects["backtest/2026-08-08/pit_parity.json"])
    assert written["status"] == "unknown"
    # And it paged.
    assert any("UNKNOWN" in m for m in _mute_alerts)
    # A verdict-bearing key must never be "ok" on this path.
    assert written.get("delta_pit_minus_current") is None


def test_compare_failed_pass_emits_unknown(fake_s3):
    ok_doc = psa.build_pass_artifact(
        _stats(1, 1, -0.1, -0.1, [0.01, 0.0], 0.0), "lookahead", "2026-08-08")
    fake_s3.objects["parity/2026-08-08/pit_stats_lookahead.json"] = (
        json.dumps(ok_doc).encode())
    fake_s3.objects["parity/2026-08-08/pit_stats_walkforward.json"] = (
        json.dumps(psa.build_failure_pass_artifact(
            "walkforward", "2026-08-08", RuntimeError("boom"))).encode())
    report = psa.run_compare_and_publish(_cfg())
    assert report["status"] == "unknown"
    assert report["pass_availability"]["walkforward"] == "failed"
    assert report["pass_availability"]["lookahead"] == "ok"


def test_compare_both_ok_emits_full_report(fake_s3):
    rng = np.random.default_rng(11)
    n = 90
    cur = _stats(1.0, 0.9, -0.03, -0.15, list(rng.normal(0.001, 0.01, n)), 0.03)
    pit = _stats(0.7, 0.9, -0.03, -0.15, list(rng.normal(0.0005, 0.01, n)), 0.02)
    pit["predictor_metadata"] = {"walk_forward": {"n_folds": 12}}
    for which, stats in (("lookahead", cur), ("walkforward", pit)):
        doc = psa.build_pass_artifact(stats, which, "2026-08-08",
                                      wall_clock_seconds=100.0)
        fake_s3.objects[psa.pass_artifact_key("2026-08-08", which)] = (
            json.dumps(doc).encode())

    report = psa.run_compare_and_publish(_cfg())
    assert report["status"] == "ok"
    assert report["delta_pit_minus_current"]["sortino_ratio"] == pytest.approx(-0.3)
    assert report["pass_artifacts"] == {
        "lookahead": "parity/2026-08-08/pit_stats_lookahead.json",
        "walkforward": "parity/2026-08-08/pit_stats_walkforward.json",
    }
    assert report["pass_wall_clock_seconds"] == {
        "lookahead": 100.0, "walkforward": 100.0,
    }
    # alarms leg ran (build_contamination_report -> evaluate_parity_alarms)
    assert "alarms" in report
    written = json.loads(fake_s3.objects["backtest/2026-08-08/pit_parity.json"])
    assert written["status"] == "ok"
    assert report["_s3_key"] == "backtest/2026-08-08/pit_parity.json"


def test_compare_report_card_reads_unknown_status():
    """The §2.3a propagation consumer check: evaluate.py's Report Card
    surfaces report.get('status', 'ok') — the UNKNOWN report must read as
    'unknown' there, never default to 'ok'."""
    report = psa.build_unknown_report("2026-08-08", {
        "lookahead": "missing", "walkforward": "ok",
    })
    assert report.get("status", "ok") == "unknown"
    # legacy compat keys mirror the incomplete-report shape
    assert report["current_status"] == "missing"
    assert report["pit_status"] == "ok"


# ---------------------------------------------------------------------------
# publish_pass_artifact — pass-stage entry
# ---------------------------------------------------------------------------


def test_publish_pass_artifact_ok(monkeypatch, fake_s3):
    monkeypatch.setattr(
        psa, "_run_predictor_pass_isolated",
        lambda cfg, which, run_date: _stats(1.0, 0.9, -0.03, -0.15, [0.01], 0.02),
    )
    ok = psa.publish_pass_artifact(_cfg(), "lookahead")
    assert ok is True
    key = "parity/2026-08-08/pit_stats_lookahead.json"
    assert key in fake_s3.objects
    written = json.loads(fake_s3.objects[key])
    _validate(written)
    assert written["status"] == "ok"
    assert written["wall_clock_seconds"] is not None


def test_publish_pass_artifact_crash_writes_failed_artifact_and_returns_false(
        monkeypatch, fake_s3, _mute_alerts):
    def _boom(cfg, which, run_date):
        raise RuntimeError("child exploded")

    monkeypatch.setattr(psa, "_run_predictor_pass_isolated", _boom)
    ok = psa.publish_pass_artifact(_cfg(), "walkforward")
    assert ok is False
    written = json.loads(
        fake_s3.objects["parity/2026-08-08/pit_stats_walkforward.json"])
    _validate(written)
    assert written["status"] == "failed"
    assert written["error_class"] == "RuntimeError"
    assert any("UNKNOWN" in m for m in _mute_alerts)


def test_publish_pass_artifact_strict_upload_failure_is_loud(monkeypatch):
    """Unlike pit_parity.json's best-effort write, a pass-artifact upload
    failure must NOT read as success — the artifact is the stage product."""
    monkeypatch.setattr(
        psa, "_run_predictor_pass_isolated",
        lambda cfg, which, run_date: _stats(1.0, 0.9, -0.03, -0.15, [0.01], 0.02),
    )

    class _BoomS3:
        def put_object(self, **kw):
            raise RuntimeError("S3 down")

    fake_boto3 = types.ModuleType("boto3")
    fake_boto3.client = lambda *a, **k: _BoomS3()
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)
    with pytest.raises(RuntimeError):
        psa.publish_pass_artifact(_cfg(), "lookahead")
