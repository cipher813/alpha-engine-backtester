"""alpha-engine-config-I8704 — a canary that graded nothing must not say `ok`.

Under `dry_run` the handler short-circuits both compute phases to
``{"status": "dry_run"}`` and skips the retrain-alert email. Every status check
downstream reads fields those payloads do not carry, so they all fall through
and the handler reported ``status=ok warnings=[]`` — champion-challenger-policy
§7.2, "a well-formed artifact containing nothing".

That was not a latent risk. Measured 2026-08-26: the deploy canary
(``infrastructure/deploy_health.sh``, payload ``{"dry_run": true}``) is the ONLY
caller invoking this Lambda, because the scheduled EventBridge rule
``alpha-engine-predictor-health-check`` has been DISABLED since the 2026-08-07
automation pause. Live CloudWatch, 2026-08-25 and 2026-08-26:

    INFO [lambda_health] [dry_run] Skipping retrain alert email
    INFO [lambda_health] Health check complete in 13.5s: status=ok warnings=[]

while ``predictor/metrics/production_health.json`` carried
``degradation_flag: true`` (``ic_ratio: 0.11``).

Also pins the liveness anchor: ``_persist_metric`` stamps ``evaluated_at`` on
every real write, so "this detector has not graded in N days" is derivable from
the artifact rather than from a rule's console state.
"""
from __future__ import annotations

import json

import pytest


# ── dry_run honesty ──────────────────────────────────────────────────────────


def _load_handler_source() -> str:
    from pathlib import Path

    return (Path(__file__).resolve().parent.parent / "lambda_health" / "handler.py").read_text()


def test_dry_run_status_is_never_ok():
    """The handler must set a distinct status for a dry run.

    Asserted on the source rather than by invoking the handler: `handler()` is
    decorated with `@monitor_handler`, runs a preflight against a live S3
    bucket and downloads a 356 MB research.db before reaching the status block,
    none of which is reachable in a unit test. The property under test is a
    single branch, and the pre-fix source did not contain it.
    """
    src = _load_handler_source()
    assert 'status = "dry_run"' in src, (
        "handler() has no dry_run status branch — a canary invocation that "
        "evaluated no metric will report `ok`. See alpha-engine-config-I8704."
    )


def test_dry_run_branch_sits_after_every_degradation_check():
    """Order matters: the dry_run override must come LAST, so it cannot mask a
    real `error` on a phase that threw before the short-circuit."""
    src = _load_handler_source()
    i_degradation = src.index('warnings.append("IC degradation detected")')
    i_dry_run = src.index('status = "dry_run"')
    assert i_dry_run > i_degradation, (
        "the dry_run status override must be evaluated after the degradation "
        "checks, not before them"
    )


def test_dry_run_emits_a_graded_nothing_warning():
    src = _load_handler_source()
    assert "NO metric was evaluated" in src, (
        "a dry run must say what it did NOT do; a bare status change still "
        "leaves a reader inferring coverage from an empty warnings list"
    )


def test_canary_asserts_on_status_code_not_on_status_field():
    """Guard the claim in the fix's own comment: making `status` honest cannot
    fail a deploy, because deploy_health.sh reads `statusCode`."""
    from pathlib import Path

    script = (
        Path(__file__).resolve().parent.parent / "infrastructure" / "deploy_health.sh"
    ).read_text()
    assert "d.get('statusCode', 0)" in script
    assert 'CANARY_STATUS" != "200"' in script


# ── liveness anchor ──────────────────────────────────────────────────────────


def test_persist_metric_stamps_evaluated_at(monkeypatch):
    from analysis import production_health

    captured: dict = {}

    class _FakeS3:
        def put_object(self, **kw):
            captured.update(json.loads(kw["Body"].decode()))

    monkeypatch.setattr(
        production_health.boto3, "client", lambda *_a, **_k: _FakeS3()
    )
    production_health._persist_metric(
        "test-bucket", "predictor/metrics/production_health.json",
        {"date": "2026-08-22", "degradation_flag": True},
    )

    assert "evaluated_at" in captured, (
        "production_health.json carries no evaluated_at — 'this detector has "
        "not graded in N days' is not derivable from the artifact "
        "(alpha-engine-config-I8704)"
    )
    assert captured["evaluated_at"].endswith("Z")
    # The caller's dict must not be mutated — callers return it to the handler.
    assert captured["degradation_flag"] is True


def test_persist_metric_does_not_mutate_the_callers_dict(monkeypatch):
    from analysis import production_health

    class _FakeS3:
        def put_object(self, **_kw):
            pass

    monkeypatch.setattr(
        production_health.boto3, "client", lambda *_a, **_k: _FakeS3()
    )
    original = {"date": "2026-08-22", "status": "ok"}
    production_health._persist_metric("b", "k", original)
    assert "evaluated_at" not in original, (
        "_persist_metric mutated its argument; the handler returns that same "
        "dict as the phase result"
    )
