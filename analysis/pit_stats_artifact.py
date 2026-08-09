"""pit_stats_artifact.py — per-pass pit_parity stats as a versioned S3 contract
(alpha-engine-config#6030).

The weekly SF's bundled ``Parity`` state is split into independent stages per
sf-pipeline-policy §2.1: ``PitParityLookahead`` and ``PitParityWalkforward``
each run ONE predictor pass on their own spot instance, ``PitParityCompare``
joins them. The two passes are independent *runs* but not independent
*products* — the artifact is the delta between them — so the split requires a
cross-stage artifact:

    s3://{bucket}/parity/{run_date}/pit_stats_{lookahead|walkforward}.json

governed by ``contracts/pit_stats_pass.schema.json`` (v1, M0 rule: a new
cross-repo artifact gets a versioned schema + producer/consumer contract tests
at birth). Pickle — the previous in-process transport
(``analysis/pit_parity.py::_run_predictor_pass_isolated``) — is not an
acceptable cross-boundary format; values are JSON with numpy coerced
explicitly (non-finite floats -> null).

Join semantics (sf-pipeline-policy §2.3a): a missing / unparseable /
non-``ok`` pass artifact does NOT abort the compare. The compare still runs
and emits ``backtest/{run_date}/pit_parity.json`` with ``status: "unknown"``
— the parity verdict is UNKNOWN, never a pass. Downstream
(``evaluate.py::_load_pit_parity_report`` -> Report Card
``pipeline_health.pit_parity_status``) reads the same ``status`` key it
already reads for ``"failed"`` / ``"incomplete"`` reports, so UNKNOWN
propagates to the machine without a shape change.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import time

import numpy as np

from analysis.pit_parity import (
    SCHEMA as PIT_PARITY_SCHEMA,
    _config_without_runtime_handles,
    _run_predictor_pass_isolated,
    build_contamination_report,
    read_prior_delta,
)

logger = logging.getLogger(__name__)

PASS_SCHEMA = "pit_stats_pass-1.0.0"
PASSES = ("lookahead", "walkforward")

# The EXACT per-pass stats fields the in-process implementation consumed from
# the pickled stats dicts (analysis/pit_parity.py::run_pit_parity +
# build_contamination_report + evaluate helpers). Do not add fields the
# compare does not read — the full in-memory stats dict (equity curves, trade
# logs, ...) deliberately stays process-local.
_SCALAR_FIELDS = (
    "sortino_ratio", "psr", "cvar_95", "max_drawdown",
    "total_return", "total_alpha",
)


def pass_artifact_key(run_date: str, which: str) -> str:
    """Canonical S3 key for one pass's stats artifact.

    Kept in lockstep with nousergon-data's SF-level consumer contract
    (``contracts/pit_stats_pass.consumer.json`` there) — change both or
    neither.
    """
    if which not in PASSES:
        raise ValueError(f"unknown pit_parity pass {which!r} (valid: {PASSES})")
    return f"parity/{run_date}/pit_stats_{which}.json"


def _coerce_number(value):
    """Explicit numpy/python scalar -> JSON number (non-finite -> None)."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _coerce_series(values):
    """1-D numeric series -> list of JSON numbers (non-finite -> None)."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64).ravel()
    return [float(v) if np.isfinite(v) else None for v in arr]


def _coerce_matrix(matrix):
    """2-D numeric matrix -> list of rows of JSON numbers (non-finite -> None)."""
    if matrix is None:
        return None
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        return None
    return [[float(v) if np.isfinite(v) else None for v in row] for row in arr]


def _coerce_jsonable(obj):
    """Recursive best-effort coercion for the predictor_metadata block."""
    if isinstance(obj, dict):
        return {str(k): _coerce_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_coerce_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if np.isfinite(f) else None
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _coerce_jsonable(obj.tolist())
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    return str(obj)


def extract_pass_stats(stats: dict) -> dict:
    """Lift the consumed stats subset out of a full in-memory pass dict,
    coercing every value to strict JSON (no pickle, no NaN literals)."""
    out: dict = {k: _coerce_number(stats.get(k)) for k in _SCALAR_FIELDS}
    out["daily_log_returns"] = _coerce_series(stats.get("daily_log_returns"))
    meta = stats.get("predictor_metadata")
    out["predictor_metadata"] = _coerce_jsonable(meta) if meta is not None else None
    out["_cscv_block_matrix"] = _coerce_matrix(stats.get("_cscv_block_matrix"))
    spec_ids = stats.get("_cscv_spec_ids")
    out["_cscv_spec_ids"] = (
        [_coerce_jsonable(v) for v in spec_ids] if spec_ids is not None else None
    )
    n_trials = stats.get("_cscv_n_trials")
    out["_cscv_n_trials"] = int(n_trials) if n_trials is not None else None
    return out


def build_pass_artifact(
    stats: dict, which: str, run_date: str, wall_clock_seconds: float | None = None,
) -> dict:
    """Assemble one pass's schema-v1 artifact from its in-memory stats dict."""
    status = stats.get("status") or "ok"
    artifact = {
        "schema": PASS_SCHEMA,
        "run_date": run_date,
        "pass": which,
        "status": status,
        "wall_clock_seconds": (
            round(float(wall_clock_seconds), 3) if wall_clock_seconds is not None else None
        ),
    }
    if status == "ok":
        artifact["stats"] = extract_pass_stats(stats)
    else:
        # A pass that returned a non-ok status dict (insufficient data,
        # executor path missing, ...) still always emits an artifact so the
        # compare can tell "ran and could not produce stats" from "never ran"
        # — both propagate as UNKNOWN, but the diagnosis differs.
        err = str(stats.get("error", ""))[:1000]
        if err:
            artifact["error_msg"] = err
    return artifact


def build_failure_pass_artifact(which: str, run_date: str, exc: BaseException) -> dict:
    """status=failed artifact for the pass-crashed path (always-emit contract
    — mirrors analysis/pit_parity.py::write_failure_artifact)."""
    return {
        "schema": PASS_SCHEMA,
        "run_date": run_date,
        "pass": which,
        "status": "failed",
        "error_class": type(exc).__name__,
        "error_msg": str(exc)[:1000],
    }


def validate_pass_artifact(artifact: dict) -> None:
    """Producer-side schema validation at write time. Raises on violation —
    a producer that ships a contract-breaking artifact must fail its own
    stage, not its consumer's."""
    from pathlib import Path

    import jsonschema

    schema_path = (
        Path(__file__).resolve().parents[1] / "contracts" / "pit_stats_pass.schema.json"
    )
    jsonschema.validate(artifact, json.loads(schema_path.read_text()))


def _put_json(bucket: str, key: str, doc: dict) -> None:
    """Strict S3 write — raises on failure. Unlike pit_parity.json's
    best-effort ``_write_artifact_to_s3``, the pass artifact IS the stage's
    product: a failed upload must fail the stage (the SF branch degrades and
    the compare emits UNKNOWN), never read as success."""
    import boto3

    boto3.client("s3").put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(doc, indent=2),
        ContentType="application/json",
    )
    logger.info("[pit_stats] artifact -> s3://%s/%s", bucket, key)


def _alert(message: str, *, dedup_key: str) -> None:
    """Best-effort WARNING page (mirrors handle_pit_parity_failure's alert
    half). Never raises."""
    try:
        from nousergon_lib.alerts import publish as _publish

        _publish(
            message,
            severity="warning",
            source="alpha-engine-backtester/pit_stats_artifact",
            dedup_key=dedup_key,
            dedup_window_min=720,
        )
    except Exception as alert_err:  # noqa: BLE001 — observability-only leg
        logger.error("[pit_stats] alert publish failed: %s", alert_err)


def publish_pass_artifact(config: dict, which: str) -> bool:
    """Run ONE pit_parity pass in an isolated subprocess and publish its
    stats artifact. Returns True iff the pass completed ``ok`` AND the
    artifact uploaded — the CLI exits non-zero otherwise so the SF branch
    records DEGRADED (fail-open at the SF layer, fail-loud at this one).
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    run_date = config.get("_run_date") or _dt.date.today().isoformat()
    key = pass_artifact_key(run_date, which)
    safe_config = _config_without_runtime_handles(config)

    started = time.monotonic()
    try:
        stats = _run_predictor_pass_isolated(safe_config, which, run_date)
    except Exception as exc:  # noqa: BLE001 — converted to failed artifact + alert
        elapsed = time.monotonic() - started
        logger.error("[pit_stats] %s pass failed after %.0fs: %s", which, elapsed, exc)
        artifact = build_failure_pass_artifact(which, run_date, exc)
        artifact["wall_clock_seconds"] = round(elapsed, 3)
        try:
            validate_pass_artifact(artifact)
            _put_json(bucket, key, artifact)
        except Exception as put_err:  # noqa: BLE001 — artifact absence => compare UNKNOWN
            logger.error("[pit_stats] failure-artifact write also failed: %s", put_err)
        _alert(
            f"pit_parity {which} pass failed on {run_date}: "
            f"{type(exc).__name__}: {str(exc)[:200]} — the PitParityCompare "
            f"stage will emit verdict UNKNOWN (see s3://{bucket}/{key})",
            dedup_key=f"pit_stats_{which}_failed_{run_date}",
        )
        return False

    elapsed = time.monotonic() - started
    artifact = build_pass_artifact(stats, which, run_date, wall_clock_seconds=elapsed)
    validate_pass_artifact(artifact)
    _put_json(bucket, key, artifact)
    if artifact["status"] != "ok":
        _alert(
            f"pit_parity {which} pass returned status="
            f"{artifact['status']!r} on {run_date} — compare will emit "
            f"verdict UNKNOWN (see s3://{bucket}/{key})",
            dedup_key=f"pit_stats_{which}_failed_{run_date}",
        )
        return False
    logger.info(
        "[pit_stats] %s pass ok in %.0fs — published s3://%s/%s",
        which, elapsed, bucket, key,
    )
    return True


def load_pass_artifact(bucket: str, run_date: str, which: str, s3_client=None):
    """Read one pass artifact. Returns ``(state, artifact_or_None)`` where
    state is 'ok' | 'failed' (artifact present, status != ok) | 'missing' |
    'unparseable'. Never raises on the §2.3a-expected absence classes; a
    non-404 S3 error IS raised (infrastructure breakage — the SF Catch
    degrades the stage, which is honest, unlike mapping it to UNKNOWN-with-
    green-stage)."""
    import boto3
    from botocore.exceptions import ClientError

    s3 = s3_client or boto3.client("s3")
    key = pass_artifact_key(run_date, which)
    try:
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return "missing", None
        raise
    try:
        artifact = json.loads(body)
    except Exception as e:  # noqa: BLE001 — unparseable == verdict-unknown input
        logger.error("[pit_stats] unparseable artifact at %s: %s", key, e)
        return "unparseable", None
    if not isinstance(artifact, dict) or artifact.get("status") != "ok" \
            or not isinstance(artifact.get("stats"), dict):
        return "failed", artifact if isinstance(artifact, dict) else None
    return "ok", artifact


def _stats_from_artifact(artifact: dict) -> dict:
    """Reconstruct the consumer-side stats dict from an ``ok`` artifact.

    Identity mapping by design (the schema's ``stats`` block mirrors the
    in-memory keys verbatim); null series entries round-trip to NaN inside
    ``np.asarray(..., dtype=float)`` exactly as the in-process path's
    non-finite values did, and are filtered by the same ``np.isfinite``
    guards downstream."""
    return dict(artifact["stats"])


def build_unknown_report(
    run_date: str, availability: dict[str, str],
) -> dict:
    """The §2.3a verdict when either pass artifact is unavailable: UNKNOWN,
    never pass. Shape extends the existing ``status``-keyed non-complete
    reports ('failed', 'incomplete') that evaluate.py / the Report Card
    already read — a new status VALUE, not a new shape."""
    return {
        "schema": PIT_PARITY_SCHEMA,
        "run_date": run_date,
        "status": "unknown",
        "verdict": "UNKNOWN",
        "pass_availability": dict(availability),
        # Compatibility keys mirroring the legacy "incomplete" report so
        # existing readers of current_status/pit_status keep working.
        "current_status": availability.get("lookahead"),
        "pit_status": availability.get("walkforward"),
        "observational": True,
    }


def run_compare_and_publish(config: dict) -> dict:
    """The PitParityCompare stage: read both pass artifacts, compute the
    contamination report (delta_pit_minus_current + PBO + materiality +
    parity_alarms via ``build_contamination_report``), and write
    ``backtest/{run_date}/pit_parity.json`` where today's consumers read it.

    Missing/unparseable/non-ok pass artifact => the verdict is UNKNOWN
    (``build_unknown_report``) and the report is still written — the compare
    stage SUCCEEDS at emitting an honest verdict; the degradation itself is
    already flagged by the failed branch's SF marker. The report upload is
    STRICT (raises) — the report is this stage's product.
    """
    import boto3

    bucket = config.get("signals_bucket", "alpha-engine-research")
    run_date = config.get("_run_date") or _dt.date.today().isoformat()
    s3 = boto3.client("s3")

    availability: dict[str, str] = {}
    artifacts: dict[str, dict | None] = {}
    for which in PASSES:
        state, artifact = load_pass_artifact(bucket, run_date, which, s3_client=s3)
        availability[which] = state
        artifacts[which] = artifact

    if any(state != "ok" for state in availability.values()):
        logger.error(
            "[pit_stats] compare inputs unavailable (%s) — emitting verdict UNKNOWN",
            availability,
        )
        report = build_unknown_report(run_date, availability)
        _put_json(bucket, f"backtest/{run_date}/pit_parity.json", report)
        _alert(
            f"pit_parity verdict UNKNOWN on {run_date}: pass artifacts "
            f"{availability} — absence of a verdict must not be read as a "
            f"clean pass (sf-pipeline-policy §2.3a). "
            f"See s3://{bucket}/backtest/{run_date}/pit_parity.json",
            dedup_key=f"pit_parity_unknown_{run_date}",
        )
        return report

    cur_stats = _stats_from_artifact(artifacts["lookahead"])
    pit_stats = _stats_from_artifact(artifacts["walkforward"])
    wf_meta = (pit_stats.get("predictor_metadata") or {}).get("walk_forward")
    prior_delta = read_prior_delta(bucket, run_date)
    report = build_contamination_report(
        cur_stats, pit_stats, run_date=run_date, wf_meta=wf_meta,
        pit_block_matrix=pit_stats.get("_cscv_block_matrix"),
        pit_spec_ids=pit_stats.get("_cscv_spec_ids"),
        n_trials=pit_stats.get("_cscv_n_trials"),
        prior_delta=prior_delta,
    )
    # Explicit ok status: a completed report previously carried NO status key
    # and readers default absent -> "ok" (evaluate.py:2296), so this is a
    # compatible, strictly-more-honest extension.
    report["status"] = "ok"
    report["pass_artifacts"] = {
        which: pass_artifact_key(run_date, which) for which in PASSES
    }
    report["pass_wall_clock_seconds"] = {
        which: (artifacts[which] or {}).get("wall_clock_seconds")
        for which in PASSES
    }
    _put_json(bucket, f"backtest/{run_date}/pit_parity.json", report)
    report["_s3_key"] = f"backtest/{run_date}/pit_parity.json"
    return report
