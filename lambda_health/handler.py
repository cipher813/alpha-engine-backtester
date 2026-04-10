"""
lambda_health/handler.py — Daily predictor health check Lambda.

Runs lightweight model health monitoring (Phases 2a, 2b, 5) on weekdays
after predictor inference. Catches model degradation within 1-2 days
instead of waiting for the Saturday backtester.

Checks:
  - Production health: 30d rolling IC, hit rate, regime IC, mode collapse
  - Calibration validation: per-bin confidence vs actual hit rate
  - Retrain alert: evaluates 5 trigger conditions, sends email if fired

Feature drift (Phase 3) is NOT recomputed — it needs ArcticDB which is
too heavy for this Lambda. Instead, reads the last weekly feature_drift.json
from S3 as input to the retrain alert evaluator.

Lambda configuration:
  Memory: 512 MB  |  Timeout: 120s  |  Runtime: container (python:3.12)

Environment variables:
  S3_BUCKET          — default: alpha-engine-research
  EMAIL_SENDER       — from-address for retrain alerts
  EMAIL_RECIPIENTS   — comma-separated recipient list
  GMAIL_APP_PASSWORD — Gmail App Password (enables SMTP path)
  AWS_REGION         — SES fallback region (default: us-east-1)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time

# Load secrets from SSM Parameter Store before any os.environ.get calls
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ssm_secrets import load_secrets
load_secrets()

log = logging.getLogger(__name__)


def handler(event: dict, context) -> dict:
    """
    AWS Lambda entry point.

    event may contain:
        date     (str)  : Override date YYYY-MM-DD (default: today UTC)
        dry_run  (bool) : If True, skip S3 writes and email (for canary tests)
    """
    # Structured logging + flow-doctor owned by log_config.py. When
    # FLOW_DOCTOR_ENABLED=1 the root logger gets a handler that captures
    # ERROR+ records — every log.error() call in the phases below is
    # automatically routed to flow-doctor without explicit fd.report() plumbing.
    from log_config import setup_logging, get_flow_doctor
    setup_logging("lambda_health")
    fd = get_flow_doctor()

    t0 = time.time()
    bucket = os.environ.get("S3_BUCKET", "alpha-engine-research")
    run_date = event.get("date")
    dry_run = event.get("dry_run", False)

    log.info("Predictor health check starting: bucket=%s date=%s dry_run=%s", bucket, run_date, dry_run)

    # Track critical errors across phases. If any phase fails, return 500
    # so the Lambda caller (EventBridge scheduler / monitoring) sees the
    # failure instead of a misleading 200 OK. The previous behavior
    # returned 200 regardless of per-phase exceptions, which meant
    # degradations and calibration drift went unnoticed.
    phase_errors: list[str] = []

    # ── Download research.db from S3 ─────────────────────────────────────
    db_path = _download_research_db(bucket)
    if not db_path:
        return _response(500, "Failed to download research.db from S3")

    results = {}

    # ── Phase 2a: Production health ──────────────────────────────────────
    production_health = None
    try:
        from analysis.production_health import compute_production_health
        if dry_run:
            log.info("[dry_run] Skipping production health S3 write")
            production_health = {"status": "dry_run"}
        else:
            production_health = compute_production_health(db_path, bucket, run_date)
        results["production_health"] = production_health
        log.info("Production health: %s", _summarize(production_health))
    except Exception as exc:
        log.error("Production health failed: %s", exc, exc_info=True)
        results["production_health"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"production_health: {exc}")

    # ── Phase 2b: Calibration validation ─────────────────────────────────
    calibration = None
    try:
        from analysis.production_health import compute_calibration_validation
        if dry_run:
            log.info("[dry_run] Skipping calibration S3 write")
            calibration = {"status": "dry_run"}
        else:
            calibration = compute_calibration_validation(db_path, bucket, run_date)
        results["calibration"] = calibration
        log.info("Calibration: %s", _summarize(calibration))
    except Exception as exc:
        log.error("Calibration validation failed: %s", exc, exc_info=True)
        results["calibration"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"calibration: {exc}")

    # ── Load last weekly feature drift (read-only, not recomputed) ───────
    feature_drift = _load_last_feature_drift(bucket)

    # ── Phase 5: Retrain alert evaluation ────────────────────────────────
    try:
        from analysis.retrain_alert import evaluate_retrain_triggers, send_retrain_alert
        alert = evaluate_retrain_triggers(production_health, feature_drift, calibration)
        results["retrain_alert"] = {
            "triggered": alert.get("triggered", False),
            "n_triggers": alert.get("n_triggers", 0),
            "summary": alert.get("summary", ""),
        }

        if alert.get("triggered") and not dry_run:
            email_config = _build_email_config()
            send_result = send_retrain_alert(alert, email_config, bucket)
            results["retrain_alert"]["email_sent"] = send_result.get("sent", False)
        elif dry_run:
            log.info("[dry_run] Skipping retrain alert email")
    except Exception as exc:
        log.error("Retrain alert evaluation failed: %s", exc, exc_info=True)
        results["retrain_alert"] = {"status": "error", "error": str(exc)}
        phase_errors.append(f"retrain_alert: {exc}")

    # ── Write health status to S3 ────────────────────────────────────────
    elapsed = time.time() - t0
    status = "ok"
    warnings = []

    if phase_errors:
        status = "error"
        warnings.extend(phase_errors)
    if production_health and production_health.get("degradation_flag"):
        status = "degraded"
        warnings.append("IC degradation detected")
    if production_health and production_health.get("mode_collapse_flag"):
        status = "degraded"
        warnings.append("Mode collapse detected")
    if results.get("retrain_alert", {}).get("triggered"):
        status = "degraded"

    if not dry_run:
        try:
            from health_status import write_health
            write_health(
                bucket=bucket,
                module_name="predictor_health_check",
                status=status,
                run_date=run_date,
                duration_seconds=round(elapsed, 1),
                summary=results,
                warnings=warnings,
            )
        except Exception as exc:
            log.error("Failed to write health status: %s", exc, exc_info=True)
            phase_errors.append(f"write_health: {exc}")

    log.info("Health check complete in %.1fs: status=%s warnings=%s", elapsed, status, warnings)

    # Return 500 when any critical phase failed, so callers see the
    # failure. Returning 200 here is what hid the 2026-04-10 predictor
    # degradation for 3 hours — Lambda dashboards showed green while the
    # underlying checks were throwing exceptions.
    if phase_errors:
        return _response(500, {
            "status": "error",
            "duration_seconds": round(elapsed, 1),
            "phase_errors": phase_errors,
            "results": results,
        })

    return _response(200, {
        "status": status,
        "duration_seconds": round(elapsed, 1),
        "results": results,
    })


def _download_research_db(bucket: str) -> str | None:
    """Download research.db from S3 to /tmp."""
    import boto3
    db_path = "/tmp/research.db"
    try:
        s3 = boto3.client("s3")
        s3.download_file(bucket, "research.db", db_path)
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        log.info("Downloaded research.db (%.1f MB) to %s", size_mb, db_path)
        return db_path
    except Exception as exc:
        log.error("Failed to download research.db: %s", exc)
        return None


def _load_last_feature_drift(bucket: str) -> dict | None:
    """Read the most recent feature_drift.json from S3 (weekly output)."""
    import boto3
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/metrics/feature_drift.json")
        return json.loads(obj["Body"].read())
    except Exception:
        log.debug("No feature_drift.json available — skipping drift input for retrain alert")
        return None


def _build_email_config() -> dict:
    """Build email config from environment variables."""
    recipients_str = os.environ.get("EMAIL_RECIPIENTS", "")
    recipients = [r.strip() for r in recipients_str.split(",") if r.strip()]
    return {
        "email_sender": os.environ.get("EMAIL_SENDER", ""),
        "email_recipients": recipients,
        "aws_region": os.environ.get("AWS_REGION", "us-east-1"),
    }


def _summarize(result: dict | None) -> str:
    """One-line summary for logging."""
    if not result:
        return "None"
    if "status" in result and result["status"] in ("skipped", "error", "dry_run"):
        return f"status={result['status']}"
    parts = []
    for key in ("rolling_30d_ic", "rolling_30d_hit_rate", "degradation_flag",
                "mode_collapse_flag", "overall_ece", "calibration_quality"):
        if key in result:
            parts.append(f"{key}={result[key]}")
    return " ".join(parts) if parts else str(result.get("status", "ok"))


def _response(code: int, body) -> dict:
    """Standard Lambda response."""
    return {
        "statusCode": code,
        "body": json.dumps(body, default=str) if isinstance(body, dict) else body,
    }
