"""
regression_monitor.py — rolling metrics, promotion baselines, and auto-rollback.

After each weekly backtester run:
  1. save_rolling_metrics() persists current metrics to S3 history
  2. check_regression() compares current metrics to the promotion baseline
  3. If thresholds are breached, rollback_all() restores previous configs

The promotion baseline is saved automatically before any optimizer writes new
params to S3.  This module is fully automated — no human approval needed.
"""

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

from optimizer.rollback import rollback_all

logger = logging.getLogger(__name__)

S3_METRICS_PREFIX = "config/metrics_history/"
S3_BASELINE_KEY = "config/promotion_baseline.json"

# Default thresholds (can be overridden via config.yaml regression_monitor section)
DEFAULT_ACCURACY_DROP_PP = 5.0     # rollback if accuracy drops > 5 percentage points
DEFAULT_SHARPE_DROP_PCT = 0.20     # rollback if Sharpe drops > 20%


def extract_metrics(portfolio_stats: dict | None, signal_quality: dict | None) -> dict:
    """Extract the metrics used for regression comparison."""
    metrics = {}

    if portfolio_stats and isinstance(portfolio_stats, dict):
        for key in ("sharpe_ratio", "total_alpha", "max_drawdown", "win_rate"):
            if key in portfolio_stats:
                metrics[key] = portfolio_stats[key]

    if signal_quality and isinstance(signal_quality, dict):
        overall = signal_quality.get("overall", {})
        for key in ("accuracy_10d", "accuracy_30d"):
            if key in overall:
                metrics[key] = overall[key]

    return metrics


def save_rolling_metrics(bucket: str, run_date: str, metrics: dict) -> None:
    """Persist rolling metrics snapshot to S3."""
    if not metrics:
        logger.info("No metrics to save for %s", run_date)
        return

    payload = {
        "run_date": run_date,
        "saved_at": date.today().isoformat(),
        **metrics,
    }
    key = f"{S3_METRICS_PREFIX}{run_date}.json"
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info("Rolling metrics saved to s3://%s/%s", bucket, key)


def save_promotion_baseline(bucket: str, metrics: dict, promoted_configs: list[str]) -> None:
    """
    Save current metrics as the pre-promotion baseline.
    Called immediately before any optimizer apply() succeeds.
    """
    payload = {
        "saved_at": date.today().isoformat(),
        "promoted_configs": promoted_configs,
        **metrics,
    }
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=S3_BASELINE_KEY,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info("Promotion baseline saved to s3://%s/%s (configs: %s)",
                bucket, S3_BASELINE_KEY, promoted_configs)


def _load_baseline(bucket: str) -> dict | None:
    """Load the promotion baseline from S3. Returns None if not found."""
    s3 = boto3.client("s3")
    try:
        resp = s3.get_object(Bucket=bucket, Key=S3_BASELINE_KEY)
        return json.loads(resp["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return None
        raise


def check_regression(
    bucket: str,
    current_metrics: dict,
    config: dict | None = None,
) -> dict:
    """
    Compare current rolling metrics against the saved promotion baseline.

    Returns:
        {
            "checked": True,
            "regression_detected": bool,
            "rollback_triggered": bool,
            "details": {"accuracy_drop": float|None, "sharpe_drop_pct": float|None},
            "baseline": dict|None,
            "current": dict,
        }
    """
    baseline = _load_baseline(bucket)
    if baseline is None:
        logger.info("No promotion baseline found — skipping regression check")
        return {"checked": False, "reason": "no baseline"}

    reg_config = (config or {}).get("regression_monitor", {})
    acc_threshold = reg_config.get("accuracy_drop_threshold_pp", DEFAULT_ACCURACY_DROP_PP)
    sharpe_threshold = reg_config.get("sharpe_drop_threshold_pct", DEFAULT_SHARPE_DROP_PCT)

    details = {}
    regression_detected = False

    # Accuracy check (10d)
    base_acc = baseline.get("accuracy_10d")
    curr_acc = current_metrics.get("accuracy_10d")
    if base_acc is not None and curr_acc is not None:
        # Accuracy is 0-1 (proportion), threshold is in percentage points
        drop_pp = (base_acc - curr_acc) * 100
        details["accuracy_drop"] = drop_pp
        if drop_pp > acc_threshold:
            regression_detected = True
            logger.warning(
                "Regression: accuracy_10d dropped %.1fpp (%.1f%% -> %.1f%%), threshold=%.1fpp",
                drop_pp, base_acc * 100, curr_acc * 100, acc_threshold,
            )

    # Sharpe check
    base_sharpe = baseline.get("sharpe_ratio")
    curr_sharpe = current_metrics.get("sharpe_ratio")
    if base_sharpe is not None and curr_sharpe is not None and base_sharpe > 0:
        drop_pct = (base_sharpe - curr_sharpe) / abs(base_sharpe)
        details["sharpe_drop_pct"] = drop_pct
        if drop_pct > sharpe_threshold:
            regression_detected = True
            logger.warning(
                "Regression: sharpe dropped %.1f%% (%.4f -> %.4f), threshold=%.1f%%",
                drop_pct * 100, base_sharpe, curr_sharpe, sharpe_threshold * 100,
            )

    result = {
        "checked": True,
        "regression_detected": regression_detected,
        "rollback_triggered": False,
        "details": details,
        "baseline": baseline,
        "current": current_metrics,
    }

    if regression_detected:
        logger.warning("Regression detected — triggering auto-rollback")
        rollback_results = rollback_all(bucket)
        result["rollback_triggered"] = True
        result["rollback_results"] = rollback_results

    return result
