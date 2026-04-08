"""
grade_history.py — Track component grades over time.

Appends each week's grading result to a JSON history file on S3, enabling
trend analysis of component health (is the CIO getting better or worse?).

Storage: s3://{bucket}/backtest/grade_history.json
Format: list of {date, overall, research, predictor, executor, components}
"""

import json
import logging
from datetime import date

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _load_history(bucket: str, key: str = "backtest/grade_history.json") -> list[dict]:
    """Load existing grade history from S3."""
    try:
        s3 = boto3.client("s3")
        resp = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            return []
        logger.warning("Failed to load grade history: %s", e)
        return []
    except Exception as e:
        logger.warning("Failed to load grade history: %s", e)
        return []


def _save_history(history: list[dict], bucket: str, key: str = "backtest/grade_history.json") -> None:
    """Save grade history to S3."""
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(history, indent=2, default=str).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Saved grade history (%d entries) to s3://%s/%s", len(history), bucket, key)


def _extract_component_grades(grading: dict) -> dict:
    """Extract a flat dict of component grades from a grading result."""
    grades = {}
    for mod_key in ("research", "predictor", "executor"):
        mod = grading.get(mod_key, {})
        grades[mod_key] = mod.get("grade")
        for comp_key, comp in mod.get("components", {}).items():
            if isinstance(comp, dict) and "grade" in comp:
                grades[f"{mod_key}.{comp_key}"] = comp.get("grade")
            elif isinstance(comp, list):
                # Sector teams
                for item in comp:
                    if isinstance(item, dict) and "team_id" in item:
                        grades[f"{mod_key}.team.{item['team_id']}"] = item.get("grade")
    return grades


def append_grades(grading: dict, run_date: str, bucket: str) -> dict:
    """Append this week's grades to the history file on S3.

    Args:
        grading: Result from compute_scorecard()
        run_date: ISO date string for this backtest run
        bucket: S3 bucket name

    Returns:
        {"status": "ok", "n_entries": int} or {"status": "skipped", "reason": str}
    """
    if not grading or grading.get("status") not in ("ok", "partial"):
        return {"status": "skipped", "reason": "no grading data"}

    overall = grading.get("overall", {}).get("grade")
    components = _extract_component_grades(grading)

    entry = {
        "date": run_date,
        "overall": overall,
        "research": grading.get("research", {}).get("grade"),
        "predictor": grading.get("predictor", {}).get("grade"),
        "executor": grading.get("executor", {}).get("grade"),
        "components": components,
    }

    history = _load_history(bucket)

    # Deduplicate by date (replace if same date already exists)
    history = [h for h in history if h.get("date") != run_date]
    history.append(entry)
    history.sort(key=lambda h: h.get("date", ""))

    # Keep last 52 weeks (1 year)
    if len(history) > 52:
        history = history[-52:]

    _save_history(history, bucket)

    return {"status": "ok", "n_entries": len(history)}


def load_grade_history(bucket: str) -> list[dict]:
    """Load the full grade history for dashboard display."""
    return _load_history(bucket)
