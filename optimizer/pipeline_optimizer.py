"""
pipeline_optimizer.py — team slot allocation + CIO fallback optimization.

4b: If a sector team consistently underperforms its sector over 8+ weeks,
    reduce its slot count. Writes to config/team_slots.json on S3.

4c: If CIO underperforms a simple score-ranking baseline over 8+ weeks,
    recommend switching to deterministic CIO mode. Writes cio_mode flag
    to config/research_params.json on S3.

Both items depend on end_to_end.py lift metrics being populated.
Min-data gate: 8 weeks of team/CIO lift data.
"""

import json
import logging
from datetime import date

import boto3

logger = logging.getLogger(__name__)

S3_TEAM_SLOTS_KEY = "config/team_slots.json"
S3_RESEARCH_PARAMS_KEY = "config/research_params.json"

_MIN_WEEKS = 8
_NEGATIVE_LIFT_THRESHOLD = -0.005  # -0.5% avg lift → underperforming
_CIO_MIN_LIFT_TO_KEEP = 0.0       # CIO must have non-negative lift vs ranking

DEFAULT_TEAM_SLOTS = {
    "technology": 3,
    "healthcare": 3,
    "financial": 3,
    "consumer": 3,
    "industrial": 3,
    "defensive": 3,
}

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("pipeline_optimizer", {})


def analyze_team_performance(e2e_lift: dict) -> dict:
    """
    Analyze sector team lift and recommend slot adjustments.

    Args:
        e2e_lift: dict from end_to_end.compute_lift_metrics()

    Returns:
        dict with per-team analysis and slot recommendations.
    """
    if not e2e_lift or e2e_lift.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No lift metrics available"}

    team_lift = e2e_lift.get("team_lift")
    if not team_lift:
        return {"status": "insufficient_data", "note": "No team lift data"}

    # team_lift is a list of dicts or a dict keyed by team_id
    if isinstance(team_lift, dict):
        teams = [team_lift]
    elif isinstance(team_lift, list):
        teams = team_lift
    else:
        return {"status": "error", "note": "Unexpected team_lift format"}

    n_dates = e2e_lift.get("n_dates", 0)
    min_weeks = _cfg.get("min_weeks", _MIN_WEEKS)
    if n_dates < min_weeks:
        return {
            "status": "insufficient_data",
            "n_weeks": n_dates,
            "min_required": min_weeks,
        }

    threshold = _cfg.get("negative_lift_threshold", _NEGATIVE_LIFT_THRESHOLD)
    team_analysis = []

    for t in teams:
        team_id = t.get("team_id", "unknown")
        lift = t.get("lift")
        lift_vs_quant = t.get("lift_vs_quant")
        n_picks = t.get("n_picks", 0)

        if lift is None:
            assessment = "no_data"
            slot_change = 0
        elif lift < threshold:
            assessment = "underperforming"
            slot_change = -1
        elif lift > abs(threshold):
            assessment = "outperforming"
            slot_change = 1
        else:
            assessment = "neutral"
            slot_change = 0

        team_analysis.append({
            "team_id": team_id,
            "lift_vs_sector": lift,
            "lift_vs_quant": lift_vs_quant,
            "n_picks": n_picks,
            "assessment": assessment,
            "recommended_slot_change": slot_change,
        })

    return {
        "status": "ok",
        "n_weeks": n_dates,
        "team_analysis": team_analysis,
    }


def recommend_team_slots(analysis: dict, current_slots: dict | None = None) -> dict:
    """Generate recommended team slot allocation."""
    if analysis.get("status") != "ok":
        return {"status": analysis.get("status", "error")}

    if current_slots is None:
        current_slots = DEFAULT_TEAM_SLOTS.copy()

    recommended = dict(current_slots)
    changes = {}

    for ta in analysis.get("team_analysis", []):
        team_id = ta.get("team_id")
        change = ta.get("recommended_slot_change", 0)
        if team_id in recommended and change != 0:
            old = recommended[team_id]
            new_val = max(1, min(old + change, 5))  # clamp to [1, 5]
            if new_val != old:
                recommended[team_id] = new_val
                changes[team_id] = new_val - old

    if not changes:
        return {"status": "no_change", "current_slots": current_slots}

    return {
        "status": "ok",
        "current_slots": current_slots,
        "recommended_slots": recommended,
        "changes": changes,
    }


def apply_team_slots(result: dict, bucket: str) -> dict:
    """Write team slot allocation to S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    payload = {
        **result.get("recommended_slots", {}),
        "updated_at": str(date.today()),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_TEAM_SLOTS_KEY, Body=body, ContentType="application/json")
    logger.info("Team slots updated in S3: %s", result.get("changes"))

    return {"applied": True, "slots": result.get("recommended_slots"), "changes": result.get("changes")}


def analyze_cio_performance(e2e_lift: dict) -> dict:
    """
    Analyze CIO lift vs score-ranking baseline.

    If CIO consistently underperforms the score-ranking baseline,
    recommend switching to deterministic mode.
    """
    if not e2e_lift or e2e_lift.get("status") != "ok":
        return {"status": "insufficient_data", "note": "No lift metrics available"}

    cio_lift = e2e_lift.get("cio_lift")
    cio_vs_ranking = e2e_lift.get("cio_vs_ranking")

    n_dates = e2e_lift.get("n_dates", 0)
    min_weeks = _cfg.get("min_weeks", _MIN_WEEKS)
    if n_dates < min_weeks:
        return {
            "status": "insufficient_data",
            "n_weeks": n_dates,
            "min_required": min_weeks,
        }

    # CIO lift: ADVANCE vs all recommendations
    cio_lift_val = None
    if isinstance(cio_lift, dict):
        cio_lift_val = cio_lift.get("lift")
    elif isinstance(cio_lift, (int, float)):
        cio_lift_val = float(cio_lift)

    # CIO vs ranking baseline
    ranking_lift_val = None
    if isinstance(cio_vs_ranking, dict):
        ranking_lift_val = cio_vs_ranking.get("lift")
    elif isinstance(cio_vs_ranking, (int, float)):
        ranking_lift_val = float(cio_vs_ranking)

    min_lift = _cfg.get("cio_min_lift", _CIO_MIN_LIFT_TO_KEEP)

    # If CIO underperforms ranking baseline, recommend deterministic
    should_fallback = (
        ranking_lift_val is not None
        and ranking_lift_val < min_lift
    )

    # If CIO lift itself is negative, also recommend fallback
    if cio_lift_val is not None and cio_lift_val < _NEGATIVE_LIFT_THRESHOLD:
        should_fallback = True

    return {
        "status": "ok",
        "n_weeks": n_dates,
        "cio_lift": cio_lift_val,
        "cio_vs_ranking_lift": ranking_lift_val,
        "recommendation": "deterministic" if should_fallback else "keep_llm",
        "reasoning": (
            f"CIO lift={cio_lift_val}, vs ranking={ranking_lift_val} — "
            + ("underperforming, recommend deterministic" if should_fallback
               else "performing adequately")
        ),
    }


def apply_cio_mode(result: dict, bucket: str) -> dict:
    """Write cio_mode to research_params.json on S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    if result.get("recommendation") != "deterministic":
        return {"applied": False, "reason": "CIO performing adequately — keeping LLM mode"}

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_RESEARCH_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    if current.get("cio_mode") == "deterministic":
        return {"applied": False, "reason": "already in deterministic mode"}

    current["cio_mode"] = "deterministic"
    current["cio_mode_updated_at"] = str(date.today())
    current["cio_mode_reason"] = result.get("reasoning")

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_RESEARCH_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("CIO mode set to deterministic in S3")

    return {"applied": True, "mode": "deterministic"}
