"""
predictor_sizing_optimizer.py — recommend using p_up for position sizing.

Computes rank IC of predictor p_up vs realized 5-day returns from
predictor_outcomes. If IC is consistently positive over sufficient samples,
recommends enabling p_up-weighted position sizing in the executor.

Min-data gate: requires >= 30 resolved predictions with 5d returns.
"""

import json
import logging
import sqlite3
from datetime import date

import boto3
import pandas as pd

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/executor_params.json"

_MIN_SAMPLES = 30
_MIN_IC_TO_ENABLE = 0.05  # rank IC must exceed 0.05 to recommend p_up sizing
_MIN_POSITIVE_WEEKS = 6   # at least 6 out of 8 rolling weeks must have positive IC
_ROLLING_WEEKS = 8

_cfg: dict = {}


def init_config(config: dict) -> None:
    global _cfg
    _cfg = config.get("predictor_sizing_optimizer", {})


def analyze(research_db_path: str) -> dict:
    """
    Compute rank IC of p_up vs realized 5d returns.

    Returns dict with IC metrics and recommendation.
    """
    min_samples = _cfg.get("min_samples", _MIN_SAMPLES)
    min_ic = _cfg.get("min_ic_to_enable", _MIN_IC_TO_ENABLE)

    try:
        conn = sqlite3.connect(research_db_path)
        df = pd.read_sql_query(
            """
            SELECT prediction_date, symbol, p_up, actual_5d_return
            FROM predictor_outcomes
            WHERE p_up IS NOT NULL AND actual_5d_return IS NOT NULL
            ORDER BY prediction_date
            """,
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if len(df) < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": len(df),
            "min_required": min_samples,
        }

    # Overall rank IC
    overall_ic = float(df["p_up"].corr(df["actual_5d_return"], method="spearman"))

    # Weekly IC for rolling consistency check
    df["week"] = pd.to_datetime(df["prediction_date"]).dt.isocalendar().week.astype(int)
    df["year_week"] = (
        pd.to_datetime(df["prediction_date"]).dt.year.astype(str) + "-W"
        + df["week"].astype(str).str.zfill(2)
    )
    weekly_ic = []
    for yw, group in df.groupby("year_week"):
        if len(group) >= 5:
            ic = float(group["p_up"].corr(group["actual_5d_return"], method="spearman"))
            weekly_ic.append({"week": yw, "ic": round(ic, 4), "n": len(group)})

    positive_weeks = sum(1 for w in weekly_ic if w["ic"] > 0)
    total_weeks = len(weekly_ic)
    min_pos_weeks = _cfg.get("min_positive_weeks", _MIN_POSITIVE_WEEKS)
    rolling = _cfg.get("rolling_weeks", _ROLLING_WEEKS)

    # Use the most recent N weeks for the rolling check
    recent_weekly = weekly_ic[-rolling:] if len(weekly_ic) >= rolling else weekly_ic
    recent_positive = sum(1 for w in recent_weekly if w["ic"] > 0)
    recent_mean_ic = (
        sum(w["ic"] for w in recent_weekly) / len(recent_weekly)
        if recent_weekly else 0
    )

    should_enable = (
        overall_ic >= min_ic
        and recent_positive >= min(min_pos_weeks, len(recent_weekly))
        and recent_mean_ic >= min_ic
    )

    # Compute value-add: p_up-weighted return vs equal-weight return
    df["rank_pct"] = df.groupby("prediction_date")["p_up"].rank(pct=True)
    weighted_return = (df["rank_pct"] * df["actual_5d_return"]).mean()
    equal_weight_return = df["actual_5d_return"].mean()
    sizing_lift = weighted_return - equal_weight_return

    return {
        "status": "ok",
        "n_samples": len(df),
        "overall_rank_ic": round(overall_ic, 4),
        "recent_mean_ic": round(recent_mean_ic, 4),
        "recent_positive_weeks": recent_positive,
        "recent_total_weeks": len(recent_weekly),
        "total_positive_weeks": positive_weeks,
        "total_weeks": total_weeks,
        "sizing_lift": round(sizing_lift, 6),
        "equal_weight_return": round(equal_weight_return, 6),
        "weighted_return": round(weighted_return, 6),
        "recommendation": "enable" if should_enable else "keep_disabled",
        "weekly_ic": weekly_ic[-12:],  # last 12 weeks for report
    }


def apply(result: dict, bucket: str) -> dict:
    """Write use_p_up_sizing flag to executor_params.json on S3."""
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    if result.get("recommendation") != "enable":
        return {
            "applied": False,
            "reason": f"IC insufficient (overall={result.get('overall_rank_ic')}, "
                      f"recent_mean={result.get('recent_mean_ic')})",
        }

    s3 = boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=S3_PARAMS_KEY)
        current = json.loads(obj["Body"].read())
    except Exception:
        current = {}

    if current.get("use_p_up_sizing") is True:
        return {"applied": False, "reason": "already enabled"}

    current["use_p_up_sizing"] = True
    current["p_up_sizing_blend"] = _cfg.get("blend_factor", 0.3)
    current["p_up_sizing_updated_at"] = str(date.today())
    current["p_up_sizing_ic"] = result.get("overall_rank_ic")

    body = json.dumps(current, indent=2)
    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("p_up sizing enabled in S3 (IC=%.3f)", result.get("overall_rank_ic", 0))

    return {"applied": True, "ic": result.get("overall_rank_ic")}
