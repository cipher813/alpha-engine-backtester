"""
veto_analysis.py — analyze Predictor veto gate effectiveness.

Sweeps confidence thresholds against historical outcomes to find the
optimal veto_confidence setting. For each threshold, measures:
- How many BUY signals would have been vetoed (predicted DOWN + high confidence)
- Whether vetoed signals actually underperformed (precision of veto)
- How much alpha was missed from false vetoes (cost of being too aggressive)

Writes recommended threshold to S3 for the Predictor Lambda to read.
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

S3_PARAMS_KEY = "config/predictor_params.json"

CONFIDENCE_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
CURRENT_DEFAULT_THRESHOLD = 0.65

# Guardrails
MIN_PREDICTIONS = 20    # minimum predictions loaded to proceed
MIN_VETO_DECISIONS = 5  # minimum vetoes at any threshold to recommend
MIN_THRESHOLD_CHANGE = 0.05  # recommended must differ from current by this much


def _load_predictions_for_dates(dates: list[str], bucket: str) -> dict:
    """
    Load predictor predictions from S3 for each date.

    Returns: {date_str: {ticker: {predicted_direction, prediction_confidence, p_up, p_down}}}
    """
    s3 = boto3.client("s3")
    predictions_by_date = {}

    for d in dates:
        key = f"predictor/predictions/{d}.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                continue
            logger.warning("S3 error loading predictions for %s: %s", d, e)
            continue

        by_ticker = {}
        for pred in data.get("predictions", []):
            ticker = pred.get("ticker")
            if ticker:
                by_ticker[ticker] = {
                    "predicted_direction": pred.get("predicted_direction"),
                    "prediction_confidence": pred.get("prediction_confidence", 0),
                    "p_up": pred.get("p_up", 0),
                    "p_down": pred.get("p_down", 0),
                }
        if by_ticker:
            predictions_by_date[d] = by_ticker

    logger.info("Loaded predictions for %d/%d dates", len(predictions_by_date), len(dates))
    return predictions_by_date


def analyze_veto_effectiveness(df: pd.DataFrame, bucket: str) -> dict:
    """
    Analyze veto gate effectiveness across confidence thresholds.

    Args:
        df: score_performance DataFrame with beat_spy_10d, return_10d columns.
        bucket: S3 bucket containing predictor/predictions/{date}.json.

    Returns:
        {
            "status": "ok" | "insufficient_data",
            "current_threshold": 0.65,
            "n_predictions_loaded": int,
            "thresholds": [{confidence, n_vetoes, true_negatives, false_negatives,
                            precision, missed_alpha}, ...],
            "recommended_threshold": float,
            "recommendation_reason": str,
        }
    """
    if df is None or df.empty:
        return {"status": "insufficient_data", "note": "No score_performance data"}

    # Only look at rows with beat_spy_10d outcome resolved
    populated = df[df["beat_spy_10d"].notna()].copy()
    if len(populated) < MIN_PREDICTIONS:
        return {
            "status": "insufficient_data",
            "n_rows": len(populated),
            "min_required": MIN_PREDICTIONS,
            "note": f"Only {len(populated)} rows with outcomes (need {MIN_PREDICTIONS})",
        }

    # Load predictions from S3
    dates = populated["score_date"].unique().tolist()
    predictions_by_date = _load_predictions_for_dates(dates, bucket)

    if not predictions_by_date:
        return {
            "status": "no_predictions",
            "note": "No predictor predictions found in S3 for the score dates",
        }

    # Join predictions with outcomes
    rows = []
    for _, row in populated.iterrows():
        d = row["score_date"]
        ticker = row["symbol"]
        preds = predictions_by_date.get(d, {}).get(ticker)
        if preds and preds.get("predicted_direction") == "DOWN":
            rows.append({
                "symbol": ticker,
                "score_date": d,
                "prediction_confidence": float(preds["prediction_confidence"]),
                "beat_spy_10d": float(row["beat_spy_10d"]),
                "return_10d": float(row.get("return_10d", 0)),
                # A signal was a BUY if score >= 65 (approximate; we use the presence
                # in score_performance as proxy for BUY-rated signals)
            })

    if not rows:
        return {
            "status": "no_down_predictions",
            "n_predictions_loaded": sum(len(v) for v in predictions_by_date.values()),
            "note": "No DOWN predictions found — veto gate has not been triggered",
        }

    down_df = pd.DataFrame(rows)
    n_down = len(down_df)
    logger.info("Found %d DOWN predictions with outcomes for veto analysis", n_down)

    # Sweep thresholds
    threshold_results = []
    for threshold in CONFIDENCE_THRESHOLDS:
        vetoed = down_df[down_df["prediction_confidence"] >= threshold]
        n_vetoes = len(vetoed)

        if n_vetoes == 0:
            threshold_results.append({
                "confidence": threshold,
                "n_vetoes": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "precision": None,
                "missed_alpha": 0.0,
            })
            continue

        # True negative = vetoed signal that actually lost (beat_spy_10d == 0)
        true_neg = int((vetoed["beat_spy_10d"] == 0).sum())
        # False negative = vetoed signal that actually won (beat_spy_10d == 1)
        false_neg = int((vetoed["beat_spy_10d"] == 1).sum())
        precision = true_neg / n_vetoes if n_vetoes > 0 else 0.0
        # Missed alpha = sum of positive returns from false negatives
        missed = float(vetoed[vetoed["beat_spy_10d"] == 1]["return_10d"].sum())

        threshold_results.append({
            "confidence": threshold,
            "n_vetoes": n_vetoes,
            "true_negatives": true_neg,
            "false_negatives": false_neg,
            "precision": round(precision, 4),
            "missed_alpha": round(missed, 4),
        })

    # Find best threshold: maximize precision while keeping missed_alpha low
    # Score = precision - 0.5 * (missed_alpha / max_missed_alpha)  [if any vetoes]
    scoreable = [t for t in threshold_results if t["n_vetoes"] >= MIN_VETO_DECISIONS]

    if not scoreable:
        return {
            "status": "insufficient_vetoes",
            "current_threshold": CURRENT_DEFAULT_THRESHOLD,
            "n_down_predictions": n_down,
            "thresholds": threshold_results,
            "note": (
                f"No threshold has {MIN_VETO_DECISIONS}+ veto decisions. "
                "Need more prediction history for reliable analysis."
            ),
        }

    max_missed = max(abs(t["missed_alpha"]) for t in scoreable) or 1.0
    for t in scoreable:
        cost_penalty = 0.5 * (abs(t["missed_alpha"]) / max_missed)
        t["_score"] = t["precision"] - cost_penalty

    best = max(scoreable, key=lambda t: t["_score"])
    recommended = best["confidence"]

    return {
        "status": "ok",
        "current_threshold": CURRENT_DEFAULT_THRESHOLD,
        "n_down_predictions": n_down,
        "n_predictions_loaded": sum(len(v) for v in predictions_by_date.values()),
        "thresholds": threshold_results,
        "recommended_threshold": recommended,
        "recommendation_reason": (
            f"Confidence {recommended:.2f}: precision {best['precision']:.1%} "
            f"with {best['missed_alpha']:.4f} missed alpha "
            f"({best['n_vetoes']} vetoes, {best['true_negatives']} correct)"
        ),
    }


def apply(result: dict, bucket: str) -> dict:
    """
    Write recommended veto threshold to S3 if guardrails pass.

    Writes to s3://{bucket}/config/predictor_params.json and archives
    to config/predictor_params_history/{date}.json.
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_threshold")
    current = result.get("current_threshold", CURRENT_DEFAULT_THRESHOLD)

    if recommended is None:
        return {"applied": False, "reason": "no recommended threshold"}

    if abs(recommended - current) < MIN_THRESHOLD_CHANGE:
        return {
            "applied": False,
            "reason": (
                f"Recommended ({recommended:.2f}) too close to current "
                f"({current:.2f}) — need {MIN_THRESHOLD_CHANGE}+ difference"
            ),
        }

    payload = {
        "veto_confidence": recommended,
        "precision": next(
            (t["precision"] for t in result.get("thresholds", [])
             if t["confidence"] == recommended), None
        ),
        "n_vetoes": next(
            (t["n_vetoes"] for t in result.get("thresholds", [])
             if t["confidence"] == recommended), None
        ),
        "updated_at": str(date.today()),
        "recommendation_reason": result.get("recommendation_reason"),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Predictor veto threshold updated in S3: %s", recommended)

    history_key = f"config/predictor_params_history/{date.today().isoformat()}.json"
    s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
    logger.info("Predictor params archived to s3://%s/%s", bucket, history_key)

    return {
        "applied": True,
        "veto_confidence": recommended,
        "previous": current,
    }
