"""
weight_optimizer.py — scoring weight recommendation based on sub-score attribution.

Joins score_performance outcomes (research.db) with sub-scores from signals.json (S3)
to compute which sub-scores (news / research) best predict outperformance.
Suggests revised weights and applies them to S3 if guardrails pass.

Horizon separation: Research uses news + research only (6–12 month fundamental
attractiveness). Technical analysis is handled by Predictor (GBM) and Executor.

Current default weights: news=0.50, research=0.50
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SUB_SCORES = ["news", "research"]
S3_WEIGHTS_KEY = "config/scoring_weights.json"

# ── Fallback defaults (override via weight_optimizer section in config.yaml) ──
_DEFAULT_WEIGHTS = {"news": 0.50, "research": 0.50}
_MAX_SINGLE_CHANGE = 0.10
_MIN_MEANINGFUL_CHANGE = 0.03
_BLEND_FACTOR = 0.20
_CONFIDENCE_LOW = 100
_CONFIDENCE_MEDIUM = 300
_HORIZON_BLEND = {"beat_spy_10d": 0.50, "beat_spy_30d": 0.50}

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load weight_optimizer section from backtester config."""
    global _cfg
    _cfg = config.get("weight_optimizer", {})


def load_with_subscores(
    df: pd.DataFrame,
    bucket: str,
    signals_prefix: str = "signals",
) -> pd.DataFrame:
    """
    Enrich a score_performance DataFrame with sub-scores from signals.json in S3.

    For each unique score_date in df, loads the corresponding signals.json and
    extracts technical/news/research sub-scores per symbol. Merges back by
    (symbol, score_date).

    Args:
        df:              score_performance DataFrame (from signal_quality.load_score_performance).
        bucket:          S3 bucket containing signals/{date}/signals.json.
        signals_prefix:  S3 prefix for signals files (default "signals").

    Returns:
        DataFrame with news_score, research_score columns added.
        Rows where sub-scores could not be resolved are kept but have NaN sub-scores.
    """
    if df.empty:
        return df

    dates = df["score_date"].unique().tolist()
    logger.info("Loading sub-scores for %d signal dates from S3...", len(dates))

    # Build lookup: {score_date: {symbol: {technical: N, news: N, research: N}}}
    subscores_by_date: dict[str, dict] = {}
    s3 = boto3.client("s3")

    for d in dates:
        key = f"{signals_prefix}/{d}/signals.json"
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                logger.debug("No signals.json for %s — sub-scores unavailable for this date", d)
            else:
                logger.warning("S3 error loading signals for %s: %s", d, e)
            continue

        by_symbol: dict[str, dict] = {}
        for stock in data.get("universe", []) + data.get("buy_candidates", []):
            ticker = stock.get("ticker") or stock.get("symbol")
            sub = stock.get("sub_scores", {})
            if ticker and sub:
                by_symbol[ticker] = {k: sub.get(k) for k in SUB_SCORES}

        subscores_by_date[d] = by_symbol
        logger.debug("Loaded sub-scores for %d symbols on %s", len(by_symbol), d)

    loaded_dates = len(subscores_by_date)
    logger.info(
        "Sub-scores loaded for %d/%d dates", loaded_dates, len(dates)
    )

    if not subscores_by_date:
        logger.warning(
            "No sub-scores found in S3. signals.json must include sub_scores per stock. "
            "Attribution and weight optimization will be skipped."
        )
        return df

    # Build a flat sub-score DataFrame and merge
    rows = []
    for d, by_symbol in subscores_by_date.items():
        for symbol, sub in by_symbol.items():
            rows.append({"symbol": symbol, "score_date": d, **{f"{k}_score": v for k, v in sub.items()}})

    sub_df = pd.DataFrame(rows)
    merged = df.merge(sub_df, on=["symbol", "score_date"], how="left")
    filled = merged[["news_score", "research_score"]].notna().any(axis=1).sum()
    logger.info("Sub-scores matched for %d/%d score_performance rows", filled, len(merged))
    return merged


def compute_weights(
    df: pd.DataFrame,
    current_weights: dict | None = None,
    min_samples: int = 30,
) -> dict:
    """
    Compute suggested scoring weights from sub-score vs. beat_spy correlations.

    Args:
        df:               score_performance DataFrame with news_score,
                          research_score columns (from load_with_subscores).
        current_weights:  Current weights dict. Defaults to DEFAULT_WEIGHTS.
        min_samples:      Minimum rows with beat_spy_10d populated to proceed.

    Returns:
        {
            "status": "ok" | "insufficient_data" | "no_subscores",
            "n_samples": int,
            "confidence": "low" | "medium" | "high",
            "current_weights": {"news": 0.50, "research": 0.50},
            "correlations": {
                "news":       {"beat_spy_10d": 0.11, "beat_spy_30d": 0.14},
                "research":   {"beat_spy_10d": 0.18, "beat_spy_30d": 0.22},
            },
            "suggested_weights": {"news": 0.48, "research": 0.52},
            "changes": {"news": -0.02, "research": +0.02},
            "note": "..."
        }
    """
    if current_weights is None:
        current_weights = _cfg.get("default_weights", _DEFAULT_WEIGHTS).copy()

    populated = df[df["beat_spy_10d"].notna()].copy()
    n = len(populated)

    if n < min_samples:
        return {
            "status": "insufficient_data",
            "n_samples": n,
            "min_required": min_samples,
            "current_weights": current_weights,
            "note": (
                f"Only {n} rows with beat_spy_10d populated "
                f"(need {min_samples}). Weight recommendation deferred."
            ),
        }

    # Check sub-score columns are present
    sub_cols = {s: f"{s}_score" for s in SUB_SCORES if f"{s}_score" in populated.columns}
    if not sub_cols:
        return {
            "status": "no_subscores",
            "n_samples": n,
            "current_weights": current_weights,
            "note": (
                "Sub-score columns not found. signals.json may not include sub_scores. "
                "Run load_with_subscores() before compute_weights()."
            ),
        }

    # Compute correlations with beat_spy outcomes
    correlations: dict[str, dict] = {}
    for label, col in sub_cols.items():
        corr: dict[str, float | None] = {}
        for target in ("beat_spy_10d", "beat_spy_30d"):
            valid = populated[[col, target]].dropna()
            corr[target] = float(valid[col].corr(valid[target])) if len(valid) >= 10 else None
        correlations[label] = corr

    # Derive suggested weights
    # Blend of 10d and 30d correlations (weights from config); clip negatives to 0
    horizon = _cfg.get("horizon_blend", _HORIZON_BLEND)
    w10 = horizon.get("beat_spy_10d", 0.50)
    w30 = horizon.get("beat_spy_30d", 0.50)
    weighted_corrs: dict[str, float] = {}
    for label, corr in correlations.items():
        c10 = corr.get("beat_spy_10d") or 0.0
        c30 = corr.get("beat_spy_30d") or 0.0
        weighted_corrs[label] = max(0.0, w10 * c10 + w30 * c30)

    total_corr = sum(weighted_corrs.values())
    if total_corr == 0:
        # No positive correlations — keep current weights
        pure_suggested = current_weights.copy()
    else:
        pure_suggested = {k: v / total_corr for k, v in weighted_corrs.items()}

    # Blend toward data-driven weights conservatively
    blend = _cfg.get("blend_factor", _BLEND_FACTOR)
    suggested = {
        k: round(current_weights.get(k, 0.0) * (1 - blend) + pure_suggested.get(k, 0.0) * blend, 3)
        for k in SUB_SCORES
    }

    # Re-normalize to ensure sum == 1.0
    total = sum(suggested.values())
    suggested = {k: round(v / total, 3) for k, v in suggested.items()}

    changes = {k: round(suggested[k] - current_weights.get(k, 0.0), 3) for k in SUB_SCORES}

    conf_med = _cfg.get("confidence_medium", _CONFIDENCE_MEDIUM)
    conf_low = _cfg.get("confidence_low", _CONFIDENCE_LOW)
    confidence = (
        "high" if n >= conf_med
        else "medium" if n >= conf_low
        else "low"
    )

    return {
        "status": "ok",
        "n_samples": n,
        "confidence": confidence,
        "current_weights": current_weights,
        "correlations": correlations,
        "suggested_weights": suggested,
        "changes": changes,
        "note": (
            f"Based on {n} signals. Confidence: {confidence}. "
            "Suggested weights blend 30% data signal with 70% current weights to avoid instability."
        ),
    }


def apply_weights(result: dict, bucket: str) -> dict:
    """
    Apply suggested weights to S3 if guardrails pass.

    Writes s3://{bucket}/config/scoring_weights.json. The research Lambda
    reads this file at cold-start and uses it in place of universe.yaml defaults.

    Guardrails (all must pass):
      - confidence must be "medium" or "high" (>= 50 samples)
      - no single weight changes by more than MAX_SINGLE_CHANGE (15%)
      - at least one weight changes by more than MIN_MEANINGFUL_CHANGE (2%)

    Args:
        result: dict returned by compute_weights().
        bucket: S3 bucket (same as signals_bucket).

    Returns:
        {"applied": True, "weights": {...}, "n_samples": int, "confidence": str}
        or {"applied": False, "reason": str}
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    confidence = result.get("confidence", "low")
    if confidence == "low":
        return {"applied": False, "reason": "confidence too low — need medium or high (50+ samples)"}

    max_single = _cfg.get("max_single_change", _MAX_SINGLE_CHANGE)
    min_meaningful = _cfg.get("min_meaningful_change", _MIN_MEANINGFUL_CHANGE)

    changes = result.get("changes", {})
    max_change = max(abs(v) for v in changes.values()) if changes else 0
    meaningful = any(abs(v) >= min_meaningful for v in changes.values())

    if max_change > max_single:
        return {
            "applied": False,
            "reason": f"largest change {max_change:.1%} exceeds {max_single:.0%} limit — skipping to avoid instability",
        }

    if not meaningful:
        return {
            "applied": False,
            "reason": f"all changes < {min_meaningful:.0%} — not worth updating",
        }

    suggested = result.get("suggested_weights", {})
    payload = {
        **suggested,
        "updated_at": str(date.today()),
        "n_samples": result.get("n_samples"),
        "confidence": confidence,
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)
    s3.put_object(
        Bucket=bucket,
        Key=S3_WEIGHTS_KEY,
        Body=body,
        ContentType="application/json",
    )
    logger.info(
        "Scoring weights updated in S3: %s (n=%s, confidence=%s)",
        suggested, payload["n_samples"], confidence,
    )

    history_key = f"config/scoring_weights_history/{date.today().isoformat()}.json"
    s3.put_object(
        Bucket=bucket,
        Key=history_key,
        Body=body,
        ContentType="application/json",
    )
    logger.info("Scoring weights archived to s3://%s/%s", bucket, history_key)

    return {
        "applied": True,
        "weights": suggested,
        "n_samples": result.get("n_samples"),
        "confidence": confidence,
    }
