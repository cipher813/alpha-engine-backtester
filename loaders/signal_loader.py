"""
signal_loader.py — reads signals/{date}/signals.json from S3.

Signal file format (written by alpha-engine-research pipeline):
{
    "date": "2026-03-06",
    "signals": {
        "PLTR": {
            "ticker": "PLTR",
            "score": 82,
            "rating": "BUY",
            "signal": "ENTER",
            "conviction": "rising",
            "sector": "Technology",
            "quant_score": 85,
            "qual_score": 79,
            "sub_scores": {"quant": 85, "qual": 79}
        },
        ...
    },
    "universe": [...],
    "buy_candidates": [...],
    "market_regime": "neutral",
    "sector_ratings": {...}
}
"""

import json
import logging
from datetime import date, timedelta

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def list_dates(bucket: str, prefix: str = "signals") -> list[str]:
    """
    Return sorted list of dates (YYYY-MM-DD) that have a signals.json in S3.

    s3://{bucket}/{prefix}/{date}/signals.json
    """
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    dates = []

    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/", Delimiter="/"):
            for cp in page.get("CommonPrefixes", []):
                # cp["Prefix"] looks like "signals/2026-03-06/"
                date_str = cp["Prefix"].rstrip("/").split("/")[-1]
                if _is_valid_date(date_str):
                    dates.append(date_str)
    except ClientError as e:
        logger.error("Failed to list signal dates from s3://%s/%s/: %s", bucket, prefix, e)
        raise

    return sorted(dates)


def load(bucket: str, signal_date: str, prefix: str = "signals") -> dict:
    """
    Load signals.json for a given date from S3.

    Returns the parsed JSON dict, or raises if not found.
    Validates basic schema: must be a dict with at least one of
    'signals', 'universe', or 'population'.
    """
    key = f"{prefix}/{signal_date}/signals.json"
    s3 = boto3.client("s3")

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"No signals found at s3://{bucket}/{key}") from e
        raise

    if not isinstance(data, dict):
        raise ValueError(f"signals.json at {key} is not a dict (got {type(data).__name__})")

    has_content = (
        data.get("signals") or data.get("universe") or data.get("population")
    )
    if not has_content:
        logger.warning(
            "signals.json at %s has no 'signals', 'universe', or 'population' key — "
            "may be malformed",
            key,
        )

    logger.debug("Loaded signals for %s: %d signals", signal_date, len(data.get("signals", {})))
    return data


def load_buy_signals(bucket: str, signal_date: str, min_score: int = 0) -> list[dict]:
    """
    Convenience wrapper: load signals for a date and return only BUY-rated rows
    with score >= min_score.
    """
    data = load(bucket, signal_date)
    signals = data.get("signals", {})
    if isinstance(signals, dict):
        signals = list(signals.values())
    return [
        s for s in signals
        if s.get("rating") == "BUY" and s.get("score", 0) >= min_score
    ]


def _is_valid_date(s: str) -> bool:
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False
