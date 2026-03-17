"""
signal_quality.py — Mode 1: aggregate score_performance from research.db.

Reads the score_performance table (populated by the research pipeline) and
computes accuracy metrics: % of BUY signals that beat SPY at 10d and 30d.

Data availability: meaningful results require ~200 populated rows.
As of 2026-03-06, score_performance has 9 rows with beat_spy_10d = NULL.
This module will return empty results until Week 4 (~200 rows with 10d returns).
"""

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Minimum rows needed before reporting results — avoid misleading metrics on tiny samples
MIN_SAMPLES = 10


def load_score_performance(db_path: str) -> pd.DataFrame:
    """
    Load score_performance from research.db.

    Returns a DataFrame with columns:
        symbol, score_date, score, price_on_date, price_10d, price_30d,
        spy_10d_return, spy_30d_return, return_10d, return_30d,
        beat_spy_10d, beat_spy_30d, eval_date_10d, eval_date_30d
    """
    path = Path(db_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"research.db not found at {path}")

    conn = sqlite3.connect(path)
    try:
        df = pd.read_sql_query(
            "SELECT * FROM score_performance ORDER BY score_date",
            conn,
            parse_dates=["score_date", "eval_date_10d", "eval_date_30d"],
        )
    finally:
        conn.close()

    logger.info("Loaded %d rows from score_performance", len(df))
    return df


def compute_accuracy(df: pd.DataFrame, min_samples: int = MIN_SAMPLES) -> dict:
    """
    Given score_performance rows, compute accuracy metrics.

    Returns a dict with:
        - overall: {accuracy_10d, accuracy_30d, avg_alpha_10d, avg_alpha_30d, n}
        - by_score_bucket: accuracy split into [60-70, 70-80, 80-90, 90+]
        - by_conviction: accuracy split by conviction (rising/stable/declining)
        - status: "insufficient_data" if not enough rows are populated yet
    """
    populated_10d = df[df["beat_spy_10d"].notna()]
    populated_30d = df[df["beat_spy_30d"].notna()]

    if len(populated_10d) < min_samples:
        logger.warning(
            "Only %d rows with beat_spy_10d populated (need %d). "
            "Results will be available after Week 4.",
            len(populated_10d),
            min_samples,
        )
        return {
            "status": "insufficient_data",
            "rows_10d_populated": len(populated_10d),
            "rows_30d_populated": len(populated_30d),
            "rows_needed": min_samples,
        }

    result = {
        "status": "ok",
        "rows_10d_populated": len(populated_10d),
        "rows_30d_populated": len(populated_30d),
        "overall": _compute_slice_metrics(populated_10d, populated_30d),
        "by_score_bucket": _accuracy_by_score_bucket(populated_10d, populated_30d),
    }

    if "conviction" in df.columns:
        result["by_conviction"] = _accuracy_by_field(populated_10d, populated_30d, "conviction")

    return result


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return (round(max(0.0, centre - spread), 4), round(min(1.0, centre + spread), 4))


def _compute_slice_metrics(df_10d: pd.DataFrame, df_30d: pd.DataFrame) -> dict:
    n_10d = len(df_10d)
    n_30d = len(df_30d)

    acc_10d = float(df_10d["beat_spy_10d"].mean()) if n_10d > 0 else None
    acc_30d = float(df_30d["beat_spy_30d"].mean()) if n_30d > 0 else None

    # Wilson score 95% confidence intervals
    ci_10d = _wilson_ci(int(df_10d["beat_spy_10d"].sum()), n_10d) if n_10d > 0 else None
    ci_30d = _wilson_ci(int(df_30d["beat_spy_30d"].sum()), n_30d) if n_30d > 0 else None

    return {
        "accuracy_10d": acc_10d,
        "accuracy_30d": acc_30d,
        "ci_95_10d": ci_10d,
        "ci_95_30d": ci_30d,
        "avg_alpha_10d": float((df_10d["return_10d"] - df_10d["spy_10d_return"]).mean()) if n_10d > 0 else None,
        "avg_alpha_30d": float((df_30d["return_30d"] - df_30d["spy_30d_return"]).mean()) if n_30d > 0 else None,
        "n_10d": n_10d,
        "n_30d": n_30d,
    }


def _accuracy_by_score_bucket(df_10d: pd.DataFrame, df_30d: pd.DataFrame) -> list[dict]:
    buckets = [(60, 70), (70, 80), (80, 90), (90, 101)]
    rows = []
    for lo, hi in buckets:
        label = f"{lo}-{min(hi, 100)}" if hi <= 100 else f"{lo}+"
        slice_10d = df_10d[(df_10d["score"] >= lo) & (df_10d["score"] < hi)]
        slice_30d = df_30d[(df_30d["score"] >= lo) & (df_30d["score"] < hi)]
        if len(slice_10d) == 0:
            continue
        rows.append({
            "bucket": label,
            **_compute_slice_metrics(slice_10d, slice_30d),
        })
    return rows


def _accuracy_by_field(df_10d: pd.DataFrame, df_30d: pd.DataFrame, field: str) -> list[dict]:
    values = df_10d[field].dropna().unique()
    rows = []
    for val in sorted(values):
        slice_10d = df_10d[df_10d[field] == val]
        slice_30d = df_30d[df_30d[field] == val]
        rows.append({
            field: val,
            **_compute_slice_metrics(slice_10d, slice_30d),
        })
    return rows
