"""
predictor_confusion.py — Confusion matrix for predictor directional predictions.

Computes a 3x3 confusion matrix (UP/FLAT/DOWN predicted vs actual) from
predictor_outcomes. Reveals whether the model confuses flat with directional
or reverses direction.

Actual direction is derived from actual_5d_return:
  UP:   actual_5d_return > +0.5%
  DOWN: actual_5d_return < -0.5%
  FLAT: otherwise
"""

import logging
import sqlite3

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds for mapping continuous return to directional label
_UP_THRESHOLD = 0.005    # +0.5%
_DOWN_THRESHOLD = -0.005  # -0.5%

DIRECTIONS = ["UP", "FLAT", "DOWN"]


def _actual_direction(ret: float) -> str:
    if ret > _UP_THRESHOLD:
        return "UP"
    elif ret < _DOWN_THRESHOLD:
        return "DOWN"
    return "FLAT"


def compute_confusion_matrix(db_path: str, min_samples: int = 30) -> dict:
    """Compute a 3x3 confusion matrix from predictor_outcomes.

    Returns:
        status: "ok" | "insufficient_data" | "error"
        n: total resolved predictions
        matrix: {predicted: {actual: count}} e.g. {"UP": {"UP": 40, "FLAT": 15, "DOWN": 5}}
        accuracy: overall directional accuracy
        per_class: {direction: {precision, recall, f1, n_predicted, n_actual}}
    """
    try:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT predicted_direction, actual_5d_return "
            "FROM predictor_outcomes "
            "WHERE predicted_direction IS NOT NULL AND actual_5d_return IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if len(df) < min_samples:
        return {
            "status": "insufficient_data",
            "n": len(df),
            "min_required": min_samples,
        }

    df["actual_direction"] = df["actual_5d_return"].apply(_actual_direction)

    # Build confusion matrix
    matrix = {}
    for pred in DIRECTIONS:
        matrix[pred] = {}
        for actual in DIRECTIONS:
            matrix[pred][actual] = int(
                ((df["predicted_direction"] == pred) & (df["actual_direction"] == actual)).sum()
            )

    n = len(df)
    correct = sum(matrix[d][d] for d in DIRECTIONS)
    accuracy = correct / n if n > 0 else None

    # Per-class precision, recall, F1
    per_class = {}
    for d in DIRECTIONS:
        n_predicted = sum(matrix[d][a] for a in DIRECTIONS)
        n_actual = sum(matrix[p][d] for p in DIRECTIONS)
        tp = matrix[d][d]

        precision = tp / n_predicted if n_predicted > 0 else None
        recall = tp / n_actual if n_actual > 0 else None
        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        per_class[d] = {
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "f1": round(f1, 4) if f1 is not None else None,
            "n_predicted": n_predicted,
            "n_actual": n_actual,
            "tp": tp,
        }

    return {
        "status": "ok",
        "n": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "matrix": matrix,
        "per_class": per_class,
    }
