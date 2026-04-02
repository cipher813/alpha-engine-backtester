"""
macro_eval.py — Macro multiplier A/B evaluation.

Compares signal accuracy with and without the macro shift applied to
determine whether sector macro multipliers improve or hurt picking quality.

The macro shift adjusts each stock's composite score by a sector-level
modifier derived from macro analysis. This evaluation computes accuracy
on the raw score (without macro) vs. the shifted score (with macro) to
see if the shift improves the win rate.

Data source:
  - cio_evaluations in research.db (has combined_score, macro_shift, final_score)
  - universe_returns in research.db (forward returns)
  - score_performance in research.db (fallback)
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_macro_evaluation(
    db_path: str,
    score_threshold: float = 60.0,
    min_samples: int = 10,
) -> dict:
    """
    A/B comparison: accuracy with vs. without macro shift.

    Approach:
    1. Load cio_evaluations with combined_score (before macro) and final_score (after macro)
    2. Join to universe_returns for actual forward returns
    3. Simulate which stocks would be BUY with and without macro shift
    4. Compare accuracy of both sets

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        with_macro: {accuracy, avg_alpha, n}  (using final_score >= threshold)
        without_macro: {accuracy, avg_alpha, n}  (using combined_score >= threshold)
        macro_lift: accuracy difference (with - without)
        assessment: "helps" | "hurts" | "neutral"
        shift_stats: {avg_shift, max_shift, min_shift, n_positive, n_negative}
    """
    if not Path(db_path).exists():
        return {"status": "error", "error": f"DB not found at {db_path}"}

    try:
        conn = sqlite3.connect(db_path)

        # Check if cio_evaluations exists and has macro_shift
        try:
            ce = pd.read_sql_query(
                "SELECT ticker, eval_date, combined_score, macro_shift, final_score, "
                "cio_decision "
                "FROM cio_evaluations "
                "WHERE combined_score IS NOT NULL AND final_score IS NOT NULL",
                conn,
            )
        except sqlite3.OperationalError:
            conn.close()
            return {"status": "insufficient_data", "error": "cio_evaluations table not found"}

        if ce.empty or len(ce) < min_samples:
            conn.close()
            return {
                "status": "insufficient_data",
                "error": f"need >= {min_samples} CIO evaluations with scores, have {len(ce)}",
            }

        # Join with universe_returns for forward returns
        try:
            ur = pd.read_sql_query(
                "SELECT ticker, eval_date, return_5d, spy_return_5d, beat_spy_5d "
                "FROM universe_returns WHERE return_5d IS NOT NULL",
                conn,
            )
        except sqlite3.OperationalError:
            conn.close()
            return {"status": "insufficient_data", "error": "universe_returns table not found"}

        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if ur.empty:
        return {"status": "insufficient_data", "error": "universe_returns is empty"}

    merged = ce.merge(ur, on=["ticker", "eval_date"], how="inner")
    if len(merged) < min_samples:
        return {
            "status": "insufficient_data",
            "error": f"only {len(merged)} CIO rows matched to universe_returns (need {min_samples})",
        }

    merged["alpha_5d"] = merged["return_5d"] - merged["spy_return_5d"]

    # With macro: stocks with final_score >= threshold
    with_macro = merged[merged["final_score"] >= score_threshold]
    # Without macro: stocks with combined_score >= threshold
    without_macro = merged[merged["combined_score"] >= score_threshold]

    def _metrics(sub: pd.DataFrame) -> dict:
        if sub.empty:
            return {"accuracy": None, "avg_alpha": None, "n": 0}
        beat_spy = sub["beat_spy_5d"].dropna()
        acc = round(float(beat_spy.mean()), 4) if not beat_spy.empty else None
        avg_a = round(float(sub["alpha_5d"].mean()), 2)
        return {"accuracy": acc, "avg_alpha": avg_a, "n": len(sub)}

    result_with = _metrics(with_macro)
    result_without = _metrics(without_macro)

    # Lift
    lift = None
    if result_with["accuracy"] is not None and result_without["accuracy"] is not None:
        lift = round(result_with["accuracy"] - result_without["accuracy"], 4)

    alpha_lift = None
    if result_with["avg_alpha"] is not None and result_without["avg_alpha"] is not None:
        alpha_lift = round(result_with["avg_alpha"] - result_without["avg_alpha"], 2)

    # Assessment
    if lift is not None:
        if lift > 0.02:
            assessment = "helps"
        elif lift < -0.02:
            assessment = "hurts"
        else:
            assessment = "neutral"
    else:
        assessment = "insufficient_data"

    # Macro shift statistics
    shifts = merged["macro_shift"].dropna()
    shift_stats = {}
    if not shifts.empty:
        shift_stats = {
            "avg_shift": round(float(shifts.mean()), 2),
            "max_shift": round(float(shifts.max()), 2),
            "min_shift": round(float(shifts.min()), 2),
            "std_shift": round(float(shifts.std()), 2),
            "n_positive": int((shifts > 0).sum()),
            "n_negative": int((shifts < 0).sum()),
            "n_zero": int((shifts == 0).sum()),
        }

    # Stocks that changed BUY status due to macro
    promoted = merged[(merged["combined_score"] < score_threshold) & (merged["final_score"] >= score_threshold)]
    demoted = merged[(merged["combined_score"] >= score_threshold) & (merged["final_score"] < score_threshold)]

    macro_impact = {
        "n_promoted": len(promoted),
        "n_demoted": len(demoted),
    }
    if not promoted.empty:
        macro_impact["promoted_avg_alpha"] = round(float(promoted["alpha_5d"].mean()), 2)
    if not demoted.empty:
        macro_impact["demoted_avg_alpha"] = round(float(demoted["alpha_5d"].mean()), 2)

    return {
        "status": "ok",
        "with_macro": result_with,
        "without_macro": result_without,
        "accuracy_lift": lift,
        "alpha_lift": alpha_lift,
        "assessment": assessment,
        "shift_stats": shift_stats,
        "macro_impact": macro_impact,
        "n_evaluated": len(merged),
    }
