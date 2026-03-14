"""
attribution.py — which sub-score (technical / news / research) drives beat-SPY?

Computes correlation between each sub-score and beat_spy_10d/30d.
This is the primary mechanism for improving research pipeline scoring weights.

Data availability: noisy with <200 rows; meaningful at Week 8+ (~500 rows).
Automated weight optimization is deferred to Phase 2 (Month 6+).

The feedback loop:
    attribution output → human review (quarterly) → manual weight change in
    alpha-engine-research scoring config → backtester regression test validates change
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

SUB_SCORES = ["technical", "news", "research"]
PREDICTOR_COLS = ["p_up", "p_down", "prediction_confidence", "predicted_direction"]


def compute_attribution(df: pd.DataFrame) -> dict:
    """
    Compute correlation between sub-scores and forward return outcomes.

    Expects score_performance rows joined with sub-score columns.
    Sub-scores are assumed to be in a 'sub_scores' JSON column or as separate
    columns named technical_score, news_score, research_score.

    Returns:
        {
            "status": "ok" | "insufficient_data",
            "correlations": {
                "technical": {"beat_spy_10d": 0.12, "beat_spy_30d": 0.09, "return_10d": 0.15, ...},
                "news": {...},
                "research": {...},
            },
            "ranking_10d": ["technical", "research", "news"],  # descending by correlation
            "ranking_30d": [...],
            "note": "..."
        }
    """
    populated = df[df["beat_spy_10d"].notna()].copy()

    if len(populated) < 50:
        return {
            "status": "insufficient_data",
            "rows_populated": len(populated),
            "note": (
                f"Attribution analysis is noisy with fewer than 50 rows "
                f"(currently {len(populated)}). Meaningful results at Week 8+ (~500 rows)."
            ),
        }

    # Resolve sub-score columns
    sub_score_cols = _resolve_sub_score_columns(populated)
    if not sub_score_cols:
        return {
            "status": "no_sub_score_columns",
            "note": (
                "No sub-score columns found. Expected 'sub_scores' JSON column or "
                "separate technical_score/news_score/research_score columns."
            ),
        }

    correlations = {}
    for label, col in sub_score_cols.items():
        corr_row = {}
        for target in ["beat_spy_10d", "beat_spy_30d", "return_10d", "return_30d"]:
            valid = populated[[col, target]].dropna()
            corr_row[target] = float(valid[col].corr(valid[target])) if len(valid) >= 10 else None
        correlations[label] = corr_row

    ranking_10d = sorted(
        correlations.keys(),
        key=lambda k: correlations[k].get("beat_spy_10d") or 0,
        reverse=True,
    )
    ranking_30d = sorted(
        correlations.keys(),
        key=lambda k: correlations[k].get("beat_spy_30d") or 0,
        reverse=True,
    )

    # Predictor correlation (optional — only if predictor columns are present)
    predictor_corr = {}
    predictor_hit_rate = None
    if "p_up" in populated.columns and "p_down" in populated.columns:
        populated["_net_pred"] = (
            pd.to_numeric(populated["p_up"], errors="coerce").fillna(0)
            - pd.to_numeric(populated["p_down"], errors="coerce").fillna(0)
        )
        for outcome_col in ["beat_spy_10d", "beat_spy_30d"]:
            if outcome_col in populated.columns:
                valid = populated[["_net_pred", outcome_col]].dropna()
                if len(valid) >= 10:
                    predictor_corr[outcome_col] = float(valid["_net_pred"].corr(valid[outcome_col]))
    if "correct_5d" in populated.columns:
        resolved = pd.to_numeric(populated["correct_5d"], errors="coerce").dropna()
        if len(resolved) >= 10:
            predictor_hit_rate = float(resolved.mean())

    return {
        "status": "ok",
        "rows_analyzed": len(populated),
        "correlations": correlations,
        "ranking_10d": ranking_10d,
        "ranking_30d": ranking_30d,
        "predictor_correlation": predictor_corr,
        "predictor_hit_rate": predictor_hit_rate,
        "note": (
            "Correlations below 0.1 should be treated as noise at current sample sizes. "
            "Automated weight optimization activates at Month 6+ (500+ rows)."
        ),
    }


def _resolve_sub_score_columns(df: pd.DataFrame) -> dict[str, str]:
    """
    Find sub-score columns in the DataFrame.

    Checks for:
    1. Separate columns: technical_score, news_score, research_score
    2. Falls back to flattening a 'sub_scores' JSON column if present

    Returns dict mapping label → column_name.
    """
    explicit = {}
    for name in SUB_SCORES:
        col = f"{name}_score"
        if col in df.columns:
            explicit[name] = col

    if explicit:
        return explicit

    # Try to expand a 'sub_scores' dict column if it was loaded as objects
    if "sub_scores" in df.columns:
        try:
            expanded = pd.json_normalize(df["sub_scores"])
            for name in SUB_SCORES:
                if name in expanded.columns:
                    df[f"_attr_{name}"] = expanded[name].values
                    explicit[name] = f"_attr_{name}"
            if explicit:
                return explicit
        except Exception as e:
            logger.debug("Could not expand sub_scores column: %s", e)

    return {}
