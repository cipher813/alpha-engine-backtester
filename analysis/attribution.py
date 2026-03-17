"""
attribution.py — which sub-score (news / research) drives beat-SPY?

Computes correlation between each sub-score and beat_spy_10d/30d.
This is the primary mechanism for improving research pipeline scoring weights.

Horizon separation: Research uses news + research only (6–12 month fundamental).
Technical analysis is handled by Predictor (GBM) and Executor (ATR/time exits).

Data availability: noisy with <200 rows; meaningful at Week 8+ (~500 rows).
"""

import logging

import pandas as pd
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

SUB_SCORES = ["news", "research"]
PREDICTOR_COLS = ["p_up", "p_down", "prediction_confidence", "predicted_direction"]


def compute_attribution(df: pd.DataFrame) -> dict:
    """
    Compute correlation between sub-scores and forward return outcomes.

    Expects score_performance rows joined with sub-score columns.
    Sub-scores are assumed to be in a 'sub_scores' JSON column or as separate
    columns named news_score, research_score.

    Returns:
        {
            "status": "ok" | "insufficient_data",
            "correlations": {
                "news": {"beat_spy_10d": 0.12, "beat_spy_30d": 0.09, ...},
                "research": {...},
            },
            "ranking_10d": ["research", "news"],  # descending by correlation
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
    p_values = {}
    for label, col in sub_score_cols.items():
        corr_row = {}
        pval_row = {}
        for target in ["beat_spy_10d", "beat_spy_30d", "return_10d", "return_30d"]:
            valid = populated[[col, target]].dropna()
            if len(valid) >= 10:
                r, p = pearsonr(valid[col], valid[target])
                corr_row[target] = round(float(r), 4)
                pval_row[target] = round(float(p), 4)
            else:
                corr_row[target] = None
                pval_row[target] = None
        correlations[label] = corr_row
        p_values[label] = pval_row

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
    predictor_pvals = {}
    predictor_hit_rate = None
    predictor_hit_rate_ci = None
    if "p_up" in populated.columns and "p_down" in populated.columns:
        populated["_net_pred"] = (
            pd.to_numeric(populated["p_up"], errors="coerce").fillna(0)
            - pd.to_numeric(populated["p_down"], errors="coerce").fillna(0)
        )
        for outcome_col in ["beat_spy_10d", "beat_spy_30d"]:
            if outcome_col in populated.columns:
                valid = populated[["_net_pred", outcome_col]].dropna()
                if len(valid) >= 10:
                    r, p = pearsonr(valid["_net_pred"], valid[outcome_col])
                    predictor_corr[outcome_col] = round(float(r), 4)
                    predictor_pvals[outcome_col] = round(float(p), 4)
    if "correct_5d" in populated.columns:
        resolved = pd.to_numeric(populated["correct_5d"], errors="coerce").dropna()
        if len(resolved) >= 10:
            predictor_hit_rate = round(float(resolved.mean()), 4)
            from analysis.signal_quality import _wilson_ci
            predictor_hit_rate_ci = _wilson_ci(int(resolved.sum()), len(resolved))

    # Flag non-significant correlations
    sig_note = []
    for label, pvals in {**p_values, "predictor": predictor_pvals}.items():
        for target, p in pvals.items():
            if p is not None and p > 0.05:
                sig_note.append(f"{label}.{target} (p={p:.3f})")

    return {
        "status": "ok",
        "rows_analyzed": len(populated),
        "correlations": correlations,
        "p_values": p_values,
        "ranking_10d": ranking_10d,
        "ranking_30d": ranking_30d,
        "predictor_correlation": predictor_corr,
        "predictor_p_values": predictor_pvals,
        "predictor_hit_rate": predictor_hit_rate,
        "predictor_hit_rate_ci_95": predictor_hit_rate_ci,
        "non_significant": sig_note if sig_note else None,
        "note": (
            "Correlations include p-values; those with p > 0.05 are flagged as non-significant. "
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
