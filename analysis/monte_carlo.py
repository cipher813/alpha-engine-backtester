"""
monte_carlo.py — Permutation test for signal alpha significance.

Uses materialized forward returns from score_performance (populated upstream
by alpha-engine-data/collectors/signal_returns.py, which JOINs polygon-sourced
universe_returns onto score rows). Reading the denormalized label column means
Monte Carlo uses the exact same ground-truth return the weight_optimizer and
veto_optimizer read — no training-serving skew between the significance gate
and the promotion gates.

Null hypothesis: scores are uninformative. Shuffling scores across rows breaks
the (score → return) association; a top-N-by-permuted-score selection per date
is equivalent to a random top-N selection per date.

Units: return_Xd and spy_Xd_return are stored as percentages (2.5 = 2.5%);
output alpha fields are also in percentage units.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_TOP_N = 5
DEFAULT_N_PERMUTATIONS = 1000
_VALID_HORIZONS = ("5d", "10d", "30d")


def run_monte_carlo(
    research_db_path: str,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    top_n: int = DEFAULT_TOP_N,
    min_score: float = 70.0,
    horizon: str = "5d",
    seed: int = 42,
) -> dict:
    """
    Permutation test for signal alpha significance using materialized labels.

    Args:
        research_db_path: path to research.db with score_performance table
        n_permutations: number of random shuffles
        top_n: number of top-scoring signals to select per date
        min_score: minimum score threshold for signal inclusion
        horizon: return horizon — one of {"5d", "10d", "30d"}. Maps to
                 score_performance columns return_{horizon} + spy_{horizon}_return.
        seed: random seed for reproducibility

    Returns dict with:
        status, actual_alpha, p_value, percentile, null_mean, null_std,
        n_permutations, n_signals, n_signal_dates, horizon, top_n, min_score,
        conclusion.
    """
    if horizon not in _VALID_HORIZONS:
        return {"status": "error", "error": f"Invalid horizon {horizon!r}; expected one of {_VALID_HORIZONS}"}

    return_col = f"return_{horizon}"
    spy_col = f"spy_{horizon}_return"  # score_performance uses horizon-middle naming

    if not Path(research_db_path).exists():
        return {"status": "error", "error": f"research.db not found at {research_db_path}"}

    # Load signals with materialized forward returns
    try:
        conn = sqlite3.connect(research_db_path)
        signals_df = pd.read_sql_query(
            f"SELECT symbol AS ticker, score_date, score, "
            f"       {return_col} AS stock_return, "
            f"       {spy_col} AS spy_return "
            f"FROM score_performance "
            f"WHERE score IS NOT NULL "
            f"  AND score_date IS NOT NULL "
            f"  AND {return_col} IS NOT NULL "
            f"  AND {spy_col} IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": f"Failed to load signals: {e}"}

    if signals_df.empty or len(signals_df) < 20:
        return {
            "status": "insufficient_data",
            "error": f"Only {len(signals_df)} signals with populated {horizon} returns",
        }

    signals_df["score_date"] = pd.to_datetime(signals_df["score_date"])
    buy_signals = signals_df[signals_df["score"] >= min_score].copy()

    if buy_signals.empty:
        return {"status": "insufficient_data", "error": "No signals above min_score threshold"}

    signal_dates = sorted(buy_signals["score_date"].unique())
    if len(signal_dates) < 5:
        return {
            "status": "insufficient_data",
            "error": f"Only {len(signal_dates)} unique signal dates",
        }

    # Actual (unpermuted) strategy alpha
    actual_alpha = _compute_portfolio_alpha(buy_signals, top_n)
    if actual_alpha is None:
        return {"status": "error", "error": "Could not compute actual strategy alpha"}

    # Null distribution: shuffle scores across rows (breaks score↔return linkage)
    rng = np.random.RandomState(seed)
    null_alphas: list[float] = []
    score_pool = buy_signals["score"].to_numpy(copy=True)

    for i in range(n_permutations):
        permuted_scores = score_pool.copy()
        rng.shuffle(permuted_scores)
        permuted = buy_signals.assign(score=permuted_scores)

        perm_alpha = _compute_portfolio_alpha(permuted, top_n)
        if perm_alpha is not None:
            null_alphas.append(perm_alpha)

        if (i + 1) % 100 == 0:
            logger.info("Monte Carlo: %d/%d permutations complete", i + 1, n_permutations)

    if not null_alphas:
        return {"status": "error", "error": "No permutations produced valid results"}

    null_arr = np.array(null_alphas)
    p_value = float(np.mean(null_arr >= actual_alpha))
    percentile = float(np.mean(null_arr < actual_alpha) * 100)
    conclusion = "significant" if p_value < 0.05 else "not_significant"

    result = {
        "status": "ok",
        "actual_alpha": round(actual_alpha, 4),
        "p_value": round(p_value, 4),
        "percentile": round(percentile, 1),
        "null_mean": round(float(null_arr.mean()), 4),
        "null_std": round(float(null_arr.std()), 4),
        "null_min": round(float(null_arr.min()), 4),
        "null_max": round(float(null_arr.max()), 4),
        "n_permutations": len(null_alphas),
        "n_signals": len(buy_signals),
        "n_signal_dates": len(signal_dates),
        "horizon": horizon,
        "top_n": top_n,
        "min_score": min_score,
        "conclusion": conclusion,
    }

    logger.info(
        "Monte Carlo complete: actual_alpha=%.4f%%, p=%.4f, percentile=%.1f%%, conclusion=%s",
        actual_alpha, p_value, percentile, conclusion,
    )
    return result


def _compute_portfolio_alpha(signals: pd.DataFrame, top_n: int) -> float | None:
    """
    Per signal date: pick top-N by score, mean(stock_return) - mean(spy_return)
    is that date's realized alpha. Average across dates → overall alpha.

    Returns None if no date produces a valid portfolio.
    """
    date_alphas: list[float] = []
    for _, day_signals in signals.groupby("score_date"):
        top = day_signals.nlargest(top_n, "score")
        if top.empty:
            continue
        portfolio_return = float(top["stock_return"].mean())
        spy_return = float(top["spy_return"].mean())
        date_alphas.append(portfolio_return - spy_return)

    if not date_alphas:
        return None
    return float(np.mean(date_alphas))
