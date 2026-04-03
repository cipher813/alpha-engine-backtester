"""
monte_carlo.py — Permutation test for signal alpha significance.

Shuffles signal-date assignments N times and compares actual strategy alpha
to the null distribution. If the actual alpha is not in the top 5% of random
permutations, the signal is likely noise.

Uses a simplified simulation (top-N by score, equal-weight, hold 5 days)
to isolate signal quality from execution mechanics.

Data sources: score_performance table in research.db, price data from S3.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HOLD_DAYS = 5
DEFAULT_TOP_N = 5
DEFAULT_N_PERMUTATIONS = 1000


def run_monte_carlo(
    research_db_path: str,
    price_data: dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame | None = None,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    hold_days: int = DEFAULT_HOLD_DAYS,
    top_n: int = DEFAULT_TOP_N,
    min_score: float = 70.0,
    seed: int = 42,
) -> dict:
    """
    Permutation test for signal alpha significance.

    Args:
        research_db_path: path to research.db with score_performance table
        price_data: {ticker: DataFrame with 'Close' column and DatetimeIndex}
        spy_prices: DataFrame with SPY 'Close' column (for alpha computation)
        n_permutations: number of random shuffles
        hold_days: holding period for simplified simulation
        top_n: number of top-scoring signals to select per date
        min_score: minimum score threshold for signal inclusion
        seed: random seed for reproducibility

    Returns dict with:
        status, actual_alpha, p_value, percentile, null_mean, null_std,
        n_permutations, n_signals, conclusion
    """
    if not Path(research_db_path).exists():
        return {"status": "error", "error": f"research.db not found at {research_db_path}"}

    # ── Load signal history ────────────────────────────────────────────
    try:
        conn = sqlite3.connect(research_db_path)
        signals_df = pd.read_sql_query(
            "SELECT ticker, score_date, score, beat_spy_5d, return_5d, spy_return_5d "
            "FROM score_performance "
            "WHERE score IS NOT NULL AND score_date IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": f"Failed to load signals: {e}"}

    if signals_df.empty or len(signals_df) < 20:
        return {"status": "insufficient_data", "error": f"Only {len(signals_df)} signals found"}

    signals_df["score_date"] = pd.to_datetime(signals_df["score_date"])
    buy_signals = signals_df[signals_df["score"] >= min_score].copy()

    if buy_signals.empty:
        return {"status": "insufficient_data", "error": "No signals above min_score threshold"}

    signal_dates = sorted(buy_signals["score_date"].unique())
    if len(signal_dates) < 5:
        return {"status": "insufficient_data", "error": f"Only {len(signal_dates)} unique signal dates"}

    # ── Compute actual strategy alpha ──────────────────────────────────
    actual_alpha = _simulate_strategy(buy_signals, price_data, spy_prices, hold_days, top_n)
    if actual_alpha is None:
        return {"status": "error", "error": "Could not compute actual strategy alpha"}

    # ── Run permutations ───────────────────────────────────────────────
    rng = np.random.RandomState(seed)
    null_alphas = []

    for i in range(n_permutations):
        # Shuffle: keep ticker-score pairs, randomize which dates they appear on
        permuted = buy_signals.copy()
        permuted_dates = permuted["score_date"].values.copy()
        rng.shuffle(permuted_dates)
        permuted["score_date"] = permuted_dates

        perm_alpha = _simulate_strategy(permuted, price_data, spy_prices, hold_days, top_n)
        if perm_alpha is not None:
            null_alphas.append(perm_alpha)

        if (i + 1) % 100 == 0:
            logger.info("Monte Carlo: %d/%d permutations complete", i + 1, n_permutations)

    if not null_alphas:
        return {"status": "error", "error": "No permutations produced valid results"}

    null_alphas = np.array(null_alphas)
    p_value = float(np.mean(null_alphas >= actual_alpha))
    percentile = float(np.mean(null_alphas < actual_alpha) * 100)

    conclusion = "significant" if p_value < 0.05 else "not_significant"

    result = {
        "status": "ok",
        "actual_alpha": round(actual_alpha, 4),
        "p_value": round(p_value, 4),
        "percentile": round(percentile, 1),
        "null_mean": round(float(np.mean(null_alphas)), 4),
        "null_std": round(float(np.std(null_alphas)), 4),
        "null_min": round(float(np.min(null_alphas)), 4),
        "null_max": round(float(np.max(null_alphas)), 4),
        "n_permutations": len(null_alphas),
        "n_signals": len(buy_signals),
        "n_signal_dates": len(signal_dates),
        "hold_days": hold_days,
        "top_n": top_n,
        "min_score": min_score,
        "conclusion": conclusion,
    }

    logger.info(
        "Monte Carlo complete: actual_alpha=%.4f, p=%.4f, percentile=%.1f%%, conclusion=%s",
        actual_alpha, p_value, percentile, conclusion,
    )

    return result


def _simulate_strategy(
    signals: pd.DataFrame,
    price_data: dict[str, pd.DataFrame],
    spy_prices: pd.DataFrame | None,
    hold_days: int,
    top_n: int,
) -> float | None:
    """
    Simplified portfolio simulation: for each signal date, select top-N by
    score, equal-weight, hold for hold_days, compute cumulative return vs SPY.
    """
    portfolio_returns = []
    spy_returns = []

    for dt in sorted(signals["score_date"].unique()):
        day_signals = signals[signals["score_date"] == dt].nlargest(top_n, "score")

        stock_rets = []
        for _, row in day_signals.iterrows():
            ticker = row["ticker"]
            df = price_data.get(ticker)
            if df is None or df.empty:
                continue

            # Find entry date (signal date or next available)
            try:
                entry_idx = df.index.searchsorted(pd.Timestamp(dt))
                if entry_idx >= len(df):
                    continue
                exit_idx = min(entry_idx + hold_days, len(df) - 1)
                if exit_idx <= entry_idx:
                    continue
                entry_price = float(df.iloc[entry_idx]["Close"])
                exit_price = float(df.iloc[exit_idx]["Close"])
                if entry_price <= 0:
                    continue
                stock_rets.append((exit_price / entry_price) - 1.0)
            except (KeyError, IndexError):
                continue

        if stock_rets:
            portfolio_returns.append(np.mean(stock_rets))

            # SPY return for same period
            if spy_prices is not None and not spy_prices.empty:
                try:
                    spy_entry_idx = spy_prices.index.searchsorted(pd.Timestamp(dt))
                    spy_exit_idx = min(spy_entry_idx + hold_days, len(spy_prices) - 1)
                    if spy_exit_idx > spy_entry_idx:
                        spy_entry = float(spy_prices.iloc[spy_entry_idx]["Close"])
                        spy_exit = float(spy_prices.iloc[spy_exit_idx]["Close"])
                        if spy_entry > 0:
                            spy_returns.append((spy_exit / spy_entry) - 1.0)
                except (KeyError, IndexError):
                    pass

    if not portfolio_returns:
        return None

    avg_portfolio = np.mean(portfolio_returns)
    avg_spy = np.mean(spy_returns) if spy_returns else 0.0

    return float(avg_portfolio - avg_spy)
