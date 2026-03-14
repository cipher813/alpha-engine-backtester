"""
synthetic/signal_generator.py — convert GBM alpha predictions to executor-compatible signals.

The executor expects signals in the same JSON envelope format produced by the
Research pipeline (signals.json). This module translates raw continuous alpha
predictions from GBMScorer into that format so the full executor pipeline
(risk guard, position sizing, ATR stops, time decay, drawdown) can run on
synthetic signals without any Research/LLM dependency.

Score mapping:
    GBM outputs small continuous alpha values (typically ±0.02 for a 5-day
    window). We centre at 50 and scale by 1000 so that a +0.02 alpha maps to
    score 70 (the executor's default min_score gate). Clamped to [0, 100].

Signal assignment:
    score >= 70 → ENTER  (strong predicted outperformance)
    score >= 40 → HOLD   (mild/neutral)
    score  < 40 → EXIT   (predicted underperformance)

Conviction:
    alpha >= 0.02 → "rising"    (strong alpha signal)
    alpha <= -0.01 → "declining" (negative alpha → half position sizing)
    else → "stable"
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Sector ETF → human-readable sector name ─────────────────────────────────
_ETF_TO_SECTOR = {
    "XLK": "Technology",
    "XLF": "Financial Services",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Cyclical",
    "XLP": "Consumer Defensive",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
    "XLB": "Basic Materials",
}


def predictions_to_signals(
    predictions: dict[str, float],
    date: str,
    sector_map: dict[str, str],
    top_n: int = 20,
    min_score: float = 70,
) -> dict:
    """
    Convert a dict of GBM alpha predictions to an executor signal envelope.

    Parameters
    ----------
    predictions : {ticker: alpha_score} — raw GBM output for one date.
    date : date string (YYYY-MM-DD) for the signal envelope.
    sector_map : {ticker: sector_etf_symbol} from sector_map.json.
    top_n : only emit ENTER signals for the top N tickers by alpha.
    min_score : minimum composite score to qualify for ENTER.

    Returns
    -------
    dict — executor-compatible signal envelope with keys:
        date, market_regime, sector_ratings, universe, buy_candidates
    """
    # Score and rank all tickers
    scored = []
    for ticker, alpha in predictions.items():
        score = _alpha_to_score(alpha)
        signal = _assign_signal(score, min_score)
        conviction = _assign_conviction(alpha)
        sector_etf = sector_map.get(ticker, "")
        sector = _ETF_TO_SECTOR.get(sector_etf, "Technology")

        scored.append({
            "ticker": ticker,
            "score": round(score, 1),
            "signal": signal,
            "conviction": conviction,
            "sector": sector,
            "rating": "BUY" if signal == "ENTER" else ("HOLD" if signal == "HOLD" else "SELL"),
            "alpha_predicted": round(alpha, 6),
        })

    # Sort by score descending to find top N candidates
    scored.sort(key=lambda s: s["score"], reverse=True)

    # Only the top_n highest-scoring tickers get ENTER; demote the rest to HOLD
    enter_count = 0
    for s in scored:
        if s["signal"] == "ENTER":
            if enter_count >= top_n:
                s["signal"] = "HOLD"
                s["rating"] = "HOLD"
            else:
                enter_count += 1

    # Build envelope
    buy_candidates = [s for s in scored if s["signal"] == "ENTER"]
    universe = [s for s in scored if s["signal"] != "ENTER"]

    # Neutral macro context — we don't have LLM macro assessment
    sector_ratings = {
        name: {"rating": "market_weight", "modifier": 0.0, "rationale": "synthetic"}
        for name in _ETF_TO_SECTOR.values()
    }

    return {
        "date": date,
        "market_regime": "neutral",
        "sector_ratings": sector_ratings,
        "buy_candidates": buy_candidates,
        "universe": universe,
    }


def _alpha_to_score(alpha: float) -> float:
    """Map continuous GBM alpha to a 0-100 composite score."""
    return max(0.0, min(100.0, 50.0 + alpha * 1000.0))


def _assign_signal(score: float, min_score: float = 70) -> str:
    """Map composite score to executor signal type."""
    if score >= min_score:
        return "ENTER"
    elif score >= 40:
        return "HOLD"
    else:
        return "EXIT"


def _assign_conviction(alpha: float) -> str:
    """Map alpha magnitude to conviction level for position sizing."""
    if alpha >= 0.02:
        return "rising"
    elif alpha <= -0.01:
        return "declining"
    else:
        return "stable"
