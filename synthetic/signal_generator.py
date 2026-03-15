"""
synthetic/signal_generator.py — convert GBM predictions + technical indicators
to executor-compatible signals.

Previous version used a broken alpha-to-score mapping (50 + alpha * 1000) that
clustered all scores at 45-55, producing zero ENTER signals.  This version
computes real technical scores from OHLCV price history and enriches them with
GBM alpha predictions (±10 pts max).

Score composition:
    technical_score = weighted RSI(14) + MACD + MA50 + MA200 + momentum
    trading_score   = technical_score + clip(gbm_alpha * max_enrichment, -10, +10)

Signal assignment:
    trading_score >= min_score AND top_n → ENTER
    trading_score < 30                   → EXIT
    else                                 → HOLD

Conviction (for position sizing):
    alpha >= 0.02   → "rising"
    alpha <= -0.01  → "declining"
    else            → "stable"
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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


# ── Technical scoring (same formulas as executor/technical_scorer.py) ────────
# Inlined here to avoid cross-repo import from alpha-engine.

def _score_rsi(rsi: float, market_regime: str = "neutral") -> float:
    if market_regime == "bull":
        overbought, oversold, max_os = 80, 30, 100.0
    elif market_regime in ("bear", "caution"):
        overbought, oversold, max_os = 70, 40, 65.0
    else:
        overbought, oversold, max_os = 70, 30, 100.0

    if rsi >= overbought:
        return 0.0
    if rsi <= oversold:
        return max_os
    return max_os * (overbought - rsi) / (overbought - oversold)


def _score_macd(macd_cross: float, macd_above_zero: bool) -> float:
    if macd_cross == 1.0:
        return 100.0 if macd_above_zero else 70.0
    if macd_cross == -1.0:
        return 30.0 if macd_above_zero else 0.0
    return 60.0 if macd_above_zero else 40.0


def _score_price_vs_ma(pct_diff: Optional[float]) -> float:
    if pct_diff is None:
        return 50.0
    if pct_diff >= 5:
        return min(100.0, 80.0 + (pct_diff - 5) * (20.0 / 15.0))
    if pct_diff >= 0:
        return 50.0 + pct_diff * 6.0
    if pct_diff > -5:
        return 50.0 + pct_diff * 4.0
    return max(0.0, 30.0 - (abs(pct_diff) - 5) * 1.5)


def _score_momentum(
    momentum_20d: Optional[float],
    percentile_rank: Optional[float] = None,
) -> float:
    if percentile_rank is not None:
        return float(percentile_rank)
    if momentum_20d is None:
        return 50.0
    return max(0.0, min(100.0, 50.0 + momentum_20d * 3.0))


def _compute_technical_score(
    indicators: dict,
    market_regime: str = "neutral",
    momentum_percentile: Optional[float] = None,
) -> float:
    rsi = _score_rsi(indicators.get("rsi_14", 50.0), market_regime)
    macd = _score_macd(indicators.get("macd_cross", 0.0), indicators.get("macd_above_zero", False))
    ma50 = _score_price_vs_ma(indicators.get("price_vs_ma50"))
    ma200 = _score_price_vs_ma(indicators.get("price_vs_ma200"))
    mom = _score_momentum(indicators.get("momentum_20d"), momentum_percentile)
    return round(max(0.0, min(100.0, rsi * 0.25 + macd * 0.20 + ma50 * 0.20 + ma200 * 0.20 + mom * 0.15)), 2)


def _compute_momentum_percentiles(
    momentum_data: dict[str, Optional[float]],
) -> dict[str, float]:
    valid = [(t, m) for t, m in momentum_data.items() if m is not None]
    if not valid:
        return {t: 50.0 for t in momentum_data}
    tickers, values = zip(*valid)
    arr = np.array(values, dtype=float)
    ranks = (arr.argsort().argsort() / max(len(arr) - 1, 1)) * 100
    result = {t: round(float(r), 1) for t, r in zip(tickers, ranks)}
    for t in momentum_data:
        result.setdefault(t, 50.0)
    return result


# ── Indicator computation from OHLCV ─────────────────────────────────────────

def _compute_indicators_from_ohlcv(
    price_history: list[dict],
    min_bars: int = 210,
) -> Optional[dict]:
    """
    Compute the 5 technical indicators from OHLCV bars.
    Returns None if insufficient data.
    """
    if not price_history or len(price_history) < min_bars:
        return None

    close = pd.Series([bar["close"] for bar in price_history], dtype=float)

    # RSI(14) — Wilder's smoothing
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    # MACD (12, 26, 9)
    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_above_zero = bool(macd_line.iloc[-1] > 0)

    diff = macd_line - signal_line
    macd_cross = 0.0
    if len(diff) >= 2:
        for i in range(max(len(diff) - 3, 0), len(diff)):
            if i == 0:
                continue
            if diff.iloc[i] >= 0 and diff.iloc[i - 1] < 0:
                macd_cross = 1.0
            elif diff.iloc[i] < 0 and diff.iloc[i - 1] >= 0:
                macd_cross = -1.0

    # Price vs 50-day MA
    ma50 = close.rolling(50).mean()
    if pd.isna(ma50.iloc[-1]) or ma50.iloc[-1] == 0:
        price_vs_ma50 = None
    else:
        price_vs_ma50 = ((close.iloc[-1] - ma50.iloc[-1]) / ma50.iloc[-1]) * 100

    # Price vs 200-day MA
    ma200 = close.rolling(200).mean()
    if pd.isna(ma200.iloc[-1]) or ma200.iloc[-1] == 0:
        price_vs_ma200 = None
    else:
        price_vs_ma200 = ((close.iloc[-1] - ma200.iloc[-1]) / ma200.iloc[-1]) * 100

    # 20-day momentum
    if len(close) >= 21:
        momentum_20d = float((close.iloc[-1] / close.iloc[-21]) - 1) * 100
    else:
        momentum_20d = None

    return {
        "rsi_14": rsi_14,
        "macd_cross": macd_cross,
        "macd_above_zero": macd_above_zero,
        "price_vs_ma50": price_vs_ma50,
        "price_vs_ma200": price_vs_ma200,
        "momentum_20d": momentum_20d,
    }


# ── Signal generation ─────────────────────────────────────────────────────────

def predictions_to_signals(
    predictions: dict[str, float],
    date: str,
    sector_map: dict[str, str],
    ohlcv_by_ticker: dict[str, list[dict]],
    market_regime: str = "neutral",
    top_n: int = 20,
    min_score: float = 60,
    gbm_enrichment_max: float = 10.0,
) -> dict:
    """
    Convert GBM alpha predictions + OHLCV price histories to executor signals.

    For each ticker:
    1. Compute 5 technical indicators from OHLCV bars up to this date
    2. Score via _compute_technical_score() → 0-100
    3. Enrich with GBM alpha: ±gbm_enrichment_max pts
    4. Assign signal (ENTER/EXIT/HOLD) based on trading_score

    Parameters
    ----------
    predictions : {ticker: alpha_score} from GBM inference.
    date : date string (YYYY-MM-DD).
    sector_map : {ticker: sector_etf_symbol}.
    ohlcv_by_ticker : {ticker: [{date, open, high, low, close}, ...]} —
                      bars up to (and including) this date only (no lookahead).
    market_regime : 'bull' | 'neutral' | 'caution' | 'bear'.
    top_n : max ENTER signals per date.
    min_score : minimum trading_score for ENTER.
    gbm_enrichment_max : max ±pts GBM can adjust technical score.
    """
    # Step 1: Compute technical indicators for all tickers with OHLCV data
    indicators_by_ticker: dict[str, dict] = {}
    for ticker in predictions:
        history = ohlcv_by_ticker.get(ticker)
        if history:
            ind = _compute_indicators_from_ohlcv(history)
            if ind is not None:
                indicators_by_ticker[ticker] = ind

    # Step 2: Compute momentum percentiles across all scored tickers
    momentum_data = {
        t: ind.get("momentum_20d") for t, ind in indicators_by_ticker.items()
    }
    percentiles = _compute_momentum_percentiles(momentum_data)

    # Step 3: Score each ticker
    scored = []
    for ticker, alpha in predictions.items():
        indicators = indicators_by_ticker.get(ticker)
        if indicators is None:
            # No price data — skip this ticker
            continue

        tech_score = _compute_technical_score(
            indicators,
            market_regime=market_regime,
            momentum_percentile=percentiles.get(ticker),
        )

        # GBM enrichment
        gbm_adj = max(-gbm_enrichment_max, min(gbm_enrichment_max, alpha * 500.0))
        trading_score = round(max(0.0, min(100.0, tech_score + gbm_adj)), 2)

        conviction = _assign_conviction(alpha)
        sector_etf = sector_map.get(ticker, "")
        sector = _ETF_TO_SECTOR.get(sector_etf, "Technology")

        signal = "HOLD"
        if trading_score >= min_score:
            signal = "ENTER"
        elif trading_score < 30:
            signal = "EXIT"

        scored.append({
            "ticker": ticker,
            "score": trading_score,
            "signal": signal,
            "conviction": conviction,
            "sector": sector,
            "rating": "BUY" if signal == "ENTER" else ("SELL" if signal == "EXIT" else "HOLD"),
            "technical_score": tech_score,
            "gbm_adjustment": round(gbm_adj, 2),
            "alpha_predicted": round(alpha, 6),
        })

    # Sort by score descending for top-N filtering
    scored.sort(key=lambda s: s["score"], reverse=True)

    # Cap ENTER signals at top_n
    enter_count = 0
    for s in scored:
        if s["signal"] == "ENTER":
            if enter_count >= top_n:
                s["signal"] = "HOLD"
                s["rating"] = "HOLD"
            else:
                enter_count += 1

    buy_candidates = [s for s in scored if s["signal"] == "ENTER"]
    universe = [s for s in scored if s["signal"] != "ENTER"]

    sector_ratings = {
        name: {"rating": "market_weight", "modifier": 1.0, "rationale": "synthetic"}
        for name in _ETF_TO_SECTOR.values()
    }

    return {
        "date": date,
        "market_regime": market_regime,
        "sector_ratings": sector_ratings,
        "buy_candidates": buy_candidates,
        "universe": universe,
    }


def _assign_conviction(alpha: float) -> str:
    if alpha >= 0.02:
        return "rising"
    elif alpha <= -0.01:
        return "declining"
    else:
        return "stable"
