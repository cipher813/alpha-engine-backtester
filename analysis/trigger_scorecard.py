"""
trigger_scorecard.py — Entry trigger effectiveness analysis.

For each trigger type (pullback, VWAP, support, time_expiry), computes:
  - Avg slippage vs signal price (signal_price → fill_price)
  - Avg slippage vs market open (approximated by price_at_order)
  - Avg realized alpha of trades entered via that trigger
  - Win rate (% that beat SPY)
  - Trade count

Data source: trades table in trades.db (trigger_type, signal_price,
fill_price, price_at_order, realized_alpha_pct, spy_return_during_hold).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_TRIGGER_CATEGORIES = {
    "pullback": "pullback",
    "vwap": "vwap",
    "support": "support",
    "time_expiry": "time_expiry",
}


def _categorize_trigger(trigger_type: str | None) -> str:
    """Map free-text trigger_type to a canonical category."""
    if not trigger_type:
        return "unknown"
    t = trigger_type.lower()
    for keyword, category in _TRIGGER_CATEGORIES.items():
        if keyword in t:
            return category
    return "other"


def compute_trigger_scorecard(trades_db_path: str, min_trades: int = 3) -> dict:
    """
    Compute entry trigger effectiveness metrics from trades.db.

    Args:
        trades_db_path: path to trades.db
        min_trades: minimum trades per trigger type to include

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        triggers: list of per-trigger dicts
        summary: overall stats across all triggers
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    try:
        conn = sqlite3.connect(trades_db_path)
        df = pd.read_sql_query(
            "SELECT ticker, date, action, fill_price, price_at_order, "
            "signal_price, trigger_type, trigger_price, "
            "realized_return_pct, realized_alpha_pct, "
            "spy_return_during_hold, slippage_vs_signal, days_held "
            "FROM trades WHERE action = 'ENTER'",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if df.empty:
        return {"status": "insufficient_data", "error": "no ENTER trades found"}

    df["trigger_category"] = df["trigger_type"].apply(_categorize_trigger)

    # Compute slippage metrics
    df["slippage_vs_signal_pct"] = None
    mask = df["signal_price"].notna() & df["fill_price"].notna() & (df["signal_price"] > 0)
    df.loc[mask, "slippage_vs_signal_pct"] = (
        (df.loc[mask, "fill_price"] - df.loc[mask, "signal_price"])
        / df.loc[mask, "signal_price"]
        * 100
    )

    df["slippage_vs_open_pct"] = None
    mask_open = df["price_at_order"].notna() & df["fill_price"].notna() & (df["price_at_order"] > 0)
    df.loc[mask_open, "slippage_vs_open_pct"] = (
        (df.loc[mask_open, "fill_price"] - df.loc[mask_open, "price_at_order"])
        / df.loc[mask_open, "price_at_order"]
        * 100
    )

    triggers = []
    for cat in sorted(df["trigger_category"].unique()):
        subset = df[df["trigger_category"] == cat]
        n = len(subset)
        if n < min_trades:
            continue

        # Classification: positive outcome = realized_alpha > 0
        alpha_valid = subset["realized_alpha_pct"].dropna()
        tp = int((alpha_valid > 0).sum())
        fp = int((alpha_valid <= 0).sum())

        trigger_data = {
            "trigger": cat,
            "n_trades": n,
            "avg_slippage_vs_signal": _safe_mean(subset["slippage_vs_signal_pct"]),
            "avg_slippage_vs_open": _safe_mean(subset["slippage_vs_open_pct"]),
            "avg_realized_alpha": _safe_mean(subset["realized_alpha_pct"]),
            "avg_realized_return": _safe_mean(subset["realized_return_pct"]),
            "avg_days_held": _safe_mean(subset["days_held"]),
            "win_rate_vs_spy": _win_rate(subset["realized_alpha_pct"]),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) > 0 else None,
            "tp": tp,
            "fp": fp,
        }
        triggers.append(trigger_data)

    if not triggers:
        return {
            "status": "insufficient_data",
            "error": f"no trigger types with >= {min_trades} trades",
            "total_entries": len(df),
        }

    all_alpha = df["realized_alpha_pct"].dropna()
    all_tp = int((all_alpha > 0).sum())
    all_fp = int((all_alpha <= 0).sum())

    summary = {
        "total_entries": len(df),
        "avg_slippage_vs_signal": _safe_mean(df["slippage_vs_signal_pct"]),
        "avg_slippage_vs_open": _safe_mean(df["slippage_vs_open_pct"]),
        "avg_realized_alpha": _safe_mean(df["realized_alpha_pct"]),
        "avg_realized_return": _safe_mean(df["realized_return_pct"]),
        "win_rate_vs_spy": _win_rate(df["realized_alpha_pct"]),
        "precision": round(all_tp / (all_tp + all_fp), 4) if (all_tp + all_fp) > 0 else None,
        "tp": all_tp,
        "fp": all_fp,
    }

    return {
        "status": "ok",
        "triggers": triggers,
        "summary": summary,
    }


def _safe_mean(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return round(float(valid.mean()), 4)


def _win_rate(alpha_series: pd.Series) -> float | None:
    valid = alpha_series.dropna()
    if valid.empty:
        return None
    return round(float((valid > 0).mean()), 4)
