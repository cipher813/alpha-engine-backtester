"""
exit_timing.py — Exit timing analysis via MFE/MAE.

For each completed trade (with entry and exit), computes:
  - Max Favorable Excursion (MFE): best unrealized return during hold
  - Max Adverse Excursion (MAE): worst unrealized return during hold
  - Capture ratio: realized return / MFE (are we capturing gains?)
  - Stop efficiency: |realized loss| / MAE (are stops placed well?)

Requires intraday/daily price data during the hold period. Uses yfinance
for historical prices when analyzing completed trades.

Data source: trades table in trades.db (roundtrip trades with entry_trade_id).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_exit_timing(
    trades_db_path: str,
    min_roundtrips: int = 5,
) -> dict:
    """
    Compute MFE/MAE analysis for completed roundtrip trades.

    A roundtrip is an EXIT trade linked to its ENTER via entry_trade_id.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_roundtrips: number of completed roundtrips analyzed
        summary: {avg_mfe, avg_mae, avg_capture_ratio, avg_realized_return}
        by_exit_type: [{exit_type, n, avg_mfe, avg_mae, avg_capture, avg_return}, ...]
        diagnosis: "exits_too_early" | "exits_well_timed" | "exits_too_late"
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    try:
        conn = sqlite3.connect(trades_db_path)

        exits = pd.read_sql_query(
            "SELECT e.ticker, e.date AS exit_date, e.fill_price AS exit_price, "
            "e.trigger_type AS exit_type, e.realized_return_pct, "
            "e.realized_alpha_pct, e.days_held, "
            "en.date AS entry_date, en.fill_price AS entry_price, "
            "en.signal_price "
            "FROM trades e "
            "JOIN trades en ON e.entry_trade_id = en.trade_id "
            "WHERE e.action IN ('EXIT', 'REDUCE') "
            "AND en.action = 'ENTER' "
            "AND e.fill_price IS NOT NULL "
            "AND en.fill_price IS NOT NULL",
            conn,
        )
        conn.close()
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if exits.empty or len(exits) < min_roundtrips:
        return {
            "status": "insufficient_data",
            "error": f"need >= {min_roundtrips} roundtrips, have {len(exits)}",
        }

    # Fetch price history for MFE/MAE calculation
    try:
        import yfinance as yf
    except ImportError:
        return {"status": "error", "error": "yfinance not installed"}

    tickers = exits["ticker"].unique().tolist()
    min_date = exits["entry_date"].min()
    max_date = exits["exit_date"].max()

    try:
        price_data = yf.download(
            tickers=tickers,
            start=min_date,
            end=pd.Timestamp(max_date) + pd.Timedelta(days=3),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=120,
        )
    except Exception as e:
        return {"status": "error", "error": f"yfinance download failed: {e}"}

    multi_ticker = len(tickers) > 1

    results = []
    for _, trade in exits.iterrows():
        entry_ts = pd.Timestamp(trade["entry_date"])
        exit_ts = pd.Timestamp(trade["exit_date"])

        try:
            if multi_ticker:
                highs = price_data[trade["ticker"]]["High"].loc[entry_ts:exit_ts]
                lows = price_data[trade["ticker"]]["Low"].loc[entry_ts:exit_ts]
            else:
                highs = price_data["High"].loc[entry_ts:exit_ts]
                lows = price_data["Low"].loc[entry_ts:exit_ts]
        except (KeyError, TypeError):
            continue

        if highs.empty or lows.empty:
            continue

        entry_px = trade["entry_price"]
        if entry_px is None or entry_px <= 0:
            continue

        max_high = float(highs.max())
        min_low = float(lows.min())

        mfe_pct = ((max_high - entry_px) / entry_px) * 100
        mae_pct = ((min_low - entry_px) / entry_px) * 100

        realized = trade.get("realized_return_pct")
        if realized is None:
            if trade["exit_price"] and entry_px:
                realized = ((trade["exit_price"] - entry_px) / entry_px) * 100
            else:
                continue

        capture_ratio = (realized / mfe_pct) if mfe_pct > 0.01 else None

        results.append({
            "ticker": trade["ticker"],
            "entry_date": trade["entry_date"],
            "exit_date": trade["exit_date"],
            "exit_type": trade.get("exit_type", "unknown"),
            "entry_price": entry_px,
            "exit_price": trade["exit_price"],
            "mfe_pct": round(mfe_pct, 2),
            "mae_pct": round(mae_pct, 2),
            "realized_return_pct": round(realized, 2),
            "capture_ratio": round(capture_ratio, 2) if capture_ratio is not None else None,
            "days_held": trade.get("days_held"),
        })

    if len(results) < min_roundtrips:
        return {
            "status": "insufficient_data",
            "error": f"only {len(results)} roundtrips with price data (need {min_roundtrips})",
        }

    rdf = pd.DataFrame(results)

    summary = {
        "n_roundtrips": len(rdf),
        "avg_mfe": round(float(rdf["mfe_pct"].mean()), 2),
        "avg_mae": round(float(rdf["mae_pct"].mean()), 2),
        "avg_realized_return": round(float(rdf["realized_return_pct"].mean()), 2),
        "avg_capture_ratio": round(float(rdf["capture_ratio"].dropna().mean()), 2)
        if rdf["capture_ratio"].notna().any() else None,
        "median_mfe": round(float(rdf["mfe_pct"].median()), 2),
        "median_mae": round(float(rdf["mae_pct"].median()), 2),
    }

    # Diagnosis
    avg_mfe = summary["avg_mfe"]
    avg_realized = summary["avg_realized_return"]
    capture = summary.get("avg_capture_ratio")

    if capture is not None and capture < 0.3:
        diagnosis = "exits_too_early"
    elif avg_realized > avg_mfe * 0.6:
        diagnosis = "exits_well_timed"
    elif capture is not None and capture > 0.8:
        diagnosis = "exits_well_timed"
    else:
        diagnosis = "exits_could_improve"

    # By exit type
    by_exit_type = []
    for et in sorted(rdf["exit_type"].dropna().unique()):
        grp = rdf[rdf["exit_type"] == et]
        if len(grp) < 2:
            continue
        by_exit_type.append({
            "exit_type": et,
            "n": len(grp),
            "avg_mfe": round(float(grp["mfe_pct"].mean()), 2),
            "avg_mae": round(float(grp["mae_pct"].mean()), 2),
            "avg_realized": round(float(grp["realized_return_pct"].mean()), 2),
            "avg_capture": round(float(grp["capture_ratio"].dropna().mean()), 2)
            if grp["capture_ratio"].notna().any() else None,
        })

    return {
        "status": "ok",
        "n_roundtrips": len(rdf),
        "summary": summary,
        "by_exit_type": by_exit_type,
        "diagnosis": diagnosis,
    }
