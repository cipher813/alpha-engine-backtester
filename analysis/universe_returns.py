"""
universe_returns.py — Full-population forward-return tracking for all ~900 S&P stocks.

Uses polygon.io grouped-daily endpoint to fetch OHLCV for the entire US market
in a single API call per date. Computes 5d/10d forward returns for every ticker,
SPY benchmark returns, and sector ETF returns for sector-relative analysis.

This is the denominator for all lift calculations in the evaluation framework:
scanner filter lift, sector team lift, CIO lift, predictor lift, execution lift.

Writer: backtester (weekly batch).
Table: universe_returns in research.db.
Rows/week: ~900. Rows/year: ~47K. Size: ~20-30MB/year.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Sector ETF mapping ──────────────────────────────────────────────────────

_SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
    "Basic Materials": "XLB",
}

_ETF_TO_SECTOR = {v: k for k, v in _SECTOR_TO_ETF.items()}

_SECTOR_ETFS = set(_SECTOR_TO_ETF.values())
_SKIP_TICKERS = _SECTOR_ETFS | {"SPY", "VIX", "^VIX", "^TNX", "^IRX"}

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS universe_returns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    eval_date TEXT NOT NULL,
    sector TEXT,
    close_price REAL,
    return_5d REAL,
    return_10d REAL,
    spy_return_5d REAL,
    spy_return_10d REAL,
    beat_spy_5d INTEGER,
    beat_spy_10d INTEGER,
    sector_etf TEXT,
    sector_etf_return_5d REAL,
    beat_sector_5d INTEGER,
    UNIQUE(ticker, eval_date)
)
"""


def ensure_table(db_path: str) -> None:
    """Create universe_returns table if it doesn't exist."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.commit()
    finally:
        conn.close()


def get_existing_dates(db_path: str) -> set[str]:
    """Return set of eval_dates already populated in universe_returns."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT eval_date FROM universe_returns"
        ).fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def build_and_insert(
    db_path: str,
    eval_dates: list[str],
    polygon_client,
    sector_map: dict[str, str] | None = None,
    dry_run: bool = False,
) -> dict:
    """
    Build universe_returns rows for the given eval_dates and insert into DB.

    For each eval_date (a Monday signal date), fetches:
    - Close prices on eval_date
    - Close prices on eval_date + 5 business days
    - Close prices on eval_date + 10 business days

    Then computes forward returns and beat_spy/beat_sector flags.

    Args:
        db_path: path to research.db
        eval_dates: list of signal dates (YYYY-MM-DD) to process
        polygon_client: PolygonClient instance
        sector_map: {ticker: sector_etf_symbol} mapping (optional)
        dry_run: if True, compute but don't write to DB

    Returns:
        dict with status, dates_processed, rows_inserted, errors
    """
    ensure_table(db_path)
    existing = get_existing_dates(db_path)

    dates_to_process = [d for d in eval_dates if d not in existing]
    if not dates_to_process:
        logger.info("All %d eval_dates already in universe_returns", len(eval_dates))
        return {"status": "ok", "dates_processed": 0, "rows_inserted": 0, "skipped": len(eval_dates)}

    logger.info(
        "Processing %d eval_dates for universe_returns (%d already exist)",
        len(dates_to_process), len(existing),
    )

    total_inserted = 0
    errors = []

    for eval_date in dates_to_process:
        try:
            rows = _build_rows_for_date(eval_date, polygon_client, sector_map)
            if not rows:
                errors.append({"date": eval_date, "error": "no rows computed"})
                continue

            if not dry_run:
                inserted = _insert_rows(db_path, rows)
                total_inserted += inserted
                logger.info(
                    "universe_returns: %s → %d rows inserted",
                    eval_date, inserted,
                )
            else:
                total_inserted += len(rows)
                logger.info(
                    "universe_returns (dry-run): %s → %d rows computed",
                    eval_date, len(rows),
                )
        except Exception as e:
            logger.warning("universe_returns: failed for %s: %s", eval_date, e)
            errors.append({"date": eval_date, "error": str(e)})

    return {
        "status": "ok" if not errors else "partial",
        "dates_processed": len(dates_to_process),
        "rows_inserted": total_inserted,
        "errors": errors,
    }


def _build_rows_for_date(
    eval_date: str,
    polygon_client,
    sector_map: dict[str, str] | None,
) -> list[dict]:
    """Build universe_returns rows for a single eval_date."""
    eval_dt = date.fromisoformat(eval_date)
    fwd_5d = _add_business_days(eval_dt, 5)
    fwd_10d = _add_business_days(eval_dt, 10)

    # Check that forward dates are in the past (returns can be computed)
    today = date.today()
    if fwd_5d >= today:
        logger.debug("Skipping %s: 5d forward date %s is in the future", eval_date, fwd_5d)
        return []

    has_10d = fwd_10d < today

    # Fetch grouped-daily prices for eval_date and forward dates
    prices_t0 = polygon_client.get_grouped_daily(eval_date)
    prices_5d = polygon_client.get_grouped_daily(str(fwd_5d))
    prices_10d = polygon_client.get_grouped_daily(str(fwd_10d)) if has_10d else {}

    if not prices_t0:
        logger.warning("No prices for eval_date %s — may be a non-trading day", eval_date)
        # Try next business day
        next_day = _add_business_days(eval_dt, 1)
        prices_t0 = polygon_client.get_grouped_daily(str(next_day))
        if not prices_t0:
            return []

    # SPY benchmark
    spy_t0 = prices_t0.get("SPY", {}).get("close")
    spy_5d = prices_5d.get("SPY", {}).get("close")
    spy_10d = prices_10d.get("SPY", {}).get("close") if has_10d else None

    spy_ret_5d = _pct_return(spy_t0, spy_5d)
    spy_ret_10d = _pct_return(spy_t0, spy_10d) if has_10d else None

    # Sector ETF returns
    sector_etf_returns_5d: dict[str, float | None] = {}
    for etf in _SECTOR_ETFS:
        etf_t0 = prices_t0.get(etf, {}).get("close")
        etf_5d = prices_5d.get(etf, {}).get("close")
        sector_etf_returns_5d[etf] = _pct_return(etf_t0, etf_5d)

    # Build rows for all tickers
    rows = []
    for ticker, bar in prices_t0.items():
        if ticker in _SKIP_TICKERS:
            continue

        close_t0 = bar.get("close")
        if close_t0 is None or close_t0 <= 0:
            continue

        close_5d = prices_5d.get(ticker, {}).get("close")
        close_10d = prices_10d.get(ticker, {}).get("close") if has_10d else None

        ret_5d = _pct_return(close_t0, close_5d)
        ret_10d = _pct_return(close_t0, close_10d) if has_10d else None

        # Sector classification
        sector_etf = sector_map.get(ticker) if sector_map else None
        sector = _ETF_TO_SECTOR.get(sector_etf, "") if sector_etf else ""
        etf_ret_5d = sector_etf_returns_5d.get(sector_etf) if sector_etf else None

        rows.append({
            "ticker": ticker,
            "eval_date": eval_date,
            "sector": sector,
            "close_price": round(close_t0, 2),
            "return_5d": round(ret_5d, 4) if ret_5d is not None else None,
            "return_10d": round(ret_10d, 4) if ret_10d is not None else None,
            "spy_return_5d": round(spy_ret_5d, 4) if spy_ret_5d is not None else None,
            "spy_return_10d": round(spy_ret_10d, 4) if spy_ret_10d is not None else None,
            "beat_spy_5d": int(ret_5d > spy_ret_5d) if ret_5d is not None and spy_ret_5d is not None else None,
            "beat_spy_10d": int(ret_10d > spy_ret_10d) if ret_10d is not None and spy_ret_10d is not None else None,
            "sector_etf": sector_etf,
            "sector_etf_return_5d": round(etf_ret_5d, 4) if etf_ret_5d is not None else None,
            "beat_sector_5d": int(ret_5d > etf_ret_5d) if ret_5d is not None and etf_ret_5d is not None else None,
        })

    return rows


def _insert_rows(db_path: str, rows: list[dict]) -> int:
    """Insert rows into universe_returns, skipping duplicates."""
    conn = sqlite3.connect(db_path)
    try:
        inserted = 0
        for row in rows:
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO universe_returns "
                    "(ticker, eval_date, sector, close_price, return_5d, return_10d, "
                    "spy_return_5d, spy_return_10d, beat_spy_5d, beat_spy_10d, "
                    "sector_etf, sector_etf_return_5d, beat_sector_5d) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        row["ticker"], row["eval_date"], row["sector"],
                        row["close_price"], row["return_5d"], row["return_10d"],
                        row["spy_return_5d"], row["spy_return_10d"],
                        row["beat_spy_5d"], row["beat_spy_10d"],
                        row["sector_etf"], row["sector_etf_return_5d"],
                        row["beat_sector_5d"],
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError:
                pass
        conn.commit()
        return inserted
    finally:
        conn.close()


def load(db_path: str, eval_date: str | None = None) -> pd.DataFrame:
    """Load universe_returns from research.db, optionally filtered by eval_date."""
    conn = sqlite3.connect(db_path)
    try:
        query = "SELECT * FROM universe_returns"
        params: list = []
        if eval_date:
            query += " WHERE eval_date = ?"
            params.append(eval_date)
        query += " ORDER BY eval_date, ticker"
        return pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()


def summary(db_path: str) -> dict:
    """Return summary stats about universe_returns coverage."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as n, COUNT(DISTINCT eval_date) as n_dates, "
            "MIN(eval_date) as min_date, MAX(eval_date) as max_date, "
            "COUNT(DISTINCT ticker) as n_tickers "
            "FROM universe_returns"
        ).fetchone()
        return {
            "total_rows": row[0],
            "n_dates": row[1],
            "min_date": row[2],
            "max_date": row[3],
            "n_tickers": row[4],
        }
    finally:
        conn.close()


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pct_return(price_start: float | None, price_end: float | None) -> float | None:
    """Compute percentage return (as decimal, e.g. 0.05 = 5%)."""
    if price_start is None or price_end is None or price_start <= 0:
        return None
    return (price_end / price_start) - 1.0


def _add_business_days(start: date, n: int) -> date:
    """Add n business days to a date (skipping weekends)."""
    current = start
    added = 0
    while added < n:
        current += timedelta(days=1)
        if current.weekday() < 5:  # Mon-Fri
            added += 1
    return current
