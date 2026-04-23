"""
store/arctic_reader.py — Read universe + macro data from ArcticDB for backtesting.

Connects to the shared ArcticDB instance on S3 and reads per-ticker DataFrames
containing OHLCV + 53 pre-computed features. Sole price source for the
backtester post-Phase-0 (backtester-audit-260415.md) — no S3 parquet cache,
no yfinance, no polygon, no IBKR fallback.

Usage:
    from store.arctic_reader import load_universe_from_arctic, _verify_arctic_fresh

    price_data, features_by_ticker = load_universe_from_arctic(bucket)

Failure semantics (matches predictor `load_price_data_from_arctic` pattern):
  * ArcticDB unreachable → RuntimeError (hard fail)
  * Per-ticker read error rate > 5% → RuntimeError
  * Individual ticker missing/empty → WARNING log, dropped
  * SPY missing from macro library (freshness gate) → RuntimeError
"""

from __future__ import annotations

import logging
import os
import time

import arcticdb as adb
import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_BUCKET = "alpha-engine-research"
ARCTIC_PREFIX = "arcticdb"

# Macro/ETF tickers that are not stock symbols
_MACRO_TICKERS = {"SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO", "^VIX", "^TNX", "^IRX"}
_SECTOR_ETFS = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"}
_SKIP_TICKERS = _MACRO_TICKERS | _SECTOR_ETFS

# OHLCV columns (kept separate for price_matrix / ohlcv_by_ticker)
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Per-ticker error rate threshold — matches predictor inference gate
_MAX_ERROR_RATE = 0.05


def _get_arctic(bucket: str) -> adb.Arctic:
    """Create ArcticDB connection. Raises on unreachable."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix={ARCTIC_PREFIX}&aws_auth=true"
    try:
        return adb.Arctic(uri)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB unreachable at {uri}: {exc}"
        ) from exc


def _safe_last_date(idx: pd.Index) -> pd.Timestamp | None:
    """Return the normalized last date from a DatetimeIndex, or None if empty/NaT."""
    if idx is None or idx.empty:
        return None
    last = idx.max()
    if pd.isna(last):
        return None
    return pd.Timestamp(last).normalize()


def _verify_arctic_fresh(bucket: str, min_date: str | None = None) -> None:
    """Assert ArcticDB's SPY close series has data through ``min_date``.

    Matches the predictor `_verify_arctic_fresh` pattern. If ``min_date`` is
    None, just asserts SPY exists and has at least one row.

    Raises RuntimeError on missing/stale SPY.
    """
    arctic = _get_arctic(bucket)
    try:
        macro_lib = arctic.get_library("macro")
    except Exception as exc:
        raise RuntimeError(f"ArcticDB macro library unreachable: {exc}") from exc

    try:
        df = macro_lib.read("SPY", columns=["Close"]).data
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB macro SPY unreadable: {exc} — DataPhase1 did not run "
            f"or the macro library is broken."
        ) from exc

    last_date = _safe_last_date(df.index)
    if last_date is None:
        raise RuntimeError("ArcticDB macro SPY has no rows — DataPhase1 has never written.")

    if min_date is not None:
        expected = pd.Timestamp(min_date).normalize()
        if last_date < expected:
            raise RuntimeError(
                f"ArcticDB macro SPY last_date={last_date.date()} is stale for "
                f"required date={expected.date()}."
            )


def load_universe_from_arctic(
    bucket: str = DEFAULT_BUCKET,
    tickers_allowlist: set[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load universe symbols from ArcticDB.

    Returns two dicts, both keyed by ticker:
        price_data:         {ticker: DataFrame[OHLCV]}  — for price_matrix and ohlcv_by_ticker
        features_by_ticker: {ticker: DataFrame[OHLCV + features]}  — for GBM inference

    Macro/ETF tickers (SPY, VIX, sector ETFs) are included in price_data but
    excluded from features_by_ticker. The macro tickers are ALWAYS loaded
    regardless of `tickers_allowlist` — SPY in particular is required for
    benchmark comparisons and cannot be filtered out.

    Parameters
    ----------
    tickers_allowlist :
        If provided, only stocks in this set are read from the universe
        library. Intersection with the actual ArcticDB symbol list is
        applied. Macro/ETF symbols are ALWAYS loaded regardless. Intended
        for smoke-harness fixtures where reading the full ~900-ticker
        universe dominates wall-clock (e.g. 2026-04-23 dry-run surfaced
        ~196s of bulk read for a 5-date smoke). Defaults to None (full
        universe) so production code is unchanged.

    Raises
    ------
    RuntimeError : ArcticDB unreachable, or per-ticker read error rate > 5%.
    """
    t0 = time.time()
    arctic = _get_arctic(bucket)
    try:
        universe = arctic.get_library("universe")
        macro_lib = arctic.get_library("macro")
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB library open failed on bucket {bucket}: {exc}"
        ) from exc

    symbols = universe.list_symbols()
    if tickers_allowlist is not None:
        # Intersect with the actual ArcticDB catalog so a bad allowlist
        # (typo, missing ticker) doesn't crash — we just read fewer
        # tickers. Loud log so operators see when the filter activates.
        requested = set(tickers_allowlist)
        symbols = [s for s in symbols if s in requested]
        log.info(
            "ArcticDB universe: filtered to %d symbols via tickers_allowlist "
            "(requested %d of available catalog)",
            len(symbols), len(requested),
        )
    else:
        log.info("ArcticDB universe: %d symbols", len(symbols))

    price_data: dict[str, pd.DataFrame] = {}
    features_by_ticker: dict[str, pd.DataFrame] = {}
    n_err = 0

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data
        except Exception as exc:
            log.warning("ArcticDB universe read failed for %s: %s", ticker, exc)
            n_err += 1
            continue

        if df.empty:
            log.warning("ArcticDB universe returned empty frame for %s", ticker)
            n_err += 1
            continue

        # Defensive dedup per 2026-04-15 duplicate-row workaround window
        df = df[~df.index.duplicated(keep="last")].sort_index()

        # price_data gets OHLCV columns only
        ohlcv_cols = [c for c in OHLCV_COLS if c in df.columns]
        price_data[ticker] = df[ohlcv_cols]

        # features_by_ticker gets the full DataFrame (OHLCV + features),
        # only for stock tickers (not macro/ETFs)
        if ticker not in _SKIP_TICKERS:
            features_by_ticker[ticker] = df

        if (i + 1) % 200 == 0:
            log.info("  Read %d/%d symbols from ArcticDB", i + 1, len(symbols))

    err_rate = n_err / max(len(symbols), 1)
    if err_rate > _MAX_ERROR_RATE:
        raise RuntimeError(
            f"ArcticDB per-ticker error rate {err_rate:.1%} exceeds "
            f"{_MAX_ERROR_RATE:.0%} threshold ({n_err} failed of {len(symbols)}) — "
            f"treating as pipeline failure."
        )

    # Macro + sector ETFs: try universe first (backfill writes full OHLCV),
    # fall back to macro library (daily_append writes Close only).
    all_macro_syms = ["SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO"] + sorted(_SECTOR_ETFS)
    for sym in all_macro_syms:
        if sym in price_data:
            continue  # already loaded from universe
        try:
            mdf = macro_lib.read(sym).data
        except Exception as exc:
            log.warning("ArcticDB macro read failed for %s: %s", sym, exc)
            continue
        if mdf.empty:
            log.warning("ArcticDB macro returned empty frame for %s", sym)
            continue
        mdf = mdf[~mdf.index.duplicated(keep="last")].sort_index()
        price_data[sym] = mdf

    elapsed = time.time() - t0
    log.info(
        "[data_source=arcticdb] Load complete in %.1fs: %d price tickers, %d feature tickers",
        elapsed, len(price_data), len(features_by_ticker),
    )

    return price_data, features_by_ticker
