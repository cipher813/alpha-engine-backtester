"""
store/arctic_reader.py — Read universe data from ArcticDB for backtesting.

Connects to the shared ArcticDB instance on S3 and reads per-ticker
DataFrames containing OHLCV + 53 pre-computed features. This replaces
the load_full_cache_from_s3() + compute_all_features() pipeline.

Usage:
    from store.arctic_reader import load_universe_from_arctic

    price_data, features_by_ticker = load_universe_from_arctic(bucket)
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


def _get_arctic(bucket: str) -> adb.Arctic:
    """Create ArcticDB connection."""
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = f"s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix={ARCTIC_PREFIX}&aws_auth=true"
    return adb.Arctic(uri)


def load_universe_from_arctic(
    bucket: str = DEFAULT_BUCKET,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """
    Load all universe symbols from ArcticDB.

    Returns two dicts, both keyed by ticker:
        price_data:         {ticker: DataFrame[OHLCV]}  — for price_matrix and ohlcv_by_ticker
        features_by_ticker: {ticker: DataFrame[OHLCV + features]}  — for GBM inference

    Macro/ETF tickers (SPY, VIX, sector ETFs) are included in price_data
    but excluded from features_by_ticker.
    """
    t0 = time.time()
    arctic = _get_arctic(bucket)
    universe = arctic.get_library("universe")
    macro_lib = arctic.get_library("macro")

    symbols = universe.list_symbols()
    log.info("ArcticDB universe: %d symbols", len(symbols))

    price_data: dict[str, pd.DataFrame] = {}
    features_by_ticker: dict[str, pd.DataFrame] = {}

    for i, ticker in enumerate(symbols):
        try:
            df = universe.read(ticker).data

            if df.empty:
                continue

            # price_data gets OHLCV columns only
            ohlcv_cols = [c for c in OHLCV_COLS if c in df.columns]
            price_data[ticker] = df[ohlcv_cols]

            # features_by_ticker gets the full DataFrame (OHLCV + features)
            # but only for stock tickers, not macro/ETFs
            if ticker not in _SKIP_TICKERS:
                features_by_ticker[ticker] = df

        except Exception as exc:
            log.debug("Failed to read %s from ArcticDB: %s", ticker, exc)

        if (i + 1) % 200 == 0:
            log.info("  Read %d/%d symbols from ArcticDB", i + 1, len(symbols))

    # Also load macro series (SPY, VIX, etc.) into price_data
    # These are stored in the macro library with just a Close column
    for key in ["SPY", "VIX", "VIX3M", "TNX", "IRX", "GLD", "USO"]:
        if key in price_data:
            continue  # already loaded from universe
        try:
            mdf = macro_lib.read(key).data
            if not mdf.empty:
                price_data[key] = mdf
        except Exception:
            log.debug("Macro series %s not in ArcticDB", key)

    # Sector ETFs
    for sym in _SECTOR_ETFS:
        if sym in price_data:
            continue
        try:
            mdf = macro_lib.read(sym).data
            if not mdf.empty:
                price_data[sym] = mdf
        except Exception:
            pass

    elapsed = time.time() - t0
    log.info(
        "ArcticDB load complete in %.1fs: %d price tickers, %d feature tickers",
        elapsed, len(price_data), len(features_by_ticker),
    )

    return price_data, features_by_ticker
