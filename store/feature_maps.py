"""store/feature_maps.py — Precomputed ATR + VWAP feature maps for
backtest simulation.

Replaces per-simulate-call ArcticDB reads (executor's
``load_atr_14_pct`` + ``load_daily_vwap``) with a single bulk read at
pipeline startup. The executor's ``atr_map`` / ``vwap_map`` kwargs
(alpha-engine PR #91) accept injected maps and skip the ArcticDB
round-trip entirely when provided.

Motivation: the 2026-04-22 Saturday SF dry-run timed out at the 2h
SSM ceiling still mid-param-sweep because each ``_simulate_single_date``
call triggered 20+ ``universe.read(ticker)`` round-trips for ATR and
VWAP. 60 combos × 2000+ dates × 20 tickers = millions of reads.
py-spy confirmed the hot path. This module ships one bulk read and
in-memory resolution for the full pipeline.

Semantics mirror executor behavior:
  - ATR: last-row ``atr_14_pct`` per ticker (matches
    ``executor/price_cache.py::load_atr_14_pct``).
  - VWAP: walk back up to ``max_lookback`` trading days from the
    simulate date, first positive non-NaN value wins (matches
    ``executor/price_cache.py::load_daily_vwap``).

A known existing behavior we preserve for this PR: ``load_atr_14_pct``
in backtest mode currently uses TODAY's ATR regardless of the simulate
date (lookahead bias). This PR is perf-only and preserves that
behavior; lookahead-free ATR lands in a follow-up.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

log = logging.getLogger(__name__)

# Matches executor/price_cache.py::_ATR_MAX_STALENESS_TRADING_DAYS + VWAP
# defaults — not imported directly because executor path is sys.path-injected
# lazily and we want this module importable in isolation for tests.
_VWAP_DEFAULT_LOOKBACK = 5


def load_precomputed_feature_maps(
    bucket: str,
    max_workers: int = 20,
) -> tuple[dict[str, float], dict[str, pd.Series]]:
    """Bulk-read atr_14_pct + VWAP for every universe ticker at once.

    Returns
    -------
    atr_by_ticker : dict[ticker, float]
        Most-recent ``atr_14_pct`` per ticker. Matches the
        last-row-wins semantics of ``executor.price_cache.load_atr_14_pct``.
        Missing / non-positive values are omitted (the executor's
        ``.get(ticker)`` lookup returns None naturally).

    vwap_series_by_ticker : dict[ticker, pd.Series]
        Full VWAP time-series per ticker (pd.Series indexed by date).
        Per-simulate-date resolution is done by
        ``resolve_vwap_map_for_date`` below.

    The concurrent reads mirror
    ``loaders/price_loader.load_slim_cache``'s ThreadPoolExecutor shape
    and the existing ``store/arctic_reader`` read loop. ArcticDB is
    thread-safe for reads.

    Failure semantics: on library open failure, raises RuntimeError
    (mirrors the executor's hard-fail contract). Per-ticker read
    failures are logged + skipped; caller sees missing tickers via
    absence from the returned dicts (same shape the executor already
    tolerates for VWAP).
    """
    # Lazy import so test suites without arcticdb installed can still
    # import this module and monkey-patch. Matches the pattern in
    # executor/price_cache.py.
    import arcticdb as adb

    # URI format matches store/arctic_reader.py::_get_arctic and
    # alpha_engine_lib's arctic_uri helper — keep consistent with
    # whatever the rest of the repo uses for consistency of log grep.
    import os
    region = os.environ.get("AWS_REGION", "us-east-1")
    uri = (
        f"s3s://s3.{region}.amazonaws.com:{bucket}"
        f"?path_prefix=arcticdb&aws_auth=true"
    )
    try:
        arctic = adb.Arctic(uri)
        universe = arctic.get_library("universe")
    except Exception as exc:
        raise RuntimeError(
            f"feature_maps: ArcticDB universe library open failed "
            f"(bucket={bucket}): {exc}"
        ) from exc

    symbols = universe.list_symbols()
    log.info(
        "feature_maps: bulk-reading atr_14_pct + VWAP for %d ticker(s)",
        len(symbols),
    )

    atr_by_ticker: dict[str, float] = {}
    vwap_series_by_ticker: dict[str, pd.Series] = {}
    n_err = 0
    n_missing_atr = 0
    n_missing_vwap = 0

    def _read_one(ticker: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            df = universe.read(ticker).data
            return ticker, df, None
        except Exception as exc:
            return ticker, None, f"{exc.__class__.__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_read_one, t): t for t in symbols}
        for fut in as_completed(futures):
            ticker, df, err = fut.result()
            if err is not None:
                log.warning("feature_maps: read failed %s — %s", ticker, err)
                n_err += 1
                continue
            if df is None or df.empty:
                n_err += 1
                continue

            # ATR: last-row atr_14_pct (matches load_atr_14_pct)
            if "atr_14_pct" in df.columns:
                val = df["atr_14_pct"].iloc[-1]
                if pd.notna(val) and val > 0:
                    atr_by_ticker[ticker] = float(val)
                else:
                    n_missing_atr += 1
            else:
                n_missing_atr += 1

            # VWAP: full series (per-date resolution happens per call)
            if "VWAP" in df.columns:
                series = df["VWAP"]
                # Normalize to tz-naive for easy .loc comparison with
                # date strings in _simulate_single_date.
                if hasattr(series.index, "tz") and series.index.tz is not None:
                    series = series.copy()
                    series.index = series.index.tz_convert("UTC").tz_localize(None)
                vwap_series_by_ticker[ticker] = series
            else:
                n_missing_vwap += 1

    log.info(
        "feature_maps: loaded atr=%d, vwap=%d, missing_atr=%d, missing_vwap=%d, "
        "read_errors=%d (of %d tickers)",
        len(atr_by_ticker), len(vwap_series_by_ticker),
        n_missing_atr, n_missing_vwap, n_err, len(symbols),
    )
    return atr_by_ticker, vwap_series_by_ticker


def resolve_vwap_map_for_date(
    vwap_series_by_ticker: dict[str, pd.Series],
    tickers: list[str],
    simulate_date: str,
    max_lookback: int = _VWAP_DEFAULT_LOOKBACK,
) -> dict[str, float]:
    """Resolve VWAP per ticker at a specific simulate date.

    Walks back up to ``max_lookback`` trading days from ``simulate_date``
    and returns the first positive, non-NaN VWAP. Mirrors
    ``executor/price_cache.py::load_daily_vwap``'s per-call resolution
    semantics, but operates on in-memory pre-loaded Series instead of
    per-call ArcticDB reads.

    Tickers with no valid VWAP in the window are omitted from the
    returned dict (matches load_daily_vwap — the daemon's VWAP trigger
    explicitly handles `if vwap and vwap > 0:`).
    """
    if not tickers:
        return {}

    target = pd.Timestamp(simulate_date)
    resolved: dict[str, float] = {}

    for ticker in tickers:
        series = vwap_series_by_ticker.get(ticker)
        if series is None or series.empty:
            continue
        # Rows at or before the simulate date, most recent first after tail()
        window = series.loc[:target].tail(max_lookback).dropna()
        positive = window[window > 0]
        if not positive.empty:
            resolved[ticker] = float(positive.iloc[-1])

    return resolved
