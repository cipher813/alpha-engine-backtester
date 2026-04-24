"""
synthetic/predictor_backtest.py — predictor-only historical backtest pipeline.

Runs GBM inference on up to 10 years of OHLCV data to generate synthetic
signals, then feeds them through the full executor pipeline (risk guard,
position sizing, ATR stops, time decay, graduated drawdown).

This tests everything downstream of Research without any LLM API calls:
    1. Load OHLCV + pre-computed features from ArcticDB (sole source)
    2. Recompute features inline only when ArcticDB coverage is insufficient
    3. Run GBM inference in daily batches (up to ~2520 days × ~900 tickers)
    4. Convert alpha predictions to executor-compatible signals
    5. Build price matrix + OHLCV histories for simulation loop

The caller (backtest.py) then passes these to _run_simulation_loop() with
the existing executor pipeline.

Data source (Phase 0 of backtester-audit-260415.md):
    ArcticDB universe library — OHLCV + 53 features per ticker.
    Legacy S3 parquet cache (predictor/price_cache/*.parquet) and local
    slim-cache fallbacks were removed on 2026-04-16; ArcticDB is the
    unified source shared with predictor training + inference.

Performance notes (10y on c5.large spot):
    - Feature computation: ~900 calls to compute_features() (~3-5 min)
    - GBM inference: ~2500 batch calls (~2-3 min)
    - Total runtime: ~8-12 min
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import time

import pandas as pd

import boto3

from synthetic.signal_generator import predictions_to_signals

logger = logging.getLogger(__name__)

# Macro series tickers in the slim cache (used for feature computation)
_MACRO_TICKERS = {"SPY", "^VIX", "^TNX", "^IRX", "GLD", "USO"}

# Sector ETF tickers (present in slim cache)
_SECTOR_ETFS = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLC", "XLB"}

# Minimum OHLCV rows required for feature computation (52-week rolling windows ≈ 260 trading days + buffer)
_MIN_ROWS_FOR_FEATURES = 265


def _log_rss(label: str) -> None:
    """Log process RSS (resident set size) at a named checkpoint.

    Noisy but invaluable for catching OOM-class issues like the 2026-04-23
    SF dry-run where predictor_data_prep blew past c5.large's 4 GB budget
    inside load_universe_from_arctic + build_ohlcv_by_ticker. Without
    these checkpoints we had to diagnose via CloudWatch CPU patterns +
    SSM-agent death instead of seeing the memory curve directly.

    Uses /proc/self/status on Linux (primary target — spot instances).
    Falls back to resource.getrusage.ru_maxrss elsewhere (Darwin for
    local tests) which reports in KB on Linux but bytes on Darwin —
    the distinction doesn't matter for a diagnostic log.

    Safe to call on any platform: any failure is swallowed since this
    is pure observability and must never fail the caller."""
    try:
        rss_bytes = None
        # Linux path — /proc/self/status line "VmRSS:  N kB"
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[-1].lower() == "kb":
                            rss_bytes = int(parts[1]) * 1024
                        break
        except FileNotFoundError:
            pass
        if rss_bytes is None:
            # Darwin / non-Linux fallback
            import resource
            rusage = resource.getrusage(resource.RUSAGE_SELF)
            # ru_maxrss is KB on Linux but BYTES on Darwin — we
            # don't know which, but this path is only hit in tests.
            rss_bytes = rusage.ru_maxrss * 1024
        rss_mb = rss_bytes / (1024 * 1024)
        logger.info("MEM %s: RSS=%.0f MB", label, rss_mb)
    except Exception as exc:
        logger.debug("MEM %s: failed to sample RSS: %s", label, exc)


def load_sector_map(predictor_path: str) -> dict[str, str]:
    """Load sector_map.json mapping tickers to sector ETF symbols."""
    map_path = Path(predictor_path) / "data" / "cache" / "sector_map.json"
    if not map_path.exists():
        logger.warning("sector_map.json not found at %s", map_path)
        return {}
    with open(map_path) as f:
        return json.load(f)


def compute_all_features(
    price_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str],
    predictor_path: str,
) -> dict[str, pd.DataFrame]:
    """
    Compute 29 technical features for each stock ticker (not macro/ETF tickers).

    Features are computed once per ticker for the full 2y series, then indexed
    by date during inference. This avoids ~450K redundant compute_features()
    calls (900 tickers × 500 dates).

    Parameters
    ----------
    price_data : {ticker: OHLCV DataFrame} from load_universe_from_arctic()
    sector_map : {ticker: sector_etf} from sector_map.json
    predictor_path : path to predictor repo root (for importing compute_features)

    Returns
    -------
    {ticker: featured_df} — DataFrames with 29 feature columns + original OHLCV,
    rows with insufficient history already dropped.
    """
    # Import predictor's feature engineer
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from data.feature_engineer import compute_features

    # Extract macro series from the cache
    spy_series = _extract_close(price_data, "SPY")
    vix_series = _extract_close(price_data, "^VIX")
    tnx_series = _extract_close(price_data, "^TNX")
    irx_series = _extract_close(price_data, "^IRX")
    gld_series = _extract_close(price_data, "GLD")
    uso_series = _extract_close(price_data, "USO")

    # Stock tickers only (exclude macro/ETF series from feature computation)
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    stock_tickers = [t for t in price_data if t not in skip_tickers]

    logger.info("Computing features for %d stock tickers...", len(stock_tickers))
    features_by_ticker: dict[str, pd.DataFrame] = {}
    skip_reasons = {"too_short": 0, "empty_features": 0, "computation_error": 0}

    for i, ticker in enumerate(stock_tickers):
        df = price_data[ticker]

        if len(df) < _MIN_ROWS_FOR_FEATURES:
            skip_reasons["too_short"] += 1
            logger.debug("Skip %s: too_short (%d rows < %d)", ticker, len(df), _MIN_ROWS_FOR_FEATURES)
            continue

        # Get the sector ETF series for this ticker
        sector_etf = sector_map.get(ticker)
        sector_etf_series = _extract_close(price_data, sector_etf) if sector_etf else None

        try:
            featured = compute_features(
                df,
                spy_series=spy_series,
                vix_series=vix_series,
                sector_etf_series=sector_etf_series,
                tnx_series=tnx_series,
                irx_series=irx_series,
                gld_series=gld_series,
                uso_series=uso_series,
            )
            if not featured.empty:
                features_by_ticker[ticker] = featured
            else:
                skip_reasons["empty_features"] += 1
                logger.debug("Skip %s: empty_features after compute", ticker)
        except Exception as e:
            logger.warning("Feature computation failed for %s: %s", ticker, type(e).__name__)
            skip_reasons["computation_error"] += 1

        if (i + 1) % 100 == 0:
            logger.info("  Features computed: %d/%d tickers", i + 1, len(stock_tickers))

    skipped = sum(skip_reasons.values())
    reasons = {k: v for k, v in skip_reasons.items() if v > 0}
    logger.info(
        "Feature computation: %d tickers OK, %d skipped%s",
        len(features_by_ticker), skipped,
        f" ({reasons})" if reasons else "",
    )
    return features_by_ticker, skip_reasons


def download_gbm_model(bucket: str = "alpha-engine-research", region: str = "us-east-1") -> str:
    """
    Download the v3 Layer-1A momentum GBM from S3 to a temp file.

    Source: ``predictor/weights/meta/momentum_model.txt`` — the Layer-1A
    quant GBM that the v3 meta-model uses as an input, re-trained every
    Saturday alongside the rest of the meta stack. Saved by the current
    ``GBMScorer.save`` which persists ``feature_names`` metadata, so the
    backtester's feature-alignment check (``scorer.feature_names``) works
    cleanly.

    Why Layer-1A specifically (not the Ridge meta-model): the 10y synthetic
    backtest needs a pure quant scorer fed per-ticker features. The Ridge
    meta combines quant output with a Research calibrator whose input
    (Research composite score) only exists from ~March 2026 onward —
    replaying the full v3 stack over 10y would require fabricating research
    signals for 9.8 years of history. Scoping predictor-backtest to Layer
    1A measures the quant component in isolation, which per
    feedback_component_baseline_validation is the right standalone
    baseline for a stacked ensemble.

    Previously loaded ``predictor/weights/gbm_latest.txt`` — a v2 artifact
    last updated 2026-03-28, ripped from production 2026-04-13. Every
    Saturday since has been measuring a dead model. Cleanup of the stale
    v2 S3 artifacts is tracked in ROADMAP P2 "v2 legacy artifact cleanup".

    Returns the local path to the downloaded model.
    """
    s3 = boto3.client("s3", region_name=region)

    model_key = "predictor/weights/meta/momentum_model.txt"
    meta_key = "predictor/weights/meta/momentum_model.txt.meta.json"

    model_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    model_tmp.close()
    try:
        s3.download_file(bucket, model_key, model_tmp.name)
        logger.info("Downloaded Layer-1A momentum GBM from s3://%s/%s", bucket, model_key)
    except Exception as exc:
        raise RuntimeError(
            f"Layer-1A momentum GBM not found at s3://{bucket}/{model_key}. "
            "Saturday PredictorTraining step must populate predictor/weights/"
            "meta/momentum_model.txt on each run — investigate the training "
            f"pipeline if this key is missing. Underlying error: {exc}"
        ) from exc

    # Download metadata — hard-fail if missing. The backtester hard-requires
    # feature_names from the meta.json for input alignment; a successful
    # download of the booster with no meta.json would crash downstream in a
    # less useful place.
    meta_path = model_tmp.name + ".meta.json"
    try:
        s3.download_file(bucket, meta_key, meta_path)
    except Exception as exc:
        raise RuntimeError(
            f"Layer-1A momentum GBM metadata not found at s3://{bucket}/"
            f"{meta_key}. feature_names alignment will fail without it. "
            f"Underlying error: {exc}"
        ) from exc

    return model_tmp.name


def run_inference(
    features_by_ticker: dict[str, pd.DataFrame],
    model_path: str,
    predictor_path: str,
    trading_dates: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Run GBM inference for all tickers across all trading dates.

    For each trading date, stacks feature vectors for all tickers with valid
    features on that date, runs one batch GBMScorer.predict() call, and
    returns predictions indexed by date.

    Parameters
    ----------
    features_by_ticker : {ticker: featured_df} from compute_all_features()
    model_path : local path to GBM model file
    predictor_path : path to predictor repo root (for importing GBMScorer)
    trading_dates : optional list of dates to run inference on. If None,
        uses the union of all available feature dates.

    Returns
    -------
    {date_str: {ticker: alpha_score}} — predictions per date per ticker.
    """
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from model.gbm_scorer import GBMScorer

    scorer = GBMScorer.load(model_path)

    # Use the model's own trained feature list rather than the current
    # GBM_FEATURES config — they drift when new features land in config.py
    # before a fresh training run promotes weights. Slicing by the model's
    # feature_names guarantees the input matrix matches regardless of config
    # drift; a fresh training run will update this list automatically.
    GBM_FEATURES = scorer.feature_names
    if not GBM_FEATURES:
        raise RuntimeError(
            "Loaded model has no feature_names metadata — cannot align "
            "input features. Retrain with a newer GBMScorer that persists "
            "feature_names in the metadata JSON."
        )
    logger.info("Predictor backtest using %d model features: %s",
                len(GBM_FEATURES), GBM_FEATURES)

    # Determine trading dates from feature data if not provided
    if trading_dates is None:
        all_dates = set()
        for df in features_by_ticker.values():
            all_dates.update(df.index.strftime("%Y-%m-%d"))
        trading_dates = sorted(all_dates)

    logger.info(
        "Running GBM inference: %d tickers × %d dates",
        len(features_by_ticker), len(trading_dates),
    )

    # Pre-extract feature vectors into {ticker: {date_str: numpy_array}}
    # This avoids ~2M slow pandas .loc[] lookups in the inner loop.
    logger.info("Pre-extracting feature vectors...")
    feature_arrays: dict[str, dict[str, np.ndarray]] = {}
    for ticker, featured_df in features_by_ticker.items():
        try:
            arr = featured_df[GBM_FEATURES].to_numpy(dtype=np.float32)
            dates = featured_df.index.strftime("%Y-%m-%d")
            # Handle duplicate dates: last value wins (same as iloc[-1])
            feature_arrays[ticker] = dict(zip(dates, arr))
        except (KeyError, ValueError):
            continue
    logger.info("Pre-extracted vectors for %d tickers", len(feature_arrays))

    predictions_by_date: dict[str, dict[str, float]] = {}

    for i, date_str in enumerate(trading_dates):
        # Collect feature vectors for all tickers on this date
        tickers_batch = []
        vectors_batch = []

        for ticker, date_to_vec in feature_arrays.items():
            vec = date_to_vec.get(date_str)
            if vec is not None and not np.any(np.isnan(vec)):
                tickers_batch.append(ticker)
                vectors_batch.append(vec)

        if not vectors_batch:
            continue

        # Batch predict
        X = np.stack(vectors_batch)  # shape (N, 29)
        alphas = scorer.predict(X)   # shape (N,)

        predictions_by_date[date_str] = {
            ticker: float(alpha)
            for ticker, alpha in zip(tickers_batch, alphas)
        }

        if (i + 1) % 50 == 0:
            logger.info("  Inference: %d/%d dates", i + 1, len(trading_dates))

    logger.info(
        "Inference complete: %d dates with predictions",
        len(predictions_by_date),
    )
    return predictions_by_date


def build_signals_by_date(
    predictions_by_date: dict[str, dict[str, float]],
    sector_map: dict[str, str],
    ohlcv_by_ticker: dict[str, list[dict]],
    top_n: int = 20,
    min_score: float = 60,
) -> dict[str, dict]:
    """
    Convert per-date predictions to executor signal envelopes using
    technical scoring from OHLCV data (not the broken alpha-to-score mapping).

    Parameters
    ----------
    predictions_by_date : {date: {ticker: alpha}} from run_inference()
    sector_map : {ticker: sector_etf} from sector_map.json
    ohlcv_by_ticker : {ticker: [{date, open, high, low, close}, ...]}
    top_n : max ENTER signals per day (prevents unrealistic portfolio churn)
    min_score : minimum trading score for ENTER signal

    Returns
    -------
    {date: signal_envelope} — each envelope is a full signals_override dict.

    Performance notes
    -----------------
    Pre-2026-04-21 implementation ran at ~2.2s per date × 2277 dates ≈ 75 min,
    which pushed the Saturday SF past its 7200s SSM ceiling. The bottleneck
    was the inner loop rebuilding ``ohlcv_up_to_date`` by scanning every
    ticker's full 10y bar list per date (~5B Python string comparisons
    total). The data already has a date axis; pandas can roll every
    indicator in one vectorized pass per ticker.

    This revision: one-shot ``precompute_indicator_series(ohlcv_by_ticker)``
    produces per-ticker date-indexed DataFrames of all 6 indicators. The
    per-date loop then does O(1) hashtable lookups via
    ``indicators_from_precomputed``. Expected speedup ~50-100x (verified
    on synthetic + production data).
    """
    from synthetic.signal_generator import (
        precompute_indicator_series,
        indicators_from_precomputed,
    )

    signals_by_date: dict[str, dict] = {}
    sorted_dates = sorted(predictions_by_date.keys())

    # One-shot vectorized indicator pass over the full history per ticker.
    logger.info(
        "  Precomputing indicator series for %d tickers (vectorized)...",
        len(ohlcv_by_ticker),
    )
    t_pre = time.time()
    precomputed = precompute_indicator_series(ohlcv_by_ticker)
    logger.info(
        "  Precompute complete: %d tickers indexed in %.1fs",
        len(precomputed), time.time() - t_pre,
    )

    for i, date_str in enumerate(sorted_dates):
        predictions = predictions_by_date[date_str]

        # O(1) hashtable lookup per ticker — the hot path that was an
        # O(bars) list-comp scan before.
        indicators_this_date = indicators_from_precomputed(
            precomputed, predictions.keys(), date_str,
        )

        envelope = predictions_to_signals(
            predictions=predictions,
            date=date_str,
            sector_map=sector_map,
            precomputed_indicators=indicators_this_date,
            top_n=top_n,
            min_score=min_score,
        )
        signals_by_date[date_str] = envelope

        if (i + 1) % 250 == 0:
            n_enter = len(envelope.get("buy_candidates", []))
            logger.info(
                "  Signal generation: %d/%d dates (ENTER=%d on %s)",
                i + 1, len(sorted_dates), n_enter, date_str,
            )

    return signals_by_date


def build_price_matrix(
    price_data: dict[str, pd.DataFrame],
    trading_dates: list[str],
) -> pd.DataFrame:
    """
    Build a price matrix from slim cache data (same format as price_loader.build_matrix).

    Returns DataFrame with DatetimeIndex (dates) and ticker columns,
    values are close prices.
    """
    # Only include stock tickers (not macro/ETF series)
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    stock_tickers = [t for t in price_data if t not in skip_tickers]

    records = {}
    for ticker in stock_tickers:
        df = price_data[ticker]
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            continue
        close = df[close_col]
        records[ticker] = close

    matrix = pd.DataFrame(records)
    # Filter to trading dates only
    matrix.index = pd.to_datetime(matrix.index)
    date_index = pd.to_datetime(trading_dates)
    matrix = matrix.reindex(date_index)

    logger.info(
        "Price matrix: %d dates × %d tickers (%.1f%% fill)",
        len(matrix), len(matrix.columns),
        matrix.notna().sum().sum() / max(matrix.size, 1) * 100,
    )
    return matrix


def build_ohlcv_by_ticker(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, list[dict]]:
    """
    Convert DataFrames to the {ticker: [{date, open, high, low, close}, ...]}
    format needed by _run_simulation_loop's price_histories parameter.

    Non-destructive — callers that own `price_data` and can afford to
    retain it use this. Memory-sensitive callers should use
    `_drain_price_data_into_ohlcv` instead.
    """
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    ohlcv: dict[str, list[dict]] = {}

    for ticker, df in price_data.items():
        if ticker in skip_tickers:
            continue
        bars = _df_to_bars(df)
        if bars:
            ohlcv[ticker] = bars

    return ohlcv


def _df_to_bars(df: pd.DataFrame) -> list[dict]:
    """Convert a single ticker's OHLCV DataFrame to the list-of-dicts form."""
    bars = []
    for dt, row in df.iterrows():
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        close = row.get("Close", row.get("close", 0))
        bars.append({
            "date": date_str,
            "open": row.get("Open", row.get("open", close)),
            "high": row.get("High", row.get("high", close)),
            "low": row.get("Low", row.get("low", close)),
            "close": close,
        })
    return bars


def build_ohlcv_df_by_ticker(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Produce the DataFrame-form ohlcv_by_ticker — the new shape for the
    backtester memory refactor (plan 2026-04-23). Each value is a
    pd.DataFrame with:

      - DatetimeIndex (sorted ascending, no duplicates)
      - Columns [open, high, low, close], dtype float64

    Normalizes ArcticDB's capitalized column names to lowercase at the
    producer boundary so downstream consumers (executor, indicator
    compute, artifact persistence) can rely on a single canonical
    column naming without per-site case handling.

    Missing OHL columns (frequent in thin-history tickers where only
    Close is populated) fall back to the Close series — matches the
    existing list-of-dicts producer's ``_df_to_bars`` semantic.

    Macro + sector ETFs are filtered at the producer boundary — they
    aren't used downstream in the list form either.

    This is the low-overhead shape (~91 MB for 911 tickers × 2500 bars
    vs. ~1.1 GB for the equivalent list-of-dicts, where Python dict
    header overhead dominates at ~240 B/row). Kills the backtester
    OOM-on-c5.large risk diagnosed in the 2026-04-23 root-cause
    analysis (see alpha-engine-backtester-pandas-refactor-plan-260423.md).
    """
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    out: dict[str, pd.DataFrame] = {}
    for ticker, df in price_data.items():
        if ticker in skip_tickers:
            continue
        if df is None or df.empty:
            continue
        cols: dict[str, pd.Series] = {}
        for title in ("Close", "Open", "High", "Low"):
            lower = title.lower()
            if title in df.columns:
                cols[lower] = df[title]
            elif lower in df.columns:
                cols[lower] = df[lower]
        if "close" not in cols:
            continue
        close = cols["close"]
        for key in ("open", "high", "low"):
            if key not in cols:
                cols[key] = close
        frame = pd.DataFrame({
            "open":  cols["open"],
            "high":  cols["high"],
            "low":   cols["low"],
            "close": cols["close"],
        }).astype(float)
        frame.index = pd.to_datetime(frame.index)
        frame = frame[~frame.index.duplicated(keep="last")].sort_index()
        out[ticker] = frame
    return out


def _df_slice_to_bars(df: pd.DataFrame, until_date: str) -> list[dict]:
    """Materialize the <= until_date slice of a single ticker's OHLCV
    DataFrame as the list-of-dicts form the executor still consumes.

    ``until_date`` is an ISO8601 string. The slice is inclusive of the
    final bar whose DatetimeIndex entry is <= ``until_date``, matching
    the historic ``[b for b in bars if b["date"] <= signal_date]``
    semantic that _simulate_single_date's bisect path replaced on
    2026-04-22. Used at the executor boundary once the producer flips
    to DataFrame form — the executor itself still expects list-of-dicts
    during Option A's coexistence window.
    """
    ts = pd.Timestamp(until_date)
    sliced = df.loc[:ts]
    bars: list[dict] = []
    for dt, row in sliced.iterrows():
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        bars.append({
            "date":  date_str,
            "open":  float(row["open"]),
            "high":  float(row["high"]),
            "low":   float(row["low"]),
            "close": float(row["close"]),
        })
    return bars


def _drain_price_data_into_ohlcv(
    price_data: dict[str, pd.DataFrame],
) -> dict[str, list[dict]]:
    """Destructively pop each price_data[ticker] into ohlcv_by_ticker, so
    both dicts never hold full-universe data simultaneously.

    After this returns, `price_data` is empty (all entries popped).

    Motivated by the 2026-04-23 OOM incident where coexisting full-
    universe price_data (~91 MB of DataFrames) + ohlcv_by_ticker
    (~1.1 GB of list-of-dicts due to Python overhead) pushed past
    c5.large's 4 GB budget during predictor_data_prep. Saves ~91 MB at
    the peak. The larger win lives in a follow-up PR that changes
    ohlcv_by_ticker's shape itself (list-of-dicts → pd.DataFrame) —
    that's a cross-repo contract change and tracked as P2 in
    SYSTEM_STATE's backtester section.
    """
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    ohlcv: dict[str, list[dict]] = {}
    tickers = list(price_data.keys())  # snapshot so we can mutate dict
    for ticker in tickers:
        df = price_data.pop(ticker)
        if ticker in skip_tickers:
            # skip_tickers' rows aren't needed downstream — drop silently
            continue
        bars = _df_to_bars(df)
        if bars:
            ohlcv[ticker] = bars
    # price_data is now empty; caller can safely `del price_data`
    return ohlcv


def _extract_close(price_data: dict[str, pd.DataFrame], ticker: str | None) -> pd.Series | None:
    """Extract Close price series for a given ticker, or None if not found."""
    if ticker is None or ticker not in price_data:
        return None
    df = price_data[ticker]
    if "Close" in df.columns:
        return df["Close"]
    elif "close" in df.columns:
        return df["close"]
    return None


def _resolve_trading_dates(
    features_by_ticker: dict[str, pd.DataFrame],
    min_trading_days: int,
    max_trading_days: int,
) -> list[str] | dict:
    """Determine common trading dates from feature data.

    Returns sorted date list on success, or error dict if insufficient dates.
    """
    all_dates = set()
    for df in features_by_ticker.values():
        all_dates.update(df.index.strftime("%Y-%m-%d"))
    trading_dates = sorted(all_dates)

    if len(trading_dates) < min_trading_days:
        return {
            "status": "insufficient_data",
            "dates_available": len(trading_dates),
            "min_required": min_trading_days,
            "note": f"Only {len(trading_dates)} trading dates with features "
                    f"(need {min_trading_days})",
        }

    if len(trading_dates) > max_trading_days:
        trading_dates = trading_dates[-max_trading_days:]
        logger.info(
            "Trimmed to most recent %d trading dates (from %s to %s)",
            len(trading_dates), trading_dates[0], trading_dates[-1],
        )
    else:
        logger.info(
            "Trading dates: %d (from %s to %s)",
            len(trading_dates), trading_dates[0], trading_dates[-1],
        )

    return trading_dates


def run(config: dict, keep_features: bool = False) -> dict:
    """
    Full predictor-only backtest pipeline.

    Steps:
        1. Resolve predictor path and load slim cache
        2. Load sector map
        3. Compute features for all stock tickers
        4. Download GBM model from S3
        5. Run inference across all trading dates
        6. Generate synthetic signals
        7. Build price matrix and OHLCV histories

    Returns a dict with all data needed by backtest.py's simulation loop:
        - signals_by_date: {date: signal_envelope}
        - price_matrix: DataFrame
        - ohlcv_by_ticker: {ticker: [{date, open, high, low, close}, ...]}
        - metadata: {n_tickers, n_dates, date_range, ...}
    """
    # Resolve predictor path
    predictor_paths = config.get("predictor_paths", [])
    if isinstance(predictor_paths, str):
        predictor_paths = [predictor_paths]
    predictor_path = next((p for p in predictor_paths if os.path.isdir(p)), None)
    if not predictor_path:
        raise ValueError(
            f"None of the predictor_paths exist: {predictor_paths}. "
            "Add the alpha-engine-predictor repo root to predictor_paths in config.yaml."
        )

    pb_config = config.get("predictor_backtest", {})
    min_trading_days = pb_config.get("min_trading_days", 252)
    max_trading_days = pb_config.get("max_trading_days", 500)
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # 1. Load price data + features from ArcticDB (sole source post-Phase-0).
    #    Hard-fail on unreachable per backtester-audit-260415.md: legacy S3
    #    parquet cache + inline slim-cache fallbacks have been removed.
    from store.arctic_reader import load_universe_from_arctic
    _log_rss("pre_arcticdb_load")
    logger.info("[data_source=arcticdb] Loading universe from ArcticDB...")
    # Smoke fixture universe filter — production default is None (full
    # universe load). When smoke_tickers is set, reader restricts the
    # stock-symbol read; macro/ETF symbols (SPY etc.) always load.
    _smoke_tickers = config.get("smoke_tickers")
    _allowlist = set(_smoke_tickers) if _smoke_tickers else None
    price_data, features_by_ticker = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=_allowlist,
    )
    data_source = "arcticdb"
    feature_skip_reasons: dict = {}
    logger.info("[data_source=arcticdb] %d tickers with pre-computed features", len(features_by_ticker))
    _log_rss("post_arcticdb_load")

    # 2. Load sector map
    sector_map = load_sector_map(predictor_path)

    # 3. Inline feature recompute is only hit when ArcticDB's feature coverage
    #    is insufficient for the requested backtest window (e.g., 10y synthetic
    #    backtest running before the feature schema was backfilled). In practice
    #    load_universe_from_arctic returns a non-empty dict for every stock
    #    ticker in the universe library; this branch is the safety net.
    if not features_by_ticker:
        logger.warning("ArcticDB returned no pre-computed features — recomputing inline from OHLCV")
        features_by_ticker, feature_skip_reasons = compute_all_features(price_data, sector_map, predictor_path)

    if not features_by_ticker:
        return {
            "status": "error",
            "error": "No tickers had sufficient data for feature computation",
            "tickers_loaded": len(price_data),
            "skip_reasons": feature_skip_reasons,
        }

    # 3b. Resolve trading dates
    trading_dates = _resolve_trading_dates(features_by_ticker, min_trading_days, max_trading_days)
    if isinstance(trading_dates, dict):
        return trading_dates  # early exit with error dict

    # 4. Build price matrix, extract SPY, drain price_data into ohlcv_by_ticker.
    #    2026-04-23 SF dry-run OOM'd on c5.large because price_data (~91 MB) and
    #    ohlcv_by_ticker (~1.1 GB, dominated by Python dict overhead in the
    #    list-of-dicts form) coexisted at peak. Destructive iteration below
    #    pops each price_data entry as we consume it, so the two dicts never
    #    hold full-universe data simultaneously. Peak for this section drops
    #    by ~91 MB — modest, but the RSS log surrounding it makes the memory
    #    curve visible for the follow-up ohlcv→DataFrame refactor (tracked
    #    as P2 in SYSTEM_STATE).
    price_matrix = build_price_matrix(price_data, trading_dates)
    _log_rss("post_price_matrix")
    spy_prices = _extract_close(price_data, "SPY")  # extracts a Series copy
    _log_rss("post_spy_extract")

    ohlcv_by_ticker = _drain_price_data_into_ohlcv(price_data)
    # price_data is empty now (all entries popped by the drain). Release
    # the dict shell too so the GC can fully reclaim.
    del price_data
    gc.collect()
    _log_rss("post_ohlcv_build_and_drain")
    logger.info("Freed raw price data (memory optimization)")

    # 5. Download GBM model
    model_path = download_gbm_model(bucket=bucket)
    _log_rss("post_gbm_download")

    # 6. Run inference
    n_feature_tickers = len(features_by_ticker)
    predictions_by_date = run_inference(
        features_by_ticker, model_path, predictor_path, trading_dates,
    )
    _log_rss("post_inference")

    # Free features and model (no longer needed unless caller needs them)
    if not keep_features:
        del features_by_ticker
        gc.collect()
        logger.info("Freed feature data (memory optimization)")
        _log_rss("post_feature_free")

    # Clean up temp model file
    try:
        os.unlink(model_path)
        meta_path = model_path + ".meta.json"
        if os.path.exists(meta_path):
            os.unlink(meta_path)
    except OSError:
        pass

    # 7. Generate signals (using technical scoring from OHLCV, enriched by GBM alpha)
    signals_by_date = build_signals_by_date(
        predictions_by_date, sector_map, ohlcv_by_ticker,
        top_n=top_n, min_score=min_score,
    )
    _log_rss("post_build_signals")

    # Metadata for reporting
    n_enter_total = sum(
        len(env.get("buy_candidates", []))
        for env in signals_by_date.values()
    )

    metadata = {
        "data_source": data_source,
        "n_tickers": n_feature_tickers,
        "n_dates": len(trading_dates),
        "date_range_start": trading_dates[0],
        "date_range_end": trading_dates[-1],
        "n_enter_signals_total": n_enter_total,
        "top_n_per_day": top_n,
        "min_score": min_score,
    }
    logger.info("Predictor backtest data ready: %s", metadata)

    result = {
        "status": "ok",
        "signals_by_date": signals_by_date,
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv_by_ticker,
        "spy_prices": spy_prices,
        "metadata": metadata,
    }

    if keep_features:
        result["features_by_ticker"] = features_by_ticker
        result["sector_map"] = sector_map
        result["trading_dates"] = trading_dates
        result["predictions_by_date"] = predictions_by_date

    return result
