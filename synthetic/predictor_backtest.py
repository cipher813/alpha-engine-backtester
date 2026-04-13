"""
synthetic/predictor_backtest.py — predictor-only historical backtest pipeline.

Runs GBM inference on up to 10 years of OHLCV data to generate synthetic
signals, then feeds them through the full executor pipeline (risk guard,
position sizing, ATR stops, time decay, graduated drawdown).

This tests everything downstream of Research without any LLM API calls:
    1. Load OHLCV data (10y from S3 full cache, or 2y local slim cache)
    2. Compute 29 technical features per ticker (one full series per ticker)
    3. Run GBM inference in daily batches (up to ~2520 days × ~900 tickers)
    4. Convert alpha predictions to executor-compatible signals
    5. Build price matrix + OHLCV histories for simulation loop

The caller (backtest.py) then passes these to _run_simulation_loop() with
the existing executor pipeline.

Data sources:
    - Full cache (10y): S3 predictor/price_cache/*.parquet — use on spot instances
    - Slim cache (2y): local predictor/data/cache/*.parquet — use on always-on EC2

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


def load_full_cache_from_s3(
    bucket: str = "alpha-engine-research",
    prefix: str = "predictor/price_cache",
    region: str = "us-east-1",
) -> dict[str, pd.DataFrame]:
    """
    Load 10-year OHLCV parquets from S3 full price cache.

    This is the primary data source for spot instance backtest runs.
    The full cache contains ~10 years of adjusted OHLCV data for ~900 tickers,
    refreshed weekly by the predictor training pipeline.

    Returns {ticker: DataFrame} with DatetimeIndex and OHLCV columns.
    """
    s3 = boto3.client("s3", region_name=region)

    # List all parquet files in the cache prefix
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".parquet"):
                keys.append(obj["Key"])

    logger.info("Loading full cache from S3: %d parquet files at s3://%s/%s", len(keys), bucket, prefix)

    price_data: dict[str, pd.DataFrame] = {}
    tmp_dir = tempfile.mkdtemp(prefix="backtest_cache_")

    for i, key in enumerate(keys):
        ticker = Path(key).stem  # e.g. "AAPL" from ".../AAPL.parquet"
        local_path = os.path.join(tmp_dir, f"{ticker}.parquet")

        try:
            s3.download_file(bucket, key, local_path)
            df = pd.read_parquet(local_path)
            if df.empty:
                continue

            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                else:
                    df.index = pd.to_datetime(df.index)

            df = df.sort_index()
            price_data[ticker] = df

            # Clean up temp file immediately to limit disk usage
            os.unlink(local_path)
        except Exception as e:
            logger.debug("Skipping %s: %s", key, e)

        if (i + 1) % 100 == 0:
            logger.info("  Loaded %d/%d parquets from S3", i + 1, len(keys))

    # Clean up temp directory
    try:
        os.rmdir(tmp_dir)
    except OSError:
        pass

    logger.info("Loaded %d tickers from S3 full cache (10y)", len(price_data))
    return price_data


def load_slim_cache(predictor_path: str) -> dict[str, pd.DataFrame]:
    """
    Read all parquet files from {predictor_path}/data/cache/ into memory.

    Returns {ticker: DataFrame} where DataFrame has DatetimeIndex and
    columns including Close, Volume, Open, High, Low.

    Includes SPY, VIX, sector ETFs, and macro series alongside ~500 stocks.
    """
    cache_dir = Path(predictor_path) / "data" / "cache"
    if not cache_dir.exists():
        raise FileNotFoundError(f"Predictor cache directory not found: {cache_dir}")

    price_data: dict[str, pd.DataFrame] = {}
    parquet_files = list(cache_dir.glob("*.parquet"))
    logger.info("Loading slim cache: %d parquet files from %s", len(parquet_files), cache_dir)

    for pf in parquet_files:
        ticker = pf.stem  # e.g. "AAPL" from "AAPL.parquet"
        try:
            df = pd.read_parquet(pf)
            if df.empty:
                continue
            # Ensure DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"])
                    df = df.set_index("Date")
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                else:
                    df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            price_data[ticker] = df
        except Exception as e:
            logger.debug("Skipping %s: %s", pf.name, e)

    logger.info("Loaded %d tickers from slim cache", len(price_data))
    return price_data


def load_sector_map(predictor_path: str) -> dict[str, str]:
    """Load sector_map.json mapping tickers to sector ETF symbols."""
    map_path = Path(predictor_path) / "data" / "cache" / "sector_map.json"
    if not map_path.exists():
        logger.warning("sector_map.json not found at %s", map_path)
        return {}
    with open(map_path) as f:
        return json.load(f)


def _load_features_from_store(
    trading_dates: list[str],
    bucket: str,
) -> dict[str, pd.DataFrame] | None:
    """
    Try to load pre-computed features from the S3 feature store.

    Reads technical + interaction group Parquet files for each trading date,
    then pivots back to {ticker: DataFrame} format for GBM inference.

    Returns None if feature store is unavailable or has insufficient coverage.
    """
    try:
        s3 = boto3.client("s3")

        # Quick availability check — does the features/ prefix exist?
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="features/", MaxKeys=1)
        if resp.get("KeyCount", 0) == 0:
            logger.info("Feature store empty at s3://%s/features/ — falling back to computation", bucket)
            return None

        # Read technical + interaction groups for each date
        frames_by_ticker: dict[str, list[pd.DataFrame]] = {}
        dates_found = 0

        for date_str in trading_dates:
            day_frames = []
            for group in ("technical", "interaction"):
                key = f"features/{date_str}/{group}.parquet"
                try:
                    import io
                    obj = s3.get_object(Bucket=bucket, Key=key)
                    buf = io.BytesIO(obj["Body"].read())
                    df = pd.read_parquet(buf, engine="pyarrow")
                    day_frames.append(df)
                except Exception:
                    continue

            if not day_frames:
                continue

            dates_found += 1
            # Merge groups on ticker + date
            merged = day_frames[0]
            for extra in day_frames[1:]:
                merged = merged.merge(extra, on=["ticker", "date"], how="left")

            # Pivot into per-ticker frames
            for _, row in merged.iterrows():
                ticker = row.get("ticker")
                if not ticker:
                    continue
                frames_by_ticker.setdefault(ticker, []).append(row)

        # Need at least 50% date coverage to be useful
        coverage = dates_found / max(len(trading_dates), 1)
        if coverage < 0.50:
            logger.info(
                "Feature store coverage %.0f%% (%d/%d dates) — too sparse, falling back",
                coverage * 100, dates_found, len(trading_dates),
            )
            return None

        # Convert to {ticker: DataFrame} with DatetimeIndex
        features_by_ticker: dict[str, pd.DataFrame] = {}
        for ticker, rows in frames_by_ticker.items():
            df = pd.DataFrame(rows).drop(columns=["ticker"], errors="ignore")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date").sort_index()
            if not df.empty:
                features_by_ticker[ticker] = df

        logger.info(
            "Feature store loaded: %d tickers, %d dates (%.0f%% coverage)",
            len(features_by_ticker), dates_found, coverage * 100,
        )
        return features_by_ticker

    except Exception as e:
        logger.warning("Feature store read failed — falling back to computation: %s", e)
        return None


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
    price_data : {ticker: OHLCV DataFrame} from load_slim_cache()
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
    Download GBM model weights + metadata from S3 to a temp file.

    Tries predictor/weights/ first, falls back to backtest/ prefix
    (EC2 IAM role may lack access to predictor/ prefix).

    Returns the local path to the model file.
    """
    s3 = boto3.client("s3", region_name=region)

    # Download booster file — try primary path, fall back to backtest/ mirror
    model_tmp = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
    model_tmp.close()
    for key in ("predictor/weights/gbm_latest.txt", "backtest/gbm_latest.txt"):
        try:
            s3.download_file(bucket, key, model_tmp.name)
            logger.info("Downloaded GBM model from s3://%s/%s", bucket, key)
            break
        except Exception as e:
            logger.debug("Could not download %s: %s", key, e)
    else:
        raise RuntimeError(
            f"GBM model not found in S3 bucket {bucket} at "
            "predictor/weights/gbm_latest.txt or backtest/gbm_latest.txt"
        )

    # Download metadata (optional)
    meta_path = model_tmp.name + ".meta.json"
    for meta_key in ("predictor/weights/gbm_latest.txt.meta.json", "backtest/gbm_latest.txt.meta.json"):
        try:
            s3.download_file(bucket, meta_key, meta_path)
            break
        except Exception:
            continue

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
    """
    signals_by_date: dict[str, dict] = {}
    sorted_dates = sorted(predictions_by_date.keys())

    for i, date_str in enumerate(sorted_dates):
        predictions = predictions_by_date[date_str]

        # Filter OHLCV to dates <= current date to prevent lookahead bias
        ohlcv_up_to_date = {}
        for ticker, bars in ohlcv_by_ticker.items():
            filtered = [bar for bar in bars if bar["date"] <= date_str]
            if filtered:
                ohlcv_up_to_date[ticker] = filtered

        envelope = predictions_to_signals(
            predictions=predictions,
            date=date_str,
            sector_map=sector_map,
            ohlcv_by_ticker=ohlcv_up_to_date,
            top_n=top_n,
            min_score=min_score,
        )
        signals_by_date[date_str] = envelope

        if (i + 1) % 50 == 0:
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
    """
    skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
    ohlcv: dict[str, list[dict]] = {}

    for ticker, df in price_data.items():
        if ticker in skip_tickers:
            continue
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
        if bars:
            ohlcv[ticker] = bars

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
    use_full_cache = pb_config.get("use_full_cache", False)
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # 1. Load price data + features — try ArcticDB first, fall back to legacy path
    use_arcticdb = pb_config.get("use_arcticdb", True)
    features_by_ticker = None
    feature_skip_reasons = {}
    price_data = None
    data_source = "legacy"

    if use_arcticdb:
        try:
            from store.arctic_reader import load_universe_from_arctic
            logger.info("[data_source=arcticdb] Loading universe from ArcticDB...")
            price_data, features_by_ticker = load_universe_from_arctic(bucket=bucket)
            if features_by_ticker:
                data_source = "arcticdb"
                logger.info("[data_source=arcticdb] %d tickers with pre-computed features — skipped recomputation", len(features_by_ticker))
            else:
                logger.warning("[data_source=arcticdb] No features returned — falling back to legacy path")
                price_data = None
        except Exception as exc:
            logger.warning("[data_source=arcticdb] Load failed — falling back to legacy path: %s", exc)

    # Legacy fallback: load from S3 parquets + compute features inline
    if price_data is None:
        data_source = "legacy"
        if use_full_cache:
            logger.info("[data_source=legacy] Loading full 10y price cache from S3...")
            price_data = load_full_cache_from_s3(bucket=bucket)
        else:
            price_data = load_slim_cache(predictor_path)

        # Trim to recent rows to limit memory
        trim_rows = max_trading_days + 300
        trimmed = 0
        for ticker in price_data:
            if len(price_data[ticker]) > trim_rows:
                price_data[ticker] = price_data[ticker].iloc[-trim_rows:]
                trimmed += 1
        if trimmed:
            logger.info("Trimmed %d tickers to most recent %d rows (memory optimization)", trimmed, trim_rows)

    # 2. Load sector map
    sector_map = load_sector_map(predictor_path)

    # 3. Compute features if not already loaded from ArcticDB
    if features_by_ticker is None:
        use_feature_store = pb_config.get("use_feature_store", True)

        if use_feature_store and not use_full_cache:
            all_dates = set()
            skip_tickers = _MACRO_TICKERS | _SECTOR_ETFS
            for t, df in price_data.items():
                if t not in skip_tickers:
                    all_dates.update(df.index.strftime("%Y-%m-%d"))
            est_dates = sorted(all_dates)[-max_trading_days:] if all_dates else []
            if est_dates:
                features_by_ticker = _load_features_from_store(est_dates, bucket)
                if features_by_ticker:
                    logger.info("Using feature store cache (%d tickers) — skipped recomputation", len(features_by_ticker))

        if features_by_ticker is None:
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

    # 4. Build price matrix and OHLCV early, then free raw price data
    price_matrix = build_price_matrix(price_data, trading_dates)
    ohlcv_by_ticker = build_ohlcv_by_ticker(price_data)
    spy_prices = _extract_close(price_data, "SPY")
    del price_data
    gc.collect()
    logger.info("Freed raw price data (memory optimization)")

    # 5. Download GBM model
    model_path = download_gbm_model(bucket=bucket)

    # 6. Run inference
    n_feature_tickers = len(features_by_ticker)
    predictions_by_date = run_inference(
        features_by_ticker, model_path, predictor_path, trading_dates,
    )

    # Free features and model (no longer needed unless caller needs them)
    if not keep_features:
        del features_by_ticker
        gc.collect()
        logger.info("Freed feature data (memory optimization)")

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
