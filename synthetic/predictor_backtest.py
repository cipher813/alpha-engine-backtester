"""
synthetic/predictor_backtest.py — predictor-only historical backtest pipeline.

Runs GBM inference on 2 years of slim-cache OHLCV data to generate synthetic
signals, then feeds them through the full executor pipeline (risk guard,
position sizing, ATR stops, time decay, graduated drawdown).

This tests everything downstream of Research without any LLM API calls:
    1. Load 2y OHLCV from predictor's slim cache (~900 tickers)
    2. Compute 29 technical features per ticker (one full series per ticker)
    3. Run GBM inference in daily batches (~500 trading days × ~900 tickers)
    4. Convert alpha predictions to executor-compatible signals
    5. Build price matrix + OHLCV histories for simulation loop

The caller (backtest.py) then passes these to _run_simulation_loop() with
the existing executor pipeline.

Performance notes:
    - Feature computation: ~900 calls to compute_features() (one per ticker)
    - GBM inference: ~500 batch calls (one per trading day, ~900 vectors each)
    - Total runtime: ~5-10 min on EC2 (dominated by feature computation)
"""

from __future__ import annotations

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
    skipped = 0

    for i, ticker in enumerate(stock_tickers):
        df = price_data[ticker]

        # Need ~265 rows for 52-week rolling windows
        if len(df) < 265:
            skipped += 1
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
        except Exception as e:
            logger.debug("Feature computation failed for %s: %s", ticker, e)
            skipped += 1

        if (i + 1) % 100 == 0:
            logger.info("  Features computed: %d/%d tickers", i + 1, len(stock_tickers))

    logger.info(
        "Feature computation complete: %d tickers with features, %d skipped",
        len(features_by_ticker), skipped,
    )
    return features_by_ticker


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
    from config import FEATURES

    scorer = GBMScorer.load(model_path)

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

    predictions_by_date: dict[str, dict[str, float]] = {}

    for i, date_str in enumerate(trading_dates):
        ts = pd.Timestamp(date_str)

        # Collect feature vectors for all tickers on this date
        tickers_batch = []
        vectors_batch = []

        for ticker, featured_df in features_by_ticker.items():
            if ts in featured_df.index:
                row = featured_df.loc[ts]
                # Handle duplicate dates (take last)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[-1]
                try:
                    vec = row[FEATURES].to_numpy(dtype=np.float32)
                    if not np.any(np.isnan(vec)):
                        tickers_batch.append(ticker)
                        vectors_batch.append(vec)
                except (KeyError, ValueError):
                    continue

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
    top_n: int = 20,
    min_score: float = 70,
) -> dict[str, dict]:
    """
    Convert per-date predictions to executor signal envelopes.

    Parameters
    ----------
    predictions_by_date : {date: {ticker: alpha}} from run_inference()
    sector_map : {ticker: sector_etf} from sector_map.json
    top_n : max ENTER signals per day (prevents unrealistic portfolio churn)
    min_score : minimum composite score for ENTER signal

    Returns
    -------
    {date: signal_envelope} — each envelope is a full signals_override dict.
    """
    signals_by_date: dict[str, dict] = {}

    for date_str, predictions in predictions_by_date.items():
        envelope = predictions_to_signals(
            predictions=predictions,
            date=date_str,
            sector_map=sector_map,
            top_n=top_n,
            min_score=min_score,
        )
        signals_by_date[date_str] = envelope

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


def run(config: dict) -> dict:
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
    min_trading_days = pb_config.get("min_trading_days", 200)
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # 1. Load slim cache
    price_data = load_slim_cache(predictor_path)

    # 2. Load sector map
    sector_map = load_sector_map(predictor_path)

    # 3. Compute features
    features_by_ticker = compute_all_features(price_data, sector_map, predictor_path)

    if not features_by_ticker:
        return {
            "status": "error",
            "error": "No tickers had sufficient data for feature computation",
        }

    # Determine common trading dates (dates where enough tickers have features)
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

    logger.info(
        "Trading dates: %d (from %s to %s)",
        len(trading_dates), trading_dates[0], trading_dates[-1],
    )

    # 4. Download GBM model
    model_path = download_gbm_model(bucket=bucket)

    # 5. Run inference
    predictions_by_date = run_inference(
        features_by_ticker, model_path, predictor_path, trading_dates,
    )

    # Clean up temp model file
    try:
        os.unlink(model_path)
        meta_path = model_path + ".meta.json"
        if os.path.exists(meta_path):
            os.unlink(meta_path)
    except OSError:
        pass

    # 6. Generate signals
    signals_by_date = build_signals_by_date(
        predictions_by_date, sector_map,
        top_n=top_n, min_score=min_score,
    )

    # 7. Build price matrix and OHLCV
    price_matrix = build_price_matrix(price_data, trading_dates)
    ohlcv_by_ticker = build_ohlcv_by_ticker(price_data)

    # Metadata for reporting
    n_enter_total = sum(
        len(env.get("buy_candidates", []))
        for env in signals_by_date.values()
    )

    metadata = {
        "n_tickers": len(features_by_ticker),
        "n_dates": len(trading_dates),
        "date_range_start": trading_dates[0],
        "date_range_end": trading_dates[-1],
        "n_enter_signals_total": n_enter_total,
        "top_n_per_day": top_n,
        "min_score": min_score,
    }
    logger.info("Predictor backtest data ready: %s", metadata)

    return {
        "status": "ok",
        "signals_by_date": signals_by_date,
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv_by_ticker,
        "metadata": metadata,
    }
