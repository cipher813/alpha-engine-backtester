"""
price_loader.py — reads prices/{date}/prices.json from S3, with fallbacks.

Fallback chain (in order):
  1. S3 prices/{date}/prices.json        — canonical source (written by research pipeline)
  2. yfinance                            — works standalone; tickers resolved from signals.json
  3. IBKR reqHistoricalData              — optional; only if ibkr_client is passed in

Price file format (written by alpha-engine-research pipeline):
{
    "date": "2026-03-06",
    "prices": {
        "PLTR": {"open": 84.12, "close": 85.47, "high": 86.10, "low": 83.90},
        "NVDA": {"open": 118.30, "close": 119.55, "high": 120.00, "low": 117.80}
    }
}

IBKR fallback:
    ibkr_client must implement:
        get_historical_bar(ticker: str, date: str) -> dict | None
            Returns {"open": ..., "close": ..., "high": ..., "low": ...} or None.
    This matches the interface being added to IBKRClient in alpha-engine (Phase 0a).
    Pass ibkr_client=None (default) to skip IBKR and rely only on yfinance.
"""

import json
import logging
from datetime import date, timedelta

import boto3
import pandas as pd
import yfinance as yf
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def load(
    bucket: str,
    price_date: str,
    tickers: list[str] | None = None,
    prefix: str = "prices",
    ibkr_client=None,
) -> dict:
    """
    Load prices for a date. Tries S3, then yfinance, then IBKR (if client provided).

    Args:
        bucket:       S3 bucket name.
        price_date:   Date string "YYYY-MM-DD".
        tickers:      Ticker list for yfinance/IBKR fallback. If None and S3 misses,
                      prices dict will be empty. Use build_matrix() to auto-resolve
                      tickers from signals.json.
        prefix:       S3 key prefix (default "prices").
        ibkr_client:  Optional IBKRClient instance with get_historical_bar(ticker, date).

    Returns:
        {"date": "YYYY-MM-DD", "prices": {"PLTR": {"open":..., "close":...}, ...},
         "source": "s3" | "yfinance" | "ibkr" | "partial"}
    """
    # 1. Try S3
    try:
        data = _load_from_s3(bucket, price_date, prefix)
        data["source"] = "s3"
        return data
    except FileNotFoundError:
        pass

    if not tickers:
        logger.warning(
            "prices.json missing for %s and no tickers provided — returning empty prices. "
            "Call build_matrix() or pass tickers= to get fallback data.",
            price_date,
        )
        return {"date": price_date, "prices": {}, "source": "none"}

    # 2. Try yfinance
    yf_data = _load_from_yfinance(price_date, tickers)
    missing = [t for t in tickers if t not in yf_data["prices"]]

    if not missing:
        logger.info("prices for %s: all %d tickers from yfinance", price_date, len(tickers))
        yf_data["source"] = "yfinance"
        return yf_data

    # 3. Fill any yfinance gaps with IBKR
    if ibkr_client is not None and missing:
        ibkr_prices = _load_from_ibkr(ibkr_client, price_date, missing)
        filled = len(ibkr_prices)
        yf_data["prices"].update(ibkr_prices)
        logger.info(
            "prices for %s: %d from yfinance, %d gap-filled from IBKR, %d still missing",
            price_date,
            len(tickers) - len(missing),
            filled,
            len(missing) - filled,
        )
        yf_data["source"] = "yfinance+ibkr" if filled else "yfinance"
    else:
        if missing:
            logger.warning(
                "prices for %s: %d/%d tickers missing from yfinance (pass ibkr_client= to fill gaps): %s",
                price_date,
                len(missing),
                len(tickers),
                missing[:10],
            )
        yf_data["source"] = "yfinance" if yf_data["prices"] else "none"

    return yf_data


def build_matrix(
    dates: list[str],
    bucket: str,
    field: str = "close",
    signals_prefix: str = "signals",
    prices_prefix: str = "prices",
    ibkr_client=None,
    _ohlcv_out: dict | None = None,
) -> pd.DataFrame:
    """
    Build a price matrix for vectorbt: rows = dates, columns = tickers.

    For each date, tries S3 prices.json first. On a miss, resolves the ticker
    list from the corresponding signals.json and falls back to yfinance (and
    optionally IBKR). This means the matrix is always populated as long as at
    least one data source is reachable.

    Args:
        dates:           List of date strings "YYYY-MM-DD".
        bucket:          S3 bucket for both signals and prices.
        field:           OHLCV field to use ("open", "close", "high", "low"). Default: "close".
        signals_prefix:  S3 prefix for signals (default "signals").
        prices_prefix:   S3 prefix for prices (default "prices").
        ibkr_client:     Optional IBKRClient for gap-filling after yfinance.

    Returns:
        DataFrame indexed by datetime, columns by ticker.
        Missing values are forward-filled then back-filled.
    """
    rows = {}
    sources = {}

    for d in sorted(dates):
        # Fast path: try S3 directly
        try:
            data = _load_from_s3(bucket, d, prices_prefix)
            rows[d] = {ticker: info[field] for ticker, info in data.get("prices", {}).items()}
            sources[d] = "s3"
        except FileNotFoundError:
            # Resolve tickers from signals.json so fallbacks know what to fetch
            tickers = _tickers_from_signals(bucket, d, signals_prefix)
            if not tickers:
                logger.warning("No signals found for %s — skipping price row", d)
                rows[d] = {}
                sources[d] = "none"
                continue

            data = load(
                bucket=bucket,
                price_date=d,
                tickers=tickers,
                prefix=prices_prefix,
                ibkr_client=ibkr_client,
            )
            rows[d] = {ticker: info[field] for ticker, info in data.get("prices", {}).items()}
            sources[d] = data.get("source", "unknown")

        # Capture full OHLCV if caller requested it
        if _ohlcv_out is not None:
            for ticker, info in data.get("prices", {}).items():
                _ohlcv_out.setdefault(ticker, []).append({
                    "date": d,
                    "open": info.get("open", info.get("close", 0)),
                    "high": info.get("high", info.get("close", 0)),
                    "low": info.get("low", info.get("close", 0)),
                    "close": info.get("close", 0),
                })

    # Sort OHLCV bars ascending by date (needed for ATR / trailing stop lookback)
    if _ohlcv_out is not None:
        for ticker in _ohlcv_out:
            _ohlcv_out[ticker].sort(key=lambda b: b["date"])

    # Log summary of data sources used
    from collections import Counter
    src_counts = Counter(sources.values())
    logger.info("build_matrix: %d dates — sources: %s", len(dates), dict(src_counts))

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# --- Private helpers ---

def _load_from_s3(bucket: str, price_date: str, prefix: str) -> dict:
    key = f"{prefix}/{price_date}/prices.json"
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        logger.debug("S3 prices for %s: %d tickers", price_date, len(data.get("prices", {})))
        return data
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            raise FileNotFoundError(f"No prices found at s3://{bucket}/{key}") from e
        raise


def _load_from_yfinance(price_date: str, tickers: list[str]) -> dict:
    """
    Fetch OHLCV for a single date from yfinance for the given tickers.

    yfinance requires end = date + 1 to include the target date's bar.
    Weekend/holiday dates return empty — caller should handle gracefully.
    """
    end_date = (date.fromisoformat(price_date) + timedelta(days=1)).isoformat()

    try:
        raw = yf.download(
            tickers,
            start=price_date,
            end=end_date,
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", price_date, e)
        return {"date": price_date, "prices": {}}

    prices = {}

    if raw.empty:
        logger.debug("yfinance returned no data for %s (weekend/holiday?)", price_date)
        return {"date": price_date, "prices": {}}

    if isinstance(raw.columns, pd.MultiIndex):
        # Multiple tickers: columns are (field, ticker)
        for ticker in tickers:
            try:
                row = raw.xs(ticker, axis=1, level=1)
                if not row.empty and not row.iloc[0].isna().all():
                    prices[ticker] = {
                        "open":  float(row["Open"].iloc[0]),
                        "close": float(row["Close"].iloc[0]),
                        "high":  float(row["High"].iloc[0]),
                        "low":   float(row["Low"].iloc[0]),
                    }
            except KeyError:
                pass
    else:
        # Single ticker
        ticker = tickers[0]
        if not raw.iloc[0].isna().all():
            prices[ticker] = {
                "open":  float(raw["Open"].iloc[0]),
                "close": float(raw["Close"].iloc[0]),
                "high":  float(raw["High"].iloc[0]),
                "low":   float(raw["Low"].iloc[0]),
            }

    logger.debug("yfinance: %d/%d tickers for %s", len(prices), len(tickers), price_date)
    return {"date": price_date, "prices": prices}


def _load_from_ibkr(ibkr_client, price_date: str, tickers: list[str]) -> dict:
    """
    Fetch prices for specific tickers from IBKR historical data.

    ibkr_client must implement:
        get_historical_bar(ticker: str, date: str) -> dict | None
        Returns {"open": float, "close": float, "high": float, "low": float} or None.

    Returns prices dict for successfully fetched tickers only.
    """
    prices = {}
    for ticker in tickers:
        try:
            bar = ibkr_client.get_historical_bar(ticker, price_date)
            if bar:
                prices[ticker] = bar
        except Exception as e:
            logger.debug("IBKR historical fetch failed for %s on %s: %s", ticker, price_date, e)
    logger.debug("IBKR: %d/%d tickers fetched for %s", len(prices), len(tickers), price_date)
    return prices


def _tickers_from_signals(bucket: str, signal_date: str, prefix: str = "signals") -> list[str]:
    """
    Extract the unique ticker list from signals.json for a given date.
    Used to drive yfinance/IBKR fallbacks when prices.json is missing.
    """
    key = f"{prefix}/{signal_date}/signals.json"
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        data = json.loads(response["Body"].read())
        tickers = list({s["symbol"] for s in data.get("signals", []) if "symbol" in s})
        logger.debug("Resolved %d tickers from signals for %s", len(tickers), signal_date)
        return tickers
    except ClientError as e:
        if e.response["Error"]["Code"] == "NoSuchKey":
            logger.warning("No signals.json found for %s — cannot resolve tickers", signal_date)
            return []
        raise
