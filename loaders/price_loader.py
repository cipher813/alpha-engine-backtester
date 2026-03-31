"""
price_loader.py — reads prices/{date}/prices.json from S3, with fallbacks.

Fallback chain (in order):
  1. S3 prices/{date}/prices.json        — canonical source (written by research pipeline)
  2. Polygon grouped-daily               — all US stocks for a single date in 1 API call
  3. yfinance                            — works standalone; tickers resolved from signals.json
  4. IBKR reqHistoricalData              — optional; only if ibkr_client is passed in

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
        data["status"] = "ok"
        return data
    except FileNotFoundError:
        pass

    if not tickers:
        logger.warning(
            "prices.json missing for %s and no tickers provided — returning empty prices. "
            "Call build_matrix() or pass tickers= to get fallback data.",
            price_date,
        )
        return {"date": price_date, "prices": {}, "source": "none", "status": "no_data"}

    # 2. Try polygon grouped-daily (all US stocks in 1 API call)
    poly_data = _load_from_polygon(price_date, tickers)
    if poly_data["prices"]:
        missing_poly = [t for t in tickers if t not in poly_data["prices"]]
        if not missing_poly:
            logger.info("prices for %s: all %d tickers from polygon", price_date, len(tickers))
            poly_data["source"] = "polygon"
            poly_data["status"] = "ok"
            return poly_data
        logger.info("prices for %s: %d/%d from polygon, falling back for %d",
                     price_date, len(tickers) - len(missing_poly), len(tickers), len(missing_poly))

    # 3. Try yfinance
    yf_data = _load_from_yfinance(price_date, tickers)
    # Merge polygon + yfinance results
    if poly_data["prices"]:
        merged = {**yf_data["prices"], **poly_data["prices"]}  # polygon wins on overlap
        yf_data["prices"] = merged
    missing = [t for t in tickers if t not in yf_data["prices"]]

    if not missing:
        source = "polygon+yfinance" if poly_data["prices"] else "yfinance"
        logger.info("prices for %s: all %d tickers from %s", price_date, len(tickers), source)
        yf_data["source"] = source
        yf_data["status"] = "ok"
        return yf_data

    # 4. Fill any remaining gaps with IBKR
    if ibkr_client is not None and missing:
        ibkr_prices = _load_from_ibkr(ibkr_client, price_date, missing)
        filled = len(ibkr_prices)
        yf_data["prices"].update(ibkr_prices)
        logger.info(
            "prices for %s: %d fetched, %d gap-filled from IBKR, %d still missing",
            price_date,
            len(tickers) - len(missing),
            filled,
            len(missing) - filled,
        )
        yf_data["source"] = "polygon+yfinance+ibkr" if poly_data["prices"] else "yfinance+ibkr"
    else:
        if missing:
            logger.warning(
                "prices for %s: %d/%d tickers missing (pass ibkr_client= to fill gaps): %s",
                price_date,
                len(missing),
                len(tickers),
                missing[:10],
            )
        yf_data["source"] = "yfinance" if yf_data["prices"] else "none"

    yf_data["status"] = "ok" if yf_data["prices"] else "no_data"
    if yf_data["source"] == "none":
        logger.warning("All price fallbacks failed for %s — no data available", price_date)

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
    no_data_dates = [d for d, s in sources.items() if s == "none"]

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    _MAX_FFILL_DAYS = 5   # max forward-fill window (1 trading week)
    _MAX_STALENESS_BDAYS = 5  # circuit breaker: halt simulation if data this stale

    # Gap detection: detect gaps before filling
    was_nan = df.isna()
    gap_lengths = was_nan.sum(axis=0)
    long_gaps = gap_lengths[gap_lengths > 5]
    price_gap_warnings = {}
    if not long_gaps.empty:
        price_gap_warnings = {str(k): int(v) for k, v in long_gaps.items()}
        logger.warning("Price gaps detected (will be ffill'd up to %d days): %s",
                        _MAX_FFILL_DAYS, price_gap_warnings)

    # Bounded forward-fill + back-fill for leading NaNs
    df.ffill(limit=_MAX_FFILL_DAYS, inplace=True)
    df.bfill(inplace=True)

    # Drop tickers with remaining NaNs after bounded fill — VectorBT treats
    # NaN prices as zero return, which silently distorts backtest results.
    remaining_nans = df.isna().sum(axis=0)
    unfilled = remaining_nans[remaining_nans > 0]
    unfilled_dict = {}
    if not unfilled.empty:
        unfilled_dict = {str(k): int(v) for k, v in unfilled.items()}
        logger.warning(
            "Dropping %d tickers with unfilled gaps after ffill(limit=%d): %s",
            len(unfilled_dict), _MAX_FFILL_DAYS, unfilled_dict,
        )
        df.drop(columns=unfilled.index, inplace=True)

    # Freshness validation: tiered staleness check
    staleness_warning = None
    stale_circuit_break = False
    if not df.empty:
        last_price_date = df.index.max()
        expected_recent = pd.Timestamp.now() - pd.tseries.offsets.BDay(2)
        expected_max = pd.Timestamp.now() - pd.tseries.offsets.BDay(_MAX_STALENESS_BDAYS)
        if last_price_date < expected_max:
            staleness_warning = (
                f"STALE price data: last date {last_price_date.date()}, "
                f"expected >= {expected_max.date()} — simulation results unreliable"
            )
            logger.error(staleness_warning)
            stale_circuit_break = True
        elif last_price_date < expected_recent:
            staleness_warning = (
                f"Stale price data: last date {last_price_date.date()}, "
                f"expected >= {expected_recent.date()}"
            )
            logger.warning(staleness_warning)

    # Store metadata on the DataFrame for downstream reporting
    df.attrs["price_gap_warnings"] = price_gap_warnings
    df.attrs["unfilled_gaps"] = unfilled_dict
    df.attrs["staleness_warning"] = staleness_warning
    df.attrs["stale_circuit_break"] = stale_circuit_break
    df.attrs["no_data_dates"] = no_data_dates

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
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "AccessDenied", "403"):
            # AccessDenied can occur when the key does not exist and the caller
            # lacks s3:ListBucket — S3 returns 403 instead of 404.
            if code != "NoSuchKey":
                logger.debug("S3 returned %s for %s (treating as missing)", code, key)
            raise FileNotFoundError(f"No prices found at s3://{bucket}/{key}") from e
        raise


def _load_from_polygon(price_date: str, tickers: list[str]) -> dict:
    """Fetch prices via polygon grouped-daily endpoint (all US stocks, 1 API call)."""
    try:
        from polygon_client import polygon_client
        grouped = polygon_client().get_grouped_daily(price_date)
        if not grouped:
            return {"date": price_date, "prices": {}}
        prices = {}
        for ticker in tickers:
            if ticker in grouped:
                g = grouped[ticker]
                prices[ticker] = {
                    "open": g["open"],
                    "close": g["close"],
                    "high": g["high"],
                    "low": g["low"],
                }
        return {"date": price_date, "prices": prices}
    except Exception as e:
        logger.warning("Polygon grouped-daily failed for %s: %s", price_date, e)
        return {"date": price_date, "prices": {}}


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
            timeout=300,
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
        # Signals may be a dict keyed by ticker or a list; universe is always a list.
        # The per-stock field is "ticker" (not "symbol").
        sigs = data.get("signals", {})
        if isinstance(sigs, dict):
            ticker_set = set(sigs.keys())
        else:
            ticker_set = {s["ticker"] for s in sigs if "ticker" in s}
        # Also pull from universe (v2 format) and buy_candidates
        for key in ("universe", "buy_candidates"):
            for s in data.get(key, []):
                if isinstance(s, dict) and "ticker" in s:
                    ticker_set.add(s["ticker"])
        tickers = sorted(ticker_set)
        logger.debug("Resolved %d tickers from signals for %s", len(tickers), signal_date)
        return tickers
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("NoSuchKey", "AccessDenied", "403"):
            logger.warning("No signals.json found for %s (%s) — cannot resolve tickers", signal_date, code)
            return []
        raise
