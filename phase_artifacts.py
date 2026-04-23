"""
phase_artifacts.py — Serialization helpers for PhaseRegistry artifacts.

Each `save_*` function uploads an artifact to S3 and returns the key so
callers can hand it to `ctx.record_artifact(key)`. Each `load_*` function
is the strict inverse.

Layout:
    s3://{bucket}/backtest/{date}/.phases/{phase}/{name}.{ext}

JSON is used for dicts-of-scalars + lists; parquet for DataFrames and
OHLCV data. Pickle is NOT used — it's fragile across Python versions
and hides artifact contents from non-Python consumers (the evaluator
and future dashboards read these).

Motivated by ROADMAP Backtester P0 "Phase-selective backtest execution —
skip already-successful phases on retry". Paired with PhaseRegistry in
pipeline_common.py.
"""

from __future__ import annotations

import io
import json
import logging

import boto3
import pandas as pd

logger = logging.getLogger(__name__)


def artifact_key(date: str, phase: str, name: str, ext: str) -> str:
    """Canonical S3 key for a phase artifact. Exposed for tests + callers
    that want to name a key without uploading (e.g. the `--dry-run` path).
    """
    return f"backtest/{date}/.phases/{phase}/{name}.{ext}"


def _client(s3_client=None):
    return s3_client if s3_client is not None else boto3.client("s3")


# ── JSON artifacts (dicts, lists, scalars) ───────────────────────────────────


def save_json(bucket: str, date: str, phase: str, name: str, obj, *, s3_client=None) -> str:
    """Serialize `obj` to JSON and upload. Returns the S3 key.

    Uses `default=str` so datetime / numpy scalars don't crash the encoder.
    Callers that need strict type preservation should use a DataFrame-
    backed artifact instead.
    """
    key = artifact_key(date, phase, name, "json")
    body = json.dumps(obj, indent=2, default=str).encode()
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json",
    )
    return key


def load_json(bucket: str, key: str, *, s3_client=None):
    """Load a JSON artifact. Raises if missing or corrupt."""
    obj = _client(s3_client).get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())


# ── DataFrame artifacts ──────────────────────────────────────────────────────


def save_dataframe(
    bucket: str, date: str, phase: str, name: str, df: pd.DataFrame,
    *, s3_client=None, preserve_index: bool = True,
) -> str:
    """Upload `df` as parquet. Returns the S3 key.

    `preserve_index=True` keeps the row index so DataFrames with a
    meaningful index (e.g. `price_matrix` indexed by date) round-trip
    correctly. Set False for row-only frames like sweep_df.
    """
    key = artifact_key(date, phase, name, "parquet")
    buf = io.BytesIO()
    df.to_parquet(buf, index=preserve_index)
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    return key


def load_dataframe(bucket: str, key: str, *, s3_client=None) -> pd.DataFrame:
    """Load a parquet DataFrame."""
    obj = _client(s3_client).get_object(Bucket=bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


# ── OHLCV artifacts (dict[ticker → list[dict]]) ─────────────────────────────
#
# The backtester carries OHLCV as {ticker: [{date, open, high, low, close}, ...]}
# — that shape is not directly parquet-friendly because pandas would
# require one frame per ticker. Instead we stack all bars into a single
# frame with a `ticker` column. Load reconstructs the original dict.
# ~900 tickers × 2500 rows = ~2.3M rows, typical size ~50-100 MB on disk.


def save_ohlcv_by_ticker(
    bucket: str, date: str, phase: str, name: str,
    ohlcv_by_ticker: dict[str, list[dict]],
    *, s3_client=None,
) -> str:
    """Stack per-ticker OHLCV lists into one parquet frame + upload."""
    rows = []
    for ticker, bars in ohlcv_by_ticker.items():
        for bar in bars:
            rows.append({"ticker": ticker, **bar})
    df = pd.DataFrame(rows)
    return save_dataframe(
        bucket, date, phase, name, df,
        s3_client=s3_client, preserve_index=False,
    )


def load_ohlcv_by_ticker(
    bucket: str, key: str, *, s3_client=None,
) -> dict[str, list[dict]]:
    """Inverse of save_ohlcv_by_ticker.

    Reshapes the stacked frame back to the per-ticker list-of-dicts
    shape that the simulation loop expects. Date ordering is preserved
    within each ticker's list because the stacked frame preserves the
    original enumeration order.
    """
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if "ticker" not in df.columns:
        raise ValueError(
            f"load_ohlcv_by_ticker: parquet at {key!r} missing 'ticker' column "
            f"(found: {list(df.columns)})"
        )
    out: dict[str, list[dict]] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        out[str(ticker)] = group.drop(columns=["ticker"]).to_dict("records")
    return out


# ── Series artifacts (e.g. spy_prices) ───────────────────────────────────────


def save_series(
    bucket: str, date: str, phase: str, name: str, series: pd.Series,
    *, s3_client=None,
) -> str:
    """Persist a pd.Series as a single-column parquet. The Series name (if
    set) is preserved as the column name; index is kept."""
    df = series.to_frame(name=series.name or "value")
    return save_dataframe(
        bucket, date, phase, name, df, s3_client=s3_client, preserve_index=True,
    )


def load_series(bucket: str, key: str, *, s3_client=None) -> pd.Series:
    """Inverse of save_series — returns the single column as a Series."""
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.shape[1] != 1:
        raise ValueError(
            f"load_series: parquet at {key!r} has {df.shape[1]} columns "
            f"(expected 1 for a Series round-trip)"
        )
    col = df.columns[0]
    return df[col]


# ── Dict-of-Series artifacts (e.g. vwap_series_by_ticker) ────────────────────


def save_dict_of_series(
    bucket: str, date: str, phase: str, name: str,
    data: dict[str, pd.Series],
    *, s3_client=None,
) -> str:
    """Stack per-ticker Series into (ticker, idx, value) rows + parquet upload.

    For ~900 tickers × ~2500 bars the compressed size is small (~20-40 MB).
    On load, rebuilt as dict[ticker → Series(index=idx)]. Series dtype +
    index dtype round-trip via parquet.
    """
    rows = []
    for ticker, s in data.items():
        if s is None:
            continue
        for idx, value in s.items():
            rows.append({"ticker": str(ticker), "idx": idx, "value": value})
    df = pd.DataFrame(rows)
    return save_dataframe(
        bucket, date, phase, name, df,
        s3_client=s3_client, preserve_index=False,
    )


def load_dict_of_series(
    bucket: str, key: str, *, s3_client=None,
) -> dict[str, pd.Series]:
    """Inverse of save_dict_of_series."""
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.empty:
        return {}
    missing = {"ticker", "idx", "value"} - set(df.columns)
    if missing:
        raise ValueError(
            f"load_dict_of_series: parquet at {key!r} missing columns "
            f"{sorted(missing)} (found: {list(df.columns)})"
        )
    out: dict[str, pd.Series] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        s = pd.Series(
            group["value"].values, index=group["idx"].values, name=str(ticker),
        )
        out[str(ticker)] = s
    return out


# ── Dict-of-DataFrames artifacts (e.g. features_by_ticker) ───────────────────


def save_dict_of_dataframes(
    bucket: str, date: str, phase: str, name: str,
    data: dict[str, pd.DataFrame],
    *, s3_client=None,
) -> str:
    """Stack per-ticker DataFrames into one parquet with a 'ticker' column.

    The original DataFrame index is materialized as a column named
    `__idx__` so it round-trips; on load the per-ticker frames set that
    column back as the index. Columns across tickers must be a compatible
    superset — a ticker missing column C will load with NaN for C.

    Used for features_by_ticker in predictor_data_prep. ~900 tickers
    × ~2500 rows × 59 feature cols compresses to ~150-250 MB parquet.
    """
    parts = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        # Materialize index as a column so it survives concat+parquet
        reset = df.reset_index().rename(columns={df.index.name or "index": "__idx__"})
        reset.insert(0, "ticker", str(ticker))
        parts.append(reset)
    if not parts:
        stacked = pd.DataFrame(columns=["ticker", "__idx__"])
    else:
        stacked = pd.concat(parts, ignore_index=True)
    return save_dataframe(
        bucket, date, phase, name, stacked,
        s3_client=s3_client, preserve_index=False,
    )


def load_dict_of_dataframes(
    bucket: str, key: str, *, s3_client=None,
) -> dict[str, pd.DataFrame]:
    """Inverse of save_dict_of_dataframes."""
    stacked = load_dataframe(bucket, key, s3_client=s3_client)
    if stacked.empty or "ticker" not in stacked.columns:
        if stacked.empty:
            return {}
        raise ValueError(
            f"load_dict_of_dataframes: parquet at {key!r} missing 'ticker' "
            f"column (found: {list(stacked.columns)})"
        )
    out: dict[str, pd.DataFrame] = {}
    for ticker, group in stacked.groupby("ticker", sort=False):
        df = group.drop(columns=["ticker"]).copy()
        if "__idx__" in df.columns:
            df = df.set_index("__idx__")
            df.index.name = None
        out[str(ticker)] = df.reset_index(drop=True) if df.index.name == "index" else df
    return out
