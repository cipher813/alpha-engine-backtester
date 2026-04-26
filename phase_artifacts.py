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
    ohlcv_by_ticker: dict,
    *, s3_client=None,
) -> str:
    """Persist ``ohlcv_by_ticker`` (DataFrame shape only — Option A
    step 9 cleanup deleted the legacy list-of-dicts producer).

    Delegates to ``save_dict_of_dataframes`` which adds a ``ticker``
    column and an ``__idx__`` column carrying the DatetimeIndex.
    """
    if not ohlcv_by_ticker:
        return save_dict_of_dataframes(
            bucket, date, phase, name, ohlcv_by_ticker, s3_client=s3_client,
        )
    sample = next(iter(ohlcv_by_ticker.values()))
    if not isinstance(sample, pd.DataFrame):
        raise TypeError(
            f"save_ohlcv_by_ticker: expected dict[str, pd.DataFrame] post "
            f"Option A step 9 cleanup; got value of type "
            f"{type(sample).__name__}. Producer must use "
            f"build_ohlcv_df_by_ticker."
        )
    return save_dict_of_dataframes(
        bucket, date, phase, name, ohlcv_by_ticker, s3_client=s3_client,
    )


def load_ohlcv_by_ticker(
    bucket: str, key: str, *, s3_client=None,
) -> dict:
    """Inverse of ``save_ohlcv_by_ticker``. Returns
    ``dict[str, pd.DataFrame]`` (DatetimeIndex per ticker).

    Hard-fails on legacy artifacts (parquet without ``__idx__`` column)
    so a stale pre-Option-A artifact from a prior run can't silently
    feed list-of-dicts shape to the now-DataFrame-only consumers. The
    auto-skip layer treats this as a real failure and re-runs the
    upstream phase.
    """
    df = load_dataframe(bucket, key, s3_client=s3_client)
    if df.empty:
        return {}
    if "ticker" not in df.columns:
        raise ValueError(
            f"load_ohlcv_by_ticker: parquet at {key!r} missing 'ticker' column "
            f"(found: {list(df.columns)})"
        )
    if "__idx__" not in df.columns:
        raise ValueError(
            f"load_ohlcv_by_ticker: parquet at {key!r} is on the legacy "
            f"list-of-dicts schema (no '__idx__' column). Option A step 9 "
            f"cleanup removed legacy support — re-run the upstream "
            f"predictor_data_prep phase to regenerate the artifact in "
            f"DataFrame shape."
        )
    out_df: dict[str, pd.DataFrame] = {}
    for ticker, group in df.groupby("ticker", sort=False):
        frame = group.drop(columns=["ticker"]).copy()
        frame = frame.set_index("__idx__")
        frame.index.name = None
        out_df[str(ticker)] = frame
    return out_df


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


def _promote_int_to_float_on_drift(
    ticker_frames: list[tuple[str, pd.DataFrame]],
) -> None:
    """In-place: for any column that's integer in some frames and float
    in others, cast every integer instance to float64. Float64 is the
    wider type and lossless for the value ranges typical of OHLCV data.

    Motivated by the 2026-04-26 spot OOM-RECOVERY validation run, where
    ``predictor_data_prep`` aborted at the post_inference checkpoint
    with ``Unable to merge: Field Volume has incompatible types:
    int64 vs double``. Audit of the live ArcticDB universe library found
    804 tickers with int64 Volume, 107 with float64 — the float ones
    are mostly recent listings (UBER, APP, VRT, DT, KD) where early
    bars had NaN volumes that pandas auto-cast to float64.

    The previous behavior was a hard-fail in ``pa.unify_schemas`` —
    which is correct for genuinely incompatible types but unhelpful
    for the int→float promotion case where the wider type
    semantically subsumes the narrower one. We handle int→float here
    as the only safe auto-promotion; anything else still raises at
    ``unify_schemas`` (e.g. string vs numeric, datetime vs object —
    real semantic divergences worth investigating).

    No-op when zero columns drift, so the common case (all tickers
    homogeneous) pays only an O(N_frames × N_cols) dtype scan.
    """
    if not ticker_frames:
        return
    from collections import defaultdict
    col_dtypes: dict[str, set[str]] = defaultdict(set)
    for _, frame in ticker_frames:
        for col in frame.columns:
            col_dtypes[col].add(str(frame[col].dtype))

    drift_cols: list[str] = []
    for col, dtypes in col_dtypes.items():
        has_int = any(d.startswith("int") or d.startswith("uint") for d in dtypes)
        has_float = any(d.startswith("float") for d in dtypes)
        if has_int and has_float:
            drift_cols.append(col)

    if not drift_cols:
        return

    import logging
    logging.getLogger(__name__).info(
        "save_dict_of_dataframes: promoting %d int-vs-float drift column(s) "
        "to float64: %s",
        len(drift_cols), drift_cols,
    )
    for _, frame in ticker_frames:
        for col in drift_cols:
            if col in frame.columns:
                dtype_str = str(frame[col].dtype)
                if dtype_str.startswith("int") or dtype_str.startswith("uint"):
                    frame[col] = frame[col].astype("float64")


def save_dict_of_dataframes(
    bucket: str, date: str, phase: str, name: str,
    data: dict[str, pd.DataFrame],
    *, s3_client=None,
) -> str:
    """Stream per-ticker DataFrames into one parquet with a 'ticker' column.

    The original DataFrame index is materialized as a column named
    `__idx__` so it round-trips; on load the per-ticker frames set that
    column back as the index. Columns across tickers must be a compatible
    superset — a ticker missing column C will load with NaN for C.

    Used for features_by_ticker in predictor_data_prep. ~900 tickers
    × ~2500 rows × 59 feature cols compresses to ~150-250 MB parquet.

    Memory-efficient streaming write: uses pyarrow ParquetWriter to
    append one row group per ticker instead of building a stacked
    in-memory DataFrame via pd.concat. Motivated by the 2026-04-24
    dry-run's predictor_data_prep save block spending 20+ min
    thrashing swap on c5.large — the prior pd.concat approach held
    (dict + parts list + concat result + parquet buffer) = ~2.3 GB
    simultaneously. This path peaks at ~1 ticker (~1.4 MB) + the
    accumulating compressed parquet buffer (~250 MB final) = ~300 MB
    peak, ~8× reduction.

    Schema unification: when tickers have mismatched columns, pyarrow's
    default behavior rejects the second row group. We unify by building
    all pyarrow Tables first (cheap, zero-copy from pandas), then
    `pa.unify_schemas()` on their schemas and casting each table to the
    unified schema before appending. This preserves the legacy
    "compatible superset" semantic (missing column → NaN on load).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Filter + build the reset-and-annotated DataFrames, but do NOT
    # concat them. Each becomes a pyarrow Table below.
    ticker_frames: list[tuple[str, pd.DataFrame]] = []
    for ticker, df in data.items():
        if df is None or df.empty:
            continue
        reset = df.reset_index().rename(columns={df.index.name or "index": "__idx__"})
        reset.insert(0, "ticker", str(ticker))
        ticker_frames.append((str(ticker), reset))

    if not ticker_frames:
        # Empty-data fallback preserves the legacy shape: a parquet with
        # just the ticker + __idx__ column headers. Load path returns {}.
        empty = pd.DataFrame(columns=["ticker", "__idx__"])
        return save_dataframe(
            bucket, date, phase, name, empty,
            s3_client=s3_client, preserve_index=False,
        )

    # Type-drift unification (added 2026-04-26 after Volume int64-vs-float64
    # mismatch on the universe library — 804 tickers had int64, 107 had
    # float64 because their early bars contained NaNs). pyarrow's
    # ``unify_schemas`` doesn't auto-promote int→float; it raises
    # ``Unable to merge: Field X has incompatible types: int64 vs double``.
    # We pre-promote at the pandas layer: any column whose dtype family
    # differs across ticker frames (int in some, float in others) gets
    # cast to float64 on the int instances. Float64 is the wider type
    # and is lossless for typical Volume values.
    _promote_int_to_float_on_drift(ticker_frames)

    # Convert each DataFrame to a pyarrow Table. preserve_index=False
    # because we materialized the original index into __idx__ above.
    tables = [
        (ticker, pa.Table.from_pandas(frame, preserve_index=False))
        for ticker, frame in ticker_frames
    ]

    # Unify schemas so row groups with missing columns widen to the
    # union (NaN-filled on load). Matches the legacy pd.concat semantic.
    unified_schema = pa.unify_schemas([t.schema for _, t in tables])
    tables = [(ticker, t.cast(unified_schema)) for ticker, t in tables]

    # Stream-write each table as its own row group. The ParquetWriter
    # buffer is the only large allocation and grows only with compressed
    # output size. No full-DataFrame concat ever materializes in memory.
    buf = io.BytesIO()
    writer = pq.ParquetWriter(buf, unified_schema, compression="snappy")
    try:
        for _, table in tables:
            writer.write_table(table)
    finally:
        writer.close()

    key = artifact_key(date, phase, name, "parquet")
    _client(s3_client).put_object(
        Bucket=bucket, Key=key, Body=buf.getvalue(),
        ContentType="application/vnd.apache.parquet",
    )
    return key


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
