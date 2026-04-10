"""
pipeline_common.py — Shared utilities for backtest.py and evaluate.py.

Config loading, research DB management, predictor metrics.
Data seeding/backfilling lives in alpha-engine-data/collectors/signal_returns.py.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_MIN_IC_SAMPLES = 10
_IC_STD_EPSILON = 1e-8


# ── Config ────────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    search_paths = [
        Path.home() / "alpha-engine-config" / "backtester" / "config.yaml",
        Path(__file__).parent.parent / "alpha-engine-config" / "backtester" / "config.yaml",
        Path(path),
    ]
    resolved = next((p for p in search_paths if p.exists()), None)
    if resolved is None:
        raise FileNotFoundError(f"Config not found. Searched: {[str(p) for p in search_paths]}")
    with open(resolved) as f:
        config = yaml.safe_load(f)
    _validate_config(config, str(resolved))
    return config


def _validate_config(config: dict, path: str) -> None:
    """Validate required config keys exist and warn about common issues."""
    warnings = []
    errors = []

    if not config.get("signals_bucket"):
        errors.append("signals_bucket is required")

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    if not executor_paths:
        warnings.append("executor_paths not set — simulate/param-sweep modes will fail")
    elif not any(os.path.isdir(p) for p in executor_paths):
        warnings.append(
            f"No executor_paths found on disk: {executor_paths}. "
            "simulate/param-sweep modes will fail."
        )

    if not config.get("email_sender") or not config.get("email_recipients"):
        warnings.append("email_sender/email_recipients not set — email reports will be skipped")

    for w in warnings:
        logger.warning("Config (%s): %s", path, w)
    if errors:
        msg = f"Config validation failed ({path}): " + "; ".join(errors)
        raise ValueError(msg)


# ── Research DB ───────────────────────────────────────────────────────────────


def pull_research_db(bucket: str, local_path: str, s3_key: str = "research.db") -> bool:
    """Pull research.db from S3 to local_path. Returns True on success."""
    s3 = boto3.client("s3")
    try:
        s3.download_file(bucket, s3_key, local_path)
        size = os.path.getsize(local_path)
        logger.info("Pulled research.db from s3://%s/%s (%s bytes)", bucket, s3_key, f"{size:,}")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            logger.warning("research.db not found in S3 — signal quality analysis will be skipped")
        else:
            logger.error("Failed to pull research.db: %s", e)
        return False


def init_research_db(db_arg: str | None, config: dict) -> None:
    """Pull or set research_db in config. Mutates config in place."""
    if db_arg:
        config["research_db"] = db_arg
        logger.info("Using local research.db: %s", db_arg)
    else:
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()
        bucket = config.get("signals_bucket", "alpha-engine-research")
        db_pulled = pull_research_db(bucket, tmp_db.name)
        if db_pulled:
            config["research_db"] = tmp_db.name
        else:
            config["research_db"] = None
        config["_db_pull_status"] = "ok" if db_pulled else "failed"


# ── Trades DB ─────────────────────────────────────────────────────────────────


def find_trades_db(config: dict) -> str | None:
    """Find trades.db from executor_paths config."""
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    for p in executor_paths:
        db_path = Path(p) / "trades.db"
        if db_path.exists():
            return str(db_path)
    return None


# ── Predictor metrics (evaluation output) ─────────────────────────────────────


def push_predictor_rolling_metrics(config: dict, db_path: str) -> None:
    """Compute 30-day rolling hit rate and IC, merge into predictor/metrics/latest.json."""
    import sqlite3 as _sqlite3
    from datetime import datetime, timedelta

    bucket = config.get("signals_bucket")
    metrics_key = "predictor/metrics/latest.json"
    if not bucket or not db_path or not os.path.exists(db_path):
        return

    try:
        cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        conn = _sqlite3.connect(db_path)
        df = pd.read_sql_query(
            "SELECT * FROM predictor_outcomes WHERE correct_5d IS NOT NULL "
            "AND prediction_date >= ?",
            conn,
            params=(cutoff,),
        )
        conn.close()
    except (_sqlite3.Error, FileNotFoundError, KeyError) as e:
        logger.warning("push_predictor_rolling_metrics: DB read failed: %s", e)
        return

    if len(df) < 5:
        logger.info("push_predictor_rolling_metrics: < 5 resolved outcomes, skipping S3 update")
        return

    hit_rate = float(pd.to_numeric(df["correct_5d"], errors="coerce").mean())

    df["net_signal"] = (
        pd.to_numeric(df["p_up"], errors="coerce").fillna(0)
        - pd.to_numeric(df["p_down"], errors="coerce").fillna(0)
    )
    df["actual"] = pd.to_numeric(df["actual_5d_return"], errors="coerce")
    valid = df.dropna(subset=["net_signal", "actual"])
    ic_30d = None
    ic_ir_30d = None
    if len(valid) >= _MIN_IC_SAMPLES:
        from scipy.stats import pearsonr
        import numpy as np
        ic_val, _ = pearsonr(valid["net_signal"], valid["actual"])
        ic_30d = round(float(ic_val), 4)
        n_chunks = max(2, len(valid) // 5)
        chunk_size = len(valid) // n_chunks
        chunk_ics = np.array([
            pearsonr(
                valid["net_signal"].iloc[i * chunk_size:(i + 1) * chunk_size],
                valid["actual"].iloc[i * chunk_size:(i + 1) * chunk_size],
            )[0]
            for i in range(n_chunks)
        ])
        ic_ir_30d = round(float(chunk_ics.mean() / (chunk_ics.std() + _IC_STD_EPSILON)), 3)

    s3 = boto3.client("s3")
    existing: dict = {}
    try:
        resp = s3.get_object(Bucket=bucket, Key=metrics_key)
        existing = json.loads(resp["Body"].read())
    except s3.exceptions.NoSuchKey:
        # Expected on first run — metrics file doesn't exist yet.
        logger.info("%s not found in S3 — initializing new metrics file", metrics_key)
    except Exception as e:
        # Non-NoSuchKey errors (S3 permissions, network, parse errors) mean
        # we might be overwriting valid existing metrics with a partial set,
        # or the entire metrics pipeline is broken. Raise so flow-doctor
        # captures it and downstream rolling-window updates don't silently
        # corrupt the metrics history.
        logger.error(
            "Failed to read existing predictor metrics from s3://%s/%s: %s",
            bucket, metrics_key, e, exc_info=True,
        )
        raise

    from datetime import datetime
    existing["hit_rate_30d_rolling"] = round(hit_rate, 4)
    existing["ic_30d"] = ic_30d
    existing["ic_ir_30d"] = ic_ir_30d
    existing["rolling_metrics_updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    existing["rolling_n"] = len(df)

    try:
        s3.put_object(
            Bucket=bucket,
            Key=metrics_key,
            Body=json.dumps(existing, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        logger.info(
            "Predictor rolling metrics updated: hit_rate=%.3f  ic_30d=%s  n=%d",
            hit_rate, ic_30d, len(df),
        )
    except Exception as e:
        # Write failure means the rolling metrics never get persisted — next
        # run reads stale values and the retrain alert evaluator bases its
        # decision on week-old IC / hit-rate. Raise so flow-doctor captures
        # it; previously this was a silent warning that kept the pipeline
        # green even when metrics went stale for weeks.
        logger.error(
            "push_predictor_rolling_metrics: S3 write failed for s3://%s/%s: %s",
            bucket, metrics_key, e, exc_info=True,
        )
        raise


# ── Sector map ────────────────────────────────────────────────────────────────


def load_sector_map(config: dict) -> dict[str, str] | None:
    """Load sector_map.json from predictor repo or S3."""
    predictor_paths = config.get("predictor_paths", [])
    if isinstance(predictor_paths, str):
        predictor_paths = [predictor_paths]
    for p in predictor_paths:
        map_path = Path(p) / "data" / "cache" / "sector_map.json"
        if map_path.exists():
            with open(map_path) as f:
                return json.load(f)

    try:
        s3 = boto3.client("s3")
        bucket = config.get("signals_bucket", "alpha-engine-research")
        resp = s3.get_object(
            Bucket=bucket, Key="predictor/price_cache/sector_map.json"
        )
        return json.load(resp["Body"])
    except Exception as e:
        logger.warning("Could not load sector_map.json: %s", e)
        return None
