"""
backtest.py — CLI entry point for alpha-engine-backtester.

Usage:
    # Mode 1 — Signal quality report
    python backtest.py --mode signal-quality

    # Mode 2 — Portfolio simulation (requires executor_path in config.yaml)
    python backtest.py --mode simulate

    # Full report (both modes)
    python backtest.py --mode all

    # Predictor-only backtest (2y historical, no LLM calls)
    python backtest.py --mode predictor-backtest

    # Upload results to S3
    python backtest.py --mode signal-quality --upload

Options:
    --mode          signal-quality | simulate | param-sweep | all | predictor-backtest
    --config        path to config.yaml (default: ./config.yaml)
    --db            path to local research.db (skips S3 pull; useful locally)
    --upload        upload results to S3
    --date          run date label for output (default: today)
    --log-level     DEBUG | INFO | WARNING (default: INFO)

research.db:
    Lives in S3 at s3://{signals_bucket}/research.db (Lambda writes it after each run).
    backtest.py pulls a fresh copy to a temp file at startup — read-only, never written back.
    Override with --db for local development.
"""

import argparse
import json
import logging
import tempfile
import os
import time as _time
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

from analysis import signal_quality, regime_analysis, score_analysis, attribution, param_sweep
from analysis import veto_analysis
from analysis import universe_returns
from analysis import end_to_end
from analysis import trigger_scorecard, alpha_distribution, veto_value
from analysis import shadow_book as shadow_book_analysis
from analysis import exit_timing, macro_eval
from analysis import sizing_ab
from optimizer import weight_optimizer, executor_optimizer, research_optimizer
from optimizer import trigger_optimizer, predictor_sizing_optimizer
from optimizer import scanner_optimizer, pipeline_optimizer
from emailer import send_report_email
from reporter import build_report, save, upload_to_s3
# pipeline_common: shared utilities also used by evaluate.py
import pipeline_common  # noqa: F401 — imported for evaluate.py reuse

logger = logging.getLogger(__name__)

_MIN_IC_SAMPLES = 10     # minimum resolved outcomes before computing Information Coefficient
_IC_STD_EPSILON = 1e-8   # avoid division by zero in IC/IR computation


def load_config(path: str) -> dict:
    from pathlib import Path
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

    # Required for all modes
    if not config.get("signals_bucket"):
        errors.append("signals_bucket is required")

    # Required for simulate/param-sweep modes
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

    # Email (optional but flagged)
    if not config.get("email_sender") or not config.get("email_recipients"):
        warnings.append("email_sender/email_recipients not set — email reports will be skipped")

    for w in warnings:
        logger.warning("Config (%s): %s", path, w)
    if errors:
        msg = f"Config validation failed ({path}): " + "; ".join(errors)
        raise ValueError(msg)


def pull_research_db(bucket: str, local_path: str, s3_key: str = "research.db") -> bool:
    """
    Pull research.db from S3 to local_path. Returns True on success.
    research.db is written to S3 by the Lambda research pipeline after each run.
    """
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


def _push_predictor_rolling_metrics(config: dict, db_path: str) -> None:
    """
    Compute 30-day rolling hit rate and IC from resolved predictor_outcomes rows
    and merge into predictor/metrics/latest.json in S3.

    Called after _backfill_predictor_outcomes() so correct_5d is populated.
    Silent on any failure — never blocks the backtest run.
    """
    import json
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
        logger.warning("_push_predictor_rolling_metrics: DB read failed: %s", e)
        return

    if len(df) < 5:
        logger.info("_push_predictor_rolling_metrics: < 5 resolved outcomes, skipping S3 update")
        return

    # hit_rate_30d_rolling
    hit_rate = float(pd.to_numeric(df["correct_5d"], errors="coerce").mean())

    # ic_30d — Pearson correlation between net directional signal and actual return
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
        ic_val, _ = pearsonr(valid["net_signal"], valid["actual"])
        ic_30d = round(float(ic_val), 4)
        # IC IR over weekly chunks
        n_chunks = max(2, len(valid) // 5)
        chunk_size = len(valid) // n_chunks
        import numpy as np
        chunk_ics = np.array([
            pearsonr(
                valid["net_signal"].iloc[i * chunk_size:(i + 1) * chunk_size],
                valid["actual"].iloc[i * chunk_size:(i + 1) * chunk_size],
            )[0]
            for i in range(n_chunks)
        ])
        ic_ir_30d = round(float(chunk_ics.mean() / (chunk_ics.std() + _IC_STD_EPSILON)), 3)

    try:
        s3 = boto3.client("s3")
        # Read existing metrics, merge rolling stats on top
        existing: dict = {}
        try:
            resp = s3.get_object(Bucket=bucket, Key=metrics_key)
            existing = json.loads(resp["Body"].read())
        except s3.exceptions.NoSuchKey:
            pass  # fresh file or not yet written
        except Exception as e:
            logger.warning("Failed to read existing predictor metrics from S3: %s", e)

        existing["hit_rate_30d_rolling"] = round(hit_rate, 4)
        existing["ic_30d"] = ic_30d
        existing["ic_ir_30d"] = ic_ir_30d
        existing["rolling_metrics_updated_at"] = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        existing["rolling_n"] = len(df)

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
        logger.warning("_push_predictor_rolling_metrics: S3 write failed: %s", e)


def _seed_predictor_outcomes(config: dict) -> None:
    """Seed predictor_outcomes rows from S3 predictions/*.json files.

    Reads each predictions/{date}.json from S3 and inserts rows that
    don't already exist in the DB. This bridges the gap between the
    predictor (which writes to S3) and the backtester (which needs
    rows in research.db to backfill actual returns).
    """
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    bucket = config.get("signals_bucket")
    if not db_path or not os.path.exists(db_path) or not bucket:
        return
    try:
        s3 = boto3.client("s3")
        # List all prediction files
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="predictor/predictions/", Delimiter="/")
        keys = [obj["Key"] for obj in resp.get("Contents", [])
                if obj["Key"].endswith(".json") and "latest" not in obj["Key"]]

        if not keys:
            logger.info("No prediction files found in S3 — skipping seed")
            return

        conn = _sqlite3.connect(db_path)
        existing = {
            (r[0], r[1]) for r in
            conn.execute("SELECT symbol, prediction_date FROM predictor_outcomes").fetchall()
        }

        inserted = 0
        for key in keys:
            try:
                obj = s3.get_object(Bucket=bucket, Key=key)
                data = json.loads(obj["Body"].read())
                pred_date = data.get("date") or key.split("/")[-1].replace(".json", "")
                for p in data.get("predictions", []):
                    ticker = p.get("ticker")
                    if not ticker or (ticker, pred_date) in existing:
                        continue
                    conn.execute(
                        """INSERT INTO predictor_outcomes
                           (symbol, prediction_date, predicted_direction, prediction_confidence,
                            p_up, p_flat, p_down, score_modifier_applied)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            ticker, pred_date,
                            p.get("predicted_direction"),
                            p.get("prediction_confidence"),
                            p.get("p_up"),
                            p.get("p_flat"),
                            p.get("p_down"),
                            0.0,
                        ),
                    )
                    existing.add((ticker, pred_date))
                    inserted += 1
            except (ClientError, json.JSONDecodeError, KeyError) as e:
                logger.info("Skipping prediction file %s: %s", key, e)
                continue

        conn.commit()
        conn.close()
        if inserted:
            logger.info("Seeded %d predictor_outcomes rows from %d S3 prediction files", inserted, len(keys))
        else:
            logger.info("All predictor_outcomes already seeded (%d files checked)", len(keys))
    except Exception as e:
        logger.warning("_seed_predictor_outcomes: %s", e)


def _seed_score_performance(config: dict) -> None:
    """
    Seed score_performance rows from S3 signals/{date}/signals.json files.

    The research pipeline's record_new_buy_scores() should insert these, but
    it was never wired into the Lambda handler. This function bridges the gap
    by reading signals from S3 and inserting any BUY-rated stocks that don't
    already have a score_performance row.

    Prices are fetched via yfinance to record entry price.
    """
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    bucket = config.get("signals_bucket")
    if not db_path or not os.path.exists(db_path) or not bucket:
        return
    conn = None
    try:
        import yfinance as yf
        from loaders import signal_loader

        conn = _sqlite3.connect(db_path)

        # Get existing (symbol, score_date) pairs
        existing = {
            (r[0], r[1]) for r in
            conn.execute("SELECT symbol, score_date FROM score_performance").fetchall()
        }

        # Load all signal dates from S3
        signal_dates = signal_loader.list_dates(bucket)

        # Collect rows that need inserting
        rows_to_insert = []
        for sig_date in signal_dates:
            try:
                signals = signal_loader.load(bucket, sig_date)
            except FileNotFoundError:
                continue

            # Extract BUY-rated stocks from universe
            for stock in signals.get("universe", []):
                ticker = stock.get("ticker")
                score = stock.get("score", 0)
                rating = stock.get("rating", "")
                if not ticker or rating != "BUY" or (ticker, sig_date) in existing:
                    continue
                rows_to_insert.append((ticker, sig_date, score))

            # Also check signals dict (v1 format)
            sigs = signals.get("signals", {})
            if isinstance(sigs, dict):
                for ticker, s in sigs.items():
                    score = s.get("score", 0)
                    rating = s.get("rating", "")
                    if rating != "BUY" or (ticker, sig_date) in existing:
                        continue
                    rows_to_insert.append((ticker, sig_date, score))

        if not rows_to_insert:
            conn.close()
            logger.info("All score_performance rows already seeded")
            return

        # Fetch entry prices via yfinance
        unique_tickers = list({r[0] for r in rows_to_insert})
        min_date = min(r[1] for r in rows_to_insert)
        price_data = yf.download(
            tickers=unique_tickers,
            start=min_date,
            end=str(date.today()),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=300,
        )

        def _get_close(ticker: str, dt_str: str) -> float | None:
            """Get closing price on or near the given date."""
            ts = pd.Timestamp(dt_str)
            # Try exact date, then next 3 trading days (for weekend signals)
            for offset in range(4):
                try_ts = ts + pd.Timedelta(days=offset)
                try:
                    if len(unique_tickers) == 1:
                        return float(price_data["Close"].loc[try_ts])
                    return float(price_data[ticker]["Close"].loc[try_ts])
                except (KeyError, TypeError):
                    continue
            return None

        inserted = 0
        for ticker, sig_date, score in rows_to_insert:
            if score is None:
                continue
            price = _get_close(ticker, sig_date)
            if price is None:
                continue
            conn.execute(
                "INSERT OR IGNORE INTO score_performance (symbol, score_date, score, price_on_date) "
                "VALUES (?, ?, ?, ?)",
                (ticker, sig_date, round(float(score), 2), round(price, 2)),
            )
            inserted += 1

        conn.commit()
        conn.close()
        if inserted:
            logger.info("Seeded %d score_performance rows from %d signal dates", inserted, len(signal_dates))
        else:
            logger.info("No new score_performance rows to insert (prices unavailable)")
    except Exception as e:
        logger.warning("_seed_score_performance: %s", e)
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def _ensure_5d_columns(conn) -> None:
    """Add 5d return columns to score_performance if they don't exist yet."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(score_performance)").fetchall()}
    for col, col_type in [
        ("price_5d", "REAL"),
        ("return_5d", "REAL"),
        ("spy_5d_return", "REAL"),
        ("beat_spy_5d", "INTEGER"),
        ("eval_date_5d", "TEXT"),
    ]:
        if col not in cols:
            conn.execute(f"ALTER TABLE score_performance ADD COLUMN {col} {col_type}")
    conn.commit()


def _backfill_score_performance_returns(config: dict) -> None:
    """
    Backfill 5d, 10d, and 30d returns for score_performance rows missing them.

    The research pipeline's run_performance_checks() does this on each Lambda run,
    but seeded rows (from _seed_score_performance) won't have returns until the
    next research run. This function fills them in via yfinance so the backtester
    has complete data.
    """
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return
    try:
        import yfinance as yf

        conn = _sqlite3.connect(db_path)
        _ensure_5d_columns(conn)

        # Repair: fix beat_spy columns where return exists but beat_spy is NULL
        # (caused by earlier bug that set beat_spy=NULL instead of 0 for losses)
        for horizon in ("5d", "10d", "30d"):
            repaired = conn.execute(
                f"UPDATE score_performance "
                f"SET beat_spy_{horizon} = CASE "
                f"  WHEN return_{horizon} > spy_{horizon}_return THEN 1 ELSE 0 END "
                f"WHERE return_{horizon} IS NOT NULL "
                f"  AND spy_{horizon}_return IS NOT NULL "
                f"  AND beat_spy_{horizon} IS NULL",
            ).rowcount
            if repaired:
                logger.info("Repaired %d beat_spy_%s values (NULL→0 for losses)", repaired, horizon)
        conn.commit()

        pending = pd.read_sql_query(
            "SELECT symbol, score_date, price_on_date FROM score_performance "
            "WHERE return_5d IS NULL OR return_10d IS NULL OR return_30d IS NULL",
            conn,
        )
        if pending.empty:
            conn.close()
            return

        today_ts = pd.Timestamp(date.today())

        # Only backfill rows where enough time has passed
        rows_5d = []
        rows_10d = []
        rows_30d = []
        for _, row in pending.iterrows():
            score_ts = pd.Timestamp(row["score_date"])
            eval_5d = score_ts + pd.offsets.BDay(5)
            eval_10d = score_ts + pd.offsets.BDay(10)
            eval_30d = score_ts + pd.offsets.BDay(30)
            if eval_5d <= today_ts:
                rows_5d.append(row)
            if eval_10d <= today_ts:
                rows_10d.append(row)
            if eval_30d <= today_ts:
                rows_30d.append(row)

        if not rows_5d and not rows_10d and not rows_30d:
            conn.close()
            logger.info("No score_performance rows eligible for return backfill yet")
            return

        # Batch yfinance download
        all_rows = rows_5d + rows_10d + rows_30d
        tickers = list({r["symbol"] for r in all_rows})
        all_tickers = tickers + (["SPY"] if "SPY" not in tickers else [])
        min_date = min(r["score_date"] for r in all_rows)

        price_data = yf.download(
            tickers=all_tickers,
            start=min_date,
            end=str(date.today()),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=300,
        )

        def _get_close(ticker: str, dt: pd.Timestamp) -> float | None:
            # Try exact date, then next 2 business days
            for offset in range(3):
                try_ts = dt + pd.Timedelta(days=offset)
                try:
                    if len(all_tickers) == 1:
                        return float(price_data["Close"].loc[try_ts])
                    return float(price_data[ticker]["Close"].loc[try_ts])
                except (KeyError, TypeError):
                    continue
            return None

        updated = 0

        for row in rows_5d:
            score_ts = pd.Timestamp(row["score_date"])
            eval_ts = score_ts + pd.offsets.BDay(5)
            exit_price = _get_close(row["symbol"], eval_ts)
            spy_entry = _get_close("SPY", score_ts)
            spy_exit = _get_close("SPY", eval_ts)

            if exit_price is None or row["price_on_date"] is None:
                continue

            ret_5d = (exit_price / row["price_on_date"]) - 1
            spy_ret = (spy_exit / spy_entry) - 1 if spy_entry and spy_exit else None
            beat = (1 if ret_5d > spy_ret else 0) if spy_ret is not None else None

            conn.execute(
                "UPDATE score_performance SET price_5d=?, return_5d=?, "
                "spy_5d_return=?, beat_spy_5d=?, eval_date_5d=? "
                "WHERE symbol=? AND score_date=? AND return_5d IS NULL",
                (
                    round(exit_price, 2),
                    round(ret_5d * 100, 2),
                    round(spy_ret * 100, 2) if spy_ret is not None else None,
                    beat,
                    str(eval_ts.date()),
                    row["symbol"], row["score_date"],
                ),
            )
            updated += 1

        for row in rows_10d:
            score_ts = pd.Timestamp(row["score_date"])
            eval_ts = score_ts + pd.offsets.BDay(10)
            exit_price = _get_close(row["symbol"], eval_ts)
            spy_entry = _get_close("SPY", score_ts)
            spy_exit = _get_close("SPY", eval_ts)

            if exit_price is None or row["price_on_date"] is None:
                continue

            ret_10d = (exit_price / row["price_on_date"]) - 1
            spy_ret = (spy_exit / spy_entry) - 1 if spy_entry and spy_exit else None
            beat = (1 if ret_10d > spy_ret else 0) if spy_ret is not None else None

            conn.execute(
                "UPDATE score_performance SET price_10d=?, return_10d=?, "
                "spy_10d_return=?, beat_spy_10d=?, eval_date_10d=? "
                "WHERE symbol=? AND score_date=? AND return_10d IS NULL",
                (
                    round(exit_price, 2),
                    round(ret_10d * 100, 2),
                    round(spy_ret * 100, 2) if spy_ret is not None else None,
                    beat,
                    str(eval_ts.date()),
                    row["symbol"], row["score_date"],
                ),
            )
            updated += 1

        for row in rows_30d:
            score_ts = pd.Timestamp(row["score_date"])
            eval_ts = score_ts + pd.offsets.BDay(30)
            exit_price = _get_close(row["symbol"], eval_ts)
            spy_entry = _get_close("SPY", score_ts)
            spy_exit = _get_close("SPY", eval_ts)

            if exit_price is None or row["price_on_date"] is None:
                continue

            ret_30d = (exit_price / row["price_on_date"]) - 1
            spy_ret = (spy_exit / spy_entry) - 1 if spy_entry and spy_exit else None
            beat = (1 if ret_30d > spy_ret else 0) if spy_ret is not None else None

            conn.execute(
                "UPDATE score_performance SET price_30d=?, return_30d=?, "
                "spy_30d_return=?, beat_spy_30d=?, eval_date_30d=? "
                "WHERE symbol=? AND score_date=? AND return_30d IS NULL",
                (
                    round(exit_price, 2),
                    round(ret_30d * 100, 2),
                    round(spy_ret * 100, 2) if spy_ret is not None else None,
                    beat,
                    str(eval_ts.date()),
                    row["symbol"], row["score_date"],
                ),
            )
            updated += 1

        conn.commit()

        if updated:
            logger.info("Backfilled returns for %d score_performance rows via yfinance", updated)

        # Completeness check: warn if eligible rows still lack returns
        still_pending = pd.read_sql_query(
            "SELECT COUNT(*) as n FROM score_performance "
            "WHERE (return_5d IS NULL AND score_date <= date('now', '-10 days')) "
            "   OR (return_10d IS NULL AND score_date <= date('now', '-14 days')) "
            "   OR (return_30d IS NULL AND score_date <= date('now', '-45 days'))",
            conn,
        )
        n_stale = int(still_pending.iloc[0]["n"]) if not still_pending.empty else 0
        if n_stale > 0:
            logger.warning(
                "Score performance backfill gap: %d rows still missing returns "
                "(eligible for backfill but yfinance failed). Accuracy metrics "
                "may be computed on incomplete data.",
                n_stale,
            )

        conn.close()
    except Exception as e:
        logger.warning("_backfill_score_performance_returns: %s", e)
        try:
            conn.close()
        except Exception:
            pass


def _populate_universe_returns(config: dict) -> dict | None:
    """
    Populate universe_returns table with forward returns for all ~900 S&P stocks.

    Uses polygon.io grouped-daily endpoint (1 API call per date) to fetch
    close prices for the entire US market. Computes 5d/10d forward returns,
    SPY benchmark, and sector ETF returns for each evaluation date.

    Only processes dates that have signals but are missing from universe_returns.
    """
    db_path = config.get("research_db")
    bucket = config.get("signals_bucket")
    if not db_path or not os.path.exists(db_path) or not bucket:
        return None

    try:
        from polygon_client import polygon_client as get_polygon_client
        from loaders import signal_loader

        client = get_polygon_client()
        signal_dates = signal_loader.list_dates(bucket)

        # Load sector_map if available
        sector_map = _load_sector_map(config)

        result = universe_returns.build_and_insert(
            db_path=db_path,
            eval_dates=signal_dates,
            polygon_client=client,
            sector_map=sector_map,
        )
        if result.get("rows_inserted", 0) > 0:
            logger.info(
                "universe_returns: inserted %d rows across %d dates",
                result["rows_inserted"], result["dates_processed"],
            )
        return result
    except Exception as e:
        logger.warning("_populate_universe_returns: %s", e)
        return {"status": "error", "error": str(e)}


def _load_sector_map(config: dict) -> dict[str, str] | None:
    """Load sector_map.json from predictor repo or S3."""
    # Try local predictor repo first
    predictor_paths = config.get("predictor_paths", [])
    if isinstance(predictor_paths, str):
        predictor_paths = [predictor_paths]
    for p in predictor_paths:
        map_path = Path(p) / "data" / "cache" / "sector_map.json"
        if map_path.exists():
            import json as _json
            with open(map_path) as f:
                return _json.load(f)

    # Try S3
    try:
        import json as _json
        s3 = boto3.client("s3")
        bucket = config.get("signals_bucket", "alpha-engine-research")
        resp = s3.get_object(
            Bucket=bucket, Key="predictor/price_cache/sector_map.json"
        )
        return _json.load(resp["Body"])
    except Exception as e:
        logger.warning("Could not load sector_map.json: %s", e)
        return None


def _find_trades_db(config: dict) -> str | None:
    """Find trades.db from executor_paths config."""
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    for p in executor_paths:
        db_path = Path(p) / "trades.db"
        if db_path.exists():
            return str(db_path)
    # Try S3-pulled copy in temp dir
    return None


def _backfill_predictor_outcomes(config: dict, df_base: pd.DataFrame) -> None:
    """
    Backfill actual_5d_return and correct_5d for pending predictor_outcomes rows.

    Fetches actual 5-trading-day returns from yfinance for each pending prediction.
    A prediction is eligible for backfill when 5+ trading days have elapsed since
    the prediction date.
    """
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return
    try:
        import yfinance as yf
    except ImportError as ie:
        logger.warning("_backfill_predictor_outcomes: missing dependency: %s", ie)
        return

    try:
        conn = _sqlite3.connect(db_path)
        pending = pd.read_sql_query(
            "SELECT id, symbol, prediction_date, predicted_direction "
            "FROM predictor_outcomes WHERE actual_5d_return IS NULL",
            conn,
        )
        if pending.empty:
            conn.close()
            return

        # Determine which predictions are old enough (5 business days elapsed)
        today_ts = pd.Timestamp(date.today())
        eligible = []
        for _, row in pending.iterrows():
            pred_ts = pd.Timestamp(row["prediction_date"])
            eval_ts = pred_ts + pd.offsets.BDay(5)
            if eval_ts <= today_ts:
                eligible.append({**row, "_eval_date": eval_ts})

        if not eligible:
            conn.close()
            logger.info("No predictor outcomes eligible for backfill yet (need 5 trading days)")
            return

        # Batch yfinance download: all unique tickers + SPY
        eligible_df = pd.DataFrame(eligible)
        tickers = list(eligible_df["symbol"].unique())
        all_tickers = tickers + ["SPY"] if "SPY" not in tickers else tickers
        min_date = eligible_df["prediction_date"].min()

        price_data = yf.download(
            tickers=all_tickers,
            start=min_date,
            end=str(date.today()),
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="ticker",
            threads=True,
            timeout=300,
        )

        def _get_close(ticker: str, dt: pd.Timestamp) -> float | None:
            try:
                if len(all_tickers) == 1:
                    return float(price_data["Close"].loc[dt])
                return float(price_data[ticker]["Close"].loc[dt])
            except (KeyError, TypeError):
                return None

        resolved = 0
        for row in eligible:
            pred_date_ts = pd.Timestamp(row["prediction_date"])
            eval_date_ts = row["_eval_date"]

            entry_price = _get_close(row["symbol"], pred_date_ts)
            exit_price = _get_close(row["symbol"], eval_date_ts)
            spy_entry = _get_close("SPY", pred_date_ts)
            spy_exit = _get_close("SPY", eval_date_ts)

            if entry_price is None or exit_price is None:
                continue

            actual_return = (exit_price / entry_price) - 1
            spy_return = (spy_exit / spy_entry) - 1 if spy_entry and spy_exit else 0
            # actual_5d_return stored as market-relative (alpha)
            actual_alpha = actual_return - spy_return

            direction = row["predicted_direction"]
            if direction == "UP":
                correct = 1 if actual_alpha > 0 else 0
            elif direction == "DOWN":
                correct = 1 if actual_alpha < 0 else 0
            elif direction == "FLAT":
                correct = 1 if abs(actual_alpha) < 0.01 else 0
            else:
                continue

            conn.execute(
                "UPDATE predictor_outcomes SET actual_5d_return=?, correct_5d=? "
                "WHERE symbol=? AND prediction_date=?",
                (round(actual_alpha * 100, 4), correct, row["symbol"], row["prediction_date"]),
            )
            resolved += 1

        conn.commit()
        conn.close()
        if resolved:
            logger.info("Backfilled %d/%d predictor outcomes via yfinance", resolved, len(eligible))
        else:
            logger.info("No predictor outcomes could be resolved (price data missing)")
    except Exception as e:
        logger.warning("_backfill_predictor_outcomes: %s", e)


def run_signal_quality(config: dict) -> tuple[dict, list, list, dict]:
    """
    Run Mode 1: aggregate score_performance from research.db.

    Returns (sq_result, regime_rows, score_rows, attr_result).
    """
    db_path = config.get("research_db")
    min_samples = config.get("min_samples", 5)
    thresholds = config.get("score_thresholds", [60, 65, 70, 75, 80, 85, 90])

    if not db_path:
        logger.error("research_db not set in config and --db not provided")
        sq_result = {"status": "db_not_found", "error": "research_db path not configured"}
        return sq_result, [], [], {"status": "insufficient_data", "note": "research_db not configured"}, None

    logger.info("Loading score_performance from %s", db_path)

    # Seed score_performance from S3 signals before loading
    # (bridges gap where research handler doesn't call record_new_buy_scores)
    _seed_score_performance(config)
    _backfill_score_performance_returns(config)

    try:
        df_base = signal_quality.load_score_performance(db_path)
        sq_result = signal_quality.compute_accuracy(df_base, min_samples=min_samples)
    except FileNotFoundError as e:
        logger.error("research.db not found: %s", e)
        sq_result = {"status": "db_not_found", "error": str(e)}
        return sq_result, [], [], {"status": "insufficient_data", "note": "research.db not found"}, None

    try:
        df_regime = regime_analysis.load_with_regime(db_path)
        regime_rows = regime_analysis.accuracy_by_regime(df_regime, min_samples=min_samples)
    except Exception as e:
        logger.warning("Regime analysis failed: %s", e)
        regime_rows = []

    score_rows = score_analysis.accuracy_by_threshold(
        df_base, thresholds=thresholds, min_samples=min_samples
    )

    attr_result = attribution.compute_attribution(df_base)
    _seed_predictor_outcomes(config)
    _backfill_predictor_outcomes(config, df_base)
    _push_predictor_rolling_metrics(config, config.get("research_db", ""))
    # universe_returns population now handled by alpha-engine-data (Phase 1)

    # Phase 2: Production model health monitoring
    _production_health = None
    _calibration = None
    try:
        from analysis.production_health import compute_production_health, compute_calibration_validation
        _db = config.get("research_db", "")
        _bucket = config.get("signals_bucket", "alpha-engine-research")
        if _db and os.path.exists(_db):
            _production_health = compute_production_health(_db, _bucket)
            _calibration = compute_calibration_validation(_db, _bucket)
    except Exception as _ph_exc:
        logger.warning("Production health analysis failed (non-fatal): %s", _ph_exc)

    # Phase 3: Feature importance drift detection
    _feature_drift = None
    try:
        from analysis.feature_drift import compute_feature_drift
        _db = config.get("research_db", "")
        _bucket = config.get("signals_bucket", "alpha-engine-research")
        if _db and os.path.exists(_db):
            _feature_drift = compute_feature_drift(_db, _bucket)
    except Exception as _fd_exc:
        logger.warning("Feature drift analysis failed (non-fatal): %s", _fd_exc)

    # Phase 5: Retrain alert evaluation
    try:
        from analysis.retrain_alert import evaluate_retrain_triggers, send_retrain_alert
        _bucket = config.get("signals_bucket", "alpha-engine-research")
        alert = evaluate_retrain_triggers(_production_health, _feature_drift, _calibration)
        if alert.get("triggered"):
            send_retrain_alert(alert, config, _bucket)
        else:
            logger.info("Retrain alert: %s", alert.get("summary", "no triggers"))
    except Exception as _ra_exc:
        logger.warning("Retrain alert evaluation failed (non-fatal): %s", _ra_exc)

    return sq_result, regime_rows, score_rows, attr_result, df_base


def _read_current_weights(config: dict) -> dict:
    """
    Read current scoring weights — the values the system is *actually* using.

    Priority order:
      1. S3 config/scoring_weights.json  (last backtester-optimized weights)
      2. Local research repo universe.yaml  (initial config)
      3. Hardcoded defaults  (bootstrap only)
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # 1. Try S3 — last known optimal from backtester
    try:
        import boto3
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="config/scoring_weights.json")
        data = json.loads(obj["Body"].read())
        weights = {k: float(data[k]) for k in ("news", "research") if k in data}
        if len(weights) == 2:
            logger.info(
                "Current scoring weights from S3 (updated %s): %s",
                data.get("updated_at", "unknown"), weights,
            )
            return weights
    except Exception as e:
        logger.info("No scoring weights in S3 (%s), trying local research repo...", e)

    # 2. Try local research repo universe.yaml
    research_paths = config.get("research_paths", [])
    if isinstance(research_paths, str):
        research_paths = [research_paths]
    research_path = next((p for p in research_paths if os.path.isdir(p)), None)

    if research_path:
        universe_yaml = os.path.join(research_path, "config", "universe.yaml")
        try:
            with open(universe_yaml) as f:
                universe = yaml.safe_load(f)
            weights = universe.get("scoring_weights", {})
            if weights:
                logger.info("Scoring weights read from %s: %s", universe_yaml, weights)
                return weights
        except Exception as e:
            logger.warning("Could not read universe.yaml from %s: %s", universe_yaml, e)
    else:
        logger.warning(
            "research_paths not found on disk — using default scoring weights. "
            "Add research repo path to research_paths in config.yaml for accurate readings."
        )

    # 3. Hardcoded defaults (bootstrap only)
    return weight_optimizer._cfg.get("default_weights", weight_optimizer._DEFAULT_WEIGHTS).copy()


def run_weight_optimizer(config: dict, df_base: pd.DataFrame, freeze: bool = False) -> dict:
    """
    Run the weight optimizer: join sub-scores from signals.json in S3 with
    score_performance outcomes, then suggest revised scoring weights.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    current_weights = _read_current_weights(config)
    min_samples = config.get("weight_optimizer_min_samples", 30)

    try:
        df_with_sub = weight_optimizer.load_with_subscores(df_base, bucket)
        result = weight_optimizer.compute_weights(
            df_with_sub,
            current_weights=current_weights,
            min_samples=min_samples,
            bucket=bucket,
        )
        if freeze:
            result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
        else:
            apply_result = weight_optimizer.apply_weights(result, bucket)
            result["apply_result"] = apply_result
        return result
    except Exception as e:
        logger.warning("Weight optimizer failed: %s", e)
        return {"status": "error", "error": str(e)}


def _setup_simulation(config: dict) -> tuple:
    """
    Resolve executor path, import executor modules, load signal dates, build price matrix.

    Returns (executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker).
    price_matrix is None when fewer than min_simulation_dates are available or no prices found.
    ohlcv_by_ticker: {ticker: [{date, open, high, low, close}, ...]} for strategy layer.
    """
    import sys
    import pandas as pd

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        raise ValueError(
            f"None of the executor_paths exist: {executor_paths}. "
            "Add the alpha-engine repo root to executor_paths in config.yaml."
        )
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient
    from loaders import signal_loader, price_loader

    bucket = config.get("signals_bucket", "alpha-engine-research")
    min_dates = config.get("min_simulation_dates", 5)
    init_cash = float(config.get("init_cash", 1_000_000.0))

    dates = signal_loader.list_dates(bucket)
    logger.info("Simulation setup: %d signal dates available in S3", len(dates))

    if len(dates) < min_dates:
        logger.warning(
            "Only %d signal dates available (need %d) — simulation skipped",
            len(dates), min_dates,
        )
        return executor_run, SimulatedIBKRClient, dates, None, init_cash, {}

    ohlcv_by_ticker = {}
    logger.info("Building price matrix for %d dates (yfinance fallback)...", len(dates))
    price_matrix = price_loader.build_matrix(dates, bucket, _ohlcv_out=ohlcv_by_ticker)

    if price_matrix.empty:
        return executor_run, SimulatedIBKRClient, dates, None, init_cash, {}

    logger.info("OHLCV captured for %d tickers (strategy layer)", len(ohlcv_by_ticker))
    return executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker


def _run_simulation_loop(
    executor_run,
    SimulatedIBKRClient,
    dates: list[str],
    price_matrix,
    config: dict,
    ohlcv_by_ticker: dict | None = None,
    signals_by_date: dict | None = None,
    spy_prices: pd.Series | None = None,
) -> dict:
    """
    Run one full simulation pass with the given config and pre-built price matrix.

    A fresh SimulatedIBKRClient is created per call so param-sweep combinations
    start from the same initial state. Prices are swapped per date; positions
    and NAV carry forward across dates within a single run.

    ohlcv_by_ticker: full OHLCV histories for strategy layer (ATR trailing stops).
        Filtered to <= signal_date before each executor call to prevent lookahead.
    signals_by_date: optional pre-built signals for each date (predictor-only mode).
        When provided, uses these instead of loading from S3 via signal_loader.
    """
    import pandas as pd
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_portfolio_stats

    init_cash = float(config.get("init_cash", 1_000_000.0))
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # Staleness circuit breaker: halt if price data is too old for reliable simulation
    if getattr(price_matrix, "attrs", {}).get("stale_circuit_break"):
        return {
            "status": "stale_prices",
            "staleness_warning": price_matrix.attrs.get("staleness_warning"),
            "note": "Price data too stale for reliable simulation",
        }

    # Build config_override from swept params that need to reach the executor
    config_override = _build_config_override(config)

    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
    all_orders = []
    dates_simulated = 0
    skip_reasons = {"no_price_index": 0, "empty_prices": 0, "no_signals": 0}

    # Use signals_by_date keys as iteration dates when available
    if signals_by_date is not None:
        sim_dates = sorted(signals_by_date.keys())
    else:
        sim_dates = dates

    for signal_date in sim_dates:
        ts = pd.Timestamp(signal_date)
        if ts not in price_matrix.index:
            # Weekend/holiday signal dates: use next available trading day's prices
            later = price_matrix.index[price_matrix.index > ts]
            if len(later) > 0:
                ts = later[0]
                logger.debug("Signal date %s not in price index — using next trading day %s",
                             signal_date, ts.date())
            else:
                skip_reasons["no_price_index"] += 1
                continue

        date_prices = price_matrix.loc[ts].dropna().to_dict()
        if not date_prices:
            skip_reasons["empty_prices"] += 1
            continue

        # Load signals: from pre-built dict or from S3
        if signals_by_date is not None:
            signals_raw = signals_by_date[signal_date]
        else:
            from loaders import signal_loader
            try:
                signals_raw = signal_loader.load(bucket, signal_date)
            except FileNotFoundError:
                skip_reasons["no_signals"] += 1
                continue

        sim_client._prices = date_prices
        sim_client._simulation_date = signal_date

        # Filter OHLCV histories to <= signal_date (no lookahead)
        price_histories = None
        if ohlcv_by_ticker:
            price_histories = {
                ticker: [b for b in bars if b["date"] <= signal_date]
                for ticker, bars in ohlcv_by_ticker.items()
            }

        orders = executor_run(
            simulate=True,
            ibkr_client=sim_client,
            signals_override=signals_raw,
            price_histories=price_histories,
            config_override=config_override,
        )
        if orders:
            all_orders.extend(orders)
        dates_simulated += 1

    _MIN_SIMULATION_COVERAGE = 0.80

    dates_expected = len(sim_dates)
    coverage = dates_simulated / dates_expected if dates_expected > 0 else 0
    skipped = {k: v for k, v in skip_reasons.items() if v > 0}
    logger.info(
        "Simulation: %d/%d dates (%.0f%% coverage), %d orders%s",
        dates_simulated, dates_expected, coverage * 100, len(all_orders),
        f" — skipped: {skipped}" if skipped else "",
    )

    if dates_expected > 0 and coverage < _MIN_SIMULATION_COVERAGE:
        return {
            "status": "insufficient_coverage",
            "dates_simulated": dates_simulated,
            "dates_expected": dates_expected,
            "coverage": round(coverage, 3),
            "skip_reasons": skipped,
            "note": (
                f"Only {dates_simulated}/{dates_expected} dates simulated "
                f"({coverage:.0%}) — below {_MIN_SIMULATION_COVERAGE:.0%} threshold"
            ),
        }

    if not all_orders:
        return {
            "status": "no_orders",
            "dates_simulated": dates_simulated,
            "dates_expected": dates_expected,
            "coverage": round(coverage, 3),
            "note": "No ENTER signals passed risk rules during the simulation period",
        }

    fees = config.get("simulation_fees", 0.001)
    sim_cfg = config.get("simulation", {})
    slippage_bps = float(sim_cfg.get("slippage_bps", 0))
    assume_next_day_fill = bool(sim_cfg.get("assume_next_day_fill", False))
    pf = orders_to_portfolio(
        all_orders, price_matrix, init_cash=init_cash, fees=fees,
        slippage_bps=slippage_bps, assume_next_day_fill=assume_next_day_fill,
    )
    stats = compute_portfolio_stats(pf, spy_prices=spy_prices)
    # Record simulation assumptions for reporting
    if slippage_bps > 0 or assume_next_day_fill:
        fill_type = "next-day close" if assume_next_day_fill else "same-day close"
        stats["simulation_assumptions"] = f"Fills: {fill_type} + {slippage_bps:.0f}bp slippage"
    stats["status"] = "ok"
    stats["dates_simulated"] = dates_simulated
    stats["dates_expected"] = dates_expected
    stats["coverage"] = round(coverage, 3)
    stats["total_orders"] = len(all_orders)
    if skipped:
        stats["skip_reasons"] = skipped
    # Pass through price data quality metadata for reporting
    if hasattr(price_matrix, 'attrs'):
        if price_matrix.attrs.get("price_gap_warnings"):
            stats["price_gap_warnings"] = price_matrix.attrs["price_gap_warnings"]
        if price_matrix.attrs.get("staleness_warning"):
            stats["staleness_warning"] = price_matrix.attrs["staleness_warning"]
        if price_matrix.attrs.get("unfilled_gaps"):
            stats["unfilled_gaps"] = price_matrix.attrs["unfilled_gaps"]
    return stats


def _seed_grid_with_current(grid: dict, current_params: dict | None) -> dict:
    """
    Inject current S3 executor param values into the sweep grid so the
    optimizer iterates on last week's best rather than searching from
    scratch. Values already in the grid are not duplicated.
    """
    if not current_params:
        return grid

    grid = {k: list(v) for k, v in grid.items()}  # shallow copy
    for key, val in current_params.items():
        if key in grid and val not in grid[key]:
            grid[key].append(val)
            grid[key].sort()
            logger.info("Seeded grid[%s] with current S3 value: %s", key, val)
    return grid


_DIRECT_RISK_PARAMS = {"min_score", "max_position_pct", "drawdown_circuit_breaker"}
_STRATEGY_EXIT_PARAMS = {
    "atr_multiplier": "atr_multiplier",
    "time_decay_reduce_days": "time_decay_reduce_days",
    "time_decay_exit_days": "time_decay_exit_days",
    "profit_take_pct": "profit_take_pct",
}
_RECOGNIZED_SWEEP_PARAMS = _DIRECT_RISK_PARAMS | set(_STRATEGY_EXIT_PARAMS)


def _build_config_override(config: dict) -> dict | None:
    """
    Map flat sweep params in config to the nested executor config structure.

    Sweep grid uses flat keys (e.g. atr_multiplier) but the executor expects
    them nested under strategy.exit_manager. This function builds the override
    dict that executor.main.run(config_override=) can merge.
    """
    override = {}

    # Direct risk params (top-level in executor's risk.yaml)
    for key in _DIRECT_RISK_PARAMS:
        if key in config:
            override[key] = config[key]

    # Strategy params → nested under strategy.exit_manager
    exit_manager_overrides = {}
    for sweep_key, config_key in _STRATEGY_EXIT_PARAMS.items():
        if sweep_key in config:
            exit_manager_overrides[config_key] = config[sweep_key]

    if exit_manager_overrides:
        override["strategy"] = {"exit_manager": exit_manager_overrides}

    # Warn about sweep params present in config but not mapped to executor
    from optimizer.executor_optimizer import SAFE_PARAMS
    sweep_params_in_config = {k for k in config if k in SAFE_PARAMS}
    unmapped = sweep_params_in_config - _RECOGNIZED_SWEEP_PARAMS
    if unmapped:
        logger.warning(
            "Sweep params not mapped to executor config (will be ignored): %s", unmapped
        )

    return override if override else None


def run_simulate(config: dict) -> dict:
    """
    Run Mode 2: replay all historical signal dates through the executor with
    SimulatedIBKRClient, then compute portfolio metrics via vectorbt.

    Returns a stats dict. Returns {"status": "insufficient_data"} if fewer than
    config["min_simulation_dates"] signal dates exist in S3.
    """
    executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv = _setup_simulation(config)
    min_dates = config.get("min_simulation_dates", 5)

    if price_matrix is None:
        return {
            "status": "insufficient_data",
            "dates_available": len(dates),
            "min_required": min_dates,
        }

    return _run_simulation_loop(
        executor_run, SimulatedIBKRClient, dates, price_matrix, config,
        ohlcv_by_ticker=ohlcv,
    )


def run_param_sweep(config: dict) -> pd.DataFrame | None:
    """
    Run Mode 2 across a grid of risk + strategy parameters. Price matrix and
    OHLCV histories are built once and reused for all combinations — only the
    simulation loop re-runs per combo.

    Returns a DataFrame sorted by sharpe_ratio, or an empty DataFrame if
    insufficient data is available.
    """
    import pandas as pd

    executor_run, SimulatedIBKRClient, dates, price_matrix, _, ohlcv = _setup_simulation(config)

    if price_matrix is None:
        logger.warning(
            "Param sweep skipped: only %d signal dates available", len(dates)
        )
        return pd.DataFrame()

    def sim_fn(combo_config: dict) -> dict:
        return _run_simulation_loop(
            executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
            ohlcv_by_ticker=ohlcv,
        )

    grid = config.get("param_sweep", param_sweep.DEFAULT_GRID)
    sweep_settings = config.get("param_sweep_settings", {})

    logger.info("Running param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
    return param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)


def run_predictor_backtest(config: dict) -> dict:
    """
    Run predictor-only historical backtest: generate synthetic signals from
    GBM inference on 2y of slim cache data, then simulate through the full
    executor pipeline (risk guard, position sizing, ATR stops, time decay,
    graduated drawdown).

    Returns a stats dict with portfolio metrics + metadata, or a status dict
    if insufficient data.
    """
    import sys
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    # Prepare data: load cache, compute features, run GBM, generate signals
    result = run_predictor_pipeline(config)

    if result.get("status") != "ok":
        return result

    signals_by_date = result["signals_by_date"]
    price_matrix = result["price_matrix"]
    ohlcv_by_ticker = result["ohlcv_by_ticker"]
    spy_prices = result.get("spy_prices")
    metadata = result["metadata"]

    # Import executor modules
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        return {"status": "error", "error": f"executor_paths not found: {executor_paths}"}
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient

    # Run simulation
    logger.info("Running predictor-only simulation: %d dates", len(signals_by_date))
    stats = _run_simulation_loop(
        executor_run, SimulatedIBKRClient,
        dates=[],  # not used when signals_by_date is provided
        price_matrix=price_matrix,
        config=config,
        ohlcv_by_ticker=ohlcv_by_ticker,
        signals_by_date=signals_by_date,
        spy_prices=spy_prices,
    )

    # Merge metadata into stats for reporting
    stats["predictor_metadata"] = metadata
    return stats


def run_predictor_param_sweep(config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Run predictor-only backtest with param sweep.

    Loads data once (features, GBM inference, signal generation), then runs
    the simulation loop for each parameter combination. Also runs Phase 4
    evaluations (ensemble mode, feature pruning) if features are available.

    Returns (single_run_stats, sweep_df).
    """
    import sys
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    # Prepare data once — keep features for Phase 4 evaluations
    result = run_predictor_pipeline(config, keep_features=True)

    if result.get("status") != "ok":
        return result, pd.DataFrame()

    signals_by_date = result["signals_by_date"]
    price_matrix = result["price_matrix"]
    ohlcv_by_ticker = result["ohlcv_by_ticker"]
    spy_prices = result.get("spy_prices")
    metadata = result["metadata"]
    features_by_ticker = result.get("features_by_ticker")
    sector_map = result.get("sector_map", {})
    trading_dates = result.get("trading_dates", [])

    # Import executor modules
    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next((p for p in executor_paths if os.path.isdir(p)), None)
    if not executor_path:
        return {"status": "error", "error": f"executor_paths not found: {executor_paths}"}, pd.DataFrame()
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient

    # Single run with default config
    logger.info("Running predictor-only simulation (default params): %d dates", len(signals_by_date))
    single_stats = _run_simulation_loop(
        executor_run, SimulatedIBKRClient,
        dates=[],
        price_matrix=price_matrix,
        config=config,
        ohlcv_by_ticker=ohlcv_by_ticker,
        signals_by_date=signals_by_date,
        spy_prices=spy_prices,
    )
    single_stats["predictor_metadata"] = metadata

    # ── Phase 4: Predictor hyperparameter feedback ───────────────────────
    predictions_by_date = result.get("predictions_by_date", {})
    if features_by_ticker and trading_dates:
        try:
            from optimizer.predictor_optimizer import (
                evaluate_ensemble_modes,
                evaluate_signal_thresholds,
                evaluate_feature_pruning,
                apply_recommendations,
            )
            bucket = config.get("signals_bucket", "alpha-engine-research")

            # Phase 4a: Ensemble mode evaluation
            ensemble_result = None
            try:
                ensemble_result = evaluate_ensemble_modes(
                    features_by_ticker, price_matrix, ohlcv_by_ticker,
                    spy_prices, sector_map, trading_dates,
                    config, single_stats,
                )
                single_stats["ensemble_eval"] = ensemble_result
            except Exception as exc:
                logger.warning("Ensemble mode evaluation failed (non-fatal): %s", exc)

            # Phase 4b: Signal threshold sweep
            threshold_result = None
            if predictions_by_date:
                try:
                    threshold_result = evaluate_signal_thresholds(
                        predictions_by_date, sector_map, ohlcv_by_ticker,
                        price_matrix, spy_prices, trading_dates,
                        config, single_stats,
                    )
                    single_stats["threshold_eval"] = threshold_result
                except Exception as exc:
                    logger.warning("Signal threshold evaluation failed (non-fatal): %s", exc)

            # Phase 4c: Feature pruning evaluation
            pruning_result = None
            try:
                pruning_result = evaluate_feature_pruning(
                    features_by_ticker, price_matrix, ohlcv_by_ticker,
                    spy_prices, sector_map, trading_dates,
                    config, single_stats,
                )
                single_stats["pruning_eval"] = pruning_result
            except Exception as exc:
                logger.warning("Feature pruning evaluation failed (non-fatal): %s", exc)

            # Apply recommendations to S3 (if any)
            try:
                apply_result = apply_recommendations(
                    ensemble_result, pruning_result, bucket,
                    threshold_result=threshold_result,
                )
                single_stats["predictor_optimizer_apply"] = apply_result
            except Exception as exc:
                logger.warning("Predictor optimizer apply failed (non-fatal): %s", exc)

        except ImportError as exc:
            logger.warning("Phase 4 optimizer not available: %s", exc)

        # Free features now that Phase 4 is done
        del features_by_ticker
        import gc
        gc.collect()

    # Param sweep — seed grid with current S3 params for iterative learning
    sweep_df = pd.DataFrame()
    grid = config.get("param_sweep")
    if grid:
        bucket = config.get("signals_bucket", "alpha-engine-research")
        grid = _seed_grid_with_current(grid, executor_optimizer.read_current_params(bucket))
        def sim_fn(combo_config: dict) -> dict:
            return _run_simulation_loop(
                executor_run, SimulatedIBKRClient,
                dates=[],
                price_matrix=price_matrix,
                config=combo_config,
                ohlcv_by_ticker=ohlcv_by_ticker,
                signals_by_date=signals_by_date,
                spy_prices=spy_prices,
            )

        sweep_settings = config.get("param_sweep_settings", {})

        logger.info("Running predictor param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
        sweep_df = param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)

    return single_stats, sweep_df


def _stop_ec2_instance() -> None:
    """Stop the current EC2 instance via metadata endpoint. Best-effort."""
    import urllib.request
    try:
        token = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                method="PUT",
            ),
            timeout=5,
        ).read().decode()
        instance_id = urllib.request.urlopen(
            urllib.request.Request(
                "http://169.254.169.254/latest/meta-data/instance-id",
                headers={"X-aws-ec2-metadata-token": token},
            ),
            timeout=5,
        ).read().decode()
        logger.info("Stopping instance %s", instance_id)
        boto3.client("ec2").stop_instances(InstanceIds=[instance_id])
    except Exception as e:
        logger.error("Failed to stop instance: %s", e)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Alpha Engine Backtester")
    parser.add_argument("--mode", choices=["signal-quality", "simulate", "param-sweep", "all", "predictor-backtest"],
                        default="signal-quality")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="Override research_db path from config")
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date label")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--stop-instance", action="store_true",
                        help="Stop this EC2 instance after completion (for scheduled runs)")
    parser.add_argument("--rollback", action="store_true",
                        help="Rollback all S3 configs to previous versions and exit")
    parser.add_argument("--freeze", action="store_true",
                        help="Skip all S3 config promotions (guardrails compute + report but never write)")
    return parser.parse_args()


def _init_pipeline(args: argparse.Namespace, config: dict) -> None:
    """Initialize optimizer modules and pull research DB if needed.

    Mutates config in place (adds research_db, _db_pull_status).
    """
    weight_optimizer.init_config(config)
    executor_optimizer.init_config(config)
    veto_analysis.init_config(config)
    research_optimizer.init_config(config)

    if args.db:
        config["research_db"] = args.db
        logger.info("Using local research.db: %s", args.db)
    elif args.mode in ("signal-quality", "all"):
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()
        bucket = config.get("signals_bucket", "alpha-engine-research")
        db_pulled = pull_research_db(bucket, tmp_db.name)
        if db_pulled:
            config["research_db"] = tmp_db.name
        else:
            config["research_db"] = None
        config["_db_pull_status"] = "ok" if db_pulled else "failed"


def _run_signal_quality_pipeline(
    args: argparse.Namespace, config: dict, fd=None,
) -> tuple[dict, list, list, dict, object, dict | None, dict | None]:
    """Run signal quality analysis, weight optimizer, veto analysis, and research params.

    Returns (sq_result, regime_rows, score_rows, attr_result, df_base, weight_result, veto_result).
    """
    sq_result, regime_rows, score_rows, attr_result, df_base = run_signal_quality(config)
    weight_result = run_weight_optimizer(config, df_base, freeze=args.freeze)

    veto_result = None
    if df_base is not None and not df_base.empty:
        try:
            bucket = config.get("signals_bucket", "alpha-engine-research")
            veto_result = veto_analysis.analyze_veto_effectiveness(df_base, bucket)
            if veto_result.get("status") == "ok":
                if args.freeze:
                    veto_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                else:
                    veto_result["apply_result"] = veto_analysis.apply(veto_result, bucket)
        except Exception as e:
            logger.error("Veto analysis failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "veto_analysis", "mode": args.mode})
            veto_result = {"status": "error", "error": str(e)}

    # Research params optimization (boost correlations)
    if df_base is not None and not df_base.empty:
        try:
            bucket = config.get("signals_bucket", "alpha-engine-research")
            current_rp = research_optimizer.read_current_params(bucket)
            corr_result = research_optimizer.compute_boost_correlations(df_base, bucket)
            if corr_result.get("status") == "ok":
                rp_result = research_optimizer.recommend(corr_result, current_rp)
                if rp_result.get("status") == "ok":
                    if args.freeze:
                        rp_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                    else:
                        rp_result["apply_result"] = research_optimizer.apply(rp_result, bucket)
        except Exception as e:
            logger.error("Research params optimization failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "research_optimizer", "mode": args.mode})

    return sq_result, regime_rows, score_rows, attr_result, df_base, weight_result, veto_result


def _run_simulation_pipeline(
    args: argparse.Namespace,
    config: dict,
    _sim_setup: tuple | None,
    current_executor_params: dict | None,
    fd=None,
) -> tuple[dict | None, object | None, dict | None]:
    """Run simulate mode, param sweep, executor optimizer, and twin simulation.

    Returns (portfolio_stats, sweep_df, executor_rec).
    """
    portfolio_stats = None
    sweep_df = None
    executor_rec = None

    # ── Simulate mode ─────────────────────────────────────────────────────
    if args.mode in ("simulate", "all"):
        try:
            if _sim_setup is None:
                portfolio_stats = {"status": "error", "error": "Simulation setup failed"}
            else:
                executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv = _sim_setup
                if price_matrix is None:
                    min_dates = config.get("min_simulation_dates", 5)
                    portfolio_stats = {
                        "status": "insufficient_data",
                        "dates_available": len(dates),
                        "min_required": min_dates,
                    }
                else:
                    portfolio_stats = _run_simulation_loop(
                        executor_run, SimulatedIBKRClient, dates, price_matrix, config,
                        ohlcv_by_ticker=ohlcv,
                    )
        except Exception as e:
            logger.error("Mode 2 simulation failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "simulation", "mode": args.mode})
            portfolio_stats = {"status": "error", "error": str(e)}

    # ── Param sweep ───────────────────────────────────────────────────────
    if args.mode in ("param-sweep", "all"):
        try:
            if _sim_setup is None:
                sweep_df = None
            else:
                executor_run, SimulatedIBKRClient, dates, price_matrix, _, ohlcv = _sim_setup
                if price_matrix is None:
                    logger.warning("Param sweep skipped: only %d signal dates available", len(dates))
                    sweep_df = pd.DataFrame()
                else:
                    def sim_fn(combo_config: dict) -> dict:
                        return _run_simulation_loop(
                            executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
                            ohlcv_by_ticker=ohlcv,
                        )
                    grid = config.get("param_sweep", param_sweep.DEFAULT_GRID)
                    grid = _seed_grid_with_current(grid, current_executor_params)
                    sweep_settings = config.get("param_sweep_settings", {})
                    logger.info("Running param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
                    sweep_df = param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)
        except Exception as e:
            logger.error("Param sweep failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "param_sweep", "mode": args.mode})
            sweep_df = None

        # Executor parameter optimization from sweep results
        if sweep_df is not None and not sweep_df.empty:
            try:
                executor_rec = executor_optimizer.recommend(
                    sweep_df, config, current_params=current_executor_params,
                )
                if executor_rec.get("status") == "ok" and _sim_setup is not None:
                    executor_run_fn, SimClientCls, sim_dates, pm, _, ohlcv_data = _sim_setup
                    if pm is not None:
                        def holdout_sim_fn(combo_config):
                            return _run_simulation_loop(
                                executor_run_fn, SimClientCls, sim_dates, pm, combo_config,
                                ohlcv_by_ticker=ohlcv_data,
                            )
                        executor_rec = executor_optimizer.validate_holdout(
                            executor_rec, holdout_sim_fn, sim_dates, config,
                        )

                # Twin simulation: current vs proposed on same dates
                if executor_rec.get("status") == "ok" and _sim_setup is not None:
                    executor_run_fn, SimClientCls, sim_dates, pm, _, ohlcv_data = _sim_setup
                    if pm is not None and current_executor_params:
                        from optimizer.twin_sim import run_twin_simulation
                        from copy import deepcopy
                        recommended = executor_rec.get("recommended_params", {})
                        current_cfg = deepcopy(config)
                        current_cfg.update(current_executor_params)
                        proposed_cfg = deepcopy(config)
                        proposed_cfg.update(recommended)
                        changed_keys = [k for k in recommended if recommended.get(k) != current_executor_params.get(k)]

                        def twin_sim_fn(cfg):
                            return _run_simulation_loop(
                                executor_run_fn, SimClientCls, sim_dates, pm, cfg,
                                ohlcv_by_ticker=ohlcv_data,
                            )
                        executor_rec["twin_sim"] = run_twin_simulation(
                            twin_sim_fn, current_cfg, proposed_cfg, changed_keys,
                        )

                if executor_rec.get("status") == "ok":
                    bucket = config.get("signals_bucket", "alpha-engine-research")
                    if args.freeze:
                        executor_rec["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                    else:
                        executor_rec["apply_result"] = executor_optimizer.apply(executor_rec, bucket)
            except Exception as e:
                logger.error("Executor optimizer failed: %s", e)
                if fd:
                    fd.report(e, severity="error", context={
                        "site": "executor_optimizer", "mode": args.mode})
                executor_rec = {"status": "error", "error": str(e)}

    return portfolio_stats, sweep_df, executor_rec


def _run_predictor_pipeline(
    args: argparse.Namespace,
    config: dict,
    executor_rec: dict | None,
    current_executor_params: dict | None,
    fd=None,
) -> tuple[dict | None, object | None, dict | None]:
    """Run predictor backtest and auto-apply executor params from predictor sweep.

    Returns (predictor_stats, predictor_sweep_df, executor_rec).
    """
    predictor_stats = None
    predictor_sweep_df = None

    try:
        predictor_stats, predictor_sweep_df = run_predictor_param_sweep(config)
    except Exception as e:
        logger.error("Predictor backtest failed: %s", e)
        if fd:
            fd.report(e, severity="error", context={
                "site": "predictor_backtest", "mode": args.mode})
        predictor_stats = {"status": "error", "error": str(e)}
        predictor_sweep_df = None

    # Auto-apply executor params from predictor sweep (if signal-based sweep
    # didn't already produce a recommendation)
    if (
        (executor_rec is None or executor_rec.get("status") not in ("ok",))
        and predictor_sweep_df is not None
        and not predictor_sweep_df.empty
    ):
        try:
            executor_rec = executor_optimizer.recommend(
                predictor_sweep_df, config, current_params=current_executor_params,
            )
            if executor_rec.get("status") == "ok":
                bucket = config.get("signals_bucket", "alpha-engine-research")
                if args.freeze:
                    executor_rec["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                else:
                    executor_rec["apply_result"] = executor_optimizer.apply(executor_rec, bucket)
        except Exception as e:
            logger.error("Executor optimizer (predictor sweep) failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "executor_optimizer_predictor", "mode": args.mode})
            executor_rec = {"status": "error", "error": str(e)}

    return predictor_stats, predictor_sweep_df, executor_rec


def _export_simulation_artifacts(
    config: dict,
    run_date: str,
    sweep_df=None,
    predictor_sweep_df=None,
    portfolio_stats: dict | None = None,
    predictor_stats: dict | None = None,
) -> None:
    """Write simulation artifacts to S3 for downstream evaluator consumption.

    The evaluator reads these artifacts to run executor optimization and
    include simulation results in its report. If the backtester fails,
    the evaluator runs without them (degraded mode).
    """
    import io
    bucket = config.get("output_bucket", config.get("signals_bucket", "alpha-engine-research"))
    prefix = f"backtest/{run_date}"
    s3 = boto3.client("s3")
    exported = []

    if sweep_df is not None and not sweep_df.empty:
        buf = io.BytesIO()
        sweep_df.to_parquet(buf, index=False)
        s3.put_object(Bucket=bucket, Key=f"{prefix}/sweep_df.parquet", Body=buf.getvalue())
        exported.append("sweep_df.parquet")

    if predictor_sweep_df is not None and not predictor_sweep_df.empty:
        buf = io.BytesIO()
        predictor_sweep_df.to_parquet(buf, index=False)
        s3.put_object(Bucket=bucket, Key=f"{prefix}/predictor_sweep_df.parquet", Body=buf.getvalue())
        exported.append("predictor_sweep_df.parquet")

    if portfolio_stats:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/portfolio_stats.json", Body=json.dumps(portfolio_stats, indent=2, default=str).encode())
        exported.append("portfolio_stats.json")

    if predictor_stats:
        s3.put_object(Bucket=bucket, Key=f"{prefix}/predictor_stats.json", Body=json.dumps(predictor_stats, indent=2, default=str).encode())
        exported.append("predictor_stats.json")

    if exported:
        logger.info("Exported simulation artifacts to s3://%s/%s/: %s", bucket, prefix, ", ".join(exported))


def _run_regression_detection(
    args: argparse.Namespace,
    config: dict,
    portfolio_stats: dict | None,
    sq_result: dict,
    weight_result: dict | None,
    executor_rec: dict | None,
    veto_result: dict | None,
) -> dict | None:
    """Run regression detection and save rolling metrics."""
    try:
        from optimizer.regression_monitor import (
            extract_metrics, save_rolling_metrics, save_promotion_baseline,
            check_regression,
        )
        bucket = config.get("signals_bucket", "alpha-engine-research")
        current_metrics = extract_metrics(portfolio_stats, sq_result)

        if current_metrics:
            save_rolling_metrics(bucket, args.date, current_metrics)

        promoted = []
        for label, res in [
            ("scoring_weights", weight_result),
            ("executor_params", executor_rec),
            ("predictor_params", veto_result),
        ]:
            if res and res.get("apply_result", {}).get("applied"):
                promoted.append(label)

        if promoted and current_metrics:
            save_promotion_baseline(bucket, current_metrics, promoted)

        if current_metrics and not args.freeze:
            return check_regression(bucket, current_metrics, config)
        return None
    except Exception as e:
        logger.error("Regression monitor failed: %s", e)
        return None


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    _health_start = _time.time()

    # Flow Doctor: structured error capture (optional dependency)
    fd = None
    try:
        import flow_doctor
        fd = flow_doctor.init(config_path=os.path.join(
            os.path.dirname(__file__), "flow-doctor.yaml"))
    except ImportError:
        pass
    except Exception as e:
        logger.warning("flow-doctor init failed: %s", e)

    config = load_config(args.config)

    # Handle --rollback before any other mode
    if args.rollback:
        from optimizer.rollback import rollback_all
        bucket = config.get("signals_bucket", "alpha-engine-research")
        results = rollback_all(bucket)
        for r in results:
            if r.get("rolled_back"):
                print(f"  Rolled back: {r['config_type']} → {r['key']}")
            else:
                print(f"  Skipped: {r.get('reason', 'unknown')}")
        return

    _init_pipeline(args, config)

    # ── Default results (overwritten by each pipeline stage) ──────────────
    sq_result: dict = {"status": "skipped"}
    regime_rows: list = []
    score_rows: list = []
    attr_result: dict = {"status": "skipped"}
    df_base = None
    weight_result = None
    veto_result = None
    portfolio_stats = None
    sweep_df = None
    executor_rec = None
    predictor_stats = None
    predictor_sweep_df = None
    e2e_lift = None
    trigger_result = None
    alpha_dist_result = None
    score_cal_result = None
    veto_val_result = None
    shadow_result = None
    exit_timing_result = None
    macro_result = None
    trigger_opt_result = None
    predictor_sizing_result = None
    scanner_opt_result = None
    team_opt_result = None
    cio_opt_result = None
    sizing_ab_result = None
    confusion_result = None

    # ── Signal quality pipeline ───────────────────────────────────────────
    if args.mode in ("signal-quality", "all"):
        (sq_result, regime_rows, score_rows, attr_result, df_base,
         weight_result, veto_result) = _run_signal_quality_pipeline(args, config, fd)

        # End-to-end pipeline lift metrics (requires universe_returns)
        db_path = config.get("research_db")
        trades_db = _find_trades_db(config)
        if db_path and os.path.exists(db_path):
            try:
                e2e_lift = end_to_end.compute_lift_metrics(
                    research_db_path=db_path,
                    trades_db_path=trades_db,
                )
                if e2e_lift.get("status") == "ok":
                    logger.info("End-to-end lift metrics computed across %d dates", e2e_lift.get("n_dates", 0))
            except Exception as e:
                logger.warning("End-to-end lift metrics failed: %s", e)

        # ── Phase 3: Component-level diagnostics ──────────────────────────
        # 3a: Entry trigger scorecard
        if trades_db:
            try:
                trigger_result = trigger_scorecard.compute_trigger_scorecard(trades_db)
                if trigger_result.get("status") == "ok":
                    logger.info("Trigger scorecard: %d triggers analyzed", len(trigger_result.get("triggers", [])))
            except Exception as e:
                logger.warning("Trigger scorecard failed: %s", e)

        # 3g: Alpha magnitude distribution
        if db_path and os.path.exists(db_path):
            try:
                alpha_dist_result = alpha_distribution.compute_alpha_distribution(db_path)
                if alpha_dist_result.get("status") == "ok":
                    logger.info("Alpha distribution computed for %d horizons", len(alpha_dist_result.get("distributions", {})))
            except Exception as e:
                logger.warning("Alpha distribution failed: %s", e)

            # Score calibration curve
            try:
                score_cal_result = alpha_distribution.compute_score_calibration(db_path)
            except Exception as e:
                logger.warning("Score calibration failed: %s", e)

        # 3e: Net veto value in dollars
        if db_path and os.path.exists(db_path):
            try:
                veto_val_result = veto_value.compute_veto_value(
                    research_db_path=db_path,
                    trades_db_path=trades_db,
                )
                if veto_val_result.get("status") == "ok":
                    logger.info("Net veto value: $%.0f", veto_val_result.get("net_veto_value", 0))
            except Exception as e:
                logger.warning("Veto value analysis failed: %s", e)

        # 3h: Predictor confusion matrix
        if db_path and os.path.exists(db_path):
            try:
                from analysis.predictor_confusion import compute_confusion_matrix
                confusion_result = compute_confusion_matrix(db_path)
                if confusion_result.get("status") == "ok":
                    logger.info("Predictor confusion matrix: %d predictions, accuracy=%.1f%%",
                                confusion_result.get("n", 0), (confusion_result.get("accuracy", 0) or 0) * 100)
            except Exception as e:
                logger.warning("Predictor confusion matrix failed: %s", e)

        # 3b: Shadow book analysis
        if trades_db:
            try:
                shadow_result = shadow_book_analysis.compute_shadow_book_analysis(
                    trades_db_path=trades_db,
                    research_db_path=db_path if db_path and os.path.exists(db_path) else None,
                )
                if shadow_result.get("status") == "ok":
                    logger.info("Shadow book: %d blocked, assessment=%s",
                                shadow_result.get("n_blocked", 0), shadow_result.get("assessment"))
            except Exception as e:
                logger.warning("Shadow book analysis failed: %s", e)

        # 3c: Exit timing analysis (MFE/MAE)
        if trades_db:
            try:
                exit_timing_result = exit_timing.compute_exit_timing(trades_db)
                if exit_timing_result.get("status") == "ok":
                    logger.info("Exit timing: %d roundtrips, diagnosis=%s",
                                exit_timing_result.get("n_roundtrips", 0), exit_timing_result.get("diagnosis"))
            except Exception as e:
                logger.warning("Exit timing analysis failed: %s", e)

        # 3f: Macro multiplier A/B evaluation
        if db_path and os.path.exists(db_path):
            try:
                macro_result = macro_eval.compute_macro_evaluation(db_path)
                if macro_result.get("status") == "ok":
                    logger.info("Macro eval: assessment=%s, lift=%s",
                                macro_result.get("assessment"), macro_result.get("accuracy_lift"))
            except Exception as e:
                logger.warning("Macro multiplier evaluation failed: %s", e)

        # ── Phase 4: Self-adjustment mechanisms ─────────────────────────────
        bucket = config.get("signals_bucket", "alpha-engine-research")

        # 4e: Trigger optimizer — disable underperforming triggers
        if trigger_result and trigger_result.get("status") == "ok":
            try:
                trigger_opt_result = trigger_optimizer.analyze(trigger_result)
                if trigger_opt_result.get("status") == "ok":
                    if args.freeze:
                        trigger_opt_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                    else:
                        trigger_opt_result["apply_result"] = trigger_optimizer.apply(trigger_opt_result, bucket)
                    logger.info("Trigger optimizer: %d triggers to disable", len(trigger_opt_result.get("disabled_triggers", [])))
            except Exception as e:
                logger.warning("Trigger optimizer failed: %s", e)

        # 4d: Predictor p_up → sizing (if IC positive)
        if db_path and os.path.exists(db_path):
            try:
                predictor_sizing_result = predictor_sizing_optimizer.analyze(db_path)
                if predictor_sizing_result.get("status") == "ok":
                    if args.freeze:
                        predictor_sizing_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                    elif predictor_sizing_result.get("recommendation") == "enable":
                        predictor_sizing_result["apply_result"] = predictor_sizing_optimizer.apply(predictor_sizing_result, bucket)
                    logger.info("Predictor sizing: IC=%.3f, recommendation=%s",
                                predictor_sizing_result.get("overall_rank_ic", 0),
                                predictor_sizing_result.get("recommendation"))
            except Exception as e:
                logger.warning("Predictor sizing optimizer failed: %s", e)

        # 4a: Scanner auto-relax (if leakage is high)
        if db_path and os.path.exists(db_path):
            try:
                scanner_analysis = scanner_optimizer.analyze(db_path)
                if scanner_analysis.get("status") == "ok":
                    current_scanner = scanner_optimizer.read_current_params(bucket)
                    scanner_opt_result = scanner_optimizer.recommend(scanner_analysis, current_scanner)
                    if scanner_opt_result.get("status") == "ok":
                        if args.freeze:
                            scanner_opt_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                        else:
                            scanner_opt_result["apply_result"] = scanner_optimizer.apply(scanner_opt_result, bucket)
                    scanner_opt_result["analysis"] = scanner_analysis
                    logger.info("Scanner optimizer: leakage=%.1f%%, high=%s",
                                scanner_analysis.get("leakage_rate", 0) * 100,
                                scanner_analysis.get("high_leakage"))
                else:
                    scanner_opt_result = scanner_analysis
            except Exception as e:
                logger.warning("Scanner optimizer failed: %s", e)

        # 4b: Team slot allocation + 4c: CIO fallback
        if e2e_lift and e2e_lift.get("status") == "ok":
            try:
                team_analysis = pipeline_optimizer.analyze_team_performance(e2e_lift)
                if team_analysis.get("status") == "ok":
                    team_opt_result = pipeline_optimizer.recommend_team_slots(team_analysis)
                    if team_opt_result.get("status") == "ok":
                        if args.freeze:
                            team_opt_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                        else:
                            team_opt_result["apply_result"] = pipeline_optimizer.apply_team_slots(team_opt_result, bucket)
                    team_opt_result["analysis"] = team_analysis
                    logger.info("Team optimizer: %s", team_opt_result.get("changes", "no changes"))
                else:
                    team_opt_result = team_analysis
            except Exception as e:
                logger.warning("Team slot optimizer failed: %s", e)

            try:
                cio_opt_result = pipeline_optimizer.analyze_cio_performance(e2e_lift)
                if cio_opt_result.get("status") == "ok":
                    if args.freeze:
                        cio_opt_result["apply_result"] = {"applied": False, "reason": "frozen (--freeze flag)"}
                    elif cio_opt_result.get("recommendation") == "deterministic":
                        cio_opt_result["apply_result"] = pipeline_optimizer.apply_cio_mode(cio_opt_result, bucket)
                    logger.info("CIO optimizer: recommendation=%s", cio_opt_result.get("recommendation"))
            except Exception as e:
                logger.warning("CIO optimizer failed: %s", e)

    # ── Simulation setup (shared by simulate + param-sweep) ───────────────
    _sim_setup = None
    if args.mode in ("simulate", "param-sweep", "all"):
        try:
            _sim_setup = _setup_simulation(config)
        except Exception as e:
            logger.error("Simulation setup failed: %s", e)
            if fd:
                fd.report(e, severity="error", context={
                    "site": "simulation_setup", "mode": args.mode})

    current_executor_params = None
    if args.mode in ("param-sweep", "all", "predictor-backtest"):
        bucket = config.get("signals_bucket", "alpha-engine-research")
        current_executor_params = executor_optimizer.read_current_params(bucket)

    # ── Simulate + param sweep + executor optimizer ───────────────────────
    if args.mode in ("simulate", "param-sweep", "all"):
        portfolio_stats, sweep_df, executor_rec = _run_simulation_pipeline(
            args, config, _sim_setup, current_executor_params, fd,
        )

    # ── 4f: Sizing A/B test ──────────────────────────────────────────────
    if args.mode in ("simulate", "all") and _sim_setup is not None:
        try:
            executor_run, SimulatedIBKRClient, dates, price_matrix, _, ohlcv = _sim_setup
            if price_matrix is not None:
                def _sizing_sim_fn(combo_config):
                    return _run_simulation_loop(
                        executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
                        ohlcv_by_ticker=ohlcv,
                    )
                sizing_ab_result = sizing_ab.run_sizing_ab(_sizing_sim_fn, config)
                if sizing_ab_result.get("status") == "ok":
                    logger.info("Sizing A/B: %s (Sharpe diff=%s)",
                                sizing_ab_result.get("assessment"),
                                sizing_ab_result.get("sharpe_diff"))
        except Exception as e:
            logger.warning("Sizing A/B test failed: %s", e)
            if fd:
                fd.report(e, severity="warning", context={
                    "site": "sizing_ab", "mode": args.mode})

    # ── Predictor backtest ────────────────────────────────────────────────
    if args.mode in ("predictor-backtest", "all"):
        predictor_stats, predictor_sweep_df, executor_rec = _run_predictor_pipeline(
            args, config, executor_rec, current_executor_params, fd,
        )

    # ── Export simulation artifacts for evaluator ────────────────────────
    if args.mode in ("simulate", "param-sweep", "all", "predictor-backtest"):
        try:
            _export_simulation_artifacts(config, args.date, sweep_df=sweep_df, predictor_sweep_df=predictor_sweep_df, portfolio_stats=portfolio_stats, predictor_stats=predictor_stats)
        except Exception as e:
            logger.warning("Simulation artifact export failed (non-fatal): %s", e)

    # ── Regression detection ──────────────────────────────────────────────
    regression_result = _run_regression_detection(
        args, config, portfolio_stats, sq_result,
        weight_result, executor_rec, veto_result,
    )

    # ── Report, upload, email, and instance stop ──────────────────────────
    # Wrapped in try/finally so --stop-instance ALWAYS runs.
    try:
        pipeline_health = {
            "db_pull_status": config.get("_db_pull_status"),
            "staleness_warning": portfolio_stats.get("staleness_warning") if portfolio_stats else None,
            "coverage": portfolio_stats.get("coverage") if portfolio_stats else None,
            "dates_simulated": portfolio_stats.get("dates_simulated") if portfolio_stats else None,
            "dates_expected": portfolio_stats.get("dates_expected") if portfolio_stats else None,
            "skip_reasons": portfolio_stats.get("skip_reasons") if portfolio_stats else None,
            "price_gap_warnings": portfolio_stats.get("price_gap_warnings") if portfolio_stats else None,
            "unfilled_gaps": portfolio_stats.get("unfilled_gaps") if portfolio_stats else None,
            "feature_skip_reasons": predictor_stats.get("skip_reasons") if predictor_stats else None,
        }

        # Compute unified scorecard grades
        from analysis.grading import compute_scorecard
        grading_result = compute_scorecard(
            signal_quality=sq_result,
            e2e_lift=e2e_lift,
            macro_eval=macro_result,
            score_calibration=score_cal_result,
            veto_result=veto_result,
            veto_value=veto_val_result,
            trigger_scorecard=trigger_result,
            shadow_book=shadow_result,
            exit_timing=exit_timing_result,
            sizing_ab=sizing_ab_result,
            predictor_sizing=predictor_sizing_result,
            portfolio_stats=portfolio_stats,
            scanner_opt=scanner_opt_result,
            cio_opt=cio_opt_result,
        )

        report_md = build_report(
            run_date=args.date,
            signal_quality=sq_result,
            regime_analysis=regime_rows,
            score_analysis=score_rows,
            attribution=attr_result,
            portfolio_stats=portfolio_stats,
            sweep_df=sweep_df,
            weight_result=weight_result,
            config=config,
            predictor_stats=predictor_stats,
            predictor_sweep_df=predictor_sweep_df,
            veto_result=veto_result,
            executor_rec=executor_rec,
            regression_result=regression_result,
            pipeline_health=pipeline_health,
            e2e_lift=e2e_lift,
            trigger_scorecard=trigger_result,
            alpha_dist=alpha_dist_result,
            score_calibration=score_cal_result,
            veto_value=veto_val_result,
            shadow_book=shadow_result,
            exit_timing=exit_timing_result,
            macro_eval=macro_result,
            trigger_opt=trigger_opt_result,
            predictor_sizing=predictor_sizing_result,
            scanner_opt=scanner_opt_result,
            team_opt=team_opt_result,
            cio_opt=cio_opt_result,
            sizing_ab=sizing_ab_result,
            grading=grading_result,
            confusion_matrix=confusion_result,
        )

        save_sweep_df = sweep_df
        if predictor_sweep_df is not None and not predictor_sweep_df.empty:
            save_sweep_df = predictor_sweep_df

        out_dir = save(
            report_md=report_md,
            signal_quality=sq_result,
            score_analysis=score_rows,
            sweep_df=save_sweep_df,
            attribution=attr_result if args.mode in ("signal-quality", "all") else None,
            run_date=args.date,
            results_dir=config.get("results_dir", "results"),
            grading=grading_result,
            trigger_scorecard=trigger_result,
            shadow_book=shadow_result,
            exit_timing=exit_timing_result,
            e2e_lift=e2e_lift,
            veto_result=veto_result,
            confusion_matrix=confusion_result,
        )

        print(f"\nReport saved to {out_dir}/")
        print(f"\n{'='*60}")
        print(report_md[:2000])
        if len(report_md) > 2000:
            print(f"\n... (truncated — see {out_dir}/report.md for full report)")

        if args.upload:
            upload_to_s3(
                local_dir=out_dir,
                bucket=config.get("output_bucket", "alpha-engine-research"),
                prefix=config.get("output_prefix", "backtest"),
                run_date=args.date,
            )
            print(f"\nUploaded to s3://{config.get('output_bucket')}/{config.get('output_prefix')}/{args.date}/")

            # Append grades to S3 history for trend tracking
            if grading_result and grading_result.get("status") in ("ok", "partial"):
                try:
                    from analysis.grade_history import append_grades
                    gh_result = append_grades(grading_result, args.date, config.get("output_bucket", "alpha-engine-research"))
                    if gh_result.get("status") == "ok":
                        logger.info("Grade history updated: %d entries", gh_result.get("n_entries", 0))
                except Exception as e:
                    logger.warning("Grade history update failed (non-fatal): %s", e)

        sender = config.get("email_sender")
        recipients = config.get("email_recipients", [])
        if sender and recipients:
            send_report_email(
                run_date=args.date,
                report_md=report_md,
                status=sq_result.get("status", "unknown"),
                sender=sender,
                recipients=recipients,
                s3_bucket=config.get("output_bucket") if args.upload else None,
                s3_prefix=config.get("output_prefix", "backtest"),
            )
        else:
            logger.warning("No email_sender/email_recipients in config — skipping email")
    except Exception as e:
        logger.error("Report/upload/email failed: %s", e)
        if fd:
            fd.report(e, severity="critical", context={
                "site": "report_upload_email",
                "mode": args.mode,
                "run_date": args.date,
                "upload": args.upload,
            })
    finally:
        try:
            from health_status import write_health
            configs_applied = []
            if weight_result and weight_result.get("apply_result", {}).get("applied"):
                configs_applied.append("scoring_weights")
            if executor_rec and executor_rec.get("apply_result", {}).get("applied"):
                configs_applied.append("executor_params")
            if veto_result and veto_result.get("apply_result", {}).get("applied"):
                configs_applied.append("predictor_params")
            bucket = config.get("signals_bucket", "alpha-engine-research")
            write_health(
                bucket=bucket,
                module_name="backtester",
                status="ok",
                run_date=args.date,
                duration_seconds=_time.time() - _health_start,
                summary={
                    "mode": args.mode,
                    "configs_applied": configs_applied,
                },
            )
        except Exception as _he:
            logger.warning("Health status write failed: %s", _he)

        if args.stop_instance:
            _stop_ec2_instance()


if __name__ == "__main__":
    main()
