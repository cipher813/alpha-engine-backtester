"""
pipeline_common.py — Shared utilities for backtest.py and evaluate.py.

Functions extracted from backtest.py that both entry points need:
config loading, research DB management, data seeding/backfilling.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import date
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


# ── Schema migrations ────────────────────────────────────────────────────────


def ensure_5d_columns(conn) -> None:
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


# ── Data seeding ──────────────────────────────────────────────────────────────


def seed_score_performance(config: dict) -> None:
    """Seed score_performance rows from S3 signals/{date}/signals.json files."""
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
        existing = {
            (r[0], r[1]) for r in
            conn.execute("SELECT symbol, score_date FROM score_performance").fetchall()
        }

        signal_dates = signal_loader.list_dates(bucket)
        rows_to_insert = []
        for sig_date in signal_dates:
            try:
                signals = signal_loader.load(bucket, sig_date)
            except FileNotFoundError:
                continue

            for stock in signals.get("universe", []):
                ticker = stock.get("ticker")
                score = stock.get("score", 0)
                rating = stock.get("rating", "")
                if not ticker or rating != "BUY" or (ticker, sig_date) in existing:
                    continue
                rows_to_insert.append((ticker, sig_date, score))

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
            ts = pd.Timestamp(dt_str)
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
        logger.warning("seed_score_performance: %s", e)
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def seed_predictor_outcomes(config: dict) -> None:
    """Seed predictor_outcomes rows from S3 predictions/*.json files."""
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    bucket = config.get("signals_bucket")
    if not db_path or not os.path.exists(db_path) or not bucket:
        return
    try:
        s3 = boto3.client("s3")
        resp = s3.list_objects_v2(Bucket=bucket, Prefix="predictor/predictions/", Delimiter="/")
        keys = [obj["Key"] for obj in resp.get("Contents", [])
                if obj["Key"].endswith(".json") and "latest" not in obj["Key"]]

        if not keys:
            logger.info("No prediction files found in S3 — skipping seed")
            return

        import sqlite3 as _sqlite3
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
        logger.warning("seed_predictor_outcomes: %s", e)


# ── Return backfilling ────────────────────────────────────────────────────────


def backfill_score_performance_returns(config: dict) -> None:
    """Backfill 5d, 10d, and 30d returns for score_performance rows missing them."""
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return
    try:
        import yfinance as yf

        conn = _sqlite3.connect(db_path)
        ensure_5d_columns(conn)

        # Repair beat_spy columns where return exists but beat_spy is NULL
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

        rows_5d = []
        rows_10d = []
        rows_30d = []
        for _, row in pending.iterrows():
            score_ts = pd.Timestamp(row["score_date"])
            if score_ts + pd.offsets.BDay(5) <= today_ts:
                rows_5d.append(row)
            if score_ts + pd.offsets.BDay(10) <= today_ts:
                rows_10d.append(row)
            if score_ts + pd.offsets.BDay(30) <= today_ts:
                rows_30d.append(row)

        if not rows_5d and not rows_10d and not rows_30d:
            conn.close()
            logger.info("No score_performance rows eligible for return backfill yet")
            return

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

        for horizon_label, rows, bday_offset in [
            ("5d", rows_5d, 5),
            ("10d", rows_10d, 10),
            ("30d", rows_30d, 30),
        ]:
            for row in rows:
                score_ts = pd.Timestamp(row["score_date"])
                eval_ts = score_ts + pd.offsets.BDay(bday_offset)
                exit_price = _get_close(row["symbol"], eval_ts)
                spy_entry = _get_close("SPY", score_ts)
                spy_exit = _get_close("SPY", eval_ts)

                if exit_price is None or row["price_on_date"] is None:
                    continue

                ret = (exit_price / row["price_on_date"]) - 1
                spy_ret = (spy_exit / spy_entry) - 1 if spy_entry and spy_exit else None
                beat = (1 if ret > spy_ret else 0) if spy_ret is not None else None

                conn.execute(
                    f"UPDATE score_performance SET price_{horizon_label}=?, return_{horizon_label}=?, "
                    f"spy_{horizon_label}_return=?, beat_spy_{horizon_label}=?, eval_date_{horizon_label}=? "
                    f"WHERE symbol=? AND score_date=? AND return_{horizon_label} IS NULL",
                    (
                        round(exit_price, 2),
                        round(ret * 100, 2),
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
        logger.warning("backfill_score_performance_returns: %s", e)
        try:
            conn.close()
        except Exception:
            pass


def backfill_predictor_outcomes(config: dict, df_base: pd.DataFrame) -> None:
    """Backfill actual_5d_return and correct_5d for pending predictor_outcomes rows."""
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return
    try:
        import yfinance as yf
    except ImportError as ie:
        logger.warning("backfill_predictor_outcomes: missing dependency: %s", ie)
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
        logger.warning("backfill_predictor_outcomes: %s", e)


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

    try:
        s3 = boto3.client("s3")
        existing: dict = {}
        try:
            resp = s3.get_object(Bucket=bucket, Key=metrics_key)
            existing = json.loads(resp["Body"].read())
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            logger.warning("Failed to read existing predictor metrics from S3: %s", e)

        from datetime import datetime
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
        logger.warning("push_predictor_rolling_metrics: S3 write failed: %s", e)


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
