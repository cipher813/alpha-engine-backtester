"""
backtest.py — CLI entry point for alpha-engine-backtester.

Usage:
    # Mode 1 — Signal quality report (works now; results meaningful at Week 4+)
    python backtest.py --mode signal-quality

    # Mode 2 — Portfolio simulation (deferred until Week 4+)
    python backtest.py --mode simulate

    # Full report (both modes)
    python backtest.py --mode all

    # Upload results to S3
    python backtest.py --mode signal-quality --upload

Options:
    --mode          signal-quality | simulate | all  (default: signal-quality)
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
import logging
import tempfile
import os
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import yaml

from analysis import signal_quality, regime_analysis, score_analysis, attribution
from reporter import build_report, save, upload_to_s3

logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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
        return sq_result, [], [], {"status": "insufficient_data", "note": "research_db not configured"}

    logger.info("Loading score_performance from %s", db_path)

    try:
        df_base = signal_quality.load_score_performance(db_path)
        sq_result = signal_quality.compute_accuracy(df_base, min_samples=min_samples)
    except FileNotFoundError as e:
        logger.error("research.db not found: %s", e)
        sq_result = {"status": "db_not_found", "error": str(e)}
        return sq_result, [], [], {"status": "insufficient_data", "note": "research.db not found"}

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

    return sq_result, regime_rows, score_rows, attr_result


def run_simulate(config: dict):
    """
    Run Mode 2: executor portfolio simulation via vectorbt.

    DEFERRED — requires:
      1. Phase 0a: SimulatedIBKRClient in alpha-engine/executor/ibkr.py
      2. Phase 0b: simulate= mode in alpha-engine/executor/main.py
      3. Phase 0c: prices.json written to S3 by alpha-engine-research pipeline
      4. 20+ trading days of signal history (available Week 4+)

    Raises NotImplementedError with a clear message until prerequisites are met.
    """
    raise NotImplementedError(
        "Mode 2 (portfolio simulation) is data-gated until Week 4+.\n"
        "Requires 20+ trading days of signals.json + prices.json in S3.\n"
        "Phase 0 (SimulatedIBKRClient, simulate= mode, price snapshots) is complete.\n"
        "Wire up run_simulate() once sufficient signal history is available."
    )


def main():
    parser = argparse.ArgumentParser(description="Alpha Engine Backtester")
    parser.add_argument("--mode", choices=["signal-quality", "simulate", "all"],
                        default="signal-quality")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--db", help="Override research_db path from config")
    parser.add_argument("--upload", action="store_true", help="Upload results to S3")
    parser.add_argument("--date", default=date.today().isoformat(), help="Run date label")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    config = load_config(args.config)

    # Pull research.db from S3 unless a local path was explicitly provided.
    # research.db lives at s3://{signals_bucket}/research.db — Lambda writes it
    # after each pipeline run. We pull read-only to a temp file.
    if args.db:
        config["research_db"] = args.db
        logger.info("Using local research.db: %s", args.db)
    elif args.mode in ("signal-quality", "all"):
        tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp_db.close()
        bucket = config.get("signals_bucket", "alpha-engine-research")
        if pull_research_db(bucket, tmp_db.name):
            config["research_db"] = tmp_db.name
        else:
            config["research_db"] = None  # run_signal_quality handles None gracefully

    portfolio_stats = None

    if args.mode in ("signal-quality", "all"):
        sq_result, regime_rows, score_rows, attr_result = run_signal_quality(config)
    else:
        sq_result = {"status": "skipped"}
        regime_rows = []
        score_rows = []
        attr_result = {"status": "skipped"}

    if args.mode in ("simulate", "all"):
        try:
            portfolio_stats = run_simulate(config)
        except NotImplementedError as e:
            logger.warning("%s", e)

    report_md = build_report(
        run_date=args.date,
        signal_quality=sq_result,
        regime_analysis=regime_rows,
        score_analysis=score_rows,
        attribution=attr_result,
        portfolio_stats=portfolio_stats,
        config=config,
    )

    out_dir = save(
        report_md=report_md,
        signal_quality=sq_result,
        score_analysis=score_rows,
        run_date=args.date,
        results_dir=config.get("results_dir", "results"),
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


if __name__ == "__main__":
    main()
