"""
backtest.py — CLI entry point for alpha-engine-backtester.

Usage:
    # Mode 1 — Signal quality report
    python backtest.py --mode signal-quality

    # Mode 2 — Portfolio simulation (requires executor_path in config.yaml)
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
from emailer import send_report_email
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


def run_simulate(config: dict) -> dict:
    """
    Run Mode 2: replay all historical signal dates through the executor with
    SimulatedIBKRClient, then compute portfolio metrics via vectorbt.

    Price data comes from S3 prices.json when available, falling back to
    yfinance for any dates where prices.json is missing — so this works
    with any number of signal dates in S3.

    Returns a stats dict with status, returns, Sharpe, drawdown, etc.
    Returns {"status": "insufficient_data", ...} if fewer than
    config["min_simulation_dates"] signal dates exist in S3.
    """
    import sys
    import pandas as pd

    executor_paths = config.get("executor_paths", [])
    if isinstance(executor_paths, str):
        executor_paths = [executor_paths]
    executor_path = next(
        (p for p in executor_paths if os.path.isdir(p)), None
    )
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
    from vectorbt_bridge import orders_to_portfolio
    from vectorbt_bridge import portfolio_stats as compute_portfolio_stats

    bucket = config.get("signals_bucket", "alpha-engine-research")
    min_dates = config.get("min_simulation_dates", 5)
    init_cash = float(config.get("init_cash", 1_000_000.0))

    dates = signal_loader.list_dates(bucket)
    logger.info("Mode 2: %d signal dates available in S3", len(dates))

    if len(dates) < min_dates:
        logger.warning(
            "Mode 2: only %d signal dates available (need %d) — skipping simulation",
            len(dates), min_dates,
        )
        return {
            "status": "insufficient_data",
            "dates_available": len(dates),
            "min_required": min_dates,
        }

    logger.info("Mode 2: building price matrix for %d dates (yfinance fallback)...", len(dates))
    price_matrix = price_loader.build_matrix(dates, bucket)

    if price_matrix.empty:
        return {"status": "no_prices", "dates_available": len(dates)}

    # One SimulatedIBKRClient for the entire run — positions and NAV carry forward
    # across dates so the portfolio accumulates naturally over time.
    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
    all_orders = []
    dates_simulated = 0

    for signal_date in dates:
        ts = pd.Timestamp(signal_date)
        if ts not in price_matrix.index:
            logger.debug("No price row for %s — skipping", signal_date)
            continue

        date_prices = price_matrix.loc[ts].dropna().to_dict()
        if not date_prices:
            logger.warning("Empty price row for %s — skipping", signal_date)
            continue

        try:
            signals_raw = signal_loader.load(bucket, signal_date)
        except FileNotFoundError:
            logger.warning("No signals.json for %s — skipping", signal_date)
            continue

        # Swap in the new day's prices; positions and NAV persist from prior day
        sim_client._prices = date_prices

        orders = executor_run(
            simulate=True,
            ibkr_client=sim_client,
            signals_override=signals_raw,
        )
        if orders:
            all_orders.extend(orders)
        dates_simulated += 1

    logger.info(
        "Mode 2: %d dates simulated, %d orders generated", dates_simulated, len(all_orders)
    )

    if not all_orders:
        return {
            "status": "no_orders",
            "dates_simulated": dates_simulated,
            "note": "No ENTER signals passed risk rules during the simulation period",
        }

    pf = orders_to_portfolio(all_orders, price_matrix, init_cash=init_cash)
    stats = compute_portfolio_stats(pf)
    stats["status"] = "ok"
    stats["dates_simulated"] = dates_simulated
    stats["total_orders"] = len(all_orders)
    return stats


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
        except Exception as e:
            logger.error("Mode 2 simulation failed: %s", e)
            portfolio_stats = {"status": "error", "error": str(e)}

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


if __name__ == "__main__":
    main()
