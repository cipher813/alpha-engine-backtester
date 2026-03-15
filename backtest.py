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
import logging
import tempfile
import os
from datetime import date
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import yaml

from analysis import signal_quality, regime_analysis, score_analysis, attribution, param_sweep
from analysis import veto_analysis
from optimizer import weight_optimizer, executor_optimizer
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
    except Exception as e:
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
    if len(valid) >= 10:
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
        ic_ir_30d = round(float(chunk_ics.mean() / (chunk_ics.std() + 1e-8)), 3)

    try:
        s3 = boto3.client("s3")
        # Read existing metrics, merge rolling stats on top
        existing: dict = {}
        try:
            resp = s3.get_object(Bucket=bucket, Key=metrics_key)
            existing = json.loads(resp["Body"].read())
        except Exception:
            pass  # fresh file or not yet written

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


def _backfill_predictor_outcomes(config: dict, df_base: pd.DataFrame) -> None:
    """Backfill actual_5d_return and correct_5d for pending predictor_outcomes rows."""
    import sqlite3 as _sqlite3
    db_path = config.get("research_db")
    if not db_path or not os.path.exists(db_path):
        return
    if "symbol" not in df_base.columns or "score_date" not in df_base.columns:
        return
    try:
        conn = _sqlite3.connect(db_path)
        pending = pd.read_sql_query(
            "SELECT * FROM predictor_outcomes WHERE actual_5d_return IS NULL",
            conn,
        )
        if pending.empty:
            conn.close()
            return
        for _, row in pending.iterrows():
            match = df_base[
                (df_base["symbol"] == row["symbol"]) &
                (df_base["score_date"] == row["prediction_date"])
            ]
            if match.empty:
                continue
            actual = pd.to_numeric(match.iloc[0].get("return_10d"), errors="coerce")
            spy = pd.to_numeric(match.iloc[0].get("spy_10d_return", 0), errors="coerce") or 0
            if pd.isna(actual):
                continue
            direction = row["predicted_direction"]
            if direction == "UP":
                correct = 1 if actual > spy else 0
            elif direction == "DOWN":
                correct = 1 if actual < spy else 0
            elif direction == "FLAT":
                correct = 1 if abs(actual - spy) < 0.01 else 0
            else:
                continue
            conn.execute(
                "UPDATE predictor_outcomes SET actual_5d_return=?, correct_5d=? "
                "WHERE symbol=? AND prediction_date=?",
                (float(actual), correct, row["symbol"], row["prediction_date"]),
            )
        conn.commit()
        conn.close()
        logger.info("Backfilled predictor outcomes from score_performance data")
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
    _backfill_predictor_outcomes(config, df_base)
    _push_predictor_rolling_metrics(config, config.get("research_db", ""))

    return sq_result, regime_rows, score_rows, attr_result, df_base


def _read_current_weights(config: dict) -> dict:
    """
    Read current scoring weights from alpha-engine-research/config/universe.yaml.
    Falls back to weight_optimizer.DEFAULT_WEIGHTS if the research repo isn't found.
    """
    research_paths = config.get("research_paths", [])
    if isinstance(research_paths, str):
        research_paths = [research_paths]
    research_path = next((p for p in research_paths if os.path.isdir(p)), None)

    if not research_path:
        logger.warning(
            "research_paths not found on disk — using default scoring weights. "
            "Add research repo path to research_paths in config.yaml for accurate readings."
        )
        return weight_optimizer.DEFAULT_WEIGHTS.copy()

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

    return weight_optimizer.DEFAULT_WEIGHTS.copy()


def run_weight_optimizer(config: dict, df_base: pd.DataFrame) -> dict:
    """
    Run the weight optimizer: join sub-scores from signals.json in S3 with
    score_performance outcomes, then suggest revised scoring weights.

    Advisory only — no weights are changed. Output included in the weekly report.
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
        )
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

    # Build config_override from swept params that need to reach the executor
    config_override = _build_config_override(config)

    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
    all_orders = []
    dates_simulated = 0

    # Use signals_by_date keys as iteration dates when available
    if signals_by_date is not None:
        sim_dates = sorted(signals_by_date.keys())
    else:
        sim_dates = dates

    for signal_date in sim_dates:
        ts = pd.Timestamp(signal_date)
        if ts not in price_matrix.index:
            continue

        date_prices = price_matrix.loc[ts].dropna().to_dict()
        if not date_prices:
            continue

        # Load signals: from pre-built dict or from S3
        if signals_by_date is not None:
            signals_raw = signals_by_date[signal_date]
        else:
            from loaders import signal_loader
            try:
                signals_raw = signal_loader.load(bucket, signal_date)
            except FileNotFoundError:
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

    logger.info(
        "Simulation loop: %d dates, %d orders", dates_simulated, len(all_orders)
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


def _build_config_override(config: dict) -> dict | None:
    """
    Map flat sweep params in config to the nested executor config structure.

    Sweep grid uses flat keys (e.g. atr_multiplier) but the executor expects
    them nested under strategy.exit_manager. This function builds the override
    dict that executor.main.run(config_override=) can merge.
    """
    override = {}

    # Direct risk params (top-level in executor's risk.yaml)
    for key in ("min_score", "max_position_pct", "drawdown_circuit_breaker"):
        if key in config:
            override[key] = config[key]

    # Strategy params → nested under strategy.exit_manager
    strategy_keys = {
        "atr_multiplier": "atr_multiplier",
        "time_decay_reduce_days": "time_decay_reduce_days",
        "time_decay_exit_days": "time_decay_exit_days",
    }
    exit_manager_overrides = {}
    for sweep_key, config_key in strategy_keys.items():
        if sweep_key in config:
            exit_manager_overrides[config_key] = config[sweep_key]

    if exit_manager_overrides:
        override["strategy"] = {"exit_manager": exit_manager_overrides}

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


def run_param_sweep(config: dict):
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
    logger.info("Running param sweep: %s", {k: len(v) for k, v in grid.items()})
    return param_sweep.sweep(grid, sim_fn, config)


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
    )

    # Merge metadata into stats for reporting
    stats["predictor_metadata"] = metadata
    return stats


def run_predictor_param_sweep(config: dict) -> tuple[dict, pd.DataFrame]:
    """
    Run predictor-only backtest with param sweep.

    Loads data once (features, GBM inference, signal generation), then runs
    the simulation loop for each parameter combination.

    Returns (single_run_stats, sweep_df).
    """
    import sys
    from synthetic.predictor_backtest import run as run_predictor_pipeline

    # Prepare data once
    result = run_predictor_pipeline(config)

    if result.get("status") != "ok":
        return result, pd.DataFrame()

    signals_by_date = result["signals_by_date"]
    price_matrix = result["price_matrix"]
    ohlcv_by_ticker = result["ohlcv_by_ticker"]
    metadata = result["metadata"]

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
    )
    single_stats["predictor_metadata"] = metadata

    # Param sweep
    sweep_df = pd.DataFrame()
    grid = config.get("param_sweep")
    if grid:
        def sim_fn(combo_config: dict) -> dict:
            return _run_simulation_loop(
                executor_run, SimulatedIBKRClient,
                dates=[],
                price_matrix=price_matrix,
                config=combo_config,
                ohlcv_by_ticker=ohlcv_by_ticker,
                signals_by_date=signals_by_date,
            )

        logger.info("Running predictor param sweep: %s", {k: len(v) for k, v in grid.items()})
        sweep_df = param_sweep.sweep(grid, sim_fn, config)

    return single_stats, sweep_df


def main():
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
    sweep_df = None
    weight_result = None
    veto_result = None
    executor_rec = None
    predictor_stats = None
    predictor_sweep_df = None

    if args.mode in ("signal-quality", "all"):
        sq_result, regime_rows, score_rows, attr_result, df_base = run_signal_quality(config)
        weight_result = run_weight_optimizer(config, df_base)

        # Phase 3: Veto threshold analysis (needs score_performance data)
        if df_base is not None and not df_base.empty:
            try:
                bucket = config.get("signals_bucket", "alpha-engine-research")
                veto_result = veto_analysis.analyze_veto_effectiveness(df_base, bucket)
                if veto_result.get("status") == "ok":
                    veto_result["apply_result"] = veto_analysis.apply(veto_result, bucket)
            except Exception as e:
                logger.error("Veto analysis failed: %s", e)
                veto_result = {"status": "error", "error": str(e)}
    else:
        sq_result = {"status": "skipped"}
        regime_rows = []
        score_rows = []
        attr_result = {"status": "skipped"}
        df_base = None

    if args.mode in ("simulate", "all"):
        try:
            portfolio_stats = run_simulate(config)
        except Exception as e:
            logger.error("Mode 2 simulation failed: %s", e)
            portfolio_stats = {"status": "error", "error": str(e)}

    if args.mode in ("param-sweep", "all"):
        try:
            sweep_df = run_param_sweep(config)
        except Exception as e:
            logger.error("Param sweep failed: %s", e)
            sweep_df = None

        # Phase 2: Executor parameter optimization from sweep results
        if sweep_df is not None and not sweep_df.empty:
            try:
                executor_rec = executor_optimizer.recommend(sweep_df, config)
                if executor_rec.get("status") == "ok":
                    bucket = config.get("signals_bucket", "alpha-engine-research")
                    executor_rec["apply_result"] = executor_optimizer.apply(executor_rec, bucket)
            except Exception as e:
                logger.error("Executor optimizer failed: %s", e)
                executor_rec = {"status": "error", "error": str(e)}

    if args.mode == "predictor-backtest":
        try:
            predictor_stats, predictor_sweep_df = run_predictor_param_sweep(config)
        except Exception as e:
            logger.error("Predictor backtest failed: %s", e)
            predictor_stats = {"status": "error", "error": str(e)}
            predictor_sweep_df = None

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
    )

    # For predictor-backtest mode, use predictor_sweep_df as the sweep output
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

    if args.stop_instance:
        import urllib.request
        try:
            # Get instance ID from EC2 metadata endpoint
            token = urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/api/token",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                    method="PUT",
                ),
                timeout=2,
            ).read().decode()
            instance_id = urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/meta-data/instance-id",
                    headers={"X-aws-ec2-metadata-token": token},
                ),
                timeout=2,
            ).read().decode()
            logger.info("Stopping instance %s", instance_id)
            boto3.client("ec2").stop_instances(InstanceIds=[instance_id])
        except Exception as e:
            logger.error("Failed to stop instance: %s", e)


if __name__ == "__main__":
    main()
