"""
backtest.py — CLI entry point for alpha-engine-backtester (simulation only).

Runs portfolio simulation, parameter sweeps, and predictor backtests.
Evaluation logic (signal quality, diagnostics, optimizers) lives in evaluate.py.

Usage:
    # Portfolio simulation (requires executor_path in config.yaml)
    python backtest.py --mode simulate

    # Parameter sweep over risk + strategy params
    python backtest.py --mode param-sweep

    # Predictor-only backtest (10y synthetic signals, no LLM calls)
    python backtest.py --mode predictor-backtest

    # Full simulation pipeline (param-sweep + predictor-backtest)
    python backtest.py --mode all

    # Upload results to S3
    python backtest.py --mode all --upload

Options:
    --mode          simulate | param-sweep | all | predictor-backtest
    --config        path to config.yaml (default: ./config.yaml)
    --db            path to local research.db (skips S3 pull; useful locally)
    --upload        upload results to S3
    --date          run date label for output (default: today)
    --log-level     DEBUG | INFO | WARNING (default: INFO)
"""

import argparse
import json
import logging
import tempfile
import os
import time as _time
from datetime import date
from pathlib import Path

from ssm_secrets import load_secrets

load_secrets()

import boto3
import pandas as pd
import yaml

from analysis import param_sweep
from optimizer import executor_optimizer
from emailer import send_report_email
from reporter import build_report, save, upload_to_s3
from pipeline_common import load_config, phase, pull_research_db

logger = logging.getLogger(__name__)


# ── Simulation setup and execution ──────────────────────────────────────────


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
    logger.info("Building price matrix for %d dates (ArcticDB)...", len(dates))
    price_matrix = price_loader.build_matrix(dates, bucket, _ohlcv_out=ohlcv_by_ticker)

    if price_matrix.empty:
        return executor_run, SimulatedIBKRClient, dates, None, init_cash, {}

    logger.info("OHLCV captured for %d tickers (strategy layer)", len(ohlcv_by_ticker))
    return executor_run, SimulatedIBKRClient, dates, price_matrix, init_cash, ohlcv_by_ticker


_SIGNAL_LIST_FIELDS = ("universe", "buy_candidates", "enter", "exit", "reduce", "hold")


def _filter_signals_to_universe(
    signals: dict,
    universe_symbols: set[str],
    rejected_counter: dict[str, int] | None,
) -> dict:
    """Return a shallow-copied signals dict where every ticker-carrying list
    is filtered to entries whose ``ticker`` is in ``universe_symbols``.

    Rationale: simulate mode replays historical signals.json files from S3.
    Past constituent turnover (e.g. TSM/ASML dropped 2026-04-20) leaves
    historical signals referencing tickers no longer in ArcticDB. Executor
    hard-fail guards (load_daily_vwap, load_atr_14_pct) then abort the
    simulation. This filter drops those tickers at the simulate boundary —
    NOT at the executor layer, because live executor must preserve EXIT/
    REDUCE/HOLD for real held positions even if the ticker somehow went
    missing from ArcticDB (different concern: alarm, don't silently skip).
    In simulate mode there are no real held positions so the drop is safe.

    ``rejected_counter`` (if provided) accumulates per-ticker reject counts
    across the simulation loop for a single aggregate WARN log at end of run.
    Consistent with feedback_no_silent_fails: rejects are counted and
    reported, not silently dropped.
    """
    filtered = dict(signals)
    for field in _SIGNAL_LIST_FIELDS:
        entries = signals.get(field)
        if not isinstance(entries, list):
            continue
        kept = []
        for e in entries:
            ticker = (e.get("ticker") if isinstance(e, dict) else None) or ""
            ticker = ticker.upper()
            if ticker and ticker in universe_symbols:
                kept.append(e)
            elif ticker and rejected_counter is not None:
                rejected_counter[ticker] = rejected_counter.get(ticker, 0) + 1
        filtered[field] = kept
    return filtered


def _build_ohlcv_date_index(ohlcv_by_ticker: dict) -> dict[str, list[str]]:
    """Precompute the date axis for every ticker's OHLCV history.

    Returns ``{ticker: [date0, date1, ...]}`` — parallel to
    ``ohlcv_by_ticker[ticker]`` (same order, same length). The dates are
    already-sorted ISO8601 strings (``loaders/price_loader.py:142`` sorts
    bars by ``date`` string), so lexicographic comparison is chronological
    and ``bisect.bisect_right(dates, signal_date)`` gives the "<= signal_date"
    cut index in ``O(log N)``.

    Built once per simulation pipeline; every per-date
    ``_simulate_single_date`` call then slices with ``bars[:cut]`` instead
    of the prior ``[b for b in bars if b["date"] <= signal_date]`` Python-
    list comprehension. For a 900-ticker × 2500-bar × 2000-date predictor
    param sweep (60 combos) this cuts the filter inner loop from
    ~270B dict-lookup/compare ops to ~108B list-ref copies — roughly
    10x faster per the same micro-benchmark shape that motivated PR #46
    (2026-04-21 ``build_signals_by_date`` vectorization, 75min → ~1min).
    """
    return {
        ticker: [b["date"] for b in bars]
        for ticker, bars in (ohlcv_by_ticker or {}).items()
    }


def _simulate_single_date(
    executor_run,
    sim_client,
    signal_date: str,
    price_matrix,
    ohlcv_by_ticker: dict | None,
    bucket: str,
    config_override: dict | None,
    signals_override: dict | None = None,
    universe_symbols: set[str] | None = None,
    rejected_ticker_counter: dict[str, int] | None = None,
    ohlcv_dates_index: dict[str, list[str]] | None = None,
    atr_by_ticker: dict[str, float] | None = None,
    vwap_series_by_ticker: dict[str, pd.Series] | None = None,
    coverage_by_ticker: dict[str, float] | None = None,
) -> tuple[list[dict] | None, str | None]:
    """
    Run the executor once for a single signal date.

    Returns ``(orders_or_none, skip_reason)``. On successful run, returns
    ``(orders_list, None)`` — orders may be an empty list. On skip, returns
    ``(None, reason_key)`` where reason_key ∈ {no_price_index, empty_prices,
    no_signals}.

    Side effect: mutates ``sim_client._prices`` and ``sim_client._simulation_date``
    so state carries forward across calls (matching the existing simulation
    loop's semantics).

    Extracted from _run_simulation_loop in 2026-04-16 to support the replay
    parity test (Phase 1.1b of backtester-audit-260415.md) without duplicating
    the per-date orchestration logic.
    """
    import pandas as pd

    ts = pd.Timestamp(signal_date)
    if ts not in price_matrix.index:
        # Weekend/holiday signal dates: use next available trading day's prices
        later = price_matrix.index[price_matrix.index > ts]
        if len(later) > 0:
            ts = later[0]
            logger.debug("Signal date %s not in price index — using next trading day %s",
                         signal_date, ts.date())
        else:
            return None, "no_price_index"

    date_prices = price_matrix.loc[ts].dropna().to_dict()
    if not date_prices:
        return None, "empty_prices"

    if signals_override is not None:
        signals_raw = signals_override
    else:
        from loaders import signal_loader
        try:
            signals_raw = signal_loader.load(bucket, signal_date)
        except FileNotFoundError:
            return None, "no_signals"

    # Filter historical signals against TODAY's ArcticDB universe. Signals
    # from weeks past may contain tickers that have since been dropped from
    # the universe (2026-04-20 TSM/ASML Research↔Executor coverage-gap fix,
    # future constituent turnover, etc.). The executor's hard-fail guards
    # (load_daily_vwap, load_atr_14_pct) raise NoSuchVersionException on any
    # such ticker and abort the whole simulate run. Dropping them per-date
    # before executor_run is the simulate-specific defense-in-depth that
    # pairs with the live-executor buy_candidate filter (alpha-engine PR #77).
    if universe_symbols is not None:
        signals_raw = _filter_signals_to_universe(
            signals_raw, universe_symbols, rejected_ticker_counter,
        )

    sim_client._prices = date_prices
    sim_client._simulation_date = signal_date

    # Filter OHLCV histories to <= signal_date (no lookahead).
    #
    # 2026-04-22: vectorized from a per-date Python list comprehension to
    # bisect+slice after the Saturday SF dry-run timed out at the 2h SSM
    # ceiling. The old path iterated every bar of every ticker doing a
    # dict-lookup + string compare per bar — for a 60-combo × 2000-date
    # × 900-ticker × 2500-bar predictor param sweep that worked out to
    # ~270B inner ops. bisect.bisect_right is O(log N) against the
    # precomputed date axis; the list slice copies only references, not
    # bar data. Same shape as PR #46's ``build_signals_by_date`` fix.
    #
    # ``ohlcv_dates_index`` is threaded from the caller when available.
    # Fallback derivation keeps the old per-call callers working while
    # we migrate — it's the scalar path, kept under test via
    # ``test_price_histories_parity.py`` so future refactors can't drift.
    price_histories = None
    if ohlcv_by_ticker:
        if ohlcv_dates_index is not None:
            from bisect import bisect_right
            price_histories = {
                ticker: bars[:bisect_right(ohlcv_dates_index[ticker], signal_date)]
                for ticker, bars in ohlcv_by_ticker.items()
            }
        else:
            price_histories = {
                ticker: [b for b in bars if b["date"] <= signal_date]
                for ticker, bars in ohlcv_by_ticker.items()
            }

    # Precomputed feature-map injection (alpha-engine PR #91). When both
    # maps are available, resolve VWAP for this simulate date against the
    # in-memory series, then pass atr_map + vwap_map via the executor
    # kwargs to skip the per-call ArcticDB reads (``load_atr_14_pct``
    # and ``load_daily_vwap``). The roadmap's Saturday SF dry-run timed
    # out at 2h in precisely these two ArcticDB hot paths.
    #
    # Semantics: ``atr_by_ticker`` is a flat {ticker: value} dict — the
    # executor does ``.get(ticker)`` lookups and tolerates missing
    # tickers via fall-through. VWAP is resolved per simulate date via
    # ``resolve_vwap_map_for_date`` which mirrors
    # ``load_daily_vwap``'s walk-back semantics (up to 5 trading days).
    atr_map: dict | None = None
    vwap_map: dict | None = None
    coverage_map: dict | None = None
    if atr_by_ticker is not None:
        atr_map = atr_by_ticker  # executor filters via .get() per ticker
    if coverage_by_ticker is not None:
        coverage_map = coverage_by_ticker  # executor filters via .get() per ticker
    if vwap_series_by_ticker is not None:
        from store.feature_maps import resolve_vwap_map_for_date
        enter_tickers = [
            s["ticker"]
            for s in (signals_raw.get("enter") or [])
            if s.get("ticker")
        ]
        vwap_map = resolve_vwap_map_for_date(
            vwap_series_by_ticker, enter_tickers, signal_date,
        )

    orders = executor_run(
        simulate=True,
        ibkr_client=sim_client,
        signals_override=signals_raw,
        price_histories=price_histories,
        config_override=config_override,
        atr_map=atr_map,
        vwap_map=vwap_map,
        coverage_map=coverage_map,
    )
    # Tag each order with the simulation date for downstream parity diffing.
    # Executor-emitted orders may not include a date field (they carry
    # fill_time instead); parity needs a `date` key per docs/trade_mapping.md.
    for order in (orders or []):
        order.setdefault("date", signal_date)
    return list(orders or []), None


def _run_simulation_loop(
    executor_run,
    SimulatedIBKRClient,
    dates: list[str],
    price_matrix,
    config: dict,
    ohlcv_by_ticker: dict | None = None,
    signals_by_date: dict | None = None,
    spy_prices: pd.Series | None = None,
    ohlcv_dates_index: dict[str, list[str]] | None = None,
    atr_by_ticker: dict[str, float] | None = None,
    vwap_series_by_ticker: dict[str, pd.Series] | None = None,
    coverage_by_ticker: dict[str, float] | None = None,
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

    # Precompute OHLCV date axis once per simulate pass. Callers sharing
    # ``ohlcv_by_ticker`` across many simulate runs (param sweep, Phase 4
    # evaluations) can build this once at the top-level and pass it
    # in — otherwise we derive it here so every per-date
    # ``_simulate_single_date`` call gets the fast bisect+slice path
    # regardless of caller. Cost is a single ``O(N_tickers × N_bars)``
    # pass, negligible compared to the inner loop savings.
    if ohlcv_by_ticker and ohlcv_dates_index is None:
        ohlcv_dates_index = _build_ohlcv_date_index(ohlcv_by_ticker)

    # Precompute ATR + VWAP maps once per simulate pass — same motivation
    # as ohlcv_dates_index. The executor's ``load_atr_14_pct`` and
    # ``load_daily_vwap`` both hit ArcticDB per ticker per call (20+
    # round-trips per simulate call). The alpha-engine PR #91 kwargs
    # ``atr_map`` + ``vwap_map`` let the backtester inject pre-resolved
    # maps and skip those reads entirely. Callers can also pass in the
    # pre-built maps to avoid rebuilding per combo in param sweep.
    if (atr_by_ticker is None or vwap_series_by_ticker is None or coverage_by_ticker is None):
        from store.feature_maps import load_precomputed_feature_maps
        _atr, _vwap, _cov = load_precomputed_feature_maps(bucket)
        if atr_by_ticker is None:
            atr_by_ticker = _atr
        if vwap_series_by_ticker is None:
            vwap_series_by_ticker = _vwap
        if coverage_by_ticker is None:
            coverage_by_ticker = _cov

    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)
    all_orders: list[dict] = []
    dates_simulated = 0
    skip_reasons = {"no_price_index": 0, "empty_prices": 0, "no_signals": 0}

    # Load today's ArcticDB universe once — used to filter historical signals
    # that reference since-dropped tickers (e.g. TSM/ASML post-2026-04-20).
    # Hard-fail on ArcticDB library-open error: that's a pipeline precondition,
    # not a simulate-mode edge case to paper over.
    universe_symbols: set[str] | None = None
    rejected_ticker_counter: dict[str, int] = {}
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        universe_symbols = get_universe_symbols(bucket)
    except Exception as exc:
        # Fail loud: simulate would otherwise crash later at load_daily_vwap
        # when a historical signal references a dropped ticker, and the
        # failure surface would be a misleading "daemon cannot plan triggers"
        # instead of a clear "ArcticDB library unreachable."
        raise RuntimeError(
            f"Simulate universe-filter bootstrap failed: could not read "
            f"ArcticDB universe symbols from bucket {bucket!r}: {exc}"
        ) from exc

    # Use signals_by_date keys as iteration dates when available
    if signals_by_date is not None:
        sim_dates = sorted(signals_by_date.keys())
    else:
        sim_dates = dates

    # Per-date heartbeat — emit an INFO line every N dates so a long sim
    # can't go fully silent for more than a minute or two at a time. Before
    # this, ~2000 signal dates iterated with zero log output; combined with
    # a DEBUG-only per-combo log in param_sweep, a predictor-param-sweep
    # could run for >100 min without a single INFO line. See ROADMAP
    # P0 "Diagnose the silent-phase bottleneck" (2026-04-22 4th dry-run).
    _HEARTBEAT_EVERY = 250
    n_dates = len(sim_dates)
    t0 = _time.monotonic()

    for idx, signal_date in enumerate(sim_dates):
        signals_override = signals_by_date[signal_date] if signals_by_date is not None else None
        orders, skip = _simulate_single_date(
            executor_run=executor_run,
            sim_client=sim_client,
            signal_date=signal_date,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            config_override=config_override,
            signals_override=signals_override,
            universe_symbols=universe_symbols,
            rejected_ticker_counter=rejected_ticker_counter,
            ohlcv_dates_index=ohlcv_dates_index,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=vwap_series_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
        )
        if skip is not None:
            skip_reasons[skip] += 1
        else:
            if orders:
                all_orders.extend(orders)
            dates_simulated += 1

        if (idx + 1) % _HEARTBEAT_EVERY == 0 or (idx + 1) == n_dates:
            elapsed = _time.monotonic() - t0
            logger.info(
                "Simulation loop: %d/%d dates processed (%.1fs elapsed, last=%s)",
                idx + 1, n_dates, elapsed, signal_date,
            )

    _MIN_SIMULATION_COVERAGE = 0.80

    dates_expected = len(sim_dates)
    coverage = dates_simulated / dates_expected if dates_expected > 0 else 0
    skipped = {k: v for k, v in skip_reasons.items() if v > 0}
    logger.info(
        "Simulation: %d/%d dates (%.0f%% coverage), %d orders%s",
        dates_simulated, dates_expected, coverage * 100, len(all_orders),
        f" — skipped: {skipped}" if skipped else "",
    )

    if rejected_ticker_counter:
        # Aggregate reject log — loud so data drift (tickers dropped from
        # the universe between signal-write time and replay time) is visible.
        top = sorted(rejected_ticker_counter.items(), key=lambda kv: -kv[1])
        total_rejects = sum(rejected_ticker_counter.values())
        logger.warning(
            "Simulate universe-filter dropped %d signal entries across %d "
            "tickers (tickers present in historical signals but absent from "
            "current ArcticDB universe). Top offenders: %s",
            total_rejects, len(rejected_ticker_counter),
            [f"{t}={n}" for t, n in top[:10]],
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


# ── Replay helper for parity testing (Phase 1.1b) ──────────────────────────


def replay_for_dates(
    dates: list[str],
    config: dict,
    *,
    warmup_from_full_history: bool = True,
) -> list[dict]:
    """
    Replay the backtester for a specific list of signal dates; return
    aggregated orders tagged with ``date``.

    Primary consumer: ``tests/test_parity_replay.py`` (Phase 1.1 replay
    parity test). See ``docs/trade_mapping.md`` for the tolerance contract
    used to diff the returned orders against ``trades.db``.

    Parameters
    ----------
    dates : signal dates to replay orders for, ``"YYYY-MM-DD"`` each.
    config : loaded via ``pipeline_common.load_config``.
    warmup_from_full_history : if True (default), replay the FULL historical
        signal stream up through the latest requested date so the sim_client's
        NAV / positions have time to evolve before the test window. Only
        orders on ``dates`` are returned. If False, only the requested dates
        are simulated starting from ``init_cash`` — fast but NAV-divergent.

    Returns
    -------
    list of order dicts — each with at minimum ``date``, ``ticker``, ``action``.
    Empty list on any simulation setup failure (stale prices, no price index).

    State-reconstruction note
    -------------------------
    Neither mode perfectly reconstructs the live executor's state at each
    date — live NAV on any given date reflects prior realized P&L that the
    backtester's simulated P&L can drift from. ``position_pct`` tolerance
    in ``docs/trade_mapping.md`` accounts for small drift; large drift is
    a signal of logic divergence worth investigating.
    """
    executor_run, SimulatedIBKRClient, all_signal_dates, price_matrix, init_cash, ohlcv_by_ticker = \
        _setup_simulation(config)

    # Hard-fail on setup-level problems per feedback_no_silent_fails.
    # Returning [] here would let the parity test interpret "no orders" as a
    # legitimate backtester outcome, surfacing every live trade as a spurious
    # "only_live" divergence — logic failure indistinguishable from data
    # failure. Raising surfaces the actual cause in the test error message.
    if price_matrix is None:
        raise RuntimeError(
            "replay_for_dates: _setup_simulation returned no price matrix — "
            "cannot replay. Likely causes: ArcticDB unreachable, empty signal "
            "history, or fewer than `min_simulation_dates` signal dates in S3."
        )
    if getattr(price_matrix, "attrs", {}).get("stale_circuit_break"):
        raise RuntimeError(
            f"replay_for_dates: price-matrix staleness circuit-breaker tripped "
            f"({price_matrix.attrs.get('staleness_warning')}). Refusing to "
            f"produce parity output against stale prices."
        )

    requested = set(dates)
    bucket = config.get("signals_bucket", "alpha-engine-research")
    config_override = _build_config_override(config)
    sim_client = SimulatedIBKRClient(prices={}, nav=init_cash)

    if warmup_from_full_history and dates:
        latest_requested = max(dates)
        sim_dates = [d for d in all_signal_dates if d <= latest_requested]
    else:
        sim_dates = sorted(dates)

    # One-time OHLCV date-axis build for the same reason as in
    # ``_run_simulation_loop`` — avoids rebuilding per ``signal_date``.
    ohlcv_dates_index = _build_ohlcv_date_index(ohlcv_by_ticker) if ohlcv_by_ticker else None

    captured: list[dict] = []
    for signal_date in sim_dates:
        orders, _skip = _simulate_single_date(
            executor_run=executor_run,
            sim_client=sim_client,
            signal_date=signal_date,
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket=bucket,
            config_override=config_override,
            signals_override=None,  # load from S3 per date
            ohlcv_dates_index=ohlcv_dates_index,
        )
        if orders and signal_date in requested:
            captured.extend(orders)

    logger.info(
        "replay_for_dates: %d orders captured across %d requested dates "
        "(warmup=%s, replayed=%d)",
        len(captured), len(requested), warmup_from_full_history, len(sim_dates),
    )
    return captured


# ── Param sweep helpers ─────────────────────────────────────────────────────


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


# ── Convenience wrappers ────────────────────────────────────────────────────


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

    # Build the OHLCV date-axis index ONCE — every combo's simulate pass
    # would otherwise rebuild it redundantly in _run_simulation_loop.
    ohlcv_dates_index = _build_ohlcv_date_index(ohlcv) if ohlcv else None

    # Precompute ATR + VWAP + coverage maps ONCE across the full combo
    # sweep. Without this, _run_simulation_loop derives them lazily per
    # combo and we repay the ~900-ticker ArcticDB bulk read 60 times.
    from store.feature_maps import load_precomputed_feature_maps
    bucket = config.get("signals_bucket", "alpha-engine-research")
    atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = load_precomputed_feature_maps(bucket)

    def sim_fn(combo_config: dict) -> dict:
        return _run_simulation_loop(
            executor_run, SimulatedIBKRClient, dates, price_matrix, combo_config,
            ohlcv_by_ticker=ohlcv,
            ohlcv_dates_index=ohlcv_dates_index,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=vwap_series_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
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
    with phase("predictor_data_prep"):
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

    # One-time OHLCV date axis build. Shared across the single-run sim,
    # every Phase 4 evaluation that also runs a full simulation
    # (ensemble_modes, signal_thresholds, feature_pruning), and every
    # param-sweep combo. Without this, each of those would rebuild the
    # index redundantly — 60+ rebuilds for a full predictor-param-sweep
    # over 2000+ dates, each ~2s. See _build_ohlcv_date_index docstring
    # for the inner-loop cost math.
    ohlcv_dates_index = _build_ohlcv_date_index(ohlcv_by_ticker) if ohlcv_by_ticker else None

    # Same one-time-share logic for ATR + VWAP precomputed maps. The
    # predictor-param-sweep is the bottleneck the Saturday SF dry-run
    # timed out on: 60 combos × 2000+ dates × per-ticker ArcticDB reads.
    # Loading once up front collapses that to a single bulk scan (~1-2
    # min for ~900 tickers with 20-way concurrency) — every subsequent
    # _simulate_single_date call reuses the in-memory maps.
    with phase("predictor_feature_maps_bulk_load"):
        from store.feature_maps import load_precomputed_feature_maps
        _bucket_for_maps = config.get("signals_bucket", "alpha-engine-research")
        atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = load_precomputed_feature_maps(_bucket_for_maps)

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
    with phase("predictor_single_run", n_dates=len(signals_by_date)):
        single_stats = _run_simulation_loop(
            executor_run, SimulatedIBKRClient,
            dates=[],
            price_matrix=price_matrix,
            config=config,
            ohlcv_by_ticker=ohlcv_by_ticker,
            signals_by_date=signals_by_date,
            spy_prices=spy_prices,
            ohlcv_dates_index=ohlcv_dates_index,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=vwap_series_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
        )
    single_stats["predictor_metadata"] = metadata

    # ── Phase 4: Predictor hyperparameter feedback ───────────────────────
    # `skip_phase4_evaluations` (config flag, set by --skip-phase4-evaluations
    # CLI or SF input): bypass the three Phase 4 evaluators wholesale. Each
    # runs a full silent simulation internally and can add tens of minutes
    # to the predictor pipeline. For dry-runs where we only care "does the
    # pipeline complete end-to-end", skipping is cheap and safe — the S3
    # config promotions will have nothing to apply, so the next real run
    # picks up the existing configs unchanged.
    predictions_by_date = result.get("predictions_by_date", {})
    if config.get("skip_phase4_evaluations"):
        logger.info(
            "Phase 4 predictor-hyperparameter feedback SKIPPED "
            "(skip_phase4_evaluations=true). Ensemble mode / signal "
            "threshold / feature pruning evaluators will not run."
        )
    elif features_by_ticker and trading_dates:
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
                with phase("phase4a_ensemble_modes"):
                    ensemble_result = evaluate_ensemble_modes(
                        features_by_ticker, price_matrix, ohlcv_by_ticker,
                        spy_prices, sector_map, trading_dates,
                        config, single_stats,
                    )
                single_stats["ensemble_eval"] = ensemble_result
            except Exception as exc:
                # Bumped from warning to error so flow-doctor captures it.
                # Previously logged at warning and the spot run stayed
                # green even when the optimizer couldn't evaluate ensemble
                # mode, which meant param recommendations were based on
                # partial sweep data.
                logger.error(
                    "Phase 4a ensemble mode evaluation failed: %s — "
                    "optimizer recommendations may be incomplete",
                    exc, exc_info=True,
                )

            # Phase 4b: Signal threshold sweep
            threshold_result = None
            if predictions_by_date:
                try:
                    with phase("phase4b_signal_thresholds"):
                        threshold_result = evaluate_signal_thresholds(
                            predictions_by_date, sector_map, ohlcv_by_ticker,
                            price_matrix, spy_prices, trading_dates,
                            config, single_stats,
                        )
                    single_stats["threshold_eval"] = threshold_result
                except Exception as exc:
                    logger.error(
                        "Phase 4b signal threshold evaluation failed: %s",
                        exc, exc_info=True,
                    )

            # Phase 4c: Feature pruning evaluation
            pruning_result = None
            try:
                with phase("phase4c_feature_pruning"):
                    pruning_result = evaluate_feature_pruning(
                        features_by_ticker, price_matrix, ohlcv_by_ticker,
                        spy_prices, sector_map, trading_dates,
                        config, single_stats,
                    )
                single_stats["pruning_eval"] = pruning_result
            except Exception as exc:
                logger.error(
                    "Phase 4c feature pruning evaluation failed: %s",
                    exc, exc_info=True,
                )

            # Apply recommendations to S3 (if any)
            try:
                apply_result = apply_recommendations(
                    ensemble_result, pruning_result, bucket,
                    threshold_result=threshold_result,
                )
                single_stats["predictor_optimizer_apply"] = apply_result
            except Exception as exc:
                # This one is especially important — if the apply fails,
                # the optimizer's recommendations don't get persisted to
                # S3, so the predictor keeps running on stale params.
                logger.error(
                    "Predictor optimizer apply failed (recommendations "
                    "not persisted to S3): %s", exc, exc_info=True,
                )

        except ImportError as exc:
            logger.error(
                "Phase 4 optimizer not available (import failed): %s",
                exc, exc_info=True,
            )

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
                ohlcv_dates_index=ohlcv_dates_index,
                atr_by_ticker=atr_by_ticker,
                vwap_series_by_ticker=vwap_series_by_ticker,
                coverage_by_ticker=coverage_by_ticker,
            )

        sweep_settings = config.get("param_sweep_settings", {})

        logger.info("Running predictor param sweep (%s): %s", sweep_settings.get("mode", "random"), {k: len(v) for k, v in grid.items()})
        with phase("predictor_param_sweep", combos=sum(len(v) for v in grid.values())):
            sweep_df = param_sweep.sweep(grid, sim_fn, config, sweep_settings=sweep_settings)

    return single_stats, sweep_df


# ── Infrastructure helpers ──────────────────────────────────────────────────


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


# ── Runtime smoke test ──────────────────────────────────────────────────────


_SMOKE_SAMPLE_TICKERS = ("AAPL", "MSFT", "NVDA", "JNJ", "PG")


def _runtime_smoke(config: dict) -> None:
    """End-to-end smoke test with minimal data.

    Exercises the SAME module imports + S3 reads + ArcticDB reads + model
    load paths as the full backtest, but scoped to a handful of tickers
    and a single recent signal date so it completes in ~30-60 seconds.

    Runs after BacktesterPreflight to catch environment issues that the
    cheap preflight can't see from import checks alone:
      - Actual ArcticDB `read()` works (not just `list_symbols`)
      - signal_loader resolves a usable signals.json
      - The Layer-1A GBM booster loads and predicts on a real feature
        tensor with `scorer.feature_names` populated

    Raises RuntimeError with a named ``[stage=X]`` prefix on the first
    failure so the operator sees exactly where in the end-to-end chain
    the real problem is — not "your 80-minute backtest died in stage N."

    Motivated by the 2026-04-21 Saturday SF dry-run where
    ``No module named 'alpha_engine_lib.arcticdb'`` surfaced ~80 minutes
    into a spot run. With preflight + runtime smoke, the same failure
    would surface in ~2 seconds (preflight) or ~30 seconds (smoke).
    """
    import numpy as np
    bucket = config.get("signals_bucket", "alpha-engine-research")

    def _fail(stage: str, exc: Exception) -> RuntimeError:
        return RuntimeError(
            f"Runtime smoke FAILED [stage={stage}]: {exc}. "
            "Full backtest is aborted to avoid 60-80 minutes of wasted "
            "spot compute. Fix the underlying issue and re-run."
        )

    # Stage 1: universe symbols end-to-end (catches lib/arcticdb issues
    # that preflight's import check only surfaces at import time).
    try:
        from alpha_engine_lib.arcticdb import get_universe_symbols
        symbols = get_universe_symbols(bucket)
        if not symbols:
            raise RuntimeError("empty universe — ArcticDB has zero symbols")
        sample = [t for t in _SMOKE_SAMPLE_TICKERS if t in symbols][:3]
        if not sample:
            raise RuntimeError(
                f"none of {_SMOKE_SAMPLE_TICKERS} are in the current universe "
                f"({len(symbols)} symbols) — universe drift or a broken library"
            )
    except Exception as exc:
        raise _fail("universe_symbols", exc) from exc
    logger.info("Smoke [universe_symbols]: %d symbols, sample=%s", len(symbols), sample)

    # Stage 2: per-ticker ArcticDB read (catches per-symbol read failures
    # that list_symbols alone wouldn't surface).
    try:
        from alpha_engine_lib.arcticdb import open_universe_lib
        lib = open_universe_lib(bucket)
        for t in sample:
            df = lib.read(t).data
            if df.empty:
                raise RuntimeError(f"{t}: empty frame")
    except Exception as exc:
        raise _fail("arcticdb_per_ticker_read", exc) from exc
    logger.info("Smoke [arcticdb_per_ticker_read]: %d tickers read OK", len(sample))

    # Stage 3: recent signals.json loads and parses. Simulate mode
    # depends on this working for every replayed date; if the most
    # recent one can't be loaded the full replay would also fail.
    try:
        from loaders import signal_loader
        # signal_loader has `list_dates` or similar — fall back to a
        # direct S3 list if needed. Scope lookback generously.
        recent = _latest_signals_date(bucket)
        if recent is None:
            raise RuntimeError("no signals/{date}/signals.json found in S3 (14d lookback)")
        signals_raw = signal_loader.load(bucket, recent)
        if not isinstance(signals_raw, dict) or not signals_raw.get("date"):
            raise RuntimeError(f"{recent}/signals.json parsed but missing 'date' field")
    except Exception as exc:
        raise _fail("signals_load", exc) from exc
    logger.info("Smoke [signals_load]: loaded %s/signals.json OK", recent)

    # Stage 4: Layer-1A GBM loads and predicts. Covers the
    # download_gbm_model + GBMScorer.load + scorer.predict path that
    # predictor-backtest mode exercises over 10y. A single tensor of
    # zeros is enough to verify feature_names is populated + the
    # booster is callable.
    try:
        from synthetic.predictor_backtest import download_gbm_model
        from model.gbm_scorer import GBMScorer
        model_path = download_gbm_model(bucket=bucket)
        try:
            scorer = GBMScorer.load(model_path)
            if not scorer.feature_names:
                raise RuntimeError("loaded scorer has empty feature_names")
            X = np.zeros((len(sample), len(scorer.feature_names)), dtype=np.float32)
            preds = scorer.predict(X)
            if len(preds) != len(sample):
                raise RuntimeError(
                    f"prediction shape mismatch: expected {len(sample)}, got {len(preds)}"
                )
        finally:
            # Always clean up the temp file — _runtime_smoke runs before
            # the full modes and the temp downloads would otherwise
            # accumulate in /tmp across smoke + full invocations.
            import os
            for p in (model_path, model_path + ".meta.json"):
                if os.path.exists(p):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
    except Exception as exc:
        raise _fail("gbm_load_predict", exc) from exc
    logger.info(
        "Smoke [gbm_load_predict]: scorer loaded, feature_names populated (%d features), "
        "predict returned %d values",
        len(scorer.feature_names), len(preds),
    )

    logger.info("Runtime smoke PASSED — proceeding to full backtest modes")


def _latest_signals_date(bucket: str, max_lookback: int = 14) -> str | None:
    """Return the most recent date (YYYY-MM-DD) whose signals.json is in S3,
    or None if none found within ``max_lookback`` calendar days.

    Walked day-by-day via HEAD object rather than listing — a single HEAD
    is cheaper than a ListObjectsV2 call and easier to reason about in
    the smoke path.
    """
    import boto3
    from datetime import date, timedelta
    s3 = boto3.client("s3")
    today = date.today()
    for days_back in range(max_lookback + 1):
        candidate = today - timedelta(days=days_back)
        key = f"signals/{candidate.isoformat()}/signals.json"
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return candidate.isoformat()
        except Exception:
            continue
    return None


# ── Pipeline orchestration ──────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Alpha Engine Backtester (simulation)")
    parser.add_argument("--mode", choices=["simulate", "param-sweep", "all", "predictor-backtest", "smoke"],
                        default="simulate",
                        help="Pipeline mode. 'smoke' runs preflight + end-to-end runtime smoke with "
                             "minimal data (~30-60s) then exits 0 — used by spot_backtest.sh --smoke-only "
                             "and as the gate before any full run.")
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
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Bypass the runtime smoke test that precedes the full modes. Only for "
                             "genuine restart cases where the operator knows the environment is good; "
                             "default behavior is to always run the ~30-60s smoke before committing "
                             "to 60-80 minutes of full work.")
    parser.add_argument("--skip-phase4-evaluations", action="store_true",
                        help="Skip Phase 4 predictor-hyperparameter feedback (ensemble mode, signal "
                             "threshold, feature pruning). Each Phase 4 evaluator runs a full silent "
                             "simulation internally; skipping all three shaves the predictor-pipeline "
                             "runtime dramatically during dry-runs where we only want 'does the "
                             "pipeline complete end-to-end'. Defaults to running. Routable from the "
                             "Saturday Step Function input as `skip_phase4_evaluations: true`.")
    return parser.parse_args()


def _init_pipeline(args: argparse.Namespace, config: dict) -> None:
    """Initialize optimizer modules and pull research DB if needed.

    Mutates config in place (adds research_db, _db_pull_status).
    """
    executor_optimizer.init_config(config)

    if args.db:
        config["research_db"] = args.db
        logger.info("Using local research.db: %s", args.db)


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

    # Precomputed feature maps — built ONCE per _run_simulation_pipeline
    # invocation, shared across simulate, param-sweep, holdout, and twin
    # sub-stages. Without this hoist, every sim_fn closure below lazily
    # derives the maps inside _run_simulation_loop, and the param-sweep
    # path pays 60× the ~900-ticker ArcticDB bulk read (2026-04-22 13:00
    # PT re-run timed out for exactly this reason — py-spy confirmed every
    # combo was re-entering load_precomputed_feature_maps). The guard
    # matches the shape of the simulate/sweep blocks: skip the read
    # entirely when _sim_setup is None or price_matrix is empty.
    ohlcv_dates_index = None
    atr_by_ticker = None
    vwap_series_by_ticker = None
    coverage_by_ticker = None
    if (
        _sim_setup is not None
        and _sim_setup[3] is not None  # price_matrix
        and args.mode in ("simulate", "param-sweep", "all")
    ):
        _ohlcv = _sim_setup[5]
        if _ohlcv:
            ohlcv_dates_index = _build_ohlcv_date_index(_ohlcv)
        try:
            from store.feature_maps import load_precomputed_feature_maps
            bucket = config.get("signals_bucket", "alpha-engine-research")
            atr_by_ticker, vwap_series_by_ticker, coverage_by_ticker = load_precomputed_feature_maps(bucket)
        except Exception as exc:
            # Fall through to lazy per-call derivation rather than abort.
            # Preserves existing behavior on a bulk-read failure — each
            # _run_simulation_loop call will hit the slower ArcticDB path
            # individually. Logged loud so the perf regression is visible.
            logger.warning(
                "feature_maps: bulk precompute failed (%s) — falling back "
                "to per-call ArcticDB reads inside _run_simulation_loop. "
                "Param sweep will run at the pre-PR-#50 rate.",
                exc,
            )

    # ── Simulate mode ─────────────────────────────────────────────────────
    if args.mode in ("simulate", "all"):
        try:
            with phase("simulate", mode=args.mode):
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
                            ohlcv_dates_index=ohlcv_dates_index,
                            atr_by_ticker=atr_by_ticker,
                            vwap_series_by_ticker=vwap_series_by_ticker,
                            coverage_by_ticker=coverage_by_ticker,
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
            with phase("param_sweep", mode=args.mode):
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
                                ohlcv_dates_index=ohlcv_dates_index,
                                atr_by_ticker=atr_by_ticker,
                                vwap_series_by_ticker=vwap_series_by_ticker,
                                coverage_by_ticker=coverage_by_ticker,
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
                with phase("executor_optimizer", mode=args.mode):
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
                                    ohlcv_dates_index=ohlcv_dates_index,
                                    atr_by_ticker=atr_by_ticker,
                                    vwap_series_by_ticker=vwap_series_by_ticker,
                                    coverage_by_ticker=coverage_by_ticker,
                                )
                            with phase("executor_holdout", mode=args.mode):
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
                                    ohlcv_dates_index=ohlcv_dates_index,
                                    atr_by_ticker=atr_by_ticker,
                                    vwap_series_by_ticker=vwap_series_by_ticker,
                                    coverage_by_ticker=coverage_by_ticker,
                                )
                            with phase("executor_twin_sim", mode=args.mode):
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


# ── Main entry point ────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()

    # Structured logging + flow-doctor singleton come from
    # alpha_engine_lib. setup_logging() configures the root logger and,
    # when FLOW_DOCTOR_ENABLED=1, initializes the shared FlowDoctor
    # instance using the config at flow-doctor.yaml and attaches its
    # ERROR-level log handler. Respects the --log-level CLI flag.
    from alpha_engine_lib.logging import setup_logging, get_flow_doctor
    _flow_doctor_yaml = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow-doctor.yaml")
    setup_logging("backtest", flow_doctor_yaml=_flow_doctor_yaml)
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    _health_start = _time.time()

    fd = get_flow_doctor()

    config = load_config(args.config)

    # Stamp CLI flags into config so deep-pipeline code (run_predictor_param_sweep
    # and below) can read them without threading args all the way down.
    if args.skip_phase4_evaluations:
        config["skip_phase4_evaluations"] = True

    # Preflight: external-world handshakes must pass before any 90-min
    # spot run starts. Raises RuntimeError (propagates to non-zero exit)
    # on missing env vars, unreachable S3, or stale ArcticDB macro/SPY.
    # Kept out of --rollback path because rollback touches S3 configs
    # only, not ArcticDB.
    if not args.rollback:
        with phase("preflight", mode=args.mode):
            from preflight import BacktesterPreflight
            BacktesterPreflight(
                bucket=config.get("signals_bucket", "alpha-engine-research"),
                mode="backtest",
                executor_paths=config.get("executor_paths") or [],
                predictor_paths=config.get("predictor_paths") or [],
            ).run()

        # Runtime smoke: end-to-end sanity with minimal data (~30-60s).
        # Runs after preflight (so any preflight failure surfaces first,
        # in seconds) and before any full mode (so an environment bug
        # doesn't burn 60-80 min of spot compute). --skip-smoke is the
        # escape hatch for genuine restart scenarios; --mode=smoke runs
        # the smoke and exits 0 without doing full work.
        if args.mode == "smoke":
            with phase("runtime_smoke", mode=args.mode):
                _runtime_smoke(config)
            logger.info("Smoke-only mode complete — exiting 0 without full run")
            return
        if not args.skip_smoke:
            with phase("runtime_smoke", mode=args.mode):
                _runtime_smoke(config)

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
    portfolio_stats = None
    sweep_df = None
    executor_rec = None
    predictor_stats = None
    predictor_sweep_df = None

    # ── Simulation setup (shared by simulate + param-sweep) ───────────────
    _sim_setup = None
    if args.mode in ("simulate", "param-sweep", "all"):
        try:
            with phase("simulation_setup", mode=args.mode):
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
        with phase("simulation_pipeline", mode=args.mode):
            portfolio_stats, sweep_df, executor_rec = _run_simulation_pipeline(
                args, config, _sim_setup, current_executor_params, fd,
            )

    # ── Predictor backtest ────────────────────────────────────────────────
    if args.mode in ("predictor-backtest", "all"):
        with phase("predictor_pipeline", mode=args.mode):
            predictor_stats, predictor_sweep_df, executor_rec = _run_predictor_pipeline(
                args, config, executor_rec, current_executor_params, fd,
            )

    # ── Export simulation artifacts for evaluator ────────────────────────
    if args.mode in ("simulate", "param-sweep", "all", "predictor-backtest"):
        try:
            with phase("export_artifacts", mode=args.mode):
                _export_simulation_artifacts(config, args.date, sweep_df=sweep_df, predictor_sweep_df=predictor_sweep_df, portfolio_stats=portfolio_stats, predictor_stats=predictor_stats)
        except Exception as e:
            logger.warning("Simulation artifact export failed (non-fatal): %s", e)

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

        # Eval-only kwargs (weight_result, veto_result, grading, etc.) default
        # to None in build_report — they are populated by evaluate.py, not here.
        report_md = build_report(
            run_date=args.date,
            signal_quality={"status": "skipped"},
            regime_analysis=[],
            score_analysis=[],
            attribution={"status": "skipped"},
            portfolio_stats=portfolio_stats,
            sweep_df=sweep_df,
            config=config,
            predictor_stats=predictor_stats,
            predictor_sweep_df=predictor_sweep_df,
            executor_rec=executor_rec,
            pipeline_health=pipeline_health,
        )

        save_sweep_df = sweep_df
        if predictor_sweep_df is not None and not predictor_sweep_df.empty:
            save_sweep_df = predictor_sweep_df

        out_dir = save(
            report_md=report_md,
            signal_quality={"status": "skipped"},
            score_analysis=[],
            sweep_df=save_sweep_df,
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
                status="simulation",
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
            if executor_rec and executor_rec.get("apply_result", {}).get("applied"):
                configs_applied.append("executor_params")
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
