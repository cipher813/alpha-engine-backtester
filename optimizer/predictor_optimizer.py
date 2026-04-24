"""
optimizer/predictor_optimizer.py — Phase 4: Predictor hyperparameter feedback.

Phase 4a: Ensemble mode evaluation
  - Downloads alternative model variants (MSE, Rank, CatBoost) from S3
  - Runs synthetic backtest with each variant using the same data pipeline
  - Compares portfolio Sharpe and recommends the best ensemble mode

Phase 4b: Signal threshold sweep
  - Sweeps alpha cutoffs on the GBM's continuous output (no retraining)
  - Only tickers with predicted alpha >= threshold enter signal generation
  - Compares portfolio Sharpe across thresholds and recommends optimal cutoff

Phase 4c: Feature pruning recommendations
  - Reads noise_candidates from training_summary_latest.json
  - Runs synthetic backtest with and without noise features
  - Recommends pruning if Sharpe holds or improves

All phases reuse the data pipeline outputs (features, prices, OHLCV, SPY)
from the primary predictor backtest — no redundant data loading.

Writes recommendations to config/predictor_params.json (merges with existing
veto_confidence field, does not overwrite).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from datetime import date

import boto3
import numpy as np
import pandas as pd

from pipeline_common import phase

log = logging.getLogger(__name__)

# Emit an INFO heartbeat every N dates inside the per-variant inference
# loop. Motivation: _run_variant_inference iterates ~2500 trading dates
# silently; with 3 variants this was 3 silent stretches inside the
# phase4a block. See ROADMAP "Diagnose the silent-phase bottleneck"
# (2026-04-22 4th dry-run).
_VARIANT_INFERENCE_HEARTBEAT = 500

# ── S3 model variant keys ────────────────────────────────────────────────────
_MODEL_VARIANTS = {
    "mse": {
        "weights_key": "predictor/weights/gbm_mse_latest.txt",
        "meta_key": "predictor/weights/gbm_mse_latest.txt.meta.json",
        "scorer_cls": "GBMScorer",
    },
    "rank": {
        "weights_key": "predictor/weights/gbm_rank_latest.txt",
        "meta_key": "predictor/weights/gbm_rank_latest.txt.meta.json",
        "scorer_cls": "GBMScorer",
    },
    "catboost": {
        "weights_key": "predictor/weights/catboost_latest.cbm",
        "meta_key": "predictor/weights/catboost_latest.cbm.meta.json",
        "scorer_cls": "CatBoostScorer",
    },
}

# Guardrails
_MIN_SHARPE_IMPROVEMENT = 0.10  # 10% relative improvement required
_MIN_YEARS_DATA = 5  # minimum years of backtest data for mode recommendation
_MIN_TRADING_DAYS_5Y = 5 * 252  # ~1260 trading days


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4a: Ensemble Mode Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ensemble_modes(
    features_by_ticker: dict[str, pd.DataFrame],
    price_matrix: pd.DataFrame,
    ohlcv_by_ticker: dict,
    spy_prices: pd.Series | None,
    sector_map: dict[str, str],
    trading_dates: list[str],
    config: dict,
    baseline_stats: dict,
) -> dict:
    """
    Evaluate alternative model variants against the baseline (promoted) model.

    Uses the same features/prices from the primary backtest — only swaps the
    GBM model for inference, then re-generates signals and re-simulates.

    Parameters
    ----------
    features_by_ticker : pre-computed features from ArcticDB or inline
    price_matrix : date x ticker close prices
    ohlcv_by_ticker : {ticker: [{date, open, high, low, close}]}
    spy_prices : SPY close series
    sector_map : {ticker: sector_etf}
    trading_dates : ordered list of date strings
    config : full backtest config dict
    baseline_stats : portfolio stats from the primary (promoted) model run

    Returns
    -------
    dict with variant results and recommendation
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    predictor_path = _resolve_predictor_path(config)
    if not predictor_path:
        return {"status": "skipped", "reason": "no_predictor_path"}

    pb_config = config.get("predictor_backtest", {})
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)

    # Check data coverage for guardrail
    n_dates = len(trading_dates)
    has_sufficient_data = n_dates >= _MIN_TRADING_DAYS_5Y

    baseline_sharpe = baseline_stats.get("sharpe_ratio", 0)
    if not baseline_sharpe or np.isnan(baseline_sharpe):
        log.info("Ensemble eval: baseline Sharpe is NaN/zero — skipping")
        return {"status": "skipped", "reason": "no_baseline_sharpe"}

    # Discover which alternative models exist on S3
    available_variants = _discover_model_variants(bucket)
    if not available_variants:
        log.info("Ensemble eval: no alternative model variants found on S3")
        return {"status": "skipped", "reason": "no_alternative_models"}

    log.info("Ensemble eval: found %d variant(s): %s", len(available_variants), list(available_variants.keys()))

    # Run backtest for each variant. Each variant is wrapped in a
    # PHASE_START/END marker so a stall inside one variant's inference
    # or simulation loop is attributable to a specific mode.
    variant_results = {}
    for mode, variant_info in available_variants.items():
        try:
            log.info("Ensemble eval [%s]: starting variant backtest", mode)
            t0 = time.monotonic()
            with phase("phase4a_variant", variant=mode):
                stats = _run_variant_backtest(
                    mode, variant_info, features_by_ticker, price_matrix,
                    ohlcv_by_ticker, spy_prices, sector_map, trading_dates,
                    config, predictor_path, top_n, min_score,
                )
            variant_results[mode] = stats
            log.info(
                "Ensemble eval [%s]: Sharpe=%.3f  alpha=%.3f  max_dd=%.3f  (%.1fs)",
                mode, stats.get("sharpe_ratio", 0),
                stats.get("total_alpha", 0), stats.get("max_drawdown", 0),
                time.monotonic() - t0,
            )
        except Exception as exc:
            log.warning("Ensemble eval [%s]: failed — %s", mode, exc)
            variant_results[mode] = {"status": "error", "error": str(exc)}

    # Compare variants to baseline
    recommendation = _pick_best_mode(baseline_stats, variant_results, has_sufficient_data)

    result = {
        "date": str(date.today()),
        "baseline_mode": "promoted",
        "baseline_sharpe": round(baseline_sharpe, 4),
        "baseline_alpha": round(baseline_stats.get("total_alpha", 0) or 0, 4),
        "variants": {
            mode: {
                "sharpe_ratio": round(s.get("sharpe_ratio", 0) or 0, 4),
                "total_alpha": round(s.get("total_alpha", 0) or 0, 4),
                "max_drawdown": round(s.get("max_drawdown", 0) or 0, 4),
                "total_trades": s.get("total_trades", 0),
            }
            for mode, s in variant_results.items()
            if "sharpe_ratio" in s
        },
        "n_trading_days": n_dates,
        "sufficient_data": has_sufficient_data,
        **recommendation,
    }

    return result


def _discover_model_variants(bucket: str) -> dict:
    """Check which alternative model files exist on S3."""
    s3 = boto3.client("s3")
    available = {}
    for mode, info in _MODEL_VARIANTS.items():
        try:
            s3.head_object(Bucket=bucket, Key=info["weights_key"])
            available[mode] = info
        except Exception:
            pass
    return available


def _run_variant_backtest(
    mode: str,
    variant_info: dict,
    features_by_ticker: dict[str, pd.DataFrame],
    price_matrix: pd.DataFrame,
    ohlcv_by_ticker: dict,
    spy_prices: pd.Series | None,
    sector_map: dict[str, str],
    trading_dates: list[str],
    config: dict,
    predictor_path: str,
    top_n: int,
    min_score: float,
) -> dict:
    """Download a model variant, run inference + signal gen + simulation."""
    bucket = config.get("signals_bucket", "alpha-engine-research")

    # Download model to temp file
    model_path = _download_variant_model(bucket, variant_info)
    try:
        # Run inference with this variant
        predictions_by_date = _run_variant_inference(
            mode, variant_info, model_path, features_by_ticker,
            predictor_path, trading_dates,
        )

        # Generate signals from predictions
        from synthetic.predictor_backtest import build_signals_by_date
        signals_by_date = build_signals_by_date(
            predictions_by_date, sector_map, ohlcv_by_ticker,
            top_n=top_n, min_score=min_score,
        )

        # Run simulation
        from backtest import _run_simulation_loop
        executor_path = _resolve_executor_path(config)
        if not executor_path:
            return {"status": "error", "error": "no executor path"}
        if executor_path not in sys.path:
            sys.path.insert(0, executor_path)
        from executor.main import run as executor_run
        from executor.ibkr import SimulatedIBKRClient

        stats = _run_simulation_loop(
            executor_run, SimulatedIBKRClient,
            dates=[],
            price_matrix=price_matrix,
            config=config,
            ohlcv_by_ticker=ohlcv_by_ticker,
            signals_by_date=signals_by_date,
            spy_prices=spy_prices,
        )
        return stats
    finally:
        _cleanup_model_files(model_path)


def _download_variant_model(bucket: str, variant_info: dict) -> str:
    """Download a model variant from S3 to a temp file."""
    s3 = boto3.client("s3")
    suffix = ".cbm" if variant_info["scorer_cls"] == "CatBoostScorer" else ".txt"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    s3.download_file(bucket, variant_info["weights_key"], tmp.name)

    meta_path = tmp.name + ".meta.json"
    try:
        s3.download_file(bucket, variant_info["meta_key"], meta_path)
    except Exception:
        pass

    return tmp.name


def _run_variant_inference(
    mode: str,
    variant_info: dict,
    model_path: str,
    features_by_ticker: dict[str, pd.DataFrame],
    predictor_path: str,
    trading_dates: list[str],
) -> dict[str, dict[str, float]]:
    """Run inference using a specific model variant.

    Shares the vectorized tensor builder with ``run_inference``. The
    pre-refactor path allocated a 911-entry dict-of-dicts with ~2.28M
    inner-loop ticks per inference pass; the tensor path does a single
    O(n_tickers) build + O(n_dates) vectorized per-date slice.
    """
    if predictor_path not in sys.path:
        sys.path.insert(0, predictor_path)
    from config import GBM_FEATURES

    if variant_info["scorer_cls"] == "CatBoostScorer":
        from model.catboost_scorer import CatBoostScorer
        scorer = CatBoostScorer.load(model_path)
    else:
        from model.gbm_scorer import GBMScorer
        scorer = GBMScorer.load(model_path)

    from synthetic.predictor_backtest import (
        build_inference_tensor,
        _predict_from_tensor,
    )

    log.info(
        "Variant [%s] inference: starting across %d dates × %d tickers",
        mode, len(trading_dates), len(features_by_ticker),
    )
    t0 = time.monotonic()
    tensor, tickers, date_to_idx = build_inference_tensor(
        features_by_ticker, GBM_FEATURES,
    )
    log.info(
        "Variant [%s] inference tensor: shape=%s usable_tickers=%d (%.1fs)",
        mode, tensor.shape, len(tickers), time.monotonic() - t0,
    )

    predictions_by_date = _predict_from_tensor(
        tensor, tickers, date_to_idx, trading_dates,
        scorer=scorer,
        heartbeat_every=_VARIANT_INFERENCE_HEARTBEAT,
        log_label=f"Variant [{mode}]",
    )

    log.info(
        "Variant [%s] inference: %d dates with predictions (%.1fs total)",
        mode, len(predictions_by_date), time.monotonic() - t0,
    )
    return predictions_by_date


def _pick_best_mode(
    baseline_stats: dict,
    variant_results: dict[str, dict],
    has_sufficient_data: bool,
) -> dict:
    """Compare variants to baseline and recommend best mode."""
    baseline_sharpe = baseline_stats.get("sharpe_ratio", 0) or 0

    best_mode = "promoted"
    best_sharpe = baseline_sharpe
    recommendation_reason = "baseline is best"

    for mode, stats in variant_results.items():
        if "sharpe_ratio" not in stats:
            continue
        variant_sharpe = stats.get("sharpe_ratio", 0) or 0
        if variant_sharpe > best_sharpe:
            best_sharpe = variant_sharpe
            best_mode = mode

    # Apply guardrails
    if best_mode != "promoted":
        improvement = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0
        if improvement < _MIN_SHARPE_IMPROVEMENT:
            return {
                "recommended_mode": None,
                "recommendation_reason": (
                    f"{best_mode} Sharpe ({best_sharpe:.3f}) is only "
                    f"{improvement:.1%} better than baseline ({baseline_sharpe:.3f}) "
                    f"— below {_MIN_SHARPE_IMPROVEMENT:.0%} threshold"
                ),
            }
        if not has_sufficient_data:
            return {
                "recommended_mode": None,
                "recommendation_reason": (
                    f"{best_mode} looks better but insufficient data "
                    f"(need {_MIN_YEARS_DATA}+ years for mode recommendation)"
                ),
            }
        recommendation_reason = (
            f"{best_mode} Sharpe ({best_sharpe:.3f}) is {improvement:.1%} "
            f"better than baseline ({baseline_sharpe:.3f})"
        )
        return {
            "recommended_mode": best_mode,
            "recommendation_reason": recommendation_reason,
        }

    return {
        "recommended_mode": None,
        "recommendation_reason": "baseline (promoted) model is best or tied",
    }


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4b: Signal Threshold Sweep
# ═════════════════════════════════════════════════════════════════════════════

# Default alpha cutoffs to sweep. Tickers with predicted alpha below the
# threshold are excluded from signal generation entirely.
_DEFAULT_ALPHA_THRESHOLDS = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03]
_MIN_THRESHOLD_SHARPE_IMPROVEMENT = 0.05  # 5% relative improvement to change


def evaluate_signal_thresholds(
    predictions_by_date: dict[str, dict[str, float]],
    sector_map: dict[str, str],
    ohlcv_by_ticker: dict,
    price_matrix: pd.DataFrame,
    spy_prices: pd.Series | None,
    trading_dates: list[str],
    config: dict,
    baseline_stats: dict,
) -> dict:
    """
    Sweep alpha cutoffs on the GBM's continuous output (no retraining).

    For each threshold, filters predictions to only include tickers with
    predicted alpha >= threshold, re-generates signals, and re-simulates
    the portfolio. Recommends the threshold that produces the best Sharpe.

    Parameters
    ----------
    predictions_by_date : {date: {ticker: alpha}} from GBM inference
    sector_map : {ticker: sector_etf}
    ohlcv_by_ticker : {ticker: [{date, open, high, low, close}]}
    price_matrix : date x ticker close prices
    spy_prices : SPY close series
    trading_dates : ordered date strings
    config : full backtest config
    baseline_stats : portfolio stats from the baseline run (threshold=0)

    Returns
    -------
    dict with per-threshold results and recommendation
    """
    pb_config = config.get("predictor_backtest", {})
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)
    thresholds = pb_config.get("alpha_thresholds", _DEFAULT_ALPHA_THRESHOLDS)

    executor_path = _resolve_executor_path(config)
    if not executor_path:
        return {"status": "skipped", "reason": "no_executor_path"}
    if executor_path not in sys.path:
        sys.path.insert(0, executor_path)

    from executor.main import run as executor_run
    from executor.ibkr import SimulatedIBKRClient
    from synthetic.predictor_backtest import build_signals_by_date
    from backtest import _run_simulation_loop

    baseline_sharpe = baseline_stats.get("sharpe_ratio", 0)
    if not baseline_sharpe or np.isnan(baseline_sharpe):
        return {"status": "skipped", "reason": "no_baseline_sharpe"}

    threshold_results = []

    # Each threshold gets its own PHASE_START/END marker so a stall in
    # a specific threshold's simulate call is attributable to the value.
    for threshold in thresholds:
        try:
            log.info("Signal threshold %.3f: starting sweep iteration", threshold)
            t0 = time.monotonic()
            with phase("phase4b_threshold", threshold=f"{threshold:.3f}"):
                # Filter predictions: only keep tickers with alpha >= threshold
                filtered_predictions = _filter_predictions_by_alpha(
                    predictions_by_date, threshold,
                )

                # Count how many ENTER-eligible predictions survive
                total_above = sum(len(d) for d in filtered_predictions.values())
                if total_above == 0:
                    log.info("Signal threshold %.3f: 0 predictions survive — skipping", threshold)
                    threshold_results.append({
                        "threshold": threshold,
                        "status": "no_predictions",
                    })
                    continue

                # Generate signals from filtered predictions
                signals_by_date = build_signals_by_date(
                    filtered_predictions, sector_map, ohlcv_by_ticker,
                    top_n=top_n, min_score=min_score,
                )

                # Simulate
                log.info(
                    "Signal threshold %.3f: starting simulation (%d signal dates)",
                    threshold, len(signals_by_date),
                )
                stats = _run_simulation_loop(
                    executor_run, SimulatedIBKRClient,
                    dates=[],
                    price_matrix=price_matrix,
                    config=config,
                    ohlcv_by_ticker=ohlcv_by_ticker,
                    signals_by_date=signals_by_date,
                    spy_prices=spy_prices,
                )

            sharpe = stats.get("sharpe_ratio", 0) or 0
            alpha = stats.get("total_alpha", 0) or 0
            trades = stats.get("total_trades", 0)

            threshold_results.append({
                "threshold": threshold,
                "sharpe_ratio": round(sharpe, 4),
                "total_alpha": round(alpha, 4),
                "max_drawdown": round(stats.get("max_drawdown", 0) or 0, 4),
                "total_trades": trades,
                "predictions_per_day": round(total_above / max(len(filtered_predictions), 1), 1),
            })

            log.info(
                "Signal threshold %.3f: Sharpe=%.3f  alpha=%.3f  trades=%d  (%.1fs)",
                threshold, sharpe, alpha, trades, time.monotonic() - t0,
            )

        except Exception as exc:
            log.warning("Signal threshold %.3f: failed — %s", threshold, exc)
            threshold_results.append({
                "threshold": threshold,
                "status": "error",
                "error": str(exc),
            })

    # Pick the best threshold
    valid_results = [r for r in threshold_results if "sharpe_ratio" in r]
    if not valid_results:
        return {"status": "skipped", "reason": "no_valid_threshold_results"}

    best = max(valid_results, key=lambda r: r["sharpe_ratio"])
    best_threshold = best["threshold"]
    best_sharpe = best["sharpe_ratio"]

    # Apply guardrail: only recommend change if meaningful improvement
    recommended_threshold = None
    reason = "no improvement over baseline"

    if best_sharpe > baseline_sharpe:
        improvement = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0
        if improvement >= _MIN_THRESHOLD_SHARPE_IMPROVEMENT:
            recommended_threshold = best_threshold
            reason = (
                f"threshold {best_threshold:.3f} Sharpe ({best_sharpe:.3f}) is "
                f"{improvement:.1%} better than baseline ({baseline_sharpe:.3f})"
            )
        else:
            reason = (
                f"best threshold {best_threshold:.3f} Sharpe ({best_sharpe:.3f}) is only "
                f"{improvement:.1%} better — below {_MIN_THRESHOLD_SHARPE_IMPROVEMENT:.0%} minimum"
            )

    result = {
        "date": str(date.today()),
        "thresholds_tested": threshold_results,
        "baseline_sharpe": round(baseline_sharpe, 4),
        "best_threshold": best_threshold,
        "best_sharpe": round(best_sharpe, 4),
        "recommended_signal_threshold": recommended_threshold,
        "recommendation_reason": reason,
    }

    log.info(
        "Signal threshold sweep: best=%.3f (Sharpe=%.3f)  recommend=%s",
        best_threshold, best_sharpe,
        recommended_threshold if recommended_threshold is not None else "no change",
    )

    return result


def _filter_predictions_by_alpha(
    predictions_by_date: dict[str, dict[str, float]],
    min_alpha: float,
) -> dict[str, dict[str, float]]:
    """Filter predictions to only include tickers with alpha >= min_alpha."""
    if min_alpha <= 0:
        return predictions_by_date  # no filtering needed

    filtered = {}
    for date_str, preds in predictions_by_date.items():
        above = {t: a for t, a in preds.items() if a >= min_alpha}
        if above:
            filtered[date_str] = above
    return filtered


# ═════════════════════════════════════════════════════════════════════════════
# Phase 4c: Feature Pruning Recommendations
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_feature_pruning(
    features_by_ticker: dict[str, pd.DataFrame],
    price_matrix: pd.DataFrame,
    ohlcv_by_ticker: dict,
    spy_prices: pd.Series | None,
    sector_map: dict[str, str],
    trading_dates: list[str],
    config: dict,
    baseline_stats: dict,
) -> dict:
    """
    Test whether removing noise features improves or preserves Sharpe.

    Reads noise_candidates from training_summary_latest.json, retrains is NOT
    needed — we simply zero out the noise features in the feature arrays before
    inference (functionally equivalent to removing them from a tree model since
    zero-variance features contribute nothing to splits).

    For a proper pruned-model evaluation, the predictor would need to retrain
    without those features. This approximation is directionally correct for
    tree-based models where unused features don't degrade performance.
    """
    bucket = config.get("signals_bucket", "alpha-engine-research")
    predictor_path = _resolve_predictor_path(config)
    if not predictor_path:
        return {"status": "skipped", "reason": "no_predictor_path"}

    # Load noise candidates from training summary
    noise_candidates = _load_noise_candidates(bucket)
    if not noise_candidates:
        log.info("Feature pruning: no noise candidates in training summary — skipping")
        return {"status": "skipped", "reason": "no_noise_candidates"}

    baseline_sharpe = baseline_stats.get("sharpe_ratio", 0)
    if not baseline_sharpe or np.isnan(baseline_sharpe):
        return {"status": "skipped", "reason": "no_baseline_sharpe"}

    log.info("Feature pruning: testing removal of %d noise features: %s", len(noise_candidates), noise_candidates)

    pb_config = config.get("predictor_backtest", {})
    top_n = pb_config.get("top_n_signals_per_day", 20)
    min_score = pb_config.get("min_score", 70)

    # Download the current promoted model
    from synthetic.predictor_backtest import download_gbm_model
    model_path = download_gbm_model(bucket=bucket)

    try:
        # Run inference with noise features zeroed in the shared
        # inference tensor (no per-ticker DataFrame.copy() — previously
        # allocated ~1.1 GB of transient DataFrame copies on the full
        # 911-ticker universe, a legitimate OOM contributor on c5.large
        # when Phase 4c runs while features_by_ticker is still resident).
        from synthetic.predictor_backtest import run_inference, build_signals_by_date
        log.info(
            "Feature pruning: starting inference (%d tickers × %d dates, "
            "zeroing %d noise features)",
            len(features_by_ticker), len(trading_dates), len(noise_candidates),
        )
        t0 = time.monotonic()
        predictions_by_date = run_inference(
            features_by_ticker, model_path, predictor_path, trading_dates,
            zero_features=noise_candidates,
        )
        log.info("Feature pruning: inference complete (%.1fs)", time.monotonic() - t0)

        log.info("Feature pruning: building signals from predictions")
        t0 = time.monotonic()
        signals_by_date = build_signals_by_date(
            predictions_by_date, sector_map, ohlcv_by_ticker,
            top_n=top_n, min_score=min_score,
        )
        log.info(
            "Feature pruning: signal build complete (%d dates, %.1fs)",
            len(signals_by_date), time.monotonic() - t0,
        )

        # Run simulation
        from backtest import _run_simulation_loop
        executor_path = _resolve_executor_path(config)
        if not executor_path:
            return {"status": "skipped", "reason": "no_executor_path"}
        if executor_path not in sys.path:
            sys.path.insert(0, executor_path)
        from executor.main import run as executor_run
        from executor.ibkr import SimulatedIBKRClient

        log.info("Feature pruning: starting simulation")
        t0 = time.monotonic()
        with phase("phase4c_pruned_simulation"):
            pruned_stats = _run_simulation_loop(
                executor_run, SimulatedIBKRClient,
                dates=[],
                price_matrix=price_matrix,
                config=config,
                ohlcv_by_ticker=ohlcv_by_ticker,
                signals_by_date=signals_by_date,
                spy_prices=spy_prices,
            )
        log.info("Feature pruning: simulation complete (%.1fs)", time.monotonic() - t0)
    finally:
        _cleanup_model_files(model_path)

    pruned_sharpe = pruned_stats.get("sharpe_ratio", 0) or 0

    # Recommend pruning if Sharpe holds steady or improves
    sharpe_change = (pruned_sharpe - baseline_sharpe) / abs(baseline_sharpe) if baseline_sharpe != 0 else 0
    recommend_pruning = sharpe_change >= -0.05  # allow up to 5% Sharpe degradation

    result = {
        "date": str(date.today()),
        "noise_candidates": noise_candidates,
        "baseline_sharpe": round(baseline_sharpe, 4),
        "pruned_sharpe": round(pruned_sharpe, 4),
        "sharpe_change_pct": round(sharpe_change * 100, 1),
        "baseline_alpha": round(baseline_stats.get("total_alpha", 0) or 0, 4),
        "pruned_alpha": round(pruned_stats.get("total_alpha", 0) or 0, 4),
        "recommend_pruning": recommend_pruning,
        "prune_features": noise_candidates if recommend_pruning else [],
        "recommendation_reason": (
            f"Pruning {len(noise_candidates)} features: Sharpe {sharpe_change:+.1%} "
            f"({baseline_sharpe:.3f} → {pruned_sharpe:.3f})"
        ),
    }

    log.info(
        "Feature pruning: %s — Sharpe %s%.1f%% (%s)",
        "RECOMMEND" if recommend_pruning else "SKIP",
        "+" if sharpe_change >= 0 else "", sharpe_change * 100,
        ", ".join(noise_candidates),
    )

    return result


def _load_noise_candidates(bucket: str) -> list[str]:
    """Load noise_candidates from the most recent training summary."""
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key="predictor/metrics/training_summary_latest.json")
        summary = json.loads(obj["Body"].read())
        candidates = summary.get("noise_candidates", [])
        return candidates if isinstance(candidates, list) else []
    except Exception as exc:
        log.debug("Failed to load noise candidates: %s", exc)
        return []


# Prior-art note: `_zero_out_features` was removed on 2026-04-24.
# It deep-copied the 911-ticker feature dict (~1.1 GB transient)
# just to zero a handful of columns. The equivalent semantics are
# now expressed via run_inference(zero_features=...), which applies
# the zero on the shared inference tensor in-place. See PR that
# removed it for parity-test coverage.


# ═════════════════════════════════════════════════════════════════════════════
# S3 Params Writer (merges into existing predictor_params.json)
# ═════════════════════════════════════════════════════════════════════════════

def apply_recommendations(
    ensemble_result: dict | None,
    pruning_result: dict | None,
    bucket: str,
    threshold_result: dict | None = None,
) -> dict:
    """
    Merge Phase 4 recommendations into config/predictor_params.json.

    Preserves existing fields (veto_confidence, etc.) and adds/updates:
      - preferred_ensemble_mode (4a)
      - recommended_signal_threshold (4b)
      - prune_features (4c)
    """
    updates = {}

    if ensemble_result and ensemble_result.get("recommended_mode"):
        updates["preferred_ensemble_mode"] = ensemble_result["recommended_mode"]
        updates["ensemble_eval_date"] = ensemble_result.get("date", str(date.today()))
        updates["ensemble_eval_reason"] = ensemble_result.get("recommendation_reason", "")

    if threshold_result and threshold_result.get("recommended_signal_threshold") is not None:
        updates["recommended_signal_threshold"] = threshold_result["recommended_signal_threshold"]
        updates["signal_threshold_eval_date"] = threshold_result.get("date", str(date.today()))
        updates["signal_threshold_eval_reason"] = threshold_result.get("recommendation_reason", "")

    if pruning_result and pruning_result.get("recommend_pruning"):
        updates["prune_features"] = pruning_result.get("prune_features", [])
        updates["pruning_eval_date"] = pruning_result.get("date", str(date.today()))
        updates["pruning_eval_reason"] = pruning_result.get("recommendation_reason", "")

    if not updates:
        log.info("Predictor optimizer: no recommendations to apply")
        return {"applied": False, "reason": "no_recommendations"}

    # Read existing params, merge updates
    s3 = boto3.client("s3")
    existing = {}
    try:
        obj = s3.get_object(Bucket=bucket, Key="config/predictor_params.json")
        existing = json.loads(obj["Body"].read())
    except Exception:
        pass

    # Archive before update
    try:
        from optimizer.rollback import save_previous
        save_previous(bucket, "predictor_params")
    except Exception as exc:
        log.debug("Rollback archive failed (non-fatal): %s", exc)

    merged = {**existing, **updates}

    try:
        body = json.dumps(merged, indent=2, default=str)
        s3.put_object(
            Bucket=bucket,
            Key="config/predictor_params.json",
            Body=body,
            ContentType="application/json",
        )
        log.info("Predictor params updated with Phase 4 recommendations: %s", list(updates.keys()))

        # Archive
        history_key = f"config/predictor_params_history/{date.today().isoformat()}.json"
        s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
    except Exception as exc:
        log.error("Failed to write predictor params to S3: %s", exc)
        return {"applied": False, "reason": f"S3 write failed: {exc}"}

    return {"applied": True, "updates": list(updates.keys()), "merged_keys": list(merged.keys())}


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_predictor_path(config: dict) -> str | None:
    paths = config.get("predictor_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    return next((p for p in paths if os.path.isdir(p)), None)


def _resolve_executor_path(config: dict) -> str | None:
    paths = config.get("executor_paths", [])
    if isinstance(paths, str):
        paths = [paths]
    return next((p for p in paths if os.path.isdir(p)), None)


def _cleanup_model_files(model_path: str) -> None:
    """Remove temp model and metadata files."""
    for path in (model_path, model_path + ".meta.json"):
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError:
            pass
