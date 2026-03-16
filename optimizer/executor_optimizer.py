"""
executor_optimizer.py — recommend optimal executor parameters from param sweep results.

Reads the param sweep DataFrame (grid search over risk + strategy params),
extracts the best-performing combination by Sharpe ratio, and writes
recommendations to S3 for the Executor Lambda to read on cold-start.

Only safe-to-tune params are recommended; drawdown circuit breaker and
sector/equity limits are excluded from auto-tuning.
"""

import json
import logging
from datetime import date

import boto3
import pandas as pd
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


def _safe_float(v) -> float | None:
    """Convert to float, returning None for NaN/None."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), 4)

S3_PARAMS_KEY = "config/executor_params.json"

# Params safe to auto-tune via sweep results.
# Excluded: drawdown_circuit_breaker, max_sector_pct, max_equity_pct (too dangerous)
SAFE_PARAMS = [
    "atr_multiplier",
    "time_decay_reduce_days",
    "time_decay_exit_days",
    "min_score",
    "max_position_pct",
]

# ── Fallback defaults (override via executor_optimizer section in config.yaml) ──
_MIN_VALID_COMBOS = 5
_MIN_SHARPE_IMPROVEMENT = 0.10

# Module-level config ref — set by init_config() from backtest.py
_cfg: dict = {}


def init_config(config: dict) -> None:
    """Load executor_optimizer section from backtester config."""
    global _cfg
    _cfg = config.get("executor_optimizer", {})


def recommend(sweep_df: pd.DataFrame, base_config: dict) -> dict:
    """
    Extract best executor params from param sweep results.

    Args:
        sweep_df: DataFrame from param_sweep.sweep(), sorted by sharpe_ratio desc.
        base_config: Base config dict (used for current/baseline values).

    Returns:
        {
            "status": "ok" | "insufficient_data" | "no_improvement",
            "baseline_params": {...},
            "recommended_params": {...},
            "baseline_sharpe": float,
            "best_sharpe": float,
            "improvement_pct": float,
        }
    """
    if sweep_df is None or sweep_df.empty:
        return {"status": "insufficient_data", "note": "No sweep results available"}

    min_combos = _cfg.get("min_valid_combos", _MIN_VALID_COMBOS)
    valid = sweep_df[sweep_df["sharpe_ratio"].notna()].copy()
    if len(valid) < min_combos:
        return {
            "status": "insufficient_data",
            "n_valid": len(valid),
            "min_required": min_combos,
            "note": f"Only {len(valid)} valid combos (need {min_combos})",
        }

    # Identify param columns (everything that's not a stat column)
    stat_cols = {
        "total_return", "total_alpha", "spy_return", "sharpe_ratio",
        "max_drawdown", "calmar_ratio", "total_trades", "win_rate",
        "error", "status", "dates_simulated", "total_orders", "note",
    }
    param_cols = [c for c in valid.columns if c in SAFE_PARAMS]

    if not param_cols:
        return {"status": "no_params", "note": "No safe params found in sweep results"}

    # Re-sort by sharpe_ratio (sweep may be sorted by total_alpha for display)
    valid = valid.sort_values("sharpe_ratio", ascending=False)

    # Baseline = worst combo by sharpe (conservative); best = top combo
    baseline_sharpe = valid.iloc[-1]["sharpe_ratio"]
    best_row = valid.iloc[0]
    best_sharpe = best_row["sharpe_ratio"]

    # Alpha metrics (informational — not used for gating)
    best_alpha = _safe_float(best_row.get("total_alpha"))
    baseline_alpha = _safe_float(valid.iloc[-1].get("total_alpha"))

    if baseline_sharpe == 0:
        improvement_pct = float("inf") if best_sharpe > 0 else 0.0
    else:
        improvement_pct = (best_sharpe - baseline_sharpe) / abs(baseline_sharpe)

    recommended = {col: best_row[col] for col in param_cols if pd.notna(best_row[col])}
    # Convert numpy types to native Python
    recommended = {k: float(v) if isinstance(v, (int, float)) else v for k, v in recommended.items()}

    baseline = {col: valid.iloc[-1][col] for col in param_cols if pd.notna(valid.iloc[-1][col])}
    baseline = {k: float(v) if isinstance(v, (int, float)) else v for k, v in baseline.items()}

    min_improvement = _cfg.get("min_sharpe_improvement", _MIN_SHARPE_IMPROVEMENT)
    if improvement_pct < min_improvement:
        return {
            "status": "no_improvement",
            "baseline_params": baseline,
            "recommended_params": recommended,
            "baseline_sharpe": round(float(baseline_sharpe), 4),
            "best_sharpe": round(float(best_sharpe), 4),
            "best_alpha": best_alpha,
            "baseline_alpha": baseline_alpha,
            "improvement_pct": round(improvement_pct, 4),
            "note": (
                f"Best Sharpe ({best_sharpe:.4f}) only {improvement_pct:.1%} better than "
                f"baseline ({baseline_sharpe:.4f}). Need {min_improvement:.0%}+ to recommend."
            ),
        }

    return {
        "status": "ok",
        "baseline_params": baseline,
        "recommended_params": recommended,
        "baseline_sharpe": round(float(baseline_sharpe), 4),
        "best_sharpe": round(float(best_sharpe), 4),
        "best_alpha": best_alpha,
        "baseline_alpha": baseline_alpha,
        "improvement_pct": round(improvement_pct, 4),
        "n_combos_tested": len(valid),
        "note": (
            f"Best combo improves Sharpe by {improvement_pct:.1%} "
            f"({baseline_sharpe:.4f} → {best_sharpe:.4f}) across {len(valid)} combos."
        ),
    }


def apply(result: dict, bucket: str) -> dict:
    """
    Write recommended executor params to S3 if recommendation is valid.

    Writes to s3://{bucket}/config/executor_params.json and archives
    to config/executor_params_history/{date}.json.

    Args:
        result: dict from recommend().
        bucket: S3 bucket name.

    Returns:
        {"applied": True, ...} or {"applied": False, "reason": ...}
    """
    if result.get("status") != "ok":
        return {"applied": False, "reason": f"status={result.get('status')}"}

    recommended = result.get("recommended_params", {})
    if not recommended:
        return {"applied": False, "reason": "no recommended params"}

    payload = {
        **recommended,
        "updated_at": str(date.today()),
        "best_sharpe": result.get("best_sharpe"),
        "best_alpha": result.get("best_alpha"),
        "improvement_pct": result.get("improvement_pct"),
        "n_combos_tested": result.get("n_combos_tested"),
    }

    s3 = boto3.client("s3")
    body = json.dumps(payload, indent=2)

    s3.put_object(Bucket=bucket, Key=S3_PARAMS_KEY, Body=body, ContentType="application/json")
    logger.info("Executor params updated in S3: %s", recommended)

    history_key = f"config/executor_params_history/{date.today().isoformat()}.json"
    s3.put_object(Bucket=bucket, Key=history_key, Body=body, ContentType="application/json")
    logger.info("Executor params archived to s3://%s/%s", bucket, history_key)

    return {
        "applied": True,
        "params": recommended,
        "best_sharpe": result.get("best_sharpe"),
        "best_alpha": result.get("best_alpha"),
        "improvement_pct": result.get("improvement_pct"),
    }
