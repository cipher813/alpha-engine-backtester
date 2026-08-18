"""
sizing_ab.py — Position sizing A/B test: current sizing vs equal-weight.

Runs two portfolio simulations with identical signal history but different
sizing configs. Compares Sharpe, return, and alpha to determine if position
sizing adjustments earn their complexity.

This is analysis-only — no S3 config writes. Results go into the weekly report.
Min-data gate: requires >= 50 trades in simulation.
"""

import logging
from copy import deepcopy

from analysis.pit_parity import _config_without_runtime_handles

logger = logging.getLogger(__name__)

_MIN_TRADES = 50


def run_sizing_ab(
    sim_fn,
    base_config: dict,
    min_trades: int = _MIN_TRADES,
) -> dict:
    """
    Run A/B comparison: current sizing vs equal-weight sizing.

    Args:
        sim_fn: callable that takes a config dict and returns portfolio_stats dict
        base_config: the current production config
        min_trades: minimum trades for valid comparison

    Returns:
        dict with comparison results.
    """
    # ``base_config`` is the live ``_run_simulation_pipeline`` config, which
    # carries the ``_phase_registry`` runtime handle (PhaseRegistry.s3_client
    # is a botocore S3Client with circular service-model references that
    # blow the recursion stack under ``copy.deepcopy`` — same failure class
    # as ``backtest.py::_build_merged_simulate_config`` (caught 2026-04-27)
    # and ``analysis/pit_parity.py`` (re-bit 2026-05-17..24). Strip it before
    # copying; sizing_ab needs no runtime handles, only data keys, so this is
    # behaviour-neutral (config-I7209 follow-up).
    safe_config = _config_without_runtime_handles(base_config)

    # Config A: current (production) sizing
    config_a = deepcopy(safe_config)

    # Config B: equal-weight (disable all sizing adjustments)
    config_b = deepcopy(safe_config)
    config_b["atr_sizing_enabled"] = False
    # confidence_sizing_enabled dropped 2026-08-17 (alpha-engine-config-I7525):
    # the executor's confidence factor is retired, so the flag no longer
    # disables anything and setting it here would imply a knob that is gone.
    config_b["staleness_discount_enabled"] = False
    config_b["earnings_sizing_enabled"] = False
    # Keep sector_adj and drawdown — those are risk management, not sizing
    config_b["sector_adj"] = {
        "overweight": 1.0,
        "market_weight": 1.0,
        "underweight": 1.0,
    }
    config_b["conviction_decline_adj"] = 1.0
    config_b["upside_fail_adj"] = 1.0

    try:
        logger.info("Running sizing A/B: current sizing (A) vs equal-weight (B)")
        stats_a = sim_fn(config_a)
        stats_b = sim_fn(config_b)
    except Exception as e:
        return {"status": "error", "error": str(e)}

    if not stats_a or not stats_b:
        return {"status": "error", "error": "One or both simulations returned empty results"}

    trades_a = stats_a.get("total_trades", 0)
    trades_b = stats_b.get("total_trades", 0)

    if trades_a < min_trades or trades_b < min_trades:
        return {
            "status": "insufficient_data",
            "trades_a": trades_a,
            "trades_b": trades_b,
            "min_required": min_trades,
        }

    sharpe_a = stats_a.get("sharpe_ratio", 0)
    sharpe_b = stats_b.get("sharpe_ratio", 0)
    return_a = stats_a.get("total_return", 0)
    return_b = stats_b.get("total_return", 0)
    alpha_a = stats_a.get("total_alpha")
    alpha_b = stats_b.get("total_alpha")
    dd_a = stats_a.get("max_drawdown", 0)
    dd_b = stats_b.get("max_drawdown", 0)

    sharpe_diff = sharpe_a - sharpe_b if sharpe_a and sharpe_b else None
    return_diff = return_a - return_b if return_a is not None and return_b is not None else None
    alpha_diff = (alpha_a - alpha_b) if alpha_a is not None and alpha_b is not None else None

    # Assessment
    if sharpe_diff is not None and sharpe_diff > 0.1:
        assessment = "sizing_helps"
        detail = f"Current sizing Sharpe {sharpe_a:.2f} vs equal-weight {sharpe_b:.2f} (+{sharpe_diff:.2f})"
    elif sharpe_diff is not None and sharpe_diff < -0.1:
        assessment = "equal_weight_better"
        detail = f"Equal-weight Sharpe {sharpe_b:.2f} vs current {sharpe_a:.2f} (+{-sharpe_diff:.2f})"
    else:
        assessment = "no_difference"
        detail = f"Sizing has minimal impact (Sharpe diff={sharpe_diff:.3f})" if sharpe_diff else "Unable to compare"

    return {
        "status": "ok",
        "current_sizing": {
            "sharpe": round(sharpe_a, 3) if sharpe_a else None,
            "total_return": round(return_a, 4) if return_a is not None else None,
            "total_alpha": round(alpha_a, 4) if alpha_a is not None else None,
            "max_drawdown": round(dd_a, 4) if dd_a else None,
            "total_trades": trades_a,
        },
        "equal_weight": {
            "sharpe": round(sharpe_b, 3) if sharpe_b else None,
            "total_return": round(return_b, 4) if return_b is not None else None,
            "total_alpha": round(alpha_b, 4) if alpha_b is not None else None,
            "max_drawdown": round(dd_b, 4) if dd_b else None,
            "total_trades": trades_b,
        },
        "sharpe_diff": round(sharpe_diff, 3) if sharpe_diff is not None else None,
        "return_diff": round(return_diff, 4) if return_diff is not None else None,
        "alpha_diff": round(alpha_diff, 4) if alpha_diff is not None else None,
        "assessment": assessment,
        "detail": detail,
    }
