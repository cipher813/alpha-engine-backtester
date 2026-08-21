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

# Materiality band on ``sharpe_diff`` — the |diff| below which the A/B verdict is
# "no_difference" rather than a direction. Re-derived on the sqrt(252) scale
# (config-I7598, deliverable 3; the one literal config-I7597/PR#694 left behind
# because another change was in flight in this file).
#
# Derivation. config-I7236 (PR#666, 3ad4948, 2026-08-13) moved every Sharpe this
# repo reports from vectorbt's sqrt(365) annualization to
# ``nousergon_lib.quant.riskstats.sharpe_ratio``'s sqrt(252)
# (``vectorbt_bridge.py:360``), so for identical returns
#
#     Sharpe_new = Sharpe_old * sqrt(252/365) = Sharpe_old * 0.8309097
#
# ``sharpe_diff`` is a DIFFERENCE of two Sharpes computed by that same function
# in the same run, so it carries the factor linearly:
#
#     round(0.1 * 0.8309097, 3) = 0.083
#
# (PR#694 quotes the factor as 0.830455, which is sqrt(252/365) mis-evaluated in
# the 4th decimal. Every literal it derived rounds identically under either
# value — 0.05 -> 0.0415, -0.3 -> -0.249, 0.5 -> 0.415 — so no published number
# is affected and none is churned here; the exact factor is used from now on.)
#
# A conversion — not a measurement — is the right answer here for the same
# reason it was in ``analysis/grading.py::_grade_position_sizing``, which grades
# this very number: there is no external convention for "a materially different
# Sharpe delta from a position-sizing scheme". This band was chosen against this
# system's own pre-fix numbers, so it carries the pre-fix scale with it, and
# leaving it at 0.1 silently raises the bar by 20.4% — the same real sizing
# improvement that used to read "sizing_helps" would read "no_difference".
_MIN_MATERIAL_SHARPE_DIFF = 0.083


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
        # config-I7596. This handler used to record NOTHING: no log line, no
        # stack, no exception type — only ``str(e)`` on a dict. That is how
        # ``{"status": "error", "error": "maximum recursion depth exceeded"}``
        # reached `backtest/{date}/sizing_ab.json` on every run from PR#655
        # onward with no traceback anywhere on the box, and why the cause had
        # to be reproduced by hand months later. It also pre-empts the caller's
        # own fail-loud handler (`backtest.py`'s sizing_ab except, which does
        # log ``exc_info=True`` and reports to flow-doctor) by never letting
        # the exception escape.
        #
        # Deviation from the RAISE default, per the fleet fail-loud rule:
        #   (a) swallowed: any failure raised by ``sim_fn`` for either arm;
        #   (b) the backtest's primary deliverables (portfolio_stats, sweep_df,
        #       executor params) are already computed when this runs, and this
        #       output feeds no order, no promotion and no NAV — the ALWAYS-EMIT
        #       artifact contract requires a status here rather than an abort;
        #   (c) recorded on BOTH durable surfaces — the log carries the full
        #       stack, and the returned dict IS the persisted artifact and now
        #       names the exception type as well as its message.
        logger.error("sizing A/B simulation failed: %s", e, exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__,
        }

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

    # Assessment. The +/-_MIN_MATERIAL_SHARPE_DIFF band is on the sqrt(252)
    # Sharpe scale — see the constant's derivation above (config-I7598).
    if sharpe_diff is not None and sharpe_diff > _MIN_MATERIAL_SHARPE_DIFF:
        assessment = "sizing_helps"
        detail = f"Current sizing Sharpe {sharpe_a:.2f} vs equal-weight {sharpe_b:.2f} (+{sharpe_diff:.2f})"
    elif sharpe_diff is not None and sharpe_diff < -_MIN_MATERIAL_SHARPE_DIFF:
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
