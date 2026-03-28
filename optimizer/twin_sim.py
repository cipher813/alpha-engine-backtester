"""
twin_sim.py — A/B twin simulation for parameter promotion transparency.

Runs identical simulation twice (current params vs proposed params) on the same
date range and price matrix. Reports side-by-side metrics so the weekly email
shows exactly what changed and why.
"""

import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

# Metrics to compare in the twin simulation
_COMPARE_METRICS = [
    ("sharpe_ratio", "Sharpe ratio", ".2f"),
    ("total_alpha", "Total alpha", ".1%"),
    ("max_drawdown", "Max drawdown", ".1%"),
    ("win_rate", "Win rate", ".1%"),
    ("total_return", "Total return", ".1%"),
    ("total_trades", "Total trades", "d"),
]


def _extract_stats(raw: dict) -> dict:
    """Extract comparable metrics from simulation output."""
    return {key: raw.get(key) for key, _, _ in _COMPARE_METRICS}


def run_twin_simulation(
    sim_fn,
    current_config: dict,
    proposed_config: dict,
    param_keys: list[str],
) -> dict:
    """
    Run the same simulation with current and proposed configs, compare results.

    Args:
        sim_fn: callable(config_dict) -> stats dict. Runs one full simulation pass.
        current_config: base config with current S3 param values applied.
        proposed_config: base config with proposed param values applied.
        param_keys: list of param names that differ between current and proposed.

    Returns:
        {
            "current_stats": {metric: value, ...},
            "proposed_stats": {metric: value, ...},
            "delta": {metric: proposed - current, ...},
            "proposed_better": bool,  # True if proposed Sharpe > current
            "param_changes": {key: {"before": v1, "after": v2}, ...},
        }
    """
    logger.info("Twin simulation: running current params...")
    try:
        current_raw = sim_fn(current_config)
    except Exception as e:
        logger.error("Twin sim current-params run failed: %s", e)
        return {"status": "error", "error": f"current run failed: {e}"}

    logger.info("Twin simulation: running proposed params...")
    try:
        proposed_raw = sim_fn(proposed_config)
    except Exception as e:
        logger.error("Twin sim proposed-params run failed: %s", e)
        return {"status": "error", "error": f"proposed run failed: {e}"}

    current_stats = _extract_stats(current_raw)
    proposed_stats = _extract_stats(proposed_raw)

    # Compute deltas
    delta = {}
    for key, _, _ in _COMPARE_METRICS:
        c = current_stats.get(key)
        p = proposed_stats.get(key)
        if c is not None and p is not None:
            delta[key] = p - c
        else:
            delta[key] = None

    # Build param change summary
    param_changes = {}
    for key in param_keys:
        param_changes[key] = {
            "before": current_config.get(key),
            "after": proposed_config.get(key),
        }

    current_sharpe = current_stats.get("sharpe_ratio")
    proposed_sharpe = proposed_stats.get("sharpe_ratio")
    proposed_better = (
        proposed_sharpe is not None
        and current_sharpe is not None
        and proposed_sharpe > current_sharpe
    )

    logger.info(
        "Twin simulation complete: current Sharpe=%.3f, proposed Sharpe=%.3f, better=%s",
        current_sharpe or 0, proposed_sharpe or 0, proposed_better,
    )

    return {
        "status": "ok",
        "current_stats": current_stats,
        "proposed_stats": proposed_stats,
        "delta": delta,
        "proposed_better": proposed_better,
        "param_changes": param_changes,
    }
