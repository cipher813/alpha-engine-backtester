"""
param_sweep.py — grid search over risk.yaml parameters using Mode 2 simulation.

Runs executor.main.run(simulate=True) for each parameter combination across all
historical signal dates and compares portfolio outcomes.

Data availability: requires Mode 2 (executor simulate= mode, Phase 0 complete)
and at least 20 trading days of signal history. Deferred to Phase 5 (Week 8+).

Prerequisite: Phase 0b (executor simulate= mode) must be deployed first.
"""

import itertools
import logging
from copy import deepcopy
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_GRID = {
    "min_score": [65, 70, 75, 80],
    "max_position_pct": [0.05, 0.10, 0.15],
    "drawdown_circuit_breaker": [0.10, 0.15, 0.20],
}


def sweep(
    grid: dict,
    run_simulation_fn: Callable[[dict], dict],
    base_config: dict,
) -> pd.DataFrame:
    """
    Grid search over parameter combinations.

    Args:
        grid: Dict mapping param name → list of values to try.
              e.g. {"min_score": [65, 70, 75], "max_position_pct": [0.05, 0.10]}
        run_simulation_fn: Callable that accepts a config dict and returns a
              stats dict (total_return, sharpe_ratio, max_drawdown, ...).
              This wraps backtest.run_simulation() with each param combination.
        base_config: Base config dict; each combination overrides relevant keys.

    Returns:
        DataFrame with one row per parameter combination, sorted by sharpe_ratio.

    NOTE: This function is a scaffold. run_simulation_fn must be implemented by
    the caller once Phase 0b (executor simulate= mode) is deployed and enough
    signal history is available.
    """
    keys = list(grid.keys())
    values = list(grid.values())
    combinations = list(itertools.product(*values))

    logger.info(
        "Running param sweep: %d combinations across %s",
        len(combinations),
        keys,
    )

    rows = []
    for combo in combinations:
        params = dict(zip(keys, combo))
        config = deepcopy(base_config)
        config.update(params)

        logger.debug("Running simulation with params: %s", params)
        try:
            stats = run_simulation_fn(config)
            rows.append({**params, **stats})
        except Exception as e:
            logger.warning("Simulation failed for params %s: %s", params, e)
            rows.append({**params, "error": str(e)})

    df = pd.DataFrame(rows)
    if "sharpe_ratio" in df.columns:
        df.sort_values("sharpe_ratio", ascending=False, inplace=True)

    return df


def best_params(sweep_df: pd.DataFrame, metric: str = "sharpe_ratio") -> dict:
    """
    Return the parameter combination with the best value of `metric`.
    """
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric '{metric}' not found in sweep results")

    best_row = sweep_df.dropna(subset=[metric]).iloc[0]
    param_cols = [c for c in sweep_df.columns if c not in ["total_return", "sharpe_ratio",
                                                             "max_drawdown", "calmar_ratio",
                                                             "total_trades", "win_rate", "error"]]
    return {col: best_row[col] for col in param_cols}
