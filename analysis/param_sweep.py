"""
param_sweep.py — parameter sweep over risk.yaml parameters using Mode 2 simulation.

Supports two modes:
  - grid:   exhaustive search over all parameter combinations (cartesian product)
  - random: randomly sample from the parameter space (Bergstra & Bengio 2012)

Random search with n trials has probability 1 - (1-p)^n of finding a combo in the
top-p fraction. With n=60 trials, there is a 95% chance of finding a top-5% combo
regardless of grid size — making it statistically on par with full grid search for
practical purposes while scaling to arbitrarily large parameter spaces.

Runs executor.main.run(simulate=True) for each parameter combination across all
historical signal dates and compares portfolio outcomes.
"""

from __future__ import annotations

import itertools
import logging
import math
import os
import random
from copy import deepcopy
from typing import Any, Callable

import pandas as pd

import deadline_budget as _budget

logger = logging.getLogger(__name__)

# ── config-I7309: the self-deadlining combo loop ────────────────────────────
# The pit_parity walk-forward pass runs TWO nested loops inside one 2700 s
# subprocess ceiling: the walk-forward FOLD loop (already self-deadlining since
# config#7199) and — when ``pit_parity_sweep`` is set — THIS combo loop, one
# full simulation per combo. Measured on watch-rerun-2026-07-31 (recorded in
# ``backtest/2026-07-31/pit_parity.json``):
#
#     Sweep combo 5/60 done in 272.5s (sweep elapsed 2144.1s)
#
# 60 combos x 272.5 s is ~16,350 s of work under a 2700 s ceiling. The pass was
# SIGKILLed at the wall with five completed combos discarded, three Saturdays
# running. This loop now asks the same question the fold loop asks — does the
# NEXT unit still fit? — and returns the combos it finished, labelled PARTIAL,
# instead of losing all of them.
#
# The budget is the deadline the pass ALREADY carries
# (``_pass_deadline_epoch``, injected by
# ``analysis.pit_parity._run_predictor_pass_isolated``). No new config surface,
# and every non-pit_parity caller — the weekly ``run_param_sweep``, the CLI,
# the tests — hands down no deadline and is completely unchanged.
_SWEEP_BUDGET_TAG = "[param_sweep]"

#: Seconds held back from the deadline for the work that follows the sweep
#: inside the same pass — ``_build_pit_parity_cscv_matrix``'s per-block
#: re-simulations, stats assembly, and the pickle/S3 write. Being killed at the
#: wall discards the whole pass; over-reserving costs at most one combo.
#:
#: **This flat value is the reserve for every caller that is NOT followed by the
#: CSCV matrix build.** For the ``pit_parity_sweep`` path it is only a FLOOR:
#: see :func:`cscv_reserve_s`, which derives the reserve from the work that
#: actually follows. A flat 600 s was a guess, and the arithmetic it has to
#: cover is not flat — it scales with the number of combos the sweep completes.
SWEEP_BUDGET_RESERVE_S = float(os.environ.get("PARAM_SWEEP_RESERVE_S", "600"))

#: Assumed cost of the FIRST combo, before this run has measured anything.
#: Seeded at the measured 2026-07-31 per-combo cost (272.5 s) rounded up — an
#: over-estimate costs one combo of coverage, an under-estimate costs the whole
#: artifact, so the first-unit guess deliberately errs high.
SWEEP_FIRST_COMBO_ESTIMATE_S = float(
    os.environ.get("PARAM_SWEEP_FIRST_COMBO_ESTIMATE_S", "300")
)

# ── config-I7309: the CSCV matrix build is the THIRD loop in the same pass ───
# ``backtest._build_pit_parity_cscv_matrix`` runs immediately after this sweep,
# inside the SAME 2700 s subprocess ceiling, re-simulating the sweep's top-K
# combos on each of ``n_blocks`` chronological blocks — ``n_blocks x top_k``
# simulations, each over roughly ``1/n_blocks`` of the date grid, so the whole
# matrix costs about ``p90(combo) x top_k x n_blocks / n_blocks`` = one full
# sweep of the top-K again.
#
# At the canonical defaults (8 blocks x top-12) with the measured 272.5 s
# per-combo cost that is ~3,270 s of work — MORE than the entire 2700 s pass
# ceiling — held back behind a flat 600 s reserve. So bounding only the combo
# loop moved the SIGKILL from the sweep to the matrix build rather than
# removing it: the pass still had nothing to publish. Both halves are bounded
# here, and the reserve is DERIVED from the matrix build's own arithmetic
# instead of guessed.
#
# These constants are defined once, here, and imported by
# ``backtest._build_pit_parity_cscv_matrix`` so the reserve arithmetic and the
# loop it is reserving for cannot drift apart.

#: Default number of chronological CSCV blocks (rows of the block matrix).
CSCV_DEFAULT_N_BLOCKS = 8

#: Default number of top-Sortino combos re-evaluated on each block (columns).
CSCV_DEFAULT_TOP_K = 12

#: Minimum block ROWS that make a CSCV PBO computable at all (``cscv_pbo``'s
#: ``min_splits``). The sweep reserves enough budget for this many rows and no
#: more: reserving for all ``n_blocks`` would starve the sweep of the combos
#: the matrix has to be built FROM, and rows beyond the minimum are bought out
#: of whatever budget the sweep leaves behind.
CSCV_MIN_BLOCKS = 4

#: Seconds the CSCV matrix loop itself holds back, for stats assembly, the
#: pickle to ``--stats-out`` and the S3 pass-artifact write. Distinct from
#: :data:`SWEEP_BUDGET_RESERVE_S`, which covers the matrix build; this covers
#: what follows the matrix build.
CSCV_BUDGET_RESERVE_S = float(os.environ.get("PIT_PARITY_CSCV_RESERVE_S", "180"))

#: Tag prefixing the CSCV loop's budget log lines, so the three self-deadlining
#: loops inside one pass (``[walk_forward]``, ``[param_sweep]``,
#: ``[cscv_matrix]``) stay distinguishable in one interleaved log.
CSCV_BUDGET_TAG = "[cscv_matrix]"

#: The ``df.attrs`` keys ``_run_combos`` publishes about its own budget. Named
#: once so the producer and every consumer that lifts them into an artifact
#: cannot drift apart — a key added here and forgotten at the consumer is a
#: number that exists only in a log on a self-terminating spot instance.
SWEEP_BUDGET_ATTRS = (
    "n_combos_planned",
    "n_combos_run",
    "n_combos_skipped_for_budget",
    "sweep_budget_stopped",
    "combo_p90_s",
    "cscv_reserve_s",
)


def cscv_reserve_s(
    config: dict,
    combo_seconds: list[float],
    n_combos_done: int,
    *,
    first_combo_estimate_s: float = SWEEP_FIRST_COMBO_ESTIMATE_S,
) -> float:
    """Seconds the combo loop must hold back so a MINIMUM-VIABLE CSCV matrix
    can still be built after it.

    Returns :data:`SWEEP_BUDGET_RESERVE_S` unchanged when no CSCV build
    follows (``pit_parity_sweep`` unset) — every non-pit_parity caller keeps
    the behaviour it had.

    The estimate is ``p90(observed combo) x top_k_effective x CSCV_MIN_BLOCKS /
    n_blocks``, because one block row re-simulates ``top_k_effective`` combos
    over ~``1/n_blocks`` of the date grid. ``top_k_effective`` is
    ``min(n_combos_done + 1, top_k)`` — the matrix can only be built from
    combos the sweep actually finished, so each additional combo raises the
    reserve by a fraction of its own cost. That is self-limiting on purpose:
    the loop stops at the point where one more combo would cost more than the
    validation of the combos already in hand, rather than sweeping a grid it
    cannot then evaluate.
    """
    if not config.get("pit_parity_sweep"):
        return SWEEP_BUDGET_RESERVE_S
    n_blocks = int(config.get("pit_parity_cscv_n_blocks", CSCV_DEFAULT_N_BLOCKS))
    top_k = int(config.get("pit_parity_cscv_top_k", CSCV_DEFAULT_TOP_K))
    if n_blocks <= 0 or top_k <= 0:
        return SWEEP_BUDGET_RESERVE_S
    top_k_effective = max(1, min(n_combos_done + 1, top_k))
    per_combo = _budget.p90(combo_seconds, default=first_combo_estimate_s)
    block_rows = min(CSCV_MIN_BLOCKS, n_blocks)
    matrix_s = per_combo * top_k_effective * block_rows / float(n_blocks)
    return CSCV_BUDGET_RESERVE_S + matrix_s


def _deepcopy_safe_config(base: dict) -> dict:
    """Deepcopy a config dict while excluding keys whose values are not
    deepcopy-safe (boto3 clients, PhaseRegistry, other runtime objects
    with cyclic refs). Underscore-prefixed keys are treated as runtime
    refs by convention and re-attached shallow to the copy.

    Without this, the 2026-04-23 post-filter smoke-param-sweep hit
    `maximum recursion depth exceeded` because `config["_phase_registry"]`
    holds a boto3 S3 client whose internal cyclic refs broke deepcopy.
    """
    serializable: dict[str, Any] = {
        k: v for k, v in base.items() if not k.startswith("_")
    }
    copied = deepcopy(serializable)
    runtime = {k: v for k, v in base.items() if k.startswith("_")}
    copied.update(runtime)
    return copied

# Core 6 parameters — high-frequency, regime-invariant risk/exit rules that
# affect every trade.  60 random trials gives 95% confidence of finding a
# top-5% combination (Bergstra & Bengio).  Grid size: 4×3×4×3×3×4 = 1,728.
#
# Deferred parameters (revisit at 6+ months of live data):
#   reduce_fraction, atr_sizing_target_risk,
#   staleness_decay_per_day, earnings_*, momentum_gate/exit_threshold,
#   correlation_block_threshold, drawdown_circuit_breaker (safety param,
#   never auto-applied).
DEFAULT_GRID = {
    "min_score": [45, 50, 55, 60, 65, 70, 75, 80],
    "max_position_pct": [0.05, 0.10, 0.15],
    "atr_multiplier": [2.0, 2.5, 3.0, 4.0],
    "time_decay_reduce_days": [5, 7, 10],
    "time_decay_exit_days": [10, 15, 20],
    "profit_take_pct": [0.15, 0.20, 0.25, 0.30],
}

# Extended grid for future use — includes low-frequency params.
# Activate by setting param_sweep in config.yaml to this grid.
EXTENDED_GRID = {
    "min_score": [45, 50, 55, 60, 65, 70, 75, 80],
    "max_position_pct": [0.05, 0.10, 0.15],
    "atr_multiplier": [2.0, 2.5, 3.0, 4.0],
    "time_decay_reduce_days": [5, 7, 10],
    "time_decay_exit_days": [10, 15, 20],
    "profit_take_pct": [0.15, 0.20, 0.25, 0.30],
    "reduce_fraction": [0.25, 0.33, 0.50],
    "atr_sizing_target_risk": [0.01, 0.02, 0.03],
    # L300 (2026-06-01) removed confidence_sizing_min/range from this sweep:
    # the sim runs with ``predictions_by_ticker={}``, so prediction_confidence
    # was always None and sweeping them was a SILENT NO-OP that still emitted
    # misleading "tuned" recommendations. As of 2026-08-17 the mechanism itself
    # is retired in the executor (alpha-engine-config-I7525, Brian ruling) — its
    # constants were on the pre-2026-05-12 confidence axis and it sat behind the
    # optimizer cutover — so the params are gone from FACTORY_DEFAULTS and from
    # the executor_params boundary, not merely absent from the grid.
    # Conviction reaches live sizing through the MVO optimizer's predicted_alpha;
    # p_up sizing (use_p_up_sizing) remains the offline-tuned per-name path.
    "staleness_decay_per_day": [0.02, 0.03, 0.05],
    "earnings_sizing_reduction": [0.30, 0.50, 0.70],
    "earnings_proximity_days": [3, 5, 7],
    "momentum_gate_threshold": [-10.0, -5.0, -2.0],
    "correlation_block_threshold": [0.70, 0.75, 0.80, 0.85],
    "momentum_exit_threshold": [-20.0, -15.0, -10.0],
    # L300 / L300-a (2026-06-01): ALL stance-conditioned params REMOVED from the
    # sweep. The backtester sim runs with ``predictions_by_ticker={}`` and
    # ``stance`` is sourced ONLY from predictions (executor/deciders.py:534,
    # ``pred_data_for_veto.get("stance")``) → stance is always None in the sim, so:
    #   • stance_size_{momentum,value,quality,catalyst} (SIZING multipliers) →
    #     stance_adj resolves to 1.0 (L300);
    #   • value_stance_drawdown_min / quality_stance_momentum_threshold (entry
    #     GATE thresholds) → the ``if stance == "value"`` / ``elif stance ==
    #     "quality"`` branches in deciders.py never execute (L300-a, audit
    #     confirmed 2026-06-01).
    # Sweeping any of them was a SILENT NO-OP reading as "tuned." The sizing
    # multipliers are tuned OFFLINE by optimizer/stance_sizing_optimizer.py
    # (realized per-stance rank-IC gate); the gate thresholds need a
    # (stance × momentum)-conditioned offline read — deferred (L300-a follow-up).
    # The executor falls back to its FACTORY_DEFAULTS for all of them (unchanged).
    # See [[feedback_no_silent_fails]].
}

# ── Extended-grid data gate (config#947) ──────────────────────────────────────
# The EXTENDED_GRID adds low-frequency params (reduce_fraction, earnings_*,
# momentum/correlation thresholds, …) whose effect only shows across regimes.
# Sweeping them on a short (~3-month) production window over-fits in-sample:
# there is not enough out-of-sample history to holdout-validate the extra
# degrees of freedom. So the extended grid is GATED on the signal-date window
# spanning at least EXTENDED_GRID_MIN_DAYS of history — the "6+ months of data"
# condition from the issue title made into a real code gate rather than an
# undocumented manual config flip.
EXTENDED_GRID_MIN_DAYS = 183  # ~6 calendar months (config#947 data gate)


def _window_span_days(dates: list[str] | None) -> int:
    """Calendar-day span between the first and last signal date.

    ``dates`` are sorted ``YYYY-MM-DD`` strings (loaders.signal_loader.list_dates).
    Returns 0 if fewer than 2 valid dates are present.
    """
    from datetime import date as _date

    if not dates:
        return 0
    parsed: list[_date] = []
    for d in dates:
        try:
            parsed.append(_date.fromisoformat(str(d)[:10]))
        except (ValueError, TypeError):
            continue
    if len(parsed) < 2:
        return 0
    return (max(parsed) - min(parsed)).days


def select_grid(
    dates: list[str] | None,
    config: dict | None = None,
) -> dict:
    """Choose the parameter grid for a sweep, gating EXTENDED_GRID on data.

    Resolution order (config#947):
      1. An explicit ``config["param_sweep"]`` always wins — operators can pin
         any grid (e.g. force EXTENDED_GRID for an offline study) and that
         override is honored verbatim, unchanged from prior behavior.
      2. Otherwise auto-select: EXTENDED_GRID only when the signal-date window
         spans >= EXTENDED_GRID_MIN_DAYS (~6 months) of history, so the extra
         low-frequency degrees of freedom can be holdout-validated; else the
         conservative DEFAULT_GRID (core 6 params).

    The ``param_sweep_settings.force_extended_grid`` escape hatch (bool) lets an
    operator opt into the extended grid without hand-copying the dict into
    config — but it STILL respects the data gate (falls back to DEFAULT_GRID and
    warns if <6 months), and only applies when no explicit ``param_sweep`` grid
    is set.
    """
    config = config or {}
    explicit = config.get("param_sweep")
    if explicit:
        return explicit

    settings = config.get("param_sweep_settings", {}) or {}
    span = _window_span_days(dates)
    gate_met = span >= EXTENDED_GRID_MIN_DAYS

    if settings.get("force_extended_grid") and not gate_met:
        logger.warning(
            "param_sweep: force_extended_grid requested but data gate NOT met "
            "(%dd < %dd of history) — falling back to DEFAULT_GRID to avoid "
            "over-fitting the extended grid in-sample (config#947)",
            span, EXTENDED_GRID_MIN_DAYS,
        )
        return DEFAULT_GRID

    if gate_met:
        logger.info(
            "param_sweep: data gate met (%dd >= %dd of history) — using "
            "EXTENDED_GRID (%d params, config#947)",
            span, EXTENDED_GRID_MIN_DAYS, len(EXTENDED_GRID),
        )
        return EXTENDED_GRID

    logger.info(
        "param_sweep: data gate not met (%dd < %dd of history) — using "
        "DEFAULT_GRID (core %d params, config#947)",
        span, EXTENDED_GRID_MIN_DAYS, len(DEFAULT_GRID),
    )
    return DEFAULT_GRID


# ── Defaults for sweep mode (override via param_sweep_settings in config.yaml) ──
_DEFAULT_SWEEP_MODE = "random"
_DEFAULT_TOP_FRACTION = 0.05      # target top 5% of parameter space
_DEFAULT_CONFIDENCE = 0.95        # 95% probability of hitting target

# Auto-scaling: sample trial_pct of the grid, clamped to [min_trials, max_trials].
# Floor guarantees statistical coverage; ceiling caps runtime for large grids.
_DEFAULT_TRIAL_PCT = 0.25         # sample 25% of the grid
_DEFAULT_MIN_TRIALS = 50          # floor: statistical minimum
_DEFAULT_MAX_TRIALS = 400         # ceiling: cap runtime


def compute_n_trials(
    top_fraction: float = 0.05,
    confidence: float = 0.95,
) -> int:
    """
    Compute the number of random trials needed to find a combo in the top-p
    fraction with the given confidence level.

    Formula: n = ceil(ln(1 - confidence) / ln(1 - top_fraction))

    Examples:
        top 5%  at 95% confidence → 59 trials
        top 5%  at 99% confidence → 90 trials
        top 1%  at 95% confidence → 299 trials
    """
    if top_fraction <= 0 or top_fraction >= 1:
        raise ValueError(f"top_fraction must be in (0, 1), got {top_fraction}")
    if confidence <= 0 or confidence >= 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    return math.ceil(math.log(1 - confidence) / math.log(1 - top_fraction))


def auto_n_trials(
    total_grid: int,
    trial_pct: float | None = None,
    min_trials: int | None = None,
    max_trials: int | None = None,
) -> int:
    """
    Compute the number of random trials scaled to grid size.

    Uses trial_pct of total_grid, clamped to [min_trials, max_trials].
    Guarantees the statistical floor (60 = 95% top-5%) for small grids,
    and caps runtime for large grids.

    Examples:
        grid=216,  30% → 65  (floor wins: 60 → 65 after rounding)
        grid=972,  30% → 292
        grid=5000, 30% → 500 (ceiling wins)
    """
    pct = trial_pct if trial_pct is not None else _DEFAULT_TRIAL_PCT
    floor = min_trials if min_trials is not None else _DEFAULT_MIN_TRIALS
    ceiling = max_trials if max_trials is not None else _DEFAULT_MAX_TRIALS

    scaled = math.ceil(total_grid * pct)
    n = max(floor, min(scaled, ceiling))

    # Never exceed the grid itself
    return min(n, total_grid)


def _generate_random_combos(
    grid: dict,
    n_trials: int,
    seed: int | None = None,
) -> list[dict]:
    """
    Sample n_trials unique parameter combinations from the grid.

    If n_trials >= total grid size, falls back to exhaustive grid (no benefit
    to random sampling when you can cover everything).
    """
    keys = list(grid.keys())
    values = list(grid.values())
    total_combos = 1
    for v in values:
        total_combos *= len(v)

    if n_trials >= total_combos:
        logger.info(
            "max_trials (%d) >= grid size (%d) — using exhaustive grid search",
            n_trials, total_combos,
        )
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    rng = random.Random(seed)
    seen: set[tuple] = set()
    combos: list[dict] = []

    while len(combos) < n_trials:
        sample = tuple(rng.choice(v) for v in values)
        if sample not in seen:
            seen.add(sample)
            combos.append(dict(zip(keys, sample)))

    return combos


def _run_combos(
    combinations: list[dict],
    run_simulation_fn: Callable[[dict], dict],
    base_config: dict,
) -> pd.DataFrame:
    """Run simulation for each parameter combination and return results DataFrame.

    Self-deadlining (config-I7309): when ``base_config`` carries a
    ``_pass_deadline_epoch``, the loop stops before starting a combo it cannot
    finish and returns the combos that DID complete, tagging
    ``df.attrs["sweep_budget_stopped"]``. Without a deadline the behaviour is
    exactly as before — every combo runs.
    """
    import time as _time
    rows = []
    n = len(combinations)
    t_sweep_start = _time.monotonic()
    reserve_s = cscv_reserve_s(base_config, [], 0)
    remaining_s = _budget.deadline_remaining_s(
        base_config, tag=_SWEEP_BUDGET_TAG, reserve_s=reserve_s,
    )
    combo_seconds: list[float] = []
    n_combos_stopped_for_budget = 0
    for i, params in enumerate(combinations, 1):
        # config-I7309: the reserve is re-derived each iteration rather than
        # held flat, because the work it covers — the CSCV matrix build — is a
        # function of how many combos this sweep has already produced. A flat
        # reserve is either too small (the matrix build is SIGKILLed, which is
        # what a bounded combo loop alone still did) or too large (combos are
        # given up to protect a matrix that never needed the room).
        reserve_s = cscv_reserve_s(base_config, combo_seconds, len(rows))
        if not _budget.next_unit_affordable(
            remaining_s, combo_seconds,
            reserve_s=reserve_s,
            first_unit_estimate_s=SWEEP_FIRST_COMBO_ESTIMATE_S,
            tag=_SWEEP_BUDGET_TAG,
        ):
            n_combos_stopped_for_budget = n - i + 1
            # LOUD: a truncated sweep is a real coverage loss and the number
            # that says whether the budget is sized right. Silently returning
            # a short frame would make an under-budgeted pass look like a
            # small grid.
            logger.error(
                "%s stopping early on budget: %d of %d combo(s) not run "
                "(p90 per-combo %.1fs, %.0fs of budget left, reserve %.0fs). "
                "The completed combos are returned as a PARTIAL sweep — this "
                "is the config-I7309 shape, not a silent short run.",
                _SWEEP_BUDGET_TAG, n_combos_stopped_for_budget, n,
                _budget.p90(combo_seconds, default=SWEEP_FIRST_COMBO_ESTIMATE_S),
                _budget.safe_remaining(remaining_s, tag=_SWEEP_BUDGET_TAG)
                if remaining_s is not None else float("inf"),
                reserve_s,
            )
            break
        # Per-combo progress at INFO so the sweep never goes silent. Each
        # combo is a full simulation (~30-90s); without this, 60 combos
        # run in complete silence at default INFO level and look like a
        # 60-min hang to any log reader. See ROADMAP Backtester P0
        # "Diagnose the silent-phase bottleneck" (2026-04-22).
        t_combo = _time.monotonic()
        logger.info("Sweep combo %d/%d: %s", i, n, params)
        try:
            # _deepcopy_safe_config is INSIDE the per-combo try (L4525): a
            # deepcopy / recursion failure on one combo's config must degrade
            # to an error-row, NOT escape the whole sweep and return
            # sweep_df=None — which the export guard then treats as a fatal
            # ABSENT sweep (recovery8 symptom). The docstring records a prior
            # recursion failure here from a boto3 client in config. Per the
            # L4523 outcome taxonomy + [[feedback_no_silent_fails]]: a single
            # bad combo is a logged warning, never a process kill.
            config = _deepcopy_safe_config(base_config)
            config.update(params)
            stats = run_simulation_fn(config)
            # Strip nested per-combo time-series before the row enters the
            # DataFrame (L4529). `vectorbt_bridge.run_vectorbt_simulation`
            # returns `daily_returns` / `daily_log_returns` as full pandas
            # Series (a ~2500-row path per combo). They are NOT scalar sweep
            # metrics; left in the row they make the column dtype `object` and
            # `sweep_df.to_parquet` dies with `ArrowInvalid: Could not convert
            # … with type Series` — which the L4518 fail-loud export guard then
            # escalates to a backtest-stage kill (→ no sweep_df.parquet /
            # portfolio_stats.json → Evaluator critical-artifact gate fails →
            # whole Saturday SF FAILED). Mirrors the existing `stats.pop(...)`
            # pattern in analysis/portfolio_optimizer_backtest.py. The sweep
            # only needs scalar metrics per combo; nothing reads these Series
            # back from the parquet. See [[feedback_no_silent_fails]].
            stats.pop("daily_returns", None)
            stats.pop("daily_log_returns", None)
            rows.append({**params, **stats})
        except Exception as e:
            logger.warning("Simulation failed for params %s: %s", params, e)
            rows.append({**params, "error": str(e)})
        combo_seconds.append(_time.monotonic() - t_combo)
        logger.info(
            "Sweep combo %d/%d done in %.1fs (sweep elapsed %.1fs)",
            i, n, combo_seconds[-1], _time.monotonic() - t_sweep_start,
        )

    df = pd.DataFrame(rows)
    # The measurement that says whether the bound is right. `combo_p90_s` is
    # what the next run's budget arithmetic uses; `n_combos_run` vs
    # `n_combos_planned` is the coverage actually achieved. A pass that is
    # persistently short needs a BIGGER BUDGET, not a bigger ceiling — and
    # that is only decidable if both numbers reach the artifact instead of
    # only the log (config-I7309).
    df.attrs["n_combos_planned"] = n
    df.attrs["n_combos_run"] = len(rows)
    df.attrs["n_combos_skipped_for_budget"] = n_combos_stopped_for_budget
    df.attrs["sweep_budget_stopped"] = bool(n_combos_stopped_for_budget)
    df.attrs["combo_p90_s"] = (
        round(_budget.p90(combo_seconds, default=SWEEP_FIRST_COMBO_ESTIMATE_S), 1)
        if combo_seconds else None
    )
    # The reserve is now derived, so it is a MEASUREMENT of this run rather
    # than a constant anyone can look up — it has to be emitted or the
    # sweep-vs-CSCV split cannot be reasoned about after the fact.
    df.attrs["cscv_reserve_s"] = round(float(reserve_s), 1)
    # Sort by sortino_ratio (primary — skilled-risk basket per
    # [[anchor_gates_on_skilled_risk_not_sharpe]] / evaluator-revamp-260506.md).
    # total_alpha is presentation/tiebreaker only — never primary, per
    # [[alpha_vs_spy_is_presentation_not_gating]]. Raw Sharpe is observability
    # only and is intentionally absent from this sort chain — fall through to
    # unsorted rather than re-anchor on Sharpe.
    _sort_sweep_df_skilled_risk(df)
    return df


def _sort_sweep_df_skilled_risk(df: "pd.DataFrame") -> None:
    """In-place sort by Sortino (primary) → total_alpha (tiebreaker).

    Safe when columns are missing or all-NaN — falls through to whichever
    column has at least one non-NaN value, in order of preference. Returns
    silently with no sort applied when no usable column exists (preferring
    the natural enumeration order over a re-anchor on Sharpe).
    """
    if "sortino_ratio" in df.columns and df["sortino_ratio"].notna().any():
        df.sort_values("sortino_ratio", ascending=False, inplace=True)
    elif "total_alpha" in df.columns and df["total_alpha"].notna().any():
        df.sort_values("total_alpha", ascending=False, inplace=True)


def sweep(
    grid: dict,
    run_simulation_fn: Callable[[dict], dict],
    base_config: dict,
    sweep_settings: dict | None = None,
) -> pd.DataFrame:
    """
    Parameter sweep over combinations from the grid.

    Args:
        grid: Dict mapping param name → list of values to try.
        run_simulation_fn: Callable that accepts a config dict and returns a
              stats dict (total_return, sharpe_ratio, max_drawdown, ...).
        base_config: Base config dict; each combination overrides relevant keys.
        sweep_settings: Dict from config.yaml param_sweep_settings section.
            Keys: mode ("grid"|"random"), max_trials (int, optional),
            trial_pct (float), min_trials (int), max_trials_cap (int),
            seed (int, optional).

    Returns:
        DataFrame with one row per parameter combination, sorted by
        ``total_alpha`` (primary) with ``sharpe_ratio`` as tiebreaker
        (per the sort applied by ``_run_combos``). Sweep metadata stored
        in df.attrs for reporting.
    """
    settings = sweep_settings or {}
    mode = settings.get("mode", _DEFAULT_SWEEP_MODE)
    seed = settings.get("seed")

    keys = list(grid.keys())
    values = list(grid.values())
    total_grid = 1
    for v in values:
        total_grid *= len(v)

    if mode == "random":
        # If max_trials is explicitly set, use it; otherwise auto-scale
        explicit_max = settings.get("max_trials")
        if explicit_max is not None:
            n = int(explicit_max)
        else:
            n = auto_n_trials(
                total_grid,
                trial_pct=settings.get("trial_pct"),
                min_trials=settings.get("min_trials"),
                max_trials=settings.get("max_trials_cap"),
            )
        combinations = _generate_random_combos(grid, n, seed=seed)
        actual_mode = "random" if len(combinations) < total_grid else "grid (auto-fallback)"
        coverage = len(combinations) / total_grid
        top_frac = _DEFAULT_TOP_FRACTION
        prob = 1 - (1 - top_frac) ** len(combinations)

        logger.info(
            "Random sweep: %d/%d combos (%.0f%% coverage). "
            "%.1f%% probability of finding top-%.0f%% combo across %s",
            len(combinations), total_grid, coverage * 100,
            prob * 100, top_frac * 100, keys,
        )
    else:
        combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        actual_mode = "grid"
        coverage = 1.0
        logger.info(
            "Grid sweep: %d combinations across %s",
            len(combinations), keys,
        )

    df = _run_combos(combinations, run_simulation_fn, base_config)

    # Gate: require at least 50% of combos to succeed.
    # config-I7309: budget-skipped combos are NOT successes. Before the
    # self-deadlining loop every planned combo produced a row, so
    # `n_total - n_failed` was a valid success count; a truncated sweep breaks
    # that identity and would have reported a short run as fully successful.
    n_total = len(combinations)
    if not df.empty and "error" in df.columns:
        n_failed = int(df["error"].notna().sum()) + int(
            df.attrs.get("n_combos_skipped_for_budget", 0)
        )
        n_valid = n_total - n_failed
        completion_pct = n_valid / n_total if n_total > 0 else 0
        if completion_pct < 0.50:
            logger.warning(
                "Param sweep: only %d/%d combos succeeded (%.0f%%) — "
                "below 50%% threshold, results may be unreliable",
                n_valid, n_total, completion_pct * 100,
            )
            df.attrs["sweep_low_completion"] = True
            df.attrs["sweep_completion_pct"] = round(completion_pct, 2)

    # Add metadata for reporting
    if not df.empty:
        df.attrs["sweep_mode"] = actual_mode
        df.attrs["sweep_total_grid"] = total_grid
        df.attrs["sweep_trials"] = len(combinations)
        df.attrs["sweep_coverage"] = coverage

    return df


def best_params(sweep_df: pd.DataFrame, metric: str = "sharpe_ratio") -> dict:
    """
    Return the parameter combination with the best value of `metric`.
    """
    if metric not in sweep_df.columns:
        raise ValueError(f"Metric '{metric}' not found in sweep results")

    best_row = sweep_df.dropna(subset=[metric]).iloc[0]
    stat_cols = {
        "total_return", "sharpe_ratio", "max_drawdown", "calmar_ratio",
        "total_trades", "win_rate", "error", "status", "dates_simulated",
        "total_orders", "note",
    }
    param_cols = [c for c in sweep_df.columns if c not in stat_cols]
    return {col: best_row[col] for col in param_cols}
