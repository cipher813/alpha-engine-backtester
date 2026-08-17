"""harness.py — the shared contribution-lift replay harness (RC v3 T5).

Report Card v3 §1 fixes ONE objective for the whole system:

    objective(cycle) = log_return_21d(portfolio, net_of_cost) − log_return_21d(SPY)

and §3 replaces every declared component weight with a **measured marginal
contribution** to that objective: replay the pipeline twice — once as
configured (the *baseline* arm), once with the component ablated (the
*ablated* arm) — and report the paired per-cycle difference.

This module is the machinery every T5 group shares. A group contributes ONE
:class:`ReplaySpec` to ``registry.SPECS``; the spec's ``build_arms`` callable
turns the live :class:`ReplayInputs` into an :class:`ArmSet` (or a
:class:`NotAvailable` when its input does not exist), and everything after
that — sizing, simulation, the objective, the paired bootstrap CI, count
matching, DSR/PBO, the emitted component dict — happens here exactly once.

Two arm shapes are supported, because "removing" a component means two
different things (spec §3):

* **picks** — ``[{"date": "YYYY-MM-DD", "picks": ["AAPL", ...]}, ...]``.
  The harness applies its own deterministic equal-weight sizing and a fixed
  hold of ``horizon_days`` trading rows. This is the *substitution* pattern:
  research-side arms differ only in WHICH names were selected, so a shared,
  boring sizing rule is exactly what isolates selection from sizing.
* **orders** — an explicit ``[{"date", "ticker", "action", "shares",
  "price_at_order"}, ...]`` list. This is the *null_arm* pattern: an
  executor-side arm has to reproduce real fills and real sizes, because the
  thing being ablated (a guard, an exit rule, a sizer) acts ON those.

Both shapes go through the SAME cost-bearing simulator
(``vectorbt_bridge.orders_to_portfolio`` + ``portfolio_stats``) that
``backtest._run_simulation_loop`` uses, at the config's own ``fees`` /
``slippage_bps`` — a replay that skips the cost model is measuring a proxy,
not the objective (spec §1).

Determinism: no RNG anywhere except the bootstrap's fixed ``seed=0``; ties in
selection are broken by ticker ascending.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from nousergon_lib.quant.horizons import DEFAULT_POLICY

from analysis.contribution_lift import objective as _objective

logger = logging.getLogger(__name__)

#: The fleet's canonical primary horizon (21 trading days). Never hardcoded —
#: the objective is defined against whatever DEFAULT_POLICY says it is.
HORIZON_DAYS: int = int(DEFAULT_POLICY.primary_horizon)

#: Cycles below this cannot support a lift claim at all (spec §1: 0.5 × n_floor).
MIN_SAMPLES: int = 30

#: The objective's own floor, inherited from the portfolio_outcome tile.
N_FLOOR: int = 60

#: The unit every contribution_lift record carries (krepis#158 `unit` field).
UNIT: str = "log_alpha_21d"

#: trial_accumulator producer name for this harness.
TRIAL_PRODUCER: str = "contribution_lift"

_VALID_CRITICALITY = ("critical", "supporting", "diagnostic")
_VALID_PATTERN = ("substitution", "null_arm")
_VALID_NA_STATUS = (
    "N/A-MISSING-INPUT",
    "N/A-LOW-N",
    "N/A-RETIRED",
    "N/A-NOT-LIFT-SHAPED",
)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


@dataclass
class ReplayInputs:
    """Everything a ``build_arms`` callable is allowed to read.

    Assembled ONCE per run by :func:`analysis.contribution_lift.inputs.
    load_replay_inputs` and handed to every spec, so eight sibling groups
    share one S3/ArcticDB load rather than each doing their own.

    ``status``/``reason`` carry the loader's own verdict: ``"ok"`` with a
    populated window, or ``"skipped"`` with a reason when there were zero
    usable signal dates. An observational artifact is emitted either way —
    absence of ``contribution_lift.json`` must always mean "the producer
    never ran" (reporter.py's always-emit contract).
    """

    run_date: str
    dates: list[str]
    signals_by_date: dict[str, dict]
    predictions_by_date: dict[str, dict]
    pillar_profiles_by_date: dict[str, dict]
    price_matrix: pd.DataFrame
    spy_prices: pd.Series
    bucket: str
    fees: float
    slippage_bps: float
    init_cash: float
    trades_db_path: str | None = None
    research_db_path: str | None = None
    s3_client: Any = None
    status: str = "ok"
    reason: str | None = None
    source_paths: list[str] = field(default_factory=list)

    @property
    def horizon_days(self) -> int:
        return HORIZON_DAYS


# --------------------------------------------------------------------------
# Arms
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Arm:
    """One replayed configuration of the pipeline.

    Exactly one of ``picks`` / ``orders`` must be set.

    ``fees`` / ``slippage_bps`` default to the run's configured cost model
    (``None`` ⇒ inherit). They exist so a group can ablate the COST MODEL
    itself (the zero-cost arm behind ``cost_adjusted_quality``); every other
    group leaves them ``None`` so both arms are priced identically and the
    measured difference is attributable to the component alone.
    """

    label: str
    picks: tuple[tuple[str, tuple[str, ...]], ...] | None = None
    orders: tuple[Mapping[str, Any], ...] | None = None
    fees: float | None = None
    slippage_bps: float | None = None

    def __post_init__(self) -> None:
        if (self.picks is None) == (self.orders is None):
            raise ValueError(
                f"Arm {self.label!r}: exactly one of picks/orders must be set "
                f"(got picks={self.picks is not None}, orders={self.orders is not None})"
            )


def picks_arm(
    label: str,
    picks_by_cycle: Sequence[Mapping[str, Any]],
    *,
    fees: float | None = None,
    slippage_bps: float | None = None,
) -> Arm:
    """Build a picks-shaped :class:`Arm` from ``[{"date", "picks"}, ...]``.

    Normalizes to an immutable, ticker-sorted structure so two runs over the
    same inputs produce byte-identical arms (determinism, contract §Rules).
    """
    normalized: list[tuple[str, tuple[str, ...]]] = []
    for cycle in picks_by_cycle:
        d = str(cycle["date"])
        tickers = tuple(sorted({str(t) for t in (cycle.get("picks") or [])}))
        normalized.append((d, tickers))
    normalized.sort(key=lambda row: row[0])
    return Arm(
        label=label,
        picks=tuple(normalized),
        fees=fees,
        slippage_bps=slippage_bps,
    )


def orders_arm(
    label: str,
    orders: Sequence[Mapping[str, Any]],
    *,
    fees: float | None = None,
    slippage_bps: float | None = None,
) -> Arm:
    """Build an orders-shaped :class:`Arm` from an explicit order list."""
    ordered = sorted(
        (dict(o) for o in orders),
        key=lambda o: (str(o.get("date")), str(o.get("ticker")), str(o.get("action"))),
    )
    return Arm(
        label=label,
        orders=tuple(ordered),
        fees=fees,
        slippage_bps=slippage_bps,
    )


@dataclass(frozen=True)
class ArmSet:
    """The arms one spec wants replayed.

    ``sweep`` carries ADDITIONAL arms beyond the baseline/ablated pair. It is
    what distinguishes a grid from a single leave-one-out comparison: PBO is
    reported only when ``sweep`` is non-empty, because CSCV-PBO answers
    "is this the cherry-picked winner of a search?" and a two-arm
    leave-one-out ran no search (spec §1).
    """

    baseline: Arm
    ablated: Arm
    sweep: tuple[Arm, ...] = ()

    def all_arms(self) -> tuple[Arm, ...]:
        return (self.baseline, self.ablated, *self.sweep)


@dataclass(frozen=True)
class NotAvailable:
    """A spec declining to measure, with the reason an operator needs.

    Never a fabricated value: a component whose input does not exist (retired
    producer, per-model predictions never persisted, a metric that is not
    lift-shaped by construction) emits an ``N/A-*`` component naming the
    missing artifact and its tracking issue.
    """

    status: str
    reason: str

    def __post_init__(self) -> None:
        if self.status not in _VALID_NA_STATUS:
            raise ValueError(
                f"NotAvailable.status must be one of {_VALID_NA_STATUS}, got {self.status!r}"
            )


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySpec:
    """One graded component's replay.

    Attributes:
        name: the evaluator tile component name — the join key between this
            artifact and ``crucible-evaluator``'s tile. Must match exactly.
        module: owning tile (``research`` / ``predictor`` / ``executor`` /
            ``agent`` / ``behavioral``).
        criticality: inherited from the component's existing tile record
            (spec §3) — never re-derived here.
        pattern: ``substitution`` (re-score/re-select) or ``null_arm``
            (disable a gate), per spec §3's two named patterns.
        issue: ``alpha-engine-config-I<N>`` this replay closes.
        build_arms: pure function of :class:`ReplayInputs`. Returns an
            :class:`ArmSet` to be measured, or :class:`NotAvailable`.
    """

    name: str
    module: str
    criticality: str
    pattern: str
    issue: str
    build_arms: Callable[[ReplayInputs], "ArmSet | NotAvailable"]

    def __post_init__(self) -> None:
        if self.criticality not in _VALID_CRITICALITY:
            raise ValueError(
                f"{self.name}: criticality must be one of {_VALID_CRITICALITY}, "
                f"got {self.criticality!r}"
            )
        if self.pattern not in _VALID_PATTERN:
            raise ValueError(
                f"{self.name}: pattern must be one of {_VALID_PATTERN}, "
                f"got {self.pattern!r}"
            )


# --------------------------------------------------------------------------
# Sizing / order construction
# --------------------------------------------------------------------------


def picks_to_orders(
    picks: Sequence[tuple[str, Sequence[str]]],
    price_matrix: pd.DataFrame,
    *,
    init_cash: float,
    per_cycle_slots: int,
    hold_days: int = HORIZON_DAYS,
) -> list[dict]:
    """Deterministic equal-weight order list for a picks-shaped arm.

    Every name gets ``init_cash / per_cycle_slots`` of notional and is held
    for ``hold_days`` trading rows. ``per_cycle_slots`` is computed ONCE
    across every arm of the set (see :func:`_slot_count`) so two arms compared
    against each other are sized identically — a per-arm budget would let the
    narrower arm silently lever up and the "lift" would be a sizing artifact.

    Mirrors ``analysis.factor_blend_counterfactual_replay.build_orders_for_
    weights``: same order shape, same floor-to-whole-shares rule, same
    snap-forward when a cycle date is not itself a price row.
    """
    dates = price_matrix.index
    orders: list[dict] = []
    if len(dates) == 0:
        return orders
    per_name_budget = float(init_cash) / max(1, int(per_cycle_slots))

    for raw_date, tickers in picks:
        cdate = pd.Timestamp(raw_date)
        if cdate not in dates:
            future = dates[dates >= cdate]
            if len(future) == 0:
                continue
            cdate = future[0]
        entry_pos = dates.get_loc(cdate)
        exit_pos = min(entry_pos + hold_days, len(dates) - 1)
        exit_date = dates[exit_pos]

        for ticker in sorted(tickers):
            if ticker not in price_matrix.columns:
                continue
            entry_price = price_matrix.at[cdate, ticker]
            if pd.isna(entry_price) or float(entry_price) <= 0.0:
                continue
            shares = float(np.floor(per_name_budget / float(entry_price)))
            if shares <= 0.0:
                continue
            orders.append({
                "date": cdate.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "action": "ENTER",
                "shares": shares,
                "price_at_order": float(entry_price),
            })
            if exit_pos > entry_pos:
                orders.append({
                    "date": exit_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "action": "EXIT",
                })
    return orders


def _slot_count(arm_set: ArmSet) -> int:
    """Widest per-cycle pick count across every arm — the shared sizing base."""
    widest = 0
    for arm in arm_set.all_arms():
        if arm.picks is None:
            continue
        for _d, tickers in arm.picks:
            widest = max(widest, len(tickers))
    return max(1, widest)


def arm_orders(arm: Arm, arm_set: ArmSet, inputs: ReplayInputs) -> list[dict]:
    """Resolve an arm to a concrete order list."""
    if arm.orders is not None:
        return [dict(o) for o in arm.orders]
    assert arm.picks is not None  # guaranteed by Arm.__post_init__
    return picks_to_orders(
        arm.picks,
        inputs.price_matrix,
        init_cash=inputs.init_cash,
        per_cycle_slots=_slot_count(arm_set),
        hold_days=inputs.horizon_days,
    )


def picks_per_cycle(arm: Arm) -> dict[str, int]:
    """``{cycle_date: n_names}`` for an arm, whichever shape it carries.

    For an orders-shaped arm the width of a cycle is the number of DISTINCT
    tickers entered on that date — which is what count-matching constrains.
    """
    if arm.picks is not None:
        return {d: len(t) for d, t in arm.picks}
    assert arm.orders is not None
    by_date: dict[str, set[str]] = {}
    for order in arm.orders:
        if str(order.get("action")) != "ENTER":
            continue
        by_date.setdefault(str(order.get("date")), set()).add(str(order.get("ticker")))
    return {d: len(t) for d, t in by_date.items()}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


def simulate_arm(arm: Arm, arm_set: ArmSet, inputs: ReplayInputs) -> dict:
    """Run one arm through the production simulator and return its stats.

    Returns the ``vectorbt_bridge.portfolio_stats`` dict verbatim plus
    ``n_orders`` — including its ``daily_log_returns`` Series, which is the
    ONLY series the objective is computed from. Sharpe/Sortino are carried
    through as diagnostics and are never used for status derivation while
    I7236 / I7237 / I7271 are open (spec §1 blocking note).
    """
    # Imported lazily: vectorbt is a heavy import and the unit tests that
    # exercise the objective/CI math never touch the simulator.
    from vectorbt_bridge import orders_to_portfolio, portfolio_stats

    orders = arm_orders(arm, arm_set, inputs)
    fees = inputs.fees if arm.fees is None else float(arm.fees)
    slippage_bps = (
        inputs.slippage_bps if arm.slippage_bps is None else float(arm.slippage_bps)
    )
    pf = orders_to_portfolio(
        orders,
        inputs.price_matrix,
        init_cash=inputs.init_cash,
        fees=fees,
        slippage_bps=slippage_bps,
    )
    stats = portfolio_stats(pf, spy_prices=inputs.spy_prices)
    stats["n_orders"] = len(orders)
    return stats


# --------------------------------------------------------------------------
# Running one spec
# --------------------------------------------------------------------------


def _na_component(
    spec: ReplaySpec, na: NotAvailable, *, run_date: str, bucket: str
) -> dict:
    return _component_skeleton(spec, run_date=run_date, bucket=bucket) | {
        "status": na.status,
        "status_reason": na.reason,
        "value": None,
        "ci_low": None,
        "ci_high": None,
        "ci_method": "bootstrap",
        "n_samples": 0,
        "count_matched": None,
        "arms": {},
        "dsr": None,
        "pbo": None,
    }


def _component_skeleton(spec: ReplaySpec, *, run_date: str, bucket: str) -> dict:
    return {
        "name": spec.name,
        "module": spec.module,
        "criticality": spec.criticality,
        "pattern": spec.pattern,
        "issue": spec.issue,
        "unit": UNIT,
        "n_floor": N_FLOOR,
        "source_path": (
            f"s3://{bucket}/backtest/{run_date}/contribution_lift.json"
            f"#components/{spec.name}"
        ),
    }


def _arm_payload(
    label: str,
    stats: Mapping[str, Any],
    per_cycle: Mapping[str, float],
    width: Mapping[str, int],
    n_trials: int,
) -> dict:
    from nousergon_lib.quant.stats.dsr import compute_dsr

    widths = sorted(set(width.values()))
    daily_returns = stats.get("daily_returns")
    dsr_value: float | None = None
    if isinstance(daily_returns, pd.Series) and len(daily_returns.dropna()) > 2:
        # compute_dsr requires n_trials >= 1; a counter that has not been
        # written yet reads 0, which is "no trial history", not "zero trials".
        dsr_result = compute_dsr(daily_returns.dropna(), max(1, int(n_trials)))
        if dsr_result.get("status") == "ok":
            dsr_value = _finite(dsr_result.get("dsr"))
    return {
        "label": label,
        "n_orders": int(stats.get("n_orders", 0)),
        # A single integer when every cycle has the same width (the normal,
        # count-matched case); the observed set otherwise, so the gap's own
        # status_reason has something concrete to name.
        "picks_per_cycle": widths[0] if len(widths) == 1 else widths,
        "total_return": _finite(stats.get("total_return")),
        "total_alpha": _finite(stats.get("total_alpha")),
        "sortino_ratio": _finite(stats.get("sortino_ratio")),
        "sharpe_ratio": _finite(stats.get("sharpe_ratio")),
        "max_drawdown": _finite(stats.get("max_drawdown")),
        "psr": _finite(stats.get("psr")),
        "dsr": dsr_value,
        "per_cycle_log_alpha_21d": {d: float(v) for d, v in sorted(per_cycle.items())},
    }


def _finite(value: Any) -> float | None:
    """Coerce to a JSON-safe finite float, or None.

    ``+inf``/``NaN`` reaching an artifact is the config-I7237 failure mode
    (metrics.json emitting ``+inf`` for flat-market Sharpe and thereby not
    being strict JSON). Recorded as an explicit null instead — the consumer
    reads null as "not defined this run", which is true, and never parses a
    bare ``Infinity`` token.
    """
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def run_spec(
    spec: ReplaySpec,
    inputs: ReplayInputs,
    *,
    n_trials: int,
    built: "ArmSet | NotAvailable | None" = None,
) -> dict:
    """Replay one spec and return its contract-shaped component dict.

    ``n_trials`` is the fleet-cumulative trial count AFTER this run's
    increment — DSR corrects for selection across every trial the fleet has
    run, not just this replay's own arms (spec §1).

    ``built`` lets the caller supply an already-constructed arm set so the
    trial count and the simulated arms describe the same object (see
    ``report._build_all``); omitted, the spec is built here.
    """
    if built is None:
        built = spec.build_arms(inputs)
    if isinstance(built, NotAvailable):
        return _na_component(
            spec, built, run_date=inputs.run_date, bucket=inputs.bucket
        )
    if not isinstance(built, ArmSet):
        raise TypeError(
            f"{spec.name}.build_arms returned {type(built).__name__}; "
            "expected ArmSet or NotAvailable"
        )

    component = _component_skeleton(
        spec, run_date=inputs.run_date, bucket=inputs.bucket
    )

    baseline_stats = simulate_arm(built.baseline, built, inputs)
    ablated_stats = simulate_arm(built.ablated, built, inputs)

    baseline_alpha = _objective.per_cycle_log_alpha(
        baseline_stats["daily_log_returns"],
        inputs.spy_prices,
        inputs.price_matrix.index,
        inputs.dates,
        horizon_days=inputs.horizon_days,
    )
    ablated_alpha = _objective.per_cycle_log_alpha(
        ablated_stats["daily_log_returns"],
        inputs.spy_prices,
        inputs.price_matrix.index,
        inputs.dates,
        horizon_days=inputs.horizon_days,
    )

    baseline_width = picks_per_cycle(built.baseline)
    ablated_width = picks_per_cycle(built.ablated)
    match = _objective.check_count_match(baseline_width, ablated_width)

    paired = _objective.paired_diffs(baseline_alpha, ablated_alpha)
    n_samples = len(paired)

    component["arms"] = {
        "baseline": _arm_payload(
            built.baseline.label, baseline_stats, baseline_alpha,
            baseline_width, n_trials,
        ),
        "ablated": _arm_payload(
            built.ablated.label, ablated_stats, ablated_alpha,
            ablated_width, n_trials,
        ),
    }
    component["dsr"] = {
        "baseline": component["arms"]["baseline"]["dsr"],
        "ablated": component["arms"]["ablated"]["dsr"],
        "n_trials": n_trials,
    }
    component["n_samples"] = n_samples
    component["ci_method"] = "bootstrap"

    # PBO is reported ONLY for a genuine grid. A baseline-vs-ablated pair is a
    # single leave-one-out comparison — nothing was selected, so there is no
    # overfit-of-selection to estimate (spec §1).
    if built.sweep:
        sweep_alpha = {built.baseline.label: baseline_alpha,
                       built.ablated.label: ablated_alpha}
        for arm in built.sweep:
            stats = simulate_arm(arm, built, inputs)
            sweep_alpha[arm.label] = _objective.per_cycle_log_alpha(
                stats["daily_log_returns"],
                inputs.spy_prices,
                inputs.price_matrix.index,
                inputs.dates,
                horizon_days=inputs.horizon_days,
            )
        component["pbo"] = _objective.compute_pbo(sweep_alpha)
    else:
        component["pbo"] = None

    if not match.matched:
        # An unmatched-width comparison silently favors whichever arm trades
        # more; spec §3 binds every T5 replay to report that as a GAP rather
        # than a measured lift. No value is emitted.
        component["status"] = "gap"
        component["status_reason"] = match.reason
        component["value"] = None
        component["ci_low"] = None
        component["ci_high"] = None
        component["count_matched"] = False
        return component

    component["count_matched"] = True

    if n_samples < MIN_SAMPLES:
        component["status"] = "N/A-LOW-N"
        component["status_reason"] = (
            f"{n_samples} paired cycles with both arms defined; "
            f"below the {MIN_SAMPLES}-cycle minimum (0.5 x n_floor={N_FLOOR})"
        )
        component["value"] = None
        component["ci_low"] = None
        component["ci_high"] = None
        return component

    ci = _objective.paired_bootstrap_ci(paired)
    component["status"] = "ok"
    component["status_reason"] = (
        f"{n_samples} paired cycles, count-matched; mean paired "
        f"{UNIT} difference over the {inputs.horizon_days}d horizon"
    )
    component["value"] = ci["estimate"]
    component["ci_low"] = ci["ci_low"]
    component["ci_high"] = ci["ci_high"]
    return component
