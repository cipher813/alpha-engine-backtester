"""deadline_budget.py — the self-deadlining loop primitive.

A long loop handed a wall-clock deadline must decide, *before* each unit of
work, whether another unit still fits. The alternative — run until the parent
SIGKILLs the process — destroys the whole artifact, including the units that
already succeeded. That failure shape has now cost this repo two artifacts:

  * the concordance sweep (crucible-backtester#633), which sized its cap on a
    MEAN and was blown through by a 30x tail; and
  * the ``pit_parity`` walk-forward pass, which hit its 2700 s subprocess
    ceiling on 2026-08-07 and 2026-08-13 with nothing to show
    (alpha-engine-config-I7309, #6036).

config#7199 landed the correct shape for the walk-forward FOLD loop inside
``synthetic/predictor_backtest.py``. This module is that shape lifted out of it
on its second adoption in this repo — the ``param_sweep`` combo loop
(``analysis/param_sweep._run_combos``), which is the *other* unbounded loop
inside the very same pass and the one the 2026-07-31 evidence names directly:

    Sweep combo 5/60 done in 272.5s (sweep elapsed 2144.1s)

60 combos at 272.5 s is ~16,350 s of work under a 2700 s ceiling. Per
``policy-shared-code`` the pattern is lifted rather than copied; both callers
delegate here so there is ONE implementation of "does the next unit fit".

Scope note: this stays repo-local (not ``nousergon-lib``) because both adopters
are in this repo. Lift it to the shared library on the first adoption OUTSIDE
crucible-backtester.

Contract
--------
``deadline_remaining_s(config, ...)`` returns ``None`` when no deadline was
handed down — every caller that is not an isolated ``pit_parity`` pass (the
CLI, the phase, tests) then behaves exactly as it did before. An unreadable
clock means "unbounded", never "stop": a budget probe must not be able to
truncate work on its own failure.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

logger = logging.getLogger(__name__)

#: Config key carrying a process's wall-clock deadline as a UNIX epoch second.
#: Injected by ``analysis.pit_parity._run_predictor_pass_isolated`` into the
#: config JSON the isolated pass child reads, so the child inherits the
#: parent's subprocess timeout instead of discovering it as a SIGKILL. Epoch
#: (not a duration) so it survives the process boundary unambiguously.
DEADLINE_EPOCH_CONFIG_KEY = "_pass_deadline_epoch"


def deadline_remaining_s(
    config: dict,
    *,
    tag: str,
    reserve_s: float,
) -> Callable[[], float] | None:
    """Build the remaining-seconds callable from ``config``'s deadline.

    Returns ``None`` when no deadline was handed down, or when the value
    present cannot be read as a number — an unreadable deadline is ignored
    LOUDLY rather than truncating the loop on a value nobody can interpret.

    ``tag`` prefixes every log line so the two adopters stay distinguishable in
    one interleaved log (``[walk_forward]`` vs ``[param_sweep]``).
    """
    deadline = config.get(DEADLINE_EPOCH_CONFIG_KEY)
    if deadline is None:
        return None
    try:
        deadline = float(deadline)
    except (TypeError, ValueError):
        logger.error(
            "%s %s=%r is not a number — ignoring the deadline rather than "
            "truncating on a value nobody can read.",
            tag, DEADLINE_EPOCH_CONFIG_KEY, deadline,
        )
        return None
    logger.info(
        "%s loop is self-deadlining: %.0fs of budget remain (reserve %.0fs "
        "held back for the work that follows the loop + the artifact write).",
        tag, deadline - time.time(), reserve_s,
    )
    return lambda: deadline - time.time()


def safe_remaining(remaining_s: Callable[[], float], *, tag: str) -> float:
    """Evaluate a remaining-seconds callable without ever raising.

    A budget probe that throws must not take the run down with it — an
    unreadable clock means "unbounded", which degrades to the pre-deadline
    behaviour (run every unit) rather than to a silent truncation nobody asked
    for.
    """
    try:
        return float(remaining_s())
    except Exception as exc:  # noqa: BLE001 — see docstring; the log IS the surface
        logger.error(
            "%s remaining-budget probe failed (%s: %s) — treating the budget "
            "as unbounded for this unit.", tag, type(exc).__name__, exc,
        )
        return float("inf")


def p90(samples: list[float], *, default: float) -> float:
    """p90 of the observed per-unit durations.

    p90 and not p50 deliberately: the concordance incident
    (crucible-backtester#633) sized a budget on a mean and the slow tail ran
    30x it, so the cap could never bind. The budget must be sized on the unit
    that might come NEXT, not on the typical one.

    ``default`` is returned before any unit has been measured.
    """
    ordered = sorted(samples)
    if not ordered:
        return default
    # Rounds UP rather than nearest-rank, so with few samples the estimate
    # walks toward the observed MAX. That is the safe direction for a
    # deadline: an over-estimate costs one unit of coverage, an under-estimate
    # costs the whole artifact. Nearest-rank on ten samples returns the 9th of
    # nine fast units and ignores the one 100x outlier entirely — the
    # concordance mistake.
    idx = min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered))))
    return ordered[idx]


def next_unit_affordable(
    remaining_s: Callable[[], float] | None,
    unit_seconds: list[float],
    *,
    reserve_s: float,
    first_unit_estimate_s: float,
    tag: str,
) -> bool:
    """True iff another unit of work plausibly fits before the deadline.

    ``remaining - reserve >= p90(observed units)``. Before any unit has been
    measured the estimate is ``first_unit_estimate_s`` — so a loop handed a
    budget it has ALREADY blown runs zero units and reports zero coverage
    rather than starting work it cannot finish. Zero coverage is UNKNOWN
    downstream, never a pass.

    ``remaining_s is None`` (no deadline handed down) is always affordable.
    """
    if remaining_s is None:
        return True
    return (
        safe_remaining(remaining_s, tag=tag) - reserve_s
    ) >= p90(unit_seconds, default=first_unit_estimate_s)
