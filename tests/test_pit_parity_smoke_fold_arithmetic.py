"""The walk-forward fold scheme has to FIT the date axis it is handed
(alpha-engine-config-I7309).

WHAT WAS RECORDED, AND WHAT WAS TRUE
------------------------------------
`alpha-engine-config-I7309` records, as blocker 2 of re-enabling `pit_parity`
on the weekly SF:

    The walk-forward pass places ZERO orders. `SKIP ENTER <ticker> — shares
    round to 0 (weight=0.000, dollar=$0, ...)` for all 20 signals ... `pit_status`
    comes back `no_orders` while `current_status` is `ok` on the same date grid.
    Reproduces in 69.9 s via the `smoke-pit-parity` slice.

The `no_orders` observation is real and reproducible. Its attribution is not.
The production walk-forward pass trades: `backtest/2026-07-31/pit_parity.json`
carries the child's own log tail — `Simulation: 1995/1995 dates (100%
coverage), 2489 orders`, then `Sweep combo 5/60 done in 272.5s` — and
`backtest/2026-08-07/pit_parity.json` carries live `ORDER REDUCE` lines from
the same pass. What produced zero orders was the SMOKE FIXTURE, and the cause
is arithmetic, not sizing:

  * `smoke-pit-parity` caps the date axis at `max_trading_days: 60`.
  * The canonical fold scheme is `min_train: 504` (~2y of warm-up before the
    first fold), so `build_walk_forward_folds` needs **514** trading dates
    before it emits anything — its `while fold_start_idx < n` loop, with
    `fold_start_idx` initialised to `min_train`, never executed.
  * Zero folds → `predictions_by_date == {}` → `build_signals_by_date`
    iterates an empty date list → the simulation loop runs zero dates →
    `no_orders`, whose note read "No ENTER signals passed risk rules during
    the simulation period".

Every step was silent. The one guard in `run_walk_forward_inference` that
would have said something (`"%d fold(s) built but ZERO test dates scored"`) is
gated on `if folds and ...` — so the zero-FOLD case was precisely the case it
could not report. The look-ahead pass runs `walk_forward=False`, a single pass
over all dates, which is why `current_status: ok` on the identical fixture.

WHY THE TESTS BELOW HAVE THIS SHAPE
-----------------------------------
Two numbers in two different files have to stay compatible, and neither file
mentions the other. A hardcoded threshold would not notice a change to either,
so the required axis length is DERIVED from the fold builder itself
(`min_trading_dates_for_one_fold`, which calls the builder rather than
re-implementing it). The assertion is merge-blocking and hermetic — no S3, no
executor, no subprocess — so the class is caught at review rather than three
Saturdays later.
"""

from __future__ import annotations

import datetime as dt

import pytest

import backtest as bt
from synthetic.pit_folds import (
    build_walk_forward_folds,
    min_trading_dates_for_one_fold,
)
from synthetic.predictor_backtest import _WF_DEFAULTS


def _resolved_wf_params(overrides: dict) -> dict:
    """The fold params a run under these overrides actually uses — the same
    `{**_WF_DEFAULTS, **wf_params}` merge `run_walk_forward_inference` does."""
    pb = overrides.get("predictor_backtest", {}) or {}
    return {**_WF_DEFAULTS, **(pb.get("walk_forward_params", {}) or {})}


# ── The regression this file exists for ────────────────────────────────────

def test_smoke_pit_parity_fixture_can_build_at_least_one_fold():
    """THE regression test for config-I7309 blocker 2.

    Fails on the tree that produced `pit_status: no_orders` for three
    consecutive weeks, passes on the fix, and fails again if either number is
    later changed in isolation.
    """
    spec = bt._SMOKE_PHASE_MODES["smoke-pit-parity"]
    overrides = spec["overrides"]
    pb = overrides["predictor_backtest"]
    params = _resolved_wf_params(overrides)

    needed = min_trading_dates_for_one_fold(**params)
    assert needed is not None, (
        f"the smoke-pit-parity fold params {params} place no fold at any axis "
        f"length within the search window — the scheme is misconfigured"
    )

    # The fixture guarantees at LEAST min_trading_days and at most
    # max_trading_days, so the binding constraint is the lower bound: the
    # smoke must build a fold on its worst-case axis, not only its best.
    floor = pb["min_trading_days"]
    assert needed <= floor, (
        f"smoke-pit-parity builds ZERO walk-forward folds: its fold scheme "
        f"{params} needs {needed} trading dates and the fixture supplies at "
        f"most {pb['max_trading_days']} (guaranteed {floor}). The pass then "
        f"produces no predictions, no signals and no orders, and reports "
        f"pit_status='no_orders' — read for three weeks as the PRODUCTION "
        f"walk-forward pass declining to trade (config-I7309). Either raise "
        f"predictor_backtest.max_trading_days/min_trading_days above "
        f"{needed} or lower walk_forward_params.min_train."
    )


def test_the_canonical_scheme_is_what_the_smoke_deviates_from():
    """Anchors the two numbers the failure turned on, so a future reader can
    see the gap rather than rediscover it: the canonical scheme needs 514
    dates; the smoke slice is two orders of magnitude shorter."""
    assert min_trading_dates_for_one_fold(**_WF_DEFAULTS) == 514
    pb = bt._SMOKE_PHASE_MODES["smoke-pit-parity"]["overrides"]["predictor_backtest"]
    assert pb["max_trading_days"] < 514, (
        "the smoke fixture now supplies enough dates for the CANONICAL scheme "
        "— drop its walk_forward_params override rather than carrying two "
        "schemes that no longer differ"
    )


def test_production_scale_axis_builds_many_folds():
    """The counterweight to the assertion above: the canonical scheme is not
    itself broken. `predictor_backtest.max_trading_days` is 2500 in the
    reference config and that axis builds 95 folds — which is why the
    production pass places thousands of orders while the smoke placed none."""
    axis = [dt.date(2015, 1, 5) + dt.timedelta(days=i) for i in range(2500)]
    folds = build_walk_forward_folds(
        axis,
        test_window=_WF_DEFAULTS["test_window"],
        min_train=_WF_DEFAULTS["min_train"],
        purge=_WF_DEFAULTS["purge"],
        embargo=_WF_DEFAULTS["embargo"],
        train_mode=_WF_DEFAULTS["train_mode"],
    )
    assert len(folds) > 50


# ── The helper the assertion rests on ──────────────────────────────────────

@pytest.mark.parametrize(
    "params",
    [
        dict(test_window=21, min_train=504, purge=21, embargo=2),
        dict(test_window=5, min_train=10, purge=2, embargo=1),
        dict(test_window=1, min_train=1, purge=0, embargo=0),
    ],
)
def test_min_trading_dates_is_exact_at_the_boundary(params):
    """`needed` must be the SMALLEST axis that works: one date shorter builds
    nothing, `needed` itself builds at least one fold. Anything looser and the
    merge-blocking assertion above would pass a fixture that still cannot run.
    """
    needed = min_trading_dates_for_one_fold(**params)
    assert needed is not None

    def _folds(n: int):
        axis = [dt.date(2000, 1, 3) + dt.timedelta(days=i) for i in range(n)]
        return build_walk_forward_folds(axis, train_mode="expanding", **params)

    assert len(_folds(needed)) >= 1
    assert len(_folds(needed - 1)) == 0


def test_min_trading_dates_returns_none_on_a_nonsense_scheme():
    """A caller must be able to tell "needs more dates" from "these parameters
    can never place a fold" — the abort message says something different in
    each case."""
    assert min_trading_dates_for_one_fold(
        test_window=0, min_train=10, purge=1, embargo=0,
    ) is None
    assert min_trading_dates_for_one_fold(
        test_window=5, min_train=0, purge=1, embargo=0,
    ) is None
