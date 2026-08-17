"""behavioral.cost_adjusted_quality — the cost-drag null arm (config-I7484).

Coverage table (spec §2, behavioral tile): ``cost_adjusted_quality`` has no
replay of any kind. The evaluator tile computes it in-tile, and that module's
own docstring says it lacks a baseline to compare against. Spec §4 group 7
files it standalone: "closest existing in-tile computation to a lift; needs a
baseline arm, not new machinery."

The baseline arm it needs is the zero-cost counterfactual. Both arms trade the
SAME names, on the same dates, in the same sizes — the ONLY difference is the
cost model:

* **baseline** — the live selection (``harness.live_picks_by_cycle``: the
  executed ENTERs from trades.db, champion-aware per config-I7501) at the
  configured ``simulation_fees`` / ``simulation.slippage_bps``.
* **ablated**  — byte-identical picks at ``fees=0, slippage_bps=0``.

So ``lift = baseline_alpha − ablated_alpha`` is the measured cost drag, and it
is expected NEGATIVE: trading costs money. That is the point. The record's
``red_line`` is 0.0 like every contribution_lift record, and a cost drag that
crosses zero (i.e. costs helping) would mean the cost model is mis-signed —
which is exactly the kind of thing the report card exists to surface.

Count-matching is satisfied BY CONSTRUCTION: the ablated arm is the same pick
list object, so ``picks_per_cycle`` is identical on every cycle by identity,
not by a downselect that could drift.

This is also the harness's own end-to-end exercise: it is deliberately the
simplest real spec, so a green ``contribution_lift.json`` on the weekly run
proves the loader, the sizing, the simulator wiring, the objective, the paired
bootstrap and the emission all work before seven sibling groups land on top.
"""

from __future__ import annotations

from analysis.contribution_lift.harness import (
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    live_picks_by_cycle,
    live_selection_label,
    no_live_selection,
    picks_arm,
)

ISSUE = "alpha-engine-config-I7484"


def build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    cycles = live_picks_by_cycle(inputs)
    if not cycles:
        return no_live_selection(
            inputs, ISSUE, needs="there is no traded book to price"
        )
    if inputs.fees <= 0.0 and inputs.slippage_bps <= 0.0:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "the run's configured cost model is already zero "
                "(simulation_fees=0, simulation.slippage_bps=0), so the "
                "zero-cost arm is identical to the baseline and the measured "
                f"drag would be a trivial 0.0 ({ISSUE})"
            ),
        )

    baseline = picks_arm(f"as-configured — {live_selection_label(inputs)}", cycles)
    ablated = picks_arm(
        "zero-cost (fees=0, slippage=0)", cycles, fees=0.0, slippage_bps=0.0
    )
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC = ReplaySpec(
    name="cost_adjusted_quality",
    module="behavioral",
    criticality="supporting",
    pattern="null_arm",
    issue=ISSUE,
    build_arms=build_arms,
)
