"""behavioral.cost_adjusted_quality — the cost-drag null arm (config-I7484).

Coverage table (spec §2, behavioral tile): ``cost_adjusted_quality`` has no
replay of any kind. The evaluator tile computes it in-tile, and that module's
own docstring says it lacks a baseline to compare against. Spec §4 group 7
files it standalone: "closest existing in-tile computation to a lift; needs a
baseline arm, not new machinery."

The baseline arm it needs is the zero-cost counterfactual. Both arms trade the
SAME names, on the same dates, in the same sizes — the ONLY difference is the
cost model:

* **baseline** — the live signals' ENTER picks at the configured
  ``simulation_fees`` / ``simulation.slippage_bps``.
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
    picks_arm,
)

ISSUE = "alpha-engine-config-I7484"

#: The signals.json per-ticker ``signal`` value that means "open a position".
_ENTER = "ENTER"


def _enter_picks(inputs: ReplayInputs) -> list[dict]:
    """``[{"date", "picks"}]`` — the ENTER-signalled names per cycle.

    Reads the live selection exactly as ``loaders/signal_loader`` documents it:
    ``signals`` is a ``{ticker: row}`` mapping (legacy payloads emit a list),
    and the ENTER discriminator is ``row["signal"]``.
    """
    cycles: list[dict] = []
    for date in inputs.dates:
        raw = (inputs.signals_by_date.get(date) or {}).get("signals") or {}
        rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
        picks = sorted({
            row["ticker"]
            for row in rows
            if isinstance(row, dict)
            and isinstance(row.get("ticker"), str)
            and str(row.get("signal", "")).upper() == _ENTER
        })
        if picks:
            cycles.append({"date": date, "picks": picks})
    return cycles


def build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    cycles = _enter_picks(inputs)
    if not cycles:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no cycle in the {len(inputs.dates)}-date window carries an "
                f"ENTER-signalled name in s3://{inputs.bucket}/signals/{{date}}/"
                f"signals.json — there is no traded book to price ({ISSUE})"
            ),
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

    baseline = picks_arm("as-configured", cycles)
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
