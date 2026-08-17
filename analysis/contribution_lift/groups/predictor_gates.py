"""predictor.{veto_gate_precision, output_distribution_gate,
direction_accuracy_vs_majority_baseline} — RC v3 T5 group `predictor_gates`
(alpha-engine-config-I7480).

Three predictor-tile components (criticality per crucible-evaluator's
``grading/tiles/predictor.py`` on main, read at authoring time — never
re-derived here):

* ``veto_gate_precision``                       — supporting  (line ~308)
* ``output_distribution_gate``                   — critical    (line ~268)
* ``direction_accuracy_vs_majority_baseline``    — supporting  (line ~384)

Only ``veto_gate_precision`` is measured; the other two are N/A by
construction, explained below.

── veto_gate_precision (null_arm) ──────────────────────────────────────────

The predictor's veto is the ``gbm_veto`` boolean — SINGLE-SOURCED per
crucible-predictor config#1815 (``inference/stages/write_output.py``: every
downstream veto surface, including the executor's block, reads this one
field; a display rule that diverged from it is exactly the bug #1815 fixed).
The executor applies it in ``executor/deciders.py`` (origin/main,
2026-08-16 read):

    L906-935:  if pred_data.get("gbm_veto"):
                   reason = f"GBM veto: α={predicted_alpha:.2%}, rank={...}"
                   plan.blocked.append({..., "block_reason": reason})
                   plan.risk_events.append({..., "rule": "predictor_gbm_veto", ...})

That ``block_reason`` string (prefix ``"GBM veto: "``) is what lands in
``executor_shadow_book`` (``executor/trade_logger.py``). This is also the
criterion ``analysis/veto_analysis.py`` (this repo) already documents as
"the" predictor veto: "predicted DOWN + high confidence" — i.e. a name the
predictor's alpha model would have entered on research's signal but
overrode, keyed on ``predicted_alpha`` (used for the veto's own α
threshold) and ``combined_rank``.

Two OTHER executor-side gates also originate from predictor fields
(``momentum_gate`` off ``predictor_momentum_veto``, and
``reversal_confirmation`` off ``momentum_confirmation``) but neither is
"the predictor veto" this component grades: both are RISK/TIMING rules
layered on top of the predictor's directional read, distinct rules with
their own ``block_reason`` prefixes ("momentum gate (predictor): ...",
"reversal-confirmation gate: ...") and their own tile surfaces. Only
``predictor_gbm_veto`` is the predictor overriding research's own ENTER
call on its own alpha/confidence — the thing ``veto_gate_precision``'s
docstring in the evaluator tile ("of the names the gate vetoed...") means.

Ablation design — PICKS AT MATCHED WIDTH, not a shadow-book replay of the
literal blocked orders:

    N(cycle)  = the live executed ENTER count that cycle (trades.db,
                trades WHERE action='ENTER', DISTINCT ticker, GROUP BY date)
    baseline  = top-N by predicted_alpha among ENTER-signalled names with
                gbm_veto falsy (the as-configured book)
    ablated   = top-N by predicted_alpha among ALL ENTER-signalled names,
                gbm_veto included (the gate disabled)
    width     = min(N, |non-vetoed candidates|) — the SAME width is used to
                cap the ablated arm's top-N, so both arms are exactly
                count-matched by construction on every cycle, never a
                down-select needed after the fact.

Why picks-at-matched-width rather than "baseline = executed orders,
ablated = executed + the vetoed names at their shadow-book price/size":
a gate's honest counterfactual is SAME-COHORT, same selection WIDTH — what
would this cycle's book have looked like at the identical size if the gate
had not filtered the ranking? Comparing the executed N against a WIDER
N+vetoed set structurally favors whichever arm holds more names (the
harness's own count-match rule forbids exactly this — contract §Rules,
``objective.check_count_match``). Picks-at-matched-width isolates the
gate's SELECTION effect (did the names it removed from the top-N belong
there?) from a sizing/breadth effect that has nothing to do with the gate.
Both arms are priced and sized by the harness's own equal-weight sizing
(``picks_to_orders``) at the SAME cost model, so the only degree of freedom
between the two arms is which names populate the top-N.

``trades_db_path`` is required only to read the live width N — not to price
or size anything (the harness's picks_arm/picks_to_orders path handles
that identically for both arms via ``ReplayInputs.price_matrix``). Absent
→ N/A-MISSING-INPUT (no live width to match against).

── output_distribution_gate (N/A-NOT-LIFT-SHAPED) ──────────────────────────

Confirmed against crucible-predictor (origin/main, 2026-08-16 read):
``model/output_distribution_gate.py`` + ``inference/stages/write_output.py``
compute ONE pass/fail boolean per inference run over the WHOLE distribution
of that cycle's predicted alphas (``validate_live_batch_distribution`` /
``validate_stratified_per_regime`` — shape checks: alpha range, per-regime
spread, etc.), written once at
``predictor/metrics/latest.json::output_distribution_gate`` — matching the
evaluator tile's own read (``grading/tiles/predictor.py`` L263-273:
``value=1.0 if passed else 0.0, n_samples=1``). There is no per-ticker
pass/fail anywhere in the predictions envelope or in
``executor_shadow_book.block_reason`` — the gate never blocks an individual
name; it blocks (or, today, only WARNS on — both blocking flags default
``False`` per ``config.py`` L414-431) the ENTIRE inference output as one
unit. A leave-one-out replay needs a per-ticker inclusion/exclusion to build
two arms that differ in WHICH names traded; a whole-distribution gate has
no such per-ticker degree of freedom to ablate — "ablating" it would mean
replaying with predictions from a run the gate actually failed, which
does not exist as a persisted counterfactual. Structurally not lift-shaped,
independent of data availability this cycle.

── direction_accuracy_vs_majority_baseline (N/A-NOT-LIFT-SHAPED) ──────────

This component (``grading/tiles/predictor.py`` L376-391) is itself already
an in-tile lift: directional classification accuracy over ALL resolved
predictions (crucible-backtester's ``predictor_outcomes`` table via
``analysis/predictor_confusion.py``) minus the trivial always-predict-the-
majority-class baseline. It is not a trading gate at all — nothing in
``executor/deciders.py`` blocks or vetoes an ENTER on directional
classification per se (the closest executor-side rule keyed on direction,
``gbm_veto``, is already graded by ``veto_gate_precision`` above); there is
no ``block_reason`` or persisted per-ticker pass/fail tied to "direction
accuracy". Because it scores over EVERY resolved prediction (not only the
subset that became ENTER picks), and because "ablating" it would mean
replaying without a direction model at all — leaving no predicted_direction
to rank picks by, i.e. no predictor to run — there is no leave-one-out arm
pair to construct. The component already computes its own baseline
comparison in-tile; a second baseline/ablated replay of the SAME
already-a-lift metric would be measuring the metric against itself.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import pandas as pd

from analysis.contribution_lift.harness import (
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    picks_arm,
)

logger = logging.getLogger(__name__)

ISSUE = "alpha-engine-config-I7480"

#: The signals.json per-ticker ``signal`` value that means "open a position"
#: (mirrors behavioral.cost_adjusted_quality._ENTER).
_ENTER = "ENTER"


def _enter_candidates(inputs: ReplayInputs, date: str) -> list[str]:
    """ENTER-signalled tickers for ``date``, per ``signals/{date}/signals.json``."""
    raw = (inputs.signals_by_date.get(date) or {}).get("signals") or {}
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return sorted({
        row["ticker"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("ticker"), str)
        and str(row.get("signal", "")).upper() == _ENTER
    })


def _live_enter_counts(trades_db_path: str) -> dict[str, int]:
    """``{date: distinct ticker count}`` of live executed ENTER orders.

    Same table/discriminator ``analysis/post_trade.py`` and
    ``analysis/shadow_book.py`` already read (``trades WHERE action='ENTER'``)
    — reused rather than re-derived. "no such table" is the only recoverable
    schema state (a fresh trades.db with no history yet); anything else
    propagates (contract §Rules: fail loud, no silent swallow).
    """
    conn = sqlite3.connect(trades_db_path)
    try:
        df = pd.read_sql_query(
            "SELECT date, ticker FROM trades WHERE action = 'ENTER'", conn,
        )
    except pd.errors.DatabaseError as exc:
        msg = str(exc).lower()
        if "no such table" in msg or "no such column" in msg:
            return {}
        raise
    finally:
        conn.close()
    if df.empty:
        return {}
    counts = df.groupby("date")["ticker"].nunique()
    return {str(d): int(n) for d, n in counts.items()}


def _rank_top_n(
    candidates: list[str], predictions: dict, n: int
) -> list[str]:
    """Top-``n`` tickers by ``predicted_alpha`` descending, ties by ticker asc.

    A candidate with no ``predicted_alpha`` on record (predictions.json
    absent/incomplete for that name-date) is excluded from the ranking
    rather than fabricated a score — it can still appear via the OTHER
    arm's candidate pool difference, which is exactly the effect being
    measured, but never via a made-up rank.
    """
    scored = []
    for ticker in candidates:
        pred = predictions.get(ticker) or {}
        alpha = pred.get("predicted_alpha")
        if isinstance(alpha, (int, float)):
            scored.append((ticker, float(alpha)))
    scored.sort(key=lambda row: (-row[1], row[0]))
    return sorted(t for t, _a in scored[:n])


def build_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    if not inputs.trades_db_path:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "ReplayInputs.trades_db_path is None — veto_gate_precision's "
                "picks-at-matched-width design needs the live executed ENTER "
                "count per cycle (trades.db, trades WHERE action='ENTER') to "
                "size the top-N comparison; nothing to replay without it "
                f"({ISSUE})"
            ),
        )
    if not Path(inputs.trades_db_path).exists():
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"trades.db not found at {inputs.trades_db_path!r} — cannot "
                f"read the live ENTER count per cycle ({ISSUE})"
            ),
        )

    live_counts = _live_enter_counts(inputs.trades_db_path)
    if not live_counts:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"trades.db at {inputs.trades_db_path!r} has no executed "
                "ENTER rows (table absent or empty) — no live width to "
                f"match the ablated arm against ({ISSUE})"
            ),
        )

    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    for date in inputs.dates:
        n_live = live_counts.get(date)
        if not n_live:
            continue
        candidates = _enter_candidates(inputs, date)
        if not candidates:
            continue
        predictions = inputs.predictions_by_date.get(date) or {}
        non_vetoed = [
            t for t in candidates
            if not (predictions.get(t) or {}).get("gbm_veto")
        ]
        width = min(n_live, len(non_vetoed))
        if width <= 0:
            continue
        baseline_picks = _rank_top_n(non_vetoed, predictions, width)
        # Same width caps the ablated arm — vetoed names are eligible, but
        # the comparison never grows wider than the as-configured book.
        ablated_picks = _rank_top_n(candidates, predictions, width)
        if len(baseline_picks) != width or len(ablated_picks) != width:
            # A candidate pool with fewer usable predicted_alpha scores than
            # `width` (e.g. predictions.json missing for some names) shrinks
            # BOTH rankings equally via _rank_top_n's own slice — but if that
            # leaves the two arms at different actual sizes for this cycle,
            # drop the cycle rather than let a partial-data cycle bias the
            # comparison. paired_diffs only pairs cycles present in both
            # arms, so this cycle simply contributes nothing.
            if len(baseline_picks) != len(ablated_picks):
                continue
        baseline_cycles.append({"date": date, "picks": baseline_picks})
        ablated_cycles.append({"date": date, "picks": ablated_picks})

    if not baseline_cycles:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                "no cycle in the replay window has both a live ENTER count "
                "and at least one ENTER-signalled candidate with a scored "
                f"predicted_alpha — nothing to rank ({ISSUE})"
            ),
        )

    baseline = picks_arm("as-configured (gbm_veto applied)", baseline_cycles)
    ablated = picks_arm("gbm_veto disabled (top-N incl. vetoed)", ablated_cycles)
    return ArmSet(baseline=baseline, ablated=ablated)


SPEC = ReplaySpec(
    name="veto_gate_precision",
    module="predictor",
    criticality="supporting",
    pattern="null_arm",
    issue=ISSUE,
    build_arms=build_arms,
)


def _output_distribution_gate_na(_inputs: ReplayInputs) -> NotAvailable:
    return NotAvailable(
        status="N/A-NOT-LIFT-SHAPED",
        reason=(
            "output_distribution_gate is a single pass/fail boolean over the "
            "WHOLE per-cycle output distribution "
            "(crucible-predictor model/output_distribution_gate.py, written "
            "once to predictor/metrics/latest.json — matches the evaluator "
            "tile's own n_samples=1 read at grading/tiles/predictor.py "
            "L263-273), never a per-ticker flag: no field in the predictions "
            "envelope or executor_shadow_book.block_reason names it. There is "
            "no per-ticker inclusion/exclusion to build a baseline/ablated "
            f"picks pair from ({ISSUE})"
        ),
    )


OUTPUT_DISTRIBUTION_GATE_SPEC = ReplaySpec(
    name="output_distribution_gate",
    module="predictor",
    criticality="critical",
    pattern="null_arm",
    issue=ISSUE,
    build_arms=_output_distribution_gate_na,
)


def _direction_accuracy_na(_inputs: ReplayInputs) -> NotAvailable:
    return NotAvailable(
        status="N/A-NOT-LIFT-SHAPED",
        reason=(
            "direction_accuracy_vs_majority_baseline is already an in-tile "
            "lift (directional accuracy over ALL resolved predictor_outcomes "
            "rows minus the always-predict-the-majority-class baseline; "
            "grading/tiles/predictor.py L376-391) — it is not a trading gate "
            "(no block_reason or persisted per-ticker pass/fail ties to it in "
            "executor_shadow_book), it scores every resolved prediction rather "
            "than only ENTER picks, and 'ablating' the direction head would "
            "leave no predicted_direction to rank picks by at all. No "
            f"leave-one-out arm pair exists to construct ({ISSUE})"
        ),
    )


DIRECTION_ACCURACY_SPEC = ReplaySpec(
    name="direction_accuracy_vs_majority_baseline",
    module="predictor",
    criticality="supporting",
    pattern="null_arm",
    issue=ISSUE,
    build_arms=_direction_accuracy_na,
)
