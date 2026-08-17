"""research.* diagnostics — the research-tile contribution replays (config-I7483).

Eight graded components on the ``research`` tile
(``crucible-evaluator/grading/tiles/research.py``). All eight use the spec §3
**substitution** pattern through PICKS: the baseline arm is the selection the
system ACTUALLY shipped that cycle (``signals.json`` rows carrying
``signal == "ENTER"``), and the ablated arm re-ranks THAT SAME cycle's signal
universe with the component's per-ticker contribution neutralised, then
down-selects to the baseline's own width so the two arms are count-matched by
construction.

Three of the eight have a live, per-ticker, PERSISTED input that the live
selection consumed, so they are measurable:

``momentum_regime_ic``
    The momentum pillar. ``factors/profiles/{date}/by_ticker.json`` is the
    universe board's own pillar substrate — verified 2026-08-16 against
    ``scanner/universe/2026-08-14/universe.json``, where the board's
    ``pillars`` block is byte-identical to the profile's ``*_score`` fields
    (``low_vol_score`` is the board's ``defensiveness``). Ablation: the
    momentum pillar is replaced by its cross-sectional mean, which makes its
    sector-neutral z identically zero and therefore removes exactly its
    contribution to the composite, leaving the other five pillars untouched.

``attractiveness_ic``
    The composite itself — the live champion feed's ranking signal
    (config-I2994). "Neutralising attractiveness" means every pillar sits at
    its cross-sectional mean, so the composite is constant across the whole
    universe and the selection carries NO ranking information; ties resolve by
    ticker ascending (the harness's determinism rule). The measured lift is
    therefore the value of the live ranking over an information-free draw from
    the same universe, which is precisely what a critical ranking metric
    should be asked to prove.

``macro_agent``
    The macro sector tilt, and the ONLY one of the eight whose ablation is
    exact rather than reconstructed. ``crucible-research``'s
    ``scoring/composite.py`` applies the tilt as an ADDITIVE point shift on
    the persisted per-ticker score::

        macro_shift = (sector_modifier - 1.0) / 0.30 * 10.0
        final_score = weighted_base + macro_shift + boosts   (clamped to [0, 100])

    Both terms are persisted in ``signals.json`` — ``score`` per ticker and
    ``sector_modifiers[sector]`` per cycle — so the ablated score is the
    persisted score with the shift subtracted back out. No re-derivation of
    the stock signal is involved.

The remaining five emit ``NotAvailable`` rather than a fabricated number:

* ``attractiveness_trajectory_ic`` — NOT-LIFT-SHAPED. Measured on the live
  board (2026-08-14): ``attractiveness_method`` is
  ``sector_neutral_zscore_percentile``, which carries no trajectory term. The
  repricing-trajectory substrate (config#1392) is pre-cutover observe-only
  evidence, so zeroing it cannot move a single live pick.
* ``thinktank_coverage_ic`` — NOT-LIFT-SHAPED. The evaluator's own component
  docstring states the Think Tank arm is an observe-only challenger shadow
  that "feeds no gate"; its shadow score is also absent from every artifact in
  ``ReplayInputs``.
* ``judge_outcome_ic`` — NOT-LIFT-SHAPED. Same shape: the evaluator declares
  it "VALIDATES (does not steer) the judge rubric layer — feeds no gate."
* ``judge_rubric_pass_rate`` — NOT-LIFT-SHAPED. A pass-rate over judge
  evaluations, with no per-ticker selection role at all.
* ``calibration_diagnostics`` — NOT-LIFT-SHAPED. An ECE over judge-vs-realized
  forecasts; a property OF the forecasts, never an input TO the selection.

Measurement caveat, stated once here and inherited by all three measurable
specs: the baseline arm is the live book, while the ablated arm is
reconstructed by re-ranking the cycle's own signal universe. For
``macro_agent`` the reconstruction is exact (the score is persisted). For the
two pillar specs the reconstruction is the equal-weighted sector-neutral
z-composite the live board declares (``pillar_weights`` all 1/6,
``attractiveness_method = sector_neutral_zscore_percentile``), so a residue of
reconstruction error rides along with the measured lift. It is bounded by the
narrowness of the pool: the ENTER-bearing cycles carry a signals universe of
~28 already-shortlisted names against ~25 ENTER picks, so at most a handful of
slots can differ. The arm labels say which arm is reconstructed.
"""

from __future__ import annotations

import logging
import math

from analysis.contribution_lift.harness import (
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    live_picks_by_date,
    live_selection_label,
    picks_arm,
)

logger = logging.getLogger(__name__)

ISSUE = "alpha-engine-config-I7483"

MODULE = "research"

#: The six pillar fields of ``factors/profiles/{date}/by_ticker.json``. Verified
#: 2026-08-16 to be the universe board's own ``pillars`` block verbatim, with
#: ``low_vol_score`` carrying the board's ``defensiveness``.
_PILLARS: tuple[str, ...] = (
    "quality_score",
    "momentum_score",
    "low_vol_score",
    "value_score",
    "growth_score",
    "stewardship_score",
)

#: Equal pillar weights — mirrors the live board's own ``pillar_weights``
#: (every pillar 0.166667 on ``scanner/universe/{date}/universe.json``). Equal
#: weights make the weight vector cancel out of the ranking entirely, so this
#: composite is invariant to the exact value as long as the live board keeps
#: them uniform. A non-uniform live weighting would need the board artifact,
#: which is not in ``ReplayInputs``.
_PILLAR_WEIGHT = 1.0 / len(_PILLARS)

#: Macro-shift parameters, mirroring ``crucible-research``
#: ``scoring/composite.py`` (``MACRO_MODIFIER_RANGE`` / ``MACRO_MAX_SHIFT_POINTS``).
#: Duplicated rather than imported because the two repos do not depend on each
#: other; a drift here shows up as a macro lift that stops matching the
#: producer, which is why the formula is spelled out in the module docstring.
_MACRO_MODIFIER_RANGE = 0.30
_MACRO_MAX_SHIFT_POINTS = 10.0

#: Tolerance for "this sector modifier is exactly neutral".
_FLAT = 1e-9


# --------------------------------------------------------------------------
# signals.json readers
# --------------------------------------------------------------------------


def _rows_by_ticker(inputs: ReplayInputs, date: str) -> dict[str, dict]:
    """``{ticker: row}`` for one cycle.

    ``signals`` is a ``{ticker: row}`` mapping in the modern payload and a list
    in legacy ones (``loaders/signal_loader`` documents both).
    """
    raw = (inputs.signals_by_date.get(date) or {}).get("signals") or {}
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    return {
        row["ticker"]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("ticker"), str)
    }


# The live book for a cycle comes from ``harness.live_picks_by_date``
# (config-I7501) — the executed ENTERs in trades.db, which is the ground truth
# in both the pre-champion and the champion era. The signals rows below are
# still read, but only for the SCORED UNIVERSE the ablated arm re-ranks.


# --------------------------------------------------------------------------
# The pillar composite (the reconstructed live ranker)
# --------------------------------------------------------------------------


def _sector_neutral_z(
    values: dict[str, float], sectors: dict[str, str]
) -> dict[str, float]:
    """Within-sector z-score of one pillar across the cycle's candidates.

    A sector with fewer than two members, or zero dispersion, contributes 0.0
    for every member: with one observation there is no cross-sectional signal
    to extract, and inventing one would put the whole ranking at the mercy of
    single-name sectors.
    """
    by_sector: dict[str, list[str]] = {}
    for ticker in values:
        by_sector.setdefault(sectors.get(ticker, "Unknown"), []).append(ticker)

    out: dict[str, float] = {}
    for members in by_sector.values():
        xs = [values[t] for t in members]
        n = len(xs)
        if n < 2:
            for t in members:
                out[t] = 0.0
            continue
        mean = sum(xs) / n
        var = sum((x - mean) ** 2 for x in xs) / n
        sd = math.sqrt(var)
        if sd <= 0.0:
            for t in members:
                out[t] = 0.0
            continue
        for t in members:
            out[t] = (values[t] - mean) / sd
    return out


def _pillar_composite(
    profiles: dict[str, dict],
    tickers: list[str],
    *,
    neutralised: tuple[str, ...] = (),
) -> dict[str, float]:
    """The live board's composite, with ``neutralised`` pillars flattened.

    Reproduces ``attractiveness_method = sector_neutral_zscore_percentile``:
    each pillar is z-scored within its sector across the cycle's candidate
    pool, then equal-weighted. The board's final percentile transform is
    monotone, so it is omitted — it cannot change the ranking, and every use
    here is a ranking.

    Replacing a pillar with its cross-sectional mean is implemented as forcing
    its z to 0.0 for every name, which is exactly what the substitution means:
    the pillar still occupies its weight, it simply carries no information.
    """
    sectors = {
        t: str((profiles.get(t) or {}).get("sector") or "Unknown") for t in tickers
    }
    composite = dict.fromkeys(tickers, 0.0)
    for pillar in _PILLARS:
        if pillar in neutralised:
            continue
        values = {t: float(profiles[t][pillar]) for t in tickers}
        for ticker, z in _sector_neutral_z(values, sectors).items():
            composite[ticker] += _PILLAR_WEIGHT * z
    return composite


def _profiled_candidates(
    rows: dict[str, dict], profiles: dict[str, dict]
) -> list[str]:
    """Cycle names carrying a COMPLETE pillar profile, ticker-ascending.

    A partial profile is dropped rather than mean-filled: a name scored on
    four pillars is not comparable to one scored on six, and filling the gap
    would manufacture the very cross-sectional signal being measured.
    """
    out: list[str] = []
    for ticker in sorted(rows):
        profile = profiles.get(ticker)
        if not isinstance(profile, dict):
            continue
        if all(isinstance(profile.get(p), (int, float)) for p in _PILLARS):
            out.append(ticker)
    return out


def _top_n(scores: dict[str, float], n: int) -> list[str]:
    """Highest-scoring ``n`` names; ties broken by ticker ascending."""
    return sorted(sorted(scores), key=lambda t: (-scores[t], t))[:n]


# --------------------------------------------------------------------------
# Pillar-substitution arms (momentum_regime_ic, attractiveness_ic)
# --------------------------------------------------------------------------


def _pillar_arms(
    inputs: ReplayInputs,
    *,
    neutralised: tuple[str, ...],
    ablated_label: str,
    what: str,
) -> ArmSet | NotAvailable:
    """Live ENTER picks vs the same universe re-ranked without ``neutralised``."""
    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    n_enter_cycles = 0
    n_profiled_cycles = 0

    live = live_picks_by_date(inputs)
    for date in inputs.dates:
        picks = list(live.get(date) or ())
        if not picks:
            continue
        rows = _rows_by_ticker(inputs, date)
        n_enter_cycles += 1
        profiles = inputs.pillar_profiles_by_date.get(date) or {}
        candidates = _profiled_candidates(rows, profiles)
        if not candidates:
            continue
        n_profiled_cycles += 1
        if len(candidates) < len(picks):
            # Cannot produce a count-matched ablated arm from a pool narrower
            # than the live book; the cycle is dropped from BOTH arms rather
            # than emitted at a different width (which the harness would then
            # correctly refuse to score as a lift).
            continue
        composite = _pillar_composite(profiles, candidates, neutralised=neutralised)
        baseline_cycles.append({"date": date, "picks": picks})
        ablated_cycles.append({"date": date, "picks": _top_n(composite, len(picks))})

    if not baseline_cycles:
        if n_enter_cycles == 0:
            reason = (
                f"no cycle in the {len(inputs.dates)}-date window carries a "
                f"priceable live selection ({live_selection_label(inputs)}), "
                f"so there is no live book to ablate {what} against ({ISSUE})"
            )
        elif n_profiled_cycles == 0:
            reason = (
                f"{n_enter_cycles} ENTER-bearing cycle(s) in the window, but none "
                f"has a complete 6-pillar profile in s3://{inputs.bucket}/factors/"
                f"profiles/{{date}}/by_ticker.json — {what} cannot be neutralised "
                f"without the pillar substrate the live board ranks on ({ISSUE})"
            )
        else:
            reason = (
                f"{n_profiled_cycles} cycle(s) carry both ENTER picks and pillar "
                f"profiles, but in every one the profiled pool is narrower than "
                f"the live book, so no count-matched ablated arm exists ({ISSUE})"
            )
        return NotAvailable(status="N/A-MISSING-INPUT", reason=reason)

    logger.info(
        "contribution_lift/%s: %d cycle(s) usable of %d ENTER-bearing",
        ablated_label, len(baseline_cycles), n_enter_cycles,
    )
    return ArmSet(
        baseline=picks_arm(
            f"as-configured — {live_selection_label(inputs)}", baseline_cycles
        ),
        ablated=picks_arm(ablated_label, ablated_cycles),
    )


def build_momentum_regime_ic_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Leave-one-out on the MOMENTUM pillar of the live board composite."""
    return _pillar_arms(
        inputs,
        neutralised=("momentum_score",),
        ablated_label=(
            "momentum pillar at cross-sectional mean "
            "(reconstructed sector-neutral pillar composite, 5 pillars live)"
        ),
        what="the momentum pillar",
    )


def build_attractiveness_ic_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Null-information arm: the whole attractiveness composite flattened."""
    return _pillar_arms(
        inputs,
        neutralised=_PILLARS,
        ablated_label=(
            "attractiveness at cross-sectional mean "
            "(information-free selection from the same universe, ties by ticker)"
        ),
        what="the attractiveness composite",
    )


# --------------------------------------------------------------------------
# macro_agent — the exact additive-tilt ablation
# --------------------------------------------------------------------------


def _macro_shift(modifier: float) -> float:
    """``(modifier - 1.0) / range * max_shift`` — crucible-research composite.py."""
    return (
        (float(modifier) - 1.0) / _MACRO_MODIFIER_RANGE * _MACRO_MAX_SHIFT_POINTS
    )


def build_macro_agent_arms(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Live ENTER picks vs the same universe re-ranked with the tilt removed.

    Only cycles whose ``sector_modifiers`` are non-flat are admitted. On a
    cycle where every modifier is exactly 1.0 the macro shift is 0.0 for every
    name, the ablation is a provable no-op, and including the cycle would fold
    pure baseline-reconstruction noise into a number labelled "macro
    contribution". Measured 2026-08-16: modifiers span [0.80, 1.25] on
    pre-2026-07-25 cycles and are identically 1.0 from the signals_envelope v1
    (no-agent producer) cutover onward.
    """
    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    n_enter_cycles = 0
    n_tilted_cycles = 0

    live = live_picks_by_date(inputs)
    for date in inputs.dates:
        picks = list(live.get(date) or ())
        if not picks:
            continue
        rows = _rows_by_ticker(inputs, date)
        n_enter_cycles += 1
        modifiers = (inputs.signals_by_date.get(date) or {}).get(
            "sector_modifiers"
        ) or {}
        tilts = {
            str(sector): float(value)
            for sector, value in modifiers.items()
            if isinstance(value, (int, float))
        }
        if not any(abs(v - 1.0) > _FLAT for v in tilts.values()):
            continue
        n_tilted_cycles += 1

        untilted: dict[str, float] = {}
        for ticker, row in rows.items():
            score = row.get("score")
            if not isinstance(score, (int, float)):
                continue
            sector = str(row.get("sector") or "Unknown")
            untilted[ticker] = float(score) - _macro_shift(tilts.get(sector, 1.0))
        if len(untilted) < len(picks):
            continue

        baseline_cycles.append({"date": date, "picks": picks})
        ablated_cycles.append({"date": date, "picks": _top_n(untilted, len(picks))})

    if not baseline_cycles:
        if n_enter_cycles == 0:
            reason = (
                f"no cycle in the {len(inputs.dates)}-date window carries a "
                f"priceable live selection ({live_selection_label(inputs)}) — "
                f"no live book to remove the macro tilt from ({ISSUE})"
            )
        elif n_tilted_cycles == 0:
            reason = (
                f"every one of the {n_enter_cycles} ENTER-bearing cycle(s) has "
                f"sector_modifiers identically 1.0 in s3://{inputs.bucket}/signals/"
                f"{{date}}/signals.json (signals_envelope v1 no-agent producer), so "
                f"the macro shift is already 0.0 for every name and the ablation "
                f"is a provable no-op rather than a measurement ({ISSUE})"
            )
        else:
            reason = (
                f"{n_tilted_cycles} tilted cycle(s) found, but none carries a "
                f"numeric per-ticker `score` for at least as many names as the "
                f"live book, so no count-matched ablated arm exists ({ISSUE})"
            )
        return NotAvailable(status="N/A-MISSING-INPUT", reason=reason)

    logger.info(
        "contribution_lift/macro_agent: %d tilted cycle(s) usable of %d "
        "ENTER-bearing", len(baseline_cycles), n_enter_cycles,
    )
    return ArmSet(
        baseline=picks_arm(
            f"as-configured — {live_selection_label(inputs)}", baseline_cycles
        ),
        ablated=picks_arm(
            "macro sector tilt removed (score - macro_shift, tilt set to 1.0)",
            ablated_cycles,
        ),
    )


# --------------------------------------------------------------------------
# The five components with no selection role
# --------------------------------------------------------------------------


def _not_lift_shaped(reason: str):
    def build_arms(_inputs: ReplayInputs) -> NotAvailable:
        return NotAvailable(status="N/A-NOT-LIFT-SHAPED", reason=reason)

    return build_arms


build_thinktank_coverage_ic_arms = _not_lift_shaped(
    "thinktank_coverage_ic grades an OBSERVE-ONLY challenger shadow score "
    "(config-I2994); crucible-evaluator grading/tiles/research.py states it "
    "'feeds no gate', so neutralising it cannot move a single live pick and "
    "the leave-one-out lift is 0.0 by construction, not by measurement. The "
    "shadow score is also absent from every ReplayInputs artifact — it would "
    f"need s3://{{bucket}}/thinktank/ratings/ loaded per ticker per cycle "
    f"before it could be ablated even if it were promoted to the live feed "
    f"({ISSUE})"
)

build_judge_outcome_ic_arms = _not_lift_shaped(
    "judge_outcome_ic is observability OF THE JUDGE: crucible-evaluator "
    "grading/tiles/research.py states it 'VALIDATES (does not steer) the judge "
    "rubric layer — feeds no gate'. The judge quality-score is not a selection "
    "input, so there is no pick it can be removed from; its per-ticker scores "
    f"are also not persisted in any ReplayInputs artifact ({ISSUE})"
)

build_judge_rubric_pass_rate_arms = _not_lift_shaped(
    "judge_rubric_pass_rate is a coverage/pass-rate over judge evaluations "
    "(backtest/{date}/agent_quality.json, research agent-quality producer "
    "config#1149). It has no per-ticker value and no selection role, so no "
    "substitution of it changes any pick — a pass-rate is graded as a process "
    f"property, never as a marginal contribution to portfolio alpha ({ISSUE})"
)

build_calibration_diagnostics_arms = _not_lift_shaped(
    "calibration_diagnostics is the judge-vs-realized ECE over "
    "backtest/{date}/portfolio_calibration.json — a property OF the forecasts, "
    "not an input TO the selection. Nothing consumes the ECE when picks are "
    "made, so ablating it is a no-op on the book; the honest read is the "
    f"calibration number itself, which the evaluator tile already grades ({ISSUE})"
)

build_attractiveness_trajectory_ic_arms = _not_lift_shaped(
    "attractiveness_trajectory_ic grades the PRE-repricing trajectory score as "
    "observe-only evidence for the config#1392 cutover, which has NOT been "
    "made: the live board at s3://{bucket}/scanner/universe/{date}/universe.json "
    "declares attractiveness_method = 'sector_neutral_zscore_percentile' "
    "(measured 2026-08-16), a pure point-in-time pillar composite with no "
    "trajectory term. Zeroing the trajectory therefore cannot move a live pick. "
    "The substrate (pre_repricing_score / attr_slope_z under "
    "s3://{bucket}/scanner/universe/trajectory/{date}/trajectory.json) is also "
    "not loaded into ReplayInputs, so this becomes measurable only when the "
    f"cutover lands AND that artifact is added to the loader ({ISSUE})"
)


# --------------------------------------------------------------------------
# Specs — criticality mirrors the evaluator tile record verbatim
# --------------------------------------------------------------------------


def _spec(name: str, criticality: str, build_arms) -> ReplaySpec:
    return ReplaySpec(
        name=name,
        module=MODULE,
        criticality=criticality,
        pattern="substitution",
        issue=ISSUE,
        build_arms=build_arms,
    )


SPECS: list[ReplaySpec] = [
    _spec("thinktank_coverage_ic", "diagnostic", build_thinktank_coverage_ic_arms),
    _spec("macro_agent", "supporting", build_macro_agent_arms),
    _spec("calibration_diagnostics", "supporting", build_calibration_diagnostics_arms),
    _spec("momentum_regime_ic", "diagnostic", build_momentum_regime_ic_arms),
    _spec("attractiveness_ic", "critical", build_attractiveness_ic_arms),
    _spec(
        "attractiveness_trajectory_ic",
        "diagnostic",
        build_attractiveness_trajectory_ic_arms,
    ),
    _spec("judge_outcome_ic", "diagnostic", build_judge_outcome_ic_arms),
    _spec("judge_rubric_pass_rate", "supporting", build_judge_rubric_pass_rate_arms),
]
