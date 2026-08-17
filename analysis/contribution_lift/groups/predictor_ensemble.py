"""predictor.* — the leave-one-L1-out replay group (config-I7479).

Coverage table (spec §2, predictor tile): ``meta_l2_ic``, ``momentum_l1_ic``,
``volatility_l1_ic`` and ``research_calibrator_l1_ic`` have **no replay of any
kind**, and ``ensemble_lift_over_best_l1`` has only an in-tile IC-space
subtraction (``L2 CPCV IC − best standalone L1 IC``) with no simulator
involvement. This module gives all five an objective-unit arm pair.

The whole group is the **substitution** pattern: every arm selects a
count-matched basket on the same cycle dates from the same candidate set, and
the arms differ ONLY in the score they rank by. Nothing is gated off, so
nothing changes the width of the book — count matching holds by construction
(each arm is ``top-N`` of one shared candidate list).

What is actually persisted (measured 2026-08-16 against live S3)
----------------------------------------------------------------
``predictor/predictions/{date}.json`` carries, per ticker, the L2 output AND
three of the L2's own input features — the three L1 model outputs — under
domain names rather than an ``l1``/``base_models`` block:

===========================  ==========================  ====================
L1 model                     META_FEATURES name          persisted key
===========================  ==========================  ====================
momentum (deterministic)     ``momentum_score``          ``momentum_confirmation``
volatility (GBM)             ``expected_move``           ``expected_move``
research calibrator (GBM)    ``research_calibrator_prob``  ``research_calibrator_prob``
===========================  ==========================  ====================

The ``momentum_score`` → ``momentum_confirmation`` rename is the non-obvious
one: ``crucible-predictor/inference/stages/run_inference.py:950`` writes
``"momentum_confirmation": round(momentum_score, 6)`` — the same scalar the
ridge consumes at line 786, rounded to 6dp.

``predictor/weights/meta/manifest.json`` carries the L2 combination rule:
``meta_coefficients`` (the fitted ``BayesianRidge`` ``coef_`` plus
``intercept``, one entry per ``META_FEATURES`` column) and
``meta_l1_standalone_alpha_ic`` (each L1 output's own cross-sectional IC
against the SAME signed-alpha label the L2 targets — the apples-to-apples set
the evaluator tile already uses for ``ensemble_lift_over_best_l1``).

The other 13 ridge inputs (``research_composite_score``,
``research_conviction``, ``sector_macro_modifier``, the six ``macro_*``, and
``regime_intensity_z``) are computed at ``run_inference.py:784-791`` and never
written to the prediction row. That does NOT block this replay, because the
ablation never reconstructs the score from scratch — see below.

How a leave-one-L1-out arm is built
-----------------------------------
The L2 is a **linear** model, so dropping one term from its score is exactly
subtracting that term::

    s_full(t)      = predicted_alpha_raw(t)            # the ridge's own output
    s_without_X(t) = s_full(t) − coef[X] · x_X(t)

which needs ONLY ``coef[X]`` and ``x_X(t)`` — both persisted — and never the
11 features that are not. The arm then re-ranks the candidate set by
``s_without_X`` and takes the same N names.

Three properties make this exact rather than approximate:

* **Per-date constants cancel.** The intercept, the six macro features and
  ``regime_intensity_z`` are market-wide: one shared value per date, zero
  cross-sectional variance. A within-date ranking is invariant to them, so
  their absence from the prediction row costs nothing.
* **Level neutralization cancels.** ``predicted_alpha`` is
  ``predicted_alpha_raw`` minus that date's cross-sectional mean
  (``inference/level_neutralization.py``); subtracting a per-date constant is
  rank-preserving, so either field yields the same ablated ordering. ``_raw``
  is preferred and ``predicted_alpha`` is the documented fallback for
  pre-neutralization prediction files.
* **The ablation applies the SAME transform the ridge was fitted through.**
  ``MetaModel.fit`` applies an optional standardize+winsorize to the
  *directional* columns (``META_STANDARDIZE_ENABLED``); when it is on,
  ``coef_`` multiplies ``clip((x−μ)/σ, ±w)`` and subtracting ``coef·x_raw``
  would be removing the wrong quantity.

  μ/σ/w are now read from ``meta_scaler`` in ``predictor/weights/meta/
  manifest.json`` (alpha-engine-config-I7502, ``crucible-predictor-PR508``),
  written unconditionally on every training run. Before that they lived only
  inside ``meta_model.pkl``, with the ``meta_model.pkl.meta.json`` sidecar as
  their nominal transport — and that sidecar had been **frozen since
  2026-05-30** beside a 2026-08-15 pickle, because
  ``training/meta_trainer.py`` set ``gate_passed = promoted`` with
  ``promoted = False`` unconditionally after the challenger-first cutover,
  making its upload branch dead code. This module's response was to DETECT
  the transform and refuse (alpha-engine-config-I7511 is the correction):
  refusing was right against the old contract and is wrong against the new
  one, because the numbers it was missing are now present.

  The refusal survives for one case only — a feature the manifest's own
  ``feature_std`` says was standardized but which ``meta_scaler`` does not
  cover. Reconstructing that would mean guessing μ/σ, and a fabricated
  ablation is worse than an ``N/A``.

Two arms do not reconstruct anything and are unaffected by all of the above:

* ``meta_l2_ic`` — the **predictor-off** arm. Baseline ranks the candidates by
  the L2's ``predicted_alpha``; the ablated arm ranks the SAME candidates by
  the research composite ``score`` from ``signals/{date}/signals.json``, i.e.
  the ordering the pipeline would have had if the predictor had never run.
  That is spec §2's own words for this gap ("no 'predictor off / L1-only'
  replay against the objective").
* ``ensemble_lift_over_best_l1`` — the ablated arm ranks by the single best
  standalone L1, chosen deterministically as the ``argmax`` of the manifest's
  ``meta_l1_standalone_alpha_ic[·].xsec_ic`` over the evaluator tile's own
  directional-L1 set, ties broken by feature name. This is the same question
  the in-tile arithmetic asks, moved out of IC space and into the objective.

Cycle width
-----------
``N`` per cycle is the number of ENTER-signalled names in that cycle's
``signals.json`` — the width the live book actually opened — capped by the
number of usable candidates. Every arm of every component takes ``top-N`` of
one shared candidate list, so ``picks_per_cycle`` is identical across arms on
every cycle by construction.

Stated assumption (carried into every ablated arm's label)
----------------------------------------------------------
``meta_coefficients`` is a single snapshot of the CURRENTLY promoted model.
The predictor retrains weekly and no per-date coefficient history is joinable
from the prediction row (it carries ``meta_model_version: "v3.0"``, not the
manifest's ``served_version``). So the historical cycles are ablated with
today's coefficients. The alternative — refusing to measure at all — would
leave five components with no objective-unit reading for the sake of a
second-order drift in one multiplier, which is the wrong trade; the
approximation is named in the arm label so it is visible on the artifact.
"""

from __future__ import annotations

import json
import logging

from analysis.contribution_lift.harness import (
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    live_widths,
    picks_arm,
)

logger = logging.getLogger(__name__)

ISSUE = "alpha-engine-config-I7479"

#: The L2's own combination rule + per-L1 standalone diagnostics.
MANIFEST_KEY = "predictor/weights/meta/manifest.json"

#: The score every baseline arm ranks by, most-preferred first. ``_raw`` is the
#: ridge's own output; ``predicted_alpha`` is the level-neutralized field and
#: differs from it only by that date's cross-sectional mean, which cannot
#: change a within-date ranking.
_ALPHA_FIELDS = ("predicted_alpha_raw", "predicted_alpha")

#: ``META_FEATURES`` name → the key it is persisted under on a prediction row.
#: The momentum entry is the rename documented in the module docstring.
_PERSISTED_FEATURE_KEY = {
    "research_calibrator_prob": "research_calibrator_prob",
    "momentum_score": "momentum_confirmation",
    "expected_move": "expected_move",
}

#: The evaluator tile's own directional-L1 set (``grading/tiles/predictor.py``
#: ``_DIRECTIONAL_L1_FEATURES``), intersected with what is persisted per
#: ticker. ``"momentum"`` in the tile's tuple is an alias with no manifest
#: entry and no prediction-row key, so it drops out here.
_BEST_L1_CANDIDATES = ("expected_move", "research_calibrator_prob", "momentum_score")

#: A directional column that WAS standardized reads a post-transform std of
#: ~1.0 (slightly under, because of the ±3σ winsorize). A raw column reads its
#: natural scale. Anything inside this band of 1.0 is treated as standardized
#: and refused. Measured 2026-08-16 on the live manifest: 0.0145
#: (research_calibrator_prob), 0.0791 (momentum_score), 0.0106 (expected_move)
#: — all decisively raw.
_STD_BAND = 0.15


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def _load_manifest(inputs: ReplayInputs) -> dict | None:
    """Read ``predictor/weights/meta/manifest.json``, or ``None`` if absent.

    Deliberately uncached: five specs each read one small JSON object once per
    weekly run, and a module-level cache would make ``build_arms`` stateful
    across runs and across tests for no measurable saving.

    Fails LOUD on anything except a missing object. A permissions error or a
    truncated body must surface as an errored module, never as five components
    quietly reporting "the predictor persists nothing".
    """
    if inputs.s3_client is None:
        return None
    try:
        body = inputs.s3_client.get_object(Bucket=inputs.bucket, Key=MANIFEST_KEY)
    except Exception as exc:  # noqa: BLE001 - re-raised unless it is a 404
        if _is_missing_key(exc):
            return None
        raise
    manifest = json.loads(body["Body"].read())
    if not isinstance(manifest, dict):
        raise TypeError(
            f"s3://{inputs.bucket}/{MANIFEST_KEY} parsed to "
            f"{type(manifest).__name__}, expected a JSON object"
        )
    return manifest


def _is_missing_key(exc: Exception) -> bool:
    """True for an S3 'this object does not exist' error, false for anything else."""
    code = getattr(exc, "response", {}).get("Error", {}).get("Code")
    return code in ("NoSuchKey", "404", "NoSuchBucket")


def _standardized_features(manifest: dict) -> set[str]:
    """Feature names whose ridge coefficient multiplies a STANDARDIZED column.

    ``MetaModel._compute_importance`` runs on the matrix the coefficients were
    fitted against — i.e. AFTER ``_apply_scaler`` — so
    ``models.meta_model.importance.feature_std`` is the definitive read on
    whether a column was transformed: ~1.0 means standardized, its natural
    scale means raw. This is the only surviving evidence, since the scaler
    itself is persisted nowhere readable (see the module docstring).

    A feature with no ``feature_std`` entry is reported as standardized —
    "cannot tell" and "unusable" are the same answer here, and the caller
    turns either into an ``N/A`` naming the missing input.
    """
    importance = ((manifest.get("models") or {}).get("meta_model") or {}).get(
        "importance"
    ) or {}
    feature_std = importance.get("feature_std")
    if not isinstance(feature_std, dict):
        return set(_PERSISTED_FEATURE_KEY)
    suspect: set[str] = set()
    for feature in _PERSISTED_FEATURE_KEY:
        std = feature_std.get(feature)
        if not isinstance(std, (int, float)) or abs(float(std) - 1.0) <= _STD_BAND:
            suspect.add(feature)
    return suspect


def _meta_scaler(manifest: dict) -> dict | None:
    """The persisted directional standardize+winsorize transform, or ``None``.

    Written unconditionally by every training run since
    ``crucible-predictor-PR508`` (alpha-engine-config-I7502). Shape, from
    ``model/meta_model.py::MetaModel._build_scaler``::

        {"directional": [name, ...], "mean": {name: mu},
         "std": {name: sigma}, "winsor": w}

    ``None`` for a legacy or unstandardized model — which is not an error and
    not a gap: with no transform fitted, the raw column IS what the ridge
    multiplied, so the identity is correct and the ablation is exact.
    """
    scaler = manifest.get("meta_scaler")
    if not isinstance(scaler, dict):
        return None
    if not isinstance(scaler.get("directional"), list):
        raise TypeError(
            f"{MANIFEST_KEY}: meta_scaler is present but its 'directional' key "
            f"is {type(scaler.get('directional')).__name__}, expected a list — "
            "a malformed scaler must not be silently treated as absent, which "
            "would reconstruct the ablation against the wrong column space"
        )
    return scaler


def _scaled(scaler: dict | None, feature: str, x: float) -> float:
    """``x`` as the fitted ridge saw it — ``clip((x-mu)/sigma, +-w)``.

    Mirrors ``MetaModel._apply_scaler`` exactly, including its degenerate-sigma
    rule (a column whose fitted std was ~0 is recorded as 1.0, so the transform
    is the identity for it rather than a divide-by-tiny-sigma amplification).
    An untransformed feature returns unchanged.
    """
    if scaler is None or feature not in (scaler.get("directional") or []):
        return float(x)
    mu = float((scaler.get("mean") or {}).get(feature, 0.0))
    sigma = float((scaler.get("std") or {}).get(feature, 1.0)) or 1.0
    w = float(scaler.get("winsor", 3.0))
    return float(min(max((float(x) - mu) / sigma, -w), w))


# --------------------------------------------------------------------------
# Per-cycle candidate construction
# --------------------------------------------------------------------------


def _alpha_of(row: dict) -> float | None:
    for field in _ALPHA_FIELDS:
        value = row.get(field)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _enter_count(inputs: ReplayInputs, date: str) -> int:
    """How many names the live book opened on ``date`` — the arm width.

    From ``harness.live_widths`` (config-I7501): the executed ENTERs in
    trades.db, which is the width in both the pre-champion and the champion
    era. The signals feed carried the width only until 2026-07-13, when
    ``scanner_predictor_direct`` moved entry selection out of it.
    """
    return live_widths(inputs).get(date, 0)


def _signal_scores(inputs: ReplayInputs, date: str) -> dict[str, float]:
    """``{ticker: research composite score}`` from that cycle's signals.json."""
    raw = (inputs.signals_by_date.get(date) or {}).get("signals") or {}
    rows = list(raw.values()) if isinstance(raw, dict) else list(raw)
    scores: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker, score = row.get("ticker"), row.get("score")
        if isinstance(ticker, str) and isinstance(score, (int, float)):
            scores[ticker] = float(score)
    return scores


def _clip_suspect(alphas: list[float]) -> bool:
    """True when this cycle's alphas look like they hit the ridge clip bound.

    ``run_inference.py`` clips the ridge output to ``±MAX_PREDICTED_RETURN``
    before it is written. Clipping is monotone, so it never disturbs a ranking
    BY the alpha — but it destroys the additive identity the leave-one-out arms
    rely on, because a clipped ``s_full`` is no longer ``Σ coef·x``. The bound
    itself is a config constant that is not persisted, so it is detected
    instead: two tickers landing on a byte-identical extreme of a 6dp
    continuous score is the clip, not a coincidence.
    """
    if len(alphas) < 2:
        return False
    return alphas.count(max(alphas)) > 1 or alphas.count(min(alphas)) > 1


def _candidates(
    inputs: ReplayInputs, date: str, *, require: str | None = None
) -> list[tuple[str, float, float]]:
    """``[(ticker, s_full, x_require)]`` for one cycle, sorted by ticker.

    ``require`` names a ``META_FEATURES`` column that must ALSO be present and
    numeric on the row; tickers missing it are excluded from every arm, so the
    baseline and the ablated arm always rank the identical candidate list.
    ``x_require`` is ``0.0`` when no feature is required.
    """
    rows = inputs.predictions_by_date.get(date) or {}
    out: list[tuple[str, float, float]] = []
    for ticker in sorted(rows):
        row = rows[ticker]
        if not isinstance(row, dict):
            continue
        alpha = _alpha_of(row)
        if alpha is None:
            continue
        feature_value = 0.0
        if require is not None:
            raw = row.get(_PERSISTED_FEATURE_KEY[require])
            if not isinstance(raw, (int, float)):
                continue
            feature_value = float(raw)
        out.append((ticker, alpha, feature_value))
    return out


def _top_n(scored: list[tuple[str, float]], n: int) -> list[str]:
    """The ``n`` highest-scoring tickers, ties broken by ticker ascending."""
    ranked = sorted(scored, key=lambda row: (-row[1], row[0]))
    return sorted(t for t, _s in ranked[:n])


def _cycle_width(inputs: ReplayInputs, date: str, n_candidates: int) -> int:
    return min(_enter_count(inputs, date), n_candidates)


# --------------------------------------------------------------------------
# Arm builders
# --------------------------------------------------------------------------


def _no_cycles(what: str) -> NotAvailable:
    return NotAvailable(
        status="N/A-MISSING-INPUT",
        reason=(
            f"no cycle in the window produced a count-matched pair of arms for "
            f"{what} — every cycle lacked either a live executed width "
            f"(trades.db ENTER) or a usable "
            f"predictor/predictions/{{date}}.json candidate set ({ISSUE})"
        ),
    )


def _drop_one_l1(inputs: ReplayInputs, feature: str) -> ArmSet | NotAvailable:
    """Baseline = rank by the L2; ablated = rank by the L2 minus ``feature``'s term."""
    if not inputs.predictions_by_date:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no cycle in the window has a "
                f"s3://{inputs.bucket}/predictor/predictions/{{date}}.json "
                f"body, so there is no L2 score to ablate ({ISSUE})"
            ),
        )

    manifest = _load_manifest(inputs)
    if manifest is None:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"s3://{inputs.bucket}/{MANIFEST_KEY} is absent, so the L2's "
                f"combination rule (meta_coefficients) cannot be read and the "
                f"'{feature}' term cannot be removed from the ridge score ({ISSUE})"
            ),
        )

    coefficients = manifest.get("meta_coefficients")
    if not isinstance(coefficients, dict) or not isinstance(
        coefficients.get(feature), (int, float)
    ):
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"meta_coefficients['{feature}'] absent from "
                f"s3://{inputs.bucket}/{MANIFEST_KEY} — the persisted L2 "
                f"combination rule does not carry a coefficient for this L1, so "
                f"its term cannot be subtracted ({ISSUE})"
            ),
        )
    coefficient = float(coefficients[feature])

    scaler = _meta_scaler(manifest)
    # The ONE case that still cannot be reconstructed: the manifest's own
    # feature_std says this column was standardized, but meta_scaler does not
    # carry it. Reconstructing would mean guessing mu/sigma, and a fabricated
    # ablation reads exactly like a measured one (alpha-engine-config-I7511).
    if feature in _standardized_features(manifest) and (
        scaler is None or feature not in (scaler.get("directional") or [])
    ):
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"the served ridge is fitted on a STANDARDIZED '{feature}' "
                f"column (models.meta_model.importance.feature_std ~= 1.0 in "
                f"{MANIFEST_KEY}), so its coefficient multiplies "
                f"clip((x-mu)/sigma, +-w) — but meta_scaler "
                f"{'is absent from the manifest' if scaler is None else 'does not list this column'}, "
                f"so mu/sigma are unknown and the term cannot be removed "
                f"without inventing them ({ISSUE})"
            ),
        )

    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    n_clipped = 0
    for date in inputs.dates:
        candidates = _candidates(inputs, date, require=feature)
        width = _cycle_width(inputs, date, len(candidates))
        if width <= 0:
            continue
        if _clip_suspect([alpha for _t, alpha, _x in candidates]):
            n_clipped += 1
            continue
        baseline_cycles.append(
            {"date": date, "picks": _top_n([(t, a) for t, a, _x in candidates], width)}
        )
        ablated_cycles.append({
            "date": date,
            "picks": _top_n(
                # coef multiplies the column AS FITTED, so the term removed is
                # coef * scaled(x) — identical to coef * x when no transform
                # was fitted (alpha-engine-config-I7511).
                [(t, a - coefficient * _scaled(scaler, feature, x))
                 for t, a, x in candidates],
                width,
            ),
        })

    if n_clipped:
        logger.warning(
            "contribution_lift/%s: dropped %d cycle(s) whose predicted_alpha hit "
            "the ridge clip bound — the additive leave-one-out identity does not "
            "hold on a clipped score",
            feature, n_clipped,
        )
    if not baseline_cycles:
        return _no_cycles(f"the '{feature}' leave-one-out")

    return ArmSet(
        baseline=picks_arm("as-configured (L2 predicted_alpha)", baseline_cycles),
        ablated=picks_arm(
            f"L2 with the {feature} term removed "
            f"(coef={coefficient:+.6g}, from the currently promoted manifest; "
            f"{'standardized via meta_scaler' if scaler and feature in (scaler.get('directional') or []) else 'raw column, no transform fitted'})",
            ablated_cycles,
        ),
    )


def _build_meta_l2(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Baseline = the L2's ranking; ablated = the research ranking, predictor off."""
    if not inputs.predictions_by_date:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no cycle in the window has a "
                f"s3://{inputs.bucket}/predictor/predictions/{{date}}.json body — "
                f"there is no L2 ranking to compare a predictor-off arm against "
                f"({ISSUE})"
            ),
        )

    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    for date in inputs.dates:
        scores = _signal_scores(inputs, date)
        candidates = [
            (ticker, alpha, scores[ticker])
            for ticker, alpha, _x in _candidates(inputs, date)
            if ticker in scores
        ]
        width = _cycle_width(inputs, date, len(candidates))
        if width <= 0:
            continue
        baseline_cycles.append(
            {"date": date, "picks": _top_n([(t, a) for t, a, _s in candidates], width)}
        )
        ablated_cycles.append(
            {"date": date, "picks": _top_n([(t, s) for t, _a, s in candidates], width)}
        )

    if not baseline_cycles:
        return _no_cycles(
            "the predictor-off arm (no cycle had both a predicted name and its "
            "research composite score)"
        )

    return ArmSet(
        baseline=picks_arm("as-configured (L2 predicted_alpha)", baseline_cycles),
        ablated=picks_arm(
            "predictor off (ranked by the research composite score in signals.json)",
            ablated_cycles,
        ),
    )


def _build_ensemble_lift(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    """Baseline = the stacked L2; ablated = the single best standalone L1."""
    if not inputs.predictions_by_date:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"no cycle in the window has a "
                f"s3://{inputs.bucket}/predictor/predictions/{{date}}.json body — "
                f"there is no stacked ranking to compare a best-single-L1 arm "
                f"against ({ISSUE})"
            ),
        )

    manifest = _load_manifest(inputs)
    if manifest is None:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"s3://{inputs.bucket}/{MANIFEST_KEY} is absent, so "
                f"meta_l1_standalone_alpha_ic cannot be read and 'the best "
                f"single L1' cannot be chosen deterministically ({ISSUE})"
            ),
        )

    standalone = manifest.get("meta_l1_standalone_alpha_ic")
    ics: dict[str, float] = {}
    if isinstance(standalone, dict) and standalone.get("status") not in (
        "not_run",
        "error",
    ):
        for feature in _BEST_L1_CANDIDATES:
            entry = standalone.get(feature)
            if isinstance(entry, dict) and isinstance(
                entry.get("xsec_ic"), (int, float)
            ):
                ics[feature] = float(entry["xsec_ic"])
    if not ics:
        return NotAvailable(
            status="N/A-MISSING-INPUT",
            reason=(
                f"meta_l1_standalone_alpha_ic in s3://{inputs.bucket}/"
                f"{MANIFEST_KEY} carries no directional-L1 xsec_ic entry "
                f"({', '.join(_BEST_L1_CANDIDATES)}), so 'the best single L1' "
                f"has no deterministic definition this cycle. Refusing to fall "
                f"back to the volatility MAGNITUDE walk-forward IC — that is "
                f"the config-I1062 false-RED ({ISSUE})"
            ),
        )

    # argmax over the signed-alpha ICs, ties broken by feature name so the
    # chosen arm is identical run to run.
    best = min(ics, key=lambda feature: (-ics[feature], feature))

    baseline_cycles: list[dict] = []
    ablated_cycles: list[dict] = []
    for date in inputs.dates:
        candidates = _candidates(inputs, date, require=best)
        width = _cycle_width(inputs, date, len(candidates))
        if width <= 0:
            continue
        baseline_cycles.append(
            {"date": date, "picks": _top_n([(t, a) for t, a, _x in candidates], width)}
        )
        ablated_cycles.append(
            {"date": date, "picks": _top_n([(t, x) for t, _a, x in candidates], width)}
        )

    if not baseline_cycles:
        return _no_cycles(f"the best-single-L1 arm ('{best}')")

    return ArmSet(
        baseline=picks_arm("as-configured (stacked L2 predicted_alpha)", baseline_cycles),
        ablated=picks_arm(
            f"best single L1 only ({best}, standalone alpha-IC "
            f"{ics[best]:+.4g})",
            ablated_cycles,
        ),
    )


def build_momentum_l1(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    return _drop_one_l1(inputs, "momentum_score")


def build_volatility_l1(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    return _drop_one_l1(inputs, "expected_move")


def build_research_calibrator_l1(inputs: ReplayInputs) -> ArmSet | NotAvailable:
    return _drop_one_l1(inputs, "research_calibrator_prob")


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------

META_L2_IC_SPEC = ReplaySpec(
    name="meta_l2_ic",
    module="predictor",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=_build_meta_l2,
)

MOMENTUM_L1_IC_SPEC = ReplaySpec(
    name="momentum_l1_ic",
    module="predictor",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_momentum_l1,
)

VOLATILITY_L1_IC_SPEC = ReplaySpec(
    name="volatility_l1_ic",
    module="predictor",
    criticality="supporting",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_volatility_l1,
)

RESEARCH_CALIBRATOR_L1_IC_SPEC = ReplaySpec(
    name="research_calibrator_l1_ic",
    module="predictor",
    criticality="supporting",
    pattern="substitution",
    issue=ISSUE,
    build_arms=build_research_calibrator_l1,
)

ENSEMBLE_LIFT_SPEC = ReplaySpec(
    name="ensemble_lift_over_best_l1",
    module="predictor",
    criticality="critical",
    pattern="substitution",
    issue=ISSUE,
    build_arms=_build_ensemble_lift,
)

#: Emission order for ``registry.SPECS`` — matches the tile's own component
#: order in ``crucible-evaluator/grading/tiles/predictor.py``.
SPECS = [
    META_L2_IC_SPEC,
    MOMENTUM_L1_IC_SPEC,
    VOLATILITY_L1_IC_SPEC,
    RESEARCH_CALIBRATOR_L1_IC_SPEC,
    ENSEMBLE_LIFT_SPEC,
]
