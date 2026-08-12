"""evaluator_attestation.py — the Evaluator stage's RUNTIME correctness verdict.

WHY THIS EXISTS
---------------
``analysis/attestation.py`` attests the **simulation engine**: fills, fees, NAV
marking, drawdown, benchmark windows, and the classification counts the Report
Card's precision is built from. It says nothing about the stage that runs next.

The Evaluator stage (``evaluate.py``, the ``EvaluatorDiagnostics`` +
``EvaluatorOptimize`` SF states) is the one that *acts*. It grades signal quality,
scores the producer arms, and — through the assembler and the champion-promotion
engine — writes ``config/executor_params.json`` and ``config/producer_champion.json``,
the two artifacts the live executor reads. It decides those things from three
families of number:

======================  =============================================================
Information coefficient ``nousergon_lib.quant.stats.information_coefficient.compute_ic``
Hit rate / accuracy     ``analysis.signal_quality.compute_accuracy``
Calibration             ``analysis.calibration_diagnostics.compute_calibration``
======================  =============================================================

Two of the three are shared-library code whose pin moves independently of this
repo, and all three run on a spot instance that resolves its own wheels at boot —
the same substrate argument ``analysis/attestation.py`` makes for vectorbt. A
changed tie-handling rule in the Spearman path, a flipped sign convention, a
different ECE binning rule, or an accuracy denominator that starts counting
unresolved outcomes would move every ranking number the Evaluator promotes off,
coherently and plausibly, with no artifact anywhere saying the arithmetic moved.

`sf-pipeline-policy.md` §2.3a: a missing **data artifact** makes a consumer fail
visibly; a missing **correctness verdict** makes every consumer succeed *as though
the check had passed*. This module is the Evaluator stage's verdict.

WHAT IT ASSERTS
---------------
Each check drives the **production** function — the same import the Evaluator's
own diagnostics call — over a hand-built input whose answer is derived from the
metric's *definition*, on paper, never by running the code under test.

``ic_perfect_monotone``
    Conviction and forward return in identical rank order over 30 pairs with no
    ties. Spearman's ρ is Pearson's r on ranks; identical rank vectors give
    r = 1 exactly. Expected **1.0**.

``ic_perfect_inverse``
    The same 30 pairs with the return order reversed — ranks are exact
    reflections, so r = -1. Expected **-1.0**.

``ic_single_adjacent_transposition``
    The monotone set with exactly one adjacent pair of ranks swapped. With no
    ties, ρ = 1 − 6·Σd²/(n(n²−1)); one adjacent swap gives Σd² = 1² + 1² = 2, so
    for n = 30, ρ = 1 − 12/(30·899) = 1 − 12/26970 = **4493/4495**. This is the
    load-bearing one: it is the only IC check whose expectation is not a
    degenerate ±1, so it is the only one that would catch a ρ computed with the
    wrong denominator. It is also derived from the closed-form no-ties identity,
    which is a *different* derivation from the rank-Pearson the implementation
    performs — the two agreeing is evidence, not tautology.

``ic_no_variance_is_not_zero_ic``
    Constant conviction. IC is undefined, not 0.0-with-confidence. The
    production contract is ``status="no_variance"``; the check asserts the
    status, encoded numerically (1.0 = the contract held). A degenerate input
    silently grading as "IC exactly zero, well measured" is the failure this
    catches.

``hit_rate_accuracy_21d`` / ``hit_rate_precision_21d`` / ``hit_rate_n_21d``
    30 of 50 rows beat SPY at the canonical 21-day horizon. Accuracy and
    precision are both 30/50 = **0.6**, n = **50**. Counts and a mean of a 0/1
    column — arithmetic that needs no second implementation to predict.

``hit_rate_avg_alpha_21d``
    The same 50 rows carry stock and SPY returns whose per-row difference is a
    fixed **0.02**, so the mean alpha is 0.02 regardless of ordering. This is the
    units check: it fails if alpha is ever computed as a ratio of returns, in
    percent, or against the wrong leg.

``hit_rate_excludes_unresolved``
    The frame carries 12 additional rows whose 21-day outcome is unresolved
    (NaN). ``n_21d`` must still be 50. "Counted something it should not have" is
    the accuracy-side twin of the classification-count failure
    ``analysis/attestation.py`` covers, and no all-resolved fixture can catch it.

``calibration_ece_perfect`` / ``calibration_brier_perfect``
    100 picks across the five default quintile bins, each bin's realized hit rate
    set exactly equal to its (constant) predicted probability: 2/20 at p=0.1,
    6/20 at 0.3, 10/20 at 0.5, 14/20 at 0.7, 18/20 at 0.9. Every bin gap is zero,
    so ECE = **0.0** exactly. The Brier score is
    Σ_bins [hits·(1−p)² + misses·p²] / 100 = (1.80 + 4.20 + 5.00 + 4.20 + 1.80)/100
    = **0.17**.

``calibration_ece_miscalibrated``
    The same 100 picks with the middle bin's hits moved from 10 to 18 — its
    realized rate becomes 0.9 against a predicted 0.5, a gap of 0.4 on 20 of 100
    samples. ECE is the sample-weighted mean absolute gap = 20·0.4/100 = **0.08**.
    A perfectly-calibrated fixture alone cannot distinguish a correct ECE from
    one that returns zero unconditionally; this one pins the scale.

TOLERANCES
----------
Every expectation above is an exact rational number in float64 (``4493/4495`` is
the only one that is not exactly representable, and it is a single division).
The bands are ``analysis.attestation``'s ``_RTOL``/``_ATOL`` — 1e-9 relative,
1e-12 absolute — chosen there as several orders of magnitude above accumulated
float64 error over these operation counts and several orders *below* any
difference that could move a grade. They are not tuned to make this battery
pass: they are the same constants the engine battery already ships with, and the
checks here agree to ~1e-16.

CONTRACT
--------
Nothing here raises. Failure paths resolve to ``UNKNOWN`` through
``analysis.attestation.run_attestation``'s taxonomy — a check that *disagreed*
is FAIL (evidence the numbers are wrong), a check that could not *run* is UNKNOWN
(absence of evidence). Collapsing the two would make a broken venv read as a
correctness regression.
"""

from __future__ import annotations

import logging

from analysis.attestation import (
    Scenario,
    read_attestation,
    run_attestation,
    verdict_is_pass,
    worst,
)

logger = logging.getLogger(__name__)

SCHEMA = "evaluator_attestation-1.0.0"

#: The Evaluator stage's verdict artifact, beside the engine's ``attestation.json``
#: under the same ``backtest/{run_date}/`` prefix every other evaluator-read
#: artifact lives in.
_KEY_TEMPLATE = "backtest/{run_date}/evaluator_attestation.json"

_IC_N = 30


def attestation_key(run_date: str) -> str:
    """S3 key of the Evaluator stage's verdict for ``run_date``."""
    return _KEY_TEMPLATE.format(run_date=run_date)


# ════════════════════════════════════════════════════════════════════════════
# Information coefficient
# ════════════════════════════════════════════════════════════════════════════

def _ic_conviction() -> list[float]:
    # Distinct, strictly increasing — no ties, so the closed-form no-ties
    # Spearman identity in the docstring applies exactly.
    return [float(i) for i in range(_IC_N)]


def _ic_perfect_monotone() -> float:
    from analysis.information_coefficient import compute_ic

    conviction = _ic_conviction()
    forward = [0.001 * i for i in range(_IC_N)]
    result = compute_ic(conviction, forward)
    if result.get("status") != "ok":
        raise AssertionError(f"expected status=ok, got {result!r}")
    return float(result["ic"])


def _ic_perfect_inverse() -> float:
    from analysis.information_coefficient import compute_ic

    conviction = _ic_conviction()
    forward = [0.001 * (_IC_N - 1 - i) for i in range(_IC_N)]
    result = compute_ic(conviction, forward)
    if result.get("status") != "ok":
        raise AssertionError(f"expected status=ok, got {result!r}")
    return float(result["ic"])


def _ic_single_adjacent_transposition() -> float:
    from analysis.information_coefficient import compute_ic

    conviction = _ic_conviction()
    forward = [0.001 * i for i in range(_IC_N)]
    # Swap one adjacent pair -> sum of squared rank differences = 2.
    forward[10], forward[11] = forward[11], forward[10]
    result = compute_ic(conviction, forward)
    if result.get("status") != "ok":
        raise AssertionError(f"expected status=ok, got {result!r}")
    return float(result["ic"])


def _ic_no_variance_contract() -> float:
    """1.0 iff constant conviction reports ``no_variance`` rather than a measured IC."""
    from analysis.information_coefficient import compute_ic

    result = compute_ic([5.0] * _IC_N, [0.001 * i for i in range(_IC_N)])
    return 1.0 if result.get("status") == "no_variance" else 0.0


# ════════════════════════════════════════════════════════════════════════════
# Hit rate / accuracy
# ════════════════════════════════════════════════════════════════════════════

#: 50 resolved rows, 30 of which beat SPY at the canonical horizon; every row's
#: stock-minus-SPY return is exactly 0.02. Plus 12 rows whose canonical-horizon
#: outcome is UNRESOLVED and must be excluded from every denominator.
_ACC_N_RESOLVED = 50
_ACC_N_BEAT = 30
_ACC_N_UNRESOLVED = 12
_ACC_ALPHA = 0.02


def _accuracy_frame():
    import numpy as np
    import pandas as pd
    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    short_h = DEFAULT_POLICY.diagnostic_horizons[0]
    long_h = DEFAULT_POLICY.primary_horizon
    short = DEFAULT_POLICY.outcome_columns(short_h)
    long = DEFAULT_POLICY.outcome_columns(long_h)

    beat_long = [1.0] * _ACC_N_BEAT + [0.0] * (_ACC_N_RESOLVED - _ACC_N_BEAT)
    # Vary the SPY leg so a mean-alpha computed off the wrong leg, or as a
    # ratio, cannot coincidentally land on 0.02.
    spy_long = [0.001 * i for i in range(_ACC_N_RESOLVED)]
    stock_long = [s + _ACC_ALPHA for s in spy_long]

    rows = {
        long.beat_spy: beat_long + [np.nan] * _ACC_N_UNRESOLVED,
        long.stock_return: stock_long + [np.nan] * _ACC_N_UNRESOLVED,
        long.spy_return: spy_long + [np.nan] * _ACC_N_UNRESOLVED,
        # The diagnostic horizon is resolved on every row, including the ones
        # whose canonical outcome is not — the two denominators are independent.
        short.beat_spy: [1.0] * (_ACC_N_RESOLVED + _ACC_N_UNRESOLVED),
        short.stock_return: [0.01] * (_ACC_N_RESOLVED + _ACC_N_UNRESOLVED),
        short.spy_return: [0.005] * (_ACC_N_RESOLVED + _ACC_N_UNRESOLVED),
        "score": [70.0 + (i % 20) for i in range(_ACC_N_RESOLVED + _ACC_N_UNRESOLVED)],
    }
    return pd.DataFrame(rows)


def _accuracy_result() -> dict:
    from analysis.signal_quality import compute_accuracy

    result = compute_accuracy(_accuracy_frame())
    if result.get("status") != "ok":
        raise AssertionError(f"expected status=ok, got {result!r}")
    return result


def _accuracy_21d() -> float:
    return float(_accuracy_result()["overall"]["accuracy_21d"])


def _precision_21d() -> float:
    return float(_accuracy_result()["overall"]["precision_21d"])


def _n_21d() -> float:
    return float(_accuracy_result()["overall"]["n_21d"])


def _avg_alpha_21d() -> float:
    return float(_accuracy_result()["overall"]["avg_alpha_21d"])


# ════════════════════════════════════════════════════════════════════════════
# Calibration
# ════════════════════════════════════════════════════════════════════════════

#: (predicted probability, n picks in the bin, hits) per default quintile bin.
_CAL_PERFECT = ((0.1, 20, 2), (0.3, 20, 6), (0.5, 20, 10), (0.7, 20, 14), (0.9, 20, 18))
#: Same, with the middle bin's realized rate moved to 0.9 against a predicted 0.5.
_CAL_SKEWED = ((0.1, 20, 2), (0.3, 20, 6), (0.5, 20, 18), (0.7, 20, 14), (0.9, 20, 18))


def _calibration_arrays(spec):
    probabilities: list[float] = []
    outcomes: list[float] = []
    for p, n, hits in spec:
        probabilities.extend([p] * n)
        outcomes.extend([1.0] * hits + [0.0] * (n - hits))
    return probabilities, outcomes


def _calibration_result(spec) -> dict:
    from analysis.calibration_diagnostics import compute_calibration

    probabilities, outcomes = _calibration_arrays(spec)
    result = compute_calibration(probabilities, outcomes)
    if result.get("status") != "ok":
        raise AssertionError(f"expected status=ok, got {result!r}")
    return result


def _calibration_ece_perfect() -> float:
    return float(_calibration_result(_CAL_PERFECT)["ece"])


def _calibration_brier_perfect() -> float:
    return float(_calibration_result(_CAL_PERFECT)["brier_score"])


def _calibration_ece_miscalibrated() -> float:
    return float(_calibration_result(_CAL_SKEWED)["ece"])


# ════════════════════════════════════════════════════════════════════════════
# Battery
# ════════════════════════════════════════════════════════════════════════════

def _SCENARIOS() -> list[Scenario]:
    return [
        Scenario(
            name="ic_perfect_monotone",
            description="Spearman rho of identical rank orders over 30 untied pairs = 1",
            expected=1.0,
            compute=_ic_perfect_monotone,
        ),
        Scenario(
            name="ic_perfect_inverse",
            description="Spearman rho of exactly reversed rank orders = -1",
            expected=-1.0,
            compute=_ic_perfect_inverse,
        ),
        Scenario(
            name="ic_single_adjacent_transposition",
            description=(
                "One adjacent rank swap, n=30: rho = 1 - 6*sum(d^2)/(n(n^2-1)) "
                "= 1 - 12/26970 = 4493/4495"
            ),
            expected=4493.0 / 4495.0,
            compute=_ic_single_adjacent_transposition,
        ),
        Scenario(
            name="ic_no_variance_is_not_zero_ic",
            description="Constant conviction reports no_variance, not a measured IC of 0",
            expected=1.0,
            compute=_ic_no_variance_contract,
        ),
        Scenario(
            name="hit_rate_accuracy_21d",
            description="30 of 50 resolved rows beat SPY at the canonical horizon",
            expected=0.6,
            compute=_accuracy_21d,
        ),
        Scenario(
            name="hit_rate_precision_21d",
            description="Precision on BUY-only selections equals accuracy: 30/50",
            expected=0.6,
            compute=_precision_21d,
        ),
        Scenario(
            name="hit_rate_excludes_unresolved",
            description=(
                "12 rows with an unresolved canonical outcome stay out of the "
                "denominator: n_21d = 50, not 62"
            ),
            expected=float(_ACC_N_RESOLVED),
            compute=_n_21d,
        ),
        Scenario(
            name="hit_rate_avg_alpha_21d",
            description="Per-row stock-minus-SPY return is a fixed 0.02, so the mean is 0.02",
            expected=_ACC_ALPHA,
            compute=_avg_alpha_21d,
        ),
        Scenario(
            name="calibration_ece_perfect",
            description="Every bin's realized rate equals its predicted probability: ECE = 0",
            expected=0.0,
            compute=_calibration_ece_perfect,
        ),
        Scenario(
            name="calibration_brier_perfect",
            description=(
                "Brier = sum_bins[hits*(1-p)^2 + misses*p^2]/100 "
                "= (1.80+4.20+5.00+4.20+1.80)/100 = 0.17"
            ),
            expected=0.17,
            compute=_calibration_brier_perfect,
        ),
        Scenario(
            name="calibration_ece_miscalibrated",
            description=(
                "Middle bin realizes 0.9 against a predicted 0.5: "
                "ECE = 20*0.4/100 = 0.08"
            ),
            expected=0.08,
            compute=_calibration_ece_miscalibrated,
        ),
    ]


def run_evaluator_attestation(run_date: str | None = None) -> dict:
    """Run the ranking-metric known-answer battery and return the §2.3a verdict.

    Never raises. Returns a dict conforming to :data:`SCHEMA`.
    """
    return run_attestation(
        run_date=run_date,
        scenario_provider=_SCENARIOS,
        component="evaluator",
        schema=SCHEMA,
    )


# ════════════════════════════════════════════════════════════════════════════
# The gate — the Evaluator stage withholds promotion on a non-PASS verdict
# ════════════════════════════════════════════════════════════════════════════

def build_stage_attestation(bucket: str, run_date: str, s3_client=None) -> dict:
    """Run this stage's battery, read the engine's verdict, and combine the two.

    §2.3a rule 1: the verdict is consumed by every stage whose output depends on
    it being true. The Evaluator promotes ``config/executor_params.json`` and
    ``config/producer_champion.json`` off the simulation engine's numbers, so it
    depends on the engine's verdict whether or not it read it before — and off
    its own ranking metrics, hence the second half.

    Returns the artifact body. Never raises.
    """
    own = run_evaluator_attestation(run_date=run_date)
    upstream = read_attestation(bucket, run_date, s3_client=s3_client)
    combined = worst(own.get("verdict"), upstream.get("verdict"))
    return {
        "schema": SCHEMA,
        "component": "evaluator",
        "run_date": run_date,
        "verdict": combined,
        "promotion_withheld": not verdict_is_pass(combined),
        "own": own,
        "upstream": {
            "component": "backtester",
            "verdict": upstream.get("verdict"),
            "reason": upstream.get("reason"),
            "as_of": upstream.get("as_of"),
            "key": upstream.get("key"),
        },
    }


def write_attestation(bucket: str, run_date: str, body: dict, s3_client=None) -> str:
    """Persist the stage verdict. Returns the key written.

    Raises on failure: this artifact IS the evidence the stage graded its own
    correctness, so a silent write failure would reproduce the absence the
    verdict exists to remove.
    """
    import json

    import boto3

    client = s3_client or boto3.client("s3")
    key = attestation_key(run_date)
    client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(body, indent=2, default=str).encode(),
        ContentType="application/json",
    )
    return key


def apply_promotion_gate(body: dict, args) -> bool:
    """Force ``--freeze`` unless the combined verdict is an explicit PASS.

    Returns True when promotion was withheld.

    Freeze is reused deliberately rather than a new suppression path: it is the
    switch every writer in ``evaluate.py`` already honours (assembler live keys,
    the four auto-apply configs, the producer-champion pointer), it is already
    covered by their tests, and a *new* bypass would be a second thing to keep
    correct. The stage still runs and still emits every diagnostic — §2.3a's
    "withholding the guarantee is not the same as failing the pipeline" — it
    simply may not act on numbers nothing attested.
    """
    if verdict_is_pass(body.get("verdict")):
        return False
    setattr(args, "freeze", True)
    logger.error(
        "CORRECTNESS GUARANTEE WITHHELD (verdict=%s): own=%s upstream=%s (%s). "
        "Forcing --freeze — no config promotion, no champion promotion this cycle. "
        "sf-pipeline-policy.md 2.3a: a missing or failed verdict is never a pass.",
        body.get("verdict"),
        (body.get("own") or {}).get("verdict"),
        (body.get("upstream") or {}).get("verdict"),
        (body.get("upstream") or {}).get("reason"),
    )
    return True
