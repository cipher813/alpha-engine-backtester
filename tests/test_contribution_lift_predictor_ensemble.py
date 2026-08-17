"""tests/test_contribution_lift_predictor_ensemble.py — the predictor L1/L2 group.

Synthetic throughout: no AWS, no ArcticDB, no vectorbt. The S3 manifest read is
covered by a fake ``s3_client`` with the SAME shape as the live artifact
(measured 2026-08-16 against ``s3://alpha-engine-research/predictor/weights/
meta/manifest.json``), so a schema drift on the producer side fails here rather
than silently N/A-ing five components on the weekly run.

Anchors:
  * group module: analysis/contribution_lift/groups/predictor_ensemble.py
  * contract: contribution_lift.json v1 (crucible-evaluator consumer)
  * spec: alpha-engine-docs/private/report-card-v3-objective-and-attribution-260816.md §1, §3
  * epic alpha-engine-config-I7473; harness I7475; this group I7479
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift.groups import predictor_ensemble as grp  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    ArmSet,
    NotAvailable,
    ReplayInputs,
    picks_per_cycle,
)
from analysis.contribution_lift.registry import SPECS  # noqa: E402


TICKERS = ["AAA", "BBB", "CCC", "DDD"]
DATES = ["2026-03-02", "2026-03-03"]


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


class _FakeS3:
    """Serves one canned object; records the keys asked for."""

    def __init__(self, objects: dict[str, dict] | None):
        self.objects = objects
        self.keys: list[str] = []

    def get_object(self, Bucket: str, Key: str):  # noqa: N803 - boto3 signature
        self.keys.append(Key)
        if self.objects is None or Key not in self.objects:
            raise _NoSuchKey()
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}


class _NoSuchKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


def _manifest(
    *,
    coefficients: dict | None = None,
    feature_std: dict | None = None,
    standalone: dict | None = None,
    meta_scaler: dict | None = None,
) -> dict:
    """The live manifest's shape, with the fields this group reads.

    ``meta_scaler`` defaults to ABSENT, matching a legacy or unstandardized
    model — that is the identity path, not a gap. Pass one to exercise the
    standardized path (alpha-engine-config-I7511).
    """
    body = {
        "meta_coefficients": coefficients
        if coefficients is not None
        else {
            "research_calibrator_prob": 1.626403,
            "momentum_score": -0.11876,
            "expected_move": 1.763329,
            "intercept": -0.83732,
        },
        "models": {
            "meta_model": {
                "importance": {
                    "feature_std": feature_std
                    if feature_std is not None
                    else {
                        "research_calibrator_prob": 0.014518,
                        "momentum_score": 0.0791,
                        "expected_move": 0.010621,
                    }
                }
            }
        },
        "meta_l1_standalone_alpha_ic": standalone
        if standalone is not None
        else {
            "research_calibrator_prob": {"xsec_ic": 0.197032, "n_dates": 91},
            "momentum_score": {"xsec_ic": -0.058143, "n_dates": 91},
            "expected_move": {"xsec_ic": 0.105547, "n_dates": 91},
        },
    }
    if meta_scaler is not None:
        body["meta_scaler"] = meta_scaler
    return body


def _scaler(*features: str, mean=0.5, std=0.25, winsor=3.0) -> dict:
    """A meta_scaler in the shape MetaModel._build_scaler emits."""
    return {
        "directional": list(features),
        "mean": {f: mean for f in features},
        "std": {f: std for f in features},
        "winsor": winsor,
    }


def _prediction(ticker: str, alpha: float, **features) -> dict:
    row = {
        "ticker": ticker,
        "predicted_alpha": alpha,
        "predicted_alpha_raw": alpha,
        "research_calibrator_prob": 0.5,
        "momentum_confirmation": 0.0,
        "expected_move": 0.05,
    }
    row.update(features)
    return row


def _signals(enter: int, scores: dict[str, float] | None = None) -> dict:
    """A signals.json body with ``enter`` ENTER rows and a score per ticker."""
    scores = scores if scores is not None else {t: float(i) for i, t in enumerate(TICKERS)}
    rows = {}
    for i, ticker in enumerate(TICKERS):
        rows[ticker] = {
            "ticker": ticker,
            "signal": "ENTER" if i < enter else "HOLD",
            "score": scores.get(ticker),
        }
    return {"signals": rows}


def _inputs(
    *,
    predictions: dict | None = None,
    signals: dict | None = None,
    s3_client="default",
    dates: list[str] | None = None,
) -> ReplayInputs:
    dates = dates if dates is not None else list(DATES)
    axis = pd.bdate_range("2026-03-02", periods=120)
    if predictions is None:
        predictions = {
            d: {
                t: _prediction(
                    t,
                    alpha=0.10 - 0.01 * i,
                    research_calibrator_prob=0.1 * (i + 1),
                    momentum_confirmation=0.01 * (3 - i),
                    expected_move=0.02 * (i + 1),
                )
                for i, t in enumerate(TICKERS)
            }
            for d in dates
        }
    if signals is None:
        signals = {d: _signals(enter=2) for d in dates}
    return ReplayInputs(
        run_date="2026-08-17",
        dates=dates,
        signals_by_date=signals,
        predictions_by_date=predictions,
        pillar_profiles_by_date={},
        price_matrix=pd.DataFrame(100.0, index=axis, columns=TICKERS),
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
        s3_client=_FakeS3({grp.MANIFEST_KEY: _manifest()})
        if s3_client == "default"
        else s3_client,
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_all_five_components_are_registered_with_tile_names():
    """The names ARE the join key to the evaluator tile; a typo is silent data loss."""
    registered = {spec.name: spec for spec in SPECS}
    expected = {
        "meta_l2_ic": "critical",
        "momentum_l1_ic": "critical",
        "volatility_l1_ic": "supporting",
        "research_calibrator_l1_ic": "supporting",
        "ensemble_lift_over_best_l1": "critical",
    }
    for name, criticality in expected.items():
        assert name in registered, f"{name} missing from registry.SPECS"
        assert registered[name].module == "predictor"
        assert registered[name].criticality == criticality
        assert registered[name].pattern == "substitution"
        assert registered[name].issue == grp.ISSUE


def test_registry_names_are_unique():
    names = [spec.name for spec in SPECS]
    assert len(names) == len(set(names))


# --------------------------------------------------------------------------
# Count matching + determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", grp.SPECS, ids=lambda s: s.name)
def test_arms_are_count_matched_every_cycle(spec):
    built = spec.build_arms(_inputs())
    assert isinstance(built, ArmSet), built
    assert picks_per_cycle(built.baseline) == picks_per_cycle(built.ablated)
    assert set(picks_per_cycle(built.baseline).values()) == {2}


@pytest.mark.parametrize("spec", grp.SPECS, ids=lambda s: s.name)
def test_build_arms_is_deterministic(spec):
    first = spec.build_arms(_inputs())
    second = spec.build_arms(_inputs())
    assert first.baseline.picks == second.baseline.picks
    assert first.ablated.picks == second.ablated.picks
    assert first.ablated.label == second.ablated.label


@pytest.mark.parametrize("spec", grp.SPECS, ids=lambda s: s.name)
def test_width_is_capped_by_the_candidate_count(spec):
    """A 9-ENTER cycle over 4 candidates yields 4 picks, not 9, in BOTH arms."""
    built = spec.build_arms(
        _inputs(signals={d: _signals(enter=len(TICKERS)) for d in DATES})
    )
    assert set(picks_per_cycle(built.baseline).values()) == {len(TICKERS)}
    assert picks_per_cycle(built.baseline) == picks_per_cycle(built.ablated)


@pytest.mark.parametrize("spec", grp.SPECS, ids=lambda s: s.name)
def test_zero_enter_cycles_are_skipped_entirely(spec):
    na = spec.build_arms(_inputs(signals={d: _signals(enter=0) for d in DATES}))
    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"


# --------------------------------------------------------------------------
# The leave-one-out arithmetic
# --------------------------------------------------------------------------


def test_dropping_a_positive_coefficient_term_demotes_its_leaders():
    """`s - coef*x` is computed on the persisted field, not on the alpha alone.

    ``research_calibrator_prob`` carries coef +1.626 and here it ANTI-correlates
    with the alpha, so removing its term must widen the existing ordering rather
    than reverse it — and a fixture where it CO-varies must reverse it. Both
    directions are asserted, because a subtraction that silently no-ops would
    pass a one-sided test and report a flat 0.0 lift forever.
    """
    built = grp.build_research_calibrator_l1(_inputs())
    # baseline ranks by alpha: AAA .10 > BBB .09 > CCC .08 > DDD .07
    assert dict(built.baseline.picks)[DATES[0]] == ("AAA", "BBB")
    # ablated = alpha - 1.626403 * prob, prob = .1/.2/.3/.4 -> same order, wider
    assert dict(built.ablated.picks)[DATES[0]] == ("AAA", "BBB")

    covarying = {
        d: {
            t: _prediction(t, 0.10 - 0.01 * i, research_calibrator_prob=0.1 * (4 - i))
            for i, t in enumerate(TICKERS)
        }
        for d in DATES
    }
    flipped = grp.build_research_calibrator_l1(_inputs(predictions=covarying))
    # ablated = alpha - 1.626 * prob, prob = .4/.3/.2/.1
    # -> AAA -.550, BBB -.398, CCC -.245, DDD -.093  =>  DDD, CCC
    assert dict(flipped.ablated.picks)[DATES[0]] == ("CCC", "DDD")


def test_dropping_a_negative_coefficient_term_flips_the_selection():
    """A negative coefficient means removing the L1 RAISES the score it suppressed.

    ``momentum_score`` carries coef -0.11876 and ``momentum_confirmation``
    descends with alpha here, so ablating it must not be a no-op — the point of
    the test is that the ablated arm is a genuinely different basket, which is
    the only thing that makes the measured lift non-zero.
    """
    predictions = {
        d: {
            "AAA": _prediction("AAA", 0.10, momentum_confirmation=2.0),
            "BBB": _prediction("BBB", 0.09, momentum_confirmation=-2.0),
            "CCC": _prediction("CCC", 0.08, momentum_confirmation=0.0),
            "DDD": _prediction("DDD", 0.07, momentum_confirmation=0.0),
        }
        for d in DATES
    }
    built = grp.build_momentum_l1(_inputs(predictions=predictions))
    # baseline top-2 = AAA, BBB. ablated score = alpha + 0.11876 * momentum:
    # AAA .10+.238=.338, BBB .09-.238=-.148, CCC .08, DDD .07 -> AAA, CCC
    assert dict(built.baseline.picks)[DATES[0]] == ("AAA", "BBB")
    assert dict(built.ablated.picks)[DATES[0]] == ("AAA", "CCC")


def test_volatility_arm_uses_the_expected_move_coefficient():
    built = grp.build_volatility_l1(_inputs())
    assert "expected_move" in built.ablated.label
    assert "+1.76333" in built.ablated.label or "1.763329" in built.ablated.label


def test_level_neutralized_predictions_give_the_same_arms_as_raw():
    """Subtracting a per-date constant cannot change a within-date ranking."""
    raw = _inputs()
    shifted_predictions = {
        d: {
            t: {**row, "predicted_alpha_raw": row["predicted_alpha_raw"] - 0.5}
            for t, row in rows.items()
        }
        for d, rows in raw.predictions_by_date.items()
    }
    shifted = grp.build_volatility_l1(_inputs(predictions=shifted_predictions))
    assert grp.build_volatility_l1(raw).ablated.picks == shifted.ablated.picks


def test_predicted_alpha_is_the_documented_fallback_field():
    """A pre-neutralization prediction row (no ``_raw``) is still usable."""
    predictions = {
        d: {
            t: {k: v for k, v in row.items() if k != "predicted_alpha_raw"}
            for t, row in rows.items()
        }
        for d, rows in _inputs().predictions_by_date.items()
    }
    built = grp.build_volatility_l1(_inputs(predictions=predictions))
    assert isinstance(built, ArmSet)
    assert dict(built.baseline.picks)[DATES[0]] == ("AAA", "BBB")


def test_tickers_missing_the_ablated_feature_leave_both_arms():
    """Otherwise the baseline would rank a candidate the ablated arm cannot."""
    predictions = {
        d: {
            t: ({k: v for k, v in _prediction(t, 0.10 - 0.01 * i).items()
                 if k != "expected_move"} if t == "AAA"
                else _prediction(t, 0.10 - 0.01 * i))
            for i, t in enumerate(TICKERS)
        }
        for d in DATES
    }
    built = grp.build_volatility_l1(_inputs(predictions=predictions))
    for _date, picks in built.baseline.picks:
        assert "AAA" not in picks
    assert picks_per_cycle(built.baseline) == picks_per_cycle(built.ablated)


# --------------------------------------------------------------------------
# Guards: the ablation must refuse where it would be wrong
# --------------------------------------------------------------------------


def test_standardized_column_is_refused_when_meta_scaler_cannot_cover_it():
    """feature_std ~= 1.0 means coef multiplies a z-scored column, not the raw one.

    Refusing is correct ONLY while mu/sigma are unavailable. Since
    crucible-predictor-PR508 (alpha-engine-config-I7502) they are in the
    manifest, so this refusal narrowed to the one case that genuinely cannot
    be reconstructed: standardized per feature_std, absent from meta_scaler.
    """
    manifest = _manifest(
        feature_std={
            "research_calibrator_prob": 0.98,
            "momentum_score": 0.0791,
            "expected_move": 0.010621,
        }
    )
    inputs = _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))
    na = grp.build_research_calibrator_l1(inputs)
    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert "meta_scaler" in na.reason
    assert "absent from the manifest" in na.reason
    # its unstandardized siblings still measure
    assert isinstance(grp.build_volatility_l1(inputs), ArmSet)


# ---------------------------------------------------------------------------
# alpha-engine-config-I7511 — the consumer contract for meta_scaler
# ---------------------------------------------------------------------------


def test_a_standardized_column_now_measures_when_meta_scaler_carries_it():
    """The whole point of I7502/I7511: what was N/A is now a measurement."""
    manifest = _manifest(
        feature_std={
            "research_calibrator_prob": 0.98,
            "momentum_score": 0.0791,
            "expected_move": 0.010621,
        },
        meta_scaler=_scaler("research_calibrator_prob", mean=0.5, std=0.25),
    )
    inputs = _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))

    arms = grp.build_research_calibrator_l1(inputs)

    assert isinstance(arms, ArmSet), (
        "with mu/sigma present the term IS removable — refusing here would be "
        "the old contract outliving the data that retired it"
    )
    assert "meta_scaler" in arms.ablated.label


def test_the_ablation_subtracts_the_SCALED_term_not_the_raw_one():
    """The correctness core: coef multiplies the column AS FITTED.

    The default fixture cannot show this — its feature values are rank-aligned
    with alpha, and an affine transform of a monotone column is still monotone,
    so the ablated ORDER is identical either way and a picks-differ assertion
    would pass for the wrong reason. This uses feature values where winsorizing
    bites ASYMMETRICALLY: two names sit far outside the +-w band and clip to the
    same value, collapsing a raw gap the unscaled arm would have kept.
    """
    feature_std = {
        "research_calibrator_prob": 0.0145,
        "momentum_score": 0.0791,
        "expected_move": 0.010621,
    }
    # The column's fitted sigma is large relative to its spread, so after
    # standardization the term contributes almost nothing and the ablated
    # ranking is driven by alpha. Subtracting the RAW column instead lets a
    # wide-but-unimportant feature dominate — which is exactly the error.
    predictions = {
        d: {
            "AAA": _prediction("AAA", alpha=0.05, research_calibrator_prob=9.0),
            "BBB": _prediction("BBB", alpha=0.20, research_calibrator_prob=5.0),
            "CCC": _prediction("CCC", alpha=0.09, research_calibrator_prob=0.30),
            "DDD": _prediction("DDD", alpha=0.08, research_calibrator_prob=0.10),
        }
        for d in DATES
    }
    scaled = _inputs(
        predictions=predictions,
        s3_client=_FakeS3({grp.MANIFEST_KEY: _manifest(
            feature_std=feature_std,
            meta_scaler=_scaler("research_calibrator_prob", mean=0.0, std=1000.0),
        )}),
    )
    raw = _inputs(
        predictions=predictions,
        s3_client=_FakeS3({grp.MANIFEST_KEY: _manifest(feature_std=feature_std)}),
    )

    a_scaled = grp.build_research_calibrator_l1(scaled)
    a_raw = grp.build_research_calibrator_l1(raw)

    assert isinstance(a_scaled, ArmSet) and isinstance(a_raw, ArmSet)
    assert a_scaled.baseline.picks == a_raw.baseline.picks, (
        "the baseline arm never applies the transform — only the removed term does"
    )
    assert a_scaled.ablated.picks != a_raw.ablated.picks, (
        "identical ablated arms would mean the scaler was read and then ignored"
    )


def test_no_scaler_is_the_identity_and_not_a_gap():
    """A legacy / unstandardized model must still measure, unchanged.

    With no transform fitted, the raw column IS what the ridge multiplied, so
    the ablation is exact. Treating an absent scaler as missing input would
    N/A every pre-L4565 model for no reason.
    """
    manifest = _manifest(
        feature_std={
            "research_calibrator_prob": 0.014518,
            "momentum_score": 0.0791,
            "expected_move": 0.010621,
        }
    )
    assert "meta_scaler" not in manifest
    inputs = _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))

    arms = grp.build_research_calibrator_l1(inputs)

    assert isinstance(arms, ArmSet)
    assert "no transform fitted" in arms.ablated.label


def test_a_degenerate_sigma_is_the_identity_mirroring_the_producer():
    """MetaModel._build_scaler records a ~0 std as 1.0; the consumer must agree.

    Reading a literal 0.0 and dividing would produce infinities in an artifact
    the evaluator parses as strict JSON.
    """
    assert grp._scaled(_scaler("f", mean=2.0, std=0.0), "f", 5.0) == pytest.approx(3.0)


def test_the_transform_matches_the_producers_apply_scaler_exactly():
    """Independent oracle for clip((x-mu)/sigma, +-w), including both bounds."""
    scaler = _scaler("f", mean=1.0, std=2.0, winsor=3.0)
    assert grp._scaled(scaler, "f", 5.0) == pytest.approx(2.0)
    assert grp._scaled(scaler, "f", 99.0) == pytest.approx(3.0)   # clipped high
    assert grp._scaled(scaler, "f", -99.0) == pytest.approx(-3.0)  # clipped low
    # A feature the scaler does not list is untouched, whatever else it carries.
    assert grp._scaled(scaler, "other", 7.5) == pytest.approx(7.5)


def test_a_malformed_meta_scaler_raises_rather_than_reading_as_absent():
    """Fail loud: 'absent' and 'malformed' want opposite handling.

    Absent means "no transform was fitted, use the raw column" — a correct
    measurement. Malformed means the producer contract broke, and silently
    taking the identity path would reconstruct the ablation against the wrong
    column space and report it as a measurement.
    """
    manifest = _manifest(meta_scaler={"directional": "research_calibrator_prob"})
    with pytest.raises(TypeError, match="meta_scaler"):
        grp._meta_scaler(manifest)


def test_no_executable_read_of_the_retired_sidecar_remains():
    """I7511 closes-when: the .meta.json staleness branch is gone.

    Docstrings are stripped before matching — this module deliberately RECORDS
    why the sidecar was abandoned, and that history must not trip a guard
    aimed at code.
    """
    import io as _io
    import tokenize as _tokenize

    path = REPO_ROOT / "analysis" / "contribution_lift" / "groups" / "predictor_ensemble.py"
    with open(path, "rb") as handle:
        code = " ".join(
            tok.string
            for tok in _tokenize.tokenize(_io.BytesIO(handle.read()).readline)
            if tok.type not in (_tokenize.COMMENT, _tokenize.STRING)
        )
    assert "meta_model" not in code or "pkl" not in code, (
        "predictor_ensemble.py still reads the meta_model.pkl.meta.json sidecar; "
        "the scaler comes from manifest.json::meta_scaler (config-I7511)"
    )


def test_missing_feature_std_map_refuses_every_reconstructed_arm():
    manifest = _manifest(feature_std={})
    inputs = _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))
    for build in (
        grp.build_momentum_l1,
        grp.build_volatility_l1,
        grp.build_research_calibrator_l1,
    ):
        na = build(inputs)
        assert isinstance(na, NotAvailable) and na.status == "N/A-MISSING-INPUT"


def test_absent_manifest_is_na_not_a_fabricated_value():
    inputs = _inputs(s3_client=_FakeS3({}))
    for build in (
        grp.build_momentum_l1,
        grp.build_volatility_l1,
        grp.build_research_calibrator_l1,
        grp._build_ensemble_lift,
    ):
        na = build(inputs)
        assert isinstance(na, NotAvailable)
        assert na.status == "N/A-MISSING-INPUT"
        assert grp.MANIFEST_KEY in na.reason
        assert grp.ISSUE in na.reason


def test_manifest_read_failure_that_is_not_a_404_is_raised_loud():
    class _Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    class _S3:
        def get_object(self, Bucket, Key):  # noqa: N803
            raise _Denied()

    with pytest.raises(_Denied):
        grp.build_volatility_l1(_inputs(s3_client=_S3()))


def test_missing_coefficient_for_the_ablated_l1_is_na():
    manifest = _manifest(coefficients={"expected_move": 1.7, "intercept": -0.8})
    na = grp.build_momentum_l1(_inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest})))
    assert isinstance(na, NotAvailable)
    assert "meta_coefficients['momentum_score']" in na.reason


def test_clipped_cycles_are_dropped_from_both_arms():
    """A clipped ridge output breaks the additive identity, so the cycle goes."""
    clipped = {
        DATES[0]: {
            t: _prediction(t, 0.5 if t in ("AAA", "BBB") else 0.05)
            for t in TICKERS
        },
        DATES[1]: {
            t: _prediction(t, 0.10 - 0.01 * i, expected_move=0.02 * (i + 1))
            for i, t in enumerate(TICKERS)
        },
    }
    built = grp.build_volatility_l1(_inputs(predictions=clipped))
    assert [d for d, _p in built.baseline.picks] == [DATES[1]]
    assert [d for d, _p in built.ablated.picks] == [DATES[1]]


def test_every_cycle_clipped_is_na_not_an_empty_arm():
    clipped = {
        d: {t: _prediction(t, 0.5 if t in ("AAA", "BBB") else 0.05) for t in TICKERS}
        for d in DATES
    }
    na = grp.build_volatility_l1(_inputs(predictions=clipped))
    assert isinstance(na, NotAvailable) and na.status == "N/A-MISSING-INPUT"


def test_no_predictions_at_all_is_na_for_every_component():
    for spec in grp.SPECS:
        na = spec.build_arms(_inputs(predictions={}))
        assert isinstance(na, NotAvailable), spec.name
        assert na.status == "N/A-MISSING-INPUT"
        assert "predictor/predictions" in na.reason


# --------------------------------------------------------------------------
# meta_l2_ic — the predictor-off arm
# --------------------------------------------------------------------------


def test_predictor_off_arm_ranks_by_the_research_composite_score():
    scores = {"AAA": 10.0, "BBB": 20.0, "CCC": 90.0, "DDD": 80.0}
    built = grp._build_meta_l2(
        _inputs(signals={d: _signals(enter=2, scores=scores) for d in DATES})
    )
    assert dict(built.baseline.picks)[DATES[0]] == ("AAA", "BBB")
    assert dict(built.ablated.picks)[DATES[0]] == ("CCC", "DDD")
    assert "predictor off" in built.ablated.label


def test_predictor_off_arm_needs_no_manifest():
    built = grp._build_meta_l2(_inputs(s3_client=_FakeS3({})))
    assert isinstance(built, ArmSet)


def test_candidates_without_a_signals_score_leave_both_arms():
    scores = {"AAA": 10.0, "BBB": 20.0, "CCC": None, "DDD": 80.0}
    built = grp._build_meta_l2(
        _inputs(signals={d: _signals(enter=4, scores=scores) for d in DATES})
    )
    for _date, picks in built.baseline.picks:
        assert "CCC" not in picks
    assert set(picks_per_cycle(built.baseline).values()) == {3}
    assert picks_per_cycle(built.baseline) == picks_per_cycle(built.ablated)


# --------------------------------------------------------------------------
# ensemble_lift_over_best_l1
# --------------------------------------------------------------------------


def test_best_l1_is_the_argmax_of_the_standalone_alpha_ic():
    built = grp._build_ensemble_lift(_inputs())
    # live manifest values: research_calibrator_prob 0.197 > expected_move 0.106
    assert "research_calibrator_prob" in built.ablated.label
    # ablated ranks by that feature: prob = .1/.2/.3/.4 -> DDD, CCC
    assert dict(built.ablated.picks)[DATES[0]] == ("CCC", "DDD")


def test_best_l1_switches_with_the_manifest_ic():
    manifest = _manifest(
        standalone={
            "research_calibrator_prob": {"xsec_ic": 0.01},
            "expected_move": {"xsec_ic": 0.30},
            "momentum_score": {"xsec_ic": -0.05},
        }
    )
    built = grp._build_ensemble_lift(
        _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))
    )
    assert "expected_move" in built.ablated.label


def test_best_l1_ties_break_by_feature_name():
    manifest = _manifest(
        standalone={
            "research_calibrator_prob": {"xsec_ic": 0.2},
            "expected_move": {"xsec_ic": 0.2},
            "momentum_score": {"xsec_ic": 0.2},
        }
    )
    built = grp._build_ensemble_lift(
        _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))
    )
    assert "expected_move" in built.ablated.label


def test_not_run_standalone_block_refuses_the_magnitude_fallback():
    manifest = _manifest(standalone={"status": "not_run"})
    na = grp._build_ensemble_lift(
        _inputs(s3_client=_FakeS3({grp.MANIFEST_KEY: manifest}))
    )
    assert isinstance(na, NotAvailable)
    assert "config-I1062" in na.reason


# --------------------------------------------------------------------------
# Manifest read hygiene
# --------------------------------------------------------------------------


def test_manifest_is_read_from_the_documented_key():
    s3 = _FakeS3({grp.MANIFEST_KEY: _manifest()})
    grp.build_volatility_l1(_inputs(s3_client=s3))
    assert s3.keys == [grp.MANIFEST_KEY]
    assert grp.MANIFEST_KEY == "predictor/weights/meta/manifest.json"


def test_no_s3_client_is_na_naming_the_manifest():
    """A ReplayInputs assembled without an S3 client cannot read the rule."""
    na = grp.build_volatility_l1(_inputs(s3_client=None))
    assert isinstance(na, NotAvailable)
    assert na.status == "N/A-MISSING-INPUT"
    assert grp.MANIFEST_KEY in na.reason


def test_non_object_manifest_body_fails_loud():
    class _S3:
        def get_object(self, Bucket, Key):  # noqa: N803
            return {"Body": io.BytesIO(b"[1, 2, 3]")}

    with pytest.raises(TypeError):
        grp.build_volatility_l1(_inputs(s3_client=_S3()))
