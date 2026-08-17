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
) -> dict:
    """The live manifest's shape, with the fields this group reads."""
    return {
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


def test_standardized_column_is_refused_naming_the_missing_scaler():
    """feature_std ~= 1.0 means coef multiplies a z-scored column, not the raw one."""
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
    # its unstandardized siblings still measure
    assert isinstance(grp.build_volatility_l1(inputs), ArmSet)


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
