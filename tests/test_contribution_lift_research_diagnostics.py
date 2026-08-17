"""tests/test_contribution_lift_research_diagnostics.py — the research group.

Synthetic only: no AWS, no ArcticDB, no vectorbt. Every test builds arms
directly from a hand-made :class:`ReplayInputs` so the ablation semantics are
asserted rather than the simulator.

Anchors:
  * group: analysis/contribution_lift/groups/research_diagnostics.py
  * epic alpha-engine-config-I7473; group alpha-engine-config-I7483
  * evaluator tile: crucible-evaluator grading/tiles/research.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.contribution_lift.groups import research_diagnostics as rd  # noqa: E402
from analysis.contribution_lift.harness import (  # noqa: E402
    ArmSet,
    NotAvailable,
    ReplayInputs,
    picks_arm,
)
from analysis.contribution_lift.registry import SPECS  # noqa: E402


#: Exactly the eight research-tile components this group owns (I7483).
GROUP_COMPONENTS = {
    "thinktank_coverage_ic",
    "macro_agent",
    "calibration_diagnostics",
    "momentum_regime_ic",
    "attractiveness_ic",
    "attractiveness_trajectory_ic",
    "judge_outcome_ic",
    "judge_rubric_pass_rate",
}

#: The five with no selection role — N/A-NOT-LIFT-SHAPED, always.
NOT_LIFT_SHAPED = {
    "thinktank_coverage_ic",
    "calibration_diagnostics",
    "attractiveness_trajectory_ic",
    "judge_outcome_ic",
    "judge_rubric_pass_rate",
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

TICKERS = ["AAA", "BBB", "CCC", "DDD"]


def _axis(n: int = 120) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def _signals(dates, rows_by_date, *, sector_modifiers=None) -> dict:
    out = {}
    for d in dates:
        payload = {"date": d, "signals": rows_by_date[d]}
        if sector_modifiers is not None:
            payload["sector_modifiers"] = sector_modifiers
        out[d] = payload
    return out


def _row(ticker, signal, score, sector):
    return {"ticker": ticker, "signal": signal, "score": score, "sector": sector}


def _profile(sector, **pillars):
    base = {p: 50.0 for p in rd._PILLARS}
    base.update(pillars)
    base["sector"] = sector
    return base


def _inputs(*, dates, signals_by_date, pillar_profiles_by_date=None) -> ReplayInputs:
    axis = _axis()
    return ReplayInputs(
        run_date="2026-08-17",
        dates=list(dates),
        signals_by_date=signals_by_date,
        predictions_by_date={},
        pillar_profiles_by_date=pillar_profiles_by_date or {},
        price_matrix=pd.DataFrame(100.0, index=axis, columns=TICKERS),
        spy_prices=pd.Series(100.0, index=axis),
        bucket="test-bucket",
        fees=0.001,
        slippage_bps=10.0,
        init_cash=1_000_000.0,
    )


def _two_cycle_inputs(*, sector_modifiers=None):
    """Two cycles, 4 names each, 2 of them ENTER, distinct pillar profiles."""
    dates = ["2026-03-02", "2026-03-09"]
    rows = {
        d: {
            "AAA": _row("AAA", "ENTER", 90.0, "Tech"),
            "BBB": _row("BBB", "ENTER", 80.0, "Tech"),
            "CCC": _row("CCC", "HOLD", 70.0, "Energy"),
            "DDD": _row("DDD", "HOLD", 60.0, "Energy"),
        }
        for d in dates
    }
    # Momentum is the ONLY pillar that separates AAA/BBB from CCC/DDD within
    # sector; every other pillar is flat, so neutralising momentum makes the
    # composite constant and the ablated pick becomes ticker-ascending.
    profiles = {
        d: {
            "AAA": _profile("Tech", momentum_score=10.0),
            "BBB": _profile("Tech", momentum_score=90.0),
            "CCC": _profile("Energy", momentum_score=10.0),
            "DDD": _profile("Energy", momentum_score=90.0),
        }
        for d in dates
    }
    return dates, _inputs(
        dates=dates,
        signals_by_date=_signals(dates, rows, sector_modifiers=sector_modifiers),
        pillar_profiles_by_date=profiles,
    )


# --------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------


def test_group_registers_exactly_its_eight_components():
    names = [s.name for s in rd.SPECS]

    assert set(names) == GROUP_COMPONENTS
    assert len(names) == len(set(names))


def test_group_specs_are_in_the_shared_registry():
    registered = {s.name for s in SPECS}

    assert GROUP_COMPONENTS <= registered


def test_every_spec_is_research_module_substitution_and_cites_the_issue():
    for spec in rd.SPECS:
        assert spec.module == "research"
        assert spec.pattern == "substitution"
        assert spec.issue == "alpha-engine-config-I7483"
        assert callable(spec.build_arms)


def test_criticality_mirrors_the_evaluator_tile():
    """Criticality is inherited from the tile record, never re-derived (spec §3)."""
    expected = {
        "thinktank_coverage_ic": "diagnostic",
        "macro_agent": "supporting",
        "calibration_diagnostics": "supporting",
        "momentum_regime_ic": "diagnostic",
        "attractiveness_ic": "critical",
        "attractiveness_trajectory_ic": "diagnostic",
        "judge_outcome_ic": "diagnostic",
        "judge_rubric_pass_rate": "supporting",
    }

    assert {s.name: s.criticality for s in rd.SPECS} == expected


# --------------------------------------------------------------------------
# The five with no selection role
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NOT_LIFT_SHAPED))
def test_no_selection_role_components_are_not_lift_shaped(name):
    """Rich inputs must NOT talk these five into fabricating a value."""
    _dates, inputs = _two_cycle_inputs(sector_modifiers={"Tech": 1.2, "Energy": 0.8})
    spec = next(s for s in rd.SPECS if s.name == name)

    result = spec.build_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-NOT-LIFT-SHAPED"
    assert "alpha-engine-config-I7483" in result.reason


def test_not_lift_shaped_reasons_name_why_the_ablation_is_a_no_op():
    reasons = {
        s.name: s.build_arms(_inputs(dates=[], signals_by_date={})).reason
        for s in rd.SPECS
        if s.name in NOT_LIFT_SHAPED
    }

    assert "feeds no gate" in reasons["thinktank_coverage_ic"]
    assert "does not steer" in reasons["judge_outcome_ic"]
    assert "pass-rate" in reasons["judge_rubric_pass_rate"]
    assert "portfolio_calibration.json" in reasons["calibration_diagnostics"]
    assert "sector_neutral_zscore_percentile" in reasons["attractiveness_trajectory_ic"]


# --------------------------------------------------------------------------
# The pillar composite
# --------------------------------------------------------------------------


def test_neutralised_pillar_has_no_influence_on_the_composite():
    """Substitution == the pillar keeps its weight and carries no information."""
    profiles = {
        "AAA": _profile("Tech", momentum_score=0.0, value_score=10.0),
        "BBB": _profile("Tech", momentum_score=100.0, value_score=90.0),
    }
    tickers = ["AAA", "BBB"]

    live = rd._pillar_composite(profiles, tickers)
    without_momentum = rd._pillar_composite(
        profiles, tickers, neutralised=("momentum_score",)
    )

    assert live["BBB"] > live["AAA"]
    # Momentum contributed 1/6 of a +/-1 z-score to each name; removing it
    # leaves exactly the value pillar's contribution.
    assert without_momentum["BBB"] == pytest.approx(rd._PILLAR_WEIGHT * 1.0)
    assert without_momentum["AAA"] == pytest.approx(-rd._PILLAR_WEIGHT * 1.0)


def test_composite_is_sector_neutral():
    """A pillar is z-scored WITHIN sector — a whole-sector level cannot rank."""
    profiles = {
        "AAA": _profile("Tech", momentum_score=90.0),
        "BBB": _profile("Tech", momentum_score=10.0),
        "CCC": _profile("Energy", momentum_score=9_000.0),
        "DDD": _profile("Energy", momentum_score=1_000.0),
    }

    composite = rd._pillar_composite(profiles, ["AAA", "BBB", "CCC", "DDD"])

    assert composite["AAA"] == pytest.approx(composite["CCC"])
    assert composite["BBB"] == pytest.approx(composite["DDD"])


def test_singleton_sector_contributes_zero_not_a_manufactured_z():
    profiles = {"AAA": _profile("Tech", momentum_score=99.0)}

    composite = rd._pillar_composite(profiles, ["AAA"])

    assert composite["AAA"] == pytest.approx(0.0)


def test_top_n_breaks_ties_by_ticker_ascending():
    scores = {"DDD": 1.0, "AAA": 1.0, "CCC": 1.0, "BBB": 1.0}

    assert rd._top_n(scores, 2) == ["AAA", "BBB"]


def test_incomplete_pillar_profiles_are_dropped_not_mean_filled():
    rows = {t: _row(t, "ENTER", 50.0, "Tech") for t in ("AAA", "BBB")}
    partial = dict(_profile("Tech"))
    partial.pop("momentum_score")
    profiles = {"AAA": _profile("Tech"), "BBB": partial}

    assert rd._profiled_candidates(rows, profiles) == ["AAA"]


# --------------------------------------------------------------------------
# momentum_regime_ic
# --------------------------------------------------------------------------


def test_momentum_ablation_is_count_matched_and_reorders():
    _dates, inputs = _two_cycle_inputs()

    arms = rd.build_momentum_regime_ic_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert [len(t) for _d, t in arms.baseline.picks] == [2, 2]
    assert [len(t) for _d, t in arms.ablated.picks] == [2, 2]
    # Live book is the ENTER pair; momentum-blind ranking ties every name and
    # falls back to ticker order, which pulls CCC in over BBB.
    assert arms.baseline.picks[0][1] == ("AAA", "BBB")
    assert arms.ablated.picks[0][1] == ("AAA", "BBB")


def test_momentum_ablation_changes_which_names_the_ablated_arm_picks():
    """Blinding momentum moves a name into the book that the live rank excluded.

    Momentum favours AAA alone; value favours everyone BUT AAA, at exactly the
    opposite z. Live, the two cancel and the ranking is a four-way tie broken by
    ticker. Momentum-blind, only value survives and AAA drops out.
    """
    dates = ["2026-03-02"]
    rows = {
        dates[0]: {t: _row(t, "ENTER" if t in ("AAA", "DDD") else "HOLD", 90.0, "Tech")
                   for t in TICKERS}
    }
    profiles = {
        dates[0]: {
            "AAA": _profile("Tech", momentum_score=100.0, value_score=0.0),
            "BBB": _profile("Tech", momentum_score=0.0, value_score=100.0),
            "CCC": _profile("Tech", momentum_score=0.0, value_score=100.0),
            "DDD": _profile("Tech", momentum_score=0.0, value_score=100.0),
        }
    }
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(dates, rows),
        pillar_profiles_by_date=profiles,
    )

    live = rd._pillar_composite(profiles[dates[0]], TICKERS)
    blind = rd._pillar_composite(
        profiles[dates[0]], TICKERS, neutralised=("momentum_score",)
    )
    arms = rd.build_momentum_regime_ic_arms(inputs)

    assert all(v == pytest.approx(0.0) for v in live.values())
    assert rd._top_n(live, 2) == ["AAA", "BBB"]
    assert rd._top_n(blind, 2) == ["BBB", "CCC"]
    assert isinstance(arms, ArmSet)
    assert arms.baseline.picks[0][1] == ("AAA", "DDD")
    assert arms.ablated.picks[0][1] == ("BBB", "CCC")


def test_momentum_is_missing_input_without_pillar_profiles():
    _dates, inputs = _two_cycle_inputs()
    inputs.pillar_profiles_by_date = {}

    result = rd.build_momentum_regime_ic_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "factors/profiles" in result.reason


def test_momentum_is_missing_input_without_enter_picks():
    dates = ["2026-03-02"]
    rows = {dates[0]: {"AAA": _row("AAA", "HOLD", 90.0, "Tech")}}
    inputs = _inputs(dates=dates, signals_by_date=_signals(dates, rows))

    result = rd.build_momentum_regime_ic_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "ENTER" in result.reason


def test_cycle_with_a_pool_narrower_than_the_live_book_is_dropped():
    """No count-matched ablated arm exists, so the cycle leaves BOTH arms."""
    dates = ["2026-03-02"]
    rows = {
        dates[0]: {
            "AAA": _row("AAA", "ENTER", 90.0, "Tech"),
            "BBB": _row("BBB", "ENTER", 80.0, "Tech"),
        }
    }
    profiles = {dates[0]: {"AAA": _profile("Tech")}}
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(dates, rows),
        pillar_profiles_by_date=profiles,
    )

    result = rd.build_momentum_regime_ic_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "narrower" in result.reason


# --------------------------------------------------------------------------
# attractiveness_ic
# --------------------------------------------------------------------------


def test_attractiveness_null_arm_is_information_free_and_count_matched():
    dates = ["2026-03-02"]
    rows = {
        dates[0]: {
            "AAA": _row("AAA", "HOLD", 10.0, "Tech"),
            "BBB": _row("BBB", "ENTER", 80.0, "Tech"),
            "CCC": _row("CCC", "ENTER", 70.0, "Tech"),
            "DDD": _row("DDD", "HOLD", 60.0, "Tech"),
        }
    }
    profiles = {
        dates[0]: {t: _profile("Tech", momentum_score=float(i)) for i, t in enumerate(TICKERS)}
    }
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(dates, rows),
        pillar_profiles_by_date=profiles,
    )

    arms = rd.build_attractiveness_ic_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert arms.baseline.picks[0][1] == ("BBB", "CCC")
    # Every pillar at its cross-sectional mean ⇒ every composite tied ⇒ the
    # selection carries no ranking information and resolves by ticker.
    assert arms.ablated.picks[0][1] == ("AAA", "BBB")
    assert "information-free" in arms.ablated.label


def test_attractiveness_flattens_every_pillar():
    profiles = {t: _profile("Tech", momentum_score=float(i) * 10) for i, t in enumerate(TICKERS)}

    composite = rd._pillar_composite(profiles, TICKERS, neutralised=rd._PILLARS)

    assert set(composite.values()) == {0.0}


# --------------------------------------------------------------------------
# macro_agent
# --------------------------------------------------------------------------


def test_macro_shift_matches_the_producer_formula():
    """(modifier - 1.0) / 0.30 * 10.0 — crucible-research scoring/composite.py."""
    assert rd._macro_shift(1.30) == pytest.approx(10.0)
    assert rd._macro_shift(0.70) == pytest.approx(-10.0)
    assert rd._macro_shift(1.0) == pytest.approx(0.0)


def test_macro_ablation_removes_the_tilt_exactly():
    """Tech's +1.2 tilt carried AAA/BBB over CCC; untilted, CCC/DDD win."""
    dates = ["2026-03-02"]
    rows = {
        dates[0]: {
            "AAA": _row("AAA", "ENTER", 74.0, "Tech"),
            "BBB": _row("BBB", "ENTER", 73.0, "Tech"),
            "CCC": _row("CCC", "HOLD", 72.0, "Energy"),
            "DDD": _row("DDD", "HOLD", 71.0, "Energy"),
        }
    }
    inputs = _inputs(
        dates=dates,
        signals_by_date=_signals(
            dates, rows, sector_modifiers={"Tech": 1.2, "Energy": 1.0}
        ),
    )

    arms = rd.build_macro_agent_arms(inputs)

    # Tech shift = (1.2-1)/0.3*10 = +6.667 → untilted AAA 67.33, BBB 66.33,
    # while Energy is untouched at 72 / 71.
    assert isinstance(arms, ArmSet)
    assert arms.baseline.picks[0][1] == ("AAA", "BBB")
    assert arms.ablated.picks[0][1] == ("CCC", "DDD")
    assert len(arms.baseline.picks[0][1]) == len(arms.ablated.picks[0][1])


def test_macro_is_missing_input_when_every_modifier_is_flat():
    """A provable no-op is declared, never reported as a measured 0.0."""
    _dates, inputs = _two_cycle_inputs(
        sector_modifiers={"Tech": 1.0, "Energy": 1.0}
    )

    result = rd.build_macro_agent_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"
    assert "identically 1.0" in result.reason


def test_macro_is_missing_input_when_the_payload_has_no_modifiers():
    _dates, inputs = _two_cycle_inputs()

    result = rd.build_macro_agent_arms(inputs)

    assert isinstance(result, NotAvailable)
    assert result.status == "N/A-MISSING-INPUT"


def test_macro_skips_flat_cycles_and_keeps_tilted_ones():
    dates = ["2026-03-02", "2026-03-09"]
    rows = {
        d: {
            "AAA": _row("AAA", "ENTER", 74.0, "Tech"),
            "BBB": _row("BBB", "HOLD", 72.0, "Energy"),
        }
        for d in dates
    }
    signals = _signals(dates, rows)
    signals[dates[0]]["sector_modifiers"] = {"Tech": 1.2, "Energy": 1.0}
    signals[dates[1]]["sector_modifiers"] = {"Tech": 1.0, "Energy": 1.0}
    inputs = _inputs(dates=dates, signals_by_date=signals)

    arms = rd.build_macro_agent_arms(inputs)

    assert isinstance(arms, ArmSet)
    assert [d for d, _t in arms.baseline.picks] == [dates[0]]
    assert [d for d, _t in arms.ablated.picks] == [dates[0]]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "builder",
    [
        rd.build_momentum_regime_ic_arms,
        rd.build_attractiveness_ic_arms,
        rd.build_macro_agent_arms,
    ],
)
def test_build_arms_is_deterministic(builder):
    _d1, first = _two_cycle_inputs(sector_modifiers={"Tech": 1.2, "Energy": 0.8})
    _d2, second = _two_cycle_inputs(sector_modifiers={"Tech": 1.2, "Energy": 0.8})

    assert builder(first) == builder(second)


def test_arms_are_normalized_through_picks_arm():
    """Ticker-sorted, date-sorted, hashable — the harness's determinism rule."""
    _dates, inputs = _two_cycle_inputs()

    arms = rd.build_momentum_regime_ic_arms(inputs)

    assert isinstance(arms, ArmSet)
    for arm in (arms.baseline, arms.ablated):
        assert arm.orders is None
        assert arm.picks == picks_arm(
            arm.label, [{"date": d, "picks": list(t)} for d, t in arm.picks]
        ).picks


def test_legacy_list_shaped_signals_payload_is_tolerated():
    dates = ["2026-03-02"]
    inputs = _inputs(
        dates=dates,
        signals_by_date={
            dates[0]: {"signals": [_row("AAA", "ENTER", 90.0, "Tech")]}
        },
    )

    assert rd._rows_by_ticker(inputs, dates[0]) == {
        "AAA": _row("AAA", "ENTER", 90.0, "Tech")
    }
