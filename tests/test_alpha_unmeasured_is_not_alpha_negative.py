"""An unmeasured alpha floor is not a failed one — alpha-engine-config-I7672.

`NaN >= alpha_floor` is False, so an all-null `total_alpha` column drops every
combo and the gate reported "All 60 valid combos backtested with total_alpha
< 0.0". Measured on the live `backtest/2026-08-14/param_sweep.csv`: 60 rows,
all `status: ok`, `total_return` and `sharpe_ratio` populated on every row, and
`total_alpha` / `spy_return` blank on every row with
`null_legs = ['spy_return' 'total_alpha']`.

That ran for seven consecutive weeks and reached the Director's weekly plan as
"the optimization loop is broken", then as a strategy finding. The backtester
was not measuring negative alpha. It was not measuring alpha.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.apply_audit import _STATUS_MAPS
from optimizer.executor_optimizer import _null_legs_summary, init_config, recommend

#: `recommend` reads a MODULE-GLOBAL `_cfg` set by `init_config` — not its
#: `base_config` argument. A test that only passes the dict silently runs with
#: `alpha_floor=None` and the gate never fires.
_CFG = {"executor_optimizer": {"alpha_floor": 0.0, "min_valid_combos": 5,
                               "min_trades_to_promote": 1,
                               "min_improvement": 0.0}}


@pytest.fixture(autouse=True)
def _init():
    init_config(_CFG)
    yield
    init_config({})


def _sweep(total_alpha, n=10, null_legs="['spy_return' 'total_alpha']"):
    """A sweep shaped like the live one: healthy internals, benchmark absent."""
    return pd.DataFrame({
        "min_score": [50 + i for i in range(n)],
        "max_position_pct": [0.05] * n,
        "total_return": [0.04 + i * 0.001 for i in range(n)],
        "sharpe_ratio": [0.45 + i * 0.001 for i in range(n)],
        "sortino_ratio": [0.9 + i * 0.001 for i in range(n)],
        "total_trades": [75] * n,
        "total_alpha": total_alpha,
        "status": ["ok"] * n,
        "null_legs": [null_legs] * n,
    })


# --------------------------------------------------------------------------
# The live shape
# --------------------------------------------------------------------------

def test_an_all_null_alpha_column_is_unmeasured_not_below_floor():
    res = recommend(_sweep([np.nan] * 10), _CFG)
    assert res["status"] == "alpha_unmeasured"
    assert res["n_measured"] == 0
    note = res["note"]
    assert "NULL on all 10 valid combos" in note
    # The sentence that must never be emitted about data that does not exist.
    assert "backtested with total_alpha < " not in note
    assert "no combo has been shown to be alpha-negative" in note


def test_the_note_names_the_null_legs_the_sweep_itself_reported():
    """`null_legs` has been on every row since at least 2026-07-24 and no
    consumer read it. Reading it turns 'alpha is missing' into 'alpha is
    missing BECAUSE the benchmark leg was not supplied' — the difference
    between a strategy finding and a plumbing one."""
    res = recommend(_sweep([np.nan] * 10), _CFG)
    assert "spy_return" in res["note"]
    assert set(res["null_legs"]) == {"spy_return", "total_alpha"}
    assert "config-I7672" in res["note"]


def test_genuinely_negative_alpha_still_blocks():
    """The gate must not be disarmed. This is the case it was built for."""
    res = recommend(_sweep([-0.02 - i * 0.001 for i in range(10)]), _CFG)
    assert res["status"] == "alpha_below_floor"
    assert res["n_measured"] == 10
    assert res["best_alpha_in_sweep"] == pytest.approx(-0.02, abs=1e-6)


def test_a_partially_null_column_reports_the_measured_denominator():
    """Surviving combos must not stand in for the whole sweep."""
    alphas = [np.nan] * 7 + [-0.03, -0.02, -0.01]
    res = recommend(_sweep(alphas), _CFG)
    assert res["status"] == "alpha_below_floor"
    assert res["n_measured"] == 3
    assert "3 MEASURED combos (of 10 valid)" in res["note"]
    assert "WARNING" not in res["note"]  # the narrowing is a log line, not the note


def test_a_measured_positive_combo_passes_the_floor():
    alphas = [np.nan] * 7 + [-0.03, -0.02, 0.05]
    res = recommend(_sweep(alphas), _CFG)
    assert res["status"] not in ("alpha_below_floor", "alpha_unmeasured")


# --------------------------------------------------------------------------
# _null_legs_summary — the column round-trips through CSV as a string
# --------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("['spy_return' 'total_alpha']", ["spy_return", "total_alpha"]),
    ("['spy_return', 'total_alpha']", ["spy_return", "total_alpha"]),
    (["spy_return", "total_alpha"], ["spy_return", "total_alpha"]),
])
def test_null_legs_summary_normalizes_both_shapes(raw, expected):
    df = pd.DataFrame({"null_legs": [raw, raw]})
    assert _null_legs_summary(df) == expected


def test_null_legs_summary_tolerates_absence():
    assert _null_legs_summary(pd.DataFrame({"a": [1]})) == []


# --------------------------------------------------------------------------
# The audit mapping — this must be LOUD
# --------------------------------------------------------------------------

def test_alpha_unmeasured_maps_to_error_not_insufficient_data():
    """`insufficient_data` grades HEALTHY in the evaluator's state machine.
    A sweep that cannot measure its own hard constraint must not be invisible,
    and it is not 'blocked' either — nothing refused a recommendation, there
    was no measurement to refuse one on."""
    outcome, blocked_by = _STATUS_MAPS["executor_params"]["alpha_unmeasured"]
    assert outcome == "error"
    assert blocked_by is None


# --------------------------------------------------------------------------
# The producer half — the pipeline must actually SUPPLY the benchmark
# --------------------------------------------------------------------------

def test_every_pipeline_sim_call_passes_the_benchmark_legs():
    """The root cause, pinned structurally.

    `_setup_simulation` builds ~904 EQUITY columns and no macro, so SPY is not
    in the price matrix, and nothing in `_run_simulation_pipeline` loaded it.
    Every `_run_simulation_loop` call there ran with `spy_prices=None`, and
    `portfolio_stats` emitted `spy_return`/`total_alpha` as null on every
    weekly artifact back to at least 2026-07-24.

    Asserted on the source rather than by running a four-hour simulation: any
    NEW sim call added to that function without the benchmark fails here, which
    is the regression that actually needs preventing. The EW legs are included
    because they are OPT-IN in `_emit_leg` — an omitted kwarg is silently None
    and is NOT recorded in `null_legs`, so their absence would go unflagged
    exactly as it did for seven weeks.
    """
    import ast
    import inspect
    import pathlib

    source = pathlib.Path("backtest.py").read_text()
    tree = ast.parse(source)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_run_simulation_pipeline"
    )
    calls = [
        c for c in ast.walk(fn)
        if isinstance(c, ast.Call)
        and getattr(c.func, "id", None) == "_run_simulation_loop"
    ]
    assert len(calls) >= 5, f"expected >=5 sim call sites, found {len(calls)}"
    for call in calls:
        kwargs = {k.arg for k in call.keywords}
        assert "spy_prices" in kwargs, (
            f"_run_simulation_loop at line {call.lineno} passes no spy_prices — "
            "its total_alpha will be null (config-I7672)"
        )
        assert "ew_high_vol_basket_returns" in kwargs, (
            f"_run_simulation_loop at line {call.lineno} passes no "
            "ew_high_vol_basket_returns — an opt-in leg, so its absence is "
            "NOT flagged in null_legs (config-I7672)"
        )


def test_load_spy_close_degrades_to_none_rather_than_raising(monkeypatch):
    """A macro-library hiccup must not kill a four-hour simulation whose
    strategy-internal metrics are entirely computable without a benchmark. The
    consumer-side gate above is what makes the degradation loud."""
    import nousergon_lib.arcticdb as ndb

    from store.arctic_reader import load_spy_close

    def _boom(*a, **k):
        raise RuntimeError("macro library unreachable")

    monkeypatch.setattr(ndb, "open_macro_lib", _boom)
    monkeypatch.setattr("store.arctic_reader._get_arctic", lambda b: None)
    assert load_spy_close("bucket") is None
