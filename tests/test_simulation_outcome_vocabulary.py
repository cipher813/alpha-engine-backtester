"""A degeneracy of the INPUT is never reported in the vocabulary of the
STRATEGY (alpha-engine-config-I7309).

`_run_simulation_loop` returned, for a run handed zero simulation dates:

    {"status": "no_orders",
     "note": "No ENTER signals passed risk rules during the simulation period"}

Both halves are false. No dates were simulated, so no ENTER signal was ever
presented to a risk rule. The sentence is a confident, plausible, specific
attribution to a subsystem the run never reached — and it is the sentence that
sent three weeks of `pit_parity` diagnosis at position sizing and the risk
guard while the actual cause was upstream (the walk-forward pass built zero
folds, so there were no signals and no dates at all; see
`tests/test_pit_parity_smoke_fold_arithmetic.py`).

`no_orders` remains a real and legitimate outcome — a market week where every
candidate was gated is a genuine result, and `assert_predictor_backtest_
deliverable` deliberately tolerates it (alpha-engine-config#7252). The fix is
not to make it louder; it is to stop an empty INPUT from borrowing its name.
"""

from __future__ import annotations

import pandas as pd
import pytest

import backtest as bt


@pytest.fixture()
def _price_matrix():
    idx = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"])
    return pd.DataFrame({"AAPL": [100.0, 101.0, 102.0]}, index=idx)


def _loop(monkeypatch, *, signals_by_date, dates, price_matrix):
    """Run `_run_simulation_loop` with the executor + feature-store seams
    stubbed. Only the OUTCOME vocabulary is under test."""
    monkeypatch.setattr(
        bt, "_build_merged_simulate_config", lambda cfg: ({}, {}),
    )
    monkeypatch.setattr(
        "store.feature_maps.load_precomputed_feature_maps",
        lambda bucket, tickers_allowlist=None: ({"AAPL": 0.02}, {}, {"AAPL": 1.0}),
    )
    monkeypatch.setattr(
        "nousergon_lib.arcticdb.get_universe_symbols", lambda bucket: {"AAPL"},
    )
    monkeypatch.setattr(
        bt, "_build_pit_universe_resolver", lambda *a, **k: None,
    )
    monkeypatch.setattr(
        bt, "_precompute_signal_lookups", lambda *a, **k: {},
    )
    # No date ever produces an order — the two statuses under test differ only
    # in whether there were dates to try.
    monkeypatch.setattr(
        bt, "_simulate_single_date", lambda **kw: ([], None),
    )

    class _Client:
        def __init__(self, prices=None, nav=0.0):
            self._positions = {}

    return bt._run_simulation_loop(
        object(), _Client,
        dates=dates,
        price_matrix=price_matrix,
        config={"signals_bucket": "test-bucket"},
        signals_by_date=signals_by_date,
    )


def test_zero_dates_is_no_dates_not_no_orders(monkeypatch, _price_matrix):
    """THE regression test for the false attribution."""
    out = _loop(
        monkeypatch, signals_by_date={}, dates=[], price_matrix=_price_matrix,
    )
    assert out["status"] == "no_dates", (
        "a run handed zero simulation dates reported the STRATEGY outcome "
        "'no_orders' — see this module's docstring (config-I7309)"
    )
    assert out["dates_expected"] == 0
    note = out["note"].lower()
    # The note must not ATTRIBUTE the empty result to the risk rules. It may
    # (and does) say explicitly that it is not a statement about them.
    assert "passed risk rules" not in note
    assert "not a statement about entries or risk rules" in note
    assert "upstream" in note


def test_dates_that_produced_no_orders_still_report_no_orders(
    monkeypatch, _price_matrix,
):
    """The counterweight: a real week where dates ran and nothing passed is
    still `no_orders`, unchanged. Widening the new status to cover this would
    break the deliberate production tolerance in
    `assert_predictor_backtest_deliverable` (config#7252)."""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07"]
    out = _loop(
        monkeypatch,
        signals_by_date={d: {"date": d, "enter": []} for d in dates},
        dates=dates,
        price_matrix=_price_matrix,
    )
    assert out["status"] == "no_orders"
    assert out["dates_expected"] == len(dates)
    assert "risk rules" in out["note"]


def test_the_reporter_does_not_render_no_dates_as_a_risk_rule_outcome():
    """The artifact is one surface; the human-readable report is the other,
    and it carried the same sentence."""
    import reporter

    lines = "\n".join(
        reporter._section_predictor_backtest(
            {"status": "no_dates", "note": "Nothing to simulate — zero dates."}
        )
    ).lower()
    assert "passed risk rules" not in lines
    assert "not measured" in lines
