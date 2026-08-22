"""The param-sweep combo loop stops on its budget instead of being killed at
the wall (alpha-engine-config-I7309 blocker 1).

THE MEASUREMENT
---------------
`pit_parity` runs each pass in a subprocess under ONE 2700 s ceiling
(`analysis/pit_parity.py::_PIT_PARITY_PASS_TIMEOUT`). The two passes are
asymptotically different work under that single ceiling:

  * the look-ahead pass runs ONE simulation — measured 922 s;
  * the walk-forward pass, when `pit_parity_sweep` is set, runs `param_sweep.
    sweep` → `_run_combos` — one full simulation PER COMBO.

`backtest/2026-07-31/pit_parity.json` carries the walk-forward child's own log
tail at the moment it was killed:

    Sweep combo 5/60 done in 272.5s (sweep elapsed 2144.1s)

60 × 272.5 s ≈ 16,350 s of work under a 2700 s ceiling. Five completed combos
were discarded with the process. It timed out again on 2026-08-07 and three
times on 2026-08-13.

THE FIX, AND WHY IT IS NOT A BIGGER CEILING
-------------------------------------------
Raising the ceiling makes an observational stage the longest thing in the
weekly pipeline and still guesses. The pass already carries a deadline
(`_pass_deadline_epoch`), and its OTHER long loop — the walk-forward fold loop
— has been self-deadlining since config#7199. This is the same shape applied to
the loop that was still unbounded, so the 2700 s ceiling becomes a true
backstop rather than the mechanism: the pass publishes what it finished.

The budget is only right if it is measured, so `combo_p90_s` and
`n_combos_run` / `n_combos_planned` reach `df.attrs` — a pass that is
persistently short needs a bigger BUDGET, and that is decidable from the
artifact instead of from a log tail.
"""

from __future__ import annotations

import time

import pytest

from analysis import param_sweep
from deadline_budget import DEADLINE_EPOCH_CONFIG_KEY


def _combos(n: int) -> list[dict]:
    return [{"min_score": 60 + i} for i in range(n)]


def _stats(_cfg: dict) -> dict:
    return {"sortino_ratio": 1.0, "total_alpha": 0.1}


# ── No deadline → nothing changes ──────────────────────────────────────────

def test_without_a_deadline_every_combo_runs():
    """The weekly `run_param_sweep`, the CLI and every test hand down no
    deadline. That path must be byte-for-byte the old behaviour."""
    seen = []

    def sim(cfg):
        seen.append(cfg["min_score"])
        return _stats(cfg)

    df = param_sweep._run_combos(_combos(6), sim, {})
    assert len(seen) == 6
    assert len(df) == 6
    assert df.attrs["sweep_budget_stopped"] is False
    assert df.attrs["n_combos_skipped_for_budget"] == 0
    assert df.attrs["n_combos_planned"] == 6
    assert df.attrs["n_combos_run"] == 6


# ── Deadline already blown → zero combos, not a doomed first combo ─────────

def test_a_blown_budget_runs_nothing_rather_than_starting_work_it_cannot_finish():
    """A pass handed a budget it has already spent must not begin a combo it
    will be killed inside — that is how five completed combos were lost."""
    seen = []

    def sim(cfg):
        seen.append(cfg["min_score"])
        return _stats(cfg)

    df = param_sweep._run_combos(
        _combos(6), sim, {DEADLINE_EPOCH_CONFIG_KEY: time.time() - 1},
    )
    assert seen == []
    assert df.attrs["sweep_budget_stopped"] is True
    assert df.attrs["n_combos_run"] == 0
    assert df.attrs["n_combos_skipped_for_budget"] == 6


# ── Deadline mid-flight → PARTIAL, with the numbers that size the next one ─

def test_the_loop_stops_partway_and_labels_the_partial(monkeypatch):
    """Budget for roughly three combos: the loop runs what fits, returns those
    rows, and says how many it did not run."""
    monkeypatch.setattr(param_sweep, "SWEEP_BUDGET_RESERVE_S", 0.0)
    monkeypatch.setattr(param_sweep, "SWEEP_FIRST_COMBO_ESTIMATE_S", 0.05)

    def sim(cfg):
        time.sleep(0.05)
        return _stats(cfg)

    df = param_sweep._run_combos(
        _combos(20), sim, {DEADLINE_EPOCH_CONFIG_KEY: time.time() + 0.25},
    )
    assert df.attrs["sweep_budget_stopped"] is True
    assert 0 < df.attrs["n_combos_run"] < 20
    assert df.attrs["n_combos_run"] + df.attrs["n_combos_skipped_for_budget"] == 20
    assert len(df) == df.attrs["n_combos_run"]
    # The measurement that says whether the bound is right.
    assert df.attrs["combo_p90_s"] is not None
    assert df.attrs["combo_p90_s"] > 0


def test_a_budget_stop_is_logged_at_error(monkeypatch, caplog):
    """A truncated sweep is a real coverage loss. Silently returning a short
    frame would make an under-budgeted pass look like a small grid."""
    monkeypatch.setattr(param_sweep, "SWEEP_BUDGET_RESERVE_S", 0.0)
    with caplog.at_level("ERROR", logger="analysis.param_sweep"):
        param_sweep._run_combos(
            _combos(4), _stats, {DEADLINE_EPOCH_CONFIG_KEY: time.time() - 1},
        )
    joined = "\n".join(r.getMessage() for r in caplog.records)
    assert "stopping early on budget" in joined
    assert "PARTIAL" in joined


# ── The completion gate must not count skipped combos as successes ─────────

def test_budget_skipped_combos_are_not_counted_as_successes():
    """`sweep`'s 50%-completion gate computed `n_total - n_failed`. Before the
    self-deadlining loop every planned combo produced a row, so that identity
    held; a truncated sweep breaks it and would have reported a short run as
    fully successful."""
    df = param_sweep.sweep(
        {"min_score": [60, 65, 70, 75, 80, 85]},
        _stats,
        {DEADLINE_EPOCH_CONFIG_KEY: time.time() - 1},
        sweep_settings={"mode": "grid"},
    )
    # Nothing ran, so there is no frame to grade — the important property is
    # that the run is not reported as complete.
    assert df.attrs.get("n_combos_run") == 0
    assert df.attrs.get("sweep_budget_stopped") is True


def test_an_unreadable_deadline_is_ignored_rather_than_truncating():
    """A budget probe must not be able to truncate work on its own failure."""
    seen = []

    def sim(cfg):
        seen.append(cfg["min_score"])
        return _stats(cfg)

    df = param_sweep._run_combos(
        _combos(3), sim, {DEADLINE_EPOCH_CONFIG_KEY: "next tuesday"},
    )
    assert len(seen) == 3
    assert df.attrs["sweep_budget_stopped"] is False


@pytest.mark.parametrize("reserve", [0.0, 10_000.0])
def test_the_reserve_is_what_protects_the_work_after_the_loop(monkeypatch, reserve):
    """The reserve exists so `_build_pit_parity_cscv_matrix`, stats assembly
    and the artifact write still have room. A large reserve must bind even
    when raw wall-clock remains."""
    monkeypatch.setattr(param_sweep, "SWEEP_BUDGET_RESERVE_S", reserve)
    monkeypatch.setattr(param_sweep, "SWEEP_FIRST_COMBO_ESTIMATE_S", 0.01)
    df = param_sweep._run_combos(
        _combos(2), _stats, {DEADLINE_EPOCH_CONFIG_KEY: time.time() + 60},
    )
    if reserve == 0.0:
        assert df.attrs["n_combos_run"] == 2
    else:
        assert df.attrs["n_combos_run"] == 0
