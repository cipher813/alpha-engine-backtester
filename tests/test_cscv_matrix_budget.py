"""The CSCV block-matrix build stops on its budget, and the sweep reserves for
it (alpha-engine-config-I7309 blocker 1, second half).

WHAT THE FIRST HALF MISSED
--------------------------
`crucible-backtester#729` bounded the param_sweep combo loop on the pass's
`_pass_deadline_epoch`, which is correct and necessary. It is not sufficient:
`backtest._build_pit_parity_cscv_matrix` runs immediately afterwards, inside
the SAME 2700 s subprocess ceiling, and had no bound at all.

Its cost at the canonical defaults, using the same measured per-combo number
the sweep fix was sized from (`backtest/2026-07-31/pit_parity.json`:
"Sweep combo 5/60 done in 272.5s"):

    n_blocks x top_k simulations, each over ~1/n_blocks of the date grid
    = 272.5 s x 12 combos x 8 blocks / 8   ~= 3,270 s

That is MORE than the entire 2700 s pass ceiling, held back behind a flat
600 s reserve. So bounding only the combo loop moved the SIGKILL from the
sweep to the matrix build rather than removing it, and the pass still had
nothing to publish — the artifact would read `status: failed`,
`timed_out: true`, and the parity branch would route RESOURCE_KILL, halting
the weekly SF (alpha-engine-config-I7267 routing, live in the definition).

TWO CHANGES, AND THE SECOND IS THE ONE THAT MAKES THE PBO SURVIVE
-----------------------------------------------------------------
1. The matrix build is self-deadlining, at WHOLE BLOCK ROW granularity. Not
   per-cell: a row missing its lower-ranked combos biases the exact ordering
   the PBO measures, so a truncated matrix must be smaller and rectangular,
   never ragged.
2. The sweep's reserve is DERIVED from the matrix build's own arithmetic
   (`param_sweep.cscv_reserve_s`) instead of being a flat 600 s. The work it
   covers scales with the combos the sweep produces, so a constant is wrong in
   both directions — too small and the matrix is killed, too large and combos
   are given up to protect a matrix that never needed the room.

The derived reserve is self-limiting on purpose: each extra combo raises the
reserve by a fraction of its own cost, so the loop stops where one more combo
would cost more than validating the combos already in hand.
"""

from __future__ import annotations

import logging
import time

import pandas as pd
import pytest

import backtest
from analysis import param_sweep
from deadline_budget import DEADLINE_EPOCH_CONFIG_KEY

# The measured per-combo cost this whole budget arithmetic is sized from.
MEASURED_COMBO_S = 272.5
# analysis/pit_parity.py::_PIT_PARITY_PASS_TIMEOUT minus its handover margin —
# the budget a pass child actually receives.
PASS_BUDGET_S = 2700 - 120


def _sweep_df(n_combos: int, **attrs) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"min_score": 60 + i, "sortino_ratio": 2.0 - i * 0.01} for i in range(n_combos)]
    )
    for k, v in attrs.items():
        df.attrs[k] = v
    return df


def _config(**over) -> dict:
    cfg = {"param_sweep": {"min_score": [60, 61, 62]}}
    cfg.update(over)
    return cfg


def _sim(_cfg: dict) -> dict:
    return {"sortino_ratio": 1.23}


def _dates(n: int = 800) -> list[str]:
    return [f"d{i:04d}" for i in range(n)]


# ── No deadline → nothing changes ──────────────────────────────────────────

def test_without_a_deadline_every_block_row_is_built():
    """The PredictorBacktest-phase call site (config#6032 reuse path) runs in
    the main pipeline process and hands down no `_pass_deadline_epoch`. That
    path must behave exactly as it did before."""
    out = backtest._build_pit_parity_cscv_matrix(
        _sweep_df(6), _sim, _dates(), _config(),
    )
    assert out["_cscv_budget_stopped"] is False
    assert out["_cscv_n_blocks_run"] == out["_cscv_n_blocks_planned"] == 8
    assert len(out["_cscv_block_matrix"]) == 8


# ── An already-blown budget builds nothing rather than starting a row ──────

def test_an_already_blown_budget_builds_zero_rows_and_says_so():
    """Zero rows is not a small matrix, it is NO matrix — the compare must
    report pbo null (honest N/A), and the artifact must record WHY. An absent
    explanation is what made this class invisible for three Saturdays."""
    cfg = _config(**{DEADLINE_EPOCH_CONFIG_KEY: time.time() - 10})
    out = backtest._build_pit_parity_cscv_matrix(_sweep_df(6), _sim, _dates(), cfg)

    assert "_cscv_block_matrix" not in out
    assert out["_cscv_n_blocks_run"] == 0
    assert out["_cscv_budget_stopped"] is True
    assert out["_cscv_n_blocks_skipped_for_budget"] == out["_cscv_n_blocks_planned"]


# ── A mid-flight deadline yields a SMALLER, RECTANGULAR matrix ─────────────

def test_a_mid_flight_deadline_truncates_at_a_row_boundary(monkeypatch):
    """The rows that were built are complete; the matrix stays rectangular.
    A ragged matrix would drop exactly the lower-ranked combos, biasing the
    ordering the PBO exists to measure.

    Scaled clock: the loop's own OBSERVED per-row cost supersedes the seed
    after the first row (that is `next_unit_affordable`'s contract), so a
    truncation test needs simulations that actually consume the clock.
    """
    monkeypatch.setattr(param_sweep, "CSCV_BUDGET_RESERVE_S", 0.0)
    n_combos = 5
    cell_s = 0.01
    row_s = cell_s * n_combos

    def slow_sim(_cfg):
        time.sleep(cell_s)
        return {"sortino_ratio": 1.23}

    cfg = _config(**{DEADLINE_EPOCH_CONFIG_KEY: time.time() + row_s * 3.4})
    df = _sweep_df(n_combos, combo_p90_s=row_s * 8.0 / n_combos)
    out = backtest._build_pit_parity_cscv_matrix(df, slow_sim, _dates(), cfg)

    matrix = out["_cscv_block_matrix"]
    assert 0 < len(matrix) < 8
    assert {len(row) for row in matrix} == {n_combos}, "matrix must be rectangular"
    assert out["_cscv_budget_stopped"] is True
    assert out["_cscv_n_blocks_run"] == len(matrix)
    assert (
        out["_cscv_n_blocks_run"] + out["_cscv_n_blocks_skipped_for_budget"]
        == out["_cscv_n_blocks_planned"]
    )


def test_the_budget_stop_logs_at_error():
    """A truncated matrix is a real coverage loss. It is not a debug detail."""
    cfg = _config(**{DEADLINE_EPOCH_CONFIG_KEY: time.time() - 10})
    logger = logging.getLogger("backtest")
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record):  # noqa: D102
            records.append(record)

    handler = _Capture()
    logger.addHandler(handler)
    try:
        backtest._build_pit_parity_cscv_matrix(_sweep_df(6), _sim, _dates(), cfg)
    finally:
        logger.removeHandler(handler)

    assert any(r.levelno >= logging.ERROR for r in records)


def test_the_first_row_estimate_is_seeded_from_the_sweeps_own_measurement(monkeypatch):
    """`combo_p90_s` is what the sweep measured THIS run. Using it — rather
    than the deliberately-high constant seed — is what lets the first row's
    affordability question be asked with a number instead of a guess.

    The budget below sits BETWEEN the two estimates, so it discriminates: with
    the sweep's measurement the loop starts, without it the fallback seed
    (larger, because it errs high on purpose) correctly declines.
    """
    monkeypatch.setattr(param_sweep, "CSCV_BUDGET_RESERVE_S", 0.0)
    n_combos = 4
    seeded = MEASURED_COMBO_S * n_combos / 8.0
    unseeded = param_sweep.SWEEP_FIRST_COMBO_ESTIMATE_S * n_combos / 8.0
    assert seeded < unseeded, "the fallback seed must err high"
    budget = (seeded + unseeded) / 2.0

    cfg = _config(**{DEADLINE_EPOCH_CONFIG_KEY: time.time() + budget})
    out = backtest._build_pit_parity_cscv_matrix(
        _sweep_df(n_combos, combo_p90_s=MEASURED_COMBO_S), _sim, _dates(), cfg,
    )
    assert out["_cscv_n_blocks_run"] >= 1

    cfg_unseeded = _config(**{DEADLINE_EPOCH_CONFIG_KEY: time.time() + budget})
    out_unseeded = backtest._build_pit_parity_cscv_matrix(
        _sweep_df(n_combos), _sim, _dates(), cfg_unseeded,
    )
    assert out_unseeded["_cscv_n_blocks_run"] == 0


# ── The derived reserve ────────────────────────────────────────────────────

def test_reserve_is_the_flat_constant_when_no_cscv_follows():
    """Every caller that is not the pit_parity walk-forward pass keeps the
    reserve it had."""
    assert param_sweep.cscv_reserve_s({}, [100.0], 3) == param_sweep.SWEEP_BUDGET_RESERVE_S


def test_reserve_covers_a_minimum_viable_cscv_at_the_observed_cost():
    """The reserve's whole job: after the sweep stops, `CSCV_MIN_BLOCKS` rows
    over the top-K in hand must still fit."""
    cfg = _config(pit_parity_sweep=True)
    n_done = 5
    reserve = param_sweep.cscv_reserve_s(cfg, [MEASURED_COMBO_S] * n_done, n_done)
    top_k = min(n_done + 1, param_sweep.CSCV_DEFAULT_TOP_K)
    expected_rows = (
        MEASURED_COMBO_S * top_k * param_sweep.CSCV_MIN_BLOCKS
        / param_sweep.CSCV_DEFAULT_N_BLOCKS
    )
    assert reserve == pytest.approx(
        param_sweep.CSCV_BUDGET_RESERVE_S + expected_rows
    )


def test_reserve_grows_with_combos_done_and_caps_at_top_k():
    """Each extra combo raises the reserve by a fraction of its own cost —
    that is the self-limiting property. It stops growing at top-K because the
    matrix only ever re-evaluates the top-K."""
    cfg = _config(pit_parity_sweep=True)
    samples = [MEASURED_COMBO_S] * 30
    reserves = [param_sweep.cscv_reserve_s(cfg, samples, k) for k in range(0, 20)]
    assert reserves == sorted(reserves)
    top_k = param_sweep.CSCV_DEFAULT_TOP_K
    assert reserves[top_k] == reserves[-1], "reserve must plateau at top-K"


def test_the_whole_walk_forward_pass_now_fits_inside_its_ceiling():
    """The regression that names the blocker.

    Simulate the two loops against the measured per-combo cost under the real
    pass budget, and assert the pass finishes with the artifact-write reserve
    intact. Before the derived reserve this arithmetic overran: the sweep
    stopped at a flat 600 s of headroom and the matrix build then needed
    ~3,270 s of it.
    """
    cfg = _config(pit_parity_sweep=True)
    spent = 0.0
    combos_done = 0
    combo_seconds: list[float] = []
    # The sweep, as _run_combos runs it.
    while combos_done < 60:
        reserve = param_sweep.cscv_reserve_s(cfg, combo_seconds, combos_done)
        if (PASS_BUDGET_S - spent) - reserve < MEASURED_COMBO_S:
            break
        spent += MEASURED_COMBO_S
        combo_seconds.append(MEASURED_COMBO_S)
        combos_done += 1

    assert combos_done >= 2, "a sweep that cannot fund two combos funds no CSCV"

    # The matrix build, as _build_pit_parity_cscv_matrix runs it.
    top_k = min(combos_done, param_sweep.CSCV_DEFAULT_TOP_K)
    row_s = MEASURED_COMBO_S * top_k / param_sweep.CSCV_DEFAULT_N_BLOCKS
    rows = 0
    while rows < param_sweep.CSCV_DEFAULT_N_BLOCKS:
        if (PASS_BUDGET_S - spent) - param_sweep.CSCV_BUDGET_RESERVE_S < row_s:
            break
        spent += row_s
        rows += 1

    assert spent <= PASS_BUDGET_S, "the pass must not overrun its own ceiling"
    assert PASS_BUDGET_S - spent >= param_sweep.CSCV_BUDGET_RESERVE_S * 0.5, (
        "the artifact write must still have room"
    )
    assert rows >= param_sweep.CSCV_MIN_BLOCKS, (
        "the reserve exists precisely so a computable PBO survives the budget"
    )


# ── The numbers reach a durable artifact ───────────────────────────────────

def test_sweep_budget_attrs_names_exactly_what_run_combos_publishes():
    """A key added to `_run_combos` and forgotten in `SWEEP_BUDGET_ATTRS` is a
    number that exists only in a log on a self-terminating spot instance."""
    df = param_sweep._run_combos([{"min_score": 60}], lambda c: {"sortino_ratio": 1.0}, {})
    published = {k for k in df.attrs if not k.startswith("_")}
    assert published == set(param_sweep.SWEEP_BUDGET_ATTRS)


def test_the_pass_artifact_carries_both_halves_of_the_budget():
    """sf-pipeline-policy §2.3a rule 4's second obligation, one layer down:
    emit BOTH sides, always. A budget block that appeared only when something
    truncated could not be told apart from a producer that stopped emitting."""
    from analysis import pit_stats_artifact as psa

    stats = {
        "status": "ok",
        "sortino_ratio": 1.0,
        "_sweep_budget": {
            "n_combos_planned": 60,
            "n_combos_run": 5,
            "n_combos_skipped_for_budget": 55,
            "sweep_budget_stopped": True,
            "combo_p90_s": MEASURED_COMBO_S,
            "cscv_reserve_s": 998.5,
        },
        "_cscv_n_blocks_planned": 8,
        "_cscv_n_blocks_run": 6,
        "_cscv_n_blocks_skipped_for_budget": 2,
        "_cscv_budget_stopped": True,
        "_cscv_block_p90_s": 170.3,
    }
    art = psa.build_pass_artifact(stats, "walkforward", "2026-08-29", 2410.0)
    psa.validate_pass_artifact(art)

    assert art["budget"]["sweep"]["n_combos_run"] == 5
    assert art["budget"]["sweep"]["budget_stopped"] is True
    assert art["budget"]["sweep"]["cscv_reserve_s"] == pytest.approx(998.5)
    assert art["budget"]["cscv"]["n_blocks_run"] == 6
    assert art["budget"]["cscv"]["budget_stopped"] is True


def test_an_untruncated_pass_still_emits_the_budget_block():
    """The ran-regardless side. Absence of the block must mean 'nobody
    measured', never 'nothing was truncated'."""
    from analysis import pit_stats_artifact as psa

    art = psa.build_pass_artifact(
        {"status": "ok", "sortino_ratio": 1.0}, "walkforward", "2026-08-29", 900.0,
    )
    psa.validate_pass_artifact(art)
    assert "budget" in art
    # Unmeasured is null, never a False/0 that reads as "measured and clean".
    assert art["budget"]["sweep"]["budget_stopped"] is None
    assert art["budget"]["cscv"]["n_blocks_run"] is None
