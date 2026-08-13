"""Pins each alpha-engine-config-I4442 per-stage script to the EXACT
invocation the corresponding nousergon-data Saturday SF state currently
makes against the monolith (infrastructure/step_function.json, states
Backtester / PredictorBacktest / PortfolioOptimizerBacktest / Parity /
Evaluator). Verified against nousergon-data origin/main 2026-08-09:

  Backtester                 spot_backtest.sh --mode=param-sweep --no-pit-parity --skip-stages=parity,evaluator
  PredictorBacktest          spot_backtest.sh --mode=predictor-backtest --no-pit-parity --skip-stages=parity,evaluator
  PortfolioOptimizerBacktest spot_backtest.sh --mode=portfolio-optimizer-backtest --no-pit-parity --skip-stages=parity,evaluator
  Parity                     spot_backtest.sh --pit-parity-enabled=1 --skip-stages=backtest,evaluator
  Evaluator                  spot_backtest.sh --no-pit-parity --skip-stages=backtest,parity

These tests do NOT read step_function.json (it lives in a different repo,
nousergon-data) — they pin the semantics INTO this repo's own scripts, so a
future edit to either side that breaks the correspondence is caught here
(the crucible-backtester side) and by nousergon-data's own SF-definition
tests (the other side) rather than only being discoverable by manually
diffing both repos.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"


def _heredoc_body(script: str, start_token: str, end_marker: str) -> str:
    text = (_INFRA / script).read_text()
    start = text.index(start_token)
    end = text.index(f"\n{end_marker}\n", start)
    return text[start:end]


def test_backtester_runs_only_param_sweep_no_pit_parity():
    body = _heredoc_body("spot_backtester.sh", "run_ssm \"backtest\"", "BACKTEST")
    assert "--mode param-sweep" in body
    assert "--pit-parity" not in body
    assert "pytest tests/test_parity_replay.py" not in body
    assert "evaluate.py" not in body


def test_predictor_backtest_runs_only_predictor_backtest_mode_no_pit_parity():
    body = _heredoc_body("spot_predictor_backtest.sh", "run_ssm \"predictor-backtest\"", "BACKTEST")
    assert "--mode predictor-backtest" in body
    assert "--pit-parity" not in body
    assert "pytest tests/test_parity_replay.py" not in body
    assert "evaluate.py" not in body


def test_portfolio_optimizer_backtest_runs_only_that_mode_no_pit_parity():
    body = _heredoc_body(
        "spot_portfolio_optimizer_backtest.sh", "run_ssm \"portfolio-optimizer\"", "BACKTEST"
    )
    assert "--mode portfolio-optimizer-backtest" in body
    assert "--pit-parity" not in body
    assert "pytest tests/test_parity_replay.py" not in body
    assert "evaluate.py" not in body


def test_parity_runs_pit_parity_and_parity_pytest_only():
    body = _heredoc_body("spot_parity.sh", "run_ssm \"parity\"", "PARITY")
    # pit_parity fires here (the ONLY stage where it's enabled, per L4486 —
    # the SF passes --pit-parity-enabled=1 --skip-stages=backtest,evaluator,
    # which does NOT include pit_parity in the skip-set).
    assert "--mode predictor-backtest --pit-parity" in body
    assert "pytest tests/test_parity_replay.py -m parity" in body
    # Neither the simulation backtest (--mode param-sweep / --mode all /
    # --upload sweep write) nor the evaluator runs here.
    assert "--upload" not in body
    assert "evaluate.py" not in body


def test_evaluator_runs_only_evaluate_py_with_skip_backtester():
    body = _heredoc_body("spot_evaluator.sh", "run_ssm \"evaluator\"", "EVALUATOR")
    assert "evaluate.py" in body
    assert "--upload" in body
    assert "--skip-backtester" in body, (
        "the Evaluator SF state always skips the backtest stage (it now lives "
        "in the three separate spot_*backtest*.sh scripts) — --skip-backtester "
        "must always be passed, not conditionally (config#2887)"
    )
    assert "backtest.py" not in body
    assert "pytest tests/test_parity_replay.py" not in body


@pytest.mark.parametrize(
    "script,floor_expected",
    [
        # alpha-engine-config-I7216: flipped False -> True. param-sweep was
        # carved out of the floor because it does not run predictor_pipeline —
        # true, and irrelevant: it reads the same ArcticDB feature store over
        # ~900 tickers and was OOM-killed on a 4 GB c5.large on 2026-08-13,
        # freezing the live entry feed for days.
        ("spot_backtester.sh", True),
        ("spot_predictor_backtest.sh", True),
        ("spot_portfolio_optimizer_backtest.sh", True),
        ("spot_parity.sh", True),
        ("spot_evaluator.sh", False),
    ],
)
def test_predictor_ram_floor_matches_monolith_case_statement(script, floor_expected):
    """The monolith applies the >=16GB floor for
    `all|predictor-backtest|portfolio-optimizer-backtest` modes (I3280).
    pit_parity (which the Parity script runs) shares the same universal
    floor. The Evaluator does not run predictor_pipeline and stays on the
    cheap default rotation.

    param-sweep (Backtester) USED to be in that second group and no longer is
    (alpha-engine-config-I7216): "does not run predictor_pipeline" was taken to
    mean "is not memory-bound", which production falsified with a kernel OOM
    kill on a 4 GB c5.large. Its floor is not evidence of a measured
    requirement — an OOM-killed process reports no peak — it is a ceiling to
    trade under until one exists."""
    text = (_INFRA / script).read_text()
    has_floor_call = "spot_common_apply_predictor_ram_floor" in text
    assert has_floor_call == floor_expected, (
        f"{script}: expected spot_common_apply_predictor_ram_floor() present={floor_expected}"
    )


@pytest.mark.parametrize(
    "script,predictor_required",
    [
        ("spot_backtester.sh", False),
        ("spot_predictor_backtest.sh", True),
        ("spot_portfolio_optimizer_backtest.sh", True),
        ("spot_parity.sh", False),
        ("spot_evaluator.sh", False),
    ],
)
def test_predictor_config_required_matches_stage_needs(script, predictor_required):
    """Only the two stages that RUN predictor_pipeline as their primary
    deliverable (predictor-backtest, portfolio-optimizer-backtest) hard-fail
    on a missing predictor.yaml — a per-stage improvement over the
    monolith's uniform soft-skip (that script couldn't know per-invocation
    whether the mode needed it; each split script does). Parity's
    pit_parity pass stays soft-skip because it is explicitly non-blocking
    by design; the evaluator doesn't run predictor_pipeline at all."""
    text = (_INFRA / script).read_text()
    expected_call = f"spot_common_resolve_predictor_config {'1' if predictor_required else '0'}"
    assert expected_call in text, f"{script}: expected `{expected_call}`"
