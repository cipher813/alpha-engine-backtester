"""
tests/test_smoke_phase_modes.py — per-phase smoke harness.

Covers the argparse surface, fixture application semantics, and budget
enforcement contract for the 4 smoke-<phase> modes. Does NOT exercise
end-to-end compute (that happens on the spot instance or via manual
invocation) — these are structural tests of the harness itself.

Motivated by ROADMAP Backtester P0 #3 "Per-phase smoke test harness".
"""

from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

import backtest


# ── Argparse surface ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", [
    "smoke-simulate",
    "smoke-param-sweep",
    "smoke-predictor-backtest",
    "smoke-phase4",
])
def test_smoke_phase_mode_accepted(mode: str):
    with patch.object(sys, "argv", ["backtest.py", "--mode", mode]):
        args = backtest._parse_args()
    assert args.mode == mode


def test_smoke_phase_mode_registry_exact():
    """Every mode declared in argparse choices must have an entry in
    _SMOKE_PHASE_MODES, and vice versa — drift between the two would
    silently leave a smoke mode without a fixture."""
    mode_registry_keys = set(backtest._SMOKE_PHASE_MODES.keys())
    assert mode_registry_keys == {
        "smoke-simulate", "smoke-param-sweep",
        "smoke-predictor-backtest", "smoke-phase4",
    }


def test_is_smoke_phase_mode():
    assert backtest._is_smoke_phase_mode("smoke-simulate") is True
    assert backtest._is_smoke_phase_mode("smoke-phase4") is True
    assert backtest._is_smoke_phase_mode("smoke") is False  # the legacy mode
    assert backtest._is_smoke_phase_mode("simulate") is False
    assert backtest._is_smoke_phase_mode("all") is False


# ── Fixture application ──────────────────────────────────────────────────────


def _blank_args(mode: str) -> argparse.Namespace:
    """Namespace with every attribute the fixture touches."""
    return argparse.Namespace(
        mode=mode,
        only_phases="",
        skip_phases="",
        force=False,
        freeze=False,
        date="2026-04-23",
        upload=True,
    )


def test_smoke_simulate_routes_to_simulate():
    args = _blank_args("smoke-simulate")
    config: dict = {}
    backtest._apply_smoke_fixture("smoke-simulate", args, config)

    assert args.mode == "simulate"
    assert config["max_signal_dates"] == 5
    assert config["min_simulation_dates"] == 2
    assert args.force is True  # always fresh — never auto-skip a smoke run
    assert args.freeze is True  # never promote synthetic recs to S3


@pytest.mark.parametrize("mode", [
    "smoke-simulate", "smoke-param-sweep",
    "smoke-predictor-backtest", "smoke-phase4",
])
def test_smoke_fixture_namespaces_date_to_isolate_markers(mode: str):
    """Fix for 2026-04-23 SF dry-run failure: smoke markers at
    backtest/{date}/.phases/ polluted full-run auto-skip on the same
    date. Fixture now prefixes args.date with ".smoke/" so markers +
    artifacts go to backtest/.smoke/{date}/ instead. Lex-sort places
    ".smoke/" before "2026-..." so production "latest date" probes
    keep resolving to real dates."""
    args = _blank_args(mode)
    config: dict = {}
    backtest._apply_smoke_fixture(mode, args, config)

    assert args.date == ".smoke/2026-04-23"
    # Critical: sort order stays safe — ".smoke/" < "2026-..."
    assert ".smoke/" < "2026-04-23"


@pytest.mark.parametrize("mode", [
    "smoke-simulate", "smoke-param-sweep",
    "smoke-predictor-backtest", "smoke-phase4",
])
def test_smoke_fixture_disables_upload(mode: str):
    """Smoke must not write top-level backtest/{date}/ artifacts (report,
    sweep_df, portfolio_stats). Namespacing args.date covers the phase
    artifacts; disabling args.upload covers the reporter.upload_to_s3
    path that writes the run summary to s3://.../backtest/{date}/."""
    args = _blank_args(mode)
    config: dict = {}
    backtest._apply_smoke_fixture(mode, args, config)

    assert args.upload is False


@pytest.mark.parametrize("mode", [
    "smoke-simulate", "smoke-param-sweep",
    "smoke-predictor-backtest", "smoke-phase4",
])
def test_every_smoke_mode_sets_smoke_tickers(mode: str):
    """Every smoke fixture must set `smoke_tickers` — it's the dominant
    speedup lever. Without it, the ArcticDB bulk read still pays full-
    universe cost and the smoke blows any reasonable budget (proven on
    the 2026-04-23 cold-spot dry-run: 5-date smoke-simulate took ~514s
    because max_signal_dates=5 alone didn't restrict the ticker axis)."""
    args = _blank_args(mode)
    config: dict = {}
    backtest._apply_smoke_fixture(mode, args, config)

    assert "smoke_tickers" in config, (
        f"Smoke fixture for {mode} must set smoke_tickers — otherwise "
        f"ArcticDB bulk reads pay full-universe cost"
    )
    tickers = config["smoke_tickers"]
    assert isinstance(tickers, (list, tuple, set))
    assert len(tickers) <= 20, (
        f"smoke_tickers for {mode} has {len(tickers)} entries — smoke "
        f"should be tiny (≤20) to keep runtime bounded"
    )
    assert all(isinstance(t, str) and t.isupper() and t.strip() for t in tickers), (
        f"smoke_tickers for {mode} must be a list of non-empty uppercase "
        f"ticker strings"
    )


def test_smoke_param_sweep_caps_combos_via_random_max_trials():
    """The fixture forces mode=random with max_trials=3 so _generate_random_combos
    samples exactly 3 combinations REGARDLESS of the effective grid size
    after config.yaml deep-merge. Observed 2026-04-23 post-bugfix smoke
    run: mode=grid produced 864 combos because config.yaml's param_sweep
    block merged into our 1-key override."""
    args = _blank_args("smoke-param-sweep")
    config: dict = {}
    backtest._apply_smoke_fixture("smoke-param-sweep", args, config)

    assert args.mode == "param-sweep"
    # Grid override hint for random sampling
    assert config["param_sweep"] == {"max_positions": [5, 10, 15]}
    # Critical: mode=random + max_trials=3 caps combos regardless of
    # grid size (fix for 2026-04-23 bug #5)
    settings = config["param_sweep_settings"]
    assert settings["mode"] == "random"
    assert settings["max_trials"] == 3
    # Seed pinned for deterministic smoke runs — same 3 combos each
    # invocation, so a regression shows as a repro'able timing change
    # rather than stochastic noise.
    assert settings["seed"] == 0


def test_smoke_predictor_backtest_shrinks_gbm_lookback():
    args = _blank_args("smoke-predictor-backtest")
    config: dict = {}
    backtest._apply_smoke_fixture("smoke-predictor-backtest", args, config)

    assert args.mode == "predictor-backtest"
    assert config["predictor_backtest"]["min_trading_days"] == 30
    assert config["predictor_backtest"]["max_trading_days"] == 60
    assert config["predictor_backtest"]["top_n_signals_per_day"] == 5
    # param_sweep nulled so only data_prep + single_run run
    assert config["param_sweep"] is None

    # only_phases restricts to data_prep + feature_maps + single_run (NOT phase4)
    only = set(args.only_phases.split(","))
    assert "predictor_data_prep" in only
    assert "predictor_single_run" in only
    assert "phase4a_ensemble_modes" not in only


def test_smoke_phase4_includes_phase4_evaluators():
    args = _blank_args("smoke-phase4")
    config: dict = {}
    backtest._apply_smoke_fixture("smoke-phase4", args, config)

    only = set(args.only_phases.split(","))
    assert "phase4a_ensemble_modes" in only
    assert "phase4b_signal_thresholds" in only
    assert "phase4c_feature_pruning" in only
    # Still includes data_prep + single_run as upstream deps
    assert "predictor_data_prep" in only
    assert "predictor_single_run" in only


def test_fixture_deep_merge_preserves_sibling_keys():
    """A fixture override for predictor_backtest.min_trading_days must
    not wipe out other predictor_backtest keys already in config."""
    args = _blank_args("smoke-predictor-backtest")
    config = {
        "predictor_backtest": {
            "existing_key": "untouched",
            "min_trading_days": 9999,  # will be overridden
        },
    }
    backtest._apply_smoke_fixture("smoke-predictor-backtest", args, config)
    assert config["predictor_backtest"]["existing_key"] == "untouched"
    assert config["predictor_backtest"]["min_trading_days"] == 30


def test_fixture_does_not_widen_operator_only_phases():
    """If the operator passed --only-phases=X, smoke doesn't widen to
    include phases they excluded. Narrowing-only semantics."""
    args = _blank_args("smoke-phase4")
    args.only_phases = "phase4a_ensemble_modes"
    config: dict = {}
    backtest._apply_smoke_fixture("smoke-phase4", args, config)
    # Operator's explicit filter is preserved, not overwritten by the full
    # list the smoke mode would have used.
    assert args.only_phases == "phase4a_ensemble_modes"


# ── Budget enforcement ───────────────────────────────────────────────────────


def test_budget_check_passes_within_budget(caplog):
    """Under-budget run logs an INFO line and returns cleanly."""
    import logging
    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-simulate": 100.0}):
        with caplog.at_level(logging.INFO, logger="backtest"):
            backtest._assert_smoke_within_budget("smoke-simulate", 42.0)
    assert any("PASSED budget check" in rec.getMessage() for rec in caplog.records)


def test_budget_check_fails_over_budget():
    """Over-budget run must SystemExit 2 with a loud message."""
    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-simulate": 60.0}):
        with pytest.raises(SystemExit, match="BUDGET EXCEEDED"):
            backtest._assert_smoke_within_budget("smoke-simulate", 120.0)


def test_budget_check_warns_when_missing(caplog):
    """No budget declared → log warning, don't fail."""
    import logging
    with patch.object(backtest, "_load_timing_budgets", return_value={}):
        with caplog.at_level(logging.WARNING, logger="backtest"):
            backtest._assert_smoke_within_budget("smoke-simulate", 42.0)
    assert any("no budget declared" in rec.getMessage() for rec in caplog.records)


# ── Integration-light: end-to-end fixture + routing sanity ───────────────────


def test_all_smoke_modes_route_to_valid_full_modes():
    """Every smoke-<phase>'s route_mode must be a real --mode choice."""
    full_mode_choices = {"simulate", "param-sweep", "all", "predictor-backtest", "smoke"}
    for mode, spec in backtest._SMOKE_PHASE_MODES.items():
        assert spec["route_mode"] in full_mode_choices, (
            f"Smoke mode {mode} routes to invalid full mode {spec['route_mode']!r}"
        )


def test_timing_budget_file_has_entry_for_every_smoke_mode():
    """Every registered smoke mode must have a budget entry — otherwise
    regressions slip through with a soft warning instead of a hard fail."""
    budgets = backtest._load_timing_budgets()
    for mode in backtest._SMOKE_PHASE_MODES:
        assert mode in budgets, (
            f"Smoke mode {mode!r} missing from timing_budget.yaml. "
            f"Add an entry so regressions fail CI instead of warn silently."
        )
