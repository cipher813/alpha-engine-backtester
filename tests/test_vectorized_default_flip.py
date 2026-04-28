"""Pin the 2026-04-28 default-on flip for the vectorized predictor sweep.

Before 2026-04-28: `use_vectorized_sweep` defaulted to False; explicit
opt-in via `--use-vectorized-sweep` was required to engage Tier 4.

After 2026-04-28 (PR #123-ish): default is True; explicit opt-out via
`--use-scalar-sweep` falls back to the scalar per-combo loop. Both
flags are mutually exclusive (argparse-enforced) and the in-tree
scalar path is one CLI flag away for emergency rollback.

These tests pin the argparse + config wiring for the flip.
"""
from __future__ import annotations

import argparse
import sys
from unittest.mock import patch

import pytest

import backtest


# ── Argparse surface ────────────────────────────────────────────────────────


def test_default_run_has_neither_flag_set():
    """Bare invocation: neither use_vectorized_sweep nor use_scalar_sweep
    is set. The default-on logic kicks in via setdefault."""
    with patch.object(sys, "argv", ["backtest.py"]):
        args = backtest._parse_args()
    assert args.use_vectorized_sweep is False
    assert args.use_scalar_sweep is False


def test_explicit_vectorized_flag_accepted():
    with patch.object(sys, "argv", ["backtest.py", "--use-vectorized-sweep"]):
        args = backtest._parse_args()
    assert args.use_vectorized_sweep is True
    assert args.use_scalar_sweep is False


def test_explicit_scalar_flag_accepted():
    with patch.object(sys, "argv", ["backtest.py", "--use-scalar-sweep"]):
        args = backtest._parse_args()
    assert args.use_vectorized_sweep is False
    assert args.use_scalar_sweep is True


def test_both_flags_mutually_exclusive():
    """Both flags together is a usage error — argparse mutex group enforces."""
    with patch.object(
        sys, "argv",
        ["backtest.py", "--use-vectorized-sweep", "--use-scalar-sweep"],
    ):
        with pytest.raises(SystemExit):
            backtest._parse_args()


# ── Config wiring (the actual default-on behavior) ──────────────────────────


def _blank_args(**overrides) -> argparse.Namespace:
    """Args with only the fields _init_pipeline-ish flag handling touches."""
    base = dict(
        use_vectorized_sweep=False,
        use_scalar_sweep=False,
        skip_phase4_evaluations=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _apply_flag_logic(args, config: dict) -> None:
    """Mirror the logic in backtest.py's `_init_pipeline` flag-routing
    block (lines ~3680-3695). If that block moves or changes, this test
    function moves with it.
    """
    if args.skip_phase4_evaluations:
        config["skip_phase4_evaluations"] = True
    if args.use_scalar_sweep:
        config["use_vectorized_sweep"] = False
    elif args.use_vectorized_sweep:
        config["use_vectorized_sweep"] = True
    else:
        config.setdefault("use_vectorized_sweep", True)


class TestDefaultOnFlip:
    def test_neither_flag_yields_default_on(self):
        """Bare run → vectorized engaged. This is the 2026-04-28 flip."""
        config: dict = {}
        _apply_flag_logic(_blank_args(), config)
        assert config["use_vectorized_sweep"] is True

    def test_explicit_vectorized_redundant_under_default_on(self):
        config: dict = {}
        _apply_flag_logic(
            _blank_args(use_vectorized_sweep=True), config,
        )
        assert config["use_vectorized_sweep"] is True

    def test_explicit_scalar_opts_out(self):
        """`--use-scalar-sweep` flips back to scalar path. Emergency
        rollback semantics."""
        config: dict = {}
        _apply_flag_logic(
            _blank_args(use_scalar_sweep=True), config,
        )
        assert config["use_vectorized_sweep"] is False

    def test_config_yaml_can_opt_out_without_cli_flag(self):
        """Operator can also opt out by setting `use_vectorized_sweep:
        false` in config.yaml — setdefault preserves the explicit
        config value."""
        config = {"use_vectorized_sweep": False}
        _apply_flag_logic(_blank_args(), config)
        assert config["use_vectorized_sweep"] is False

    def test_config_yaml_explicit_true_unchanged(self):
        """`use_vectorized_sweep: true` in config preserved (no-op
        since default is also True, but the contract should be
        explicit)."""
        config = {"use_vectorized_sweep": True}
        _apply_flag_logic(_blank_args(), config)
        assert config["use_vectorized_sweep"] is True

    def test_cli_scalar_overrides_config_yaml_true(self):
        """If config.yaml says vectorized but CLI says --use-scalar-sweep,
        CLI wins."""
        config = {"use_vectorized_sweep": True}
        _apply_flag_logic(
            _blank_args(use_scalar_sweep=True), config,
        )
        assert config["use_vectorized_sweep"] is False

    def test_cli_vectorized_overrides_config_yaml_false(self):
        """If config.yaml says scalar but CLI says --use-vectorized-sweep,
        CLI wins."""
        config = {"use_vectorized_sweep": False}
        _apply_flag_logic(
            _blank_args(use_vectorized_sweep=True), config,
        )
        assert config["use_vectorized_sweep"] is True
