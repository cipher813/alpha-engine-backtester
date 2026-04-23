"""
tests/test_smoke_harness_hardening.py — bugs fixed post 2026-04-23 dry-run.

Covers:
- _deepcopy_safe_config strips non-deepcopy-safe runtime keys
- PhaseRegistry.phase_errors tracks in-invocation error markers
- _assert_smoke_within_budget fails on phase_errors even when wall-clock
  is under budget (false-PASS guard)
- Smoke fixtures include preflight + runtime_smoke in only_phases so
  they actually run instead of getting SKIP-but-body-still-executes
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import backtest
from analysis.param_sweep import _deepcopy_safe_config
from pipeline_common import PhaseRegistry
from tests.test_phase_registry import _FakeS3


# ── _deepcopy_safe_config ───────────────────────────────────────────────────


class _UnDeepcopyable:
    """Stand-in for a boto3 client — raises on deepcopy."""
    def __deepcopy__(self, memo):
        raise RuntimeError("boto3 clients can't be deepcopied cleanly")


def test_deepcopy_safe_strips_underscore_prefixed_runtime_refs():
    """Non-deepcopy-safe objects stored under underscore keys (the runtime
    convention) must not be deepcopied — they're re-attached shallow to
    the copy instead. Without this the 2026-04-23 smoke-param-sweep run
    hit `maximum recursion depth exceeded`."""
    unsafe = _UnDeepcopyable()
    base = {
        "max_positions": 10,
        "nested": {"a": 1, "b": [2, 3]},
        "_phase_registry": unsafe,
        "_s3_client": unsafe,
    }
    copied = _deepcopy_safe_config(base)

    # Regular keys deepcopied — nested dict is independent
    assert copied["max_positions"] == 10
    assert copied["nested"] == base["nested"]
    assert copied["nested"] is not base["nested"]  # deep-copied

    # Runtime keys re-attached shallow — same identity
    assert copied["_phase_registry"] is unsafe
    assert copied["_s3_client"] is unsafe


def test_deepcopy_safe_without_underscore_keys_is_just_deepcopy():
    base = {"a": 1, "b": {"c": 2}}
    copied = _deepcopy_safe_config(base)
    assert copied == base
    assert copied["b"] is not base["b"]


def test_deepcopy_safe_preserves_base_dict():
    """Caller's base config must be unchanged after the helper runs."""
    unsafe = _UnDeepcopyable()
    base = {"x": 1, "_registry": unsafe}
    _deepcopy_safe_config(base)
    assert base["x"] == 1
    assert base["_registry"] is unsafe
    assert set(base.keys()) == {"x", "_registry"}


# ── PhaseRegistry.phase_errors tracking ─────────────────────────────────────


def test_phase_errors_tracks_in_invocation_errors():
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)

    assert r.phase_errors == []

    with pytest.raises(RuntimeError, match="boom"):
        with r.phase("doomed_phase"):
            raise RuntimeError("boom")

    assert r.phase_errors == ["doomed_phase"]


def test_phase_errors_multiple_errors_accumulate():
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)

    with pytest.raises(RuntimeError):
        with r.phase("first_bad"):
            raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        with r.phase("second_bad"):
            raise RuntimeError("y")
    with r.phase("good_one"):
        pass

    assert r.phase_errors == ["first_bad", "second_bad"]


def test_phase_errors_clean_run_stays_empty():
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)

    with r.phase("p1"):
        pass
    with r.phase("p2"):
        pass

    assert r.phase_errors == []


# ── _assert_smoke_within_budget inner-error guard ──────────────────────────


def test_budget_check_fails_on_inner_phase_error_under_budget():
    """False-PASS guard: wall-clock 96s < 500s budget, but inner phase
    errored → must SystemExit with actionable message."""
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    r.phase_errors = ["param_sweep"]

    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-param-sweep": 500.0}):
        with pytest.raises(SystemExit, match="inner phase.*param_sweep"):
            backtest._assert_smoke_within_budget(
                "smoke-param-sweep", 96.0, registry=r,
            )


def test_budget_check_passes_clean_run_with_registry():
    """Registry passed, no errors, under budget → normal PASS path."""
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    # phase_errors left empty

    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-simulate": 400.0}):
        # Should not raise
        backtest._assert_smoke_within_budget(
            "smoke-simulate", 120.0, registry=r,
        )


def test_budget_check_inner_error_trumps_budget_pass(caplog):
    """Even if budget check would pass, inner error still fails smoke."""
    s3 = _FakeS3()
    r = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    r.phase_errors = ["simulation_setup"]

    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-simulate": 9999.0}):
        with pytest.raises(SystemExit) as exc_info:
            backtest._assert_smoke_within_budget(
                "smoke-simulate", 1.0, registry=r,
            )
    # Error message should name the failing phase + distinguish from budget exceed
    assert "simulation_setup" in str(exc_info.value)
    assert "FAILED" in str(exc_info.value)


def test_budget_check_no_registry_keeps_legacy_behavior():
    """Backward compat: when registry=None, only wall-clock matters."""
    with patch.object(backtest, "_load_timing_budgets",
                      return_value={"smoke-simulate": 400.0}):
        # Should not raise (under budget, no registry to inspect)
        backtest._assert_smoke_within_budget("smoke-simulate", 120.0)


# ── Fixture only_phases inclusion ───────────────────────────────────────────


@pytest.mark.parametrize("mode", [
    "smoke-predictor-backtest",
    "smoke-phase4",
])
def test_smoke_fixture_includes_preflight_and_runtime_smoke(mode):
    """Fix for 2026-04-23 bug #3: only_phases-restricted smoke modes
    need preflight + runtime_smoke in the list, otherwise the phases'
    bodies execute silently (bodies don't check ctx.skipped) while
    their PHASE log says SKIP. Waste ~90s per invocation."""
    spec = backtest._SMOKE_PHASE_MODES[mode]
    assert "preflight" in spec["only_phases"], (
        f"{mode} fixture missing 'preflight' — body will run without "
        f"PHASE markers per 2026-04-23 bug #3"
    )
    assert "runtime_smoke" in spec["only_phases"], (
        f"{mode} fixture missing 'runtime_smoke'"
    )


@pytest.mark.parametrize("mode", [
    "smoke-simulate",
    "smoke-param-sweep",
])
def test_smoke_modes_without_only_phases_keep_unconstrained(mode):
    """Symmetry check: smoke modes that don't use only_phases at all
    keep that shape — adding preflight etc. only matters when the list
    is non-empty."""
    spec = backtest._SMOKE_PHASE_MODES[mode]
    assert spec["only_phases"] is None, (
        f"{mode} was changed to use only_phases — the inclusion guard "
        f"above would need to cover it too"
    )
