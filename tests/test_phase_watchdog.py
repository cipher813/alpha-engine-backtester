"""
tests/test_phase_watchdog.py — Phase-watchdog tripwire.

Covers:
  - _start_watchdog fires after cap_s
  - _start_watchdog cancellation prevents fire
  - PhaseRegistry.phase() raises PhaseTimeoutError when the watchdog
    trips (vs plain KeyboardInterrupt from operator Ctrl+C)
  - PhaseRegistry.phase() without a cap keeps pre-watchdog behavior
  - load_phase_hard_caps parses the timing_budget.yaml block correctly

Motivated by the 2026-04-22 4th Saturday SF dry-run — 110 minutes of
silent compute burn with zero diagnostic signal. Watchdog converts that
into "cap-minute abort + all-thread stack dump on stderr."
"""

from __future__ import annotations

import json
import threading
import time

import pytest

from pipeline_common import (
    PhaseRegistry,
    PhaseTimeoutError,
    _start_watchdog,
    load_phase_hard_caps,
)

# Re-use the fake S3 helper from the existing phase-registry tests
from tests.test_phase_registry import _FakeS3


def _registry(
    bucket: str = "test-bucket",
    hard_caps: dict[str, float] | None = None,
    s3: _FakeS3 | None = None,
) -> tuple[PhaseRegistry, _FakeS3]:
    s3 = s3 or _FakeS3()
    return PhaseRegistry(
        date="2026-04-24",
        bucket=bucket,
        hard_caps=hard_caps,
        s3_client=s3,
    ), s3


# ── _start_watchdog direct tests ─────────────────────────────────────────────


def test_start_watchdog_fires_after_cap_via_injected_handler():
    """Watchdog Timer fires the injected handler after cap_s. We inject
    a no-op handler so the real one (_thread.interrupt_main) doesn't kill
    the test runner."""
    fired = threading.Event()
    captured: dict = {}

    def _noop_trip(name: str, cap_s: float) -> None:
        captured["name"] = name
        captured["cap_s"] = cap_s
        fired.set()

    timer, state = _start_watchdog("test_phase", 0.05, on_trip=_noop_trip)
    try:
        assert fired.wait(timeout=1.0), "watchdog did not fire within 1s"
        assert state["tripped"] is True
        assert captured == {"name": "test_phase", "cap_s": 0.05}
    finally:
        timer.cancel()


def test_start_watchdog_cancel_prevents_fire():
    """Cancelling the timer before cap_s elapses keeps `tripped` False."""
    fired = threading.Event()

    def _noop_trip(name: str, cap_s: float) -> None:
        fired.set()

    timer, state = _start_watchdog("test_phase", 0.5, on_trip=_noop_trip)
    timer.cancel()
    # Wait well past the cap to confirm it truly didn't fire
    assert not fired.wait(timeout=0.8)
    assert state["tripped"] is False


# ── PhaseRegistry.phase() integration ────────────────────────────────────────


def test_phase_registry_watchdog_trip_raises_phase_timeout(monkeypatch):
    """When the watchdog trips (state['tripped']=True) and a
    KeyboardInterrupt propagates, PhaseRegistry.phase() converts it to
    PhaseTimeoutError. We monkey-patch _start_watchdog so no real timer
    fires, then manually raise KeyboardInterrupt to simulate the trip
    path deterministically."""
    # Force the "tripped" state from the moment the watchdog is started
    fake_state = {"tripped": True, "name": "test_phase", "cap_s": 0.1}
    fake_timer = threading.Timer(9999, lambda: None)
    fake_timer.start()

    def _fake_start_watchdog(name, cap_s, on_trip=None):
        return fake_timer, fake_state

    monkeypatch.setattr(
        "pipeline_common._start_watchdog", _fake_start_watchdog,
    )

    registry, _s3 = _registry(hard_caps={"test_phase": 0.1})

    with pytest.raises(PhaseTimeoutError) as excinfo:
        with registry.phase("test_phase"):
            # Simulate what _thread.interrupt_main() would cause
            raise KeyboardInterrupt("simulated watchdog trip")

    msg = str(excinfo.value)
    assert "test_phase" in msg
    assert "hard cap" in msg
    assert "0.1" in msg


def test_phase_registry_no_cap_no_watchdog_started(monkeypatch):
    """A phase without a hard_cap entry must not start a watchdog."""
    started: list[tuple[str, float]] = []

    def _spy_start_watchdog(name, cap_s, on_trip=None):
        started.append((name, cap_s))
        return threading.Timer(9999, lambda: None), {"tripped": False}

    monkeypatch.setattr(
        "pipeline_common._start_watchdog", _spy_start_watchdog,
    )

    registry, _s3 = _registry(hard_caps={"other_phase": 60.0})

    with registry.phase("uncapped_phase"):
        pass

    assert started == [], f"watchdog should not have started: {started}"


def test_phase_registry_keyboard_interrupt_without_trip_is_not_phase_timeout(monkeypatch):
    """If the user Ctrl+Cs mid-phase, we must NOT lie and call it a
    phase timeout — only the watchdog-tripped state should convert
    KeyboardInterrupt to PhaseTimeoutError."""
    fake_state = {"tripped": False, "name": "test_phase", "cap_s": 60.0}
    fake_timer = threading.Timer(9999, lambda: None)
    fake_timer.start()

    def _fake_start_watchdog(name, cap_s, on_trip=None):
        return fake_timer, fake_state

    monkeypatch.setattr(
        "pipeline_common._start_watchdog", _fake_start_watchdog,
    )

    registry, _s3 = _registry(hard_caps={"test_phase": 60.0})

    with pytest.raises(KeyboardInterrupt):
        with registry.phase("test_phase"):
            raise KeyboardInterrupt("operator cancel")


def test_phase_registry_normal_completion_cancels_watchdog(monkeypatch):
    """A phase that exits normally must cancel the watchdog so it
    doesn't fire after the phase ends."""
    cancelled: list[bool] = []

    class _SpyTimer:
        def __init__(self):
            self._cancelled = False

        def cancel(self):
            self._cancelled = True
            cancelled.append(True)

    spy_timer = _SpyTimer()

    def _fake_start_watchdog(name, cap_s, on_trip=None):
        return spy_timer, {"tripped": False}

    monkeypatch.setattr(
        "pipeline_common._start_watchdog", _fake_start_watchdog,
    )

    registry, _s3 = _registry(hard_caps={"test_phase": 60.0})

    with registry.phase("test_phase"):
        pass

    assert cancelled == [True], "watchdog timer should have been cancelled"


# ── load_phase_hard_caps ─────────────────────────────────────────────────────


def test_load_phase_hard_caps_parses_block(tmp_path):
    yaml_body = (
        "full_run_hard_caps_seconds:\n"
        "  phase4a_ensemble_modes: 2700\n"
        "  phase4b_signal_thresholds: 3600.5\n"
    )
    f = tmp_path / "timing_budget.yaml"
    f.write_text(yaml_body)
    caps = load_phase_hard_caps(f)
    assert caps == {
        "phase4a_ensemble_modes": 2700.0,
        "phase4b_signal_thresholds": 3600.5,
    }


def test_load_phase_hard_caps_missing_file_returns_empty(tmp_path):
    caps = load_phase_hard_caps(tmp_path / "nonexistent.yaml")
    assert caps == {}


def test_load_phase_hard_caps_missing_block_returns_empty(tmp_path):
    f = tmp_path / "timing_budget.yaml"
    f.write_text("smoke_budgets_seconds:\n  smoke-simulate: 400\n")
    caps = load_phase_hard_caps(f)
    assert caps == {}


def test_load_phase_hard_caps_skips_non_numeric_entries(tmp_path):
    yaml_body = (
        "full_run_hard_caps_seconds:\n"
        "  phase_good: 100\n"
        "  phase_bad: not-a-number\n"
        "  phase_ok: 200\n"
    )
    f = tmp_path / "timing_budget.yaml"
    f.write_text(yaml_body)
    caps = load_phase_hard_caps(f)
    assert caps == {"phase_good": 100.0, "phase_ok": 200.0}
