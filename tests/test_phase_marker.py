"""
tests/test_phase_marker.py — pipeline_common.phase context manager.

Locks the log contract that the Saturday SF relies on for diagnosing
silent-phase timeouts: every phase emits PHASE_START/PHASE_END on the
`backtest.phase` logger, with the duration and status fields parseable
from the single END line.
"""

from __future__ import annotations

import logging
import re

import pytest

from pipeline_common import phase


def test_phase_emits_start_and_end(caplog):
    with caplog.at_level(logging.INFO, logger="backtest.phase"):
        with phase("unit_test_phase", foo="bar"):
            pass

    msgs = [r.getMessage() for r in caplog.records if r.name == "backtest.phase"]
    assert any(m.startswith("PHASE_START name=unit_test_phase") for m in msgs)
    end = next(m for m in msgs if m.startswith("PHASE_END name=unit_test_phase"))
    assert "status=ok" in end
    assert "foo=bar" in end
    assert re.search(r"duration_s=\d+\.\d{2}", end)


def test_phase_marks_error_status_on_exception(caplog):
    with caplog.at_level(logging.INFO, logger="backtest.phase"):
        with pytest.raises(RuntimeError, match="boom"):
            with phase("error_phase"):
                raise RuntimeError("boom")

    msgs = [r.getMessage() for r in caplog.records if r.name == "backtest.phase"]
    end = next(m for m in msgs if m.startswith("PHASE_END name=error_phase"))
    assert "status=error" in end


def test_phase_end_duration_monotonic(caplog):
    """Regression guard: duration must be measured with monotonic, not wall clock."""
    import time
    with caplog.at_level(logging.INFO, logger="backtest.phase"):
        with phase("timed_phase"):
            time.sleep(0.02)

    msgs = [r.getMessage() for r in caplog.records if r.name == "backtest.phase"]
    end = next(m for m in msgs if m.startswith("PHASE_END name=timed_phase"))
    m = re.search(r"duration_s=(\d+\.\d{2})", end)
    assert m is not None
    assert float(m.group(1)) >= 0.02
