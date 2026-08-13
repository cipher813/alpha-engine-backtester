"""A failed predictor pipeline records WHERE it failed, not just what it said.

2026-08-13, weekly execution ``watch-rerun-2026-08-13-2``.
``backtest/2026-08-13/predictor_stats.json`` was::

    {"status": "error", "error": "cannot convert float NaN to integer"}

That is the whole record. ``_run_predictor_pipeline`` caught the exception,
logged it through ``logger.error("...: %s", e)`` with no ``exc_info``, and put
``str(e)`` in the artifact — so neither the log nor the artifact named the
raising frame. The spot instance self-terminated minutes later, taking the only
copy of the stack with it, and the root cause could not be located from the
artifact that was written precisely to record it.

The cost was not cosmetic. ``predictor_stats.json`` is the walk-forward pass's
reuse source (config#6032): a ``status != ok`` artifact is correctly refused,
so the Parity stage fell back to re-running the full predictor pipeline in a
subprocess, which then exceeded its 2700s per-pass ceiling. The week's
contamination verdict came out ``UNKNOWN`` on
``{'lookahead': 'missing', 'walkforward': 'failed'}`` — downstream of this one
unlabelled exception.

These tests pin the artifact's failure contract: an error record names its
exception class and carries the traceback, bounded so it cannot bloat an object
that is uploaded whole.
"""

from __future__ import annotations

import argparse
import json

import pytest

import backtest


def _args() -> argparse.Namespace:
    return argparse.Namespace(mode="predictor-backtest")


def _raise_from_a_named_frame(_config):
    """Stand-in for run_predictor_param_sweep, raising the 2026-08-13 error
    from a frame with a findable name."""
    def _inner_frame_that_should_appear_in_the_traceback():
        raise ValueError("cannot convert float NaN to integer")

    _inner_frame_that_should_appear_in_the_traceback()


@pytest.fixture()
def failed_stats(monkeypatch) -> dict:
    monkeypatch.setattr(
        backtest, "run_predictor_param_sweep", _raise_from_a_named_frame
    )
    stats, sweep_df, _rec = backtest._run_predictor_pipeline(
        _args(), {}, None, None
    )
    assert sweep_df is None
    return stats


def test_error_record_still_carries_status_and_message(failed_stats) -> None:
    """The pre-existing contract is unchanged — consumers keying on
    ``status``/``error`` (e.g. pit_parity's reuse validator) keep working."""
    assert failed_stats["status"] == "error"
    assert failed_stats["error"] == "cannot convert float NaN to integer"


def test_error_record_names_the_exception_class(failed_stats) -> None:
    assert failed_stats["error_class"] == "ValueError"


def test_error_record_carries_the_raising_frame(failed_stats) -> None:
    """The whole point: the artifact alone must locate the defect."""
    tb = failed_stats["traceback"]
    assert "Traceback (most recent call last)" in tb
    assert "_inner_frame_that_should_appear_in_the_traceback" in tb, (
        "the traceback does not reach the raising frame — the artifact is "
        "back to naming only the message"
    )


def test_traceback_is_bounded(monkeypatch) -> None:
    """predictor_stats.json is uploaded whole; an unbounded traceback from a
    deep recursion must not be able to inflate it."""
    def _deep(_config, depth=200):
        def rec(n):
            if n == 0:
                raise ValueError("cannot convert float NaN to integer")
            rec(n - 1)
        rec(depth)

    monkeypatch.setattr(backtest, "run_predictor_param_sweep", _deep)
    stats, _sweep, _rec = backtest._run_predictor_pipeline(
        _args(), {}, None, None
    )
    assert len(stats["traceback"]) <= 4000


def test_error_record_is_json_serialisable(failed_stats) -> None:
    """_export_simulation_artifacts json.dumps() this object; a non-encodable
    field would turn a diagnosable failure into a lost one."""
    encoded = json.dumps(failed_stats, default=backtest._json_stats_default)
    assert "_inner_frame_that_should_appear_in_the_traceback" in encoded


def test_logger_receives_the_stack(monkeypatch) -> None:
    """`logger.error(..., exc_info=True)` — the log surface must not regress
    to the message-only form even if the artifact keeps the traceback."""
    seen: list[dict] = []

    def _capture(msg, *a, **kw):
        seen.append(kw)

    monkeypatch.setattr(backtest, "run_predictor_param_sweep", _raise_from_a_named_frame)
    monkeypatch.setattr(backtest.logger, "error", _capture)
    backtest._run_predictor_pipeline(_args(), {}, None, None)
    assert seen, "the failure was not logged at all"
    assert any(kw.get("exc_info") for kw in seen), (
        "predictor-pipeline failure logged without exc_info — the spot log "
        "again carries the message but not the frame"
    )
