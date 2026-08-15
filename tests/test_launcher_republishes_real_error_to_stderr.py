"""A failing stage's real cause must reach STDERR, not only STDOUT.

**The 2026-08-15 weekly-SF failure (config-I7396 / config-I7399).**
``PredictorBacktest`` died and every surface an operator reads said the same
thing::

    ERROR: backtest.py failed (rc=1). Spot run marked FAILED.

The actual cause — a ``ValueError`` with a full traceback — existed the whole
time, in two places neither the Step Function nor the alert consults:
``backtest/{run_date}/predictor_stats.json`` in S3, and the tail of the
stage's own stdout.

Why the stdout copy is structurally unreachable, and why bounding the log
chatter does not fix it: the launcher runs the stage as ``python … 2>&1``, so
the traceback lands on **stdout**, and SSM's ``GetCommandInvocation`` returns
only the **FIRST** 24 KB of stdout. Truncation is from the front, so anything
at the end of a multi-thousand-line run is gone no matter how quiet the run
is made. ``StandardErrorContent`` comes back intact, and it is what the Step
Function copies verbatim into its failure cause.

So each launcher tees the stage output to a file and, on non-zero rc, tails
it to STDERR. This test pins that shape in every stage launcher rather than
the one that happened to fail — the same block is copy-pasted across four of
them, which is precisely how a fix to one silently fails to reach the others.
"""

from __future__ import annotations

import pathlib

import pytest

_INFRA = pathlib.Path(__file__).resolve().parent.parent / "infrastructure"

# The per-stage launchers that run a long Python stage on a spot box. Each
# maps 1:1 to a weekly-SF state whose failure reaches the operator only
# through SSM's StandardErrorContent.
_STAGE_LAUNCHERS = [
    "spot_predictor_backtest.sh",
    "spot_backtester.sh",
    "spot_portfolio_optimizer_backtest.sh",
    "spot_evaluator.sh",
]


def _text(name: str) -> str:
    p = _INFRA / name
    assert p.is_file(), f"{p} missing — the launcher was renamed or removed"
    return p.read_text()


def test_the_guarded_set_is_non_empty():
    """A list that silently emptied would make every assertion below vacuous."""
    assert _STAGE_LAUNCHERS
    for name in _STAGE_LAUNCHERS:
        assert (_INFRA / name).is_file(), name


@pytest.mark.parametrize("name", _STAGE_LAUNCHERS)
def test_stage_output_is_captured_to_a_file(name):
    text = _text(name)
    assert "_STAGE_TAIL_LOG=" in text, (
        f"{name}: no stage-output capture. Without it the failure arm has "
        "nothing to republish, and the real cause exists only in the first "
        "24 KB of stdout that SSM returns."
    )
    assert 'tee "\\$_STAGE_TAIL_LOG"' in text, (
        f"{name}: stage output is not tee'd to the capture file. Piping "
        "through tee (rather than redirecting) keeps the live progress "
        "stream intact for a stage that runs for tens of minutes."
    )


@pytest.mark.parametrize("name", _STAGE_LAUNCHERS)
def test_failure_arm_tails_the_capture_to_stderr(name):
    text = _text(name)
    assert 'tail -n 60 "\\$_STAGE_TAIL_LOG"' in text, (
        f"{name}: the failure arm does not tail the captured output."
    )
    tail_line = next(
        line for line in text.split("\n")
        if 'tail -n 60 "\\$_STAGE_TAIL_LOG"' in line
    )
    assert ">&2" in tail_line, (
        f"{name}: the tail goes to stdout, which is exactly where the cause "
        "was already unreachable. It must go to STDERR."
    )


@pytest.mark.parametrize("name", _STAGE_LAUNCHERS)
def test_the_tail_precedes_the_exit(name):
    """Emitted before the stage exits, or it never runs at all."""
    text = _text(name)
    tail_at = text.index('tail -n 60 "\\$_STAGE_TAIL_LOG"')
    assert 'exit "\\$_' in text[tail_at:], (
        f"{name}: no stage exit follows the tail — either the tail moved out "
        "of the failure arm or the arm stopped exiting with the real rc"
    )


@pytest.mark.parametrize("name", _STAGE_LAUNCHERS)
def test_the_resource_kill_classification_survives(name):
    """The tail is additive. config-I7258's rc classification — the thing that
    makes an OOM say OOM instead of "failed" — must still be there, and must
    still come FIRST so it is the line the operator reads at the top."""
    text = _text(name)
    assert "SIGKILLed" in text and "timed out" in text, (
        f"{name}: config-I7258's OOM/timeout classification was lost"
    )
    assert text.index("SIGKILLed") < text.index(
        'tail -n 60 "\\$_STAGE_TAIL_LOG"'
    ), f"{name}: the raw tail now precedes the named resource-kill verdict"
