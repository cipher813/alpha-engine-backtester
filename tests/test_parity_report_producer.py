"""The parity_report.json producer and its absence detector (config#7199).

`backtest/{run_date}/parity_report.json` is the backtester-to-executor replay
parity artifact — the evidence that the simulation's fills resemble what the
live executor would actually do. It had no real write between **2026-07-18**
(for run_date 2026-07-17) and 2026-08-13.

Two independent defects, and the second outranks the first:

1. **The producer stopped existing.** `pytest` was never declared by this repo;
   it arrived transitively through the co-installed predictor requirements until
   PR #550 (2026-07-20, config#3031) correctly stopped co-installing them. From
   the 2026-07-24 run onward the parity stage printed `No module named pytest`
   and exited in one second.
2. **Nothing could tell.** The upload was guarded by `if [ -f … ]` with **no
   `else`**, so a missing artifact was indistinguishable from a skipped upload,
   the non-zero exit degraded to a WARNING on stderr, and the SF took its
   success path. The sibling `report.md` in the same S3 prefix kept updating
   weekly, so every freshness check on that prefix stayed green.

These are structural tests — they assert the properties on the shipped files,
because both defects are invisible in any unit test of the Python they guard.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]

#: Every launcher that invokes the parity replay and uploads its artifact.
_LAUNCHERS = (
    "infrastructure/spot_parity_replay.sh",
    "infrastructure/spot_parity.sh",
    "infrastructure/spot_backtest.sh",
)


def test_pytest_is_a_declared_dependency():
    """The artifact's producer is `python -m pytest`, so pytest is a RUNTIME
    dependency of this repo. Relying on it arriving through another repo's
    requirements is what broke — and the break was invisible because the
    interpreter, not the test, is what failed."""
    req = (_REPO / "requirements.txt").read_text()
    assert re.search(r"^pytest[><=~]", req, re.MULTILINE), (
        "requirements.txt declares no pytest — the parity_report.json producer "
        "cannot run on the spot instance"
    )


_GUARD = 'if [ -f "\\$PARITY_REPORT_DIR/parity_report.json" ]; then'


def _guard_block(script: str) -> str:
    """The text of the upload guard, from `if` to its closing `fi`.

    Indentation varies — `spot_backtest.sh` nests the block two levels deeper
    inside its heredoc — so the closing `fi` is matched at any indent.
    """
    body = (_REPO / script).read_text()
    assert _GUARD in body, f"{script}: the upload guard moved — re-point this test"
    tail = body[body.index(_GUARD):]
    end = re.search(r"^\s*fi\s*$", tail, re.MULTILINE)
    assert end, f"{script}: the upload guard has no closing fi"
    return tail[: end.end()]


@pytest.mark.parametrize("script", _LAUNCHERS)
def test_a_missing_parity_report_is_detected(script):
    """The `if [ -f … ]` upload guard must have an `else`. Without one, an
    absent artifact produces no signal of any kind."""
    block = _guard_block(script)
    assert re.search(r"^\s*else\s*$", block, re.MULTILINE), (
        f"{script}: the parity_report.json upload guard has no `else` — a "
        f"missing artifact is silent, which is the 4-week outage this test exists "
        f"to prevent"
    )
    assert "parity_report_missing.json" in block, (
        f"{script}: the absence is not RECORDED. An absent artifact must become "
        f"a present artifact that says it is absent, so downstream readers "
        f"render FAILED rather than a stale ABSENT row (sf-pipeline-policy §2.3a)."
    )


def test_the_absence_marker_is_valid_json_carrying_a_fail_verdict():
    """The marker is emitted by `printf` from a shell heredoc — a format string
    nothing else validates. Extract it and parse it."""
    body = (_REPO / "infrastructure/spot_parity_replay.sh").read_text()
    m = re.search(r"printf '(\{\"schema\":\"parity_report-0\.0\.0\".*?\})", body)
    assert m, "the absence-marker printf format is not where this test expects it"
    template = m.group(1)
    doc = json.loads(template % ("2026-08-15", 1, 1))
    assert doc["status"] == "failed"
    assert doc["verdict"] == "FAIL"
    assert doc["run_date"] == "2026-08-15"


def test_only_the_split_branch_fails_its_stage_on_the_absence():
    """Blast-radius asymmetry, deliberate and worth pinning.

    `spot_parity_replay.sh` IS one SF branch whose every failure path converges
    on `ParityReplayDegraded`, siblings untouched — so exiting non-zero costs
    that branch alone and the artifact is its whole product.

    The two bundled launchers still run co-tenant stages behind one script, so a
    non-zero exit there would kill stages that do not consume the parity
    artifact — trading one blindness for an outage (sf-pipeline-policy §2.1).
    They record the absence and continue.
    """
    def _else_block(script):
        block = _guard_block(script)
        return block[block.index("else"):]

    assert re.search(r"^\s*exit 1\s*$", _else_block(_LAUNCHERS[0]), re.MULTILINE)
    for bundled in _LAUNCHERS[1:]:
        assert not re.search(r"^\s*exit 1\s*$", _else_block(bundled), re.MULTILINE), (
            f"{bundled}: a bundled launcher must not exit non-zero here — it "
            f"would take co-tenant stages down with it"
        )
