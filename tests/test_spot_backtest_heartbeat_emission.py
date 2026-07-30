"""config-I5786 regression: every heartbeat's gate must be satisfiable by a real caller.

**What happened.** `spot_backtest.sh` gated the `backtester` heartbeat on
``! skip(backtest) && ! skip(parity)`` — correct while backtest and parity were
one SF state. On 2026-05-16 (nousergon-data#250, the preflight task split) they
became two, and every caller since passes a ``--skip-stages`` set excluding one
or the other. The conjunction became **unsatisfiable by every caller**.

The last ``Process=backtester`` datapoint in CloudWatch is 2026-05-13 — the final
Saturday run before that merge. `alpha-engine-backtester-no-heartbeat` went to
ALARM on 2026-06-03 (its 8-day window) and stayed there for 57 days, correctly
reporting a signal no code path could produce, while its siblings
(`predictor-training`, `evaluator`) published normally throughout.

**The class, and why a static assertion on the gate is not enough.** The gate was
never wrong in isolation — it was invalidated by a refactor somewhere else, and
its failure mode was silence. So these tests do not assert what the condition
says; they **execute the real gating block** against the exact
``--skip-stages`` values the Step Function passes, and assert on what comes out.
A future split that makes some other heartbeat unemittable fails here.

The caller list below is the coupling point. It is duplicated from
``nousergon-data/infrastructure/step_function.json`` deliberately: that file is
in another repo, this repo's CI cannot see it, and a cross-repo read that
silently skips when the checkout is absent is the *other* half of this same
failure (nous-ergon-ops-PR293). A wrong-but-present list fails loudly; an absent
one passes quietly.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"

# Every --skip-stages value the Saturday SF passes to this script, by state.
# Source: nousergon-data/infrastructure/step_function.json (verified 2026-07-30).
SF_CALLERS = {
    "Backtester": "parity,evaluator",
    "Parity": "backtest,evaluator",
    "Evaluator": "backtest,parity",
    "PredictorBacktest": "parity,evaluator",
    "PortfolioOptimizerBacktest": "parity,evaluator",
}

# Heartbeats this script is responsible for emitting, and the state that must
# emit each. A heartbeat with no state here is one nothing can produce.
HEARTBEAT_OWNERS = {
    "backtester": "Backtester",
    "evaluator": "Evaluator",
}


def _gating_block() -> str:
    """The real `_stage_in_skip` definition plus the emission conditionals."""
    s = SCRIPT.read_text()
    start = s.index("_stage_in_skip() {")
    end = s.index("_emit_heartbeat evaluator", start)
    end = s.index("fi", end) + len("fi")
    return s[start:end]


def _emitted(skip_stages: str) -> set[str]:
    """Run the script's own gating logic and return the heartbeats it emits."""
    harness = (
        "set -euo pipefail\n"
        f"SKIP_STAGES='{skip_stages}'\n"
        "_emit_heartbeat() { echo \"$1\"; }\n"
        f"{_gating_block()}\n"
    )
    proc = subprocess.run(["bash", "-c", harness], capture_output=True, text=True)
    assert proc.returncode == 0, f"gating block failed: {proc.stderr}"
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


@pytest.mark.parametrize("heartbeat,state", sorted(HEARTBEAT_OWNERS.items()))
def test_every_heartbeat_is_emitted_by_its_owning_state(heartbeat: str, state: str) -> None:
    """The regression itself: a gate no caller can satisfy emits nothing, forever."""
    emitted = _emitted(SF_CALLERS[state])
    assert heartbeat in emitted, (
        f"the {state} SF state passes --skip-stages={SF_CALLERS[state]!r} and does "
        f"NOT emit the {heartbeat!r} heartbeat. Its alarm "
        f"(alpha-engine-{heartbeat}-no-heartbeat) will sit in ALARM forever, "
        f"reporting a signal no code path can produce — which is exactly what "
        f"happened between 2026-05-13 and 2026-07-30 (config-I5786). Emitted: "
        f"{sorted(emitted)}"
    )


def test_no_heartbeat_gate_is_unsatisfiable_by_every_caller() -> None:
    """The general form, so a future split is caught even for a new heartbeat.

    Any heartbeat name the script can emit must be emitted by at least one real
    caller. A name that appears in the script and in no caller's output is a
    metric that exists only in source.
    """
    names = set(re.findall(r"_emit_heartbeat\s+([a-z][a-z0-9-]*)", _gating_block()))
    assert names, "no _emit_heartbeat calls found — the extraction is wrong, not the script"

    reachable: set[str] = set()
    for skip in SF_CALLERS.values():
        reachable |= _emitted(skip)

    unreachable = sorted(names - reachable)
    assert not unreachable, (
        f"heartbeats no SF caller can emit: {unreachable}. Their alarms cannot "
        f"ever clear, and the silence is indistinguishable from a dead workload. "
        f"Either fix the gate or delete the emission — and if a new SF state "
        f"should emit it, add that state to SF_CALLERS."
    )


def test_a_stage_that_is_skipped_does_not_emit_its_heartbeat() -> None:
    """The gate still gates — the fix must not make emission unconditional.

    A control that cannot produce a negative result is not informative, so this
    asserts the other direction: skipping the stage suppresses the heartbeat.
    """
    assert "backtester" not in _emitted("backtest,parity,evaluator")
    assert "evaluator" not in _emitted("backtest,parity,evaluator")
