"""The EXIT trap must reach ``terminate-instances`` on a NON-reclaim failure.

THE DEFECT

``cleanup()`` asks ``krepis.ec2_spot relaunch-decision`` whether the spot was
reclaimed by AWS. That CLI answers by EXIT CODE — ``0`` = relaunch,
``NO_RELAUNCH_EXIT_CODE`` (75) = hold. Hold is the designed answer for every
failure that is *not* a reclaim, which is nearly all of them: a crash, an OOM,
a workload timeout, a bad config.

Both launchers run under ``set -e``, and the call was written::

    _decide_out="$(... relaunch-decision ... )"
    _decide_rc=$?

An assignment whose value comes from a command substitution is a simple command
whose exit status IS the substitution's. So on the ordinary hold answer, errexit
fired and destroyed the shell *inside the EXIT trap*: ``_decide_rc=$?`` was
unreachable, and so were ``terminate-instances``, the S3 staging teardown, and
the closing ``exit "$exit_code"`` that re-raises the workload's real status.
Because ``set -e`` does not re-enter a trap it is already running, the abort
emitted nothing whatsoever — the log simply stops mid-cleanup.

MEASURED CONSEQUENCE

Confirmed in the sibling repo ``crucible-predictor`` on
``ne-weekly-freshness-pipeline`` execution ``watch-rerun-2026-08-10-9``
(2026-08-11): cleanup's stdout ends after its own diagnostic,
``==> Terminating spot instance ...`` never appears, and CloudTrail records NO
``TerminateInstances`` call for ``i-092854cbb6e62b753`` in the window while
three unrelated instances were terminated normally. The r5.large ran on until
its own systemd watchdog stopped it, and the launcher exited 75 instead of the
workload's status. Fixed there in ``crucible-predictor-PR467``; this file is the
same defect in this repo's two copies.

WHY THIS REPO IS THE WIDEST BLAST RADIUS OF THE THREE

``_spot_common.sh`` is sourced by ten per-stage launchers, so a single unguarded
substitution leaked an instance for every non-reclaim failure of every one of
them. ``spot_backtest.sh`` carries its own copy of the same block.

METHOD

The real ``cleanup`` is lifted out of the real script (brace-matched, so the
text executed is the text in the repository), installed as the EXIT trap, and
the harness then fails the way a failed SSM step does. ``aws`` and the lib CLI
are stubbed; the stub CLI prints the CLI's real TAB-separated line and exits 75.
Reverting the guard in either launcher fails this test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

#: The CLI's documented "do not relaunch" status. Any non-zero value reproduces
#: the defect; 75 is what the real path returns.
_NO_RELAUNCH_EXIT_CODE = 75

#: (script, function to lift, instance-id var, staging var). ``_spot_common.sh``
#: defines ``cleanup`` *inside* ``spot_common_install_cleanup_trap``, so that
#: outer function is what gets lifted — calling it defines cleanup and installs
#: the trap exactly as production does. ``spot_backtest.sh`` has cleanup at top
#: level and the harness installs the trap itself.
_LAUNCHERS = (
    ("_spot_common.sh", "spot_common_install_cleanup_trap", True),
    ("spot_backtest.sh", "cleanup", False),
)


def _function_text(source: str, name: str) -> str:
    """Return a shell function's full text, brace-matched."""
    marker = "\n%s() {" % name
    assert marker in source, f"{name}() not found"
    start = source.index(marker) + 1
    depth = 0
    for idx in range(start, len(source)):
        if source[idx] == "{":
            depth += 1
        elif source[idx] == "}":
            depth -= 1
            if depth == 0:
                return source[start : idx + 1]
    raise AssertionError(f"unbalanced braces in {name}()")


def _write_stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


@pytest.fixture(autouse=True)
def _requires_bash():
    if shutil.which("bash") is None:  # pragma: no cover - bash is a hard dep
        pytest.skip("bash unavailable")


@pytest.mark.parametrize(
    ("script_name", "function_name", "installs_own_trap"),
    _LAUNCHERS,
    ids=[s for s, _, _ in _LAUNCHERS],
)
def test_cleanup_terminates_the_instance_when_the_decision_is_hold(
    script_name: str,
    function_name: str,
    installs_own_trap: bool,
    tmp_path: Path,
) -> None:
    script = _INFRA / script_name
    source = script.read_text()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # `aws` records every invocation so the assertion is on what cleanup
    # actually CALLED, not on what it printed.
    calls = tmp_path / "aws-calls.log"
    _write_stub(
        bin_dir / "aws",
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" >> " + str(calls) + "\nexit 0\n",
    )
    # Stand-in for $LIB_PYTHON: answers "hold" exactly as the real CLI does —
    # the TAB-separated line on stdout, then the no-relaunch exit status.
    hold_python = tmp_path / "hold-python"
    _write_stub(
        hold_python,
        "#!/usr/bin/env bash\n"
        "printf 'hold\\tnot-reclaim:other\\tother\\t1\\n'\n"
        f"exit {_NO_RELAUNCH_EXIT_CODE}\n",
    )

    lifted = _function_text(source, function_name)
    trap_install = "" if installs_own_trap else "trap cleanup EXIT\n"
    invoke = f"{function_name}\n" if installs_own_trap else ""

    harness = tmp_path / "harness.sh"
    harness.write_text(
        "#!/usr/bin/env bash\n"
        # The condition under test — both launchers set exactly this.
        "set -euo pipefail\n"
        f"{lifted}\n"
        "AWS_REGION=us-east-1\n"
        "INSTANCE_ID=i-0000000000test0000\n"
        "S3_STAGING=s3://test-bucket/tmp/spot/test\n"
        "REPO_ROOT=" + str(tmp_path) + "\n"
        "SPOT_STAGE_NAME=test-stage\n"
        "LAST_SSM_DESC='test step'\n"
        "MAX_RUNTIME_SECONDS=5400\n"
        "SF_EXECUTION_TIMEOUT=''\n"
        "SPOT_ATTEMPT=1\n"
        "MAX_SPOT_ATTEMPTS=2\n"
        f"LIB_PYTHON={hold_python}\n"
        "_ORIG_ARGS=()\n"
        f"{invoke}"
        f"{trap_install}"
        # Fail the way a failed SSM step does.
        "exit 3\n"
    )
    harness.chmod(0o755)

    proc = subprocess.run(
        ["bash", str(harness)],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(tmp_path)},
        timeout=60,
    )
    aws_calls = calls.read_text() if calls.exists() else ""

    assert "terminate-instances" in aws_calls, (
        f"{script_name}: cleanup never reached terminate-instances on a HELD "
        "relaunch decision — the spot instance is leaked and runs until its own "
        "watchdog stops it.\n"
        f"aws calls seen:\n{aws_calls or '  (none)'}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    assert proc.returncode == 3, (
        f"{script_name}: the launcher exited {proc.returncode}, not the "
        "workload's status 3. Exiting with the decision CLI's 75 misreports a "
        "training/backtest failure as a spot-relaunch verdict to the "
        "orchestration wrapper."
    )
