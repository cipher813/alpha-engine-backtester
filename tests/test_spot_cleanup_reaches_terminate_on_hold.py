"""The EXIT trap must reach ``terminate-instances`` on a NON-reclaim failure.

THE DEFECT (historical — fixed by crucible-backtester-PR636)

``cleanup()`` used to ask ``krepis.ec2_spot relaunch-decision`` whether the
spot was reclaimed by AWS, reading the verdict off the CLI's EXIT CODE —
``0`` = relaunch, ``NO_RELAUNCH_EXIT_CODE`` (75) = hold. Both launchers run
under ``set -e``, and the call was written::

    _decide_out="$(... relaunch-decision ... )"
    _decide_rc=$?

An assignment whose value comes from a command substitution is a simple
command whose exit status IS the substitution's, so on the ordinary hold
answer errexit fired and destroyed the shell *inside the EXIT trap* before
``_decide_rc=$?`` could run — skipping ``terminate-instances`` entirely.
PR636 fixed that with ``|| _decide_rc=$?``.

THE CURRENT DEFECT (alpha-engine-config-I7009)

``krepis.ec2_spot relaunch-decision`` grew ``--json`` in krepis-PR133
(released 0.51.0). With ``--json`` the verdict is a field on stdout and the
CLI exits 0 whenever it reached ANY decision, hold included. A non-zero exit
with ``--json`` means only "the CLI could not answer" — not a verdict. The
PR636 guard was written for the OLD exit-code contract (0/75); this test
pins the NEW one: both a JSON-encoded hold (rc=0) and a genuine CLI failure
(rc!=0, no JSON to parse) MUST still reach ``terminate-instances`` and MUST
NOT relaunch.

METHOD

The real ``cleanup`` is lifted out of the real script (brace-matched, so the
text executed is the text in the repository), installed as the EXIT trap,
and the harness then fails the way a failed SSM step does. ``aws`` is
stubbed; ``$LIB_PYTHON`` is stubbed to intercept only the
``-m krepis.ec2_spot relaunch-decision`` invocation (answering per scenario)
and to fall through to the real interpreter for every other invocation —
notably the ``$LIB_PYTHON -c '...json.load...'`` call the launchers now use
to read the verdict, which needs a real Python. Reverting either launcher to
read the exit code as the verdict fails this test.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

#: (script, function to lift, installs its own trap).
#: ``_spot_common.sh`` defines ``cleanup`` *inside*
#: ``spot_common_install_cleanup_trap``, so that outer function is what gets
#: lifted — calling it defines cleanup and installs the trap exactly as
#: production does. ``spot_backtest.sh`` has cleanup at top level and the
#: harness installs the trap itself.
_LAUNCHERS = (
    ("_spot_common.sh", "spot_common_install_cleanup_trap", True),
    ("spot_backtest.sh", "cleanup", False),
)

#: (scenario id, stub body for the intercepted `-m krepis.ec2_spot ...` call).
#: Both scenarios are a HOLD and must not relaunch.
_HOLD_SCENARIOS = (
    (
        "json-hold",
        # --json: a reachable, non-relaunch verdict — exits 0.
        "printf '{\"relaunch\": false, \"reason\": \"not-reclaim:other\"}\\n'\nexit 0\n",
    ),
    (
        "cli-failure",
        # The CLI could not answer at all — non-zero exit, nothing parseable
        # on stdout. Per I7009 this is NOT a verdict and must still hold.
        "exit 9\n",
    ),
)


def _function_text(source: str, name: str) -> str:
    """Return a shell function's full text, brace-matched."""
    marker = "\n" + name + "() {"
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
    ("scenario_id", "decide_stub_body"),
    _HOLD_SCENARIOS,
    ids=[s for s, _ in _HOLD_SCENARIOS],
)
@pytest.mark.parametrize(
    ("script_name", "function_name", "installs_own_trap"),
    _LAUNCHERS,
    ids=[s for s, _, _ in _LAUNCHERS],
)
def test_cleanup_terminates_the_instance_on_hold(
    script_name: str,
    function_name: str,
    installs_own_trap: bool,
    scenario_id: str,
    decide_stub_body: str,
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
    # Stand-in for $LIB_PYTHON. Intercepts ONLY the
    # `-m krepis.ec2_spot relaunch-decision` invocation with the scenario's
    # canned answer; every other invocation (notably the launcher's own
    # `$LIB_PYTHON -c '...json...'` verdict parse) falls through to the real
    # interpreter, because that parse must actually work for this test to
    # exercise the real code path.
    decide_python = tmp_path / "decide-python"
    _write_stub(
        decide_python,
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "-m" ] && [ "$2" = "krepis.ec2_spot" ]; then\n'
        f"{decide_stub_body}"
        "fi\n"
        f'exec {sys.executable} "$@"\n',
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
        f"LIB_PYTHON={decide_python}\n"
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
        f"{script_name} [{scenario_id}]: cleanup never reached "
        "terminate-instances on a HELD relaunch decision — the spot instance "
        "is leaked and runs until its own watchdog stops it.\n"
        f"aws calls seen:\n{aws_calls or '  (none)'}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )

    assert "RECLAIMED by AWS" not in proc.stdout, (
        f"{script_name} [{scenario_id}]: cleanup relaunched on a HOLD verdict "
        f"instead of holding.\nstdout:\n{proc.stdout}"
    )

    assert proc.returncode == 3, (
        f"{script_name} [{scenario_id}]: the launcher exited {proc.returncode}, "
        "not the workload's status 3. Misreading the decision CLI's exit "
        "status as the verdict misreports a training/backtest failure as a "
        "spot-relaunch verdict to the orchestration wrapper."
    )
