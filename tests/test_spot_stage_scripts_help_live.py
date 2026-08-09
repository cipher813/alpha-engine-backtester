"""Live-executable (but zero-spend) coverage for the alpha-engine-config-I4442
per-stage spot scripts: each script's `--help` path actually runs the real
bash file end to end and must exit 0 having made no AWS/network call and no
S3/config write, before any spot is provisioned.

Unlike the SSM/EC2 workload itself (which cannot run in CI — see the
static-analysis tests in test_spot_stage_scripts_structure.py), `--help` is
handled by spot_common_parse_flags() before spot_common_normalize_run_date()
or spot_common_launch_instance() run, so this is safe to execute directly:
no AWS credentials, no network egress, sub-second. Not marked `live` (that
marker is reserved for tests needing real AWS/ArcticDB per conftest.py) —
this is exactly the CI-safe "does it even run" smoke the split was
supposed to make possible per stage.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

STAGE_SCRIPTS = [
    "spot_backtester.sh",
    "spot_predictor_backtest.sh",
    "spot_portfolio_optimizer_backtest.sh",
    "spot_parity.sh",
    "spot_evaluator.sh",
]

# Tokens that would indicate the script fell through into real work instead
# of stopping at --help.
FORBIDDEN_OUTPUT_TOKENS = [
    "Requesting spot instance",
    "Waiting for instance",
    "Staging configs",
    "Bootstrapping spot",
    "Installing Python dependencies",
]

if sys.platform == "win32":
    pytest.skip("bash-only scripts", allow_module_level=True)


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_help_exits_zero_with_zero_spend(name):
    result = subprocess.run(
        ["bash", str(_INFRA / name), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f"{name} --help exited {result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    combined = result.stdout + result.stderr
    for token in FORBIDDEN_OUTPUT_TOKENS:
        assert token not in combined, (
            f"{name} --help reached {token!r} — it must stop before any spend"
        )
    assert name in result.stdout, f"{name} --help did not print its own usage banner"


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_unknown_flag_hard_fails(name):
    """no-silent-fails: an unrecognized flag (including the retired
    stage-multiplexing flags --mode / --skip-stages) must exit non-zero
    with a clear message, never silently ignore it."""
    result = subprocess.run(
        ["bash", str(_INFRA / name), "--mode=all"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "Unknown flag" in (result.stdout + result.stderr)


def test_common_sh_direct_execution_fails_loud():
    result = subprocess.run(
        ["bash", str(_INFRA / "_spot_common.sh")],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "must be sourced" in (result.stdout + result.stderr)


def test_evaluator_rejects_unknown_eval_half():
    result = subprocess.run(
        ["bash", str(_INFRA / "spot_evaluator.sh"), "--eval-half=bogus", "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode != 0
    assert "unknown --eval-half" in (result.stdout + result.stderr)
