"""Structural + static-analysis guards for the alpha-engine-config-I4442
per-stage spot script split.

`infrastructure/spot_backtest.sh` (the 1600+-line monolith) bundled the
Backtester / PredictorBacktest / PortfolioOptimizerBacktest / Parity /
Evaluator stages behind --mode / --skip-stages / --pit-parity-enabled
flags, even though the Saturday SF already runs them as five independent
SF states. This split extracts shared spot infrastructure into
`_spot_common.sh` and gives each stage its own thin, independently
runnable entry script. `spot_backtest.sh` itself is UNCHANGED and stays
the currently-wired SF path — these tests only cover the five new scripts
and the shared file, mirroring the existing test_spot_backtest_*.py
static-analysis pattern (the SSM/EC2 path cannot be exercised in CI).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"

STAGE_SCRIPTS = [
    "spot_backtester.sh",
    "spot_predictor_backtest.sh",
    "spot_portfolio_optimizer_backtest.sh",
    "spot_parity.sh",
    "spot_evaluator.sh",
]

STAGE_NAMES = {
    "spot_backtester.sh": "backtester",
    "spot_predictor_backtest.sh": "predictor-backtest",
    "spot_portfolio_optimizer_backtest.sh": "portfolio-optimizer",
    "spot_parity.sh": "parity",
    "spot_evaluator.sh": "evaluator",
}

# Every stage-selecting flag CASE-BRANCH from the monolith that must NOT
# reappear on the split scripts — each new script IS one stage, hardcoded.
# (Bare substrings like "--skip-stages" legitimately appear in prose:
# header docblocks and the --help usage() heredoc both QUOTE the
# monolith-equivalent invocation as documentation.)
FORBIDDEN_MULTIPLEX_TOKENS = [
    "--skip-stages)",
    "--skip-stages=*)",
    "--pit-parity-enabled) PIT_PARITY_ENABLED",
    "--smoke-only)",
    "--mode) BACKTEST_MODE",
    "--mode=*) BACKTEST_MODE",
]


def _text(name: str) -> str:
    return (_INFRA / name).read_text()


def test_all_six_files_exist():
    assert _COMMON.is_file(), "_spot_common.sh missing"
    for name in STAGE_SCRIPTS:
        assert (_INFRA / name).is_file(), f"{name} missing"


def test_monolith_is_unchanged_entry_point_still_present():
    """spot_backtest.sh must stay the currently-wired SF path in this PR —
    the cutover is a separate nousergon-data PR, sequenced after."""
    monolith = _INFRA / "spot_backtest.sh"
    assert monolith.is_file()
    text = monolith.read_text()
    assert "--skip-stages" in text
    assert "BACKTEST_MODE" in text


def test_common_refuses_direct_execution():
    text = _COMMON.read_text()
    assert 'BASH_SOURCE[0]" == "${0}' in text or 'BASH_SOURCE[0]}" == "${0}' in text
    assert "must be sourced" in text


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_sources_common(name):
    text = _text(name)
    assert 'source "$SCRIPT_DIR/_spot_common.sh"' in text
    assert "spot_common_init_defaults" in text


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_fails_loud(name):
    text = _text(name)
    assert text.startswith("#!/usr/bin/env bash\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    assert lines[0] == "set -euo pipefail", (
        f"{name}: first non-comment/non-blank line must be `set -euo pipefail`, got {lines[0]!r}"
    )


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_carries_its_own_stage_name(name):
    text = _text(name)
    expected = STAGE_NAMES[name]
    assert f'SPOT_STAGE_NAME="{expected}"' in text


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_has_a_named_preflight_function(name):
    """weekly-sf-policy.md §2.2: every stage proves its preconditions
    before it spends. Each split script defines its OWN preflight_<x>()
    function (distinct from the shared dispatcher-side syntax/lib-pin
    check in _spot_common.sh), called before spot_common_launch_instance."""
    text = _text(name)
    m = re.search(r"^preflight_\w+\(\)\s*\{", text, re.MULTILINE)
    assert m, f"{name}: no preflight_<stage>() function defined"
    fn_name = text[m.start():].split("(", 1)[0].strip()
    # The function must actually be CALLED, and before the spot launch.
    call_idx = text.find(f"\n{fn_name}\n")
    launch_idx = text.index("spot_common_launch_instance")
    assert call_idx != -1, f"{name}: {fn_name}() defined but never called"
    assert call_idx < launch_idx, f"{name}: {fn_name}() must run BEFORE spot_common_launch_instance (before spend)"


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_has_no_multiplexing_flags(name):
    """The header docblock legitimately QUOTES the monolith-equivalent
    invocation (e.g. `--skip-stages=parity,evaluator`) as documentation —
    only the executable code must be free of stage-multiplexing dispatch."""
    text = _text(name)
    code_lines = [
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    for token in FORBIDDEN_MULTIPLEX_TOKENS:
        assert token not in code, (
            f"{name}: found stage-multiplexing token {token!r} in executable code — split "
            f"scripts must not re-implement --mode/--skip-stages/--pit-parity-enabled/"
            f"--smoke-only dispatch"
        )


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_supports_help_before_any_spend(name):
    """--help must be parsed and handled before spot_common_normalize_run_date
    / spot_common_launch_instance — the flag exists precisely so the script
    can be exercised (this repo's CI included) with zero AWS calls."""
    text = _text(name)
    assert "SHOW_HELP" in text
    help_idx = text.index('if [ "$SHOW_HELP" = "1" ]')
    launch_idx = text.index("spot_common_launch_instance")
    assert help_idx < launch_idx


def test_common_initializes_every_launch_scoped_var_unconditionally():
    """Motivating defect (2026-07-25 weekly SF recovery, the reason this
    issue exists): a variable initialized only inside one mode's heredoc
    branch was unbound under `set -u` in every OTHER mode
    (`_BACKTEST_WAS_SKIPPED` in the monolith hit exactly this class).
    Grep-based guard: every var this file's functions reference via `$VAR`
    or `${VAR` must have an unconditional (not inside `if`/`case`)
    assignment inside spot_common_init_defaults()."""
    text = _COMMON.read_text()
    m = re.search(
        r"^spot_common_init_defaults\(\) \{\n(.*?)\n\}\n",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert m, "spot_common_init_defaults() not found"
    body = m.group(1)

    # No line inside the init function may be indented under an `if`/`case`
    # (i.e. every assignment in this function is unconditional — the
    # function's own body has zero control-flow branches).
    forbidden = re.findall(r"^\s*(if|case)\b", body, re.MULTILINE)
    assert not forbidden, (
        f"spot_common_init_defaults() must assign every var UNCONDITIONALLY "
        f"(no if/case branching inside it) — found: {forbidden}"
    )

    required_vars = [
        "AWS_REGION", "S3_BUCKET", "BRANCH", "INSTANCE_TYPES", "INSTANCE_TYPE",
        "AMI_ID", "MAX_RUNTIME_SECONDS", "KEY_NAME", "SECURITY_GROUP", "SUBNETS",
        "IAM_PROFILE", "LIB_PYTHON", "SHOW_HELP", "PREFLIGHT_ONLY", "RUN_DATE",
        "MAX_SPOT_ATTEMPTS", "SPOT_ATTEMPT", "SF_EXECUTION_TIMEOUT",
        "BACKTEST_SKIP_PHASE4_FLAG", "BACKTEST_PHASE_FLAGS",
        "INSTANCE_ID", "RUN_ID", "S3_STAGING_PREFIX", "S3_STAGING",
        "LAST_SSM_DESC", "EXECUTOR_CONFIG", "PREDICTOR_CONFIG",
        "STAGED_PREDICTOR_CONFIG", "ENV_SOURCE", "REMOTE_PYTHON",
    ]
    for var in required_vars:
        assert re.search(rf"^\s*{re.escape(var)}=", body, re.MULTILINE), (
            f"{var} is not unconditionally initialized in spot_common_init_defaults()"
        )


@pytest.mark.parametrize("name", STAGE_SCRIPTS)
def test_stage_script_calls_init_defaults_before_parsing_flags(name):
    """Every script must call spot_common_init_defaults() (unconditional
    init of every var) BEFORE spot_common_parse_flags — otherwise a flag
    parse could reference a var that has no default yet under `set -u`."""
    text = _text(name)
    init_idx = text.index("spot_common_init_defaults")
    parse_idx = text.index("spot_common_parse_flags")
    assert init_idx < parse_idx
