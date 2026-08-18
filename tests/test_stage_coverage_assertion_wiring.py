"""Per-stage output-coverage assertion wiring (alpha-engine-config-I7214).

Brian's rescope ruling on I7214: the end-of-run `StageCoverageAssert` SF
state was NON-SOTA (a new state/Lambda/IAM surface). The assertion belongs
in each stage's OWN launcher script, asserting its declared output
immediately before the script exits, via the ONE shared implementation
landed as `krepis.stage_coverage` (a separate krepis PR). krepis, not
nousergon_lib, is the sanctioned `python -m` entrypoint namespace for
launcher-side dispatch: on nousergon-lib >=0.81.0, `python -m
nousergon_lib.<mod>` is a guard-less re-export shim that exits 0 SILENTLY
under runpy without executing the inner dispatch (nousergon-data#1646/
#1649 — see nousergon-data/tests/test_spot_data_weekly_ssm_transport.py::
test_uses_lib_ssm_dispatcher_chokepoint, which pins the same rule for
spot_data_weekly.sh). Every existing `-m` invocation in this repo's
launchers already targets krepis (`krepis.ec2_spot`, `krepis.
ssm_dispatcher`, `krepis.ssm_log_capture`) — zero `nousergon_lib -m`
precedent exists here. krepis is PyPI-published (not git-tag pinned), so
this repo's `krepis[openai]>=0.55.0` floor in requirements.txt already
covers a future release that adds the module — no pin bump needed.
This repo's launchers only ever CALL that module through the documented
CLI front door — they never reimplement its stage list or read
ARTIFACT_REGISTRY.yaml directly (policy-shared-code: no per-repo fork of
the shared primitive).

OBSERVE MODE ONLY (sf-pipeline-policy.md §2.1): the call sites below must
never set --enforce / STAGE_COVERAGE_ENFORCE=1, and a failed/absent module
must never fail the stage — the `|| echo ... >&2` fallback (never `|| true`,
which would make an absent module indistinguishable from a covered stage)
is asserted structurally.

krepis.stage_coverage does not exist at any published krepis release yet
(confirmed absent from the krepis 0.54.0 installed in this environment) —
these are static-analysis guards only; the SSM/EC2 runtime path (like the
rest of the test_spot_backtest_*.py / test_spot_stage_scripts_*.py suite)
is
not exercised in CI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from tests._sibling_checkout import is_ci, resolve_sf_defs_dir

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
_COMMON = _INFRA / "_spot_common.sh"

# Every spot_*.sh in THIS repo that the live weekly SF definition currently
# dispatches to, mapped to the SF Task-state name(s) it must assert
# against. Traced from nousergon-data/infrastructure/step_function.json's
# `commands` array for each Task state (2026-08-13) — each Backtester-
# family/pit-parity/parity state invokes its own dedicated script directly
# (the I4442/I4497 split cutover), NOT the shared monolith. spot_evaluator.sh
# is invoked TWICE, under --eval-half=diagnostics and --eval-half=optimize,
# for the two Evaluator SF states.
SF_WIRED_SINGLE_STAGE = {
    "spot_backtester.sh": "Backtester",
    "spot_predictor_backtest.sh": "PredictorBacktest",
    "spot_portfolio_optimizer_backtest.sh": "PortfolioOptimizerBacktest",
    "spot_pit_lookahead.sh": "PitParityLookahead",
    "spot_pit_walkforward.sh": "PitParityWalkforward",
    "spot_parity_replay.sh": "ParityReplay",
    "spot_parity_compare.sh": "PitParityCompare",
}

# spot_evaluator.sh serves two SF states via one script, distinguished by
# --eval-half — see the case statement asserted in
# test_evaluator_derives_coverage_stage_from_eval_half below.
SF_WIRED_MULTI_STAGE_SCRIPT = "spot_evaluator.sh"
SF_WIRED_MULTI_STAGE_MAPPING = {
    "diagnostics": "EvaluatorDiagnostics",
    "optimize": "EvaluatorOptimize",
}

# Explicit exclusions, WITH the reason each is not SF-wired today — an
# enumeration test that lists only what exists is blind to where one is
# missing, so every retired/rollback script this repo ships is named here
# rather than silently absent from SF_WIRED_SINGLE_STAGE.
NOT_SF_WIRED = {
    "spot_backtest.sh": (
        "retired monolith — every Task state that pre-I4442/I4497 called "
        "this with --mode/--skip-stages now calls its own dedicated "
        "per-stage script instead (retained only as the documented "
        "manual rollback path, per its own header docblock)."
    ),
    "spot_parity.sh": (
        "retired bundled-parity script — replaced by the ParityParallel "
        "branch's four independent scripts (spot_pit_lookahead.sh, "
        "spot_pit_walkforward.sh, spot_parity_replay.sh, "
        "spot_parity_compare.sh) per alpha-engine-config#6030."
    ),
    "spot_backtest_and_evaluate.sh": (
        "not referenced anywhere in nousergon-data/infrastructure/"
        "step_function.json — a standalone manual/local convenience "
        "wrapper, not an SF dispatch target."
    ),
}

ASSERT_MODULE_INVOCATION = "-m krepis.stage_coverage assert"


def _text(name: str) -> str:
    return (_INFRA / name).read_text()


def _all_spot_launcher_scripts() -> list[str]:
    return sorted(
        p.name
        for p in _INFRA.glob("spot_*.sh")
    )


# ── Totality: every spot_*.sh accounted for ──────────────────────────────


def test_every_spot_launcher_is_classified():
    """No spot_*.sh may be silently unaccounted-for: it is either in the
    SF-wired set (single- or multi-stage) or explicitly excluded with a
    reason in NOT_SF_WIRED. New scripts added later must update one of
    these three sets, or this test fails loudly instead of the totality
    guard quietly missing them."""
    classified = (
        set(SF_WIRED_SINGLE_STAGE)
        | {SF_WIRED_MULTI_STAGE_SCRIPT}
        | set(NOT_SF_WIRED)
    )
    unclassified = set(_all_spot_launcher_scripts()) - classified
    assert not unclassified, (
        f"spot_*.sh script(s) neither asserted-on nor explicitly excluded: "
        f"{sorted(unclassified)} — add to SF_WIRED_SINGLE_STAGE or "
        f"NOT_SF_WIRED (with reason)"
    )


def test_sf_wired_set_matches_live_step_function_json_when_reachable():
    """Best-effort cross-check against the live SF definition (sibling
    nousergon-data checkout).

    Resolution centralized in tests/_sibling_checkout.py (alpha-engine-
    config-I7605 / I7619, mirroring crucible-dashboard's reference fix): CI
    checks out nousergon-data (sparse) and sets ``SF_DEFS_DIR``; a dev
    laptop uses ``~/Development/nousergon-data``. The prior resolution via
    ``Path(__file__).resolve().parents[2]`` was itself an instance of the
    class this issue guards against — it silently resolved to the WRONG
    directory (one level too shallow) when this repo is checked out into a
    nested worktree (e.g. ``~/Development/.worktrees/<repo>-<branch>/``),
    always finding nothing and always taking the skip path with no signal
    that the resolution itself was broken.

    On CI a missing checkout is a broken guard, not an absent layout — hard
    fails rather than skipping (see module docstring: `krepis.stage_coverage`
    call sites are static-analysis guards only, but the invariant they
    protect is real and must be exercised somewhere). On a dev laptop
    without the nousergon-data sibling, skips with a named reason. When the
    checkout IS reachable, every script this test claims is SF-wired must
    actually appear in the JSON's `commands` blob, and every script
    explicitly excluded as retired must NOT appear as a live dispatch
    target."""
    sf_defs_dir = resolve_sf_defs_dir()
    sf_path = Path(sf_defs_dir) / "infrastructure" / "step_function.json"
    if not sf_path.is_file():
        message = (
            f"{sf_path} not present. CI checks out nousergon-data and sets "
            f"SF_DEFS_DIR; a dev laptop uses ~/Development/nousergon-data."
        )
        if is_ci():
            pytest.fail(
                f"{message} On CI this is a broken guard, not an absent "
                "layout — skipping here would report a cross-repo "
                "invariant as satisfied without ever evaluating it."
            )
        pytest.skip(message)

    blob = sf_path.read_text()
    for script in SF_WIRED_SINGLE_STAGE:
        assert script in blob, f"{script} claimed SF-wired but absent from live step_function.json"
    assert SF_WIRED_MULTI_STAGE_SCRIPT in blob

    for script, reason in NOT_SF_WIRED.items():
        if script == "spot_backtest_and_evaluate.sh":
            assert script not in blob, f"{script}: {reason} — but it IS present in the live JSON"


# ── Per-launcher assertion wiring ─────────────────────────────────────────


@pytest.mark.parametrize("script,stage", sorted(SF_WIRED_SINGLE_STAGE.items()))
def test_single_stage_launcher_asserts_its_own_stage(script, stage):
    text = _text(script)
    assert ASSERT_MODULE_INVOCATION in text, (
        f"{script}: missing `{ASSERT_MODULE_INVOCATION}` call"
    )
    assert '--stage "$_COVERAGE_STAGE"' in text, (
        f"{script}: assertion does not pass --stage \"$_COVERAGE_STAGE\""
    )
    # Hardcoded via _COVERAGE_STAGE, not re-derived from a flag — these
    # scripts are already 1:1 with one SF state (I4442/I4497 split).
    assert f'_COVERAGE_STAGE="{stage}"' in text


@pytest.mark.parametrize("script,stage", sorted(SF_WIRED_SINGLE_STAGE.items()))
def test_single_stage_launcher_assertion_is_on_the_success_path(script, stage):
    """The assertion must run AFTER the remote SSM heredoc has completed
    (i.e. after the stage's actual work finished), not before it and not
    inside the heredoc itself (which runs on a DIFFERENT box under
    $REMOTE_PYTHON, not the dispatcher-side $LIB_PYTHON)."""
    text = _text(script)
    assert_idx = text.index(ASSERT_MODULE_INVOCATION)
    # `run_ssm "..." <<HEREDOC_TAG ... HEREDOC_TAG` — the assertion must
    # come after the LAST heredoc closing delimiter, i.e. after the last
    # occurrence of the pattern `run_ssm "` in the file.
    run_ssm_idx = text.rindex('run_ssm "')
    assert assert_idx > run_ssm_idx, (
        f"{script}: stage-coverage assertion must run after the remote "
        f"SSM workload, not before/inside it"
    )
    line_start = text.rindex("\n", 0, assert_idx) + 1
    assert "$LIB_PYTHON" in text[line_start:assert_idx + len(ASSERT_MODULE_INVOCATION) + 5], (
        f"{script}: assertion must use dispatcher-side $LIB_PYTHON, not "
        f"$REMOTE_PYTHON (the remote spot box has no S3-check-only need "
        f"for the module and is torn down immediately after)"
    )


@pytest.mark.parametrize("script", sorted(SF_WIRED_SINGLE_STAGE) + [SF_WIRED_MULTI_STAGE_SCRIPT])
def test_launcher_assertion_never_bare_true_swallowed(script):
    """A bare `|| true` would make an absent module (expected until a
    krepis release carrying stage_coverage publishes) indistinguishable
    from a genuinely covered stage — the exact silence this mechanism
    exists to remove.
    The fallback must be a visible, stage-named WARNING on stderr."""
    text = _text(script)
    assert_idx = text.index(ASSERT_MODULE_INVOCATION)
    line_end = text.index("\n", assert_idx)
    line = text[text.rindex("\n", 0, assert_idx) + 1 : line_end]
    assert "|| true" not in line, f"{script}: assertion line swallows failure with `|| true`"
    assert "|| echo" in line and ">&2" in line, (
        f"{script}: assertion fallback must echo a WARNING to stderr, got: {line!r}"
    )
    assert "config-I7214" in line


@pytest.mark.parametrize("script", sorted(SF_WIRED_SINGLE_STAGE) + [SF_WIRED_MULTI_STAGE_SCRIPT])
def test_launcher_assertion_exit_code_never_fails_the_stage(script):
    """`set -euo pipefail` is active for the whole script — the assertion
    line MUST end in `|| <fallback>` (never a bare command) or a non-zero
    exit from the (currently absent) module would kill the launcher and
    fail the SF stage, which is exactly the hard-fail path Brian's
    OBSERVE-MODE ruling forbids before 2026-08-15."""
    text = _text(script)
    assert_idx = text.index(ASSERT_MODULE_INVOCATION)
    line_end = text.index("\n", assert_idx)
    line = text[text.rindex("\n", 0, assert_idx) + 1 : line_end]
    assert re.search(r"\|\|\s*echo", line), (
        f"{script}: assertion line has no `||` fallback — a real MISSING "
        f"or a missing module would fail the stage under set -e: {line!r}"
    )


def test_evaluator_derives_coverage_stage_from_eval_half():
    text = _text(SF_WIRED_MULTI_STAGE_SCRIPT)
    for eval_half, stage in SF_WIRED_MULTI_STAGE_MAPPING.items():
        pattern = rf'{re.escape(eval_half)}\)\s*_COVERAGE_STAGE="{re.escape(stage)}"'
        assert re.search(pattern, text), (
            f"spot_evaluator.sh: --eval-half={eval_half} does not map to "
            f"_COVERAGE_STAGE={stage!r}"
        )
    # The mapping must be a case/branch on $EVAL_HALF — never a hardcoded
    # single stage name (that would file one half's assertion under the
    # other's name).
    assert 'case "$EVAL_HALF" in' in text


def test_evaluator_assertion_guarded_by_derived_stage_not_hardcoded():
    text = _text(SF_WIRED_MULTI_STAGE_SCRIPT)
    assert_idx = text.index(ASSERT_MODULE_INVOCATION)
    preceding = text[:assert_idx]
    assert 'if [ -n "$_COVERAGE_STAGE" ]' in preceding[-400:], (
        "spot_evaluator.sh: assertion must be guarded on a non-empty "
        "derived _COVERAGE_STAGE (the --eval-half=all case maps to no "
        "single SF state)"
    )
    assert '--stage "$_COVERAGE_STAGE"' in text


# ── OBSERVE MODE — no promotion flags shipped ─────────────────────────────


@pytest.mark.parametrize("script", sorted(SF_WIRED_SINGLE_STAGE) + [SF_WIRED_MULTI_STAGE_SCRIPT, "_spot_common.sh"])
def test_call_sites_never_set_enforce(script):
    """The single promotion flag (--enforce / STAGE_COVERAGE_ENFORCE=1) is
    a deliberate, separately-reviewed diff — shipping it now would turn a
    real MISSING into a hard stage failure before Brian's 2026-08-15
    observe-mode window closes."""
    text = _text(script)
    assert "--enforce" not in text, f"{script}: must not set --enforce"
    assert "STAGE_COVERAGE_ENFORCE=1" not in text, f"{script}: must not set STAGE_COVERAGE_ENFORCE=1"
    assert "STAGE_COVERAGE_ENFORCE=true" not in text


# ── Shared window-start plumbing ──────────────────────────────────────────


def test_common_sets_stage_window_start_unconditionally_near_the_top():
    text = _COMMON.read_text()
    m = re.search(r'^_STAGE_WINDOW_START="\$\{_STAGE_WINDOW_START:-.*\}"', text, re.MULTILINE)
    assert m, "_spot_common.sh: _STAGE_WINDOW_START assignment not found"
    # Must be set before spot_common_init_defaults() is even defined, i.e.
    # near the top of the sourced file, so every launcher has it before
    # any flag parsing.
    init_fn_idx = text.index("spot_common_init_defaults()")
    assert m.start() < init_fn_idx


@pytest.mark.parametrize("script", sorted(SF_WIRED_SINGLE_STAGE) + [SF_WIRED_MULTI_STAGE_SCRIPT])
def test_launcher_passes_window_start_to_the_assertion(script):
    text = _text(script)
    assert_idx = text.index(ASSERT_MODULE_INVOCATION)
    line_end = text.index("\n", assert_idx)
    line = text[text.rindex("\n", 0, assert_idx) + 1 : line_end]
    assert '--window-start "$_STAGE_WINDOW_START"' in line, (
        f"{script}: assertion must pass --window-start \"$_STAGE_WINDOW_START\""
    )
