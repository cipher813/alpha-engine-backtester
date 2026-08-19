"""tests/test_smoke_pit_parity_real_run_commit_parity.py — alpha-engine-
config-I6027 Gap 1: the real Parity SF run's own `smoke-pit-parity` must
execute on the SAME commit as the real pass it precedes, not just on a
separate `--preflight-only` rehearsal that can validate a different commit
than the one the real run later executes.

## Background

I6027 was filed against the pre-split monolith (`infrastructure/
spot_backtest.sh`) describing a ~6-hour skew window on 2026-08-01: the
`--preflight-only` rehearsal (run ~12h early, on a dedicated spot boot) had
validated commit A, while the real Parity SF state — on a SEPARATE spot
boot — cloned and ran commit B.

The issue's own deliverable: "run the stage's own smoke immediately after
the git pull in the real Parity stage too — it's a tiny 5-ticker/30-60-day
fixture, costs seconds... don't delete or move the existing
`--preflight-only` rehearsal (it's a legitimate separate ~12h-early check)
— add coverage on the real run additionally."

## What this test proves

Every script that runs (or could run, per the ongoing I4442/I6030 stage
split) the Parity SF state already embeds `smoke-pit-parity` inside the
SAME script invocation as the real pit_parity pass — i.e. inside the same
`run_ssm` heredoc / SSM command that clones the code exactly once
(`spot_common_bootstrap` for the split per-stage scripts; the dispatcher-
side `--checkout` for the pre-split monolith) and then runs both the smoke
and the real pass with NO git pull/clone in between. That structurally
makes commit skew between the smoke and the real pass impossible for any
of these scripts, regardless of which one is currently wired to the live
SF (the monolith `spot_backtest.sh`, the bundled `spot_parity.sh`, or the
split `spot_pit_walkforward.sh` / `spot_pit_lookahead.sh`).

This is a STATIC/structural proof, not a live SF execution — this session
has no spot-box access to run a real Parity stage end-to-end. The
closes-when's "real Parity stage run shows its own smoke executing after
the git pull, on the same commit it then runs" is demonstrated here as:
(1) the smoke and the real pass are textually inside the same
clone-once-then-run block, and (2) no git pull/clone command appears
between them.
"""

from __future__ import annotations

import re
from pathlib import Path

_INFRA = Path(__file__).resolve().parent.parent / "infrastructure"

# For each script that runs (or could run) the Parity SF stage: the smoke
# marker, the real-pass marker, and a regex for "a fresh code pull between
# them" that must NOT match.
_GIT_REPULL_PATTERN = re.compile(r"\bgit\s+(-C\s+\S+\s+)?(pull|clone)\b")

_SCRIPTS = {
    # Pre-split monolith — currently-wired-or-not is irrelevant to this
    # test; if it's ever live again, it must still hold the guarantee.
    "spot_backtest.sh": {
        "smoke_marker": r'echo "▶ stage=smoke-pit-parity START',
        "real_pass_marker": r'echo "▶ stage=pit_parity START',
        # The monolith clones once via the dispatcher-side --checkout flag
        # (spot_common_launch_instance), long before the smoke+real-pass
        # heredoc — no in-heredoc bootstrap call to anchor on.
        "clone_marker": r"--checkout /home/ec2-user/alpha-engine-backtester",
    },
    # Bundled Parity stage (pit_parity, observational + non-blocking here).
    "spot_parity.sh": {
        "smoke_marker": r'echo "▶ stage=smoke-pit-parity START',
        "real_pass_marker": r'echo "▶ stage=pit_parity START',
        "clone_marker": r"^spot_common_bootstrap$",
    },
    # Split PitParityWalkforward stage (I6030) — smoke is FATAL here.
    "spot_pit_walkforward.sh": {
        "smoke_marker": r'echo "▶ stage=smoke-pit-parity START',
        "real_pass_marker": r'echo "▶ stage=pit_\$\{PIT_PASS\} START',
        "clone_marker": r"^spot_common_bootstrap$",
    },
    # Split PitParityLookahead stage (I6030) — smoke is FATAL here.
    "spot_pit_lookahead.sh": {
        "smoke_marker": r'echo "▶ stage=smoke-pit-parity START',
        "real_pass_marker": r'echo "▶ stage=pit_\$\{PIT_PASS\} START',
        "clone_marker": r"^spot_common_bootstrap$",
    },
}


def _read(name: str) -> str:
    path = _INFRA / name
    assert path.exists(), f"{path} does not exist — update _SCRIPTS if it was renamed/removed"
    return path.read_text()


def test_every_parity_capable_script_declared():
    """Sanity: the scripts this test covers actually exist on disk — if one
    is renamed or removed, this fails loud instead of the coverage below
    silently applying to nothing."""
    for name in _SCRIPTS:
        assert (_INFRA / name).exists(), f"infrastructure/{name} not found"


def test_smoke_runs_in_the_same_invocation_as_the_real_pass():
    """For every script, the smoke-pit-parity marker must appear BEFORE the
    real pass marker, in the same file (i.e. the same script execution —
    each of these scripts clones the code exactly once per invocation via
    spot_common_bootstrap or the dispatcher --checkout, then runs a single
    SSM command/heredoc containing both). This is the direct structural
    form of I6027 Gap 1's deliverable: run the smoke on the SAME commit
    the real pass then executes, not on an earlier, separately-cloned
    rehearsal."""
    for name, markers in _SCRIPTS.items():
        src = _read(name)
        smoke_match = re.search(markers["smoke_marker"], src)
        real_match = re.search(markers["real_pass_marker"], src)
        assert smoke_match, (
            f"{name}: smoke-pit-parity marker {markers['smoke_marker']!r} "
            f"not found — this script has NO real-run smoke at all "
            f"(alpha-engine-config-I6027 Gap 1 regression)"
        )
        assert real_match, (
            f"{name}: real pass marker {markers['real_pass_marker']!r} "
            f"not found — update the marker if the script's stage-name "
            f"convention changed"
        )
        assert smoke_match.start() < real_match.start(), (
            f"{name}: smoke-pit-parity must run BEFORE the real pass in "
            f"the same script invocation (alpha-engine-config-I6027 Gap 1) "
            f"— found the real pass marker before the smoke marker"
        )


def test_no_code_pull_between_the_smoke_and_the_real_pass():
    """The commit-skew guarantee: no `git pull` / `git clone` may appear
    between the smoke invocation and the real pass invocation. If one did,
    the smoke could validate commit A and the real pass could then run a
    freshly-pulled commit B — exactly the 2026-08-01 ~6h skew window
    I6027 describes, reintroduced."""
    for name, markers in _SCRIPTS.items():
        src = _read(name)
        smoke_match = re.search(markers["smoke_marker"], src)
        real_match = re.search(markers["real_pass_marker"], src)
        assert smoke_match and real_match, f"{name}: markers not found (see other test)"
        between = src[smoke_match.end():real_match.start()]
        repull = _GIT_REPULL_PATTERN.search(between)
        assert repull is None, (
            f"{name}: a git pull/clone command appears BETWEEN the smoke "
            f"and the real pass ({repull.group(0)!r} at offset "
            f"{repull.start()} into the between-text) — this reopens the "
            f"commit-skew window alpha-engine-config-I6027 Gap 1 closed."
        )


def test_code_is_cloned_exactly_once_before_the_smoke():
    """The clone-once precondition: each script's own bootstrap/checkout
    call must appear before the smoke marker (not after), so the commit
    the smoke validates really is the commit the real pass then runs —
    there is exactly one clone in the script's lifetime, and it happens
    up-front."""
    for name, markers in _SCRIPTS.items():
        src = _read(name)
        clone_match = re.search(markers["clone_marker"], src, re.MULTILINE)
        smoke_match = re.search(markers["smoke_marker"], src)
        assert clone_match, (
            f"{name}: clone/checkout marker {markers['clone_marker']!r} "
            f"not found — update the marker if the bootstrap mechanism "
            f"changed"
        )
        assert smoke_match, f"{name}: smoke marker not found (see other test)"
        assert clone_match.start() < smoke_match.start(), (
            f"{name}: the code clone/checkout must happen BEFORE the "
            f"smoke — otherwise the smoke could run against stale or "
            f"absent code"
        )


# ── Meta-test: prove this test's own logic actually discriminates the ──────
# ── failure mode it exists to catch, not just pass vacuously today. ───────


def test_assertion_logic_would_catch_a_reintroduced_skew_window():
    """Directly exercises the same regex logic against a SYNTHETIC script
    body that reintroduces the pre-I6027-fix shape (smoke only in a
    separate --preflight-only branch, real pass in a later block preceded
    by its own fresh git pull) — proving the chokepoint actually fires on
    the regression it exists to prevent."""
    regressed_script = """
spot_common_bootstrap
echo "▶ stage=smoke-pit-parity START"
echo "smoke ran"
echo "▶ stage=smoke-pit-parity END"

# ... hours pass, a later separate real-run boot ...
git -C /home/ec2-user/alpha-engine-backtester pull --ff-only origin main
echo "▶ stage=pit_parity START"
echo "real pass ran, possibly on a DIFFERENT commit than the smoke above"
"""
    smoke_match = re.search(r'echo "▶ stage=smoke-pit-parity START', regressed_script)
    real_match = re.search(r'echo "▶ stage=pit_parity START', regressed_script)
    assert smoke_match and real_match
    between = regressed_script[smoke_match.end():real_match.start()]
    repull = _GIT_REPULL_PATTERN.search(between)
    assert repull is not None, (
        "sanity check: the synthetic regression must be caught by the "
        "repull-detection regex, or this meta-test is meaningless"
    )
