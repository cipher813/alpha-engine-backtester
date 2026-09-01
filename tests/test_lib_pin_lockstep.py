"""Pin ``requirements.txt`` and all three Lambda ``Dockerfile``s to the same
alpha-engine-lib version.

The backtester repo ships three Lambdas (lambda_health, lambda_concordance,
lambda_counterfactual), each with its own Dockerfile and hardcoded
``pip install "alpha-engine-lib@vX.Y.Z"`` line. They don't read
``requirements.txt`` for the lib install, so bumping the project-root pin
alone leaves all three Lambda images stuck on whatever tag was hardcoded.

This drift class has bitten production multiple times across the org:

  - 2026-05-06 (research): requirements.txt bumped @v0.4.0 → @v0.5.1
    but Dockerfile kept v0.3.0; Research Lambda canary failed with
    ``ModuleNotFoundError: alpha_engine_lib.agent_schemas``.
  - 2026-05-12 (predictor + data): requirements bumped to @v0.12.0 but
    Lambda-side pins stayed stale; canary failed with
    ``ModuleNotFoundError: alpha_engine_lib.secrets``.

This test re-greps all four deploy artifacts on every CI run.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Distribution name renamed alpha-engine-lib -> nousergon-lib at lib 0.60.0
# (config#1245 / #1172). Accept either spelling so the lockstep guard keeps
# working across the crossing; the version-equality assertion below is the
# load-bearing part.
#
# alpha-engine-config-I7605: also accepts a bare 40-hex commit SHA, not just
# a vX.Y.Z tag. All four artifacts here provisionally pin nousergon-lib-PR325
# (the producer_champion_audit contract, unmerged/untagged at PR-open time)
# by its branch-tip commit — the branch is reachable regardless of merge
# order, unlike a tag that doesn't exist yet. This is a documented interim
# state, not a permanent convention: once PR325 merges and autobumps a real
# tag, a follow-up commit moves all four pins to that tag and this SHA
# alternative stops matching anything in this repo again.
_REQUIREMENTS_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+|[0-9a-f]{40})"
)
_DOCKERFILE_PIN_RE = re.compile(
    r'"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/nousergon/nousergon-lib@(v[0-9]+\.[0-9]+\.[0-9]+|[0-9a-f]{40})"'
)


def _read_pin(filename: str, regex: re.Pattern[str]) -> str:
    text = (_REPO_ROOT / filename).read_text()
    match = regex.search(text)
    assert match is not None, (
        f"could not find alpha-engine-lib pin in {filename}"
    )
    return match.group(1)


def test_all_deploy_artifacts_pin_same_lib_version():
    pins = {
        "requirements.txt": _read_pin("requirements.txt", _REQUIREMENTS_PIN_RE),
        "lambda_health/Dockerfile": _read_pin("lambda_health/Dockerfile", _DOCKERFILE_PIN_RE),
        "lambda_concordance/Dockerfile": _read_pin(
            "lambda_concordance/Dockerfile", _DOCKERFILE_PIN_RE
        ),
        "lambda_counterfactual/Dockerfile": _read_pin(
            "lambda_counterfactual/Dockerfile", _DOCKERFILE_PIN_RE
        ),
    }
    unique = set(pins.values())
    assert len(unique) == 1, (
        f"alpha-engine-lib pin drift across deploy artifacts:\n"
        + "\n".join(f"  {name}: {pin}" for name, pin in pins.items())
        + "\n\nAll four must move in lockstep — each Lambda Dockerfile has "
        f"its own hardcoded `pip install \"alpha-engine-lib@vX.Y.Z\"` line "
        f"that is independent of requirements.txt."
    )


# ── A commit-SHA pin is a CI failure here, not a Step Functions degradation ──
#
# alpha-engine-config-I7301 deliverable 3. This repo is one half of the only
# multi-repo co-install site in the fleet: `infrastructure/spot_backtest.sh`
# installs THIS repo's requirements and then `crucible-predictor`'s into ONE
# venv, so the two nousergon-lib pins must be equal or the second install
# silently downgrades the shared lib (the 2026-05-12 incident).
#
# `crucible-predictor` pinned the lib to a bare commit SHA on 2026-07-31
# (crucible-predictor#422). The commit was not an ancestor of `main` — the
# branch was squash-merged — so production installed a tree that never landed
# and the parity invariant went unverifiable for thirteen days. Nothing in CI
# said so. The first signal was the weekly Step Function's `LibPinDriftCheck`
# reporting `reason: sha_pinned` and degrading the run, which is the wrong
# layer: a committed repo state that recurs on every run until a human edits
# the file is a CI failure, not a runtime gate that fires forever.
#
# The tag-shaped `_read_pin` assertions above already fail on a SHA pin, but
# they fail as *"could not find alpha-engine-lib pin"* — a message that reads
# like a moved file or a renamed dependency. This test names the condition, so
# the red build states the fix.
_SHA_PIN_RE = re.compile(
    r"(?:alpha-engine-lib|nousergon-lib)\[[^\]]*\]\s*@\s*git\+https://github\.com/"
    r"nousergon/nousergon-lib@([0-9a-f]{7,40})\b"
)

_PIN_FILES = (
    "requirements.txt",
    "lambda_health/Dockerfile",
    "lambda_concordance/Dockerfile",
    "lambda_counterfactual/Dockerfile",
)


def test_no_deploy_artifact_pins_the_lib_by_commit_sha():
    """Every alpha-engine-lib pin is a `vX.Y.Z` release tag.

    A commit SHA is not comparable to `crucible-predictor`'s tag or to the
    weekly pipeline's `MIN_LIB_VERSION` floor, so it makes the co-install
    parity invariant permanently unverifiable rather than merely unsatisfied.
    """
    offenders = {
        name: match.group(1)
        for name in _PIN_FILES
        if (match := _SHA_PIN_RE.search((_REPO_ROOT / name).read_text()))
    }
    assert not offenders, (
        "alpha-engine-lib is pinned by commit SHA in:\n"
        + "\n".join(f"  {name}: {sha}" for name, sha in offenders.items())
        + "\n\nPin a released vX.Y.Z tag instead, equal to crucible-predictor's "
        "pin. A SHA pin cannot be compared to the co-install partner's pin or "
        "to the weekly pipeline's MIN_LIB_VERSION floor, so `LibPinDriftCheck` "
        "reports `sha_pinned` and DEGRADES the Saturday run on every "
        "invocation until the file is edited (alpha-engine-config-I7301)."
    )
