"""A failure path must not delete the evidence its own message points at.

**The 2026-08-15 weekly-SF failure (alpha-engine-config-I7396 → I7442).** The
`PredictorBacktest` stage died and printed:

    ERROR: SSM step 'predictor-backtest' terminal status=Failed …
      — full remote log: s3://alpha-engine-research/tmp/spot_predictor-backtest/
        20260815T123311Z-i-08a4371deec28ef07/ssm-output/

Four lines later, the same exit path printed *"Instance terminated; S3 staging
cleaned."* The prefix the error named was **empty** by the time anyone read it,
and so was its parent.

That copy is not redundant. SSM's ``GetCommandInvocation`` returns only the
**first** 24 KB of stdout, so on any long stage the tail — which is where a
traceback lives — exists nowhere else. A message pointing at evidence the same
exit path just removed is worse than no message.

**What changed at I7442.** #675 fixed this repo with a retain-on-failure branch
plus a per-stage launch-time prune. Three sibling `_spot_common.sh` copies and
four monoliths still ran the unguarded delete, and mirroring 55 lines of bash
into each is what `policy-shared-code` forbids. Teardown now runs through
``krepis.spot_evidence teardown``, which copies the prefix to
``_spot_evidence/`` and deletes staging only if that copy succeeded — so the
ordering is a property of that module's call graph rather than of a branch
here, and it holds for every launcher family at once.

These tests therefore pin the CLASS property: no launcher in this repo may
carry an unguarded staging delete, and the teardown must degrade to retention
rather than to deletion.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parent.parent / "infrastructure"
COMMON = INFRA / "_spot_common.sh"


@pytest.fixture(scope="module")
def body() -> str:
    assert COMMON.is_file(), f"{COMMON} missing"
    return COMMON.read_text(encoding="utf-8")


def _teardown_fn(body: str) -> str:
    """The teardown helper is defined INSIDE `spot_common_install_cleanup_trap`.

    Deliberately nested: `tests/test_spot_cleanup_reaches_terminate_on_hold.py`
    lifts the trap installer alone into a harness, and a sibling top-level
    helper would be `command not found` there — which under `set -e` in an EXIT
    trap replaced the workload's exit 3 with a 127. A teardown step must never
    be able to rewrite the status it is tearing down.
    """
    start = body.index("    spot_common_teardown_staging() {")
    return body[start : body.index("\n    }\n", start)]


def _cleanup_fn(body: str) -> str:
    start = body.index("    cleanup() {")
    end = body.index('\n        exit "$exit_code"', start)
    return body[start:end]


class TestNoLauncherDeletesItsOwnStaging:
    """The class guard. Fixing one call site of a systemic defect is not a fix."""

    def test_no_shell_script_in_infrastructure_removes_S3_STAGING(self):
        offenders = []
        for path in sorted(INFRA.glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "aws s3 rm" in stripped and "S3_STAGING" in stripped:
                    offenders.append(f"{path.name}:{n}: {stripped}")
        assert not offenders, (
            "an unguarded staging delete is back — this is the "
            "alpha-engine-config-I7442 defect, and it destroys the only "
            "un-truncated copy of a failure's output:\n  "
            + "\n  ".join(offenders)
        )

    def test_every_launcher_that_stages_also_tears_down_through_the_chokepoint(self):
        for path in sorted(INFRA.glob("spot_*.sh")):
            text = path.read_text(encoding="utf-8")
            if "S3_STAGING=" not in text:
                continue  # sources _spot_common.sh, covered below
            assert "krepis.spot_evidence teardown" in text, (
                f"{path.name} provisions its own staging prefix but never "
                "hands it to the teardown chokepoint"
            )


class TestTeardownGoesThroughTheChokepoint:
    def test_cleanup_calls_the_shared_teardown(self, body):
        assert "spot_common_teardown_staging \"$exit_code\"" in _cleanup_fn(body)

    def test_the_teardown_helper_invokes_krepis(self, body):
        fn = _teardown_fn(body)
        assert "krepis.spot_evidence teardown" in fn
        assert "--exit-code" in fn, (
            "the workload's exit status is what decides whether evidence is "
            "preserved; without it every run looks like a success"
        )
        assert "--staging" in fn and "--slug" in fn

    def test_an_unavailable_chokepoint_degrades_to_RETENTION_not_deletion(self, body):
        """The merge-order safety property, and the fail-safe direction.

        A box whose krepis pin predates `spot_evidence` must keep the evidence,
        never fall back to the delete this whole change exists to remove.
        """
        fn = _teardown_fn(body)
        assert "RETAINED" in fn
        code = "\n".join(
            line for line in fn.splitlines() if not line.strip().startswith("#")
        )
        assert "aws s3 rm" not in code

    def test_the_teardown_never_aborts_the_exit_path(self, body):
        fn = _teardown_fn(body)
        assert fn.rstrip().endswith("return 0"), (
            "a janitor that can change the trap's exit status masks the "
            "workload's own failure"
        )

    def test_the_retired_bash_prune_is_gone(self, body):
        """It moved into `krepis.spot_evidence`, which sweeps both trees on
        every teardown — including for a stage that has stopped running, which
        a launch-time prune could never reach."""
        assert "spot_common_prune_stale_staging" not in body
        assert "SPOT_STAGING_RETENTION_DAYS" not in body


class TestTheResourceLimitFlagWaitsForThePinBump:
    """`--resource-limit` is deliberately ABSENT, and that is load-bearing.

    It is a NEW `krepis.ssm_dispatcher` flag (krepis-PR161). `$LIB_PYTHON` is
    the dispatch box's venv, pinned to a krepis release that predates it, so
    argparse would reject the unknown flag and EVERY SSM step would fail on
    merge — in exactly the window before the pin bump. `krepis.spot_evidence`
    degrades safely when absent (the teardown retains); this flag has no safe
    degradation at all.

    It lands with the pin bump, `alpha-engine-config-I7556`. Until then this
    test is what stops it being reintroduced by someone reading
    sf-pipeline-policy §3 obligation 3 and adding the obvious line.
    """

    def test_no_launcher_passes_the_flag_yet(self):
        offenders = []
        for path in sorted(INFRA.glob("*.sh")):
            for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                if "--resource-limit" in line:
                    offenders.append(f"{path.name}:{n}")
        assert not offenders, (
            "--resource-limit is passed to krepis.ssm_dispatcher before the "
            "dispatch box's krepis pin ships it (alpha-engine-config-I7556). "
            "An unknown argparse flag fails EVERY SSM step: "
            + ", ".join(offenders)
        )
