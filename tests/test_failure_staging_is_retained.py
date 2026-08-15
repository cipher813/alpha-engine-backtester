"""A failure path must not delete the evidence its own message points at.

**The 2026-08-15 weekly-SF failure (alpha-engine-config-I7396).** The
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

Retention has to be bounded by something, so each stage prunes its own staging
prefix at launch. These tests pin both halves: they ship together and cannot
drift apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

COMMON = Path(__file__).resolve().parent.parent / "infrastructure" / "_spot_common.sh"


@pytest.fixture(scope="module")
def body() -> str:
    assert COMMON.is_file(), f"{COMMON} missing"
    return COMMON.read_text(encoding="utf-8")


def _cleanup_fn(body: str) -> str:
    start = body.index("    cleanup() {")
    end = body.index('\n        exit "$exit_code"', start)
    return body[start:end]


class TestFailureRetainsTheEvidence:
    def test_the_staging_delete_is_conditional(self, body):
        fn = _cleanup_fn(body)
        for n, line in enumerate(fn.split("\n"), 1):
            if "aws s3 rm" in line and "S3_STAGING" in line:
                break
        else:
            pytest.fail("the staging delete vanished entirely — that is not this fix")
        assert 'if [ "$exit_code" -ne 0 ]; then' in fn, (
            "the staging delete is unconditional again: a failed stage will "
            "delete the only full copy of its own output"
        )

    def test_the_success_path_still_cleans(self, body):
        """Retention is failure-only; steady-state storage must be unchanged."""
        fn = _cleanup_fn(body)
        tail = fn[fn.index("Terminating spot instance"):]
        else_at = tail.index("else")
        rm_at = tail.index("aws s3 rm")
        assert rm_at > else_at, (
            "the staging delete moved out of the success branch — either "
            "nothing is cleaned, or the failure branch deletes too"
        )

    def test_the_failure_branch_prints_where_the_evidence_is(self, body):
        fn = _cleanup_fn(body)
        assert "RETAINED for diagnosis" in fn
        assert "ssm-output" in fn, (
            "the retained pointer must name the ssm-output subprefix — the "
            "generic staging path is not where the operator needs to look"
        )

    def test_no_line_claims_cleaned_on_the_failure_path(self, body):
        """The 2026-08-15 message said 'staging cleaned' on a FAILED run."""
        fn = _cleanup_fn(body)
        head, _, tail = fn.partition('if [ "$exit_code" -ne 0 ]; then\n            # config-I7396')
        assert tail, "the retention branch is gone"
        failure_branch = tail.split("        else")[0]
        assert "staging cleaned" not in failure_branch


class TestRetentionIsBounded:
    def test_a_prune_helper_exists(self, body):
        assert "spot_common_prune_stale_staging()" in body, (
            "staging is retained on failure with nothing bounding it"
        )

    def test_the_prune_runs_at_launch(self, body):
        assert re.search(r"^\s*spot_common_prune_stale_staging\s*$", body, re.M), (
            "the prune helper is defined but never called"
        )

    def test_the_prune_is_scoped_to_this_stage_only(self, body):
        start = body.index("spot_common_prune_stale_staging() {")
        fn = body[start:body.index("\n}\n", start)]
        assert 'tmp/spot_${SPOT_STAGE_NAME}/' in fn, (
            "the prune is not scoped to this stage's own prefix — one stage "
            "must never delete another's retained evidence"
        )
        assert "--recursive" in fn

    def test_the_retention_window_is_named_and_overridable(self, body):
        assert "SPOT_STAGING_RETENTION_DAYS" in body
        assert re.search(r'SPOT_STAGING_RETENTION_DAYS="\$\{SPOT_STAGING_RETENTION_DAYS:-\d+\}"', body)

    def test_the_prune_never_aborts_the_launch(self, body):
        """A housekeeping step must not stop the run it houses."""
        start = body.index("spot_common_prune_stale_staging() {")
        fn = body[start:body.index("\n}\n", start)]
        assert fn.rstrip().endswith("return 0"), (
            "the prune can propagate a non-zero status into a `set -e` launch"
        )
