"""Pins the alpha-engine-config-I7179 cost-sink environment wiring on the
concordance Lambda, and guards against the retrofit that would defeat it.

``replay/runner.py``'s ``_invoke_target_model`` builds
``krepis.llm.LLMClient`` with no ``cost_sink`` — every DeepSeek call the
concordance Lambda makes was landing on no per-call cost record, so its
spend was attributed to nothing. The fix is NOT a ``cost_sink=`` passed at
that call site (krepis-PR140 makes that a per-call-site retrofit that
reproduces the gap for the next call site added) — it is
``infrastructure/deploy_concordance.sh`` merging
``KREPIS_COST_SINK_BUCKET``/``KREPIS_COST_SINK_PREFIX`` onto the live Lambda
so krepis>=0.57.0's ``LLMClient`` resolves a default sink from the
environment.

Both literal values are asserted explicitly: the whole failure class this
guards is a value that *looks* right (present, non-empty, plausible bucket
name) and points nowhere — a wrong prefix silently makes every cost row
invisible to the aggregator, exactly as already happened once in
crucible-research/thinktank/run.py.
"""

from __future__ import annotations

from pathlib import Path

_DEPLOY_SCRIPT = (
    Path(__file__).resolve().parent.parent
    / "infrastructure"
    / "deploy_concordance.sh"
)
_RUNNER = Path(__file__).resolve().parent.parent / "replay" / "runner.py"

_EXPECTED_BUCKET = "alpha-engine-research"
_EXPECTED_PREFIX = "decision_artifacts/_cost_raw"


def _deploy_text() -> str:
    return _DEPLOY_SCRIPT.read_text()


def _runner_text() -> str:
    return _RUNNER.read_text()


def test_deploy_script_merges_cost_sink_env_onto_concordance_function():
    text = _deploy_text()
    assert "krepis.aws merge-lambda-env" in text, (
        "deploy_concordance.sh must wire the cost-sink environment via "
        "krepis.aws merge-lambda-env (I7179) rather than a per-call-site "
        "cost_sink= retrofit in replay/runner.py"
    )

    idx = text.index("krepis.aws merge-lambda-env")
    # The merge-lambda-env invocation is one line — pull a generous window
    # around it rather than assuming exact line boundaries.
    window = text[max(0, idx - 200) : idx + 600]

    assert "alpha-engine-replay-concordance" in window, (
        "merge-lambda-env must target the concordance function by name"
    )
    assert f"KREPIS_COST_SINK_BUCKET={_EXPECTED_BUCKET}" in window, (
        f"cost-sink bucket must be the literal '{_EXPECTED_BUCKET}' — a "
        "different bucket silently makes every cost row unrecoverable"
    )
    assert f"KREPIS_COST_SINK_PREFIX={_EXPECTED_PREFIX}" in window, (
        f"cost-sink prefix must be the literal '{_EXPECTED_PREFIX}' — a "
        "different prefix makes the rows invisible to the aggregator "
        "(this exact mistake previously hit crucible-research/thinktank/run.py)"
    )


def test_cost_sink_merge_runs_before_publish_version():
    text = _deploy_text()
    merge_idx = text.index("krepis.aws merge-lambda-env")
    publish_idx = text.index("aws lambda publish-version")
    assert merge_idx < publish_idx, (
        "the cost-sink env merge must land BEFORE publish-version — "
        "publishing first would promote a version whose config predates "
        "the sink wiring"
    )


def test_runner_does_not_construct_its_own_cost_sink():
    text = _runner_text()
    assert "S3JsonlCostSink" not in text, (
        "replay/runner.py must not construct an S3JsonlCostSink itself — "
        "the cost sink is an environment fact (wired by "
        "deploy_concordance.sh's merge-lambda-env), not a call-site fact. "
        "A call-site retrofit here reproduces the I7179 gap for the next "
        "krepis.llm.LLMClient call site added."
    )
    assert "cost_sink=" not in text, (
        "replay/runner.py must not pass cost_sink= explicitly to "
        "LLMClient — see module note above"
    )
