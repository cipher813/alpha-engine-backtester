"""The replay Lambda deploys must strip denied environment variables, and
must do it where the promotion can carry the change to traffic.

alpha-engine-config-I7925. `alpha-engine-replay-concordance` and
`alpha-engine-replay-counterfactual` were two of eleven fleet-wide Lambdas
carrying a `GITHUB_TOKEN` set by hand and refreshed by nothing — the
environment is live-only state that no repo, IaC file or script here ever
wrote. The environment carried a STALE COPY of the credential — set
from an older SSM parameter version and never re-derived on deploy —
that GitHub rejected while the SSM parameter's own value remained valid
the whole time (alpha-engine-config-I7968 tracks the mis-attribution).
On 2026-08-21 a first-party
dependency picked it up out of site-packages, sent it to GitHub, got a 401,
and halted the preopen trading pipeline 3.4 seconds after start
(alpha-engine-config-I7924). `alpha-engine-predictor-inference`
(crucible-predictor) was the one of eleven that broke and was fixed first
(crucible-predictor-PR535).

I7925's own audit attributed these two functions to crucible-research.
Verified live: `infrastructure/deploy_concordance.sh` and `infrastructure/
deploy_counterfactual.sh`, in THIS repo, own and deploy them (ECR repo name,
LAMBDA_FUNCTION, IAM role, and the GHA deploy.yml matrix all name this
repo) — crucible-research's deploy.sh has no reference to either function.

Both scripts now converge the environment against a deny-list before
`publish-version`. These tests pin the two properties that make that
convergence real, because both fail SILENTLY: a removal placed after
`publish-version` never reaches the published version the `live` alias
serves (L4497), and a removal that also promotes the alias would race the
deploy's own promotion (Step 8/the final `update-alias` in each script).
"""

from __future__ import annotations

from pathlib import Path

import pytest

_INFRA = Path(__file__).resolve().parents[1] / "infrastructure"
_SCRIPTS = {
    "concordance": _INFRA / "deploy_concordance.sh",
    "counterfactual": _INFRA / "deploy_counterfactual.sh",
}


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_deploy_script_exists(name: str) -> None:
    assert _SCRIPTS[name].is_file(), f"{_SCRIPTS[name]} is missing"


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_github_token_is_on_the_deny_list(name: str) -> None:
    """The credential that caused I7924 must be named, not merely implied."""
    code = _SCRIPTS[name].read_text(encoding="utf-8")
    assert "LAMBDA_ENV_DENIED_KEYS=(" in code, (
        f"{name}: the deploy no longer declares a denied-key set — a "
        "variable set by hand now outlives every deploy again "
        "(alpha-engine-config-I7925)"
    )
    declaration = code.split("LAMBDA_ENV_DENIED_KEYS=(", 1)[1].split(")", 1)[0]
    assert "GITHUB_TOKEN" in declaration


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_removal_uses_the_shared_cli_not_a_bare_aws_call(name: str) -> None:
    """`aws lambda update-function-configuration --environment` REPLACES the
    whole variable map, deleting every operator-set flag codified nowhere
    (the router-addressing vars this script itself merges in, for
    concordance). The read-modify-write chokepoint is
    `krepis.aws remove-lambda-env`."""
    code = _SCRIPTS[name].read_text(encoding="utf-8")
    assert "krepis.aws remove-lambda-env" in code


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_removal_runs_before_the_version_is_published(name: str) -> None:
    """A removal after `publish-version` mutates $LATEST only. The published
    version — and therefore the `live` alias — would keep the variable, and
    the deploy would report success having changed nothing that serves
    traffic."""
    code = _SCRIPTS[name].read_text(encoding="utf-8")
    remove_at = code.index("remove-lambda-env")
    publish_at = code.index("aws lambda publish-version")
    assert remove_at < publish_at, (
        f"{name}: the environment convergence must precede publish-version, "
        "or the published version keeps the denied variable (L4497)"
    )


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_removal_defers_promotion_to_the_deploy(name: str) -> None:
    """Both scripts publish a version and only promote the `live` alias
    later, on canary success. A removal that also promoted would publish a
    second version mid-deploy and race that promotion."""
    code = _SCRIPTS[name].read_text(encoding="utf-8")
    step = code.split("krepis.aws remove-lambda-env", 1)[1].split("\n\n", 1)[0]
    assert "--defer-publish" in step
    assert "--promote-alias" not in step


@pytest.mark.parametrize("name", sorted(_SCRIPTS))
def test_removal_is_idempotent_across_deploys(name: str) -> None:
    """Every deploy after the first finds the key already gone; without
    --missing-ok the CLI refuses and `set -euo pipefail` aborts the
    deploy."""
    code = _SCRIPTS[name].read_text(encoding="utf-8")
    step = code.split("krepis.aws remove-lambda-env", 1)[1].split("\n\n", 1)[0]
    assert "--missing-ok" in step


def test_krepis_floor_can_supply_the_subcommand() -> None:
    """`remove-lambda-env` ships in krepis 0.59.23. deploy_concordance.sh /
    deploy_counterfactual.sh run on the GHA runner, which installs krepis
    per the floor requirements.txt declares (`krepis[openai]>=X`), NOT the
    per-Lambda Dockerfile pins. An older floor makes the deploy step exit 2
    on an unknown subcommand."""
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    line = next(
        ln
        for ln in req.read_text(encoding="utf-8").splitlines()
        if ln.startswith("krepis[")
    )
    version = line.split(">=", 1)[1].split()[0].strip()
    parts = tuple(int(p) for p in version.split("."))
    assert parts >= (0, 59, 23), (
        f"requirements.txt's krepis floor is {version}; remove-lambda-env "
        "needs >= 0.59.23"
    )


def test_deploy_workflow_runner_install_matches_the_floor() -> None:
    """The GHA runner installs krepis via an explicit `pip install` step
    (requirements.txt is never `pip install -r`'d on the runner for these
    two scripts), so THAT floor — not requirements.txt's — is what
    actually gates whether `remove-lambda-env` exists when the step runs."""
    workflow = (
        Path(__file__).resolve().parents[1] / ".github" / "workflows" / "deploy.yml"
    )
    code = workflow.read_text(encoding="utf-8")
    assert 'pip install "krepis>=0.59.23"' in code, (
        "the runner-side krepis install step must float to at least 0.59.23 "
        "or remove-lambda-env is unavailable when deploy_*.sh calls it"
    )
