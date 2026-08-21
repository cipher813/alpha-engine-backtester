"""No live call site in this repo constructs its own provider client.

alpha-engine-config-I7878, under Brian's 2026-08-03 ruling that no agent
may be directly linked to OpenRouter (alpha-engine-config-I6367).

This is deliberately NOT a duplicate of the `OpenRouter direct-linkage
guard` workflow. That guard greps for OpenRouter literals and is satisfied
by an allowlist entry; this asserts the POSITIVE property the migration
bought — that the replay/concordance path reaches its model through
`krepis.router` and holds no provider name, no base URL and no provider
credential of its own (model-router-policy §2 layer 5, principles §2.8).

A grep guard cannot see the difference between "migrated" and "the literal
was renamed". This can.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]

#: Every module that reaches a model. Adding one without adding it here is
#: the gap that let this repo carry an unmigrated call site for 17 days
#: after the ruling — it was invisible because nothing enumerated the set.
_LLM_CALL_SITES = (
    "replay/runner.py",
    "replay/batch.py",
    "lambda_concordance/handler.py",
)

#: Provider hosts and credential names are ASSEMBLED, never written out. This
#: file is scanned by the fleet's `OpenRouter direct-linkage guard`, and a
#: test asserting a literal's ABSENCE is indistinguishable, to a grep, from a
#: call site using it. Spelling them here would earn an allowlist entry for
#: the very test that exists to prove no entry is needed.
_FORBIDDEN_HOSTS = tuple(
    f"{vendor}.{tld}" for vendor, tld in (
        ("openrouter", "ai"),
        ("api.deepseek", "com"),
        ("api.anthropic", "com"),
    )
)

_FORBIDDEN_CREDENTIALS = tuple(
    f"{vendor}_API_KEY" for vendor in ("OPENROUTER", "ANTHROPIC", "DEEPSEEK")
)


@pytest.mark.parametrize("relpath", _LLM_CALL_SITES)
def test_no_provider_endpoint_literal(relpath):
    src = (_REPO / relpath).read_text(encoding="utf-8")
    for needle in _FORBIDDEN_HOSTS:
        assert needle not in src, (
            f"{relpath} names the provider endpoint {needle!r}. The endpoint "
            f"is a registry fact resolved by krepis.router; a copy here is a "
            f"routing table at layer 5."
        )


@pytest.mark.parametrize("relpath", _LLM_CALL_SITES)
def test_no_modelspec_is_constructed_at_the_call_site(relpath):
    """`ModelSpec(provider=..., ...)` built here is the exact pre-migration
    shape. The spec must come back from the router, which is what makes the
    provider choice a configuration fact rather than a code fact."""
    tree = ast.parse((_REPO / relpath).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "ModelSpec", (
                f"{relpath} constructs a ModelSpec directly. Use "
                f"krepis.router.resolve_model_spec / resolve_group_spec — "
                f"see alpha-engine-config-I7878."
            )


@pytest.mark.parametrize("relpath", _LLM_CALL_SITES)
def test_no_provider_credential_is_resolved(relpath):
    """A provider key resolved here would be a credential this process has
    no legitimate use for — the router edge authenticates it with its OWN
    per-consumer credential (ROUTER_CONSUMER_REPLAY on the Lambda). An
    unusable key present in a process is a standing liability with no
    upside (model-router-policy R25)."""
    src = (_REPO / relpath).read_text(encoding="utf-8")
    for needle in _FORBIDDEN_CREDENTIALS:
        assert needle not in src, (
            f"{relpath} still references {needle}. After I7878 no provider "
            f"credential belongs on this path."
        )


def test_the_target_model_reaches_the_router_and_nothing_else(
    real_router_resolution,
):
    """The one positive assertion: `resolve_target_spec` delegates to
    `krepis.router.resolve_model_spec` and passes only the caller's
    execution context and wire format — the two things a consumer is
    allowed to say about routing (R29).

    Takes ``real_router_resolution`` so conftest's autouse stub does not
    replace the function whose source this reads."""
    import inspect

    from replay import runner

    src = inspect.getsource(runner.resolve_target_spec)
    assert "resolve_model_spec" in src
    assert "exec_context=REPLAY_EXEC_CONTEXT" in src
    assert 'wire="openai"' in src
    for forbidden in ("base_url", "api_key", "provider="):
        assert forbidden not in src, (
            f"resolve_target_spec passes {forbidden!r} — a consumer declares "
            f"where it runs and what wire it speaks, and nothing else about "
            f"routing (model-router-policy §2 layer 5)."
        )
