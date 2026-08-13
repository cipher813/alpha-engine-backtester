"""Every per-stage spot launcher makes a DELIBERATE executor risk.yaml
freshness decision, and the resolver actually refreshes it.

alpha-engine-config-I7247. `spot_common_resolve_executor_config()` resolves
the executor's `risk.yaml` by searching paths directly under
`$HOME/alpha-engine-config/...` on the persistent weekly-SF dispatch box.
Unlike `predictor.yaml` (a DIFFERENT repo, staged in by an explicit `cp` —
config-I7216, crucible-backtester-PR657, mirrored in
crucible-predictor-PR487), risk.yaml is read straight out of the
`alpha-engine-config` checkout: its presence never depended on any SF
state's copy action, only on the checkout existing.

That checkout is refreshed (`git pull --ff-only`) as a SIDE EFFECT of two
independently-skippable early SF states (MorningEnrich/skip_morning_enrich,
DataPhase1/skip_data_phase1), plus incidentally PredictorTraining/
ModelZooSelect. A mechanical rerun that sets all three skip flags — recovery
from a failure downstream of them, e.g. a Backtester-family failure — never
refreshes the checkout for that execution. `spot_common_resolve_executor_config`
still finds a file (the checkout is standing infrastructure, not created
per-run), so nothing fails loud; the Backtester/Evaluator/Parity family
stages then run against whatever risk.yaml content was on disk from a
previous week's pull. This is a STALENESS exposure, not the missing-file
class PR657 fixed for predictor.yaml — no live failure was observed, only a
silent-degrade window.

The fix mirrors PR657's shape: the guarantee moves to the layer that
DECLARES the requirement (the resolver itself) rather than depending on
three unrelated upstream skip flags. Since risk.yaml is read directly out of
a standing checkout rather than copied in from one, "self-staging" here
means "self-refreshing": `spot_common_resolve_executor_config` now runs
`git -C <config_root> pull --ff-only` against every checkout candidate,
unconditionally, before searching it for risk.yaml.

This suite pins two things: (1) every launcher makes an explicit, reviewable
choice about whether it needs the resolver, so a new stage cannot silently
inherit the staleness gap by omission (mirrors
`tests/test_spot_stage_launcher_ram_floors.py` and crucible-predictor's
`tests/test_spot_stage_launcher_config_staging.py`); and (2) the resolver's
refresh actually works, verified BY EXECUTION against a real git checkout —
not by reading the source and assuming.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INFRA = REPO_ROOT / "infrastructure"
COMMON = INFRA / "_spot_common.sh"

RESOLVER_CALL = "spot_common_resolve_executor_config"

# Launchers that must resolve risk.yaml through the (now self-refreshing)
# executor config resolver.
_REQUIRES_RESOLVER = {
    "spot_backtester.sh",
    "spot_evaluator.sh",
    "spot_parity.sh",
    "spot_parity_compare.sh",
    "spot_parity_replay.sh",
    "spot_pit_lookahead.sh",
    "spot_pit_walkforward.sh",
    "spot_portfolio_optimizer_backtest.sh",
    "spot_predictor_backtest.sh",
}

# Launchers deliberately left off the resolver. Listed explicitly so the
# choice is recorded and reviewable — moving one here is a decision someone
# made, not a line nobody wrote.
_DELIBERATELY_NOT_RESOLVER = {
    # Retired monolith (config-I4442/I4497, per-stage cutover 2026-08-09) —
    # kept unchanged as the rollback path; carries its own historical config
    # resolution and is out of scope for per-stage-launcher hardening.
    "spot_backtest.sh",
    # Convenience wrapper that shells out to spot_backtest.sh (the monolith)
    # for its backtest leg and to the always-on ae-data instance for its
    # evaluator leg — never calls the per-stage resolver itself.
    "spot_backtest_and_evaluate.sh",
}


def _stage_launchers() -> list[Path]:
    return sorted(p for p in INFRA.glob("spot_*.sh") if p.name != "_spot_common.sh")


def test_every_launcher_is_classified():
    """A new stage script cannot silently inherit the staleness gap."""
    found = {p.name for p in _stage_launchers()}
    classified = _REQUIRES_RESOLVER | _DELIBERATELY_NOT_RESOLVER
    unclassified = found - classified
    assert not unclassified, (
        f"spot launcher(s) {sorted(unclassified)} are in neither "
        f"_REQUIRES_RESOLVER nor _DELIBERATELY_NOT_RESOLVER. Decide which, "
        f"and record it — inheriting a stale risk.yaml by omission is "
        f"exactly the gap alpha-engine-config-I7247 closed."
    )
    stale = classified - found
    assert not stale, f"classified launcher(s) {sorted(stale)} no longer exist"


@pytest.mark.parametrize("name", sorted(_REQUIRES_RESOLVER))
def test_executor_config_launchers_call_the_resolver(name):
    text = (INFRA / name).read_text()
    assert RESOLVER_CALL in text, (
        f"{name} must resolve executor risk.yaml via {RESOLVER_CALL}(), the "
        f"chokepoint that now self-refreshes the alpha-engine-config "
        f"checkout before searching it (alpha-engine-config-I7247)."
    )


def test_the_resolver_is_defined_exactly_once():
    text = COMMON.read_text()
    assert text.count(f"{RESOLVER_CALL}() {{") == 1, (
        f"{RESOLVER_CALL}() must be defined exactly once in _spot_common.sh "
        "— a second definition is a fork of the invariant, not a second "
        "copy of it."
    )


def _resolver_body() -> str:
    text = COMMON.read_text()
    start = text.index(f"{RESOLVER_CALL}() {{")
    # Function ends at the first line consisting of just "}" after start.
    end = text.index("\n}\n", start)
    return text[start:end]


def test_the_resolver_refreshes_every_checkout_candidate_before_searching():
    """Static pin: the refresh loop runs BEFORE the candidate search loop,
    and covers both checkout locations the search loop itself checks
    (~/alpha-engine-config and ~/Development/alpha-engine-config)."""
    body = _resolver_body()
    refresh_at = body.find("git pull --ff-only")
    search_at = body.find('for candidate in \\')
    assert refresh_at != -1, (
        f"{RESOLVER_CALL} no longer runs `git pull --ff-only` — the "
        "self-refresh this issue added is gone (alpha-engine-config-I7247)."
    )
    assert search_at != -1, "could not find the candidate search loop"
    assert refresh_at < search_at, (
        "the checkout refresh must run BEFORE the candidate search loop, or "
        "the search can still see pre-refresh (stale) content."
    )
    assert '"$HOME/alpha-engine-config"' in body
    assert '"$HOME/Development/alpha-engine-config"' in body


def test_the_refresh_is_best_effort_never_a_hard_fail():
    """A failed `git pull` (offline runner, detached HEAD, local edits) must
    not itself abort the stage — the existing checkout, however stale, is
    still the best available input. Only a genuinely missing risk.yaml is a
    hard fail (the pre-existing `exit 1` below the search loop)."""
    body = _resolver_body()
    assert "if ! git -C" in body, (
        "the pull must be guarded — an unguarded `git ... pull` under "
        "`set -e` would turn an offline dispatch box into a hard failure "
        "for every stage, which is a worse regression than the staleness "
        "bug this closes."
    )
    assert "exit 1" not in body.split("git pull --ff-only")[0].split(
        "for config_root"
    )[-1], "the refresh loop itself must never call exit"


def _run_resolver_against_home(home: Path) -> subprocess.CompletedProcess:
    script = textwrap.dedent(
        f"""
        set -u
        HOME="{home}"
        source "{COMMON}"
        spot_common_resolve_executor_config
        echo "RESOLVED_PATH=$EXECUTOR_CONFIG"
        cat "$EXECUTOR_CONFIG"
        """
    )
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )


def _init_origin_and_clone(tmp_path: Path, content: str) -> tuple[Path, Path]:
    origin = tmp_path / "origin_config"
    home = tmp_path / "home"
    home.mkdir()
    risk_dir = origin / "experiments" / "reference" / "executor"
    risk_dir.mkdir(parents=True)
    (risk_dir / "risk.yaml").write_text(content)

    def git(*args, cwd=origin):
        subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("add", "-A")
    git(
        "-c", "user.email=test@test.invalid",
        "-c", "user.name=Test",
        "commit", "-q", "-m", "initial",
    )
    subprocess.run(
        ["git", "clone", "-q", str(origin), str(home / "alpha-engine-config")],
        check=True,
        capture_output=True,
    )
    return origin, home


def test_the_resolver_serves_fresh_content_after_origin_advances(tmp_path):
    """The regression itself, verified BY EXECUTION against a real git
    checkout — not by reading the source and assuming.

    Reproduces the bug's exact shape: a dispatch-box checkout that is behind
    origin (standing in for every-config-refreshing-state-skipped), then
    asserts risk.yaml resolves to the CURRENT content, not the stale clone
    content. Against unfixed origin/main this fails: the second call still
    prints "note: v1" because nothing in the resolver ever runs `git pull`.
    """
    origin, home = _init_origin_and_clone(tmp_path, "note: v1\n")

    first = _run_resolver_against_home(home)
    assert first.returncode == 0, first.stderr
    assert "note: v1" in first.stdout

    # Advance origin — simulating the upstream repo moving while the
    # dispatch-box checkout stays where it was (every config-refreshing SF
    # state skipped in this rerun).
    (origin / "experiments" / "reference" / "executor" / "risk.yaml").write_text(
        "note: v2\n"
    )
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git", "-C", str(origin),
            "-c", "user.email=test@test.invalid",
            "-c", "user.name=Test",
            "commit", "-q", "-m", "v2",
        ],
        check=True,
        capture_output=True,
    )

    second = _run_resolver_against_home(home)
    assert second.returncode == 0, second.stderr
    assert "note: v2" in second.stdout, (
        "spot_common_resolve_executor_config served STALE content after "
        "origin advanced — the checkout was never refreshed "
        "(alpha-engine-config-I7247 regression)."
    )
    assert "note: v1" not in second.stdout


def test_the_resolver_survives_an_unreachable_origin(tmp_path):
    """A `git pull` failure (origin gone, network down) must not turn into a
    hard failure — the stage must still resolve the existing checkout."""
    origin, home = _init_origin_and_clone(tmp_path, "note: v1\n")
    # Make the origin remote unreachable without touching the clone itself.
    import shutil

    shutil.rmtree(origin)

    result = _run_resolver_against_home(home)
    assert result.returncode == 0, (
        "resolution must survive a failed refresh (offline dispatch box) — "
        f"got rc={result.returncode}, stderr={result.stderr}"
    )
    assert "note: v1" in result.stdout
    assert "config-I7247" in result.stderr
