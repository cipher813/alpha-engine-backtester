"""Guard: no test file resolves a path outside this repo's root without a
documented, CI-provisioned fallback (alpha-engine-config-I7605 / I7619).

The defect this guards against: a test read a sibling repo's artifact from
a hardcoded sibling checkout path (e.g. ``~/Development/<repo>``), so its
pass/fail depended on which branch/state that checkout happened to be in on
the machine running the suite — not on the published contract or a
controlled CI checkout.

This guard fails on any test file matching the sibling-checkout-path tell
(``Path.home() / "Development" / <repo>``, ``os.path.expanduser("~/Development/
...")``, an ``os.environ["..._DIR"]``-gated variant, or a
``Path(__file__).resolve().parents[N] / "<sibling-repo>"`` variant) that
isn't in the allowlist. Passing plain `pytest` collects and runs it — no
extra infrastructure needed to prove the invariant on a laptop with zero
sibling checkouts present.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
TESTS_DIR = Path(__file__).parent

#: Sanctioned pre-existing sibling-checkout reads. Each entry names why it
#: cannot use the nousergon-lib-contract fix and confirms the file has both
#: a CI env-var override AND a hard-fail-on-CI guard — the two properties
#: that make "sibling checkout" safe rather than "verdict depends on laptop
#: state" (see module docstring).
_ALLOWLIST = {
    "test_parity_replay_atr_contract.py": (
        "imports live executor.ibkr / decide_entries source (crucible-"
        "executor) to reproduce a production ATR-wiring defect exactly — "
        "no JSON-Schema-able contract to lift; the import IS the test."
    ),
    "test_portfolio_optimizer_backtest.py": (
        "TestEndToEndIntegration exercises the real optimizer kernel "
        "against the live crucible-executor checkout — same class as above."
    ),
    "test_vectorized_sim.py": (
        "imports executor.ibkr.SimulatedIBKRClient as the scalar reference "
        "the vectorized simulator must match bit-for-bit — no contract to "
        "lift, the parity target IS the sibling's source."
    ),
    "test_sizing_shootout.py": (
        "puts crucible-executor on sys.path defensively for the sizing "
        "arms under test; centralized resolver, no direct executor import "
        "in this file today but kept consistent with its siblings."
    ),
    "test_simulate_via_deciders.py": (
        "imports executor.ibkr.SimulatedIBKRClient + calls into backtest.py "
        "functions that lazily import executor.deciders — live-source "
        "parity target, not a data contract."
    ),
    "test_vectorized_exits.py": (
        "TestEndToEndParityVsScalarEvaluateExits pins vectorized exits "
        "against the scalar executor.strategies.exit_manager.evaluate_exits "
        "— live-source parity target."
    ),
    "test_vectorized_sweep_dispatch.py": (
        "dispatches into backtest._run_vectorized_param_sweep, which "
        "lazily imports executor.* internals on some paths; centralized "
        "resolver kept for consistency with its siblings."
    ),
    "test_vectorized_sweep.py": (
        "orchestrator-level sibling of test_vectorized_sweep_dispatch.py; "
        "centralized resolver kept for consistency, no contract to lift."
    ),
    "test_vectorized_entries.py": (
        "TestEndToEndParityVsScalarDecideEntries pins vectorized entries "
        "against the scalar executor.deciders.decide_entries — live-source "
        "parity target, not a data contract."
    ),
    "test_param_sweep_decision_capture_suppress.py": (
        "whole-module skip: exercises backtest.py's param-sweep decision-"
        "capture path against the real executor package — live-source "
        "integration, not a data contract."
    ),
    "test_stage_coverage_assertion_wiring.py": (
        "best-effort cross-check of this repo's static SF-wiring list "
        "against nousergon-data's live step_function.json — an SF "
        "definition, not a data contract; mirrors crucible-dashboard's "
        "SF_DEFS_DIR reference fix (alpha-engine-config-I7605)."
    ),
}

_SIBLING_CHECKOUT_TELL = re.compile(
    r'Path\.home\(\)\s*/\s*["\']Development["\']'
    r'|os\.environ\[["\'][A-Z_]*_DIR["\']\]'
    r'|os\.path\.expanduser\(["\']~/Development/'
    r'|parents\[\d+\]\s*/\s*["\'][a-z][a-z0-9_-]*["\']\s*/\s*["\']infrastructure["\']'
)

_THIS_FILE = Path(__file__).name


def _test_files():
    return sorted(
        p for p in TESTS_DIR.glob("test_*.py") if p.is_file() and p.name != _THIS_FILE
    )


def test_no_undocumented_sibling_checkout_path_resolution():
    offenders = []
    for path in _test_files():
        if path.name in _ALLOWLIST:
            continue
        text = path.read_text()
        if _SIBLING_CHECKOUT_TELL.search(text):
            offenders.append(path.name)
    assert not offenders, (
        f"test file(s) resolve a path outside this repo's root with no "
        f"documented CI-provisioned fallback: {offenders}. Either fix at the "
        f"contract layer (nousergon_lib.contracts — preferred whenever the "
        f"sibling repo publishes a JSON Schema), vendor the fixture locally, "
        f"centralize the resolution in tests/_sibling_checkout.py with an "
        f"env-var override + hard-fail-on-CI guard, or add a reviewed entry "
        f"to _ALLOWLIST in this file naming why not."
    )


def test_allowlist_entries_still_exist_and_are_still_safe():
    for name, _reason in _ALLOWLIST.items():
        path = TESTS_DIR / name
        assert path.exists(), f"allowlisted {name} no longer exists — remove its entry"
        text = path.read_text()
        assert "_sibling_checkout" in text, (
            f"{name} is allowlisted as a centrally-resolved sibling checkout "
            f"but no longer imports from tests/_sibling_checkout.py."
        )
