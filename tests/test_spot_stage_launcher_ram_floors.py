"""Every per-stage spot launcher makes a DELIBERATE RAM-floor decision.

alpha-engine-config-I7216. On 2026-08-13 `backtest.py --mode param-sweep` was
OOM-killed on a 4 GB c5.large. The launcher that runs it, `spot_backtester.sh`,
carried this:

    # param-sweep does not run predictor_pipeline — stays on the cheap default
    # rotation (no RAM floor needed).

The premise is true and the conclusion does not follow: param-sweep reads the
same ArcticDB feature store over ~900 tickers, and not loading the GBM tensor
does not make it cheap.

**Why this suite exists separately from `test_spot_backtest_ram_floors.py`.**
That one covers `spot_backtest.sh`, the RETIRED monolith kept only as a
rollback path (repointed to per-stage scripts 2026-08-09, config-I4442/I4497).
The Saturday SF invokes the per-stage launchers in this suite. A fix applied to
the monolith alone changes nothing that runs — which is exactly the mistake
this file was written after making.

The assertion is deliberately not "every launcher has a floor". Several run
`evaluate.py` and there is no evidence they need one, and a blanket 16 GB floor
would spend real money on every run to fix a problem nobody has measured. What
is asserted is that each launcher appears in ONE of the two lists below, so a
new stage cannot inherit the cheap 4 GB-first rotation by omission.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
INFRA = REPO_ROOT / "infrastructure"

FLOOR_CALL = "spot_common_apply_predictor_ram_floor"

# Launchers whose workload is known to need >4 GB. `param-sweep` is here
# because production proved it, not because it loads the predictor tensor.
_REQUIRES_FLOOR = {
    "spot_backtester.sh",                     # backtest.py --mode param-sweep (OOM 2026-08-13)
    "spot_predictor_backtest.sh",             # predictor_pipeline
    "spot_portfolio_optimizer_backtest.sh",   # predictor_pipeline
    "spot_parity.sh",
    "spot_pit_lookahead.sh",
    "spot_pit_walkforward.sh",
}

# Launchers deliberately left on the default rotation. Listed explicitly so the
# choice is recorded and reviewable; moving one here is a decision someone made,
# not a line nobody wrote. If any of these is ever OOM-killed, it moves up.
_DELIBERATELY_NO_FLOOR = {
    "spot_evaluator.sh",
    "spot_parity_compare.sh",
    "spot_parity_replay.sh",
    "spot_backtest_and_evaluate.sh",
    "spot_backtest.sh",  # retired monolith; carries its own in-file floor logic
}


def _stage_launchers() -> list[Path]:
    return sorted(p for p in INFRA.glob("spot_*.sh") if p.name != "_spot_common.sh")


def test_every_launcher_is_classified():
    """A new stage script cannot silently inherit the cheap rotation."""
    found = {p.name for p in _stage_launchers()}
    classified = _REQUIRES_FLOOR | _DELIBERATELY_NO_FLOOR
    unclassified = found - classified
    assert not unclassified, (
        f"spot launcher(s) {sorted(unclassified)} are in neither _REQUIRES_FLOOR "
        f"nor _DELIBERATELY_NO_FLOOR. Decide which, and record it — inheriting "
        f"the 4 GB-first default by omission is how param-sweep was OOM-killed "
        f"(alpha-engine-config-I7216)."
    )
    stale = classified - found
    assert not stale, f"classified launcher(s) {sorted(stale)} no longer exist"


@pytest.mark.parametrize("name", sorted(_REQUIRES_FLOOR))
def test_memory_hungry_launchers_apply_the_floor(name):
    text = (INFRA / name).read_text()
    assert FLOOR_CALL in text, (
        f"{name} runs a memory-hungry stage but never calls {FLOOR_CALL}(), so "
        f"it launches on the default 4 GB-first rotation."
    )


def test_the_param_sweep_launcher_applies_the_floor_before_collapsing():
    """Order matters: the floor must be applied before the override collapse.

    `spot_common_collapse_instance_type` honours an explicit --instance-type;
    the floor must run first so the operator override still wins and the floor
    is not silently discarded.
    """
    text = (INFRA / "spot_backtester.sh").read_text()
    floor_at = text.find(FLOOR_CALL)
    collapse_at = text.find("spot_common_collapse_instance_type")
    assert floor_at != -1 and collapse_at != -1
    assert floor_at < collapse_at, (
        "the RAM floor must be applied BEFORE spot_common_collapse_instance_type"
    )


def test_the_falsified_no_floor_claim_is_gone():
    """The exact sentence the defect lived inside must not come back."""
    text = (INFRA / "spot_backtester.sh").read_text()
    assert "no RAM floor needed" not in text, (
        "spot_backtester.sh still claims param-sweep needs no RAM floor — "
        "production falsified that on 2026-08-13 (OOM on a 4 GB c5.large)."
    )


def test_the_default_rotation_is_still_memory_heterogeneous():
    """Pins WHY a floor is required rather than optional.

    The shared default is c5.large,m5.large,c6i.large,c5a.large — 4/8/4/4 GB.
    While that is true, a memory-bound stage without a floor succeeds or OOMs
    depending on which spot capacity pool answers, which is why days of
    failures read as flakiness. If this rotation is ever made homogeneous, this
    test fails and the floors can be revisited.
    """
    common = (INFRA / "_spot_common.sh").read_text()
    m = re.search(r'INSTANCE_TYPES="\$\{INSTANCE_TYPES:-([^}"]+)\}"', common)
    assert m, "could not find the default INSTANCE_TYPES rotation in _spot_common.sh"
    types = [t.strip() for t in m.group(1).split(",")]
    ram = {"c5.large": 4, "c6i.large": 4, "c5a.large": 4, "m5.large": 8}
    sizes = {ram[t] for t in types if t in ram}
    assert len(sizes) > 1, (
        "the default rotation is now memory-homogeneous — revisit whether the "
        "per-stage floors are still the right mechanism"
    )
