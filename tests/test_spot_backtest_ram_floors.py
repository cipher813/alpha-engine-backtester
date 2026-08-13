"""Every memory-hungry backtest mode declares an instance-type RAM floor.

alpha-engine-config-I7216. On 2026-08-13 `--mode param-sweep` was OOM-killed
on a 4 GB c5.large:

    bash: line 16: 26748 Killed  python -u backtest.py --mode param-sweep ...

`param-sweep` had been deliberately carved OUT of the existing predictor RAM
floor, on the stated assumption that it "doesn't load the predictor tensor and
stays on the cheap 4 GB-first rotation".

Two properties this suite pins, because the failure was invisible in both:

1. **Every mode that needs memory has a floor.** The predictor floor was added
   after a 2026-06-01 OOM and covered only the modes that OOMed *then*; the
   next mode to grow past 4 GB inherited the cheap rotation silently.
2. **A floor's types are memory-HOMOGENEOUS.** The default rotation is
   `c5.large,m5.large,c6i.large,c5a.large` — 4 GB, 8 GB, 4 GB, 4 GB. A
   memory-bound job on that rotation succeeds or OOMs depending on which spot
   capacity pool answers, which is why this read as flakiness for days rather
   than as a memory fault.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "infrastructure" / "spot_backtest.sh"

# Instance RAM in GB for every type this launcher may select. Kept explicit
# rather than parsed from AWS so the assertion is hermetic and reviewable.
_TYPE_RAM_GB = {
    "c5.large": 4, "c6i.large": 4, "c5a.large": 4,
    "m5.large": 8, "m6i.large": 8, "m5a.large": 8, "m6a.large": 8,
    "t3.large": 8,
    "m5.xlarge": 16, "m6i.xlarge": 16, "m5a.xlarge": 16,
    "c5.2xlarge": 16, "c6i.2xlarge": 16,
    "r5.large": 16,
}

# Modes known to need more than the cheap rotation, and the floor each must
# clear. Adding a mode here without giving it a floor in the launcher fails.
_MODES_REQUIRING_A_FLOOR = {
    "all": 16,
    "predictor-backtest": 16,
    "portfolio-optimizer-backtest": 16,
    "param-sweep": 8,
    "simulate": 8,
    "signal-quality": 8,
}


def _launcher_text() -> str:
    return LAUNCHER.read_text()


def _floor_vars() -> dict[str, list[str]]:
    """Extract every `_*_RAM_FLOOR_TYPES=...` assignment, resolved to types.

    One level of variable indirection is resolved, because a floor that
    deliberately reuses another tier (`_A="$_B"`) is a legitimate and clearer
    way to say "the same tier", and the assertions below must see the real
    instance types rather than the reference.
    """
    raw: dict[str, str] = {}
    for m in re.finditer(r'(_[A-Z_]*RAM_FLOOR_TYPES)="([^"]+)"', _launcher_text()):
        raw[m.group(1)] = m.group(2)

    out: dict[str, list[str]] = {}
    for name, value in raw.items():
        ref = re.fullmatch(r"\$\{?(_[A-Z_]*RAM_FLOOR_TYPES)\}?", value.strip())
        if ref:
            assert ref.group(1) in raw, (
                f"{name} references {ref.group(1)}, which is not defined in the launcher"
            )
            value = raw[ref.group(1)]
        out[name] = [t.strip() for t in value.split(",") if t.strip()]
    return out


def _case_arms() -> dict[str, str]:
    """Map each mode named in the BACKTEST_MODE case to its floor variable."""
    text = _launcher_text()
    arms: dict[str, str] = {}
    block = re.search(r'case "\$BACKTEST_MODE" in(.+?)\nesac', text, re.S)
    assert block, "BACKTEST_MODE case block not found — did the launcher restructure?"
    for arm in re.finditer(
        r'^\s{4}([a-z0-9|\-]+)\)(.*?);;', block.group(1), re.S | re.M
    ):
        modes, body = arm.group(1), arm.group(2)
        var = re.search(r'INSTANCE_TYPES="\$(_[A-Z_]*RAM_FLOOR_TYPES)"', body)
        if not var:
            continue
        for mode in modes.split("|"):
            arms[mode] = var.group(1)
    return arms


class TestEveryHungryModeHasAFloor:
    @pytest.mark.parametrize(
        ("mode", "min_gb"), sorted(_MODES_REQUIRING_A_FLOOR.items())
    )
    def test_mode_selects_a_floor_meeting_its_minimum(self, mode, min_gb):
        arms = _case_arms()
        assert mode in arms, (
            f"mode {mode!r} selects no RAM floor — it inherits the default "
            f"4 GB-first rotation. That is exactly how param-sweep was "
            f"OOM-killed on 2026-08-13 (alpha-engine-config-I7216)."
        )
        types = _floor_vars()[arms[mode]]
        unknown = [t for t in types if t not in _TYPE_RAM_GB]
        assert not unknown, f"unknown instance type(s) {unknown} — add RAM to _TYPE_RAM_GB"
        worst = min(_TYPE_RAM_GB[t] for t in types)
        assert worst >= min_gb, (
            f"mode {mode!r} floor {arms[mode]} can land on {worst} GB "
            f"(needs ≥{min_gb} GB): {types}"
        )


class TestFloorsAreMemoryHomogeneous:
    @pytest.mark.parametrize("var", sorted(_floor_vars()))
    def test_a_floor_never_mixes_ram_sizes(self, var):
        """The defect's real shape.

        A floor whose members differ in RAM makes the effective memory budget
        a function of spot capacity, so the same job intermittently OOMs and
        reads as flakiness rather than as a memory fault.
        """
        types = _floor_vars()[var]
        sizes = {_TYPE_RAM_GB[t] for t in types if t in _TYPE_RAM_GB}
        assert len(sizes) == 1, (
            f"{var} mixes RAM sizes {sorted(sizes)}: {types}. A memory-bound "
            f"job on a mixed rotation succeeds or dies by capacity lottery."
        )

    def test_a_floor_keeps_more_than_one_capacity_pool(self):
        # The rotation exists for InsufficientInstanceCapacity resilience
        # (2026-05-22). A floor that collapses to one type trades an OOM for
        # a capacity stall.
        for var, types in _floor_vars().items():
            assert len(types) >= 2, f"{var} has only {types} — no capacity fallback"


class TestOperatorOverrideStillWins:
    def test_every_floor_arm_is_guarded_on_an_empty_instance_type(self):
        # An explicit --instance-type is the operator's choice, including a
        # deliberately small debug box; a floor must not override it.
        block = re.search(
            r'case "\$BACKTEST_MODE" in(.+?)\nesac', _launcher_text(), re.S
        ).group(1)
        for arm in re.finditer(r'INSTANCE_TYPES="\$_[A-Z_]*RAM_FLOOR_TYPES"', block):
            preceding = block[: arm.start()]
            assert '-z "$INSTANCE_TYPE"' in preceding.rsplit(";;", 1)[-1], (
                "a RAM floor is applied without checking that the operator "
                "left --instance-type unset"
            )
