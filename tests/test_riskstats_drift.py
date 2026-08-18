"""Drift guard: every risk-ratio path in this repo vs nousergon_lib (config-I7597).

`nousergon_lib.quant.riskstats` is the fleet's only implementation of Sharpe,
Sortino and the downside deviation. Most call sites in this repo now call it
directly, so they cannot drift. `synthetic/vectorized_stats.py` is the exception:
it computes both statistics for a whole [n_combos, n_days] matrix at once and a
per-row Python call into the stdlib library would dominate a sweep's runtime, so
it keeps its own vectorized kernel. This file is what stops that kernel — and the
thin adapters around the library — from drifting away from the definition.

CORPUS is kept byte-identical to
`nousergon-lib/tests/test_quant_riskstats_drift_corpus.py`, which pins the
library's own answers against values written out from the definition. If a
library change moves a number, that file fails first and this one says who else
it moved.

Collected by plain `pytest` — deliberately NOT part of `analysis/self_test.py`'s
battery, which only runs inside a report-card cycle.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from nousergon_lib.quant import riskstats

from analysis import factor_blend_sensitivity, risk_ratio_ci
from optimizer import pillar_weight_optimizer
from synthetic import vectorized_stats

# Keep byte-identical to the nousergon-lib copy.
CORPUS: dict[str, list[float]] = {
    "mixed": [0.01, -0.02, 0.015, -0.005, 0.03, -0.01, 0.0, 0.02, -0.03, 0.005],
    "all_positive": [0.01, 0.02, 0.005, 0.03, 0.015],
    "all_negative": [-0.01, -0.02, -0.005, -0.04],
    "all_zero": [0.0, 0.0, 0.0, 0.0, 0.0],
    "zero_vol_positive": [0.01] * 8,
    "zero_vol_negative": [-0.01] * 8,
    "two_obs": [0.01, -0.01],
    "single_obs": [0.02],
    "empty": [],
    "tiny_downside": [0.01, 0.02, 0.03, -1e-9],
}

# Series long enough for the 2-D kernel (which needs >= 2 columns) and for the
# `min_rows`-style gates. Degenerates that are shorter are exercised separately.
_MATRIX_NAMES = [n for n, v in CORPUS.items() if len(v) >= 2]

_TOL = dict(rel=1e-9, abs=1e-12)


# --------------------------------------------------------------------------
# synthetic/vectorized_stats.py — the one kernel that still computes locally
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(_MATRIX_NAMES))
def test_vectorized_sharpe_matches_library(name: str) -> None:
    r = CORPUS[name]
    got = float(vectorized_stats.compute_sharpe_ratio(np.array([r], dtype=np.float64))[0])
    want = riskstats.sharpe_ratio(r)
    # The kernel reports 0.0 where the library reports None (undefined). That
    # sentinel difference is deliberate and load-bearing for the sweep parquet;
    # everything else must agree to 1e-9.
    if want is None:
        assert got == 0.0, f"{name}: kernel must emit its 0.0 sentinel, got {got}"
    else:
        assert got == pytest.approx(want, **_TOL), name


@pytest.mark.parametrize("name", sorted(_MATRIX_NAMES))
def test_vectorized_sortino_matches_library(name: str) -> None:
    r = CORPUS[name]
    got = float(vectorized_stats.compute_sortino_ratio(np.array([r], dtype=np.float64))[0])
    want = riskstats.sortino_ratio(r)
    if want is None:
        assert got == 0.0, f"{name}: kernel must emit its 0.0 sentinel, got {got}"
    else:
        assert got == pytest.approx(want, **_TOL), name


def _ref_sortino_n_down(r: list[float], ppy: int = 252) -> float | None:
    """The RETIRED n_down convention, written out from the definition.

    Deliberately not a library call. config-I7638 deleted the ``"downside"``
    branch from ``nousergon_lib.quant.riskstats`` — asking the library for the
    convention it just removed would make this file's verdict depend on which
    library version happens to be installed, which is exactly the drift this
    file exists to catch. Written out here, the test pins the kernel against
    BOTH conventions on any pinned library.
    """
    if len(r) < 2:
        return None
    mean = sum(r) / len(r)
    short = [x for x in r if x < 0.0]
    if not short:
        return None
    dd = math.sqrt(sum(d * d for d in short) / len(short))
    if not dd:
        return None
    return (mean / dd) * math.sqrt(ppy)


def test_vectorized_kernel_uses_the_full_sample_denominator() -> None:
    """The kernel must be on the config-I7271 n-denominator, not n_down."""
    r = CORPUS["mixed"]
    got = float(vectorized_stats.compute_sortino_ratio(np.array([r], dtype=np.float64))[0])
    n_down_variant = _ref_sortino_n_down(r)
    assert n_down_variant is not None
    assert got != pytest.approx(n_down_variant, **_TOL)
    assert got == pytest.approx(riskstats.sortino_ratio(r), **_TOL)


def test_vectorized_kernels_agree_row_by_row_on_a_stacked_matrix() -> None:
    """The 2-D path must give the same answer as the 1-row path per combo."""
    rows = [CORPUS[n] for n in sorted(_MATRIX_NAMES) if len(CORPUS[n]) == 8]
    assert len(rows) >= 2, "need >= 2 same-length rows to stack"
    m = np.array(rows, dtype=np.float64)
    stacked_sharpe = vectorized_stats.compute_sharpe_ratio(m)
    stacked_sortino = vectorized_stats.compute_sortino_ratio(m)
    for i, row in enumerate(rows):
        one = np.array([row], dtype=np.float64)
        assert stacked_sharpe[i] == pytest.approx(
            float(vectorized_stats.compute_sharpe_ratio(one)[0]), **_TOL
        )
        assert stacked_sortino[i] == pytest.approx(
            float(vectorized_stats.compute_sortino_ratio(one)[0]), **_TOL
        )


# --------------------------------------------------------------------------
# The repointed adapters — pinned so a future edit cannot quietly re-derive
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CORPUS))
def test_bridge_sortino_matches_library(name: str) -> None:
    from vectorbt_bridge import _compute_sortino_ratio

    got = _compute_sortino_ratio(pd.Series(CORPUS[name], dtype="float64"))
    want = riskstats.sortino_ratio(CORPUS[name])
    assert got == pytest.approx(0.0 if want is None else want, **_TOL), name


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_risk_ratio_ci_sharpe_and_ir_match_library(name: str) -> None:
    arr = np.array(CORPUS[name], dtype=np.float64)
    want = riskstats.sharpe_ratio(CORPUS[name])
    for fn in (risk_ratio_ci._sharpe, risk_ratio_ci._information_ratio):
        got = fn(arr)
        if want is None:
            assert got is None, f"{name}/{fn.__name__}: expected None, got {got}"
        else:
            assert got == pytest.approx(want, **_TOL), f"{name}/{fn.__name__}"


@pytest.mark.parametrize("name", sorted(CORPUS))
def test_sortino_call_sites_are_on_the_fleet_denominator(name: str) -> None:
    """Every Sortino call site in this repo is on the config-I7271 n-denominator.

    config-I7618 converted the last three holdouts (``risk_ratio_ci._sortino``,
    ``factor_blend_sensitivity._sortino``, ``pillar_weight_optimizer._sortino``)
    off the retired n_down variant, which divided by the count of
    below-target observations and understated each by sqrt(n / n_down). This
    pins them to ``"full"`` so a silent change of convention fails here.
    """
    r = CORPUS[name]
    ann = riskstats.sortino_ratio(r, denominator="full")
    raw = riskstats.sortino_ratio(r, periods_per_year=1, denominator="full")

    got_ci = risk_ratio_ci._sortino(np.array(r, dtype=np.float64))
    assert (got_ci is None) == (ann is None), name
    if ann is not None:
        assert got_ci == pytest.approx(ann, **_TOL), name

    for fn in (factor_blend_sensitivity._sortino, pillar_weight_optimizer._sortino):
        got = fn(pd.Series(r, dtype="float64"))
        assert (got is None) == (raw is None), f"{name}/{fn.__module__}"
        if raw is not None:
            assert got == pytest.approx(raw, **_TOL), f"{name}/{fn.__module__}"


def test_no_source_file_asks_for_the_downside_denominator() -> None:
    """config-I7618 class guard: the non-standard variant cannot be reintroduced.

    A per-call-site numeric test only catches the sites it already knows about;
    this catches the NEXT one. ``tests/`` is no longer exempt: config-I7638
    rewrote ``test_the_two_conventions_really_do_differ`` to write the retired
    convention out from the definition, so nothing in this repo needs to name
    the variant in a live call any more.

    The library no longer HAS the branch either — config-I7638 converted
    ``nousergon_lib.quant.stats.regime_sortino``, the last two call sites, and
    deleted it, so a call site anywhere in the fleet now raises ValueError
    rather than silently computing the retired convention. This guard is the
    repo-local half: it fails in review rather than at runtime.
    """
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    skip_dirs = {".git", ".venv", "venv", "build", "dist",
                 "__pycache__", "node_modules", ".worktrees"}
    # Spelled in pieces so this detector is not its own first offender.
    retired = "down" + "side"
    needles = (f'denominator="{retired}"', f"denominator='{retired}'")
    offenders = []
    for path in repo_root.rglob("*.py"):
        if any(part in skip_dirs for part in path.relative_to(repo_root).parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue  # a comment may explain the variant it does not use
            if any(needle in line for needle in needles):
                offenders.append(f"{path.relative_to(repo_root)}:{i}")
    assert not offenders, (
        f'denominator="{retired}" is the retired n_down convention '
        "config-I7271 ruled against, config-I7618 removed from this repo and "
        'config-I7638 deleted from the library. Use denominator="full". '
        "Offenders: " + ", ".join(offenders)
    )


def test_the_two_conventions_really_do_differ() -> None:
    """Guard against the variants collapsing into each other and hiding drift.

    The retired side comes from :func:`_ref_sortino_n_down`, not the library —
    see the note there. What this pins is that the number this repo publishes is
    the FULL-sample one and is sqrt(n / n_down) away from what it used to be; it
    keeps saying that on a library where the retired branch no longer exists.
    """
    r = CORPUS["mixed"]
    n, n_down = len(r), sum(1 for x in r if x < 0)
    full = riskstats.sortino_ratio(r)
    down = _ref_sortino_n_down(r)
    assert full is not None and down is not None
    assert full / down == pytest.approx(math.sqrt(n / n_down), rel=1e-12)
