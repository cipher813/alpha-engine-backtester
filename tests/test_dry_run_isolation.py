"""
tests/test_dry_run_isolation.py — --dry-run safety bundle.

Verifies that ``_apply_dry_run_isolation`` applies all four guards
that isolate a dry-run from production S3 state:

  - args.date prefixed with ``.dry-run/`` (mirror of smoke's ``.smoke/``)
  - args.freeze = True (no optimizer config writes)
  - args.upload = False (no reporter upload)
  - args.force = True (no auto-skip)

Motivated by the 2026-04-24 ask: manual spot runs for ad-hoc
validation must not contaminate phase markers written by the
scheduled Saturday SF on the same calendar date.
"""

from __future__ import annotations

from types import SimpleNamespace

from backtest import _apply_dry_run_isolation


def _fresh_args() -> SimpleNamespace:
    """Baseline args namespace as if argparse had populated defaults
    and the operator had passed no explicit flags."""
    return SimpleNamespace(
        mode="all",
        date="2026-04-24",
        dry_run=True,
        freeze=False,
        upload=True,
        force=False,
        skip_phases="",
        only_phases="",
        force_phases="",
    )


def test_dry_run_prefixes_date_with_dot_dry_run():
    args = _fresh_args()
    _apply_dry_run_isolation(args)
    assert args.date == ".dry-run/2026-04-24"


def test_dry_run_sets_freeze_true():
    args = _fresh_args()
    _apply_dry_run_isolation(args)
    assert args.freeze is True


def test_dry_run_sets_upload_false():
    args = _fresh_args()
    _apply_dry_run_isolation(args)
    assert args.upload is False


def test_dry_run_sets_force_true():
    args = _fresh_args()
    _apply_dry_run_isolation(args)
    assert args.force is True


def test_dry_run_mode_preserved():
    """Dry-run does NOT change the backtest mode — the operator keeps
    their chosen --mode (all, simulate, param-sweep, etc.)."""
    args = _fresh_args()
    args.mode = "simulate"
    _apply_dry_run_isolation(args)
    assert args.mode == "simulate"


def test_dry_run_and_smoke_namespaces_are_distinct():
    """A .dry-run/ prefix must never collide with .smoke/ — both
    hierarchically isolate from prod on the same calendar date."""
    dry_args = _fresh_args()
    _apply_dry_run_isolation(dry_args)
    # The smoke equivalent (emulated — we don't call _apply_smoke_fixture
    # here because it needs a full config dict, but we know its output
    # from backtest.py:1723 is args.date = f".smoke/{args.date}").
    assert dry_args.date.startswith(".dry-run/")
    assert not dry_args.date.startswith(".smoke/")
    assert not dry_args.date.startswith("2026-")


def test_dry_run_preserves_phase_selection_flags():
    """--dry-run must not clobber --only-phases or --skip-phases —
    operators may want to dry-run a single phase."""
    args = _fresh_args()
    args.only_phases = "phase4a_ensemble_modes"
    args.skip_phases = ""
    _apply_dry_run_isolation(args)
    assert args.only_phases == "phase4a_ensemble_modes"
    assert args.skip_phases == ""
