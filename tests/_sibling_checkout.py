"""Single source of truth for this repo's sibling-checkout reads
(alpha-engine-config-I7605 / I7619).

Both are genuinely environment-dependent: they walk another repo's live
source/artifacts with no JSON-Schema-able contract to lift (crucible-
executor's ``executor.*`` package is Python source, and nousergon-data's
``step_function.json`` is an SF definition, not a data contract). The fix is
NOT to remove the sibling read — it is to make its absence loud on CI (a
missing checkout there is a broken guard, not a valid state) and quiet only
on a dev laptop that legitimately hasn't cloned the sibling.

Centralizing the executor resolution (rather than each test file rolling its
own ``os.path.expanduser("~/Development/alpha-engine")`` + ad hoc skipif)
fixes two defects at once:

1. Ten near-identical inline definitions of ``_EXECUTOR_ROOT`` — one to
   maintain instead of ten to keep in sync.
2. An import-order bug: several files did their ``sys.path.insert`` at
   MODULE level, so whether ``executor`` was importable when
   ``test_actionable_signals_parity.py`` ran its own
   ``from executor.signal_reader import get_actionable_signals`` depended on
   pytest's alphabetical collection order — files starting with a letter
   after ``a`` inserted too late to help it. This module is imported from
   conftest.py, which loads before every test module, so the order bug
   can't recur.
"""

from __future__ import annotations

import os
import sys

import pytest

_ON_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}


def resolve_executor_root() -> str:
    """Resolve the crucible-executor sibling checkout that
    ``executor.ibkr``, ``executor.decide_entries``, ``executor.signal_reader``
    etc. are imported from.

    CI checks out crucible-executor (sparse) and sets ``EXECUTOR_ROOT_DIR``;
    a dev laptop uses the conventional ``~/Development/alpha-engine`` sibling
    clone (a laptop naming convention for the checkout directory — the repo
    itself is ``crucible-executor``).
    """
    env_dir = os.environ.get("EXECUTOR_ROOT_DIR")
    return env_dir if env_dir else os.path.expanduser("~/Development/alpha-engine")


def ensure_executor_on_sys_path() -> str:
    """Resolve the executor root and, if present, put it on sys.path.
    Returns the resolved root regardless of whether it exists, so callers
    can still build a skip/fail condition off it."""
    root = resolve_executor_root()
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
    return root


def executor_root_missing_hard_fail_on_ci(root: str) -> bool:
    """Call at module level right after computing ``root``. Returns True if
    the checkout is missing — existing ``@pytest.mark.skipif(<this>, ...)``
    sites reuse the return value, so per-test skip granularity on a dev
    laptop is unchanged from before this fix.

    On CI, a missing checkout is a broken guard, not a valid state: this
    hard-fails collection immediately instead of returning True and letting
    the file silently skip forever, indistinguishable from a passing suite.
    """
    missing = not os.path.isdir(root)
    if missing and _ON_CI:
        pytest.fail(
            f"{root} not present. CI checks out crucible-executor and sets "
            "EXECUTOR_ROOT_DIR; a dev laptop uses ~/Development/alpha-engine. "
            "On CI this is a broken guard, not an absent layout — skipping "
            "here would report a cross-repo invariant as satisfied without "
            "ever evaluating it.",
            pytrace=False,
        )
    return missing


def executor_missing_reason(root: str) -> str:
    """Reason string for the dev-laptop-only skip path (CI already hard-
    failed above before any skipif using this reason is even reached)."""
    return f"crucible-executor sibling checkout not present at {root} (see EXECUTOR_ROOT_DIR)"


def resolve_sf_defs_dir() -> str:
    """Resolve the nousergon-data sibling checkout that
    ``infrastructure/step_function.json`` (the live weekly SF definition) is
    read from. Mirrors crucible-dashboard's reference fix (alpha-engine-
    config-I7605): CI checks out nousergon-data (sparse) and sets
    ``SF_DEFS_DIR``; a dev laptop uses ``~/Development/nousergon-data``.
    """
    env_dir = os.environ.get("SF_DEFS_DIR")
    return env_dir if env_dir else os.path.expanduser("~/Development/nousergon-data")


def is_ci() -> bool:
    return _ON_CI
