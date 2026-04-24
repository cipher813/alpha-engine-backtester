"""Add project root to sys.path so optimizer.* and analysis.* imports work,
plus centralized arcticdb stubbing for unit tests."""
import sys
import os
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub arcticdb by default for all unit tests — they must never hit real S3,
# and CI (GitHub Actions) has no AWS credentials, so real arcticdb calls
# would 403 (observed 2026-04-24 CI on PR #76). Integration tests that need
# the real module (parity replay on the spot) set USE_REAL_ARCTICDB=1 before
# invoking pytest; spot_backtest.sh's parity stage passes this env var.
#
# History: this lived as an unconditional sys.modules.setdefault inside
# test_parity_replay.py — which ran at module-import time and silently
# shadowed the real arcticdb for the parity integration test itself,
# producing a false-positive "ArcticDB universe library returned 0 symbols"
# failure (MagicMock.list_libraries() iterates to []). Moving it here
# centralizes the stub and lets integration tests opt in.
if not os.environ.get("USE_REAL_ARCTICDB"):
    sys.modules.setdefault("arcticdb", MagicMock())
