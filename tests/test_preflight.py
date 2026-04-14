"""
Tests for BacktesterPreflight mode composition.

The BasePreflight primitives (check_env_vars / check_s3_bucket /
check_arcticdb_fresh) are tested in alpha-engine-lib. These tests
only verify that each mode calls the expected primitives in the
expected order.
"""

from __future__ import annotations

import sys
import os
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preflight import BacktesterPreflight


class TestBacktesterPreflight:
    def test_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="unknown mode"):
            BacktesterPreflight(bucket="b", mode="bogus")

    def test_backtest_mode_checks_arcticdb(self):
        pf = BacktesterPreflight(bucket="b", mode="backtest")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_called_once_with("macro", "SPY", max_stale_days=8)

    def test_evaluate_mode_skips_arcticdb(self):
        pf = BacktesterPreflight(bucket="b", mode="evaluate")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_not_called()

    def test_lambda_health_mode_skips_arcticdb(self):
        pf = BacktesterPreflight(bucket="b", mode="lambda_health")
        with patch.object(pf, "check_env_vars") as env, \
             patch.object(pf, "check_s3_bucket") as s3, \
             patch.object(pf, "check_arcticdb_fresh") as adb:
            pf.run()
        env.assert_called_once_with("AWS_REGION")
        s3.assert_called_once()
        adb.assert_not_called()
