"""
tests/test_simulation_setup_auto_skip.py — integration test for the
simulation_setup auto-skip + artifact load path.

Exercises _save_simulation_setup → _load_simulation_setup round-trip
without actually running the executor. Uses the in-memory S3 fake and
stub executor module so the test stays offline.

Motivated by ROADMAP Backtester P0 "Phase-selective backtest execution
— skip already-successful phases on retry": a retry of a failed dry-run
must rebuild `_sim_setup` from S3 artifacts, not re-invoke price_loader.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from pipeline_common import PhaseRegistry
import backtest as bt
from tests.test_phase_registry import _FakeS3


@pytest.fixture
def s3():
    return _FakeS3()


@pytest.fixture
def fake_executor_module(tmp_path, monkeypatch):
    """Stand up a stub `executor` package so _load_simulation_setup can
    re-import executor.main.run + executor.ibkr.SimulatedIBKRClient
    without the real alpha-engine repo on the path."""
    pkg_dir = tmp_path / "executor"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "main.py").write_text("def run(*a, **kw): return []\n")
    (pkg_dir / "ibkr.py").write_text(
        "class SimulatedIBKRClient:\n"
        "    def __init__(self, **kw): pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    # Clear any previous executor module from cache
    for mod in list(sys.modules):
        if mod.startswith("executor"):
            del sys.modules[mod]
    yield str(tmp_path)
    for mod in list(sys.modules):
        if mod.startswith("executor"):
            del sys.modules[mod]


def _fake_sim_setup(price_matrix: pd.DataFrame, ohlcv: dict, dates: list[str]):
    """Build a _sim_setup tuple with stub callables."""
    executor_run = MagicMock()
    SimulatedIBKRClient = MagicMock()
    return (executor_run, SimulatedIBKRClient, dates, price_matrix, 1_000_000.0, ohlcv)


def test_save_then_load_simulation_setup_roundtrip(s3, fake_executor_module):
    """End-to-end: save on first run, load on retry reconstructs equivalently."""
    dates = ["2026-01-01", "2026-01-02", "2026-01-03"]
    price_matrix = pd.DataFrame(
        {"AAPL": [100.0, 101.0, 102.0], "MSFT": [200.0, 201.0, 202.0]},
        index=pd.to_datetime(dates),
    )
    ohlcv = {
        "AAPL": [
            {"date": "2026-01-01", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},
            {"date": "2026-01-02", "open": 100.5, "high": 102.0, "low": 100.0, "close": 101.5},
        ],
    }

    # First run: save path. Injects the fake s3 client explicitly.
    registry1 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    setup = _fake_sim_setup(price_matrix, ohlcv, dates)
    with registry1.phase("simulation_setup", supports_auto_skip=True) as ctx:
        assert ctx.skipped is False
        bt._save_simulation_setup(
            ctx, "b", "2026-04-23", setup, s3_client=registry1.s3_client,
        )

    # New registry instance (fresh cache, same S3) simulates the retry.
    config = {
        "executor_paths": [fake_executor_module],
        "signals_bucket": "b",
        "init_cash": 1_000_000.0,
    }
    registry2 = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)

    run, reason = registry2.should_run("simulation_setup", supports_auto_skip=True)
    assert run is False
    assert reason == "auto_skip_marker_ok"

    # Full reload via _load_simulation_setup — uses registry.s3_client
    # to find the fake client, so no boto3 monkey-patching is needed.
    reloaded = bt._load_simulation_setup(config, registry2)
    _executor_run, _SimClient, reloaded_dates, reloaded_pm, reloaded_cash, reloaded_ohlcv = reloaded
    assert reloaded_dates == dates
    assert reloaded_cash == 1_000_000.0
    pd.testing.assert_frame_equal(reloaded_pm, price_matrix, check_freq=False)
    assert list(reloaded_ohlcv.keys()) == ["AAPL"]
    assert len(reloaded_ohlcv["AAPL"]) == 2

    marker = registry2.load_marker("simulation_setup")
    assert marker is not None
    keys = marker["artifact_keys"]
    # Three artifacts persisted: price_matrix, ohlcv_by_ticker, dates
    suffixes = sorted(k.rsplit("/", 1)[-1] for k in keys)
    assert suffixes == [
        "dates.json", "ohlcv_by_ticker.parquet", "price_matrix.parquet",
    ]


def test_save_skips_when_price_matrix_none(s3):
    """Degraded state (no price matrix) must not write partial artifacts —
    next retry re-runs setup fresh."""
    registry = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    setup = _fake_sim_setup(None, {}, ["2026-01-01"])
    with registry.phase("simulation_setup", supports_auto_skip=True) as ctx:
        bt._save_simulation_setup(ctx, "b", "2026-04-23", setup)

    # Marker written (status=ok) but with no artifact_keys.
    marker = registry.load_marker("simulation_setup")
    assert marker is not None
    assert marker["status"] == "ok"
    assert marker["artifact_keys"] == []


def test_load_raises_when_marker_missing_artifact(s3, fake_executor_module):
    """If a marker exists but its artifact_keys list doesn't contain the
    expected suffix, loading raises loud — never a silent empty state."""
    # Seed a malformed marker (present, status=ok, empty artifact_keys)
    s3.seed("b", "2026-04-23", "simulation_setup", {
        "phase": "simulation_setup", "date": "2026-04-23",
        "status": "ok", "artifact_keys": [],
    })
    config = {
        "executor_paths": [fake_executor_module],
        "signals_bucket": "b",
        "init_cash": 1_000_000.0,
    }
    registry = PhaseRegistry(date="2026-04-23", bucket="b", s3_client=s3)
    with pytest.raises(RuntimeError, match="price_matrix.parquet"):
        bt._load_simulation_setup(config, registry)
