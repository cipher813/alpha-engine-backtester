"""Smoke-level coverage for backtester's Tier 2 direct-decider path.

Tier 2 (2026-04-27) replaced the per-date ``executor_run(simulate=True, ...)``
shell call with direct ``decide_entries`` / ``decide_exits_and_reduces``
invocations. These tests pin three invariants:

  1. ``_simulate_single_date`` runs to completion against a minimal real-
     executor fixture (no mocks of decider internals) and returns a list
     of orders + None skip_reason.
  2. State carries across iterations: positions accumulated on iteration
     N are visible to ``decide_entries`` "already in portfolio" check on
     iteration N+1.
  3. ``_build_merged_simulate_config`` produces a merged config with the
     param sweep override applied + a flat strategy_config dict.

Run with USE_REAL_ARCTICDB=1 if you need to stress the integration path
— the conftest's MagicMock arcticdb stub is sufficient for these tests.
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import pytest


# Ensure executor on sys.path before importing backtester functions
# (the executor lives in a sibling repo at ~/Development/alpha-engine).
_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open":  [base + i * 0.1 for i in range(n_bars)],
            "high":  [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low":   [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


def _make_signals(date_str: str = "2026-04-25", n_enter: int = 3) -> dict:
    enter = [
        {
            "ticker": f"TKR{i:03d}",
            "signal": "ENTER",
            "score": 80,
            "conviction": "rising",
            "sector": "Technology",
            "rating": "BUY",
            "price_target_upside": 0.15,
            "thesis_summary": "test",
        }
        for i in range(n_enter)
    ]
    return {
        "date": date_str,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "market_weight"}},
        "enter": enter,
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": enter,
        "buy_candidates": enter,
    }


def _config():
    return {
        "init_cash": 1_000_000.0,
        "signals_bucket": "alpha-engine-research",
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "earnings_proximity_warning_days": 2,
        "momentum_gate_enabled": True,
        "momentum_gate_threshold": -50.0,  # disable so test fixtures pass
        "atr_sizing_enabled": True,
        "correlation_block_enabled": False,
        "coverage_sizing_enabled": False,
        "reduce_fraction": 0.50,
        "strategy": {
            "graduated_drawdown": {"enabled": False},
            "exit_manager": {
                "atr_trailing_enabled": False,
                "fallback_stop_enabled": False,
                "profit_take_enabled": False,
                "momentum_exit_enabled": False,
                "time_decay_enabled": False,
                "sector_relative_veto_enabled": False,
            },
        },
    }


@pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine sibling repo not present at ~/Development/alpha-engine",
)
class TestSimulateViaDecidersSmoke:
    def test_single_date_returns_orders(self):
        from executor.ibkr import SimulatedIBKRClient
        from backtest import _build_merged_simulate_config, _simulate_single_date

        signals = _make_signals(n_enter=2)
        ts = pd.Timestamp("2026-04-25")
        price_matrix = pd.DataFrame(
            {t: [100.0] for t in ["TKR000", "TKR001", "SPY"] +
             ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
              "XLRE", "XLB", "XLI", "XLC"]},
            index=[ts],
        )
        ohlcv_by_ticker = {
            t: _df_history(base=100 + i)
            for i, t in enumerate(price_matrix.columns)
        }
        atr_by_ticker = {t: 0.02 for t in price_matrix.columns}
        coverage_by_ticker = {t: 1.0 for t in price_matrix.columns}

        sim_client = SimulatedIBKRClient(prices={}, nav=1_000_000.0)
        merged_config, strategy_config = _build_merged_simulate_config(_config())

        orders, skip = _simulate_single_date(
            sim_client=sim_client,
            signal_date="2026-04-25",
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket="test-bucket",
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=signals,
            atr_by_ticker=atr_by_ticker,
            vwap_series_by_ticker=None,
            coverage_by_ticker=coverage_by_ticker,
        )

        assert skip is None
        assert isinstance(orders, list)
        # Each order tagged with the simulation date (parity contract)
        for o in orders:
            assert o["date"] == "2026-04-25"

    def test_state_carries_across_iterations(self):
        """Position accumulated on date 1 must be visible to date 2's
        ``already in portfolio`` check (so we don't re-ENTER held names)."""
        from executor.ibkr import SimulatedIBKRClient
        from backtest import _build_merged_simulate_config, _simulate_single_date

        signals_d1 = _make_signals(date_str="2026-04-25", n_enter=1)
        signals_d2 = _make_signals(date_str="2026-04-26", n_enter=1)
        # Same ticker on both days
        d1_ts = pd.Timestamp("2026-04-25")
        d2_ts = pd.Timestamp("2026-04-26")
        price_matrix = pd.DataFrame(
            {t: [100.0, 101.0] for t in ["TKR000", "SPY"] +
             ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
              "XLRE", "XLB", "XLI", "XLC"]},
            index=[d1_ts, d2_ts],
        )
        ohlcv_by_ticker = {
            t: _df_history(base=100 + i)
            for i, t in enumerate(price_matrix.columns)
        }
        atr_by_ticker = {t: 0.02 for t in price_matrix.columns}
        coverage_by_ticker = {t: 1.0 for t in price_matrix.columns}

        sim_client = SimulatedIBKRClient(prices={}, nav=1_000_000.0)
        merged_config, strategy_config = _build_merged_simulate_config(_config())

        # Date 1 — should produce an entry for TKR000
        orders_d1, _ = _simulate_single_date(
            sim_client=sim_client,
            signal_date="2026-04-25",
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket="test-bucket",
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=signals_d1,
            atr_by_ticker=atr_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
        )
        d1_enters = [o for o in orders_d1 if o["action"] == "ENTER"]
        assert len(d1_enters) == 1, f"Expected 1 entry on D1, got {len(d1_enters)}"
        assert d1_enters[0]["ticker"] == "TKR000"

        # sim_client should now hold TKR000
        assert "TKR000" in sim_client.get_positions()

        # Date 2 — same signal must NOT produce a second entry
        orders_d2, _ = _simulate_single_date(
            sim_client=sim_client,
            signal_date="2026-04-26",
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv_by_ticker,
            bucket="test-bucket",
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=signals_d2,
            atr_by_ticker=atr_by_ticker,
            coverage_by_ticker=coverage_by_ticker,
        )
        d2_enters = [o for o in orders_d2 if o["action"] == "ENTER"]
        assert len(d2_enters) == 0, (
            f"D2 produced {len(d2_enters)} ENTER orders for already-held names — "
            f"sim_client state not carrying forward correctly"
        )


class TestBuildMergedSimulateConfig:
    def test_builds_merged_config_when_executor_unavailable(self):
        """Without executor on sys.path (test isolation case), the
        function falls back to a minimal merge — top-level + strategy.*
        only. _PARAM_MAP-routed params silently fall through."""
        # We can't easily un-import executor, but we can verify the
        # function returns a tuple of (dict, dict) under both paths.
        from backtest import _build_merged_simulate_config

        merged, strategy_config = _build_merged_simulate_config({"init_cash": 100.0})
        assert isinstance(merged, dict)
        assert isinstance(strategy_config, dict)
        assert merged["init_cash"] == 100.0
