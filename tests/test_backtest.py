"""Unit tests for backtest.py — config override mapping, simulation loop gates."""
import logging
import pytest
from unittest.mock import patch, MagicMock

import pandas as pd

from backtest import _build_config_override, _run_simulation_loop


# ── _build_config_override ───────────────────────────────────────────────────


class TestBuildConfigOverride:

    def test_direct_risk_params_mapped_to_top_level(self):
        """min_score, max_position_pct should be top-level in override."""
        config = {"min_score": 75, "max_position_pct": 0.10}
        override = _build_config_override(config)
        assert override["min_score"] == 75
        assert override["max_position_pct"] == 0.10
        assert "strategy" not in override

    def test_strategy_params_nested_under_exit_manager(self):
        """atr_multiplier etc. should be nested under strategy.exit_manager."""
        config = {"atr_multiplier": 3.0, "time_decay_exit_days": 15}
        override = _build_config_override(config)
        em = override["strategy"]["exit_manager"]
        assert em["atr_multiplier"] == 3.0
        assert em["time_decay_exit_days"] == 15

    def test_profit_take_pct_mapped(self):
        """profit_take_pct should be mapped to strategy.exit_manager."""
        config = {"profit_take_pct": 0.20}
        override = _build_config_override(config)
        assert override["strategy"]["exit_manager"]["profit_take_pct"] == 0.20

    def test_no_sweep_params_returns_none(self):
        """Config with no recognized sweep params should return None."""
        config = {"signals_bucket": "test", "email_sender": "x@y.com"}
        override = _build_config_override(config)
        assert override is None

    def test_unmapped_safe_param_logs_warning(self, caplog):
        """SAFE_PARAMS key not in _RECOGNIZED_SWEEP_PARAMS should warn."""
        # reduce_fraction is in SAFE_PARAMS but not mapped in _build_config_override
        config = {"reduce_fraction": 0.33}
        with caplog.at_level(logging.WARNING):
            _build_config_override(config)
        assert "not mapped" in caplog.text.lower() or "ignored" in caplog.text.lower()

    def test_mixed_params(self):
        """Both direct and strategy params should coexist."""
        config = {"min_score": 70, "atr_multiplier": 2.5, "profit_take_pct": 0.25}
        override = _build_config_override(config)
        assert override["min_score"] == 70
        assert override["strategy"]["exit_manager"]["atr_multiplier"] == 2.5
        assert override["strategy"]["exit_manager"]["profit_take_pct"] == 0.25


# ── _run_simulation_loop — stale_circuit_break gate ──────────────────────────


class TestSimulationLoopGates:

    def test_stale_circuit_break_halts_simulation(self):
        """stale_circuit_break=True should return stale_prices immediately."""
        price_matrix = pd.DataFrame({"AAPL": [100.0]}, index=pd.to_datetime(["2026-03-01"]))
        price_matrix.attrs["stale_circuit_break"] = True
        price_matrix.attrs["staleness_warning"] = "STALE: last date 2026-03-01"

        result = _run_simulation_loop(
            executor_run=MagicMock(),
            SimulatedIBKRClient=MagicMock,
            dates=["2026-03-01"],
            price_matrix=price_matrix,
            config={"init_cash": 100000},
        )
        assert result["status"] == "stale_prices"

    @patch("loaders.signal_loader.load")
    def test_no_stale_flag_proceeds(self, mock_signal_load):
        """Without stale flag, simulation should proceed (may produce no_orders)."""
        mock_signal_load.side_effect = FileNotFoundError("no signals in test")

        price_matrix = pd.DataFrame(
            {"AAPL": [100.0, 101.0]},
            index=pd.to_datetime(["2026-03-01", "2026-03-02"]),
        )
        price_matrix.attrs["stale_circuit_break"] = False

        mock_executor = MagicMock(return_value=[])
        mock_client_cls = MagicMock()
        mock_client_cls.return_value = MagicMock()

        result = _run_simulation_loop(
            executor_run=mock_executor,
            SimulatedIBKRClient=mock_client_cls,
            dates=["2026-03-01", "2026-03-02"],
            price_matrix=price_matrix,
            config={"init_cash": 100000},
        )
        # Should not be stale_prices
        assert result["status"] != "stale_prices"
