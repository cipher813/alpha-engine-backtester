"""Tests for backtester log_config — structured logging + flow-doctor singleton."""

import logging
import os
from unittest.mock import patch, MagicMock

import pytest

import log_config
from log_config import JSONFormatter, get_flow_doctor, setup_logging


@pytest.fixture(autouse=True)
def reset_singleton():
    """Ensure the singleton is reset between tests."""
    log_config._fd_instance = None
    yield
    log_config._fd_instance = None


class TestSetupLogging:
    def test_text_mode_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BACKTESTER_JSON_LOGS", None)
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            root = logging.getLogger()
            assert len(root.handlers) == 1
            assert root.level == logging.INFO

    def test_json_mode(self):
        with patch.dict(os.environ, {"BACKTESTER_JSON_LOGS": "1"}):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            root = logging.getLogger()
            assert isinstance(root.handlers[0].formatter, JSONFormatter)

    def test_clears_existing_handlers(self):
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        root.addHandler(logging.StreamHandler())
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
        assert len(root.handlers) == 1


class TestFlowDoctorSingleton:
    def test_get_flow_doctor_returns_none_when_disabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("FLOW_DOCTOR_ENABLED", None)
            setup_logging("test")
            assert get_flow_doctor() is None

    def test_get_flow_doctor_returns_instance_when_enabled(self):
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("backtest")
                assert get_flow_doctor() is mock_fd

    def test_setup_loads_from_yaml(self):
        """Shared instance must load from flow-doctor.yaml."""
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("backtest")
                mock_module.init.assert_called_once()
                call_kwargs = mock_module.init.call_args.kwargs
                assert "config_path" in call_kwargs
                assert call_kwargs["config_path"].endswith("flow-doctor.yaml")

    def test_init_failure_propagates(self):
        """flow-doctor init failures should propagate (strict mode default)."""
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("log_config.flow_doctor") as mock_module:
                mock_module.init.side_effect = RuntimeError("config error")
                with pytest.raises(RuntimeError, match="config error"):
                    setup_logging("test")

    def test_singleton_shared_across_call_sites(self):
        """Multiple get_flow_doctor() calls return the same instance."""
        mock_fd = MagicMock()
        mock_handler = MagicMock(spec=logging.Handler)
        mock_handler.level = logging.ERROR
        with patch.dict(os.environ, {"FLOW_DOCTOR_ENABLED": "1"}):
            with patch("log_config.flow_doctor") as mock_module:
                mock_module.init.return_value = mock_fd
                mock_module.FlowDoctorHandler.return_value = mock_handler
                setup_logging("backtest")
                assert get_flow_doctor() is get_flow_doctor()
                mock_module.init.assert_called_once()


class TestJSONFormatter:
    def test_formats_basic_record(self):
        import json
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py",
            lineno=1, msg="something failed", args=(), exc_info=None,
        )
        result = json.loads(formatter.format(record))
        assert result["level"] == "ERROR"
        assert result["msg"] == "something failed"
        assert "ts" in result

    def test_includes_exception(self):
        import json
        import sys
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="test.py",
                lineno=1, msg="failed", args=(), exc_info=sys.exc_info(),
            )
        result = json.loads(formatter.format(record))
        assert "ValueError" in result["exc"]
