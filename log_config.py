"""
Structured logging configuration for the backtester.

Mirrors the pattern used in alpha-engine/executor/log_config.py: this module
owns the single shared FlowDoctor instance for the entire backtester
process, exposed via ``get_flow_doctor()``. All call sites (backtest.py,
evaluate.py, lambda_health/handler.py) should call ``get_flow_doctor()``
instead of calling ``flow_doctor.init()`` themselves — running multiple
independent FlowDoctor instances with separate SQLite stores, rate
limiters, and dedup states is a footgun.

Flow Doctor is enabled when ``FLOW_DOCTOR_ENABLED=1`` in the environment,
which is the default on the spot instance launched by
infrastructure/spot_backtest.sh and on the always-on EC2 running the
lambda_health handler.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import flow_doctor

_FLOW_DOCTOR_YAML_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "flow-doctor.yaml"
)

# Singleton — populated once by setup_logging() and retrieved by call sites
# via get_flow_doctor(). None until setup_logging() runs with
# FLOW_DOCTOR_ENABLED=1.
_fd_instance: Optional[flow_doctor.FlowDoctor] = None


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects (for structured log aggregation)."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "ctx"):
            log_entry["ctx"] = record.ctx
        return json.dumps(log_entry, default=str)


def get_flow_doctor() -> Optional[flow_doctor.FlowDoctor]:
    """Return the shared flow-doctor instance, or None if not initialized.

    Call sites use this to access flow-doctor without creating duplicate
    instances. Returns None if setup_logging() was never called with
    FLOW_DOCTOR_ENABLED=1, or if flow-doctor init failed in strict=False
    mode (which is not the default).
    """
    return _fd_instance


def _attach_flow_doctor(name: str) -> None:
    """Initialize the shared flow-doctor instance and attach a log handler."""
    global _fd_instance
    _fd_instance = flow_doctor.init(config_path=_FLOW_DOCTOR_YAML_PATH)
    handler = flow_doctor.FlowDoctorHandler(_fd_instance, level=logging.ERROR)
    logging.getLogger().addHandler(handler)


def setup_logging(name: str = "backtester") -> None:
    """
    Configure root logger for a backtester entry point.

    JSON mode: ``BACKTESTER_JSON_LOGS=1`` (for spot-instance / production)
    Text mode: default (for local dev / dry-run)
    Flow Doctor: ``FLOW_DOCTOR_ENABLED=1`` (for spot-instance / production)

    Args:
        name: short identifier for the entry point (backtest, evaluate,
            lambda_health). Appears in the text-mode log format and in
            the flow-doctor flow_name.
    """
    json_mode = os.environ.get("BACKTESTER_JSON_LOGS", "0") == "1"

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if os.environ.get("FLOW_DOCTOR_ENABLED", "0") == "1":
        _attach_flow_doctor(name)
