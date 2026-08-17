"""analysis.contribution_lift — the shared contribution-lift replay harness.

Report Card v3 T5 (epic alpha-engine-config-I7473, harness I7475). Measures
each graded component's marginal contribution to the ONE objective — per-cycle
net-of-cost 21d log-alpha vs SPY — by replaying the pipeline with the
component as-configured and with it ablated, both through the production
cost-bearing simulator.

Public entry point: ``build_contribution_lift_report``.
"""

from analysis.contribution_lift.harness import (  # noqa: F401
    HORIZON_DAYS,
    Arm,
    ArmSet,
    NotAvailable,
    ReplayInputs,
    ReplaySpec,
    orders_arm,
    picks_arm,
)
from analysis.contribution_lift.inputs import load_replay_inputs  # noqa: F401
from analysis.contribution_lift.report import (  # noqa: F401
    build_contribution_lift_report,
)

__all__ = [
    "Arm",
    "ArmSet",
    "HORIZON_DAYS",
    "NotAvailable",
    "ReplayInputs",
    "ReplaySpec",
    "build_contribution_lift_report",
    "load_replay_inputs",
    "orders_arm",
    "picks_arm",
]
