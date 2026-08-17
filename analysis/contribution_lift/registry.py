"""registry.py — the list of replays the harness runs.

Kept deliberately tiny. Each T5 group lands ONE module under
``analysis/contribution_lift/groups/<group>.py`` exposing a module-level
``SPEC: ReplaySpec``, and appends ONE line here. Nothing else in this file
should ever need to change — a registry that grows logic is a registry two
sibling agents will conflict on.

Order is the emission order of ``components`` in the artifact; keep it grouped
by owning tile so the evaluator's tiles read contiguously.
"""

from __future__ import annotations

from analysis.contribution_lift.groups import cost_adjusted_quality
from analysis.contribution_lift.harness import ReplaySpec

SPECS: list[ReplaySpec] = [
    cost_adjusted_quality.SPEC,
]
