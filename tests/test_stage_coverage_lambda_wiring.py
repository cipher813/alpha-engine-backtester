"""Per-stage output-coverage assertion wiring for the two Lambda handlers
in this repo (alpha-engine-config-I7214) — ReplayConcordance and
Counterfactual, the only crucible-backtester stages that are Lambda
handlers rather than spot launchers (see
test_stage_coverage_assertion_wiring.py for the shell-launcher coverage).

Both handlers call the ONE shared `krepis.stage_coverage` module landed by
a separate krepis PR. krepis is PyPI-published (this repo's
`requirements.txt` already floors it at `krepis[openai]>=0.55.0` — a
future release adding the module needs no pin-bump PR here), and does NOT
carry that module at any released version yet — these tests verify BOTH
sides of that gap: (1) the handler degrades loudly-but-harmlessly
(ImportError caught, logged, handler outcome unchanged) against today's
installed krepis, matching this environment's actually-installed version
(confirmed absent: `python -c "import krepis.stage_coverage"` fails here
exactly as it will in CI), and (2) once the module IS importable
(simulated via a fake module injected into sys.modules), its verdict lands
under the `stage_coverage` key in the returned payload.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent

_HANDLERS = {
    "lambda_concordance": {
        "path": _REPO_ROOT / "lambda_concordance" / "handler.py",
        "compute_target": "replay.batch.compute_and_emit_concordance",
        "sf_stage": "ReplayConcordance",
    },
    "lambda_counterfactual": {
        "path": _REPO_ROOT / "lambda_counterfactual" / "handler.py",
        "compute_target": "replay.counterfactual.compute_and_emit",
        "sf_stage": "Counterfactual",
    },
}


def _load_handler_module(name: str, path: Path):
    module_name = f"{name}_stage_coverage_test_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture(params=sorted(_HANDLERS))
def handler_case(request):
    name = request.param
    cfg = _HANDLERS[name]
    mod = _load_handler_module(name, cfg["path"])
    mod._init_done = False
    yield name, mod, cfg


def _ok_summary(handler_name: str) -> dict:
    if handler_name == "lambda_concordance":
        return {
            "artifacts_discovered": 3,
            "per_target_model": [
                {
                    "target_model": "deepseek/deepseek-v4-flash",
                    "n_artifacts_replayed": 3,
                    "replay_failures": [],
                    "budget_stopped": False,
                },
            ],
        }
    return {
        "agents_analyzed": 2,
        "agents_skipped_thin_sample": [],
        "agents_unsupported": [],
        "load_failures": [],
        "fit_failures": [],
    }


# ── Real environment: krepis.stage_coverage is NOT importable ────────
# (Confirmed absent from the installed krepis at the current pin —
# this is not a mock, it's this handler's genuine import path.)


class TestModuleAbsentDegradesLoudlyNotSilently:
    def test_import_error_does_not_change_handler_outcome(self, handler_case):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({}, context=None)
        # Handler's own status/summary contract is unaffected by the
        # absent module — this IS the observe-mode degrade contract.
        assert result["status"] == "OK"
        assert "summary" in result

    def test_import_error_is_logged_not_swallowed(self, handler_case):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)), \
             patch.object(mod.logger, "error") as mock_error:
            mod.handler({}, context=None)
        assert mock_error.called, (
            f"{name}: ImportError for krepis.stage_coverage must be "
            f"logged (logger.error), not silently passed"
        )
        logged_msg = mock_error.call_args[0][0]
        assert "stage-coverage" in logged_msg

    def test_stage_coverage_key_absent_when_module_unavailable(self, handler_case):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({}, context=None)
        assert "stage_coverage" not in result


# ── Simulated post-pin-bump environment: module IS importable ───────────────


@pytest.fixture
def fake_stage_coverage_module():
    """Inject a fake krepis.stage_coverage into sys.modules so the
    handler's `from krepis.stage_coverage import assert_stage_coverage`
    succeeds — simulating the state after the nousergon-lib pin bump lands
    (a separate wave, per the PR body's merge order). Restores prior state
    on teardown so this fixture cannot leak into other tests."""
    calls = []

    def _assert_stage_coverage(stage, *, run_date, window_start):
        calls.append({"stage": stage, "run_date": run_date, "window_start": window_start})
        return {"stage": stage, "run_date": run_date, "status": "COVERED"}

    fake_mod = types.ModuleType("krepis.stage_coverage")
    fake_mod.assert_stage_coverage = _assert_stage_coverage

    had_parent = "krepis" in sys.modules
    parent = sys.modules.get("krepis")
    had_submodule_attr = had_parent and hasattr(parent, "stage_coverage")

    if not had_parent:
        import krepis as parent  # noqa: PLC0415 — real import, establishes sys.modules entry
    sys.modules["krepis.stage_coverage"] = fake_mod
    setattr(parent, "stage_coverage", fake_mod)

    yield calls

    del sys.modules["krepis.stage_coverage"]
    if had_submodule_attr:
        pass  # was already there before us (unlikely) — leave as-is
    else:
        if hasattr(parent, "stage_coverage"):
            delattr(parent, "stage_coverage")


class TestModulePresentVerdictLandsInPayload:
    def test_verdict_merged_under_stage_coverage_key(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            result = mod.handler({}, context=None)
        assert "stage_coverage" in result, f"{name}: verdict did not land in the returned payload"
        assert result["stage_coverage"]["stage"] == cfg["sf_stage"]
        assert result["stage_coverage"]["status"] == "COVERED"

    def test_called_with_correct_sf_stage_name(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({}, context=None)
        assert len(fake_stage_coverage_module) == 1
        assert fake_stage_coverage_module[0]["stage"] == cfg["sf_stage"]

    def test_window_start_is_tz_aware_datetime_captured_at_entry(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        before = datetime.now(timezone.utc)
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({}, context=None)
        after = datetime.now(timezone.utc)
        window_start = fake_stage_coverage_module[0]["window_start"]
        assert window_start.tzinfo is not None
        assert before <= window_start <= after

    def test_run_date_derived_from_end_time_iso_when_present(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({"end_time_iso": "2026-05-09T00:00:00Z"}, context=None)
        assert fake_stage_coverage_module[0]["run_date"] == "2026-05-09"

    def test_run_date_falls_back_to_now_when_end_time_absent(self, handler_case, fake_stage_coverage_module):
        name, mod, cfg = handler_case
        today = datetime.now(timezone.utc).date().isoformat()
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], return_value=_ok_summary(name)):
            mod.handler({}, context=None)
        assert fake_stage_coverage_module[0]["run_date"] == today

    def test_verdict_call_does_not_raise_on_error_status_path(self, handler_case, fake_stage_coverage_module):
        """The assertion sits after the try/except around compute_and_emit*
        — an ERROR-status early return must NOT reach the assertion (there
        is no successful output to assert coverage of)."""
        name, mod, cfg = handler_case
        with patch.object(mod, "_ensure_init"), \
             patch(cfg["compute_target"], side_effect=RuntimeError("boom")):
            result = mod.handler({}, context=None)
        assert result["status"] == "ERROR"
        assert "stage_coverage" not in result
        assert fake_stage_coverage_module == []
