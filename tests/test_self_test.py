"""Tests for `analysis/self_test.py` — the published known-answer self-test.

Two layers, and the second is the load-bearing one:

1. **The battery agrees on THIS runner.** Every case passes here too, so a CI
   failure and an in-pipeline failure mean the same thing and can be compared.
2. **The runner's outcome taxonomy holds.** A case that DISAGREES is FAIL, a
   case that could not RUN is UNKNOWN, and a case that exceeds its budget is
   FAIL (Brian ruling 2026-08-13) — never UNKNOWN. These are asserted with
   substituted cases rather than by breaking the engine, because the taxonomy is
   what must not be reimplemented or drifted, and it is the part that decides
   whether a harness fault gets reported as a correctness regression.
"""

from __future__ import annotations

import json
import math

import pytest

from analysis import self_test as st


# ── layer 1: the real battery ───────────────────────────────────────────────

@pytest.fixture(scope="module")
def body():
    return st.run_self_test(run_date="2026-08-15")


def test_every_case_passes_on_this_runner(body):
    failures = [c for c in body["cases"] if c["verdict"] != st.PASS]
    assert not failures, json.dumps(failures, indent=2, default=str)
    assert body["verdict"] == st.PASS
    assert body["n_cases"] == len(st.build_cases())


def test_brians_named_cases_are_all_present(body):
    """The battery must keep covering what was actually asked for.

    A case silently dropped in a refactor is a coverage regression nothing else
    would notice — the artifact would still say PASS, on fewer questions.
    """
    names = {c["case"] for c in body["cases"]}
    assert {
        "single_round_trip",
        "flat_market_total_return",
        "flat_market_max_drawdown",
        "flat_market_sharpe_is_undefined",
        "fee_charged_once_per_side",
        "currency_scale_invariance",
        "size_linearity",
        "closed_form_sharpe",
        "null_control_alpha_ci_covers_zero",
    } <= names


def test_single_round_trip_is_the_hand_computed_dollar_answer(body):
    case = next(c for c in body["cases"] if c["case"] == "single_round_trip")
    assert case["expected"] == 99.00
    assert abs(case["actual"] - 99.00) <= case["tolerance"]


def test_expected_sharpe_is_derived_here_not_by_the_engine():
    """The expectation must be reproducible from the scenario definition alone.

    Recomputed in the test from the same first principles the module documents.
    An expectation obtained by running the code under test would agree with
    whatever that code ever does — which is the defect this module exists to
    close, so it is asserted rather than assumed.
    """
    observations = [0.0] + [0.02, -0.01] * 10
    n = len(observations)
    mean = sum(observations) / n
    var = sum((r - mean) ** 2 for r in observations) / (n - 1)
    assert st._expected_closed_form_sharpe() == pytest.approx(
        mean / math.sqrt(var) * math.sqrt(365), rel=0, abs=1e-15,
    )


def test_artifact_carries_the_provenance_header(body):
    """The library versions ARE the deliverable — this is an instrument check."""
    assert body["schema"] == "backtest_self_test-1.0.0"
    assert body["component"] == "backtester"
    assert body["run_date"] == "2026-08-15"
    assert body["python"]
    assert "code_sha" in body
    for dist in ("vectorbt", "numpy", "pandas", "nousergon-lib"):
        assert dist in body["libraries"]
        assert body["libraries"][dist], f"{dist} version is empty"


def test_every_case_row_carries_the_full_shape(body):
    for case in body["cases"]:
        assert set(case) >= {
            "case", "description", "inputs", "expected", "actual",
            "abs_error", "tolerance", "verdict",
        }
        assert case["inputs"], f"{case['case']} publishes no inputs to re-derive from"
        assert case["verdict"] in (st.PASS, st.FAIL, st.UNKNOWN)


def test_artifact_is_json_serialisable_and_finite(body):
    """A NaN or inf anywhere makes the artifact invalid JSON for a strict reader.

    ``allow_nan=False`` is the assertion: it RAISES on a non-finite float
    anywhere in the body. (Substring-matching the rendered text for "NaN" would
    be wrong — the word legitimately appears in a case description.)
    """
    text = json.dumps(body, allow_nan=False, default=str)
    assert json.loads(text)["verdict"] == body["verdict"]


def test_battery_is_cheap_enough_to_run_every_cycle(body):
    assert body["wall_clock_seconds"] < 60.0


# ── layer 2: the outcome taxonomy ───────────────────────────────────────────

def _case(name="c", expected=1.0, compute=lambda: 1.0, tolerance=0.0):
    return st.Case(name=name, description="d", inputs={"k": 1},
                   expected=expected, compute=compute, tolerance=tolerance)


def test_disagreement_is_FAIL_not_UNKNOWN():
    body = st.run_self_test(case_provider=lambda: [_case(expected=1.0, compute=lambda: 2.0)])
    assert body["cases"][0]["verdict"] == st.FAIL
    assert body["verdict"] == st.FAIL
    assert body["n_failed"] == 1


def test_a_case_that_could_not_run_is_UNKNOWN_not_FAIL():
    def _boom():
        raise RuntimeError("import blew up")

    body = st.run_self_test(case_provider=lambda: [_case(compute=_boom)])
    assert body["cases"][0]["verdict"] == st.UNKNOWN
    assert body["cases"][0]["error_class"] == "RuntimeError"
    assert body["verdict"] == st.UNKNOWN


def test_a_timeout_is_FAIL_never_UNKNOWN():
    """Brian ruling 2026-08-13. A self-test that cannot finish is itself evidence."""
    import time as _time

    body = st.run_self_test(
        case_provider=lambda: [_case(compute=lambda: _time.sleep(2) or 1.0)],
        case_timeout_seconds=0.2,
    )
    assert body["cases"][0]["verdict"] == st.FAIL
    assert body["cases"][0]["timed_out"] is True
    assert body["verdict"] == st.FAIL


def test_a_battery_that_could_not_be_built_is_UNKNOWN_and_does_not_raise():
    def _boom():
        raise ImportError("no engine")

    body = st.run_self_test(case_provider=_boom)
    assert body["verdict"] == st.UNKNOWN
    assert body["status"] == "error"
    assert body["cases"] == []


def test_an_empty_battery_is_UNKNOWN_never_PASS():
    """Zero cases must never read as a clean bill of health."""
    body = st.run_self_test(case_provider=lambda: [])
    assert body["verdict"] == st.UNKNOWN


def test_fail_beats_unknown_in_the_overall_verdict():
    def _boom():
        raise RuntimeError("x")

    body = st.run_self_test(case_provider=lambda: [
        _case(name="a", expected=1.0, compute=lambda: 2.0),
        _case(name="b", compute=_boom),
    ])
    assert body["verdict"] == st.FAIL


def test_verdict_is_pass_only_on_the_literal_PASS():
    assert st.verdict_is_pass("PASS")
    for other in (None, "", "ok", "pass", "UNKNOWN", "FAIL", True):
        assert not st.verdict_is_pass(other)


def test_run_self_test_never_raises_even_on_a_broken_provider():
    class _Exploding:
        def __iter__(self):
            raise ValueError("not iterable")

    body = st.run_self_test(case_provider=lambda: _Exploding())
    assert body["verdict"] == st.UNKNOWN


def test_missing_distribution_is_recorded_explicitly_never_omitted():
    resolved = st.resolved_library_versions(("nousergon-lib", "definitely-not-a-package"))
    assert resolved["definitely-not-a-package"] == "<not installed>"
    assert "nousergon-lib" in resolved


def test_self_test_key_is_the_declared_artifact_path():
    assert st.self_test_key("2026-08-15") == "backtest/2026-08-15/self_test.json"


# ── wiring: the artifact is actually emitted by the stage ───────────────────

def test_reporter_emits_self_test_json_in_the_always_emit_block():
    """String-level wiring assertion: the artifact must be in `reporter.save`'s
    always-emit list, or it silently stops being published and nothing fails."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "reporter.py").read_text()
    assert '("self_test.json", self_test),' in source
    assert "from analysis.self_test import run_self_test as _run_self_test" in source


def test_evaluate_records_the_verdict_on_the_completeness_tracker():
    """A non-PASS verdict must raise the stage's EXISTING degraded flag."""
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "evaluate.py").read_text()
    assert 'name="self_test"' in source
    assert '_self_test_is_pass(_self_test.get("verdict")) else "degraded"' in source
    assert "self_test=_self_test," in source
