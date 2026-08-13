"""The self-deadlining pit_parity walk-forward pass and its PARTIAL verdict
(alpha-engine-config#7199).

Two independent properties are under test, and they fail in different ways:

1. **The fold loop stops instead of being killed.** A pass SIGKILLed at its
   2700s ceiling produces no artifact at all — that is the 2026-08-07 cycle,
   where `pit_parity.json` read `status: failed` and the report card was still
   written `status: ok`, grade 55.7. A pass that stops early keeps every fold it
   completed.
2. **The partial can never read as a pass.** A contamination check that covered
   62% of the window and found nothing is a genuinely valuable result and an
   incomplete claim. The verdict vocabulary has to carry both facts at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from analysis.pit_parity import (
    DEADLINE_EPOCH_CONFIG_KEY,
    VERDICT_FAIL,
    VERDICT_PARTIAL,
    VERDICT_PASS,
    VERDICT_UNKNOWN,
    build_contamination_report,
    compute_verdict,
    coverage_from_wf_meta,
    verdict_is_pass,
)


# ════════════════════════════════════════════════════════════════════════════
# The affordability primitive
# ════════════════════════════════════════════════════════════════════════════

def _pb():
    """Import lazily — predictor_backtest pulls pandas, which the verdict half
    of this file does not need."""
    return pytest.importorskip("synthetic.predictor_backtest")


def test_deadline_key_matches_the_producer_side_copy():
    """The constant is duplicated (analysis/ must not import the predictor
    pipeline behind one string). This is the CI check that stops the copies
    drifting — the whole deadline handoff is silently inert if they disagree."""
    assert _pb().DEADLINE_EPOCH_CONFIG_KEY == DEADLINE_EPOCH_CONFIG_KEY


def test_no_deadline_never_stops_early():
    pb = _pb()
    assert pb._deadline_remaining_s({}) is None


def test_unreadable_deadline_is_ignored_not_guessed():
    """A garbage deadline must degrade to "no deadline", never to a truncation
    computed off a value nobody can read."""
    pb = _pb()
    assert pb._deadline_remaining_s({DEADLINE_EPOCH_CONFIG_KEY: "soon"}) is None


def test_first_fold_estimate_used_before_any_measurement():
    """Fold 0 has no observations to estimate from, so the floor estimate is
    what decides — a run handed a budget it has already blown must score ZERO
    folds rather than start one it cannot finish."""
    pb = _pb()
    assert not pb._next_fold_affordable(lambda: 10.0, [], reserve_s=0.0)
    assert pb._next_fold_affordable(
        lambda: pb.WF_FIRST_FOLD_ESTIMATE_S + 1.0, [], reserve_s=0.0,
    )


def test_estimate_tracks_the_slow_tail_not_the_median():
    """p50-sizing is the exact mistake the concordance cap made (150 artifacts
    at "3-5s" that measured 6-137s). With one slow fold among nine fast ones,
    a p50 budget says yes and the p90 budget says no."""
    pb = _pb()
    samples = [1.0] * 9 + [100.0]
    assert np.median(samples) < 50.0  # a p50 budget of 50s would say "yes"
    assert not pb._next_fold_affordable(lambda: 50.0, samples, reserve_s=0.0)


def test_reserve_is_actually_held_back():
    """The reserve is what pays for the simulation and the artifact write after
    the fold loop. Spending it on one more fold is how a run ends with full
    coverage and no artifact."""
    pb = _pb()
    assert pb._next_fold_affordable(lambda: 100.0, [10.0], reserve_s=0.0)
    assert not pb._next_fold_affordable(lambda: 100.0, [10.0], reserve_s=95.0)


def test_a_throwing_clock_means_unbounded_not_truncated():
    pb = _pb()

    def _boom():
        raise RuntimeError("no clock")

    assert pb._next_fold_affordable(_boom, [10.0], reserve_s=0.0)


# ════════════════════════════════════════════════════════════════════════════
# The verdict
# ════════════════════════════════════════════════════════════════════════════

_FULL = {"coverage_fraction": 1.0, "budget_stopped": False,
         "complete": True, "measured": True}
_PART = {"coverage_fraction": 0.62, "budget_stopped": True,
         "complete": False, "measured": True}
_ZERO = {"coverage_fraction": 0.0, "budget_stopped": False,
         "complete": True, "measured": True}


def test_full_coverage_clean_is_pass():
    v, _ = compute_verdict(material=False, coverage=_FULL, delta_available=True)
    assert v == VERDICT_PASS
    assert verdict_is_pass(v)


def test_partial_coverage_clean_is_partial_and_never_a_pass():
    v, reason = compute_verdict(
        material=False, coverage=_PART, delta_available=True,
    )
    assert v == VERDICT_PARTIAL
    assert not verdict_is_pass(v)
    assert "62.0%" in reason and "UNVERIFIED" in reason


def test_material_contamination_on_partial_coverage_still_fails():
    """Incomplete coverage withholds a clean bill of health. It must not
    withhold a POSITIVE finding — otherwise a truncated run launders evidence
    of leakage into an "incomplete, look again next week"."""
    v, _ = compute_verdict(material=True, coverage=_PART, delta_available=True)
    assert v == VERDICT_FAIL


def test_zero_coverage_with_no_budget_stop_is_unknown_not_partial():
    """Nothing was compared, so nothing can be claimed. (Zero coverage caused
    by the CLOCK is a different finding — see the timeout section below.)"""
    v, _ = compute_verdict(material=False, coverage=_ZERO, delta_available=True)
    assert v == VERDICT_UNKNOWN


def test_no_delta_is_unknown():
    v, _ = compute_verdict(material=None, coverage=_FULL, delta_available=False)
    assert v == VERDICT_UNKNOWN


def test_unmeasured_coverage_is_partial_never_pass():
    """A pass artifact written before this change reports no coverage at all.
    Reading that silence as full coverage is the §2.3a failure one layer down,
    so it renders PARTIAL."""
    coverage = coverage_from_wf_meta({"n_folds": 4})
    assert coverage["measured"] is False
    assert coverage["coverage_fraction"] is None
    v, _ = compute_verdict(material=False, coverage=coverage, delta_available=True)
    assert v == VERDICT_PARTIAL


def test_coverage_from_absent_meta_is_not_a_fabricated_one():
    coverage = coverage_from_wf_meta(None)
    assert coverage["coverage_fraction"] is None
    assert coverage["complete"] is None
    assert coverage["measured"] is False


# ════════════════════════════════════════════════════════════════════════════
# The report body
# ════════════════════════════════════════════════════════════════════════════

def _stats(returns, **kw):
    out = {
        "sortino_ratio": 1.0, "psr": 0.5, "cvar_95": -0.02,
        "max_drawdown": -0.1, "total_return": 0.2, "total_alpha": 0.05,
        "daily_log_returns": list(returns),
    }
    out.update(kw)
    return out


def _wf_meta(**kw):
    meta = {"enabled": True, "n_folds": 10, "n_folds_scored": 10,
            "n_test_dates_planned": 100, "n_test_dates_scored": 100,
            "coverage_fraction": 1.0, "budget_stopped": False,
            "complete": True, "covered_through": "2026-08-07"}
    meta.update(kw)
    return meta


def test_full_coverage_report_carries_a_pass_verdict_and_real_deltas():
    rng = np.random.default_rng(0)
    stream = rng.normal(0.0, 0.001, 200)
    report = build_contamination_report(
        _stats(stream), _stats(stream),
        run_date="2026-08-15", wf_meta=_wf_meta(),
    )
    assert report["verdict"] == VERDICT_PASS
    assert report["delta_basis"] == "full_window"
    assert report["coverage"]["coverage_fraction"] == 1.0
    assert report["delta_pit_minus_current"]["sortino_ratio"] is not None


def test_partial_report_withholds_the_scalar_basket_but_keeps_the_headline():
    """On a truncated walk-forward pass the two sides' Sortino/PSR/CVaR are
    computed over DIFFERENT windows, so differencing them measures the missing
    tail rather than contamination. Those legs are nulled. `log_cum_return` is
    a sum over the return stream, so it is honestly recomputable on the aligned
    common prefix and survives."""
    rng = np.random.default_rng(1)
    cur = rng.normal(0.0, 0.001, 200)
    pit = cur[:124]  # the pass stopped at 62% of the window
    report = build_contamination_report(
        _stats(cur), _stats(pit),
        run_date="2026-08-15",
        wf_meta=_wf_meta(coverage_fraction=0.62, budget_stopped=True,
                         complete=False, n_test_dates_scored=62),
    )
    assert report["verdict"] == VERDICT_PARTIAL
    assert report["delta_basis"] == "aligned_common_prefix_log_return_only"
    delta = report["delta_pit_minus_current"]
    assert delta["sortino_ratio"] is None
    assert delta["psr"] is None
    assert delta["max_drawdown"] is None
    # Identical prefixes => an exactly-zero aligned headline, not a null.
    assert delta["log_cum_return"] == pytest.approx(0.0, abs=1e-12)


def test_partial_report_status_stays_ok_but_verdict_does_not():
    """The STAGE succeeded — it produced an honest comparison. The CLAIM is
    incomplete. Collapsing those two is what let 2026-08-07 grade `status: ok`
    with a timed-out contamination check."""
    rng = np.random.default_rng(2)
    cur = rng.normal(0.0, 0.001, 100)
    report = build_contamination_report(
        _stats(cur), _stats(cur[:50]),
        run_date="2026-08-15",
        wf_meta=_wf_meta(coverage_fraction=0.5, budget_stopped=True,
                         complete=False),
    )
    assert not verdict_is_pass(report["verdict"])
    assert report["verdict_reason"]


def test_verdict_is_present_on_every_report_shape(monkeypatch):
    """The §2.3a property, asserted structurally: no path through this module
    emits a contamination report without a verdict field. A consumer must never
    have to infer the verdict from its absence."""
    import analysis.pit_parity as pp

    report = pp.build_contamination_report(
        _stats([0.001] * 10), _stats([0.001] * 10),
        run_date="2026-08-15", wf_meta=_wf_meta(),
    )
    assert report["verdict"] in {VERDICT_PASS, VERDICT_FAIL,
                                 VERDICT_PARTIAL, VERDICT_UNKNOWN}

    monkeypatch.setattr(pp, "_write_artifact_to_s3", lambda *a, **k: None)
    failed = pp.write_failure_artifact({"_run_date": "2026-08-15"},
                                       RuntimeError("timed out"))
    assert failed["verdict"] == VERDICT_FAIL


# ════════════════════════════════════════════════════════════════════════════
# A timeout is a FAILURE, not an absence (Brian ruling 2026-08-13)
# ════════════════════════════════════════════════════════════════════════════

from analysis.pit_parity import (  # noqa: E402
    failure_verdict,
    is_timeout_failure,
    write_failure_artifact,
)


def test_the_literal_2026_08_07_error_is_classified_as_a_timeout():
    """The exact recorded body from the cycle that motivated this arc."""
    assert is_timeout_failure(
        "RuntimeError",
        "pit_parity walkforward pass timed out after 2700s: ...",
    )


def test_an_explicit_flag_beats_the_message_match():
    """Producers set `timed_out`; the string match exists only for artifacts
    written before the flag did."""
    assert is_timeout_failure("RuntimeError", "anything", timed_out=True)
    assert not is_timeout_failure("RuntimeError", "timed out", timed_out=False)


def test_a_timeout_verdict_is_fail_and_every_other_crash_is_unknown():
    assert failure_verdict("RuntimeError", "timed out after 2700s") == VERDICT_FAIL
    assert failure_verdict("RecursionError", "maximum depth") == VERDICT_UNKNOWN
    assert failure_verdict("MemoryError", "oom") == VERDICT_UNKNOWN


def test_timed_out_failure_artifact_carries_fail_not_unknown(monkeypatch):
    import analysis.pit_parity as pp

    monkeypatch.setattr(pp, "_write_artifact_to_s3", lambda *a, **k: None)
    report = write_failure_artifact(
        {"_run_date": "2026-08-15"},
        RuntimeError("pit_parity walkforward pass timed out after 2700s: ..."),
    )
    assert report["verdict"] == VERDICT_FAIL
    assert report["timed_out"] is True
    assert report["status"] == "failed"


def test_a_non_timeout_crash_artifact_stays_unknown(monkeypatch):
    import analysis.pit_parity as pp

    monkeypatch.setattr(pp, "_write_artifact_to_s3", lambda *a, **k: None)
    report = write_failure_artifact(
        {"_run_date": "2026-08-15"}, RecursionError("maximum recursion depth"),
    )
    assert report["verdict"] == VERDICT_UNKNOWN
    assert report["timed_out"] is False


def test_zero_coverage_after_a_budget_stop_is_fail_not_unknown():
    """The pass ran and produced nothing before its wall. That is the timeout
    condition expressed through the deadline rather than through SIGKILL, and
    the ruling applies to it identically."""
    v, reason = compute_verdict(
        material=False,
        coverage={"coverage_fraction": 0.0, "budget_stopped": True,
                  "complete": False, "measured": True},
        delta_available=True,
    )
    assert v == VERDICT_FAIL
    assert "budget" in reason


def test_zero_coverage_without_a_budget_stop_is_still_unknown():
    """Zero folds for a reason other than the clock — an empty grid, a feature
    store with nothing in it — is an absence, not a wall-kill."""
    v, _ = compute_verdict(
        material=False,
        coverage={"coverage_fraction": 0.0, "budget_stopped": False,
                  "complete": True, "measured": True},
        delta_available=True,
    )
    assert v == VERDICT_UNKNOWN


def test_compare_turns_a_timed_out_pass_artifact_into_failed():
    from analysis.pit_stats_artifact import build_unknown_report

    report = build_unknown_report(
        "2026-08-15",
        {"lookahead": "ok", "walkforward": "failed"},
        {"lookahead": None,
         "walkforward": {"status": "failed", "timed_out": True,
                         "error_class": "RuntimeError",
                         "error_msg": "pit_parity walkforward pass timed out "
                                      "after 2700s: ..."}},
    )
    assert report["verdict"] == VERDICT_FAIL
    # The dashboard's integrity panel upper-cases `status` — it must keep
    # reading FAILED, which is what it already showed while the report card
    # said `ok`.
    assert report["status"] == "failed"
    assert report["timed_out_passes"] == ["walkforward"]


def test_compare_with_a_merely_absent_pass_artifact_stays_unknown():
    from analysis.pit_stats_artifact import build_unknown_report

    report = build_unknown_report(
        "2026-08-15", {"lookahead": "ok", "walkforward": "missing"},
        {"lookahead": None, "walkforward": None},
    )
    assert report["verdict"] == VERDICT_UNKNOWN
    assert report["status"] == "unknown"
