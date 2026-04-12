"""
grading.py — Unified component scorecard for the weekly backtester report.

Consumes results from all analysis modules and produces a single scorecard
with 0-100 grades and letter grades for every system component:

  Research: Scanner, 6 Sector Teams, Macro Agent, CIO, Composite Scoring
  Predictor: GBM Model, Veto Gate, Confidence Calibration
  Executor: Entry Triggers, Risk Guard, Exit Rules, Position Sizing, Portfolio

Each grade combines precision, recall, and domain-specific metrics into a
weighted composite. Components with insufficient data receive a grade of None
and are excluded from module-level averages.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grade bands
# ---------------------------------------------------------------------------

GRADE_BANDS = [
    (90, "A"),
    (80, "A-"),
    (73, "B+"),
    (65, "B"),
    (58, "B-"),
    (50, "C+"),
    (42, "C"),
    (35, "C-"),
    (28, "D+"),
    (20, "D"),
    (0, "F"),
]


def _letter(score: float | None) -> str:
    """Map a 0-100 numeric grade to a letter grade."""
    if score is None:
        return "N/A"
    score = max(0.0, min(100.0, score))
    for threshold, letter in GRADE_BANDS:
        if score >= threshold:
            return letter
    return "F"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _pct_to_grade(pct: float | None, baseline: float = 0.50,
                  ceiling: float = 0.80) -> float | None:
    """Map a 0-1 percentage to a 0-100 grade.

    baseline maps to 30 (D+), ceiling maps to 95 (A).
    Linear interpolation between them; clamped to [0, 100].
    """
    if pct is None:
        return None
    if ceiling == baseline:
        return 50.0
    raw = 30.0 + (pct - baseline) / (ceiling - baseline) * 65.0
    return _clamp(raw)


def _lift_to_grade(lift: float | None, floor: float = -2.0,
                   ceiling: float = 3.0) -> float | None:
    """Map a lift value (percentage points) to a 0-100 grade.

    floor maps to 0, 0.0 maps to 40, ceiling maps to 100.
    """
    if lift is None:
        return None
    if lift <= 0:
        # Negative lift: 0 at floor, 40 at zero
        if floor == 0:
            return 40.0
        raw = 40.0 * (1.0 - lift / floor)
    else:
        # Positive lift: 40 at zero, 100 at ceiling
        if ceiling == 0:
            return 40.0
        raw = 40.0 + 60.0 * (lift / ceiling)
    return _clamp(raw)


def _ic_to_grade(ic: float | None) -> float | None:
    """Map an information coefficient to a 0-100 grade.

    IC 0.00 → 20, IC 0.05 → 55, IC 0.10 → 90.
    """
    if ic is None:
        return None
    raw = 20.0 + ic * 700.0
    return _clamp(raw)


def _ratio_to_grade(ratio: float | None, target: float = 0.75) -> float | None:
    """Map a 0-1 ratio (e.g. capture ratio) to a grade.

    0.0 → 0, target → 80, 1.0 → 100.
    """
    if ratio is None:
        return None
    if ratio <= 0:
        return 0.0
    if ratio >= 1.0:
        return 100.0
    if ratio <= target:
        raw = 80.0 * (ratio / target)
    else:
        raw = 80.0 + 20.0 * ((ratio - target) / (1.0 - target))
    return _clamp(raw)


def _weighted_avg(components: list[tuple[float, float | None]]) -> float | None:
    """Weighted average of (weight, grade) pairs, skipping Nones."""
    total_w = 0.0
    total_v = 0.0
    for w, g in components:
        if g is not None:
            total_w += w
            total_v += w * g
    if total_w == 0:
        return None
    return total_v / total_w


def _safe_get(d: dict | None, *keys, default=None) -> Any:
    """Safely traverse nested dicts."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur


# ---------------------------------------------------------------------------
# Component grading functions
# ---------------------------------------------------------------------------

def _grade_scanner(e2e: dict | None, scanner_opt: dict | None) -> dict:
    """Grade the quant scanner filter."""
    sl = _safe_get(e2e, "scanner_lift")
    if not sl or _safe_get(sl, "n_passing") is None:
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    lift = _safe_get(sl, "lift")
    n_passing = _safe_get(sl, "n_passing", default=0)
    n_universe = _safe_get(sl, "n_universe", default=1)
    clf = _safe_get(sl, "classification")

    # Precision/recall from classification metrics (if available)
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.65) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.10, ceiling=0.40) if recall is not None else None

    # Fallback to lift if no classification data
    lift_g = _lift_to_grade(lift, floor=-1.5, ceiling=2.5)

    # Leakage from scanner_opt (lower is better)
    leakage = _safe_get(scanner_opt, "leakage_pct")
    if leakage is not None:
        leakage_g = _clamp(95.0 - leakage * 283.0)
    else:
        leakage_g = None

    if precision_g is not None and recall_g is not None:
        grade = _weighted_avg([
            (0.35, precision_g),
            (0.25, recall_g),
            (0.20, lift_g),
            (0.20, leakage_g),
        ])
    else:
        grade = _weighted_avg([
            (0.55, lift_g),
            (0.45, leakage_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift is not None:
        detail["lift"] = f"{lift:+.2f}%"
    if leakage is not None:
        detail["leakage"] = f"{leakage:.0%}"
    detail["n_passing"] = n_passing
    detail["n_universe"] = n_universe

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_sector_team(team: dict) -> dict:
    """Grade a single sector team from e2e_lift team_lift entry."""
    team_id = team.get("team_id", "unknown")
    n_picks = team.get("n_picks", 0)

    if n_picks < 3:
        return {
            "team_id": team_id, "grade": None, "letter": "N/A",
            "reason": f"only {n_picks} picks", "n_picks": n_picks,
        }

    lift_vs_sector = team.get("lift")
    lift_vs_quant = team.get("lift_vs_quant")
    clf = team.get("classification")

    # Classification metrics (if available)
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.70) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.10, ceiling=0.50) if recall is not None else None

    # Lift-based grades (always available)
    lift_sector_g = _lift_to_grade(lift_vs_sector, floor=-2.0, ceiling=3.0)
    lift_quant_g = _lift_to_grade(lift_vs_quant, floor=-2.0, ceiling=3.0)

    if precision_g is not None and recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_sector_g),
            (0.25, lift_quant_g),
        ])
    else:
        grade = _weighted_avg([
            (0.55, lift_sector_g),
            (0.45, lift_quant_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift_vs_sector is not None:
        detail["lift_vs_sector"] = f"{lift_vs_sector:+.2f}%"
    if lift_vs_quant is not None:
        detail["lift_vs_quant"] = f"{lift_vs_quant:+.2f}%"
    detail["n_picks"] = n_picks

    return {
        "team_id": team_id, "grade": grade, "letter": _letter(grade),
        "detail": detail,
    }


def _grade_macro(macro_eval: dict | None) -> dict:
    """Grade the macro agent's contribution."""
    if not macro_eval or macro_eval.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    acc_lift = macro_eval.get("accuracy_lift")
    alpha_lift = macro_eval.get("alpha_lift")
    assessment = macro_eval.get("assessment", "neutral")

    acc_g = _lift_to_grade(acc_lift, floor=-5.0, ceiling=5.0) if acc_lift is not None else None
    alpha_g = _lift_to_grade(alpha_lift, floor=-1.0, ceiling=2.0) if alpha_lift is not None else None

    grade = _weighted_avg([
        (0.50, acc_g),
        (0.50, alpha_g),
    ])

    detail = {}
    if acc_lift is not None:
        detail["accuracy_lift"] = f"{acc_lift:+.1f}pp"
    if alpha_lift is not None:
        detail["alpha_lift"] = f"{alpha_lift:+.2f}%"
    detail["assessment"] = assessment

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_cio(e2e: dict | None, cio_opt: dict | None) -> dict:
    """Grade the CIO's selection decisions."""
    cio_lift = _safe_get(e2e, "cio_lift")
    cio_vs = _safe_get(e2e, "cio_vs_ranking")

    if not cio_lift or _safe_get(cio_lift, "n_advance", default=0) < 3:
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    clf = _safe_get(cio_lift, "classification")
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.75) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.30, ceiling=0.70) if recall is not None else None

    # Lift-based grades (fallback/complement)
    adv_lift = cio_lift.get("lift")
    lift_g = _lift_to_grade(adv_lift, floor=-3.0, ceiling=3.0)

    # CIO vs mechanical ranking baseline
    ranking_lift = _safe_get(cio_vs, "lift")
    ranking_g = _lift_to_grade(ranking_lift, floor=-2.0, ceiling=2.0) if ranking_lift is not None else None

    if precision_g is not None and recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_g),
            (0.25, ranking_g),
        ])
    else:
        # Fallback: rejection spread as recall proxy
        reject_avg = cio_lift.get("reject_avg")
        advance_avg = cio_lift.get("advance_avg")
        if reject_avg is not None and advance_avg is not None:
            rejection_spread = advance_avg - reject_avg
            rejection_g = _lift_to_grade(rejection_spread, floor=-2.0, ceiling=4.0)
        else:
            rejection_g = None
        grade = _weighted_avg([
            (0.40, lift_g),
            (0.30, rejection_g),
            (0.30, ranking_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if adv_lift is not None:
        detail["selection_lift"] = f"{adv_lift:+.2f}%"
    if ranking_lift is not None:
        detail["vs_ranking"] = f"{ranking_lift:+.2f}%"
    detail["n_advance"] = cio_lift.get("n_advance", 0)
    detail["n_reject"] = cio_lift.get("n_reject", 0)

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_composite_scoring(signal_quality: dict | None,
                             score_cal: dict | None) -> dict:
    """Grade the composite scoring system (monotonicity + bucket accuracy)."""
    if not signal_quality or signal_quality.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    overall = signal_quality.get("overall", {})
    buckets = signal_quality.get("by_score_bucket", [])

    # Overall accuracy at 10d
    acc_10d = overall.get("accuracy_10d")
    acc_g = _pct_to_grade(acc_10d, baseline=0.45, ceiling=0.70)

    # High-score bucket accuracy (90+ should be highest)
    high_bucket = next((b for b in buckets if b.get("bucket") == "90+"), None)
    high_acc = _safe_get(high_bucket, "accuracy_10d") if high_bucket else None
    high_g = _pct_to_grade(high_acc, baseline=0.50, ceiling=0.80)

    # Monotonicity from calibration
    monotonic = _safe_get(score_cal, "monotonic")
    mono_g = 90.0 if monotonic else (40.0 if monotonic is not None else None)

    grade = _weighted_avg([
        (0.40, acc_g),
        (0.30, high_g),
        (0.30, mono_g),
    ])

    detail = {}
    if acc_10d is not None:
        detail["accuracy_10d"] = f"{acc_10d:.1%}"
    if high_acc is not None:
        detail["90+_accuracy"] = f"{high_acc:.1%}"
    if monotonic is not None:
        detail["monotonic"] = "YES" if monotonic else "NO"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_gbm_model(predictor_sizing: dict | None,
                     veto_result: dict | None) -> dict:
    """Grade the GBM prediction model quality."""
    ic = _safe_get(predictor_sizing, "overall_rank_ic")
    hit_rate = None

    # Try to get hit rate from predictor_sizing weekly data
    recent_weeks = _safe_get(predictor_sizing, "weekly_ic") or []
    n_positive = _safe_get(predictor_sizing, "recent_positive_weeks", default=0)
    n_total = _safe_get(predictor_sizing, "recent_total_weeks", default=0)

    if not predictor_sizing or predictor_sizing.get("status") != "ok":
        # Fall back to veto result for any signal of model quality
        if not veto_result or veto_result.get("status") not in ("ok", "insufficient_lift"):
            return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    ic_g = _ic_to_grade(ic)

    # Stability: fraction of recent weeks with positive IC
    if n_total > 0:
        stability = n_positive / n_total
        stability_g = _pct_to_grade(stability, baseline=0.40, ceiling=0.85)
    else:
        stability_g = None

    # Sizing lift (does p_up signal correlate with returns?)
    sizing_lift = _safe_get(predictor_sizing, "sizing_lift")
    sizing_g = _lift_to_grade(sizing_lift, floor=-1.0, ceiling=2.0) if sizing_lift is not None else None

    grade = _weighted_avg([
        (0.45, ic_g),
        (0.30, stability_g),
        (0.25, sizing_g),
    ])

    detail = {}
    if ic is not None:
        detail["rank_ic"] = f"{ic:.4f}"
    if n_total > 0:
        detail["stability"] = f"{n_positive}/{n_total} weeks positive"
    if sizing_lift is not None:
        detail["sizing_lift"] = f"{sizing_lift:+.2f}%"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_veto_gate(veto_result: dict | None,
                     veto_value: dict | None) -> dict:
    """Grade the predictor's veto system."""
    if not veto_result or veto_result.get("status") not in ("ok", "insufficient_lift"):
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    # Find the recommended threshold's metrics
    thresholds = veto_result.get("thresholds", [])
    rec_thresh = veto_result.get("recommended_threshold")
    rec_row = next((t for t in thresholds if t.get("confidence") == rec_thresh), None)

    precision = _safe_get(rec_row, "precision")
    recall = _safe_get(rec_row, "recall")
    f1 = _safe_get(rec_row, "f1")
    lift = _safe_get(rec_row, "lift")

    precision_g = _pct_to_grade(precision, baseline=0.45, ceiling=0.80)
    recall_g = _pct_to_grade(recall, baseline=0.10, ceiling=0.50) if recall is not None else None

    # Net dollar value (positive = veto system saves money)
    net_value = _safe_get(veto_value, "net_value")
    if net_value is not None:
        value_g = _clamp(50.0 + net_value / 20.0)
    else:
        value_g = None

    if recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.20, _lift_to_grade(lift, floor=-5.0, ceiling=20.0) if lift is not None else None),
            (0.30, value_g),
        ])
    else:
        grade = _weighted_avg([
            (0.40, precision_g),
            (0.30, _lift_to_grade(lift, floor=-5.0, ceiling=20.0) if lift is not None else None),
            (0.30, value_g),
        ])

    detail = {}
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if lift is not None:
        detail["lift"] = f"{lift:+.1f}pp"
    if net_value is not None:
        detail["net_value"] = f"${net_value:+,.0f}"
    detail["threshold"] = rec_thresh

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_entry_triggers(trigger_scorecard: dict | None) -> dict:
    """Grade entry trigger effectiveness."""
    if not trigger_scorecard or trigger_scorecard.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    summary = trigger_scorecard.get("summary", {})
    triggers = trigger_scorecard.get("triggers", [])

    # Overall slippage vs signal (negative = bought below signal = good)
    slip = summary.get("avg_slippage_vs_signal")
    if slip is not None:
        # -1% → 90, 0% → 55, +1% → 20
        slip_g = _clamp(55.0 - slip * 35.0)
    else:
        slip_g = None

    # Overall win rate
    win_rate = summary.get("win_rate_vs_spy")
    win_g = _pct_to_grade(win_rate, baseline=0.40, ceiling=0.65)

    # Overall avg alpha
    avg_alpha = summary.get("avg_realized_alpha")
    alpha_g = _lift_to_grade(avg_alpha, floor=-3.0, ceiling=5.0)

    grade = _weighted_avg([
        (0.35, slip_g),
        (0.35, win_g),
        (0.30, alpha_g),
    ])

    detail = {}
    if slip is not None:
        detail["avg_slippage"] = f"{slip:+.2f}%"
    if win_rate is not None:
        detail["win_rate"] = f"{win_rate:.1%}"
    if avg_alpha is not None:
        detail["avg_alpha"] = f"{avg_alpha:+.2f}%"
    detail["n_triggers"] = len(triggers)
    detail["total_entries"] = summary.get("total_entries", 0)

    # Per-trigger mini-grades
    trigger_grades = []
    for t in triggers:
        t_slip = t.get("avg_slippage_vs_signal")
        t_win = t.get("win_rate_vs_spy")
        t_slip_g = _clamp(55.0 - t_slip * 35.0) if t_slip is not None else None
        t_win_g = _pct_to_grade(t_win, baseline=0.40, ceiling=0.65)
        t_grade = _weighted_avg([(0.5, t_slip_g), (0.5, t_win_g)])
        trigger_grades.append({
            "trigger": t.get("trigger"),
            "grade": t_grade,
            "letter": _letter(t_grade),
            "n_trades": t.get("n_trades", 0),
        })
    detail["per_trigger"] = trigger_grades

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_risk_guard(shadow_book: dict | None) -> dict:
    """Grade the risk guard's blocking decisions."""
    if not shadow_book or shadow_book.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    assessment = shadow_book.get("assessment", "neutral")
    guard_lift = shadow_book.get("guard_lift")
    n_blocked = shadow_book.get("n_blocked", 0)
    clf = shadow_book.get("classification")

    # Classification metrics: precision = % blocked that were actual losers
    precision = _safe_get(clf, "precision")
    recall = _safe_get(clf, "recall")
    f1 = _safe_get(clf, "f1")

    precision_g = _pct_to_grade(precision, baseline=0.40, ceiling=0.75) if precision is not None else None
    recall_g = _pct_to_grade(recall, baseline=0.05, ceiling=0.30) if recall is not None else None

    # Guard lift: positive = blocked entries were worse than traded (good)
    lift_g = _lift_to_grade(guard_lift, floor=-3.0, ceiling=3.0) if guard_lift is not None else None

    # Assessment mapping
    assessment_scores = {
        "appropriate": 80.0,
        "too_tight": 45.0,
        "too_loose": 35.0,
        "neutral": 55.0,
        "insufficient_return_data": None,
    }
    assess_g = assessment_scores.get(assessment)

    # Fallback: blocked_beat_spy_pct if no classification
    if precision_g is None:
        blocked_beat = shadow_book.get("blocked_beat_spy_pct")
        if blocked_beat is not None:
            precision_g = _clamp(95.0 - blocked_beat * 95.0)

    if recall_g is not None:
        grade = _weighted_avg([
            (0.30, precision_g),
            (0.20, recall_g),
            (0.25, lift_g),
            (0.25, assess_g),
        ])
    else:
        grade = _weighted_avg([
            (0.35, precision_g),
            (0.35, lift_g),
            (0.30, assess_g),
        ])

    detail = {
        "assessment": assessment,
        "n_blocked": n_blocked,
    }
    if precision is not None:
        detail["precision"] = f"{precision:.1%}"
    if recall is not None:
        detail["recall"] = f"{recall:.1%}"
    if f1 is not None:
        detail["f1"] = f"{f1:.3f}"
    if guard_lift is not None:
        detail["guard_lift"] = f"{guard_lift:+.2f}%"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_exit_rules(exit_timing: dict | None) -> dict:
    """Grade exit rule effectiveness."""
    if not exit_timing or exit_timing.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    summary = exit_timing.get("summary", {})
    diagnosis = exit_timing.get("diagnosis", "unknown")

    capture = summary.get("avg_capture_ratio")
    capture_g = _ratio_to_grade(capture, target=0.70)

    avg_return = summary.get("avg_realized_return")
    return_g = _lift_to_grade(avg_return, floor=-5.0, ceiling=5.0) if avg_return is not None else None

    # Diagnosis bonus/penalty
    diag_scores = {
        "exits_well_timed": 85.0,
        "exits_could_improve": 55.0,
        "exits_too_early": 35.0,
    }
    diag_g = diag_scores.get(diagnosis)

    grade = _weighted_avg([
        (0.40, capture_g),
        (0.30, return_g),
        (0.30, diag_g),
    ])

    detail = {"diagnosis": diagnosis}
    if capture is not None:
        detail["capture_ratio"] = f"{capture:.2f}"
    if avg_return is not None:
        detail["avg_return"] = f"{avg_return:+.2f}%"
    detail["n_roundtrips"] = exit_timing.get("n_roundtrips", 0)

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_position_sizing(sizing_ab: dict | None) -> dict:
    """Grade position sizing vs equal-weight baseline."""
    if not sizing_ab or sizing_ab.get("status") != "ok":
        return {"grade": None, "letter": "N/A", "reason": "insufficient data"}

    sharpe_diff = sizing_ab.get("sharpe_diff")
    alpha_diff = sizing_ab.get("alpha_diff")
    assessment = sizing_ab.get("assessment", "no_difference")

    # Sharpe improvement: 0 → 50, +0.3 → 80, +0.5 → 95
    sharpe_g = _lift_to_grade(sharpe_diff, floor=-0.3, ceiling=0.5) if sharpe_diff is not None else None

    # Alpha improvement
    alpha_g = _lift_to_grade(alpha_diff, floor=-2.0, ceiling=3.0) if alpha_diff is not None else None

    grade = _weighted_avg([
        (0.55, sharpe_g),
        (0.45, alpha_g),
    ])

    detail = {"assessment": assessment}
    if sharpe_diff is not None:
        detail["sharpe_diff"] = f"{sharpe_diff:+.3f}"
    if alpha_diff is not None:
        detail["alpha_diff"] = f"{alpha_diff:+.2f}%"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


def _grade_portfolio(signal_quality: dict | None,
                     portfolio_stats: dict | None) -> dict:
    """Grade overall portfolio construction and performance."""
    overall = _safe_get(signal_quality, "overall") or {}

    acc_10d = overall.get("accuracy_10d")
    avg_alpha = overall.get("avg_alpha_10d")

    acc_g = _pct_to_grade(acc_10d, baseline=0.45, ceiling=0.70)
    alpha_g = _lift_to_grade(avg_alpha, floor=-2.0, ceiling=4.0) if avg_alpha is not None else None

    # Portfolio simulation stats (Sharpe, drawdown)
    sharpe = _safe_get(portfolio_stats, "sharpe_ratio")
    max_dd = _safe_get(portfolio_stats, "max_drawdown")

    if sharpe is not None:
        # Sharpe 0 → 30, 1.0 → 65, 2.0 → 95
        sharpe_g = _clamp(30.0 + sharpe * 32.5)
    else:
        sharpe_g = None

    if max_dd is not None:
        # -5% → 85, -10% → 65, -20% → 30, -30% → 10
        dd_g = _clamp(95.0 + max_dd * 2.83)  # max_dd is negative
    else:
        dd_g = None

    grade = _weighted_avg([
        (0.30, acc_g),
        (0.25, alpha_g),
        (0.25, sharpe_g),
        (0.20, dd_g),
    ])

    detail = {}
    if acc_10d is not None:
        detail["accuracy_10d"] = f"{acc_10d:.1%}"
    if avg_alpha is not None:
        detail["avg_alpha_10d"] = f"{avg_alpha:+.2f}%"
    if sharpe is not None:
        detail["sharpe"] = f"{sharpe:.2f}"
    if max_dd is not None:
        detail["max_drawdown"] = f"{max_dd:.1%}"

    return {"grade": grade, "letter": _letter(grade), "detail": detail}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_scorecard(
    signal_quality: dict | None = None,
    e2e_lift: dict | None = None,
    macro_eval: dict | None = None,
    score_calibration: dict | None = None,
    veto_result: dict | None = None,
    veto_value: dict | None = None,
    trigger_scorecard: dict | None = None,
    shadow_book: dict | None = None,
    exit_timing: dict | None = None,
    sizing_ab: dict | None = None,
    predictor_sizing: dict | None = None,
    portfolio_stats: dict | None = None,
    scanner_opt: dict | None = None,
    cio_opt: dict | None = None,
) -> dict:
    """Compute the unified system scorecard.

    Returns a dict with:
        status: "ok" | "partial" | "insufficient_data"
        overall: {grade, letter}
        research: {grade, letter, components: {...}}
        predictor: {grade, letter, components: {...}}
        executor: {grade, letter, components: {...}}
    """
    # -----------------------------------------------------------------------
    # Research components
    # -----------------------------------------------------------------------
    scanner = _grade_scanner(e2e_lift, scanner_opt)
    macro = _grade_macro(macro_eval)
    cio = _grade_cio(e2e_lift, cio_opt)
    composite = _grade_composite_scoring(signal_quality, score_calibration)

    # Sector teams
    # team_lift is contractually a list[dict] (see end_to_end._team_lift).
    # Defensive isinstance check here guards against producer regressions
    # where a status dict leaks through — iterating a dict yields its keys
    # (strings), which crashes _grade_sector_team.get() with AttributeError.
    # That's exactly what happened on 2026-04-11.
    team_lift_list = _safe_get(e2e_lift, "team_lift") or []
    if not isinstance(team_lift_list, list):
        team_lift_list = []
    teams = [_grade_sector_team(t) for t in team_lift_list]

    # Average team grade
    team_grades = [t["grade"] for t in teams if t.get("grade") is not None]
    avg_team_grade = sum(team_grades) / len(team_grades) if team_grades else None

    research_grade = _weighted_avg([
        (0.10, scanner.get("grade")),
        (0.30, avg_team_grade),
        (0.10, macro.get("grade")),
        (0.25, cio.get("grade")),
        (0.25, composite.get("grade")),
    ])

    research = {
        "grade": research_grade,
        "letter": _letter(research_grade),
        "components": {
            "scanner": scanner,
            "sector_teams": teams,
            "sector_teams_avg": {"grade": avg_team_grade, "letter": _letter(avg_team_grade)},
            "macro_agent": macro,
            "cio": cio,
            "composite_scoring": composite,
        },
    }

    # -----------------------------------------------------------------------
    # Predictor components
    # -----------------------------------------------------------------------
    gbm = _grade_gbm_model(predictor_sizing, veto_result)
    veto = _grade_veto_gate(veto_result, veto_value)

    predictor_grade = _weighted_avg([
        (0.55, gbm.get("grade")),
        (0.45, veto.get("grade")),
    ])

    predictor = {
        "grade": predictor_grade,
        "letter": _letter(predictor_grade),
        "components": {
            "gbm_model": gbm,
            "veto_gate": veto,
        },
    }

    # -----------------------------------------------------------------------
    # Executor components
    # -----------------------------------------------------------------------
    triggers = _grade_entry_triggers(trigger_scorecard)
    guard = _grade_risk_guard(shadow_book)
    exits = _grade_exit_rules(exit_timing)
    sizing = _grade_position_sizing(sizing_ab)
    portfolio = _grade_portfolio(signal_quality, portfolio_stats)

    executor_grade = _weighted_avg([
        (0.15, triggers.get("grade")),
        (0.20, guard.get("grade")),
        (0.20, exits.get("grade")),
        (0.15, sizing.get("grade")),
        (0.30, portfolio.get("grade")),
    ])

    executor = {
        "grade": executor_grade,
        "letter": _letter(executor_grade),
        "components": {
            "entry_triggers": triggers,
            "risk_guard": guard,
            "exit_rules": exits,
            "position_sizing": sizing,
            "portfolio": portfolio,
        },
    }

    # -----------------------------------------------------------------------
    # Overall
    # -----------------------------------------------------------------------
    overall_grade = _weighted_avg([
        (0.40, research_grade),
        (0.25, predictor_grade),
        (0.35, executor_grade),
    ])

    # Determine status
    graded_count = sum(1 for g in [research_grade, predictor_grade, executor_grade] if g is not None)
    if graded_count == 0:
        status = "insufficient_data"
    elif graded_count < 3:
        status = "partial"
    else:
        status = "ok"

    return {
        "status": status,
        "overall": {"grade": overall_grade, "letter": _letter(overall_grade)},
        "research": research,
        "predictor": predictor,
        "executor": executor,
    }
