"""alpha-engine-config-I8701 — the hardcoded 30-day lookback under-samples.

`pipeline_common.push_predictor_rolling_metrics` already fixed this on its own
side: a prediction's label takes `forward_days` TRADING days (~30 calendar days
at 21) to close, so a 30-CALENDAR-day window sees graded rows only at its very
start, and they slide out within days of the database being refreshed.
`analysis/production_health.py` kept the literal 30.

Measured 2026-08-26 against the SAME research.db (the S3 copy is refreshed
weekly, last written 2026-08-22):

    run 2026-08-22, window [07-23, 08-22] -> n=25, degradation_flag TRUE
    run 2026-08-26, window [07-27, 08-26] -> n=0, n_any_horizon=0, "skipped"

Four days of window drift took every graded row out. That made the weekday
rule un-paused by nousergon-data-PR1543 actively harmful: on the four weekdays
that are not the refresh day it computed nothing and OVERWROTE the good
artifact with an `insufficient_samples` payload, destroying the very
`degradation_flag` the detector exists to raise. Confirmed by doing it — a
manual verification invocation on 2026-08-26T18:21Z replaced the 08-22 payload.

Persisting on the skip path stays: it fixed the 2026-05-15 stale-file landmine.
The WINDOW was the defect.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from pipeline_common import ACTIVE_HORIZON_DAYS

_COLS = (
    "symbol, prediction_date, predicted_direction, prediction_confidence, "
    "p_up, p_flat, p_down, score_modifier_applied, actual_5d_return, "
    "correct_5d, actual_log_alpha, horizon_days, correct"
)


def _make_db(tmp_path, rows):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE predictor_outcomes ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, prediction_date TEXT, "
        "predicted_direction TEXT, prediction_confidence REAL, p_up REAL, "
        "p_flat REAL, p_down REAL, score_modifier_applied REAL, "
        "actual_5d_return REAL, correct_5d INTEGER, actual_log_alpha REAL, "
        "horizon_days REAL, correct INTEGER)"
    )
    conn.executemany(
        f"INSERT INTO predictor_outcomes ({_COLS}) VALUES ({','.join('?' * 13)})",
        rows,
    )
    conn.commit()
    conn.close()
    return str(db)


def _row(i, pred_date, *, p_up, alpha):
    return (
        f"T{i}", pred_date, "UP", 0.6, p_up, 0.1, 1.0 - p_up - 0.1, 0.0,
        alpha * 100.0, 1, alpha, float(ACTIVE_HORIZON_DAYS), 1,
    )


@pytest.fixture()
def _no_s3(monkeypatch):
    from analysis import production_health

    captured: dict = {}

    class _FakeS3:
        def put_object(self, **kw):
            import json
            captured.update(json.loads(kw["Body"].decode()))

    monkeypatch.setattr(
        production_health.boto3, "client", lambda *_a, **_k: _FakeS3()
    )
    return captured


# ── The derivation ───────────────────────────────────────────────────────────


def test_default_lookback_is_derived_not_thirty():
    from analysis.production_health import _default_lookback_days

    got = _default_lookback_days()
    assert got != 30, (
        "the lookback is still the hardcoded 30 — graded rows at a "
        f"{ACTIVE_HORIZON_DAYS}-trading-day horizon cannot stay inside it "
        "(alpha-engine-config-I8701)"
    )
    expected = int(ACTIVE_HORIZON_DAYS * 7 / 5) + 1 + 30
    assert got == expected


def test_both_producers_derive_the_same_window():
    """The two realized-health producers must not disagree about how far back
    'recent' reaches — that disagreement is the whole of -I8701."""
    import inspect

    from analysis.production_health import _default_lookback_days
    import pipeline_common

    sibling = inspect.getsource(pipeline_common.push_predictor_rolling_metrics)
    assert "grade_window_calendar + 30" in sibling
    expected = int(ACTIVE_HORIZON_DAYS * 7 / 5) + 1 + 30
    assert _default_lookback_days() == expected


# ── The behaviour that actually broke ────────────────────────────────────────


def test_rows_survive_four_days_of_window_drift(tmp_path, _no_s3):
    """The exact 2026-08-22 -> 2026-08-26 regression, reproduced.

    Graded rows sit ~30 calendar days back because that is when their label
    closed. Under the old 30-day window a run four days later saw none of them.
    """
    from analysis.production_health import compute_production_health

    # Rows whose labels closed ~32 days before the FIRST run date.
    first_run = datetime(2026, 8, 22)
    later_run = datetime(2026, 8, 26)
    pred_date = (first_run - timedelta(days=32)).strftime("%Y-%m-%d")
    rows = [_row(i, pred_date, p_up=0.5 + i * 0.01, alpha=0.001 * i) for i in range(25)]
    db = _make_db(tmp_path, rows)

    for run in (first_run, later_run):
        result = compute_production_health(
            db, "test-bucket", run_date=run.strftime("%Y-%m-%d")
        )
        assert result.get("status") != "skipped", (
            f"run {run:%Y-%m-%d} reported {result.get('reason')!r} — the window "
            "slid off every graded row, and this payload would OVERWRITE a good "
            "one in S3 (alpha-engine-config-I8701)"
        )
        assert result["n_resolved"] == 25


def test_the_old_thirty_day_window_still_drops_them(tmp_path, _no_s3):
    """The companion that proves the guard above can fail: pass the legacy 30
    explicitly and the later run goes blind again."""
    from analysis.production_health import compute_production_health

    first_run = datetime(2026, 8, 22)
    later_run = datetime(2026, 8, 26)
    pred_date = (first_run - timedelta(days=32)).strftime("%Y-%m-%d")
    rows = [_row(i, pred_date, p_up=0.5 + i * 0.01, alpha=0.001 * i) for i in range(25)]
    db = _make_db(tmp_path, rows)

    result = compute_production_health(
        db, "test-bucket", run_date=later_run.strftime("%Y-%m-%d"), lookback_days=30
    )
    assert result.get("status") == "skipped"


def test_an_explicit_lookback_is_still_honoured(tmp_path, _no_s3):
    """Callers that pass a window keep getting it — the change is to the
    DEFAULT, not to the contract."""
    from analysis.production_health import compute_production_health

    run = datetime(2026, 8, 26)
    pred_date = (run - timedelta(days=5)).strftime("%Y-%m-%d")
    rows = [_row(i, pred_date, p_up=0.5 + i * 0.01, alpha=0.001 * i) for i in range(12)]
    db = _make_db(tmp_path, rows)

    result = compute_production_health(
        db, "test-bucket", run_date=run.strftime("%Y-%m-%d"), lookback_days=90
    )
    assert result["lookback_days"] == 90
