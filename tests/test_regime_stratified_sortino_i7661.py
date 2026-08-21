"""alpha-engine-config-I7661 — the T2 Sortino runner, end to end.

What was measured on live state before this fix (2026-08-21):

* ``s3://alpha-engine-research/regime/stratified_sortino/latest.json``
  (run_id 2608151626, trading_day 2026-08-14) carried ``spread_10d`` /
  ``spread_30d``, both ``insufficient_sample``, with the ONLY populated
  stratum being ``caution`` — a regime label the 3-class taxonomy retired in
  May — at ``mean_log_alpha = -6.436916136981``.
* ``research.db``: 34 rows carry a paired ``return_10d`` + ``spy_10d_return``,
  all dated 2026-03-04..03-13. The long store ``score_performance_outcomes``
  holds no horizon-30 rows at all and 534 horizon-21 rows with ``log_alpha``
  populated, through 2026-07-10.
* ``crucible-dashboard/views/15_Regime.py`` reads ``spread_21d`` /
  ``spread_5d`` — keys the producer never wrote.

Three regressions are pinned: the runner declares its units, the artifact
carries a policy-derived shape the dashboard can read, and an artifact built
from frozen inputs reports ``unmeasurable`` rather than a fresh-looking
success.
"""

from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from analysis.regime_stratified_sortino import ReturnUnits, ReturnUnitsError
from analysis.regime_stratified_sortino_runner import (
    OUTCOME_RETURN_UNITS,
    run_regime_stratified_sortino,
)


class _FakeS3:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()
        return {}

    def get_object(self, *, Bucket, Key):
        return {"Body": BytesIO(self._objects[(Bucket, Key)])}


def _db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("""
            CREATE TABLE score_performance (
                id INTEGER PRIMARY KEY, ticker TEXT, score_date TEXT,
                market_regime TEXT,
                return_21d REAL, return_5d REAL,
                spy_21d_return REAL, spy_5d_return REAL,
                beat_spy_21d INTEGER, beat_spy_5d INTEGER
            )
        """)
        cols = ["ticker", "score_date", "market_regime", "return_21d", "return_5d",
                "spy_21d_return", "spy_5d_return", "beat_spy_21d", "beat_spy_5d"]
        for r in rows:
            conn.execute(
                f"INSERT INTO score_performance ({','.join(cols)}) "
                f"VALUES ({','.join('?' * len(cols))})",
                tuple(r.get(c) for c in cols),
            )
        conn.commit()
    finally:
        conn.close()


def _rows(*, end: str, n: int = 30) -> list[dict]:
    """PERCENT-point rows — the units attach_outcomes emits."""
    dates = pd.date_range(end=pd.Timestamp(end), periods=n, freq="D")
    out = []
    for regime, base in (("bull", 3.0), ("bear", -2.5)):
        for i, d in enumerate(dates):
            out.append({
                "ticker": f"{regime[:2].upper()}{i}",
                "score_date": d.date().isoformat(),
                "market_regime": regime,
                "return_21d": round(base + (i % 6) * 0.3, 2),
                "return_5d": round(base / 3 + (i % 6) * 0.1, 2),
                "spy_21d_return": 1.2,
                "spy_5d_return": 0.4,
                "beat_spy_21d": int(base > 1.2),
                "beat_spy_5d": int(base > 1.2),
            })
    return out


def _run(db_path: Path, s3: _FakeS3):
    with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
        mock_boto3.client.return_value = s3
        return run_regime_stratified_sortino(
            db_path=str(db_path), s3_bucket="test-bucket",
        )


# ── 1. The units are declared, and they are the ones the source uses ────────


def test_runner_declares_percent_units():
    """`attach_outcomes` reproduces round(decimal*100, 2). A runner that
    declared FRACTION would be the live defect verbatim."""
    assert OUTCOME_RETURN_UNITS is ReturnUnits.PERCENT


def test_percent_rows_produce_ordinary_small_log_alphas(tmp_path):
    """Guard-fails-without-the-fix: the pre-I7661 code fed these same rows to
    log(1 + r) as fractions. Every bear pick (-2.5pp) drove 1+r negative,
    clipped to 1e-9, and produced log_alpha ~= -20.7 — which is how the live
    artifact came to publish a mean of -6.44."""
    db = tmp_path / "research.db"
    _db(db, _rows(end="2026-07-10"))
    result = _run(db, _FakeS3())

    strata = {(s["market_regime"], s["horizon_days"]): s
              for s in result["payload"]["strata"]}
    for (_regime, _h), s in strata.items():
        if s["n_picks"] >= 20:
            assert -0.20 < s["mean_log_alpha"] < 0.20, s
            assert s["mean_log_alpha"] > -6.0


def test_fraction_rows_are_refused_because_the_runner_declares_percent(tmp_path):
    """A source swap back to decimals must BREAK the run, not quietly change
    what every published Sortino means."""
    db = tmp_path / "research.db"
    rows = _rows(end="2026-07-10")
    for r in rows:
        for c in ("return_21d", "return_5d", "spy_21d_return", "spy_5d_return"):
            r[c] = r[c] / 100.0
    _db(db, rows)
    with pytest.raises(ReturnUnitsError):
        _run(db, _FakeS3())


# ── 2. The artifact shape the dashboard actually reads ──────────────────────


def test_artifact_keys_match_the_dashboard_consumer(tmp_path):
    db = tmp_path / "research.db"
    _db(db, _rows(end="2026-07-10"))
    s3 = _FakeS3()
    result = _run(db, s3)
    payload = result["payload"]

    assert set(payload) >= {"spread_21d", "spread_5d", "horizons",
                            "input_window", "status", "status_reason"}
    assert "spread_10d" not in payload
    assert "spread_30d" not in payload
    assert payload["horizons"] == [21, 5]
    # And the same body reached both S3 keys.
    bodies = {json.dumps(json.loads(v), sort_keys=True) for v in s3._objects.values()}
    assert len(bodies) == 1


def test_strata_are_populated_not_a_well_formed_empty(tmp_path):
    db = tmp_path / "research.db"
    _db(db, _rows(end="2026-07-10"))
    result = _run(db, _FakeS3())
    populated = [s for s in result["payload"]["strata"] if s["n_picks"] > 0]
    assert populated, "every stratum was empty — the metric measured nothing"
    assert {s["market_regime"] for s in populated} == {"bull", "bear"}
    assert {s["horizon_days"] for s in populated} == {21, 5}


# ── 3. Freshness on the inputs ──────────────────────────────────────────────


def test_frozen_march_inputs_report_unmeasurable(tmp_path, caplog):
    """The live condition: rows that stop in March, republished every week.
    No write-time check can see this; an input-date predicate can."""
    db = tmp_path / "research.db"
    _db(db, _rows(end="2026-03-13"))
    with caplog.at_level("ERROR"):
        result = _run(db, _FakeS3())

    assert result["status"] == "unmeasurable"
    assert "2026-03-13" in result["status_reason"]
    assert result["payload"]["status"] == "unmeasurable"
    assert result["payload"]["input_window"]["max_score_date"] == "2026-03-13"
    assert any("UNMEASURABLE" in r.getMessage() for r in caplog.records)


def test_a_normally_lagged_window_is_not_flagged(tmp_path):
    """A horizon-21 outcome cannot resolve for 21 trading days, so a healthy
    run's newest input is already a month behind. Flagging that would be the
    mirror defect (champion-challenger-policy §7.1)."""
    db = tmp_path / "research.db"
    recent = (pd.Timestamp.today().normalize() - pd.Timedelta(days=30)).date().isoformat()
    _db(db, _rows(end=recent))
    result = _run(db, _FakeS3())
    assert result["status"] == "ok"
    assert result["status_reason"] == ""
