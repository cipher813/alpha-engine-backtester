"""Tests for analysis/regime_stratified_sortino_runner.py — T2 pipeline wiring.

Exercises the end-to-end runner (load score_performance → stratify →
spread → assemble → publish canonical eval-artifact) against
synthetic SQLite DBs + in-memory S3 stubs. No real boto3.

The compute layer (regime_stratified_sortino.py) is covered by
test_regime_stratified_sortino.py — these tests focus on the
runner-specific pieces: artifact write shape, S3 prefix correctness,
graceful handling of empty / pre-migration DBs, and the dry-run path.
"""
from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from unittest.mock import patch



from analysis.regime_stratified_sortino_runner import (
    REGIME_STRATIFIED_SORTINO_PREFIX,
    run_regime_stratified_sortino,
)


class _FakeS3:
    """In-memory boto3 S3 stub."""

    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None) -> dict:
        self._objects[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode("utf-8")
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict:
        if (Bucket, Key) not in self._objects:
            raise KeyError(Key)
        return {"Body": BytesIO(self._objects[(Bucket, Key)])}


def _create_score_perf_db(path: Path, rows: list[dict]) -> None:
    """Create a synthetic score_performance table populated with rows."""
    conn = sqlite3.connect(path)
    try:
        # Policy horizons (21 primary / 5 diagnostic), and returns in the
        # PERCENT-point convention attach_outcomes emits — the units the
        # runner declares (alpha-engine-config-I7661).
        conn.execute("""
            CREATE TABLE score_performance (
                id INTEGER PRIMARY KEY,
                ticker TEXT,
                score_date TEXT,
                eval_date_21d TEXT,
                eval_date_5d TEXT,
                market_regime TEXT,
                return_21d REAL,
                return_5d REAL,
                spy_21d_return REAL,
                spy_5d_return REAL,
                beat_spy_21d INTEGER,
                beat_spy_5d INTEGER
            )
        """)
        cols = [
            "ticker", "score_date", "eval_date_21d", "eval_date_5d",
            "market_regime", "return_21d", "return_5d",
            "spy_21d_return", "spy_5d_return",
            "beat_spy_21d", "beat_spy_5d",
        ]
        placeholders = ",".join(["?"] * len(cols))
        for r in rows:
            conn.execute(
                f"INSERT INTO score_performance ({','.join(cols)}) VALUES ({placeholders})",
                tuple(r.get(c) for c in cols),
            )
        conn.commit()
    finally:
        conn.close()


def _seed_well_populated_rows(*, end: str | None = None) -> list[dict]:
    """Enough rows to clear DEFAULT_MIN_PICKS_PER_STRATUM in bull + bear.

    Values are PERCENT POINTS — the convention ``attach_outcomes`` reproduces
    from the long store and the one the runner declares. Seeding fractions
    here would make every test pass against a runner that lies about its units.

    ``end`` is relative to the REAL wall clock (``date.today()``), not a
    hardcoded literal — the runner's own freshness gate compares its newest
    measured input against ``now_dual().trading_day`` (a live date), so a
    fixed literal is a ticking time bomb: this suite passed for weeks against
    a hardcoded ``"2026-07-13"`` and then failed by itself, with no code
    change, once wall-clock date crossed the gate's 30d-horizon + 14d-grace
    allowance (measured failing 2026-08-27, `alpha-engine-config-I8769`
    incidental find while unrelated backtester work was in flight). 10 days
    back leaves comfortable headroom under that allowance regardless of when
    this suite runs.
    """
    if end is None:
        end = (date.today() - timedelta(days=10)).isoformat()
    rows: list[dict] = []
    base_dates = pd.date_range(end=pd.Timestamp(end), periods=80, freq="W-MON")
    for i, d in enumerate(base_dates[:40]):
        rows.append({
            "ticker": f"BU{i}",
            "score_date": d.date().isoformat(),
            "eval_date_21d": (d + pd.Timedelta(days=21)).date().isoformat(),
            "eval_date_5d": (d + pd.Timedelta(days=5)).date().isoformat(),
            "market_regime": "bull",
            "return_21d": round(4.0 + (i % 7) * 0.2, 2),
            "return_5d": round(1.0 + (i % 7) * 0.1, 2),
            "spy_21d_return": 1.5,
            "spy_5d_return": 0.4,
            "beat_spy_21d": 1,
            "beat_spy_5d": 1,
        })
    for i, d in enumerate(base_dates[40:]):
        rows.append({
            "ticker": f"BE{i}",
            "score_date": d.date().isoformat(),
            "eval_date_21d": (d + pd.Timedelta(days=21)).date().isoformat(),
            "eval_date_5d": (d + pd.Timedelta(days=5)).date().isoformat(),
            "market_regime": "bear",
            "return_21d": round(-3.0 + (i % 5) * 0.2, 2),
            "return_5d": round(-0.8 + (i % 5) * 0.1, 2),
            "spy_21d_return": -1.0,
            "spy_5d_return": -0.3,
            "beat_spy_21d": 0,
            "beat_spy_5d": 0,
        })
    return rows


# pandas imported here to avoid clutter at top
import pandas as pd  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# run_regime_stratified_sortino — end-to-end
# ─────────────────────────────────────────────────────────────────────


class TestRunRegimeStratifiedSortino:
    def test_well_populated_db_produces_artifact(self, tmp_path):
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )

        assert result["status"] == "ok"
        assert result["wrote"] is True
        assert result["artifact_key"].startswith("regime/stratified_sortino/")
        assert result["latest_key"] == "regime/stratified_sortino/latest.json"
        # Both bull + bear strata × 2 horizons = at least 4 (depending on
        # how many of the supported horizons land)
        assert result["n_strata"] >= 4

    def test_artifact_carries_t2_schema(self, tmp_path):
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )
        payload = result["payload"]
        assert payload["schema_version"] == 1
        assert payload["eval_tier"] == "T2_downstream_stratified_sortino"
        # Policy-derived spread keys (alpha-engine-config-I7661) — the names
        # crucible-dashboard's views/15_Regime.py has always read.
        assert "spread_21d" in payload
        assert "spread_5d" in payload
        assert "spread_10d" not in payload
        assert "spread_30d" not in payload
        assert payload["horizons"] == [21, 5]
        assert "strata" in payload
        assert "method_metadata" in payload
        # Freshness is recorded on the INPUTS, not the write time.
        assert "input_window" in payload
        assert payload["input_window"]["n_rows"] > 0
        assert payload["method_metadata"]["return_units"] == "percent"

    def test_artifact_body_matches_latest_sidecar(self, tmp_path):
        """T2's sidecar mirrors the artifact body — small enough that
        duplication is fine + simpler consumer. Pin so a refactor doesn't
        silently divergent the two."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )

        artifact_keys = [
            k for (b, k) in s3._objects.keys()
            if b == "test-bucket" and k.startswith("regime/stratified_sortino/")
        ]
        # Two writes — forensic + latest sidecar
        assert "regime/stratified_sortino/latest.json" in artifact_keys
        artifact_key = next(
            k for k in artifact_keys
            if k != "regime/stratified_sortino/latest.json"
        )
        artifact_body = s3._objects[("test-bucket", artifact_key)]
        latest_body = s3._objects[("test-bucket", "regime/stratified_sortino/latest.json")]
        assert artifact_body == latest_body

    def test_no_write_when_bucket_missing(self, tmp_path):
        """Without s3_bucket, runner skips the write but still returns
        the payload — useful for ad-hoc CLI replays via the spot or local."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        result = run_regime_stratified_sortino(
            db_path=str(db), s3_bucket=None,
        )
        assert result["status"] == "ok"
        assert result["wrote"] is False
        assert "payload" in result
        assert "artifact_key" not in result

    def test_dry_run_no_write(self, tmp_path):
        """``write=False`` returns the payload without touching S3 even
        if a bucket is supplied — useful for unit/integration tests."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, _seed_well_populated_rows())
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket", write=False,
            )
        assert result["wrote"] is False
        assert s3._objects == {}

    def test_empty_db_returns_placeholder_payload(self, tmp_path):
        """Pre-data state: score_performance exists but is empty. Runner
        emits a payload with empty strata / null spread — must NOT crash
        so the evaluator's tracker can mark it 'no_data' gracefully."""
        db = tmp_path / "research.db"
        _create_score_perf_db(db, rows=[])
        s3 = _FakeS3()
        with patch("analysis.regime_stratified_sortino_runner.boto3") as mock_boto3:
            mock_boto3.client.return_value = s3
            result = run_regime_stratified_sortino(
                db_path=str(db), s3_bucket="test-bucket",
            )
        # An empty DB measured NOTHING. Reporting "ok" here is the defect
        # alpha-engine-config-I7661 closes: a well-formed artifact containing
        # no measurement must not render as an empty success
        # (champion-challenger-policy §7.2).
        assert result["status"] == "unmeasurable"
        assert "nothing was measured" in result["status_reason"]
        assert result["payload"]["status"] == "unmeasurable"
        assert result["wrote"] is True
        assert result["n_strata"] == 0
        assert result["input_window"] == {
            "min_score_date": None, "max_score_date": None, "n_rows": 0,
        }
        # Spread is "insufficient_sample" with no strata
        assert result["spread_21d_interpretation"] == "insufficient_sample"
        assert result["spread_5d_interpretation"] == "insufficient_sample"

    def test_prefix_constant_is_canonical(self):
        """The prefix anchors the dashboard reader + judge auditor; pin
        the value so a refactor can't silently move the artifact."""
        assert REGIME_STRATIFIED_SORTINO_PREFIX == "regime/stratified_sortino"


# ─────────────────────────────────────────────────────────────────────
# evaluate.py wiring — pin the module is registered
# ─────────────────────────────────────────────────────────────────────


def test_evaluate_imports_regime_stratified_sortino_runner():
    """Catch a refactor that removes the import or moves the runner —
    silent drift in evaluate.py would leave the eval running but the
    T2 module silently missing from the Saturday eval results."""
    import evaluate
    assert hasattr(evaluate, "regime_stratified_sortino_runner")


def test_evaluate_diagnostic_includes_t2_module():
    """The T2 module hook must appear inside _run_diagnostics — pinned
    by searching the source for the registry key. Catches accidental
    removal during evaluate.py refactors."""
    src = Path(__file__).resolve().parents[1] / "evaluate.py"
    body = src.read_text()
    assert '"regime_stratified_sortino"' in body, (
        "evaluate.py must register the regime_stratified_sortino module "
        "via tracker.run_module so it runs each Saturday alongside the "
        "other diagnostics. Wire it into _run_diagnostics."
    )
    assert "regime_stratified_sortino_runner.run_regime_stratified_sortino" in body
