"""alpha-engine-config-I8704 — /tmp exhaustion on the predictor health check.

Found by RUNNING the freshly un-paused detector, not by reading it. Live
CloudWatch, 2026-08-26:

    17:51:34Z  dry_run=True   Downloaded research.db (355.9 MB) to /tmp/research.db
    17:55:09Z  dry_run=False  ERROR Failed to download research.db:
                              [Errno 28] No space left on device

Lambda's default `/tmp` is 512 MB. research.db was 373,198,848 bytes (356 MB)
and grows weekly, so a warm container still holding a previous copy has no room
for boto3's managed download of a second.

Two independent defects, both fixed here:

1. The canary downloaded a 356 MB database it never reads. Under `dry_run` both
   compute phases short-circuit and the email is skipped, so the download was
   pure waste — and it was what left the poisoned warm container behind.
2. `_download_research_db` assumed an empty `/tmp`. A warm container must never
   need 2x the file on disk.

`deploy_health.sh` additionally raises ephemeral storage to 2048 MB. That is
headroom for growth, not the fix — the two changes above make the current size
work on their own, which is why they are tested here and the size is not.
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_HANDLER = Path(__file__).resolve().parent.parent / "lambda_health" / "handler.py"


@pytest.fixture(autouse=True)
def _mock_preflight():
    """Same short-circuit as tests/test_health_handler.py — these tests
    exercise handler orchestration, not the preflight."""
    with patch("preflight.BacktesterPreflight") as mock_pf:
        mock_pf.return_value.run = MagicMock()
        yield mock_pf


# ── 1. The canary must not download the database ─────────────────────────────


@patch("lambda_health.handler._load_last_feature_drift", return_value=None)
@patch("lambda_health.handler._download_research_db")
@patch("lambda_health.handler.compute_calibration_validation", create=True)
@patch("lambda_health.handler.compute_production_health", create=True)
def test_dry_run_does_not_download_research_db(
    mock_health, mock_cal, mock_download, mock_drift
):
    from lambda_health.handler import handler

    result = handler({"dry_run": True}, None)

    mock_download.assert_not_called(), (
        "the canary downloaded 356 MB it never reads, and left it in a warm "
        "/tmp where it broke the next real invocation "
        "(alpha-engine-config-I8704)"
    )
    # Still a clean canary — the deploy gates on this.
    assert result["statusCode"] == 200


@patch("lambda_health.handler._load_last_feature_drift", return_value=None)
@patch("lambda_health.handler._download_research_db", return_value=None)
def test_a_real_run_still_fails_loudly_when_the_download_fails(mock_db, mock_drift):
    """Skipping the download under dry_run must not weaken the real path: a
    failed download is still a 500, never a misleading 200."""
    from lambda_health.handler import handler

    result = handler({}, None)
    assert result["statusCode"] == 500
    assert "research.db" in result["body"]


# ── 2. A warm container must not need two copies ─────────────────────────────


def test_download_removes_a_stale_copy_first(tmp_path, monkeypatch):
    """The warm-container case that actually failed in production."""
    import boto3 as _boto3

    import lambda_health.handler as h

    stale = tmp_path / "research.db"
    stale.write_bytes(b"x" * 4096)  # stands in for the 356 MB copy

    seen: dict = {}

    class _FakeS3:
        def download_file(self, bucket, key, path):
            # By the time boto3 is asked for the file, the stale copy must be
            # gone — that is the whole property under test.
            seen["existed_at_download"] = os.path.exists(path)
            Path(path).write_bytes(b"y" * 8192)

    monkeypatch.setattr(_boto3, "client", lambda *_a, **_k: _FakeS3())

    out = h._download_research_db("test-bucket", db_path=str(stale))

    assert out == str(stale)
    assert seen.get("existed_at_download") is False, (
        "_download_research_db called boto3 while a stale copy was still on "
        "disk — on Lambda that needs 2x356 MB in a 512 MB /tmp and dies with "
        "[Errno 28] (alpha-engine-config-I8704)"
    )


def test_download_default_path_is_unchanged():
    import lambda_health.handler as h

    assert h._RESEARCH_DB_PATH == "/tmp/research.db"


def test_download_source_removes_before_downloading():
    """Structural companion to the behavioural test above: the removal must
    precede the download call, not merely exist somewhere in the function."""
    import inspect

    import lambda_health.handler as h

    src = inspect.getsource(h._download_research_db)
    assert "os.remove(db_path)" in src, (
        "no stale-copy removal in _download_research_db "
        "(alpha-engine-config-I8704)"
    )
    assert src.index("os.remove(db_path)") < src.index("download_file"), (
        "the stale copy must be removed BEFORE the download is started; "
        "removing it afterwards frees nothing at the moment it is needed"
    )


# ── 3. Headroom is declared in the deploy script, not set by hand ────────────


def test_deploy_declares_ephemeral_storage():
    """A console-only ephemeral-storage change does not survive a function
    replacement. It belongs in the deploy script."""
    script = (
        Path(__file__).resolve().parent.parent / "infrastructure" / "deploy_health.sh"
    ).read_text()
    assert "--ephemeral-storage" in script
    assert '"Size": 2048' in script
    # And it must be applied before the canary runs, so the canary exercises
    # the configuration the live alias will get.
    assert script.index("--ephemeral-storage") < script.index("Running canary")
