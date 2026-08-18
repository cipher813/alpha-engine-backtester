"""Regression guard: portfolio_stats.json must never carry a stringified
pandas Series (alpha-engine-config-I7639).

``daily_returns`` / ``daily_log_returns`` are pandas Series in the in-memory
``portfolio_stats`` dict. ``json.dumps(..., default=str)`` cannot encode a
Series natively, falls back to ``str(series)``, and pandas truncates a long
Series' ``repr`` with a literal ``"..."`` in the middle — measured on
``s3://alpha-engine-research/backtest/2026-08-14/portfolio_stats.json``:
``daily_returns`` was a 246-character string with ``"..."`` in it, not data.

The real artifact is ``simulate/portfolio_daily_returns.parquet``
(config-I7616); ``evaluate.py`` reads that, not the JSON keys. The fix drops
both keys before either JSON write (the phase checkpoint in ``backtest.py``'s
simulate block, and the evaluator-facing copy in
``_export_simulation_artifacts``) rather than serializing them correctly —
two representations of one series is how they drift.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

import backtest


class _FakeS3:
    def __init__(self):
        self.store: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, ContentType=None):
        self.store[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, *, Bucket, Key):  # pragma: no cover - unused here
        raise NotImplementedError


def _stats_with_series() -> dict:
    idx = pd.date_range("2026-06-01", periods=400, freq="D")
    return {
        "status": "ok",
        "total_return": 0.05,
        "sharpe_ratio": 1.2,
        "daily_returns": pd.Series(range(400), index=idx, dtype=float),
        "daily_log_returns": pd.Series(range(400), index=idx, dtype=float),
    }


def test_export_simulation_artifacts_drops_the_series_keys(monkeypatch):
    """The evaluator-facing ``backtest/{date}/portfolio_stats.json`` write —
    the exact artifact the issue measured — must not carry either Series."""
    fake_s3 = _FakeS3()
    monkeypatch.setattr(backtest.boto3, "client", lambda *a, **k: fake_s3)

    backtest._export_simulation_artifacts(
        {"output_bucket": "b"}, "2026-08-14", portfolio_stats=_stats_with_series(),
    )

    body = json.loads(fake_s3.store["backtest/2026-08-14/portfolio_stats.json"])
    assert "daily_returns" not in body
    assert "daily_log_returns" not in body
    # The scalar fields must survive untouched.
    assert body["total_return"] == 0.05
    assert body["sharpe_ratio"] == 1.2


def test_export_simulation_artifacts_would_have_written_a_truncated_repr(monkeypatch):
    """Reproduces the pre-fix defect directly against the raw json.dumps call
    the old code used, so the hazard is not merely asserted from memory."""
    stats = _stats_with_series()
    body = json.dumps(stats, indent=2, default=str)
    parsed = json.loads(body)
    assert isinstance(parsed["daily_returns"], str), (
        "a Series reaching json.dumps(default=str) must serialize as a "
        "string repr, not real data — this is the defect this file guards "
        "against"
    )
    assert "..." in parsed["daily_returns"], (
        "pandas truncates a >~60-row Series repr with a literal '...' — the "
        "measured shape of the corrupted artifact"
    )
