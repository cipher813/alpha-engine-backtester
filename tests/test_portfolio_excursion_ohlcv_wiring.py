"""Tests for evaluate.py's OHLCV wiring into portfolio_excursion (config-I7600).

`config.get("_ohlcv_by_ticker")` was never written by anything in this repo —
a full-repo grep confirmed the read at evaluate.py:2440 was the only hit. The
observable consequence: `portfolio_excursion.json` carried
`status: "insufficient_data", reason: "no ohlc data"` on every date, and that
reason string is indistinguishable from a genuine transient data gap.

This file covers the NEW wiring — `_load_ohlcv_for_excursion`, which reads
the OHLCV the backtester already persisted for this date's simulation via
the `simulation_setup` phase marker (same S3 `.phases/` namespace
`PhaseRegistry` reads elsewhere in evaluate.py), never re-fetching — plus an
integration-level fixture proving `compute_portfolio_excursion_summary`
resolves to `status: "ok"` end-to-end, which config-I7600 deliverable 4 asks
for: the analysis itself was already covered in isolation
(tests/test_team_skill_metrics.py), but nothing had ever exercised it wired
up to real-shaped OHLC data end to end.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


def _ohlc_df(n_days: int = 15, start: str = "2026-01-05") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "open": [100.0 + i for i in range(n_days)],
            "high": [105.0 + i for i in range(n_days)],
            "low": [95.0 + i for i in range(n_days)],
            "close": [101.0 + i for i in range(n_days)],
        },
        index=idx,
    )


class TestLoadOhlcvForExcursion:
    def test_marker_missing_is_insufficient_data_not_defect(self):
        """No simulation_setup marker at all (e.g. a date predating
        config-I7600, or the phase genuinely never ran) is a legitimate
        upstream data gap, not a wiring defect — must be tagged accordingly
        so an operator reading logs doesn't chase a phantom bug."""
        from evaluate import _load_ohlcv_for_excursion

        registry = MagicMock()
        registry.load_marker.return_value = None

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is None
        assert reason.startswith("insufficient_data:")
        registry.load_marker.assert_called_once_with("simulation_setup")

    def test_marker_read_error_is_a_defect(self):
        from evaluate import _load_ohlcv_for_excursion

        registry = MagicMock()
        registry.load_marker.side_effect = RuntimeError("S3 unreachable")

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is None
        assert reason.startswith("defect:")

    def test_marker_present_without_ohlcv_artifact_key_is_a_defect(self):
        """The marker exists (simulation_setup ran) but carries no
        ohlcv_by_ticker key — this IS the shape of the original config-I7600
        bug: an upstream artifact that should exist but the read path can't
        find. Loud, not silently 'no ohlc data'."""
        from evaluate import _load_ohlcv_for_excursion

        registry = MagicMock()
        registry.load_marker.return_value = {
            "artifact_keys": ["backtest/2026-08-14/simulation_setup/price_matrix.parquet"],
        }

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is None
        assert reason.startswith("defect:")

    def test_load_failure_is_a_defect(self, monkeypatch):
        from evaluate import _load_ohlcv_for_excursion
        import phase_artifacts

        registry = MagicMock()
        registry.load_marker.return_value = {
            "artifact_keys": ["backtest/2026-08-14/simulation_setup/ohlcv_by_ticker.parquet"],
        }
        registry.s3_client = MagicMock()

        monkeypatch.setattr(
            phase_artifacts, "load_ohlcv_by_ticker",
            MagicMock(side_effect=RuntimeError("corrupt parquet")),
        )

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is None
        assert reason.startswith("defect:")

    def test_empty_persisted_ohlc_is_insufficient_data_not_defect(self, monkeypatch):
        """The wiring worked — the marker + artifact key + load all succeeded
        — but the persisted set was genuinely empty. That is a real data
        gap, not a defect in this read path."""
        from evaluate import _load_ohlcv_for_excursion
        import phase_artifacts

        registry = MagicMock()
        registry.load_marker.return_value = {
            "artifact_keys": ["backtest/2026-08-14/simulation_setup/ohlcv_by_ticker.parquet"],
        }
        registry.s3_client = MagicMock()

        monkeypatch.setattr(
            phase_artifacts, "load_ohlcv_by_ticker", MagicMock(return_value={}),
        )

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is None
        assert reason.startswith("insufficient_data:")

    def test_success_returns_ohlc_and_no_reason(self, monkeypatch):
        from evaluate import _load_ohlcv_for_excursion
        import phase_artifacts

        registry = MagicMock()
        registry.load_marker.return_value = {
            "artifact_keys": ["backtest/2026-08-14/simulation_setup/ohlcv_by_ticker.parquet"],
        }
        registry.s3_client = MagicMock()
        real_ohlc = {"AAPL": _ohlc_df()}

        monkeypatch.setattr(
            phase_artifacts, "load_ohlcv_by_ticker", MagicMock(return_value=real_ohlc),
        )

        ohlc, reason = _load_ohlcv_for_excursion({}, registry)
        assert ohlc is real_ohlc
        assert reason is None


class TestPortfolioExcursionEndToEndWithRealisticFixture:
    """config-I7600 deliverable 4: a test asserting portfolio_excursion
    produces status: 'ok' on a realistic date fixture — coverage that never
    existed because nothing had ever run this analysis with wired data."""

    def test_realistic_fixture_yields_status_ok(self):
        from analysis.team_skill_metrics import compute_portfolio_excursion_summary

        tickers = ["AAA", "BBB", "CCC", "DDD", "EEE"]
        picks = pd.DataFrame([
            {"symbol": t, "score_date": "2026-01-05", "score": 65 + i * 5,
             "beat_spy_10d": 1}
            for i, t in enumerate(tickers)
        ])
        ohlc = {t: _ohlc_df(n_days=20) for t in tickers}

        result = compute_portfolio_excursion_summary(
            picks, ohlc, horizon_days=10, score_threshold=60,
        )

        assert result["status"] == "ok"
        assert result["n"] == len(tickers)
        assert result["mean_mfe"] > 0
