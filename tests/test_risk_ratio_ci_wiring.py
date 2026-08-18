"""Tests for evaluate.py's risk_ratio_ci wiring (config#976 → config-I7616).

REPOINTED. The monitor used to be fed a portfolio sleeve rebuilt from
``e2e_lift.team_lift`` picks, because backtest.py's simulate stage produced —
in the old helper's own words — "a process-separate, JSON-only artifact that
does not carry raw series back to evaluate.py". Two facts made that wiring
worthless:

  * ``team_lift`` has been ``[]`` on every report card since 2026-07-17 (the
    six-team research graph was retired 2026-07-12), so the sleeve was empty on
    every run and the published CI was ``insufficient_data`` unconditionally;
  * ``config["_prices"]``, which the sleeve also needed, was read at
    evaluate.py and written by nothing (config-I7616).

``backtest.py`` now persists the deployed strategy's daily returns as
``simulate/portfolio_daily_returns.parquet``, so the monitor reads the series
``compute_risk_ratio_ci``'s docstring always named. This file covers the new
loaders; the pure computation stays covered by tests/test_risk_ratio_ci.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest


def _series(n: int = 200) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=n, freq="B")
    return pd.Series([0.001 * ((i % 7) - 3) for i in range(n)], index=idx,
                     name="daily_return")


# ── The generic phase-artifact loader ────────────────────────────────────────


class TestLoadPhaseArtifact:
    def test_marker_missing_is_insufficient_data_not_defect(self):
        from evaluate import _load_phase_artifact

        registry = MagicMock()
        registry.load_marker.return_value = None
        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=lambda *a, **k: None, label="x",
        )
        assert obj is None
        assert reason.startswith("insufficient_data:")
        registry.load_marker.assert_called_once_with("simulate")

    def test_marker_read_error_is_a_defect(self):
        from evaluate import _load_phase_artifact

        registry = MagicMock()
        registry.load_marker.side_effect = RuntimeError("s3 exploded")
        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=lambda *a, **k: None, label="x",
        )
        assert obj is None
        assert reason.startswith("defect:")
        assert "s3 exploded" in reason

    def test_missing_artifact_key_is_a_defect(self):
        """A marker that ran OK but carries no such artifact is OUR bug —
        the stage claimed success without producing what it promised."""
        from evaluate import _load_phase_artifact

        registry = MagicMock()
        registry.load_marker.return_value = {"artifact_keys": ["a/b/other.json"]}
        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=lambda *a, **k: None, label="x",
        )
        assert obj is None
        assert reason.startswith("defect:")

    def test_loader_raising_is_a_defect(self):
        from evaluate import _load_phase_artifact

        registry = MagicMock()
        registry.load_marker.return_value = {"artifact_keys": ["a/b/x.parquet"]}

        def boom(*a, **k):
            raise ValueError("corrupt parquet")

        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=boom, label="x",
        )
        assert obj is None
        assert reason.startswith("defect:")
        assert "corrupt parquet" in reason

    def test_empty_artifact_is_insufficient_data(self):
        from evaluate import _load_phase_artifact

        registry = MagicMock()
        registry.load_marker.return_value = {"artifact_keys": ["a/b/x.parquet"]}
        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=lambda *a, **k: pd.Series(dtype=float), label="x",
        )
        assert obj is None
        assert reason.startswith("insufficient_data:")

    def test_success_returns_the_artifact_and_no_reason(self):
        from evaluate import _load_phase_artifact

        s = _series(5)
        registry = MagicMock()
        registry.load_marker.return_value = {"artifact_keys": ["a/b/x.parquet"]}
        obj, reason = _load_phase_artifact(
            {}, registry, phase="simulate", suffix="/x.parquet",
            loader=lambda *a, **k: s, label="x",
        )
        assert reason is None
        pd.testing.assert_series_equal(obj, s)


# ── The three concrete loaders ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "fn_name,phase,suffix",
    [
        ("_load_price_matrix", "simulation_setup", "/price_matrix.parquet"),
        ("_load_spy_prices", "predictor_data_prep", "/spy_prices.parquet"),
        ("_load_portfolio_daily_returns", "simulate",
         "/portfolio_daily_returns.parquet"),
    ],
)
def test_each_loader_reads_the_phase_the_producer_writes(fn_name, phase, suffix):
    """Pins each loader to the phase/artifact ``backtest.py`` actually persists.

    A loader pointed at a phase nothing writes degrades exactly like the dead
    config keys did — silently, forever — so the pairing is asserted, not
    assumed.
    """
    import evaluate

    registry = MagicMock()
    registry.load_marker.return_value = None
    getattr(evaluate, fn_name)({}, registry)
    registry.load_marker.assert_called_once_with(phase)


def test_portfolio_daily_returns_round_trip_feeds_the_monitor():
    """End to end: the persisted series → compute_risk_ratio_ci → an ``ok`` card.

    The old wiring could never reach this state: the sleeve was empty on every
    run, so ``status`` was ``insufficient_data`` on every published artifact.
    """
    from analysis.risk_ratio_ci import compute_risk_ratio_ci
    from evaluate import _load_portfolio_daily_returns

    pf = _series(200)
    registry = MagicMock()
    registry.load_marker.return_value = {
        "artifact_keys": [
            "backtest/2026-08-14/.phases/simulate/portfolio_daily_returns.parquet",
        ],
    }
    # Substitute the parquet read; everything above it (marker lookup, key
    # match, reason classification) is the code under test.
    import phase_artifacts

    orig_load_series = phase_artifacts.load_series
    try:
        phase_artifacts.load_series = lambda bucket, key, *, s3_client=None: pf
        series, reason = _load_portfolio_daily_returns({}, registry)
    finally:
        phase_artifacts.load_series = orig_load_series

    assert reason is None
    pd.testing.assert_series_equal(series, pf)
    spy = pd.Series([0.0005] * len(pf), index=pf.index)
    result = compute_risk_ratio_ci(series, spy)
    assert result["status"] == "ok"
    assert result["n_samples"] == len(pf)
    assert set(result["ratios"]) == {"sharpe_ratio", "sortino_ratio", "information_ratio"}


def test_none_series_yields_insufficient_data_shape():
    from analysis.risk_ratio_ci import compute_risk_ratio_ci

    result = compute_risk_ratio_ci(None, None)
    assert result["status"] == "insufficient_data"
    assert result["all_magnitude_certain"] is False


# ── team_metrics retirement ──────────────────────────────────────────────────


class TestTeamMetricsRetirement:
    def test_retirement_record_is_declared_and_dated(self):
        """config-I7616: 'declared, with a date, not left emitting
        insufficient_data forever'. A reader must be able to tell a retired
        component from a broken one, from the artifact alone."""
        from evaluate import TEAM_METRICS_RETIREMENT

        assert TEAM_METRICS_RETIREMENT["status"] == "retired"
        assert TEAM_METRICS_RETIREMENT["retired_on"] == "2026-08-18"
        assert TEAM_METRICS_RETIREMENT["reason"]
        assert "7616" in TEAM_METRICS_RETIREMENT["tracker"]
        assert TEAM_METRICS_RETIREMENT["superseded_by"]

    def test_status_is_not_insufficient_data(self):
        from evaluate import TEAM_METRICS_RETIREMENT

        assert TEAM_METRICS_RETIREMENT["status"] != "insufficient_data"

    def test_the_team_lift_sleeve_helper_is_gone(self):
        """The helper aggregated team_lift picks into a portfolio sleeve. Its
        input has been permanently empty since 2026-07-17; keeping it would
        leave a second, dead definition of 'the portfolio return series'."""
        import evaluate

        assert not hasattr(evaluate, "_portfolio_daily_returns_from_team_lift")


# ── horizon sweep ────────────────────────────────────────────────────────────


def test_metric_bundle_horizon_is_the_policy_primary_not_a_literal_ten():
    """config-I7208: the producer never emitted a 10d horizon and the fleet
    objective is 21d. The bundle hardcoded ``horizon_days=10`` twice."""
    from nousergon_lib.quant.horizons import DEFAULT_POLICY

    from evaluate import _METRIC_BUNDLE_HORIZON_DAYS

    assert _METRIC_BUNDLE_HORIZON_DAYS == int(DEFAULT_POLICY.primary_horizon)
    assert _METRIC_BUNDLE_HORIZON_DAYS == 21


# ── producer/consumer contract for the new artifact ──────────────────────────


def test_simulate_phase_persists_the_series_the_monitor_reads():
    """The producer/consumer pair, pinned in one place (M0 contract rule).

    ``_load_portfolio_daily_returns`` matches on a key suffix. If backtest.py
    ever renames the artifact, or drops the ``save_series`` call, the loader
    degrades to ``insufficient_data:*`` forever and nothing else fails — which
    is exactly how ``config["_prices"]`` stayed dead for months.
    """
    import pathlib as _pathlib

    from phase_artifacts import artifact_key

    key = artifact_key("2026-08-14", "simulate", "portfolio_daily_returns", "parquet")
    assert key.endswith("/portfolio_daily_returns.parquet")

    repo_root = _pathlib.Path(__file__).resolve().parent.parent
    src = (repo_root / "backtest.py").read_text(encoding="utf-8")
    assert '"simulate", "portfolio_daily_returns"' in src, (
        "backtest.py's simulate phase no longer persists portfolio_daily_returns "
        "— evaluate._load_portfolio_daily_returns has no producer."
    )
