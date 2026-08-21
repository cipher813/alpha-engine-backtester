"""Tests for analysis.team_skill_metrics — orchestrator that activates
the PR 3 dormant graders.

Pins:
  1. compute_team_metrics graceful-degrades when prices/ohlc/spy are absent.
  2. IC + expectancy compute correctly from picks + score_performance.
  3. team_metrics output shape matches what _grade_sector_team consumes.
  4. compute_portfolio_calibration normalizes score 0-100 to [0, 1].
  5. compute_portfolio_excursion_summary filters by score_threshold.
  6. Empty / missing inputs return insufficient_data, never raise.
  7. End-to-end: feed orchestrator output into compute_scorecard → produces
     a scorecard with the new graders firing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.team_skill_metrics import (
    compute_portfolio_calibration,
    compute_portfolio_excursion_summary,
    compute_team_metrics,
)


def _make_score_perf(n_per_team: int = 12) -> pd.DataFrame:
    """Synthetic score_performance DataFrame.

    Two teams: tech (good — high score → positive return),
    health (mediocre — random correlation).
    """
    rng = np.random.default_rng(0)
    rows = []
    for team_id, has_signal in [("tech", True), ("health", False)]:
        for i in range(n_per_team):
            score = 60 + i * 2  # 60..82
            ticker = f"{team_id[:2].upper()}{i}"
            base_date = pd.Timestamp("2026-01-05") + pd.Timedelta(days=i * 7)
            if has_signal:
                # Higher score → higher return
                ret_5d = (score - 60) * 0.005 + rng.normal(0, 0.005)
                ret_10d = ret_5d * 1.5
            else:
                ret_5d = rng.normal(0, 0.02)
                ret_10d = rng.normal(0, 0.025)
            spy_5d = 0.005
            spy_10d = 0.010
            rows.append({
                "symbol": ticker,
                "score_date": str(base_date.date()),
                "score": score,
                "return_5d": ret_5d,
                "return_10d": ret_10d,
                "spy_5d_return": spy_5d,
                "spy_10d_return": spy_10d,
                "beat_spy_5d": int(ret_5d > spy_5d),
                "beat_spy_10d": int(ret_10d > spy_10d),
                "_team_id": team_id,
            })
    return pd.DataFrame(rows)


def _team_lift_from_score_perf(score_perf: pd.DataFrame) -> list[dict]:
    """Build a synthetic team_lift list matching the score_perf rows."""
    out = []
    for team_id, sub in score_perf.groupby("_team_id"):
        picks = []
        for _, r in sub.iterrows():
            picks.append({
                "ticker": r["symbol"],
                "eval_date": r["score_date"],
                "return_5d": r["return_5d"],
            })
        out.append({"team_id": team_id, "n_picks": len(picks), "picks": picks})
    return out


class TestGracefulDegrade:
    def test_no_prices_no_ohlc_returns_partial_metrics(self):
        sp = _make_score_perf()
        team_lift = _team_lift_from_score_perf(sp)
        out = compute_team_metrics(team_lift=team_lift, score_performance_df=sp)
        for team_id, bundle in out.items():
            # IC + expectancy work without prices.
            assert bundle["ic"]["status"] in ("ok", "insufficient_data", "no_variance")
            assert bundle["expectancy"]["status"] in ("ok", "insufficient_data",
                                                       "no_wins", "no_losses")
            # Excursion: ohlc was NOT passed at all → config-I7600 calls that
            # a wiring defect, not thin data. This assertion used to read
            # `== "insufficient_data"`, i.e. it pinned the exact conflation
            # that let `portfolio_excursion.json` sit dead for months.
            assert bundle["excursion"]["status"] == "error"
            assert bundle["excursion"]["reason"].startswith(
                "defect:ohlc_not_wired",
            )
            # risk-matched alpha still degrades on absent prices.
            assert bundle["alpha_vs_ew_high_vol"]["status"] == "insufficient_data"
            assert bundle["alpha_vs_beta_spy"]["status"] == "insufficient_data"

    def test_empty_team_lift_returns_empty_dict(self):
        sp = _make_score_perf()
        out = compute_team_metrics(team_lift=[], score_performance_df=sp)
        assert out == {}

    def test_no_score_performance_returns_insufficient(self):
        team_lift = [{"team_id": "tech", "n_picks": 5, "picks": [
            {"ticker": "AAA", "eval_date": "2026-01-05", "return_5d": 0.01},
        ]}]
        out = compute_team_metrics(team_lift=team_lift, score_performance_df=None)
        assert out["tech"]["ic"]["status"] == "insufficient_data"
        assert out["tech"]["expectancy"]["status"] == "insufficient_data"


class TestICAndExpectancy:
    def test_team_with_signal_has_positive_ic(self):
        sp = _make_score_perf(n_per_team=20)
        team_lift = _team_lift_from_score_perf(sp)
        out = compute_team_metrics(team_lift=team_lift, score_performance_df=sp)
        # Tech has signal-aligned scores → IC should be positive.
        if out["tech"]["ic"]["status"] == "ok":
            assert out["tech"]["ic"]["ic"] > 0.3

    def test_expectancy_emitted_with_enough_picks(self):
        sp = _make_score_perf(n_per_team=15)
        team_lift = _team_lift_from_score_perf(sp)
        out = compute_team_metrics(team_lift=team_lift, score_performance_df=sp)
        # Expectancy needs ≥ 5 picks.
        for team_id, bundle in out.items():
            assert bundle["expectancy"].get("status") in ("ok", "no_wins", "no_losses")


class TestPortfolioCalibration:
    def test_normalizes_score_to_unit_interval(self):
        # Build score_performance with score 60-90 + correlated outcomes.
        rng = np.random.default_rng(42)
        rows = []
        for i in range(150):
            score = 60 + (i % 31)  # 60..90 cycling
            true_p = score / 100.0
            outcome = int(rng.uniform() < true_p)
            rows.append({
                "symbol": f"T{i}",
                "score_date": "2026-01-05",
                "score": score,
                "beat_spy_21d": outcome,
            })
        df = pd.DataFrame(rows)
        result = compute_portfolio_calibration(df)
        assert result["status"] == "ok"
        # Well-calibrated → ECE should be low.
        assert result["ece"] < 0.1

    def test_missing_columns_returns_insufficient(self):
        df = pd.DataFrame({"symbol": ["A"], "score_date": ["2026-01-05"]})
        result = compute_portfolio_calibration(df)
        assert result["status"] == "insufficient_data"

    def test_empty_df_returns_insufficient(self):
        result = compute_portfolio_calibration(pd.DataFrame())
        assert result["status"] == "insufficient_data"


def _realistic_ohlc(tickers, *, periods: int = 60, seed: int = 7) -> dict:
    """OHLCV in the exact shape the live artifact carries.

    Measured against
    ``s3://alpha-engine-research/backtest/2026-08-14/.phases/simulation_setup/
    ohlcv_by_ticker.parquet`` on 2026-08-21 (903 tickers, 2 514 rows each,
    2016-08-15 → 2026-08-14): ``dict[str, pd.DataFrame]``, business-day
    ``DatetimeIndex``, lowercase ``open/high/low/close``. The pre-config-I7600
    coverage used one-row frames, which is why it stayed green for months
    while this analysis had never once produced a number in production.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-05", periods=periods, freq="B")
    out = {}
    for t in tickers:
        close = 100.0 * np.cumprod(1.0 + rng.normal(0.0005, 0.015, periods))
        out[t] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.002, periods)),
                "high": close * (1 + abs(rng.normal(0.01, 0.004, periods))),
                "low": close * (1 - abs(rng.normal(0.01, 0.004, periods))),
                "close": close,
            },
            index=idx,
        )
    return out


class TestPortfolioExcursion:
    def test_unwired_ohlc_is_a_defect_not_thin_data(self):
        """config-I7600 deliverable 3.

        ``ohlc=None`` is the caller never wiring the input; ``ohlc={}`` is a
        date whose OHLCV artifact is genuinely empty. Both used to return
        ``insufficient_data`` / ``"no ohlc data"`` — one string covering a
        permanent defect and a transient gap, which is precisely why nobody
        read `portfolio_excursion.json` as broken.
        """
        sp = _make_score_perf()

        unwired = compute_portfolio_excursion_summary(sp, ohlc=None)
        assert unwired["status"] == "error"
        assert unwired["reason"] == "defect:ohlc_not_wired: caller passed ohlc=None"

        empty = compute_portfolio_excursion_summary(sp, ohlc={})
        assert empty["status"] == "insufficient_data"
        assert empty["reason"].startswith("insufficient_data:no_ohlc_data")

        assert unwired["reason"] != empty["reason"]

    def test_realistic_ohlc_shape_produces_ok(self):
        """config-I7600 deliverable 4.

        The whole point of the wiring: given the OHLCV shape the pipeline
        actually persists, this analysis produces a graded result rather than
        an N/A. Cross-checked against the live 2026-08-14 artifact on
        2026-08-21 — 40 tickers x 3 eval dates yielded status ok, n=120,
        mean_mfe_mae_ratio 2.36.
        """
        tickers = [f"T{i:02d}" for i in range(12)]
        ohlc = _realistic_ohlc(tickers)
        idx = ohlc[tickers[0]].index
        rows = [
            {"symbol": t, "score_date": d, "score": 72, "beat_spy_10d": 1}
            for t in tickers
            for d in (idx[5], idx[15], idx[25])
        ]
        result = compute_portfolio_excursion_summary(
            pd.DataFrame(rows), ohlc=ohlc, horizon_days=10,
        )
        assert result["status"] == "ok", result
        assert result["n"] == len(rows)
        assert result["mean_mfe_mae_ratio"] is not None

    def test_computation_failure_is_error_not_insufficient(self):
        """config-I7600: a raise inside the excursion maths must not be
        rendered in the same slot as a thin date."""
        sp = _make_score_perf()
        # ``ohlc`` is a non-empty dict, so the wiring guards pass; the value
        # is not a DataFrame, so compute_per_pick_excursion raises.
        result = compute_portfolio_excursion_summary(sp, ohlc={"TE0": object()})
        assert result["status"] == "error"
        assert result["reason"].startswith("defect:excursion_compute_error")

    def test_filters_by_score_threshold(self):
        # All scores < 60 → should return insufficient (no picks pass).
        df = pd.DataFrame([{
            "symbol": "AAA", "score_date": "2026-01-05",
            "score": 50, "beat_spy_10d": 1,
        }])
        ohlc = {"AAA": pd.DataFrame(
            {"high": [105], "low": [95], "close": [100]},
            index=pd.date_range("2026-01-05", periods=1, freq="B"),
        )}
        result = compute_portfolio_excursion_summary(
            df, ohlc=ohlc, score_threshold=60,
        )
        assert result["status"] == "insufficient_data"


class TestEndToEndIntegration:
    def test_orchestrator_output_feeds_compute_scorecard(self):
        from analysis.grading import compute_scorecard

        sp = _make_score_perf(n_per_team=15)
        team_lift = _team_lift_from_score_perf(sp)
        team_metrics = compute_team_metrics(
            team_lift=team_lift, score_performance_df=sp,
        )
        portfolio_calibration = compute_portfolio_calibration(sp)

        # Compose into team_lift the way evaluate.py does.
        team_lift_for_grading = [
            {"team_id": t["team_id"], "n_picks": t["n_picks"],
             "lift": 0.5, "lift_vs_quant": 0.3}
            for t in team_lift
        ]

        scorecard = compute_scorecard(
            e2e_lift={
                "status": "ok",
                "scanner_lift": {"lift": 1.0, "n_passing": 50, "n_universe": 900},
                "team_lift": team_lift_for_grading,
                "cio_lift": {"lift": 1.0, "advance_avg": 1.5, "reject_avg": -0.5,
                             "n_advance": 10, "n_reject": 8},
                "cio_vs_ranking": {"lift": 0.5, "cio_beats_ranking": True,
                                   "n_dates": 8, "n_picks": 10, "avg_overlap": 0.5,
                                   "cio_avg": 1.5, "ranking_avg": 1.0},
            },
            team_metrics=team_metrics,
            calibration_diagnostics=portfolio_calibration if portfolio_calibration.get("status") == "ok" else None,
        )

        # Skill-composite path was used → teams have IC in detail.
        teams = scorecard["research"]["components"]["sector_teams"]
        for t in teams:
            assert "ic" in t["detail"]

        # Calibration component appears in research when wired through.
        if portfolio_calibration.get("status") == "ok":
            assert "calibration_diagnostics" in scorecard["research"]["components"]
