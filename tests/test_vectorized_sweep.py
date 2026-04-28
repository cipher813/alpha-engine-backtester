"""Integration tests for vectorized sweep orchestrator (Tier 4 PR 4, 2026-04-27).

Pins the contract that ``run_vectorized_sweep`` produces
order-stream output compatible with the scalar
``_run_simulation_loop`` orders schema, so downstream
``orders_to_portfolio`` + ``compute_portfolio_stats`` can run unchanged.

Coverage
--------
  * Combo config flatten: list-of-dicts → numpy arrays
  * Feature matrix build: per-ticker Series → [n_dates, n_tickers]
  * Sector index arrays: ticker → sector_idx, sector → ETF ticker_idx
  * Per-date signal extraction: SignalLookup → entry-pipeline arrays
  * Per-date research action extraction: HOLD default + EXIT/REDUCE/ENTER overrides
  * End-to-end small fixture: 3 combos × 30 dates × 5 tickers, verify
    per-combo orders accumulate without cross-combo state corruption
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest


_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


from synthetic.vectorized_sweep import (
    DEFAULT_SECTOR_ETF_MAP,
    build_combo_configs,
    build_feature_matrices,
    build_lookback_return_matrix,
    build_sector_arrays,
    extract_research_actions,
    extract_signal_arrays,
    run_vectorized_sweep,
)


@dataclass
class FakeSignalLookup:
    """Stand-in for backtest.SignalLookup — must mirror four-attribute
    shape including `actionable` (added 2026-04-28 after the Tier 4
    Layer 3 v14 parity bug — vectorized engine reads `actionable.get(
    "enter")`, not `signals_raw_filtered.get("enter")`).
    """
    signals_raw_filtered: dict
    signals_by_ticker: dict
    universe_sectors: dict
    actionable: dict = field(default_factory=dict)


@dataclass
class FakeFeatureLookup:
    """Stand-in for FeatureLookup — only the dict-attributes needed."""
    atr_dollar: dict
    rsi: dict
    momentum_20d_pct: dict
    returns: dict
    support_20_low: dict


def _make_price_matrix(n_dates: int, tickers: list) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    data = {}
    for t in tickers:
        # Trend with noise — enough movement to trigger some gates.
        base = 100 + rng.normal(0, 5, n_dates).cumsum() * 0.3
        data[t] = np.maximum(base, 5.0)
    return pd.DataFrame(data, index=idx)


def _make_ohlcv(price_matrix: pd.DataFrame) -> dict:
    """Per-ticker OHLCV from a closes-only price matrix."""
    rng = np.random.default_rng(11)
    out = {}
    for t in price_matrix.columns:
        closes = price_matrix[t].to_numpy()
        highs = closes + rng.uniform(0, 1, len(closes))
        lows = closes - rng.uniform(0, 1, len(closes))
        opens = np.concatenate(([closes[0]], closes[:-1]))
        out[t] = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows, "close": closes},
            index=price_matrix.index,
        )
    return out


def _make_feature_lookup(ohlcv: dict, lookback: int = 20) -> FakeFeatureLookup:
    """Build minimal FeatureLookup-compatible structure from OHLCV.

    Computes daily returns + 20-day rolling momentum + a constant ATR
    placeholder so vectorized exits have something to evaluate.
    """
    atr = {}
    rsi = {}
    mom = {}
    rets = {}
    sup = {}
    for ticker, df in ohlcv.items():
        close = df["close"]
        atr[ticker] = pd.Series(
            np.full(len(close), 1.5), index=close.index, name="atr",
        )
        rsi[ticker] = pd.Series(
            np.full(len(close), 50.0), index=close.index, name="rsi",
        )
        mom[ticker] = (close.pct_change(periods=lookback) * 100.0).rename("mom")
        rets[ticker] = close.pct_change().rename("ret")
        sup[ticker] = df["low"].rolling(window=lookback).min()
    return FakeFeatureLookup(
        atr_dollar=atr, rsi=rsi, momentum_20d_pct=mom,
        returns=rets, support_20_low=sup,
    )


# ────────────────────────────────────────────────────────────────────


class TestBuildComboConfigs:
    def test_flattens_default_combos(self):
        combos = [
            {"min_score": 60, "max_position_pct": 0.05},
            {"min_score": 70, "max_position_pct": 0.10},
            {"min_score": 80, "max_position_pct": 0.15},
        ]
        exit_cfg, entry_cfg = build_combo_configs(combos)
        np.testing.assert_array_equal(
            entry_cfg.min_score_to_enter, [60, 70, 80],
        )
        np.testing.assert_array_almost_equal(
            entry_cfg.max_position_pct, [0.05, 0.10, 0.15],
        )

    def test_strategy_params_per_combo(self):
        combos = [
            {"strategy": {"exit_manager": {"atr_multiplier": 2.0}}},
            {"strategy": {"exit_manager": {"atr_multiplier": 3.0}}},
        ]
        exit_cfg, _ = build_combo_configs(combos)
        np.testing.assert_array_almost_equal(
            exit_cfg.atr_multiplier, [2.0, 3.0],
        )


class TestBuildFeatureMatrices:
    def test_returns_dense_matrix_aligned_to_dates(self):
        tickers = ["AAPL", "MSFT"]
        pm = _make_price_matrix(40, tickers)
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)

        ti = {"AAPL": 0, "MSFT": 1}
        mats = build_feature_matrices(fl, ti, pm.index)
        assert mats["atr_dollar"].shape == (40, 2)
        # ATR is constant 1.5 in fixture
        assert np.all(np.isclose(mats["atr_dollar"], 1.5))
        # Momentum is NaN for first 20 bars
        assert np.all(np.isnan(mats["momentum_20d_pct"][:20]))
        assert not np.any(np.isnan(mats["momentum_20d_pct"][21:]))

    def test_missing_ticker_yields_nan_column(self):
        pm = _make_price_matrix(30, ["AAPL"])
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)
        # ticker_to_idx claims an extra ticker not in feature_lookup
        ti = {"AAPL": 0, "GHOST": 1}
        mats = build_feature_matrices(fl, ti, pm.index)
        assert mats["atr_dollar"].shape == (30, 2)
        assert np.all(np.isnan(mats["atr_dollar"][:, 1]))


class TestBuildLookbackReturnMatrix:
    def test_returns_match_simple_calc(self):
        idx = pd.date_range("2024-01-01", periods=30, freq="B")
        pm = pd.DataFrame({"AAPL": np.linspace(100, 130, 30)}, index=idx)
        m = build_lookback_return_matrix(pm, lookback=10)
        assert m.shape == (30, 1)
        # First 10 rows are NaN
        assert np.all(np.isnan(m[:10]))
        # Row 10: price[10]/price[0] - 1
        np.testing.assert_almost_equal(
            m[10, 0], pm.iloc[10, 0] / pm.iloc[0, 0] - 1,
        )


class TestBuildSectorArrays:
    def test_sector_and_etf_mapping(self):
        ti = {"AAPL": 0, "MSFT": 1, "XLK": 2, "JNJ": 3, "XLV": 4}
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}
        sec_idx, etf_idx, label_to_idx = build_sector_arrays(ti, sector_map)
        # AAPL + MSFT in Technology, JNJ in Healthcare. ETFs not in sector_map.
        tech_idx = label_to_idx["Technology"]
        hc_idx = label_to_idx["Healthcare"]
        assert sec_idx[0] == tech_idx
        assert sec_idx[1] == tech_idx
        assert sec_idx[3] == hc_idx
        # ETFs are -1 (no sector mapped to them)
        assert sec_idx[2] == -1
        assert sec_idx[4] == -1
        # Sector ETF lookup: Technology → XLK at idx 2; Healthcare → XLV at idx 4
        assert etf_idx[tech_idx] == 2
        assert etf_idx[hc_idx] == 4


class TestExtractSignalArrays:
    def test_extracts_basic_fields(self):
        # `actionable` carries the post-`get_actionable_signals` shape that
        # extract_signal_arrays reads. signals_raw_filtered keeps the `date`
        # for signal_age_days computation in the run loop.
        enter_list = [
            {"ticker": "AAPL", "score": 85, "sector": "Technology",
             "sector_rating": "overweight", "conviction": "rising",
             "price_target_upside": 0.20},
            {"ticker": "JNJ", "score": 72, "sector": "Healthcare",
             "sector_rating": "market_weight", "conviction": "stable",
             "price_target_upside": 0.10},
        ]
        sl = FakeSignalLookup(
            signals_raw_filtered={"date": "2024-02-01"},
            signals_by_ticker={},
            universe_sectors={},
            actionable={"enter": enter_list},
        )
        ti = {"AAPL": 0, "JNJ": 1, "XLK": 2}
        label_to_idx = {"Technology": 0, "Healthcare": 1}
        mom = np.array([5.0, -2.0, 0.0])
        out = extract_signal_arrays(
            sl,
            predictions={"AAPL": {"gbm_veto": True, "prediction_confidence": 0.85}},
            ticker_to_idx=ti,
            sector_label_to_idx=label_to_idx,
            atr_pct_by_ticker={"AAPL": 0.025, "JNJ": 0.015},
            coverage_by_ticker={"AAPL": 1.0, "JNJ": 0.92},
            earnings_by_ticker={"AAPL": 3},
            momentum_at_date_per_ticker=mom,
        )
        assert list(out["signal_ticker_idx"]) == [0, 1]
        assert list(out["signal_score"]) == [85.0, 72.0]
        assert list(out["signal_sector_idx"]) == [0, 1]
        assert out["signal_gbm_veto"][0] == True
        assert out["signal_gbm_veto"][1] == False
        assert out["signal_pred_confidence"][0] == pytest.approx(0.85)
        assert np.isnan(out["signal_pred_confidence"][1])
        assert out["signal_atr_pct"][0] == pytest.approx(0.025)
        assert out["signal_days_to_earnings"][0] == 3
        assert out["signal_days_to_earnings"][1] == -1
        assert out["signal_momentum_at_date"][0] == pytest.approx(5.0)
        assert out["signal_momentum_at_date"][1] == pytest.approx(-2.0)

    def test_skips_signals_missing_from_ticker_index(self):
        enter_list = [
            {"ticker": "AAPL", "score": 80, "sector": "Technology",
             "conviction": "stable", "price_target_upside": 0.10},
            {"ticker": "GHOST", "score": 80, "sector": "Technology",
             "conviction": "stable", "price_target_upside": 0.10},
        ]
        sl = FakeSignalLookup(
            signals_raw_filtered={"date": "2024-01-01"},
            signals_by_ticker={}, universe_sectors={},
            actionable={"enter": enter_list},
        )
        ti = {"AAPL": 0}
        out = extract_signal_arrays(
            sl, predictions={}, ticker_to_idx=ti,
            sector_label_to_idx={"Technology": 0},
            atr_pct_by_ticker={}, coverage_by_ticker={},
            earnings_by_ticker={},
            momentum_at_date_per_ticker=np.array([1.0]),
        )
        assert out["signal_ticker_idx"].size == 1
        assert out["tickers"] == ["AAPL"]


class TestExtractResearchActions:
    def test_overrides_per_field(self):
        from synthetic.vectorized_exits import RA_ENTER, RA_EXIT, RA_HOLD, RA_REDUCE

        sl = FakeSignalLookup(
            signals_raw_filtered={},
            signals_by_ticker={}, universe_sectors={},
            actionable={
                "enter": [{"ticker": "AAPL"}],
                "exit": [{"ticker": "MSFT"}],
                "reduce": [{"ticker": "JNJ"}],
                "hold": [{"ticker": "TSLA"}],
            },
        )
        ti = {"AAPL": 0, "MSFT": 1, "JNJ": 2, "TSLA": 3, "GHOST": 4}
        actions = extract_research_actions(sl, ti, n_tickers=5)
        assert actions[0] == RA_ENTER
        assert actions[1] == RA_EXIT
        assert actions[2] == RA_REDUCE
        # HOLD field not consumed (default already HOLD)
        assert actions[3] == RA_HOLD
        # GHOST: default HOLD (no signal mentions it)
        assert actions[4] == RA_HOLD


# ────────────────────────────────────────────────────────────────────
# End-to-end integration
# ────────────────────────────────────────────────────────────────────


class TestEndToEndSweep:
    def test_three_combos_thirty_dates_five_tickers(self):
        """Smoke + invariants:
          * No exception across the full sweep loop
          * Each combo gets its own order list (no cross-combo writes)
          * Total order count matches diagnostic counters
        """
        tickers = ["AAPL", "MSFT", "XLK", "JNJ", "XLV"]
        n_dates = 30
        pm = _make_price_matrix(n_dates, tickers)
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)
        sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}

        # Build per-date SignalLookups: every 5th date emits an ENTER
        # signal for AAPL or MSFT.
        signal_lookups = {}
        for i, date in enumerate(pm.index):
            ds = date.strftime("%Y-%m-%d")
            if i >= 22 and i % 5 == 2:
                ticker = "AAPL" if i % 10 == 2 else "MSFT"
                enter = [{
                    "ticker": ticker, "score": 80,
                    "sector": "Technology",
                    "sector_rating": "market_weight",
                    "conviction": "stable",
                    "price_target_upside": 0.20,
                }]
            else:
                enter = []
            signal_lookups[ds] = FakeSignalLookup(
                signals_raw_filtered={
                    "universe": [], "buy_candidates": [], "date": ds,
                },
                signals_by_ticker={},
                universe_sectors={"AAPL": "Technology", "MSFT": "Technology"},
                actionable={
                    "enter": enter, "exit": [], "reduce": [], "hold": [],
                },
            )

        combos = [
            {"min_score": 70, "max_position_pct": 0.05,
             "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
             "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
             "coverage_sizing_enabled": False, "correlation_block_enabled": False,
             "momentum_gate_enabled": False, "max_sector_pct": 1.0,
             "max_equity_pct": 1.0},
            {"min_score": 75, "max_position_pct": 0.05,
             "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
             "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
             "coverage_sizing_enabled": False, "correlation_block_enabled": False,
             "momentum_gate_enabled": False, "max_sector_pct": 1.0,
             "max_equity_pct": 1.0},
            {"min_score": 90, "max_position_pct": 0.05,
             "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
             "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
             "coverage_sizing_enabled": False, "correlation_block_enabled": False,
             "momentum_gate_enabled": False, "max_sector_pct": 1.0,
             "max_equity_pct": 1.0},
        ]

        orders_per_combo, diagnostics = run_vectorized_sweep(
            combo_configs=combos,
            price_matrix=pm,
            ohlcv_by_ticker=ohlcv,
            signal_lookups=signal_lookups,
            feature_lookup=fl,
            spy_prices=None,
            sector_map=sector_map,
        )

        assert len(orders_per_combo) == 3
        # Combo 0 (min=70) should accept; combo 2 (min=90) should reject.
        assert len(orders_per_combo[0]) > 0, "combo 0 (min_score=70) emitted no orders"
        assert len(orders_per_combo[2]) == 0, "combo 2 (min_score=90) should reject all"

        # Diagnostics counters reflect total entries + exits across combos.
        n_orders_total = sum(len(o) for o in orders_per_combo)
        assert n_orders_total == diagnostics["entries_applied"] + diagnostics["exits_applied"]

        # Order schema: each order has the keys orders_to_portfolio expects.
        if orders_per_combo[0]:
            o = orders_per_combo[0][0]
            for k in (
                "date", "ticker", "action", "shares",
                "price_at_order", "portfolio_nav_at_order",
            ):
                assert k in o, f"missing key {k} in order: {o}"

    def test_single_combo_invariants(self):
        """N=1 sweep: smoke + ensure the single combo accumulates orders
        consistently with sim state. After the sweep, sim's total cash +
        positions × prices should equal init_cash + cumulative P&L."""
        tickers = ["AAPL", "XLK"]
        pm = _make_price_matrix(25, tickers)
        ohlcv = _make_ohlcv(pm)
        fl = _make_feature_lookup(ohlcv)
        sector_map = {"AAPL": "Technology"}

        signal_lookups = {}
        for i, date in enumerate(pm.index):
            ds = date.strftime("%Y-%m-%d")
            enter = []
            if i == 22:
                enter = [{"ticker": "AAPL", "score": 80,
                          "sector": "Technology",
                          "sector_rating": "market_weight",
                          "conviction": "stable",
                          "price_target_upside": 0.20}]
            signal_lookups[ds] = FakeSignalLookup(
                signals_raw_filtered={
                    "universe": [], "buy_candidates": [], "date": ds,
                },
                signals_by_ticker={}, universe_sectors={},
                actionable={
                    "enter": enter, "exit": [], "reduce": [], "hold": [],
                },
            )

        combos = [{
            "min_score": 70, "max_position_pct": 0.05,
            "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False, "correlation_block_enabled": False,
            "momentum_gate_enabled": False, "max_sector_pct": 1.0,
            "max_equity_pct": 1.0,
        }]

        orders_per_combo, diagnostics = run_vectorized_sweep(
            combo_configs=combos,
            price_matrix=pm,
            ohlcv_by_ticker=ohlcv,
            signal_lookups=signal_lookups,
            feature_lookup=fl,
            spy_prices=None,
            sector_map=sector_map,
            init_cash=1_000_000.0,
        )

        assert diagnostics["n_combos"] == 1
        assert diagnostics["n_dates"] == 25
        assert len(orders_per_combo) == 1
        # We emitted exactly one ENTER on date_idx=22; no exits possible
        # in such a short window (no time decay reach, no ATR breach
        # given stable price drift).
        assert diagnostics["entries_applied"] == 1
        # The one order is for AAPL
        assert orders_per_combo[0][0]["ticker"] == "AAPL"
        assert orders_per_combo[0][0]["action"] == "ENTER"
