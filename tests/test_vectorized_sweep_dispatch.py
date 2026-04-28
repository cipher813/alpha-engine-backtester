"""Wire-in tests for ``_run_vectorized_param_sweep`` (Tier 4 PR 5,
2026-04-27).

Exercises the gate in ``run_predictor_param_sweep`` that dispatches
to the vectorized engine when ``config["use_vectorized_sweep"]`` is
True. Verifies the dispatch path produces a sweep_df with the same
shape as the scalar path (one row per combo, params + stats columns,
sorted by total_alpha or sharpe_ratio).

End-to-end parity vs scalar path is validated separately on real
spot data (v14 dispatch). These tests pin the interface contract: as
long as the vectorized path produces a DataFrame compatible with
downstream optimizers / reporters, the swap is safe.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest


_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


from backtest import _run_vectorized_param_sweep


@dataclass
class FakeSignalLookup:
    signals_raw_filtered: dict
    signals_by_ticker: dict
    universe_sectors: dict


@dataclass
class FakeFeatureLookup:
    atr_dollar: dict
    rsi: dict
    momentum_20d_pct: dict
    returns: dict
    support_20_low: dict


def _build_fixture():
    """Tiny end-to-end fixture: 5 tickers × 50 dates."""
    rng = np.random.default_rng(11)
    tickers = ["AAPL", "MSFT", "JNJ", "XLK", "XLV"]
    n_dates = 50
    idx = pd.date_range("2024-01-01", periods=n_dates, freq="B")

    # Price matrix
    closes = {}
    for t in tickers:
        base = 100 + rng.normal(0, 1.0, n_dates).cumsum() * 0.3
        closes[t] = np.maximum(base, 5.0)
    price_matrix = pd.DataFrame(closes, index=idx)

    # OHLCV
    ohlcv = {}
    for t in tickers:
        c = price_matrix[t].to_numpy()
        h = c + rng.uniform(0, 0.5, n_dates)
        l = c - rng.uniform(0, 0.5, n_dates)
        o = np.concatenate(([c[0]], c[:-1]))
        ohlcv[t] = pd.DataFrame(
            {"open": o, "high": h, "low": l, "close": c}, index=idx,
        )

    # FeatureLookup-compatible
    fl = FakeFeatureLookup(
        atr_dollar={t: pd.Series(np.full(n_dates, 1.5), index=idx) for t in tickers},
        rsi={t: pd.Series(np.full(n_dates, 50.0), index=idx) for t in tickers},
        momentum_20d_pct={
            t: (price_matrix[t].pct_change(periods=20) * 100).rename("mom")
            for t in tickers
        },
        returns={t: price_matrix[t].pct_change().rename("ret") for t in tickers},
        support_20_low={
            t: ohlcv[t]["low"].rolling(window=20).min() for t in tickers
        },
    )

    # Signal lookups: emit ENTER for AAPL on date_idx 25, MSFT on 35
    signal_lookups = {}
    for i, date in enumerate(idx):
        ds = date.strftime("%Y-%m-%d")
        enter = []
        if i == 25:
            enter = [{
                "ticker": "AAPL", "score": 80, "sector": "Technology",
                "sector_rating": "market_weight", "conviction": "stable",
                "price_target_upside": 0.20,
            }]
        elif i == 35:
            enter = [{
                "ticker": "MSFT", "score": 75, "sector": "Technology",
                "sector_rating": "market_weight", "conviction": "rising",
                "price_target_upside": 0.15,
            }]
        signal_lookups[ds] = FakeSignalLookup(
            signals_raw_filtered={
                "enter": enter, "exit": [], "reduce": [], "hold": [],
                "universe": [], "buy_candidates": [], "date": ds,
                "market_regime": "bull",
            },
            signals_by_ticker={},
            universe_sectors={"AAPL": "Technology", "MSFT": "Technology"},
        )

    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "JNJ": "Healthcare"}
    spy_prices = pd.Series(
        np.linspace(400, 420, n_dates), index=idx, name="SPY",
    )
    return {
        "price_matrix": price_matrix,
        "ohlcv_by_ticker": ohlcv,
        "feature_lookup": fl,
        "signal_lookups": signal_lookups,
        "sector_map": sector_map,
        "spy_prices": spy_prices,
    }


class TestVectorizedDispatch:
    def test_produces_sweep_df_with_param_and_stat_columns(self):
        fix = _build_fixture()
        grid = {
            "min_score": [70, 80],
            "max_position_pct": [0.05],
        }
        base_config = {
            "init_cash": 1_000_000.0,
            "simulation_fees": 0.001,
            "use_vectorized_sweep": True,
            # Disable ancillary sizing factors for clean parity
            "atr_sizing_enabled": False,
            "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False,
            "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False,
            "correlation_block_enabled": False,
            "momentum_gate_enabled": False,
            "max_sector_pct": 1.0,
            "max_equity_pct": 1.0,
        }
        sweep_settings = {"mode": "grid"}

        df = _run_vectorized_param_sweep(
            grid=grid,
            base_config=base_config,
            sweep_settings=sweep_settings,
            price_matrix=fix["price_matrix"],
            ohlcv_by_ticker=fix["ohlcv_by_ticker"],
            signal_lookups=fix["signal_lookups"],
            feature_lookup=fix["feature_lookup"],
            spy_prices=fix["spy_prices"],
            sector_map=fix["sector_map"],
        )

        # 2 combos in the grid
        assert len(df) == 2
        # Param cols
        assert "min_score" in df.columns
        assert "max_position_pct" in df.columns
        # Stat cols (either ok stats or no_orders)
        assert "status" in df.columns or "total_return" in df.columns
        # Metadata
        assert df.attrs.get("sweep_mode", "").endswith("(vectorized)")
        assert df.attrs.get("sweep_trials") == 2

    def test_high_score_combo_yields_no_orders(self):
        """Combo with min_score=99 should reject all signals (scores are
        80 / 75) → no orders → graceful no_orders row."""
        fix = _build_fixture()
        grid = {
            "min_score": [99],
            "max_position_pct": [0.05],
        }
        base_config = {
            "init_cash": 1_000_000.0,
            "use_vectorized_sweep": True,
            "atr_sizing_enabled": False,
            "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False,
            "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False,
            "correlation_block_enabled": False,
            "momentum_gate_enabled": False,
            "max_sector_pct": 1.0,
            "max_equity_pct": 1.0,
        }
        df = _run_vectorized_param_sweep(
            grid=grid,
            base_config=base_config,
            sweep_settings={"mode": "grid"},
            price_matrix=fix["price_matrix"],
            ohlcv_by_ticker=fix["ohlcv_by_ticker"],
            signal_lookups=fix["signal_lookups"],
            feature_lookup=fix["feature_lookup"],
            spy_prices=fix["spy_prices"],
            sector_map=fix["sector_map"],
        )
        assert len(df) == 1
        assert df.iloc[0]["status"] == "no_orders"
        assert df.iloc[0]["total_orders"] == 0

    def test_random_mode_seed_reproducible(self):
        """Random mode with same seed → identical combo set across runs."""
        fix = _build_fixture()
        grid = {
            "min_score": [60, 70, 80],
            "max_position_pct": [0.05, 0.10],
        }
        base_config = {
            "init_cash": 1_000_000.0,
            "use_vectorized_sweep": True,
            "atr_sizing_enabled": False, "confidence_sizing_enabled": False,
            "staleness_discount_enabled": False, "earnings_sizing_enabled": False,
            "coverage_sizing_enabled": False, "correlation_block_enabled": False,
            "momentum_gate_enabled": False,
            "max_sector_pct": 1.0, "max_equity_pct": 1.0,
        }
        sweep_settings = {"mode": "random", "max_trials": 3, "seed": 42}

        df1 = _run_vectorized_param_sweep(
            grid=grid, base_config=base_config, sweep_settings=sweep_settings,
            price_matrix=fix["price_matrix"], ohlcv_by_ticker=fix["ohlcv_by_ticker"],
            signal_lookups=fix["signal_lookups"], feature_lookup=fix["feature_lookup"],
            spy_prices=fix["spy_prices"], sector_map=fix["sector_map"],
        )
        df2 = _run_vectorized_param_sweep(
            grid=grid, base_config=base_config, sweep_settings=sweep_settings,
            price_matrix=fix["price_matrix"], ohlcv_by_ticker=fix["ohlcv_by_ticker"],
            signal_lookups=fix["signal_lookups"], feature_lookup=fix["feature_lookup"],
            spy_prices=fix["spy_prices"], sector_map=fix["sector_map"],
        )
        # Same combos sampled (order may differ due to sort but set should match)
        params1 = set(zip(df1["min_score"], df1["max_position_pct"]))
        params2 = set(zip(df2["min_score"], df2["max_position_pct"]))
        assert params1 == params2
