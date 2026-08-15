"""Regression: an exit on an unpriced ticker must not corrupt cash.

Root cause of the 2026-08-15 weekly-SF PredictorBacktest failure. Two
independent defects, one per test class:

1. ``compute_vectorized_exits`` branches 4 (momentum) and 5 (time decay) are
   price-independent, so they fired for a held ticker with no price on that
   date. ``apply_vectorized_exits`` then passed the raw NaN price straight to
   ``apply_sell``, which credited ``shares * NaN`` into ``cash``. NAV is
   ``cash + positions @ prices``, so that combo's NAV was NaN for every
   remaining date.
2. Nothing at the ``apply_buy`` / ``apply_sell`` boundary rejected the leg, so
   the corruption surfaced ~1700 dates later inside
   ``compute_vectorized_entries``'s share-count contract as
   "non-finite nav_per_combo at combo_idx=[50]" — a guard reporting the
   symptom in the wrong stage.
"""

import numpy as np
import pytest

from synthetic.vectorized_exits import (
    ACTION_EXIT,
    ExitDecisions,
    VectorizedExitConfig,
    apply_vectorized_exits,
    compute_vectorized_exits,
)
from synthetic.vectorized_sim import VectorizedSimulator

RA_HOLD = 0


def _sim(n_combos=1, tickers=("AAA", "BBB")):
    sim = VectorizedSimulator(
        n_combos=n_combos, ticker_index={t: i for i, t in enumerate(tickers)},
    )
    return sim


def _exit_config(n_combos, *, time_decay_exit_days=5):
    """Config with ONLY the price-independent time-decay branch armed."""
    return VectorizedExitConfig.from_uniform(
        n_combos,
        atr_trailing_enabled=False,
        fallback_stop_enabled=False,
        profit_take_enabled=False,
        momentum_exit_enabled=False,
        sector_relative_veto_enabled=False,
        position_loss_floor_enabled=False,
        time_decay_enabled=True,
        time_decay_reduce_days=3,
        time_decay_exit_days=time_decay_exit_days,
        reduce_fraction=0.5,
    )


def _compute(sim, prices, date_idx, config):
    n_t = sim.n_tickers
    return compute_vectorized_exits(
        sim,
        prices=prices,
        atr_dollar_at_date=np.full(n_t, np.nan),
        rsi_at_date=np.full(n_t, np.nan),
        momentum_at_date=np.full(n_t, np.nan),
        sector_lookback_return=np.full(n_t, np.nan),
        research_action_per_ticker=np.full(n_t, RA_HOLD, dtype=np.int8),
        sector_idx_per_ticker=np.full(n_t, -1, dtype=np.int32),
        sector_etf_ticker_idx=np.full(n_t, -1, dtype=np.int32),
        date_idx=date_idx,
        config=config,
    )


class TestExitDefersWhenUnpriced:
    def test_time_decay_exit_does_not_fire_on_unpriced_ticker(self):
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        config = _exit_config(1, time_decay_exit_days=5)
        # Day 10: well past the exit threshold, but AAA has no price today.
        prices = np.array([np.nan, 20.0])

        decisions = _compute(sim, prices, date_idx=10, config=config)

        assert decisions.exit_action[0, 0] == 0, (
            "an exit was emitted for a ticker with no price — there is no "
            "print to transact against"
        )
        assert decisions.deferred_no_price[0, 0]
        assert int(decisions.deferred_no_price.sum()) == 1

    def test_exit_fires_on_the_next_priced_date(self):
        """Deferral, not cancellation."""
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        config = _exit_config(1, time_decay_exit_days=5)

        assert _compute(sim, np.array([np.nan, 20.0]), 10, config).exit_action[0, 0] == 0
        later = _compute(sim, np.array([55.0, 20.0]), 11, config)
        assert later.exit_action[0, 0] == ACTION_EXIT
        assert not later.deferred_no_price[0, 0]

    def test_nav_stays_finite_across_an_unpriced_exit_window(self):
        """The end-to-end property the weekly SF actually needs."""
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        config = _exit_config(1, time_decay_exit_days=5)
        # AAA stops printing on date 4 — BEFORE its day-5 time-decay exit
        # would fire, so the exit day itself falls inside the unpriced window.
        for date_idx in range(1, 15):
            prices = np.array([np.nan if date_idx >= 4 else 50.0, 20.0])
            decisions = _compute(sim, prices, date_idx, config)
            apply_vectorized_exits(sim, decisions, prices)
            sim.update_nav(prices)
            assert np.isfinite(sim.nav).all(), (
                f"NAV went non-finite at date_idx={date_idx}"
            )
        assert np.isfinite(sim.cash).all()

    def test_zero_and_negative_prices_are_treated_as_unpriced(self):
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        config = _exit_config(1, time_decay_exit_days=5)
        for bad in (0.0, -1.0):
            decisions = _compute(sim, np.array([bad, 20.0]), 10, config)
            assert decisions.exit_action[0, 0] == 0, f"price={bad} exited"


class TestOrderLegFinitenessContract:
    """The backstop: a missed caller-side gate must be loud AT the corruption
    site, not 1700 dates downstream."""

    def test_apply_sell_rejects_nan_price(self):
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        with pytest.raises(ValueError, match="non-finite or non-positive order leg"):
            sim.apply_sell(
                np.array([0]), np.array([0]), np.array([100.0]),
                np.array([np.nan]),
            )
        assert np.isfinite(sim.cash).all()

    def test_apply_buy_rejects_nan_price(self):
        sim = _sim()
        with pytest.raises(ValueError, match="BUY"):
            sim.apply_buy(
                np.array([0]), np.array([0]), np.array([10.0]),
                np.array([np.nan]), date_idx=0,
            )
        assert np.isfinite(sim.cash).all()

    def test_apply_sell_rejects_nan_shares(self):
        sim = _sim()
        with pytest.raises(ValueError, match="SELL"):
            sim.apply_sell(
                np.array([0]), np.array([0]), np.array([np.nan]),
                np.array([10.0]),
            )

    def test_error_names_the_combo_and_ticker(self):
        sim = _sim(n_combos=3)
        with pytest.raises(ValueError) as exc:
            sim.apply_sell(
                np.array([2]), np.array([1]), np.array([5.0]),
                np.array([np.nan]),
            )
        assert "combo=2" in str(exc.value)
        assert "ticker_idx=1" in str(exc.value)

    def test_apply_vectorized_exits_never_reaches_the_backstop(self):
        """The two layers compose: with the price gate in place, the
        contract assertion is unreachable through the normal exit path."""
        sim = _sim()
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100.0]), np.array([50.0]),
            date_idx=0,
        )
        prices = np.array([np.nan, 20.0])
        decisions = _compute(sim, prices, 10, _exit_config(1))
        apply_vectorized_exits(sim, decisions, prices)  # must not raise
        assert np.isfinite(sim.cash).all()


def test_exit_decisions_deferred_field_defaults_to_none():
    """Back-compat for callers constructing ExitDecisions directly."""
    z = np.zeros((1, 1))
    assert ExitDecisions(z, z, z).deferred_no_price is None
