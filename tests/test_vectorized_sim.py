"""Parity tests for VectorizedSimulator (Tier 4 PR 1, 2026-04-27).

Pins the invariant that the matrix-axis sim produces byte-equal
state evolution to N independent ``SimulatedIBKRClient`` scalar
sims. The whole Tier 4 architecture rests on this — if the
vectorized state diverges from the scalar reference under any
order sequence, downstream PRs (vectorized exits, entries, sweep
wiring) silently inherit the bug.

Coverage:
  * NAV update: cash + positions @ prices
  * peak_nav: running max
  * highest_high: running max for held positions only
  * Drawdown multiplier: graduated tier lookup matches scalar
    risk_guard.compute_drawdown_multiplier
  * BUY: assignment semantics (matches SimulatedIBKRClient)
  * SELL: full-exit + partial-reduce match
  * Mixed sequences: deterministic randomized order streams
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest


# Ensure executor on sys.path for SimulatedIBKRClient import
_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


from synthetic.vectorized_sim import VectorizedSimulator, _NO_ENTRY


def _ticker_index(*tickers: str) -> dict:
    return {t: i for i, t in enumerate(tickers)}


class TestInitialization:
    def test_init_state_shape_and_values(self):
        ti = _ticker_index("AAPL", "MSFT", "NVDA")
        sim = VectorizedSimulator(n_combos=4, ticker_index=ti, init_cash=500_000)

        assert sim.positions.shape == (4, 3)
        assert sim.cash.shape == (4,)
        assert np.all(sim.cash == 500_000)
        assert np.all(sim.nav == 500_000)
        assert np.all(sim.peak_nav == 500_000)
        assert np.all(sim.entry_dates == _NO_ENTRY)
        assert np.all(sim.avg_costs == 0)
        assert np.all(sim.highest_high == 0)
        assert sim.n_tickers == 3
        assert sim.n_combos == 4

    def test_rejects_zero_combos(self):
        with pytest.raises(ValueError, match="n_combos"):
            VectorizedSimulator(n_combos=0, ticker_index=_ticker_index("AAPL"))

    def test_rejects_empty_ticker_index(self):
        with pytest.raises(ValueError, match="ticker_index"):
            VectorizedSimulator(n_combos=3, ticker_index={})

    def test_rejects_non_contiguous_ticker_index(self):
        with pytest.raises(ValueError, match="0\\.\\."):
            # Indices skip 1 — must be 0..N-1
            VectorizedSimulator(
                n_combos=3, ticker_index={"AAPL": 0, "MSFT": 2},
            )


class TestNAVUpdate:
    def test_nav_initial_no_positions(self):
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL", "MSFT"),
            init_cash=1_000_000,
        )
        sim.update_nav(np.array([150.0, 300.0]))
        # No positions held → NAV == cash
        assert np.all(sim.nav == 1_000_000)

    def test_nav_with_positions_matches_dot_product(self):
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL", "MSFT"),
            init_cash=1_000_000,
        )
        # Combo 0: 100 AAPL, Combo 1: 50 MSFT, Combo 2: 100 AAPL + 25 MSFT
        sim.apply_buy(
            combo_idx=np.array([0, 1, 2, 2]),
            ticker_idx=np.array([0, 1, 0, 1]),
            shares=np.array([100, 50, 100, 25]),
            prices=np.array([150.0, 300.0, 150.0, 300.0]),
            date_idx=0,
        )
        prices = np.array([155.0, 310.0])
        sim.update_nav(prices)

        # Manual: NAV = cash + positions @ prices
        # cash[0] = 1M - 100*150 = 985000; nav = 985000 + 100*155 = 1,000,500
        # cash[1] = 1M - 50*300 = 985000;  nav = 985000 + 50*310 = 1,000,500
        # cash[2] = 1M - 100*150 - 25*300 = 977500; nav = 977500 + 100*155 + 25*310 = 1,000,750
        assert sim.nav[0] == pytest.approx(1_000_500)
        assert sim.nav[1] == pytest.approx(1_000_500)
        assert sim.nav[2] == pytest.approx(1_000_750)

    def test_peak_nav_running_max(self):
        sim = VectorizedSimulator(
            n_combos=2, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        sim.apply_buy(
            combo_idx=np.array([0, 1]),
            ticker_idx=np.array([0, 0]),
            shares=np.array([100, 100]),
            prices=np.array([150.0, 150.0]),
            date_idx=0,
        )
        # Sequence: price up, peak captured; then down, peak preserved
        sim.update_nav(np.array([160.0]))
        first_nav = sim.nav.copy()
        first_peak = sim.peak_nav.copy()

        sim.update_nav(np.array([140.0]))
        # Peak must NOT be reduced when nav drops
        assert np.all(sim.peak_nav == first_peak)
        assert np.all(sim.nav < first_peak)

        # And rises again above prior peak: peak updates
        sim.update_nav(np.array([170.0]))
        assert sim.peak_nav[0] > first_peak[0]
        assert sim.peak_nav[1] > first_peak[1]

    def test_nav_handles_nan_prices_safely(self):
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL", "MSFT"),
            init_cash=1_000_000,
        )
        # No held positions; NaN price doesn't blow up
        sim.update_nav(np.array([np.nan, 300.0]))
        assert sim.nav[0] == pytest.approx(1_000_000)


class TestHighestHigh:
    def test_running_max_for_held_positions(self):
        sim = VectorizedSimulator(
            n_combos=2, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        sim.apply_buy(
            combo_idx=np.array([0]),
            ticker_idx=np.array([0]),
            shares=np.array([100]),
            prices=np.array([150.0]),
            date_idx=0,
        )
        # Combo 0 holds AAPL at 150 — highest_high seeds at 150 (apply_buy sets it)
        assert sim.highest_high[0, 0] == 150.0
        # Combo 1 doesn't hold anything → highest_high stays 0
        assert sim.highest_high[1, 0] == 0.0

        sim.update_highest_high(np.array([155.0]))
        assert sim.highest_high[0, 0] == 155.0
        assert sim.highest_high[1, 0] == 0.0  # unheld stays 0

        sim.update_highest_high(np.array([148.0]))
        # 148 < 155 → no update for combo 0
        assert sim.highest_high[0, 0] == 155.0

        sim.update_highest_high(np.array([170.0]))
        assert sim.highest_high[0, 0] == 170.0


class TestApplyBuy:
    def test_buy_updates_state_correctly(self):
        sim = VectorizedSimulator(
            n_combos=2, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        sim.apply_buy(
            combo_idx=np.array([0]),
            ticker_idx=np.array([0]),
            shares=np.array([100]),
            prices=np.array([150.0]),
            date_idx=42,
        )
        assert sim.positions[0, 0] == 100.0
        assert sim.avg_costs[0, 0] == 150.0
        assert sim.entry_dates[0, 0] == 42
        assert sim.highest_high[0, 0] == 150.0
        assert sim.cash[0] == 1_000_000 - 100 * 150
        # combo 1 untouched
        assert sim.positions[1, 0] == 0.0
        assert sim.cash[1] == 1_000_000

    def test_buy_assignment_semantics_overwrite(self):
        """Matches SimulatedIBKRClient.place_market_order BUY: a second
        BUY overwrites the position rather than adding shares."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
        )
        sim.apply_buy(np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0)
        # First buy
        assert sim.positions[0, 0] == 100
        cash_after_first = sim.cash[0]

        # Second buy of same ticker — overwrite
        sim.apply_buy(np.array([0]), np.array([0]), np.array([200]), np.array([160.0]), 1)
        assert sim.positions[0, 0] == 200  # NOT 300
        assert sim.avg_costs[0, 0] == 160.0
        # Cash debited again — caller responsibility (production gate
        # prevents this via "already in portfolio" check)
        assert sim.cash[0] == cash_after_first - 200 * 160


class TestApplySell:
    def test_full_exit_when_shares_matches_held(self):
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
        )
        sim.apply_buy(np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0)
        cash_before = sim.cash[0]

        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([100]), np.array([160.0]),
        )
        assert sim.positions[0, 0] == 0.0
        assert sim.avg_costs[0, 0] == 0.0
        assert sim.entry_dates[0, 0] == _NO_ENTRY
        assert sim.highest_high[0, 0] == 0.0
        # Proceeds: 100 shares × 160 = 16000 added to cash
        assert sim.cash[0] == cash_before + 100 * 160

    def test_full_exit_when_sell_exceeds_held(self):
        """Per SimulatedIBKRClient: SELL with shares > held still
        zeroes out the position; cash credited for actual held."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
        )
        sim.apply_buy(np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0)
        cash_before = sim.cash[0]

        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([200]), np.array([160.0]),  # over-sell
        )
        assert sim.positions[0, 0] == 0.0
        # Only 100 actually sold → cash + 100*160
        assert sim.cash[0] == cash_before + 100 * 160

    def test_partial_reduce(self):
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
        )
        sim.apply_buy(np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0)
        cash_before = sim.cash[0]

        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([40]), np.array([160.0]),
        )
        assert sim.positions[0, 0] == 60  # 100 - 40
        # avg_costs preserved on partial sells (matches scalar)
        assert sim.avg_costs[0, 0] == 150.0
        # entry_dates preserved
        assert sim.entry_dates[0, 0] == 0
        assert sim.highest_high[0, 0] == 150.0  # not reset
        # Cash + 40 * 160
        assert sim.cash[0] == cash_before + 40 * 160


class TestDrawdownMultiplier:
    def test_no_drawdown_returns_full_sizing(self):
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        # No price moves → no drawdown → multiplier = 1.0
        circuit_breaker = np.full(3, 0.08)
        tiers = [(-0.02, 0.75), (-0.04, 0.50), (-0.06, 0.25)]
        mult = sim.drawdown_multiplier(circuit_breaker, tiers)
        np.testing.assert_array_equal(mult, [1.0, 1.0, 1.0])

    def test_tier_lookup_matches_scalar(self):
        """Drawdown of -3% → tier 1 (0.75); -5% → tier 2 (0.50);
        -10% → circuit breaker (0.0)."""
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        # Manually inject drawdown by setting peak_nav above nav
        sim.peak_nav = np.array([1_000_000, 1_000_000, 1_000_000], dtype=np.float64)
        sim.nav = np.array([970_000, 950_000, 900_000], dtype=np.float64)
        # drawdown = [-0.03, -0.05, -0.10]

        circuit_breaker = np.full(3, 0.08)
        tiers = [(-0.02, 0.75), (-0.04, 0.50), (-0.06, 0.25)]
        mult = sim.drawdown_multiplier(circuit_breaker, tiers)
        # combo 0 (-3%): -3 ≤ -2 → tier 1 (0.75)
        # combo 1 (-5%): -5 ≤ -2 (0.75); -5 ≤ -4 (0.50). Last applies → 0.50
        # combo 2 (-10%): all tiers breached → tier 3 (0.25); but -10 ≤ -8 circuit → 0.0
        np.testing.assert_array_almost_equal(mult, [0.75, 0.50, 0.0])

    def test_per_combo_circuit_breaker(self):
        """Different combos can have different circuit breaker thresholds
        (when swept). Verify each combo's halt fires independently."""
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        sim.peak_nav = np.array([1_000_000, 1_000_000, 1_000_000], dtype=np.float64)
        sim.nav = np.array([950_000, 950_000, 950_000], dtype=np.float64)
        # All combos at -5% drawdown.

        # Combo 0: cb=0.04 → halts (5% > 4%)
        # Combo 1: cb=0.06 → tier (-0.04 mult applies, last that breaches)
        # Combo 2: cb=0.10 → no halt
        circuit_breaker = np.array([0.04, 0.06, 0.10])
        tiers = [(-0.02, 0.75), (-0.04, 0.50)]
        mult = sim.drawdown_multiplier(circuit_breaker, tiers)
        np.testing.assert_array_almost_equal(mult, [0.0, 0.50, 0.50])


class TestParityVsScalarSimulatedIBKRClient:
    """End-to-end byte-equal NAV evolution between vectorized and N
    independent scalar sim_clients."""

    @pytest.mark.skipif(
        not os.path.isdir(_EXECUTOR_ROOT),
        reason="alpha-engine sibling repo not present at ~/Development/alpha-engine",
    )
    def test_nav_evolution_matches_scalar_after_random_orders(self):
        from executor.ibkr import SimulatedIBKRClient

        rng = np.random.default_rng(42)
        ti = _ticker_index("AAPL", "MSFT", "NVDA", "AMZN")
        n_combos = 5
        init_cash = 1_000_000.0

        # Vectorized
        vsim = VectorizedSimulator(
            n_combos=n_combos, ticker_index=ti, init_cash=init_cash,
        )

        # Scalar reference: N independent SimulatedIBKRClient instances
        scalar_sims = [
            SimulatedIBKRClient(prices={}, nav=init_cash) for _ in range(n_combos)
        ]

        # Generate a random order stream + price walk for 30 dates.
        # Per-date invariant: no duplicate (combo, ticker) BUY or SELL
        # within the same batch (matches production orchestration —
        # one exit decision and one entry decision per (combo, ticker)).
        for date_idx in range(30):
            # Random prices
            prices_arr = 100 + rng.normal(0, 5, 4) + date_idx * 0.5
            prices_arr = np.maximum(prices_arr, 1.0)
            prices_dict = {t: float(prices_arr[i]) for t, i in ti.items()}

            # Update prices on scalar sims
            for s in scalar_sims:
                s._prices = prices_dict
                s._simulation_date = f"date_{date_idx}"

            # Generate ~3 BUY orders + ~2 SELL orders per date.
            # Track (combo, ticker) pairs to prevent duplicates within this batch.
            seen_buy: set = set()
            buy_combos, buy_tickers, buy_shares, buy_prices = [], [], [], []
            for _ in range(3):
                c = int(rng.integers(0, n_combos))
                t_idx = int(rng.integers(0, 4))
                key = (c, t_idx)
                if key in seen_buy:
                    continue
                seen_buy.add(key)
                shares = int(rng.integers(10, 100))
                ticker_str = list(ti.keys())[t_idx]
                if ticker_str in scalar_sims[c]._positions:
                    continue
                buy_combos.append(c)
                buy_tickers.append(t_idx)
                buy_shares.append(shares)
                buy_prices.append(prices_arr[t_idx])
                scalar_sims[c].place_market_order(ticker_str, "BUY", shares)

            seen_sell: set = set()
            sell_combos, sell_tickers, sell_shares_arr, sell_prices = [], [], [], []
            for _ in range(2):
                c = int(rng.integers(0, n_combos))
                ticker_str = list(scalar_sims[c]._positions.keys())
                if not ticker_str:
                    continue
                target = ticker_str[int(rng.integers(0, len(ticker_str)))]
                t_idx = ti[target]
                key = (c, t_idx)
                if key in seen_sell:
                    continue
                seen_sell.add(key)
                held = scalar_sims[c]._positions[target]["shares"]
                # Need held >= 10 for rng range [10, held+1). Partial sells
                # can reduce the position below 10; skip those.
                if held < 10:
                    continue
                shares = int(rng.integers(10, held + 1))  # may equal held → full exit
                sell_combos.append(c)
                sell_tickers.append(t_idx)
                sell_shares_arr.append(shares)
                sell_prices.append(prices_arr[t_idx])
                scalar_sims[c].place_market_order(target, "SELL", shares)

            # Apply to vectorized
            if buy_combos:
                vsim.apply_buy(
                    np.array(buy_combos), np.array(buy_tickers),
                    np.array(buy_shares), np.array(buy_prices),
                    date_idx=date_idx,
                )
            if sell_combos:
                vsim.apply_sell(
                    np.array(sell_combos), np.array(sell_tickers),
                    np.array(sell_shares_arr), np.array(sell_prices),
                )

            # Update NAV on both
            vsim.update_nav(prices_arr)
            scalar_navs = np.array([s.get_portfolio_nav() for s in scalar_sims])

            # Parity check at every date
            np.testing.assert_allclose(
                vsim.nav, scalar_navs,
                rtol=1e-9, atol=1e-9,
                err_msg=f"NAV drift at date {date_idx}: "
                f"vectorized={vsim.nav}, scalar={scalar_navs}",
            )
            # peak_nav also
            scalar_peaks = np.array([s._peak_nav for s in scalar_sims])
            np.testing.assert_allclose(
                vsim.peak_nav, scalar_peaks,
                rtol=1e-9, atol=1e-9,
                err_msg=f"peak_nav drift at date {date_idx}",
            )

    @pytest.mark.skipif(
        not os.path.isdir(_EXECUTOR_ROOT),
        reason="alpha-engine sibling repo not present",
    )
    def test_cash_position_state_matches_scalar(self):
        """After a fixed sequence of orders, both representations of
        per-combo state agree."""
        from executor.ibkr import SimulatedIBKRClient

        ti = _ticker_index("AAPL", "MSFT")
        vsim = VectorizedSimulator(n_combos=2, ticker_index=ti)
        scalar = [SimulatedIBKRClient(prices={"AAPL": 150.0, "MSFT": 300.0}, nav=1_000_000) for _ in range(2)]
        for s in scalar:
            s._simulation_date = "2024-01-01"

        # Combo 0: BUY AAPL 100 @ 150
        # Combo 1: BUY MSFT 50 @ 300
        scalar[0].place_market_order("AAPL", "BUY", 100)
        scalar[1].place_market_order("MSFT", "BUY", 50)
        vsim.apply_buy(
            np.array([0, 1]), np.array([0, 1]),
            np.array([100, 50]), np.array([150.0, 300.0]),
            date_idx=0,
        )

        # Cash matches
        for c in range(2):
            assert vsim.cash[c] == pytest.approx(scalar[c]._cash)

        # Move prices BEFORE the SELL so scalar's place_market_order
        # uses the same execution price as vectorized apply_sell.
        scalar[0]._prices = {"AAPL": 155.0, "MSFT": 300.0}
        scalar[1]._prices = {"AAPL": 155.0, "MSFT": 300.0}

        # Combo 0 SELLs AAPL fully
        scalar[0].place_market_order("AAPL", "SELL", 100)
        vsim.apply_sell(np.array([0]), np.array([0]), np.array([100]), np.array([155.0]))

        for c in range(2):
            assert vsim.cash[c] == pytest.approx(scalar[c]._cash), f"cash mismatch combo {c}"

        # Position state
        assert vsim.positions[0, 0] == 0  # combo 0 sold AAPL
        assert "AAPL" not in scalar[0]._positions
        assert vsim.positions[1, 1] == 50
        assert scalar[1]._positions["MSFT"]["shares"] == 50


# ── Fee-rate parity (added 2026-04-28 to close v17 absolute-stats gap) ──────


class TestFeeRate:
    """Pin per-side fee semantics. Mirrors vectorbt's
    `Portfolio.from_orders(fees=...)` convention: fees are a per-side
    fraction applied to BOTH buy and sell sides.
    """

    def test_default_fee_rate_is_zero_preserves_legacy_behavior(self):
        """Backward-compat invariant: tests + fixtures that don't pass
        fee_rate get zero-fee accounting (matches the pre-2026-04-28
        contract)."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000,
        )
        assert sim.fee_rate == 0.0
        sim.apply_buy(
            combo_idx=np.array([0]), ticker_idx=np.array([0]),
            shares=np.array([100]), prices=np.array([150.0]), date_idx=0,
        )
        # No fee → cash debit is exactly shares × price
        assert sim.cash[0] == 1_000_000 - 100 * 150

    def test_buy_with_fee_rate_deducts_extra(self):
        """fee_rate=0.001 → cash -= shares × price × 1.001."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        sim.apply_buy(
            combo_idx=np.array([0]), ticker_idx=np.array([0]),
            shares=np.array([100]), prices=np.array([150.0]), date_idx=0,
        )
        # 100 × 150 = 15000 notional; fee = 15000 × 0.001 = 15
        # cash debit = 15015
        expected = 1_000_000 - 15_000 - 15
        assert sim.cash[0] == pytest.approx(expected)

    def test_sell_with_fee_rate_credits_less(self):
        """fee_rate=0.001 → cash += shares × price × 0.999."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        # Buy 100 shares at $150
        sim.apply_buy(
            combo_idx=np.array([0]), ticker_idx=np.array([0]),
            shares=np.array([100]), prices=np.array([150.0]), date_idx=0,
        )
        cash_after_buy = sim.cash[0]
        # Sell at $160 — proceeds 16000 minus fee 16
        sim.apply_sell(
            combo_idx=np.array([0]), ticker_idx=np.array([0]),
            shares=np.array([100]), prices=np.array([160.0]),
        )
        expected_credit = 100 * 160 - 100 * 160 * 0.001
        assert sim.cash[0] == pytest.approx(cash_after_buy + expected_credit)

    def test_round_trip_at_flat_price_loses_two_fees(self):
        """Buy at $150 → Sell at $150 with fee_rate=0.001 should leave
        cash at init - 2 × (notional × fee_rate). Mirrors vectorbt's
        round-trip fee handling."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0,
        )
        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([100]), np.array([150.0]),
        )
        # Each side: 100 × 150 × 0.001 = 15. Round-trip total: 30.
        assert sim.cash[0] == pytest.approx(1_000_000 - 30)
        assert sim.positions[0, 0] == 0.0

    def test_fee_rate_applies_independently_per_combo(self):
        """All combos share one fee_rate; per-combo cash effects scale
        with each combo's order activity."""
        sim = VectorizedSimulator(
            n_combos=3, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        # Combo 0: 100 shares × $150
        # Combo 1: 200 shares × $100
        # Combo 2: untouched
        sim.apply_buy(
            combo_idx=np.array([0, 1]),
            ticker_idx=np.array([0, 0]),
            shares=np.array([100, 200]),
            prices=np.array([150.0, 100.0]),
            date_idx=0,
        )
        # Combo 0: 15000 notional + 15 fee = 15015
        # Combo 1: 20000 notional + 20 fee = 20020
        # Combo 2: untouched
        assert sim.cash[0] == pytest.approx(1_000_000 - 15015)
        assert sim.cash[1] == pytest.approx(1_000_000 - 20020)
        assert sim.cash[2] == pytest.approx(1_000_000)

    def test_partial_sell_applies_fee_to_actual_shares(self):
        """Sell half a position with fee → cash credit = (shares × price)
        × (1 - fee_rate) on the actual shares sold (not the requested)."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0,
        )
        cash_after_buy = sim.cash[0]
        # Sell 40 of 100 shares at $160
        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([40]), np.array([160.0]),
        )
        # Proceeds: 40 × 160 × 0.999 = 6396
        expected = cash_after_buy + 40 * 160 * 0.999
        assert sim.cash[0] == pytest.approx(expected)
        assert sim.positions[0, 0] == 60  # 100 - 40 remaining

    def test_oversell_applies_fee_to_held_amount(self):
        """Sell shares > held → only `held` shares actually transact,
        fee applies to the held amount (matches scalar
        SimulatedIBKRClient.place_market_order semantics)."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0,
        )
        cash_after_buy = sim.cash[0]
        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([200]), np.array([160.0]),
        )
        # Only 100 shares actually sold; fee on 100 × 160
        expected = cash_after_buy + 100 * 160 * 0.999
        assert sim.cash[0] == pytest.approx(expected)
        assert sim.positions[0, 0] == 0.0

    def test_round_trip_at_higher_price_yields_net_gain_minus_fees(self):
        """Realistic case: buy at $150, sell at $160. Gross gain = $1000
        on 100 shares. With 10 bps fees: net = 1000 - 15 (buy fee) - 16
        (sell fee) = 969."""
        sim = VectorizedSimulator(
            n_combos=1, ticker_index=_ticker_index("AAPL"),
            init_cash=1_000_000, fee_rate=0.001,
        )
        sim.apply_buy(
            np.array([0]), np.array([0]), np.array([100]), np.array([150.0]), 0,
        )
        sim.apply_sell(
            np.array([0]), np.array([0]), np.array([100]), np.array([160.0]),
        )
        # Net: 1_000_000 + (160 - 150) × 100 - 15 - 16 = 1_000_969
        expected = 1_000_000 + 1000 - 15 - 16
        assert sim.cash[0] == pytest.approx(expected)
