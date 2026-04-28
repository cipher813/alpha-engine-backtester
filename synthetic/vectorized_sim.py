"""Matrix-first vectorized simulator (Tier 4 PR 1, 2026-04-27).

Path-dependent strategy backtester with combos as a numpy axis. All
state matrices are dense ``[n_combos, n_tickers]`` or ``[n_combos]``
arrays. Per-date evaluation is pure linear algebra (broadcasting,
matmul, masking); the time loop is the only sequential dimension
(irreducible due to path-dependent portfolio state evolution).

Replaces the scalar-per-combo ``SimulatedIBKRClient`` model used by
``_simulate_single_date`` for the predictor_param_sweep path. Live
executor + run_simulate (single-combo) still use the scalar deciders;
this module is backtester-only.

PR 1 scope (this commit):
  * State matrices: positions, cash, nav, peak_nav, entry_dates,
    avg_costs, highest_high
  * ``update_nav(prices)`` — mark-to-market via ``cash + positions @ prices``
  * ``drawdown_multiplier(circuit_breakers, tiers)`` — graduated tier
    lookup as numpy array selection
  * ``apply_buy(...)`` / ``apply_sell(...)`` — vectorized state mutation
    for use by Tier 4 PR 2 (exits) and PR 3 (entries)

Out of scope for PR 1:
  * Exit checks (ATR trailing, profit-take, momentum, time decay) — PR 2
  * Entry checks (already-held, score, sector cap, sizing, correlation) — PR 3
  * predictor_param_sweep wiring — PR 4

Linear algebra primitives used:
  * NAV: ``cash + positions @ prices``       — matmul
  * Sector exposure: ``(positions * prices) @ sector_one_hot``
  * Same-sector held mask: ``positions > 0`` AND ``sector_idx[ticker] == sector_idx[candidate]``
  * Drawdown: ``(nav - peak_nav) / peak_nav`` — broadcast subtract+divide
  * Tier multiplier: ``np.where`` cascade over sorted thresholds
  * Order application: direct assignment / scatter via boolean indexing

Memory footprint at production scale (60 combos × 911 tickers):
  * positions / avg_costs / highest_high: 3 × 60 × 911 × 8 bytes = ~1.3 MB
  * entry_dates: 60 × 911 × 4 bytes = ~220 KB
  * cash / nav / peak_nav: 3 × 60 × 8 bytes = 1.4 KB
  * Total: ~1.5 MB. Trivial.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for "ticker not held" in entry_dates matrix.
_NO_ENTRY = -1


@dataclass
class VectorizedSimulator:
    """Matrix-axis simulator for parameter-sweep backtesting.

    Constructor args
    ----------------
    n_combos : int
        Number of parameter-sweep combinations to run in parallel.
    ticker_index : dict[str, int]
        Maps ticker symbol to column index in state matrices. Built
        once at simulation setup. All tickers the simulator may
        encounter must be in this index — runtime additions are not
        supported (matrices are statically sized).
    init_cash : float
        Starting cash per combo. Default 1,000,000.

    State (initialized in __post_init__)
    ------------------------------------
    positions[combo, ticker] : float64
        Shares held. 0 means unheld.
    cash[combo] : float64
        Cash per combo.
    nav[combo] : float64
        Mark-to-market NAV. Updated by ``update_nav(prices)``.
    peak_nav[combo] : float64
        Running max of nav per combo. Updated each ``update_nav`` call.
    entry_dates[combo, ticker] : int32
        Date index (into the simulation's date axis) when the position
        was entered. ``_NO_ENTRY`` (-1) for unheld.
    avg_costs[combo, ticker] : float64
        Average cost basis. 0 for unheld.
    highest_high[combo, ticker] : float64
        Running max of ``high`` since entry, for ATR trailing-stop
        computation. 0 for unheld. Updated by ``update_nav`` (which
        also takes the daily high vector — see signature note below).

    Notes on ``highest_high``
    -------------------------
    The current ``update_nav`` accepts only ``prices`` (close). To
    update ``highest_high`` correctly, the caller must invoke
    ``update_highest_high(highs)`` separately each date. This split
    keeps ``update_nav`` minimal for PR 1; PR 2 (exits) needs both
    so will pass ``highs`` from the per-date OHLCV at that point.
    """

    n_combos: int
    ticker_index: dict
    init_cash: float = 1_000_000.0
    # Per-side fee fraction (e.g. 0.001 = 10 bps). Mirrors vectorbt's
    # `Portfolio.from_orders(fees=...)` semantics — applied to BOTH buys
    # and sells. apply_buy: cash -= shares × price × (1 + fee_rate).
    # apply_sell: cash += shares × price × (1 - fee_rate). Default 0.0
    # preserves prior test fixtures + sim contracts; production
    # callers (run_vectorized_sweep) read `simulation_fees` from config
    # (default 0.001, same key the scalar path reads). Closes the
    # 2026-04-28 v17 absolute-stats gap with scalar single_run.
    fee_rate: float = 0.0

    positions: np.ndarray = field(init=False)
    cash: np.ndarray = field(init=False)
    nav: np.ndarray = field(init=False)
    peak_nav: np.ndarray = field(init=False)
    entry_dates: np.ndarray = field(init=False)
    avg_costs: np.ndarray = field(init=False)
    highest_high: np.ndarray = field(init=False)

    @property
    def n_tickers(self) -> int:
        return len(self.ticker_index)

    def __post_init__(self) -> None:
        if self.n_combos <= 0:
            raise ValueError(f"n_combos must be positive, got {self.n_combos}")
        if not self.ticker_index:
            raise ValueError("ticker_index must be non-empty")
        # Validate ticker_index has 0..N-1 column indices
        n = self.n_tickers
        idx_values = sorted(self.ticker_index.values())
        if idx_values != list(range(n)):
            raise ValueError(
                f"ticker_index values must be 0..{n-1}, got {idx_values[:5]}..."
            )

        m = n
        c = self.n_combos
        init = float(self.init_cash)

        self.positions = np.zeros((c, m), dtype=np.float64)
        self.cash = np.full(c, init, dtype=np.float64)
        self.nav = np.full(c, init, dtype=np.float64)
        self.peak_nav = np.full(c, init, dtype=np.float64)
        self.entry_dates = np.full((c, m), _NO_ENTRY, dtype=np.int32)
        self.avg_costs = np.zeros((c, m), dtype=np.float64)
        self.highest_high = np.zeros((c, m), dtype=np.float64)

    # ── Per-date state updates ────────────────────────────────────────

    def update_nav(self, prices: np.ndarray) -> None:
        """Mark-to-market NAV per combo. Updates ``nav`` and
        ``peak_nav`` in place.

        prices : np.ndarray of shape ``[n_tickers]`` — close prices for
                 the current date, indexed by the ticker_index column.
                 NaN tolerated for unpriced tickers (their contribution
                 to NAV via held positions is treated as 0).

        Computation:
            nav = cash + positions @ prices_safe
        where ``prices_safe`` replaces NaN with 0 (unheld positions
        already have ``positions[combo, ticker] = 0``, but we want
        held positions with missing prices to not corrupt NAV either).
        Real production: missing prices for held positions should be
        rare — a held ticker without a price means data outage,
        which the caller should surface separately.
        """
        if prices.shape != (self.n_tickers,):
            raise ValueError(
                f"prices shape mismatch: expected ({self.n_tickers},), "
                f"got {prices.shape}"
            )
        # Replace NaN with 0 for matmul. positions for unheld tickers
        # are already 0 so they don't contribute regardless.
        prices_safe = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        # nav = cash + positions @ prices  — matmul, the linear-algebra core
        np.matmul(self.positions, prices_safe, out=self.nav)
        self.nav += self.cash
        np.maximum(self.peak_nav, self.nav, out=self.peak_nav)

    def update_highest_high(self, highs: np.ndarray) -> None:
        """Update ``highest_high`` matrix for held positions.

        highs : np.ndarray of shape ``[n_tickers]`` — daily high prices
                for the current date.

        For each ``(combo, ticker)`` where ``positions > 0``, update
        ``highest_high[combo, ticker] = max(highest_high[combo, ticker],
        high[ticker])``. For unheld positions, ``highest_high`` stays 0
        (it's reset to 0 on exit and to the entry price on entry).
        """
        if highs.shape != (self.n_tickers,):
            raise ValueError(
                f"highs shape mismatch: expected ({self.n_tickers},), "
                f"got {highs.shape}"
            )
        held_mask = self.positions > 0
        # Broadcast: highs[None, :] is [1, n_tickers]; held_mask is
        # [n_combos, n_tickers]. np.where selects elementwise.
        candidate_high = np.broadcast_to(highs[None, :], self.highest_high.shape)
        np.maximum(
            self.highest_high,
            np.where(held_mask, candidate_high, self.highest_high),
            out=self.highest_high,
        )

    # ── Drawdown ──────────────────────────────────────────────────────

    def drawdown(self) -> np.ndarray:
        """Current drawdown per combo as a fraction (negative when
        below peak). Shape: ``[n_combos]``."""
        return np.where(
            self.peak_nav > 0,
            (self.nav - self.peak_nav) / self.peak_nav,
            0.0,
        )

    def drawdown_multiplier(
        self,
        circuit_breaker_per_combo: np.ndarray,
        tiers: list,
    ) -> np.ndarray:
        """Compute graduated drawdown sizing multiplier per combo.

        circuit_breaker_per_combo : np.ndarray ``[n_combos]``
            Per-combo absolute drawdown threshold (positive number).
            E.g. 0.08 means halt at -8% drawdown.
        tiers : list of ``(threshold, multiplier)`` pairs
            Sorted by threshold descending (least negative first), e.g.
            ``[(-0.02, 0.75), (-0.04, 0.50), (-0.06, 0.25)]``.
            Tiers are NOT swept across combos in current production;
            same tier list applies to all combos.

        Returns
        -------
        multiplier : np.ndarray ``[n_combos]`` in [0, 1]
            1.0 = full sizing (drawdown shallower than all tiers)
            0.0 = circuit breaker (drawdown <= -circuit_breaker)
            else: deepest tier whose threshold has been breached

        Mirrors ``executor.risk_guard.compute_drawdown_multiplier`` —
        per-combo equivalent computed via numpy ``np.where`` cascade.
        Live executor still calls the scalar version; vectorized sim
        uses this for parity.
        """
        if circuit_breaker_per_combo.shape != (self.n_combos,):
            raise ValueError(
                f"circuit_breaker_per_combo shape mismatch: expected "
                f"({self.n_combos},), got {circuit_breaker_per_combo.shape}"
            )

        dd = self.drawdown()  # [n_combos]
        multiplier = np.ones(self.n_combos, dtype=np.float64)
        # Walk through tiers; later tiers (deeper) overwrite earlier
        # multipliers. Matches the scalar reference's "last tier whose
        # threshold is breached applies."
        for threshold, tier_mult in tiers:
            multiplier = np.where(dd <= threshold, tier_mult, multiplier)
        # Hard halt at circuit breaker: drawdown <= -circuit_breaker.
        # Matches scalar reference's hard_halt logic.
        multiplier = np.where(dd <= -circuit_breaker_per_combo, 0.0, multiplier)
        return multiplier

    # ── State mutation primitives (for PR 2 exits + PR 3 entries) ────

    def apply_buy(
        self,
        combo_idx: np.ndarray,
        ticker_idx: np.ndarray,
        shares: np.ndarray,
        prices: np.ndarray,
        date_idx: int,
    ) -> None:
        """Apply BUY orders to state matrices.

        combo_idx, ticker_idx, shares, prices : np.ndarray, all same shape ``[k]``
            Each element is one BUY order: combo ``combo_idx[i]`` buys
            ``shares[i]`` of ticker ``ticker_idx[i]`` at ``prices[i]``.
        date_idx : int
            Date index for entry_dates assignment.

        State updates (matches SimulatedIBKRClient.place_market_order BUY):
          positions[combo, ticker] = shares (assignment — no add)
          avg_costs[combo, ticker] = price
          entry_dates[combo, ticker] = date_idx
          highest_high[combo, ticker] = price (entry price seeds the running max)
          cash[combo] -= shares * price (accumulated via np.add.at for
                                          duplicate combo handling)

        Note: BUY assignment matches scalar SimulatedIBKRClient
        semantics (re-buying the same ticker overwrites the position).
        In practice the "already in portfolio" gate prevents this in
        steady-state; defensive coverage included.
        """
        if not (combo_idx.shape == ticker_idx.shape == shares.shape == prices.shape):
            raise ValueError("BUY input shape mismatch across orders")
        if combo_idx.size == 0:
            return
        self.positions[combo_idx, ticker_idx] = shares.astype(np.float64)
        self.avg_costs[combo_idx, ticker_idx] = prices.astype(np.float64)
        self.entry_dates[combo_idx, ticker_idx] = np.int32(date_idx)
        self.highest_high[combo_idx, ticker_idx] = prices.astype(np.float64)
        # cash debit: scatter-subtract per combo. np.add.at handles
        # duplicate combo_idx entries correctly (a single combo doing
        # multiple BUYs in one date — rare but possible).
        # Fees applied as `shares × price × (1 + fee_rate)` to mirror
        # vectorbt's per-side fee semantics; fee_rate=0.0 (default)
        # preserves the legacy zero-fee accounting.
        notional = shares.astype(np.float64) * prices.astype(np.float64)
        if self.fee_rate != 0.0:
            np.add.at(self.cash, combo_idx, -(notional * (1.0 + self.fee_rate)))
        else:
            np.add.at(self.cash, combo_idx, -notional)

    def apply_sell(
        self,
        combo_idx: np.ndarray,
        ticker_idx: np.ndarray,
        shares: np.ndarray,
        prices: np.ndarray,
    ) -> None:
        """Apply SELL orders to state matrices.

        combo_idx, ticker_idx, shares, prices : np.ndarray, shape ``[k]``
            Each element is one SELL order. ``shares[i]`` is the
            number of shares to sell.

        State updates (matches SimulatedIBKRClient.place_market_order SELL):
          held = positions[combo, ticker]
          if shares >= held: full exit — positions, avg_costs,
                             entry_dates, highest_high all reset
          else:              partial reduce — positions[combo, ticker] -= shares
          cash[combo] += shares * price (proceeds, scatter-add)

        Per scalar reference: when a SELL exhausts a position the
        position dict entry is removed; partial sells just reduce shares.
        """
        if not (combo_idx.shape == ticker_idx.shape == shares.shape == prices.shape):
            raise ValueError("SELL input shape mismatch across orders")
        if combo_idx.size == 0:
            return
        # Guard: numpy fancy-indexed assignment is last-write-wins on
        # duplicate (combo, ticker) pairs, which would silently corrupt
        # state if two SELL orders target the same position. Production
        # orchestration (vectorized exits in PR 2) emits at most one
        # exit per (combo, ticker) per date so duplicates shouldn't
        # arise. Fail loudly if they do — preferable to a silent state
        # divergence that drifts from scalar parity over many dates.
        flat = combo_idx.astype(np.int64) * self.n_tickers + ticker_idx.astype(np.int64)
        if len(set(flat.tolist())) != len(flat):
            raise ValueError(
                "apply_sell: duplicate (combo, ticker) pair in single batch — "
                "numpy fancy-indexed assignment would corrupt state. Caller "
                "must aggregate shares per (combo, ticker) before calling."
            )
        held = self.positions[combo_idx, ticker_idx]
        sell_shares = shares.astype(np.float64)
        full_exit_mask = sell_shares >= held
        # Partial sell: reduce by sell_shares
        self.positions[combo_idx, ticker_idx] = np.where(
            full_exit_mask, 0.0, held - sell_shares,
        )
        # Full-exit: reset avg_costs / entry_dates / highest_high
        if np.any(full_exit_mask):
            full_combo = combo_idx[full_exit_mask]
            full_ticker = ticker_idx[full_exit_mask]
            self.avg_costs[full_combo, full_ticker] = 0.0
            self.entry_dates[full_combo, full_ticker] = _NO_ENTRY
            self.highest_high[full_combo, full_ticker] = 0.0
        # Cash credit: scatter-add proceeds. Use the lesser of
        # sell_shares vs held (you can't sell more than you hold —
        # scalar reference clips this implicitly via the held check).
        # Fees applied as `shares × price × (1 - fee_rate)` to mirror
        # vectorbt's per-side fee semantics on the sell side.
        actual_shares = np.where(full_exit_mask, held, sell_shares)
        proceeds = actual_shares * prices.astype(np.float64)
        if self.fee_rate != 0.0:
            proceeds = proceeds * (1.0 - self.fee_rate)
        np.add.at(self.cash, combo_idx, proceeds)

    # ── Convenience accessors ────────────────────────────────────────

    def held_mask(self) -> np.ndarray:
        """Boolean ``[n_combos, n_tickers]`` mask: True where shares > 0."""
        return self.positions > 0

    def n_held_per_combo(self) -> np.ndarray:
        """Count of held positions per combo. Shape ``[n_combos]``."""
        return self.held_mask().sum(axis=1).astype(np.int32)

    def __repr__(self) -> str:
        return (
            f"VectorizedSimulator(n_combos={self.n_combos}, "
            f"n_tickers={self.n_tickers}, "
            f"avg_held={self.n_held_per_combo().mean():.1f})"
        )
