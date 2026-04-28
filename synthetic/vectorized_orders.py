"""Per-combo columnar order accumulator for the vectorized sweep.

Background: Tier 4 Layer 3 v15 (2026-04-28) caught an OOM kill on
c5.large after the vectorized sweep produced 1.49M order events
(238,956 entries + 1,248,812 exits) across 60 combos × 2500 dates ×
907 tickers. The sweep stored orders as `list[list[dict]]` — every
order a Python dict with 7 string keys → ~300 bytes per order →
~450 MB held simultaneously across all 60 combos at sweep end. The
post-sweep DataFrame construction tipped the c5.large (~3.7 GB
usable) over the kernel's OOM line.

This module replaces the in-loop accumulator with columnar storage
backed by `array.array` primitives (~64 bytes per order vs ~300),
materializing to the canonical per-combo dict-list ONLY when the
consumer reads a specific combo. Consumers free each combo's buffer
after stats are computed (`store.release(combo_idx)`), keeping
peak memory bounded to one materialized combo at a time.

Memory comparison (60 combos × 25k orders avg):
  list[list[dict]] : ~450 MB held simultaneously
  VectorizedOrderStore: ~90 MB at sweep end + ~7 MB peak materialization

The downstream contract (per-combo `list[dict]` shape) is preserved
exactly — `orders_per_combo[combo_idx]` returns a freshly-materialized
list with the same keys and value types as the prior implementation.
"""
from __future__ import annotations

from array import array
from typing import Sequence

import pandas as pd

from synthetic.vectorized_exits import (
    ACTION_EXIT,
    ACTION_REDUCE,
    REASON_ATR,
    REASON_FALLBACK,
    REASON_PROFIT,
    REASON_MOMENTUM,
    REASON_TIME_EXIT,
    REASON_TIME_REDUCE,
)


# Action code for ENTER orders. The exit pipeline's ACTION_EXIT=1 +
# ACTION_REDUCE=2 are reused; ENTER gets a distinct code so the
# materializer can dispatch correctly. ACTION_NONE=0 from
# vectorized_exits is intentionally not reused as ENTER — the exit
# code's "0 means no decision" semantic must stay separate from "0
# means an entry order was recorded".
ACTION_ENTER = 3


_EXIT_REASON_TO_STR: dict[int, str] = {
    REASON_ATR: "atr_trailing_stop",
    REASON_FALLBACK: "fallback_stop",
    REASON_PROFIT: "profit_take",
    REASON_MOMENTUM: "momentum_exit",
    REASON_TIME_EXIT: "time_decay_exit",
    REASON_TIME_REDUCE: "time_decay_reduce",
}


class _OrderBuffer:
    """Columnar storage for one combo's orders.

    Uses `array.array` (C-backed contiguous primitives, ~8 bytes per
    cell) instead of Python list-of-dict (~300 bytes per order).
    Append is O(1) amortized; materialization to dict-list is O(n)
    one-time at the consumer boundary.

    Schema (columns):
        date_idx (i, int32)         — index into the dates DatetimeIndex
        ticker_idx (i, int32)       — index into the tickers list
        action_code (b, int8)       — ACTION_ENTER / EXIT / REDUCE
        shares (q, int64)           — share count (signed; large
                                       positions possible)
        price (d, float64)          — fill price
        nav (d, float64)            — portfolio NAV at order time
        extra (d, float64)          — entries: position_pct;
                                       exits: float-encoded reason code
                                       (the union saves one column;
                                       reason codes fit cleanly in
                                       a float since they're small ints)
    """

    __slots__ = ("date_idx", "ticker_idx", "action_code", "shares",
                 "price", "nav", "extra")

    def __init__(self) -> None:
        self.date_idx = array("i")
        self.ticker_idx = array("i")
        self.action_code = array("b")
        self.shares = array("q")
        self.price = array("d")
        self.nav = array("d")
        self.extra = array("d")

    def add_entry(
        self,
        date_idx: int,
        ticker_idx: int,
        shares: int,
        price: float,
        nav: float,
        position_pct: float,
    ) -> None:
        self.date_idx.append(date_idx)
        self.ticker_idx.append(ticker_idx)
        self.action_code.append(ACTION_ENTER)
        self.shares.append(shares)
        self.price.append(price)
        self.nav.append(nav)
        self.extra.append(position_pct)

    def add_exit(
        self,
        date_idx: int,
        ticker_idx: int,
        action_code: int,
        shares: int,
        price: float,
        nav: float,
        reason_code: int,
    ) -> None:
        # action_code must be ACTION_EXIT or ACTION_REDUCE; caller's
        # responsibility to validate (vectorized_sweep already filters).
        self.date_idx.append(date_idx)
        self.ticker_idx.append(ticker_idx)
        self.action_code.append(action_code)
        self.shares.append(shares)
        self.price.append(price)
        self.nav.append(nav)
        self.extra.append(float(reason_code))

    def __len__(self) -> int:
        return len(self.date_idx)

    def to_dict_list(
        self,
        dates: pd.DatetimeIndex,
        tickers: Sequence[str],
    ) -> list[dict]:
        """Materialize to the canonical per-order dict shape.

        Keys match the legacy `list[list[dict]]` accumulator exactly:
          - ENTER: date, ticker, action="ENTER", shares,
                   price_at_order, portfolio_nav_at_order, position_pct
          - EXIT/REDUCE: date, ticker, action ("EXIT"|"REDUCE"), shares,
                         price_at_order, portfolio_nav_at_order,
                         exit_reason
        """
        n = len(self)
        if n == 0:
            return []

        # Pre-stringify dates ONCE per call — strftime is slow and the
        # buffer typically has many orders per date, so this saves
        # significant time vs per-row strftime.
        unique_date_idx = sorted(set(self.date_idx))
        date_str_cache = {
            i: dates[i].strftime("%Y-%m-%d") for i in unique_date_idx
        }

        out: list[dict] = []
        for i in range(n):
            ac = self.action_code[i]
            d_str = date_str_cache[self.date_idx[i]]
            ticker = tickers[self.ticker_idx[i]]
            if ac == ACTION_ENTER:
                out.append({
                    "date": d_str,
                    "ticker": ticker,
                    "action": "ENTER",
                    "shares": int(self.shares[i]),
                    "price_at_order": float(self.price[i]),
                    "portfolio_nav_at_order": float(self.nav[i]),
                    "position_pct": float(self.extra[i]),
                })
            elif ac == ACTION_EXIT:
                out.append({
                    "date": d_str,
                    "ticker": ticker,
                    "action": "EXIT",
                    "shares": int(self.shares[i]),
                    "price_at_order": float(self.price[i]),
                    "portfolio_nav_at_order": float(self.nav[i]),
                    "exit_reason": _EXIT_REASON_TO_STR.get(
                        int(self.extra[i]), "",
                    ),
                })
            elif ac == ACTION_REDUCE:
                out.append({
                    "date": d_str,
                    "ticker": ticker,
                    "action": "REDUCE",
                    "shares": int(self.shares[i]),
                    "price_at_order": float(self.price[i]),
                    "portfolio_nav_at_order": float(self.nav[i]),
                    "exit_reason": _EXIT_REASON_TO_STR.get(
                        int(self.extra[i]), "",
                    ),
                })
            # Unknown action_code: skip silently (defensive — should
            # never happen given vectorized_sweep's call sites). If
            # this becomes a problem, raise here instead.
        return out


class VectorizedOrderStore:
    """Per-combo columnar order accumulator with materialize-on-demand.

    Drop-in replacement for `list[list[dict]]` — supports `len()` and
    indexed access (`store[combo_idx]` returns a fresh `list[dict]`)
    so existing consumers iterate without changes.

    Production hot path: after consuming a combo's orders, call
    `store.release(combo_idx)` to drop the underlying buffer. This
    keeps peak memory bounded to one materialized combo at a time
    (~7 MB) plus the un-released buffers (~1.5 MB each).

    Tests don't need to release — buffers stay alive for re-inspection,
    and the in-buffer footprint is already 4-7× smaller than the prior
    list-of-dicts.
    """

    def __init__(self, n_combos: int) -> None:
        if n_combos < 0:
            raise ValueError(f"n_combos must be >= 0, got {n_combos}")
        self._buffers: list[_OrderBuffer | None] = [
            _OrderBuffer() for _ in range(n_combos)
        ]
        self._n_combos = n_combos
        # Late-bound: set via finalize() after the sweep loop knows
        # the final dates + ticker list.
        self._dates: pd.DatetimeIndex | None = None
        self._tickers: Sequence[str] | None = None

    def add_entry(
        self,
        combo_idx: int,
        date_idx: int,
        ticker_idx: int,
        shares: int,
        price: float,
        nav: float,
        position_pct: float,
    ) -> None:
        buf = self._buffers[combo_idx]
        if buf is None:
            raise RuntimeError(
                f"VectorizedOrderStore: combo {combo_idx} buffer was "
                f"already released; cannot add more orders. Caller "
                f"must release AFTER all per-combo orders are recorded."
            )
        buf.add_entry(date_idx, ticker_idx, shares, price, nav, position_pct)

    def add_exit(
        self,
        combo_idx: int,
        date_idx: int,
        ticker_idx: int,
        action_code: int,
        shares: int,
        price: float,
        nav: float,
        reason_code: int,
    ) -> None:
        buf = self._buffers[combo_idx]
        if buf is None:
            raise RuntimeError(
                f"VectorizedOrderStore: combo {combo_idx} buffer was "
                f"already released; cannot add more orders."
            )
        buf.add_exit(
            date_idx, ticker_idx, action_code, shares, price, nav,
            reason_code,
        )

    def finalize(
        self,
        dates: pd.DatetimeIndex,
        tickers: Sequence[str],
    ) -> None:
        """Set the lookups needed for materialization.

        Called once by `run_vectorized_sweep` after the loop completes,
        before the store is handed to the consumer.
        """
        self._dates = dates
        self._tickers = tickers

    def __len__(self) -> int:
        return self._n_combos

    def __getitem__(self, combo_idx: int) -> list[dict]:
        """Materialize one combo's orders to the canonical dict-list.

        Idempotent — calling multiple times for the same combo
        re-materializes from the buffer. Production callers should
        invoke `release(combo_idx)` after they're done with a combo
        to free the buffer; tests can skip that to inspect repeatedly.
        """
        if not (0 <= combo_idx < self._n_combos):
            raise IndexError(
                f"VectorizedOrderStore: combo_idx {combo_idx} out of "
                f"range [0, {self._n_combos})"
            )
        if self._dates is None or self._tickers is None:
            raise RuntimeError(
                "VectorizedOrderStore: finalize(dates, tickers) must "
                "be called before materializing any combo."
            )
        buf = self._buffers[combo_idx]
        if buf is None:
            return []  # released — caller treats as no-orders combo
        return buf.to_dict_list(self._dates, self._tickers)

    def release(self, combo_idx: int) -> None:
        """Drop the underlying buffer for one combo. Subsequent
        ``__getitem__`` returns ``[]``."""
        if 0 <= combo_idx < self._n_combos:
            self._buffers[combo_idx] = None

    def total_orders(self) -> int:
        """Sum of orders across all unreleased combos. Used for
        diagnostics + sweep-summary log lines."""
        return sum(len(b) for b in self._buffers if b is not None)
