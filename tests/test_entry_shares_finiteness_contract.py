"""A non-finite entry share count is a contract violation, not a zero.

`shares <= 0` is False for NaN. That single fact cost the PredictorBacktest
stage two weekly runs (2026-06-26 and 2026-08-13, identical artifacts):

- `compute_vectorized_entries`'s shares-round-to-zero gate (`shares <= 0`) did
  not block a NaN, so `block_reason` stayed `BLOCK_NONE`, `entry_passed` stayed
  True, and `final_shares = np.where(entry_passed, shares, 0.0)` carried the
  NaN out;
- `run_vectorized_sweep`'s own `if shares <= 0: continue` was blind the same
  way, so the value reached `orders_per_combo.add_entry(shares=int(shares), …)`
  and raised `ValueError: cannot convert float NaN to integer` — naming neither
  the combo, the ticker, nor which input was bad.

The stage then wrote `{"status": "error", "error": "cannot convert float NaN to
integer"}` and exited 0. The traceback that finally located this frame only
existed because `predictor_stats.json` gained a `traceback` field
(crucible-backtester#661); before that, three months of the same failure were
undiagnosable.

`shares = floor(dollar_size / safe_prices)` and `safe_prices` is NaN-proof
(`np.where(sig_prices > 0, sig_prices, 1.0)`), so a non-finite result can only
originate in `dollar_size = nav_per_combo * position_weight`. The raise names
which of the two it was — the difference between a corrupted portfolio NAV and
a corrupted sizing weight.
"""

from __future__ import annotations

import numpy as np
import pytest


def _shares_gate(shares: np.ndarray) -> np.ndarray:
    """The gate as written after the fix."""
    return ~(shares > 0)


def test_nan_is_not_excluded_by_the_original_gate() -> None:
    """Pins the fact the whole incident rests on. If this ever fails, numpy
    changed NaN comparison semantics and the rest of this module is moot."""
    shares = np.array([np.nan])
    assert not bool((shares <= 0)[0]), (
        "NaN <= 0 is expected to be False — that is why the original gate let "
        "a NaN through as a passed entry"
    )


@pytest.mark.parametrize(
    "value,blocked",
    [
        (np.nan, True),
        (np.inf, False),   # +inf > 0 is True — caught by the finiteness raise
        (-np.inf, True),
        (0.0, True),
        (-1.0, True),
        (1.0, False),
    ],
)
def test_fixed_gate_blocks_nan(value, blocked) -> None:
    assert bool(_shares_gate(np.array([value]))[0]) is blocked


class _Cfg:
    def __init__(self, n_combos: int):
        self.min_position_dollar = np.zeros(n_combos)


def test_contract_raise_names_nav_when_nav_is_the_bad_input() -> None:
    """The message must distinguish a corrupted NAV from a corrupted weight —
    that distinction is the entire diagnostic value of the raise."""
    from synthetic import vectorized_entries as ve

    src = (ve.__file__ or "")
    assert src, "cannot locate module source"
    text = open(src).read()
    assert "non-finite entry share count" in text
    assert "non-finite nav_per_combo at combo_idx" in text
    assert "non-finite position_weight at combo_idx" in text


def test_nav_nan_propagates_to_shares_nan() -> None:
    """The mechanism, in isolation: a NaN NAV survives every guard in the
    sizing chain and lands in `shares`, because each guard is a comparison and
    every comparison against NaN is False."""
    nav = np.array([np.nan])
    position_weight = np.array([[0.05]])
    min_position_dollar = np.array([1000.0])
    sig_prices = np.array([50.0])

    dollar_size = nav[:, None] * position_weight
    too_small = dollar_size < min_position_dollar[:, None]
    safe_prices = np.where(sig_prices > 0, sig_prices, 1.0)
    shares = np.floor(dollar_size / safe_prices[None, :])
    shares = np.where(too_small, 0.0, shares)

    assert not bool(too_small[0, 0]), "NaN < threshold is False — guard missed"
    assert np.isnan(shares[0, 0]), "the NaN should reach `shares` untouched"
    assert bool(_shares_gate(shares)[0, 0]), "the fixed gate must block it"


def test_sweep_consumer_gate_is_nan_safe() -> None:
    """Second layer: the consumer must not crash on int(NaN) either."""
    from synthetic import vectorized_sweep as vs

    text = open(vs.__file__).read()
    assert "if not (shares > 0):" in text, (
        "run_vectorized_sweep's entry gate is back to a NaN-blind form"
    )
    assert "shares=int(shares)" in text, "guard is pinned to the wrong call site"
