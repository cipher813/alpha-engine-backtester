"""Vectorized entry decisions (Tier 4 PR 3, 2026-04-27).

Implements the entry pipeline from
``executor.deciders.decide_entries`` (which wraps
``executor.position_sizer.compute_position_size`` and
``executor.risk_guard.check_order``) as matrix ops on
``VectorizedSimulator`` state.

Per-(combo, signal) gates evaluated as boolean matrices
``[n_combos, n_signals]``:

  1. Already held — positions[c, signal_ticker_idx[s]] > 0
  2. Score gate — signal_score[s] >= min_score_per_combo[c]
  3. Momentum gate — momentum_at_date[s] >= momentum_gate_per_combo[c]
                     (skipped per-combo when ``momentum_gate_enabled`` is False)
  4. Drawdown halt — dd_multiplier[c] > 0
  5. Bear regime + underweight sector block
  6. GBM veto — predictions[s].gbm_veto (per-signal, identical across combos)
  7. Position size cap — dollar_size[c, s] / nav[c] <= effective_max_pct[c]
  8. Sector cap — sector_exposure[c, signal_sector[s]] + new[c, s]
                  <= max_sector_pct[c] × nav[c]
  9. Equity cap — total_equity[c] + new[c, s] <= max_equity_pct[c] × nav[c]
 10. Correlation block — mean(corr(candidate, same-sector held in combo c))
                         <= correlation_block_threshold[c]
 11. Shares-round-to-zero — shares[c, s] >= 1 AND
                            dollar_size[c, s] >= min_position_dollar[c]

Entry passes ⇔ all gates pass. State updates via ``apply_buy``.

Sizing pipeline matches scalar ``compute_position_size``:

  base_weight  = 1 / n_signals
  raw_weight   = base × sector_adj × conviction_adj × upside_adj
                  × dd_multiplier × atr_adj × confidence_adj
                  × p_up_adj (optional blend) × staleness_adj
                  × earnings_adj × coverage_adj
  position_pct = min(raw_weight, max_pct)
  ATR cap (when atr_adj < 1.0):
              position_pct = min(position_pct, base × atr_adj × dd_mult, max_pct)
  dollar_size = nav × position_pct
  shares      = floor(dollar_size / price)
  if dollar_size < min_position_dollar: shares = 0

Carryover infrastructure:
  * VectorizedSimulator (PR 1) — state matrices read here
  * FeatureLookup (Tier 3 Part B) — momentum + correlation returns source

Future PRs (sweep wiring, PR 4) will assemble per-date inputs from
SignalLookup + FeatureLookup + research signals JSON.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Block reason codes — surfaced for shadow-book persistence.
BLOCK_NONE = 0
BLOCK_ALREADY_HELD = 1
BLOCK_SCORE = 2
BLOCK_MOMENTUM_GATE = 3
BLOCK_DRAWDOWN_HALT = 4
BLOCK_BEAR_UNDERWEIGHT = 5
BLOCK_GBM_VETO = 6
BLOCK_POSITION_CAP = 7
BLOCK_SECTOR_CAP = 8
BLOCK_EQUITY_CAP = 9
BLOCK_CORRELATION = 10
BLOCK_SHARES_ZERO = 11
BLOCK_NO_PRICE = 12


# Conviction codes — caller translates string conviction to these.
CONV_RISING = 0
CONV_STABLE = 1
CONV_DECLINING = 2

# Sector-rating codes.
SR_OVERWEIGHT = 0
SR_MARKET_WEIGHT = 1
SR_UNDERWEIGHT = 2

# Market regime codes (one per simulation, not per combo).
REGIME_BULL = 0
REGIME_NEUTRAL = 1
REGIME_BEAR = 2
REGIME_CAUTION = 3


@dataclass(frozen=True)
class VectorizedEntryConfig:
    """Per-combo entry-decision parameters.

    Each field is shape ``[n_combos]``. All numeric thresholds + multipliers
    that downstream gates consume are arrays so the same loop body
    evaluates 60 combo-specific decisions at once.

    Build via ``from_uniform`` for tests / single-combo parity, or
    construct directly for actual sweeps.
    """

    # Score gate
    min_score_to_enter: np.ndarray            # [n_combos]

    # Momentum confirmation gate
    momentum_gate_enabled: np.ndarray         # bool
    momentum_gate_threshold: np.ndarray       # float (e.g. -5.0 = -5% 20d)

    # Position size caps
    max_position_pct: np.ndarray
    bear_max_position_pct: np.ndarray         # used when regime == BEAR
    max_sector_pct: np.ndarray
    max_equity_pct: np.ndarray

    # Drawdown halt: handled by VectorizedSimulator.drawdown_multiplier;
    # entry pipeline reads `dd_multiplier` from caller.

    # Bear regime gate
    bear_block_underweight: np.ndarray        # bool

    # Sizing multipliers
    sector_adj_overweight: np.ndarray
    sector_adj_market_weight: np.ndarray
    sector_adj_underweight: np.ndarray
    conviction_decline_adj: np.ndarray
    upside_fail_adj: np.ndarray
    min_price_target_upside: np.ndarray

    # ATR sizing
    atr_sizing_enabled: np.ndarray            # bool
    atr_sizing_target_risk: np.ndarray
    atr_sizing_floor: np.ndarray              # min atr_adj (default 0.5)
    atr_sizing_ceiling: np.ndarray            # max atr_adj (default 1.5)

    # Confidence sizing
    confidence_sizing_enabled: np.ndarray     # bool
    confidence_sizing_min: np.ndarray
    confidence_sizing_range: np.ndarray

    # p_up sizing (Phase 4d)
    use_p_up_sizing: np.ndarray               # bool
    p_up_sizing_blend: np.ndarray

    # Staleness discount
    staleness_discount_enabled: np.ndarray    # bool
    signal_cadence_days: np.ndarray
    staleness_decay_per_day: np.ndarray
    staleness_floor: np.ndarray

    # Earnings sizing
    earnings_sizing_enabled: np.ndarray       # bool
    earnings_proximity_days: np.ndarray
    earnings_sizing_reduction: np.ndarray

    # Feature-coverage derate
    coverage_sizing_enabled: np.ndarray       # bool
    coverage_derate_floor: np.ndarray

    # Min position dollar
    min_position_dollar: np.ndarray

    # Correlation block
    correlation_block_enabled: np.ndarray     # bool
    correlation_block_threshold: np.ndarray
    correlation_lookback_days: np.ndarray     # int — caller must build returns_window matrix to this length

    @property
    def n_combos(self) -> int:
        return int(self.min_score_to_enter.shape[0])

    @classmethod
    def from_uniform(cls, n_combos: int, **overrides) -> "VectorizedEntryConfig":
        """Build a config with all combos sharing scalar defaults.

        Defaults match ``executor.position_sizer._DEFAULT_SECTOR_ADJ``
        + ``executor.risk_guard.check_order`` + sweep DEFAULT_GRID + the
        scalar fallback constants in ``compute_position_size``.
        """
        defaults = {
            "min_score_to_enter": 70.0,
            "momentum_gate_enabled": True,
            "momentum_gate_threshold": -5.0,
            "max_position_pct": 0.05,
            "bear_max_position_pct": 0.025,
            "max_sector_pct": 0.25,
            "max_equity_pct": 0.90,
            "bear_block_underweight": True,
            "sector_adj_overweight": 1.05,
            "sector_adj_market_weight": 1.00,
            "sector_adj_underweight": 0.85,
            "conviction_decline_adj": 0.70,
            "upside_fail_adj": 0.70,
            "min_price_target_upside": 0.05,
            "atr_sizing_enabled": True,
            "atr_sizing_target_risk": 0.02,
            "atr_sizing_floor": 0.5,
            "atr_sizing_ceiling": 1.5,
            "confidence_sizing_enabled": True,
            "confidence_sizing_min": 0.7,
            "confidence_sizing_range": 0.6,
            "use_p_up_sizing": False,
            "p_up_sizing_blend": 0.3,
            "staleness_discount_enabled": True,
            "signal_cadence_days": 7,
            "staleness_decay_per_day": 0.03,
            "staleness_floor": 0.70,
            "earnings_sizing_enabled": True,
            "earnings_proximity_days": 5,
            "earnings_sizing_reduction": 0.50,
            "coverage_sizing_enabled": True,
            "coverage_derate_floor": 0.25,
            "min_position_dollar": 500.0,
            "correlation_block_enabled": True,
            "correlation_block_threshold": 0.80,
            "correlation_lookback_days": 60,
        }
        defaults.update(overrides)

        bool_fields = {
            "momentum_gate_enabled", "bear_block_underweight",
            "atr_sizing_enabled", "confidence_sizing_enabled",
            "use_p_up_sizing", "staleness_discount_enabled",
            "earnings_sizing_enabled", "coverage_sizing_enabled",
            "correlation_block_enabled",
        }
        int_fields = {
            "signal_cadence_days", "earnings_proximity_days",
            "correlation_lookback_days",
        }

        kwargs = {}
        for k, v in defaults.items():
            if k in bool_fields:
                kwargs[k] = np.full(n_combos, bool(v), dtype=bool)
            elif k in int_fields:
                kwargs[k] = np.full(n_combos, int(v), dtype=np.int32)
            else:
                kwargs[k] = np.full(n_combos, float(v), dtype=np.float64)
        return cls(**kwargs)


@dataclass
class EntryDecisions:
    """Result of ``compute_vectorized_entries``.

    entry_passed : bool[n_combos, n_signals]
    entry_shares : float64[n_combos, n_signals]
    entry_dollar : float64[n_combos, n_signals]
    block_reason : int8[n_combos, n_signals]   — BLOCK_* code (0 if passed)
    """

    entry_passed: np.ndarray
    entry_shares: np.ndarray
    entry_dollar: np.ndarray
    block_reason: np.ndarray


# ────────────────────────────────────────────────────────────────────
# Sizing helpers (vectorized scalars across combos × signals)
# ────────────────────────────────────────────────────────────────────


def _compute_sector_adj(
    rating_codes: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Sector rating adjustment per (combo, signal).

    rating_codes : int8[n_signals] — SR_* codes
    Returns float64[n_combos, n_signals] using a per-combo lookup.
    """
    # [n_combos, 3] table indexed by SR code.
    table = np.stack([
        config.sector_adj_overweight,
        config.sector_adj_market_weight,
        config.sector_adj_underweight,
    ], axis=1)  # [n_combos, 3]
    # Default any out-of-range code to market_weight (1.0).
    safe_codes = np.clip(rating_codes, 0, 2)
    return table[:, safe_codes]  # [n_combos, n_signals]


def _compute_conviction_adj(
    conviction_codes: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """1.0 for rising/stable, conviction_decline_adj[c] for declining."""
    declining = (conviction_codes == CONV_DECLINING)  # [n_signals]
    return np.where(
        declining[None, :],
        config.conviction_decline_adj[:, None],
        1.0,
    )


def _compute_upside_adj(
    upsides: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Apply upside_fail_adj[c] when signal upside < min_price_target_upside[c].

    upsides : float64[n_signals], NaN means missing → no penalty.
    """
    upside_known = ~np.isnan(upsides)
    fail_mask = upside_known[None, :] & (
        upsides[None, :] < config.min_price_target_upside[:, None]
    )
    return np.where(fail_mask, config.upside_fail_adj[:, None], 1.0)


def _compute_atr_adj(
    atr_pct: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """ATR-based sizing adjustment per (combo, signal).

    atr_pct : float64[n_signals] — ATR as fraction of price; NaN → no adj.
    """
    enabled = config.atr_sizing_enabled[:, None]
    valid = ~np.isnan(atr_pct) & (atr_pct > 0)
    safe_atr = np.where(valid, atr_pct, 1.0)  # avoid div/0 in inactive cells
    raw = config.atr_sizing_target_risk[:, None] / safe_atr[None, :]
    clamped = np.minimum(
        np.maximum(raw, config.atr_sizing_floor[:, None]),
        config.atr_sizing_ceiling[:, None],
    )
    return np.where(enabled & valid[None, :], clamped, 1.0)


def _compute_confidence_adj(
    confidence: np.ndarray, p_up: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Confidence-weighted sizing per (combo, signal).

    confidence : float64[n_signals] — clamped to [0, 1]; NaN → no conf adj.
    p_up       : float64[n_signals] — clamped to [0, 1]; NaN skips p_up blend.
    """
    conf_enabled = config.confidence_sizing_enabled[:, None]
    conf_known = ~np.isnan(confidence)
    clamped_conf = np.clip(np.where(conf_known, confidence, 0.0), 0.0, 1.0)
    base_conf_adj = (
        config.confidence_sizing_min[:, None]
        + config.confidence_sizing_range[:, None] * clamped_conf[None, :]
    )
    conf_adj = np.where(conf_enabled & conf_known[None, :], base_conf_adj, 1.0)

    # p_up blend (Phase 4d) — applies only when use_p_up_sizing is True.
    p_up_enabled = config.use_p_up_sizing[:, None]
    p_up_known = ~np.isnan(p_up)
    clamped_pu = np.clip(np.where(p_up_known, p_up, 0.0), 0.0, 1.0)
    p_up_adj = 0.7 + 0.6 * clamped_pu  # [n_signals]
    blend = config.p_up_sizing_blend[:, None]
    blended = conf_adj * (1 - blend) + p_up_adj[None, :] * blend
    return np.where(p_up_enabled & p_up_known[None, :], blended, conf_adj)


def _compute_staleness_adj(
    signal_age_days: int, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Staleness discount per combo (signal_age_days is global per-date).

    Returns [n_combos, 1] for broadcasting with [n_combos, n_signals].
    """
    enabled = config.staleness_discount_enabled
    cadence = config.signal_cadence_days
    decay = config.staleness_decay_per_day
    floor = config.staleness_floor
    effective_age = np.maximum(0, signal_age_days - cadence).astype(np.float64)
    raw = np.maximum(1.0 - decay * effective_age, floor)
    val = np.where(enabled & (effective_age > 0), raw, 1.0)
    return val[:, None]  # broadcast over n_signals


def _compute_earnings_adj(
    days_to_earnings: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Earnings-proximity sizing reduction per (combo, signal).

    days_to_earnings : int32[n_signals] — -1 for unknown.
    """
    known = days_to_earnings >= 0
    proximate = known[None, :] & (
        days_to_earnings[None, :] <= config.earnings_proximity_days[:, None]
    )
    enabled = config.earnings_sizing_enabled[:, None]
    return np.where(
        enabled & proximate,
        1.0 - config.earnings_sizing_reduction[:, None],
        1.0,
    )


def _compute_coverage_adj(
    feature_coverage: np.ndarray, config: VectorizedEntryConfig,
) -> np.ndarray:
    """Feature-coverage derate per (combo, signal).

    feature_coverage : float64[n_signals] — fraction of non-NaN features
                       in [0, 1]; NaN means coverage info unavailable.
    """
    known = ~np.isnan(feature_coverage)
    clamped = np.clip(np.where(known, feature_coverage, 1.0), 0.0, 1.0)
    floored = np.maximum(clamped[None, :], config.coverage_derate_floor[:, None])
    enabled = config.coverage_sizing_enabled[:, None]
    return np.where(enabled & known[None, :], floored, 1.0)


# ────────────────────────────────────────────────────────────────────
# Main entry-decision pipeline
# ────────────────────────────────────────────────────────────────────


def compute_vectorized_entries(
    sim,
    *,
    # Per-signal inputs (one element per ENTER candidate today).
    signal_ticker_idx: np.ndarray,            # int32[n_signals] — column in sim.ticker_index
    signal_score: np.ndarray,                 # float64[n_signals]
    signal_sector_idx: np.ndarray,            # int32[n_signals] — sector index for cap math
    signal_sector_rating: np.ndarray,         # int8[n_signals] — SR_* codes
    signal_conviction: np.ndarray,            # int8[n_signals] — CONV_* codes
    signal_upside: np.ndarray,                # float64[n_signals]; NaN if missing
    signal_atr_pct: np.ndarray,               # float64[n_signals]; NaN if missing
    signal_pred_confidence: np.ndarray,       # float64[n_signals]; NaN if missing
    signal_p_up: np.ndarray,                  # float64[n_signals]; NaN if missing
    signal_days_to_earnings: np.ndarray,      # int32[n_signals]; -1 if unknown
    signal_feature_coverage: np.ndarray,      # float64[n_signals]; NaN if unknown
    signal_gbm_veto: np.ndarray,              # bool[n_signals]
    signal_momentum_at_date: np.ndarray,      # float64[n_signals]; NaN means skip momentum gate (matches scalar's "no history" early-return)
    # Per-(combo) and global inputs.
    prices: np.ndarray,                       # float64[n_tickers] — current close
    nav_per_combo: np.ndarray,                # float64[n_combos]
    dd_multiplier_per_combo: np.ndarray,      # float64[n_combos]
    market_regime: int,                       # REGIME_* code
    signal_age_days: int,
    config: VectorizedEntryConfig,
    # Correlation block inputs (precomputed; pass None to disable).
    correlation_matrix: np.ndarray | None = None,   # float64[n_tickers, n_tickers]
    sector_idx_per_ticker: np.ndarray | None = None,  # int32[n_tickers]
) -> EntryDecisions:
    """Compute per-(combo, signal) entry decisions as a matrix.

    Returns ``EntryDecisions`` whose ``entry_passed`` matrix encodes
    which (combo, signal) pairs will execute.

    The first failing gate sets ``block_reason``; subsequent gates do
    not overwrite (matches scalar ``decide_entries`` cascade order).
    """
    n_combos = sim.n_combos
    n_tickers = sim.n_tickers
    n_signals = signal_ticker_idx.shape[0]

    # Validate shapes (defensive — caller bugs are easier to catch here
    # than at the matmul site).
    sig_arrays = {
        "signal_score": signal_score,
        "signal_sector_idx": signal_sector_idx,
        "signal_sector_rating": signal_sector_rating,
        "signal_conviction": signal_conviction,
        "signal_upside": signal_upside,
        "signal_atr_pct": signal_atr_pct,
        "signal_pred_confidence": signal_pred_confidence,
        "signal_p_up": signal_p_up,
        "signal_days_to_earnings": signal_days_to_earnings,
        "signal_feature_coverage": signal_feature_coverage,
        "signal_gbm_veto": signal_gbm_veto,
        "signal_momentum_at_date": signal_momentum_at_date,
    }
    for name, arr in sig_arrays.items():
        if arr.shape != (n_signals,):
            raise ValueError(
                f"{name} shape mismatch: expected ({n_signals},), got {arr.shape}"
            )
    if prices.shape != (n_tickers,):
        raise ValueError(f"prices shape mismatch: expected ({n_tickers},), got {prices.shape}")
    if nav_per_combo.shape != (n_combos,):
        raise ValueError(f"nav_per_combo shape mismatch")
    if dd_multiplier_per_combo.shape != (n_combos,):
        raise ValueError(f"dd_multiplier_per_combo shape mismatch")

    # If no signals, return empty matrices.
    if n_signals == 0:
        return EntryDecisions(
            entry_passed=np.zeros((n_combos, 0), dtype=bool),
            entry_shares=np.zeros((n_combos, 0), dtype=np.float64),
            entry_dollar=np.zeros((n_combos, 0), dtype=np.float64),
            block_reason=np.zeros((n_combos, 0), dtype=np.int8),
        )

    block_reason = np.zeros((n_combos, n_signals), dtype=np.int8)

    # ── 1. Already-held ─────────────────────────────────────────────
    already_held = sim.positions[:, signal_ticker_idx] > 0  # [n_combos, n_signals]
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & already_held, BLOCK_ALREADY_HELD, block_reason,
    ).astype(np.int8)

    # ── 2. No price ─────────────────────────────────────────────────
    sig_prices = prices[signal_ticker_idx]  # [n_signals]
    no_price = (np.isnan(sig_prices)) | (sig_prices <= 0)
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & no_price[None, :], BLOCK_NO_PRICE, block_reason,
    ).astype(np.int8)

    # ── 3. Score gate ───────────────────────────────────────────────
    score_failed = (
        signal_score[None, :] < config.min_score_to_enter[:, None]
    )  # [n_combos, n_signals]
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & score_failed, BLOCK_SCORE, block_reason,
    ).astype(np.int8)

    # ── 4. Momentum gate ────────────────────────────────────────────
    # Scalar: skip momentum gate if price_history is None or len < 21.
    # Vectorized: signal_momentum_at_date is NaN for those signals → skip.
    mom_known = ~np.isnan(signal_momentum_at_date)
    mom_failed = (
        config.momentum_gate_enabled[:, None]
        & mom_known[None, :]
        & (signal_momentum_at_date[None, :] < config.momentum_gate_threshold[:, None])
    )
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & mom_failed, BLOCK_MOMENTUM_GATE, block_reason,
    ).astype(np.int8)

    # ── 5. Drawdown halt ────────────────────────────────────────────
    dd_halt = (dd_multiplier_per_combo <= 0.0)[:, None]  # [n_combos, 1]
    dd_blocked = np.broadcast_to(dd_halt, (n_combos, n_signals))
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & dd_blocked, BLOCK_DRAWDOWN_HALT, block_reason,
    ).astype(np.int8)

    # ── 6. Bear regime + underweight sector block ───────────────────
    is_bear = (market_regime == REGIME_BEAR)
    if is_bear:
        underweight = (signal_sector_rating == SR_UNDERWEIGHT)
        bear_block = config.bear_block_underweight[:, None] & underweight[None, :]
        block_reason = np.where(
            (block_reason == BLOCK_NONE) & bear_block, BLOCK_BEAR_UNDERWEIGHT, block_reason,
        ).astype(np.int8)

    # ── 7. GBM veto (per-signal — same across combos) ───────────────
    gbm_blocked = signal_gbm_veto[None, :]
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & gbm_blocked, BLOCK_GBM_VETO, block_reason,
    ).astype(np.int8)

    # ── 8. Sizing (compute regardless; gates will mask the eligible set) ──
    base_weight = 1.0 / max(n_signals, 1)
    sector_adj = _compute_sector_adj(signal_sector_rating, config)
    conviction_adj = _compute_conviction_adj(signal_conviction, config)
    upside_adj = _compute_upside_adj(signal_upside, config)
    atr_adj = _compute_atr_adj(signal_atr_pct, config)
    confidence_adj = _compute_confidence_adj(
        signal_pred_confidence, signal_p_up, config,
    )
    staleness_adj = _compute_staleness_adj(signal_age_days, config)  # [n_combos, 1]
    earnings_adj = _compute_earnings_adj(signal_days_to_earnings, config)
    coverage_adj = _compute_coverage_adj(signal_feature_coverage, config)

    raw_weight = (
        base_weight
        * sector_adj * conviction_adj * upside_adj
        * dd_multiplier_per_combo[:, None]
        * atr_adj * confidence_adj
        * staleness_adj * earnings_adj * coverage_adj
    )

    # max_position_pct is regime-conditional.
    if is_bear:
        max_pct = config.bear_max_position_pct
    else:
        max_pct = config.max_position_pct
    position_weight = np.minimum(raw_weight, max_pct[:, None])

    # ATR cap: when atr_adj < 1.0, also bound by base*atr_adj*dd_mult
    atr_active = atr_adj < 1.0
    atr_only_weight = (
        base_weight * atr_adj * dd_multiplier_per_combo[:, None]
    )
    position_weight = np.where(
        atr_active,
        np.minimum(np.minimum(position_weight, atr_only_weight), max_pct[:, None]),
        position_weight,
    )

    # Negative dd_multiplier is invalid; clamp to 0 → dollar_size 0.
    position_weight = np.maximum(position_weight, 0.0)
    dollar_size = nav_per_combo[:, None] * position_weight

    # Floor by min_position_dollar — set shares to 0 if below threshold.
    too_small = dollar_size < config.min_position_dollar[:, None]
    safe_prices = np.where(sig_prices > 0, sig_prices, 1.0)
    shares = np.floor(dollar_size / safe_prices[None, :])
    shares = np.where(too_small | no_price[None, :], 0.0, shares)

    # ── 9. Position cap (already enforced via max_pct above) ────────
    # Kept as an explicit gate for parity/auditability when raw_weight
    # exceeds max_pct: scalar's check_order checks position_pct against
    # effective_max_pct AFTER sizing. Sizing capped at max_pct already
    # so this gate only fires from numerical edge cases (we keep it as
    # a safety net).
    position_pct_per = position_weight  # already bounded by max_pct
    position_cap_failed = position_pct_per > max_pct[:, None] + 1e-12
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & position_cap_failed,
        BLOCK_POSITION_CAP, block_reason,
    ).astype(np.int8)

    # ── 10. Sector cap ──────────────────────────────────────────────
    # sector_exposure[c, sec] = sum over t of (positions[c, t] × prices[t]) where sector_idx_per_ticker[t] == sec
    if sector_idx_per_ticker is not None:
        n_sectors = int(np.max(sector_idx_per_ticker)) + 1 if sector_idx_per_ticker.size else 0
        n_sectors = max(n_sectors, int(np.max(signal_sector_idx)) + 1 if signal_sector_idx.size else 0, 1)
        # Build sector one-hot [n_tickers, n_sectors].
        sector_onehot = np.zeros((n_tickers, n_sectors), dtype=np.float64)
        valid_sec = sector_idx_per_ticker >= 0
        rows = np.where(valid_sec)[0]
        cols = sector_idx_per_ticker[valid_sec]
        sector_onehot[rows, cols] = 1.0
        # Position dollar value per ticker: positions × prices (NaN-safe).
        prices_safe = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
        position_value = sim.positions * prices_safe[None, :]  # [n_combos, n_tickers]
        sector_exposure = position_value @ sector_onehot       # [n_combos, n_sectors]
        # Per-(combo, signal): cap_value = max_sector_pct[c] × nav[c]
        sector_cap_value = config.max_sector_pct[:, None] * nav_per_combo[:, None]  # [n_combos, 1]
        # exposure_for_signal_sector[c, s] = sector_exposure[c, signal_sector_idx[s]]
        exposure_for_signal = sector_exposure[:, signal_sector_idx]  # [n_combos, n_signals]
        proposed = exposure_for_signal + dollar_size
        sector_cap_failed = proposed > sector_cap_value
        block_reason = np.where(
            (block_reason == BLOCK_NONE) & sector_cap_failed,
            BLOCK_SECTOR_CAP, block_reason,
        ).astype(np.int8)

    # ── 11. Equity cap ──────────────────────────────────────────────
    prices_safe = np.nan_to_num(prices, nan=0.0, posinf=0.0, neginf=0.0)
    total_equity = (sim.positions * prices_safe[None, :]).sum(axis=1)  # [n_combos]
    equity_cap_value = config.max_equity_pct * nav_per_combo  # [n_combos]
    proposed_equity = total_equity[:, None] + dollar_size  # [n_combos, n_signals]
    equity_cap_failed = proposed_equity > equity_cap_value[:, None]
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & equity_cap_failed,
        BLOCK_EQUITY_CAP, block_reason,
    ).astype(np.int8)

    # ── 12. Correlation block ───────────────────────────────────────
    if (
        correlation_matrix is not None
        and sector_idx_per_ticker is not None
        and np.any(config.correlation_block_enabled)
    ):
        # corr_to_held[c, s, t] = correlation_matrix[signal_ticker_idx[s], t]
        # Mask: held[c, t] AND sector_idx_per_ticker[t] == signal_sector_idx[s]
        # Sum + count → mean per (c, s); block if mean > threshold[c]
        held_mask = sim.held_mask()  # [n_combos, n_tickers]
        # candidate_corr_vec[s, t] = corr_matrix[signal_ticker_idx[s], t]
        candidate_corr_vec = correlation_matrix[signal_ticker_idx, :]  # [n_signals, n_tickers]
        # same_sector[s, t] = sector_idx_per_ticker[t] == signal_sector_idx[s]
        # Build via broadcasting: [n_signals, n_tickers]
        same_sector = (
            sector_idx_per_ticker[None, :] == signal_sector_idx[:, None]
        )  # [n_signals, n_tickers]
        # Effective mask per (c, s, t) = held[c, t] & same_sector[s, t]
        # Use einsum-style: held_mask[c, None, t] & same_sector[None, s, t]
        eff_mask = held_mask[:, None, :] & same_sector[None, :, :]  # [n_combos, n_signals, n_tickers]
        # Exclude the candidate ticker itself from comparisons (held[c, candidate_t]
        # is True if combo c already holds candidate, and same_sector is True
        # there — but already-held already blocked entry; defensive).
        # Don't compute self-corr (it's 1.0 and would inflate the mean).
        # Per-(s, t) self-mask: t == signal_ticker_idx[s].
        n_t_arange = np.arange(n_tickers)
        not_self = n_t_arange[None, :] != signal_ticker_idx[:, None]  # [n_signals, n_tickers]
        eff_mask = eff_mask & not_self[None, :, :]
        # Replace NaN correlations with 0; mask handles inclusion.
        corr_safe = np.nan_to_num(candidate_corr_vec, nan=0.0)
        # Sum and count.
        sum_corr = (eff_mask * corr_safe[None, :, :]).sum(axis=2)  # [n_combos, n_signals]
        count = eff_mask.sum(axis=2)                                # [n_combos, n_signals]
        # Mean = sum / count where count > 0; else NaN (no comparable held).
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_corr = np.where(count > 0, sum_corr / np.maximum(count, 1), np.nan)
        corr_failed = (
            config.correlation_block_enabled[:, None]
            & ~np.isnan(mean_corr)
            & (mean_corr > config.correlation_block_threshold[:, None])
        )
        block_reason = np.where(
            (block_reason == BLOCK_NONE) & corr_failed, BLOCK_CORRELATION, block_reason,
        ).astype(np.int8)

    # ── 13. Shares-round-to-zero (final gate) ───────────────────────
    shares_zero = shares <= 0
    block_reason = np.where(
        (block_reason == BLOCK_NONE) & shares_zero, BLOCK_SHARES_ZERO, block_reason,
    ).astype(np.int8)

    entry_passed = block_reason == BLOCK_NONE
    # For final emitted shares, mask out blocked entries (caller doesn't
    # accidentally consume sized values for blocked rows).
    final_shares = np.where(entry_passed, shares, 0.0)
    final_dollar = np.where(entry_passed, dollar_size, 0.0)

    return EntryDecisions(
        entry_passed=entry_passed,
        entry_shares=final_shares,
        entry_dollar=final_dollar,
        block_reason=block_reason,
    )


def apply_vectorized_entries(
    sim,
    decisions: EntryDecisions,
    *,
    signal_ticker_idx: np.ndarray,
    prices: np.ndarray,
    date_idx: int,
) -> int:
    """Apply approved entries via ``sim.apply_buy``.

    Aggregates by (combo, ticker) to handle the (rare) case of two
    signals on the same ticker — although ``decide_entries`` shouldn't
    emit duplicates, this guards against sweep configs that might.

    Returns the count of (combo, ticker) cells where a BUY was applied.
    """
    passed = decisions.entry_passed
    if not np.any(passed):
        return 0
    combo_idx, signal_idx = np.nonzero(passed)
    ticker_idx = signal_ticker_idx[signal_idx]
    shares = decisions.entry_shares[combo_idx, signal_idx]
    px = prices[ticker_idx]

    # Detect duplicate (combo, ticker) pairs and sum their shares
    # before calling apply_buy. apply_buy uses fancy-indexed assignment
    # which is last-write-wins on dupes — would silently drop a BUY.
    flat = combo_idx.astype(np.int64) * sim.n_tickers + ticker_idx.astype(np.int64)
    unique_flat, inverse = np.unique(flat, return_inverse=True)
    if unique_flat.size < flat.size:
        # Aggregate shares (sum) and use the first price per (c, t) — for
        # ENTER orders all signals share the same per-ticker price.
        agg_shares = np.zeros(unique_flat.size, dtype=np.float64)
        np.add.at(agg_shares, inverse, shares)
        agg_combo = (unique_flat // sim.n_tickers).astype(np.int64)
        agg_ticker = (unique_flat % sim.n_tickers).astype(np.int64)
        agg_price = prices[agg_ticker]
        sim.apply_buy(agg_combo, agg_ticker, agg_shares, agg_price, date_idx=date_idx)
        return int(unique_flat.size)
    sim.apply_buy(
        combo_idx.astype(np.int64),
        ticker_idx.astype(np.int64),
        shares,
        px,
        date_idx=date_idx,
    )
    return int(combo_idx.size)


def compute_correlation_matrix(
    returns_window: np.ndarray,
) -> np.ndarray:
    """Pearson correlation matrix from a returns window.

    returns_window : float64[n_tickers, lookback]
        Daily returns per ticker over a fixed lookback. NaN-bearing
        rows yield NaN correlations (matches scalar ``check_correlation``
        behavior of skipping degenerate-variance pairs).

    Returns float64[n_tickers, n_tickers].
    """
    n_tickers, lookback = returns_window.shape
    if lookback < 2:
        return np.full((n_tickers, n_tickers), np.nan)
    # np.corrcoef handles full-NaN rows gracefully. For partial NaN, we
    # zero-fill to keep the matmul well-defined; the resulting correlation
    # is biased but matches the scalar's "use all available aligned
    # values" behavior closely enough that downstream parity holds within
    # the lookback chosen for the sweep (60 bars, all complete).
    safe = np.nan_to_num(returns_window, nan=0.0)
    # corrcoef returns [n_tickers, n_tickers]; suppress divide-by-zero
    # warnings for degenerate rows (zero variance after fill).
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(safe)
