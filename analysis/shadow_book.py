"""
shadow_book.py — Risk guard shadow book analysis.

Compares forward returns of blocked entries vs. traded entries to evaluate
whether the risk guard is too conservative, appropriately calibrated, or
too loose.

Data sources:
  - executor_shadow_book in trades.db (blocked entries)
  - trades in trades.db (executed entries)
  - universe_returns in research.db (forward returns for blocked stocks)

UNITS — read this before touching any threshold below
-----------------------------------------------------
``universe_returns.return_5d`` is a **DECIMAL FRACTION** (0.05 = +5%).
Measured against live ``s3://alpha-engine-research/research.db``
(2026-08-21, 2,012,661 non-null rows, eval_date 2025-12-08..2026-08-10):
median 0.0004, p99 0.3023, min -0.9944. The producer is
``nousergon-data/collectors/universe_returns.py``, which writes
``round(close_end / close_start - 1.0, 4)``.

A column named ``return_5d`` ALSO exists in ``score_performance`` in the same
database, in **2dp PERCENT POINTS** — measured range [-20.42, 22.70], median
-0.02, against the same quantity in the long-format
``score_performance_outcomes`` store at [-0.2042, 0.2270]: exactly 100x.
Two columns, one name, opposite conventions
(``alpha-engine-config-I7936``; sibling of ``alpha-engine-config-I7661``,
where percent points fed into ``log(1 + r)`` and every published Sortino
measured nothing).

So this module DECLARES which convention it reads and RAISES when the values
contradict the declaration. It does not coerce: a reader that cannot tell
which convention it received has no safe default, and a wrong guess produces
plausible numbers rather than an error.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.regime_stratified_sortino import ReturnUnits, ReturnUnitsError

logger = logging.getLogger(__name__)


# The convention this module reads. Not a guess — see the module docstring for
# the live measurement behind it.
UNIVERSE_RETURNS_UNITS: ReturnUnits = ReturnUnits.FRACTION

# Units tripwire on the declared column. A percent-point column mislabelled as
# a fraction has a MEDIAN absolute value of a few units; a genuine per-pick 5d
# decimal return has a median of a few thousandths. The median is the
# discriminating test — the max alone is not, because a real universe carries
# legitimate extremes (see _MAX_TRADEABLE_ABS_RETURN below). Mirrors the bound
# nousergon_lib.quant.stats.regime_sortino uses for the same decision
# (alpha-engine-config-I7661); lift to the lib on the third adoption per
# policy-shared-code -- tracked as alpha-engine-config-I7936 residue.
_MAX_PLAUSIBLE_MEDIAN_FRACTION: float = 0.5
_MEDIAN_CHECK_MIN_ROWS: int = 10

# Rows beyond this |return| are not tradeable 5-day equity outcomes. The live
# universe carries Nasdaq TEST SECURITIES (ZWZZT, ZVZZT, ZJZZT, ...) whose
# synthetic prices produce arithmetically-correct nonsense: ZWZZT closed at
# 19.44 on 2026-03-30 and 129,998.70 five sessions later, so
# return_5d = 129998.70 / 19.44 - 1 = 6686.1759 -- the live max of the column,
# and the anchor observation on alpha-engine-config-I7936. It is NOT a units
# error. Those rows contribute 70% of the universe-wide mean 5d return
# (0.004758 with them, 0.001418 without). Excluding them here keeps this
# module honest while the producer-side exclusion lands
# (nousergon-data, alpha-engine-config-I7936).
_MAX_TRADEABLE_ABS_RETURN: float = 5.0

# Material difference between the mean 5-day forward return of traded vs
# blocked entries, in the SAME decimal-fraction units as the column
# (0.01 = 100 bps). This was ``0.5`` -- 50 percentage points -- which is a
# threshold written for percent points and applied to a decimal column: the
# `appropriate` and `too_tight` branches were unreachable, so every run this
# module has ever published reported ``assessment: "neutral"`` regardless of
# what the guard did. Same root class as the column-name ambiguity above; the
# rename would not have caught it, only stating the units does
# (alpha-engine-config-I7936).
_GUARD_LIFT_MATERIAL: float = 0.01


def _assert_declared_units(values: pd.Series, *, column: str) -> None:
    """Raise if ``column`` contradicts :data:`UNIVERSE_RETURNS_UNITS`.

    Fails LOUD rather than coercing. The failure this guards is not
    hypothetical: the fleet stores this quantity both ways, and a source swap
    that silently changed which one reached this module would change what
    every ``guard_lift`` below means by two orders of magnitude without
    changing a single line of code (alpha-engine-config-I7936 / I7661).
    """
    finite = values.astype("float64")
    finite = finite[np.isfinite(finite.to_numpy())]
    if finite.size < _MEDIAN_CHECK_MIN_ROWS:
        return
    # The MEDIAN, over every finite row -- deliberately NOT pre-filtered by
    # _MAX_TRADEABLE_ABS_RETURN. A median is already robust to the handful of
    # test-security rows, and trimming first would discard exactly the large
    # values that make a percent-point column recognisable.
    median_abs = float(np.median(np.abs(finite.to_numpy())))
    if median_abs > _MAX_PLAUSIBLE_MEDIAN_FRACTION:
        raise ReturnUnitsError(
            f"{column}: declared {UNIVERSE_RETURNS_UNITS.value!r} but the MEDIAN "
            f"absolute value over {finite.size} rows is "
            f"{median_abs:.6g} -- a typical stock moving {median_abs:.0%} over "
            f"5 sessions is not a return distribution this module sees. The "
            f"source is most likely percent points, i.e. score_performance's "
            f"return_5d rather than universe_returns' "
            f"(alpha-engine-config-I7936)."
        )


def _drop_untradeable(ur: pd.DataFrame, *, column: str = "return_5d") -> pd.DataFrame:
    """Drop rows whose return is outside any tradeable 5-day equity outcome.

    See :data:`_MAX_TRADEABLE_ABS_RETURN`. Logged at WARNING with the offending
    tickers so the exclusion is never silent -- a filter nobody can see is how
    the universe came to carry test securities unnoticed in the first place.
    """
    if column not in ur.columns or ur.empty:
        return ur
    vals = ur[column].astype("float64").to_numpy()
    bad = np.isfinite(vals) & (np.abs(vals) > _MAX_TRADEABLE_ABS_RETURN)
    if not bad.any():
        return ur
    offenders = sorted(set(ur.loc[bad, "ticker"])) if "ticker" in ur.columns else []
    logger.warning(
        "shadow_book: dropped %d universe_returns row(s) with |%s| > %g "
        "(not tradeable 5-day outcomes; tickers=%s) -- "
        "alpha-engine-config-I7936",
        int(bad.sum()), column, _MAX_TRADEABLE_ABS_RETURN, offenders[:20],
    )
    return ur.loc[~bad].copy()


def compute_shadow_book_analysis(
    trades_db_path: str,
    research_db_path: str | None = None,
    min_blocks: int = 3,
) -> dict:
    """
    Compare blocked entries vs. traded entries.

    Uses universe_returns to get forward returns for blocked stocks (since they
    don't have realized PnL). Falls back to simple count/reason analysis if
    research.db isn't available.

    Returns dict with:
        status: "ok" | "insufficient_data" | "error"
        n_blocked: total blocked entries
        n_traded: total traded entries
        blocked_avg_return: avg 5d forward return of blocked stocks
        traded_avg_return: avg 5d forward return of traded stocks
        guard_lift: traded_avg - blocked_avg (positive = guard is helping)
        by_reason: breakdown by block_reason
        assessment: "too_tight" | "appropriate" | "too_loose"
    """
    if not Path(trades_db_path).exists():
        return {"status": "error", "error": f"trades.db not found at {trades_db_path}"}

    # Narrow scope of broad-except per backtester-audit-260415 Phase 1.2:
    # shadow_book is a required input to the replay parity test (Phase 1.1),
    # so silent-fail on schema corruption or permission errors would mask
    # parity-test regressions. Only "table missing" is legitimately recoverable
    # (fresh trades.db on first boot, pre-shadow-book-schema); everything else
    # propagates to surface as a loud pipeline failure.
    conn = sqlite3.connect(trades_db_path)
    try:
        try:
            shadow = pd.read_sql_query(
                "SELECT ticker, date, block_reason, research_score, "
                "prediction_confidence, predicted_direction, "
                "intended_position_pct, intended_dollars, "
                "current_price, market_regime "
                "FROM executor_shadow_book",
                conn,
            )
        except pd.errors.DatabaseError as exc:
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return {
                    "status": "insufficient_data",
                    "error": f"shadow_book schema not present: {exc}",
                }
            raise  # schema corruption / disk error / unexpected condition

        try:
            trades = pd.read_sql_query(
                "SELECT ticker, date, fill_price, "
                "realized_return_pct, realized_alpha_pct, "
                "trigger_type, days_held "
                "FROM trades WHERE action = 'ENTER'",
                conn,
            )
        except pd.errors.DatabaseError as exc:
            msg = str(exc).lower()
            if "no such table" in msg or "no such column" in msg:
                return {
                    "status": "insufficient_data",
                    "error": f"trades schema not present: {exc}",
                }
            raise
    finally:
        conn.close()

    if shadow.empty:
        return {"status": "insufficient_data", "error": "no blocked entries in shadow book"}

    if len(shadow) < min_blocks:
        return {
            "status": "insufficient_data",
            "error": f"need >= {min_blocks} blocked entries, have {len(shadow)}",
        }

    result: dict = {
        "status": "ok",
        "n_blocked": len(shadow),
        "n_traded": len(trades),
    }

    # Join with universe_returns if available to get forward returns for blocked stocks
    blocked_returns = None
    traded_returns = None
    if research_db_path and Path(research_db_path).exists():
        try:
            rconn = sqlite3.connect(research_db_path)
            ur = pd.read_sql_query(
                "SELECT ticker, eval_date, return_5d, return_10d, "
                "spy_return_5d, beat_spy_5d "
                "FROM universe_returns WHERE return_5d IS NOT NULL",
                rconn,
            )
            rconn.close()

            # DECLARE, then verify. The units check runs before any mean is
            # taken, and its ReturnUnitsError is deliberately re-raised past
            # the fail-soft handler below (see the `except` clause) -- a
            # forward-return enrichment that silently degrades is acceptable,
            # one that silently changes what the numbers MEAN is not.
            _assert_declared_units(ur["return_5d"], column="universe_returns.return_5d")
            ur = _drop_untradeable(ur, column="return_5d")

            if not ur.empty:
                # Blocked stock returns
                blocked_merged = shadow.merge(
                    ur,
                    left_on=["ticker", "date"],
                    right_on=["ticker", "eval_date"],
                    how="inner",
                )
                if not blocked_merged.empty:
                    blocked_returns = blocked_merged

                # Traded stock returns (from universe, not realized PnL)
                traded_merged = trades.merge(
                    ur,
                    left_on=["ticker", "date"],
                    right_on=["ticker", "eval_date"],
                    how="inner",
                )
                if not traded_merged.empty:
                    traded_returns = traded_merged
        except ReturnUnitsError:
            # NOT fail-soft. ReturnUnitsError subclasses ValueError, so without
            # this clause the broad handler below would swallow the one error
            # this module raises on purpose -- the exact silent-swallow the
            # fail-loud rule forbids on a units boundary
            # (alpha-engine-config-I7936).
            raise
        except (sqlite3.Error, pd.errors.DatabaseError, pd.errors.MergeError, KeyError, ValueError) as e:
            # Fail-soft: forward-return enrichment is optional. Narrowed to the
            # real failure surface here — sqlite/read_sql errors (missing
            # universe_returns table or unreadable research.db) and the
            # merge/key/value errors a schema mismatch in the joined columns
            # would raise. The analysis proceeds without blocked/traded
            # forward returns and reports "insufficient_return_data".
            logger.debug("Could not join universe_returns: %s", e)

    if blocked_returns is not None and not blocked_returns.empty:
        blocked_avg = round(float(blocked_returns["return_5d"].mean()), 4)
        blocked_beat_spy = round(float(blocked_returns["beat_spy_5d"].mean()), 4) if "beat_spy_5d" in blocked_returns else None
        result["blocked_avg_return_5d"] = blocked_avg
        result["blocked_beat_spy_pct"] = blocked_beat_spy
        result["blocked_with_returns"] = len(blocked_returns)
    else:
        blocked_avg = None

    if traded_returns is not None and not traded_returns.empty:
        traded_avg = round(float(traded_returns["return_5d"].mean()), 4)
        traded_beat_spy = round(float(traded_returns["beat_spy_5d"].mean()), 4) if "beat_spy_5d" in traded_returns else None
        result["traded_avg_return_5d"] = traded_avg
        result["traded_beat_spy_pct"] = traded_beat_spy
        result["traded_with_returns"] = len(traded_returns)
    elif not trades.empty and trades["realized_alpha_pct"].notna().any():
        traded_avg = round(float(trades["realized_alpha_pct"].dropna().mean()), 4)
        result["traded_avg_alpha"] = traded_avg
    else:
        traded_avg = None

    if blocked_avg is not None and traded_avg is not None:
        guard_lift = round(traded_avg - blocked_avg, 4)
        result["guard_lift"] = guard_lift

        if guard_lift > _GUARD_LIFT_MATERIAL:
            result["assessment"] = "appropriate"
        elif guard_lift < -_GUARD_LIFT_MATERIAL:
            result["assessment"] = "too_tight"
        else:
            result["assessment"] = "neutral"
    else:
        result["assessment"] = "insufficient_return_data"

    # Classification: selected=blocked, positive=would have lost (didn't beat SPY)
    # TP = blocked AND didn't beat SPY (correct block)
    # FP = blocked AND beat SPY (incorrectly blocked a winner)
    # FN = traded AND didn't beat SPY (should have been blocked)
    # TN = traded AND beat SPY (correctly allowed)
    if (blocked_returns is not None and not blocked_returns.empty
            and traded_returns is not None and not traded_returns.empty
            and "beat_spy_5d" in blocked_returns.columns
            and "beat_spy_5d" in traded_returns.columns):
        from analysis.classification_metrics import compute_binary_metrics
        b = blocked_returns[blocked_returns["beat_spy_5d"].notna()]
        t = traded_returns[traded_returns["beat_spy_5d"].notna()]
        if not b.empty and not t.empty:
            tp = int((b["beat_spy_5d"] == 0).sum())
            fp = int((b["beat_spy_5d"] == 1).sum())
            fn = int((t["beat_spy_5d"] == 0).sum())
            tn = int((t["beat_spy_5d"] == 1).sum())
            result["classification"] = compute_binary_metrics(tp, fp, fn, tn)

    # Breakdown by block reason
    by_reason = []
    for reason in sorted(shadow["block_reason"].unique()):
        grp = shadow[shadow["block_reason"] == reason]
        reason_data = {
            "block_reason": reason,
            "count": len(grp),
            "pct_of_blocks": round(len(grp) / len(shadow), 4),
            "avg_score": round(float(grp["research_score"].dropna().mean()), 1)
            if grp["research_score"].notna().any() else None,
        }
        if blocked_returns is not None:
            reason_merged = blocked_returns[blocked_returns["block_reason"] == reason]
            if not reason_merged.empty:
                reason_data["avg_return_5d"] = round(float(reason_merged["return_5d"].mean()), 4)
                reason_data["n_with_returns"] = len(reason_merged)
        by_reason.append(reason_data)

    result["by_reason"] = by_reason
    return result
