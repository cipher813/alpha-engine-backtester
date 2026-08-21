"""alpha-engine-config-I7936 — the consumer contract for universe_returns.return_5d.

Two columns named ``return_5d`` live in the same database with OPPOSITE unit
conventions. Measured on live ``s3://alpha-engine-research/research.db``
(2026-08-21):

* ``universe_returns.return_5d`` -- DECIMAL FRACTION. 2,012,661 non-null rows,
  eval_date 2025-12-08..2026-08-10, median 0.0004, p99 0.3023, min -0.9944,
  max 6686.1759. Producer: ``nousergon-data/collectors/universe_returns.py``,
  ``round(close_end / close_start - 1.0, 4)``.
* ``score_performance.return_5d`` -- 2dp PERCENT POINTS. 533 non-null rows,
  score_date 2026-03-04..2026-07-02, median -0.02, range [-20.42, 22.70]. The
  same quantity in the long-format ``score_performance_outcomes`` store sits at
  [-0.2042, 0.2270]: exactly 100x.

``shadow_book.py`` consumes the decimal one. This file pins that it says so and
that it breaks loudly if it ever receives the other.

The 6686.1759 is NOT a units error and this file records why: ZWZZT is a Nasdaq
TEST SECURITY. It closed at 19.44 on 2026-03-30 and 129,998.70 five sessions
later, so 129998.70 / 19.44 - 1 = 6686.1759 -- arithmetically correct, and not
a tradeable outcome.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from analysis.shadow_book import (
    UNIVERSE_RETURNS_UNITS,
    _GUARD_LIFT_MATERIAL,
    _MAX_TRADEABLE_ABS_RETURN,
    _assert_declared_units,
    _drop_untradeable,
    compute_shadow_book_analysis,
)
from analysis.regime_stratified_sortino import ReturnUnits, ReturnUnitsError


# ── 1. The convention is declared, not assumed ──────────────────────────────


def test_module_declares_the_convention_it_reads():
    assert UNIVERSE_RETURNS_UNITS is ReturnUnits.FRACTION


def test_decimal_fractions_pass_the_tripwire():
    """The live distribution: a median absolute 5d move of a few thousandths."""
    s = pd.Series([0.004, -0.011, 0.032, -0.0007, 0.019, 0.0004,
                   -0.058, 0.12, -0.003, 0.0021, 0.30, -0.9944])
    _assert_declared_units(s, column="universe_returns.return_5d")


def test_percent_points_raise_rather_than_being_coerced():
    """The live defect, verbatim: score_performance's return_5d handed to a
    reader that declared fractions. Median |r| jumps from thousandths to units,
    and guard_lift would silently mean something 100x different."""
    s = pd.Series([0.4, -1.1, 3.2, -0.07, 1.9, 0.04,
                   -5.8, 12.0, -0.3, 0.21, 22.7, -20.42])
    with pytest.raises(ReturnUnitsError) as exc:
        _assert_declared_units(s, column="score_performance.return_5d")
    assert "I7936" in str(exc.value)


def test_the_tripwire_ignores_untradeable_outliers_when_judging_units():
    """ZWZZT's 6686.1759 must not be what decides the units question -- a max
    is not the discriminating statistic, the median is."""
    s = pd.Series([0.004, -0.011, 0.032, -0.0007, 0.019, 0.0004,
                   -0.058, 0.12, -0.003, 0.0021, 6686.1759])
    _assert_declared_units(s, column="universe_returns.return_5d")


# ── 2. The 6686.1759, explained and excluded ────────────────────────────────


def test_zwzzt_arithmetic_reproduces_the_anchor_observation():
    """Not a units error: a test security's synthetic prices."""
    assert round(129998.70 / 19.44 - 1.0, 4) == 6686.1759


def test_untradeable_rows_are_dropped_and_logged(caplog):
    ur = pd.DataFrame({
        "ticker": ["AAPL", "MSFT", "ZWZZT"],
        "return_5d": [0.012, -0.004, 6686.1759],
    })
    with caplog.at_level("WARNING"):
        out = _drop_untradeable(ur)
    assert list(out["ticker"]) == ["AAPL", "MSFT"]
    assert any("ZWZZT" in r.getMessage() for r in caplog.records)
    assert 6686.1759 > _MAX_TRADEABLE_ABS_RETURN


# ── 3. The guard_lift threshold is in the column's units ────────────────────


def test_guard_lift_threshold_is_reachable_in_decimal_units():
    """It was 0.5 -- 50 percentage points of mean 5-day forward return between
    traded and blocked entries. Both non-neutral branches were unreachable, so
    every published assessment read 'neutral'. The live p99 of the column is
    0.3023; a DIFFERENCE OF MEANS of 0.5 is not a thing this can observe."""
    assert _GUARD_LIFT_MATERIAL < 0.3023


# ── 4. End to end: a percent-units source breaks the run ────────────────────


def _dbs(tmp_path, return_5d_values):
    trades = tmp_path / "trades.db"
    conn = sqlite3.connect(trades)
    conn.execute(
        "CREATE TABLE executor_shadow_book (ticker TEXT, date TEXT, "
        "block_reason TEXT, research_score REAL, prediction_confidence REAL, "
        "predicted_direction TEXT, intended_position_pct REAL, "
        "intended_dollars REAL, current_price REAL, market_regime TEXT)"
    )
    conn.execute(
        "CREATE TABLE trades (ticker TEXT, date TEXT, action TEXT, "
        "fill_price REAL, realized_return_pct REAL, realized_alpha_pct REAL, "
        "trigger_type TEXT, days_held INTEGER)"
    )
    for i in range(6):
        conn.execute(
            "INSERT INTO executor_shadow_book VALUES (?,?,?,?,?,?,?,?,?,?)",
            (f"T{i}", "2026-06-01", "risk_cap", 70.0, 0.6, "up", 1.0, 100.0, 10.0, "bull"),
        )
        conn.execute(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?)",
            (f"U{i}", "2026-06-01", "ENTER", 10.0, 1.0, 1.0, "signal", 5),
        )
    conn.commit()
    conn.close()

    research = tmp_path / "research.db"
    rconn = sqlite3.connect(research)
    rconn.execute(
        "CREATE TABLE universe_returns (ticker TEXT, eval_date TEXT, "
        "return_5d REAL, return_10d REAL, spy_return_5d REAL, beat_spy_5d INTEGER)"
    )
    for i, v in enumerate(return_5d_values):
        rconn.execute(
            "INSERT INTO universe_returns VALUES (?,?,?,?,?,?)",
            (f"T{i % 6}" if i % 2 == 0 else f"U{i % 6}", "2026-06-01",
             v, v, 0.002, int(v > 0.002)),
        )
    rconn.commit()
    rconn.close()
    return str(trades), str(research)


def test_end_to_end_decimal_source_is_accepted(tmp_path):
    vals = [0.01, -0.02, 0.03, -0.004, 0.012, -0.031,
            0.008, 0.021, -0.017, 0.004, -0.009, 0.015]
    t, r = _dbs(tmp_path, vals)
    result = compute_shadow_book_analysis(t, r)
    assert result["status"] == "ok"


def test_end_to_end_percent_source_raises_not_degrades(tmp_path):
    """The fail-soft handler catches ValueError, and ReturnUnitsError IS a
    ValueError -- without the explicit re-raise this would have degraded to
    'insufficient_return_data' and published a plausible-looking result."""
    vals = [1.0, -2.0, 3.0, -0.4, 1.2, -3.1,
            0.8, 2.1, -1.7, 0.4, -0.9, 1.5]
    t, r = _dbs(tmp_path, vals)
    with pytest.raises(ReturnUnitsError):
        compute_shadow_book_analysis(t, r)
