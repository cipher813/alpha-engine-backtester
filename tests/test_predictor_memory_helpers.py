"""
tests/test_predictor_memory_helpers.py — helpers added for the 2026-04-23
OOM incident in predictor_data_prep.

Covers:
- _drain_price_data_into_ohlcv empties the source dict as it builds
- _drain equivalence with non-destructive build_ohlcv_by_ticker
- _drain skips macro/ETF tickers the same way
- _log_rss never raises (pure observability — must not fail callers)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pandas as pd
import pytest

from synthetic.predictor_backtest import (
    _drain_price_data_into_ohlcv,
    _log_rss,
    build_ohlcv_by_ticker,
)


def _sample_price_data() -> dict[str, pd.DataFrame]:
    idx = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"])
    return {
        "AAPL": pd.DataFrame(
            {"Open": [100.0, 101.0, 102.0], "High": [101.0, 102.0, 103.0],
             "Low": [99.0, 100.0, 101.0], "Close": [100.5, 101.5, 102.5]},
            index=idx,
        ),
        "MSFT": pd.DataFrame(
            {"Open": [200.0, 201.0], "High": [201.0, 202.0],
             "Low": [199.0, 200.0], "Close": [200.5, 201.5]},
            index=idx[:2],
        ),
        "SPY": pd.DataFrame(
            {"Open": [400.0], "High": [401.0], "Low": [399.0], "Close": [400.5]},
            index=idx[:1],
        ),  # macro — should be skipped
        "XLK": pd.DataFrame(
            {"Open": [150.0], "High": [151.0], "Low": [149.0], "Close": [150.5]},
            index=idx[:1],
        ),  # sector ETF — should be skipped
    }


def test_drain_empties_source_dict():
    """After drain, price_data must be empty — that's the whole point."""
    price_data = _sample_price_data()
    assert len(price_data) == 4

    _drain_price_data_into_ohlcv(price_data)

    assert len(price_data) == 0


def test_drain_produces_same_output_as_build():
    """Destructive drain must be a strict replacement for
    build_ohlcv_by_ticker — same ticker set, same bar contents."""
    drain_input = _sample_price_data()
    build_input = _sample_price_data()

    drained = _drain_price_data_into_ohlcv(drain_input)
    built = build_ohlcv_by_ticker(build_input)

    assert set(drained.keys()) == set(built.keys())
    for ticker in drained:
        assert drained[ticker] == built[ticker], f"mismatch on {ticker}"


def test_drain_skips_macro_and_sector_etfs():
    """SPY (macro) and XLK (sector ETF) must not appear in the output
    dict — downstream simulate loop filters to stock tickers only."""
    price_data = _sample_price_data()
    ohlcv = _drain_price_data_into_ohlcv(price_data)

    assert "SPY" not in ohlcv
    assert "XLK" not in ohlcv
    assert "AAPL" in ohlcv
    assert "MSFT" in ohlcv


def test_drain_output_bar_shape():
    """Each bar must have the 5 keys the simulate loop expects."""
    price_data = _sample_price_data()
    ohlcv = _drain_price_data_into_ohlcv(price_data)

    aapl_bars = ohlcv["AAPL"]
    assert len(aapl_bars) == 3
    for bar in aapl_bars:
        assert set(bar.keys()) == {"date", "open", "high", "low", "close"}
        assert isinstance(bar["date"], str)
        assert len(bar["date"]) == 10  # YYYY-MM-DD


def test_drain_preserves_date_order_per_ticker():
    """Bars must come out in original DataFrame index order."""
    price_data = _sample_price_data()
    ohlcv = _drain_price_data_into_ohlcv(price_data)

    msft_dates = [b["date"] for b in ohlcv["MSFT"]]
    assert msft_dates == ["2026-01-01", "2026-01-02"]


# ── _log_rss ────────────────────────────────────────────────────────────────


def test_log_rss_emits_info_line_with_mb(caplog):
    with caplog.at_level(logging.INFO, logger="synthetic.predictor_backtest"):
        _log_rss("test_checkpoint")

    msgs = [r.getMessage() for r in caplog.records
            if r.name == "synthetic.predictor_backtest"]
    assert any(
        "MEM test_checkpoint" in m and "RSS=" in m and "MB" in m
        for m in msgs
    ), f"no MEM line in {msgs!r}"


def test_log_rss_never_raises_on_platform_issues():
    """Pure observability — must survive any runtime error (no /proc,
    no resource module, permission denied, etc.) since we sprinkle
    these calls on hot paths."""
    # Force both code paths to fail and ensure no exception escapes
    with patch("builtins.open", side_effect=FileNotFoundError), \
         patch("resource.getrusage", side_effect=RuntimeError("boom")):
        # Should not raise
        _log_rss("breakage_test")
