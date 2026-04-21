"""Guard: simulate-mode signals filter must drop tickers absent from ArcticDB
universe before passing to the executor.

Rationale: the backtester replays historical signals.json files from S3.
Tickers that were in past universes but have since been dropped (e.g. TSM
and ASML, removed 2026-04-20 by the Research↔Executor universe-coverage
fix) leak into replayed signals. The executor's hard-fail guards
(load_daily_vwap, load_atr_14_pct) then raise NoSuchVersionException and
abort the whole simulation — which is exactly what broke the 2026-04-20
Saturday SF dry-run (backtester portfolio_stats failure).

Live executor must preserve EXIT/REDUCE/HOLD for held positions even when a
ticker goes missing from ArcticDB, so that filter is scoped narrowly to
buy_candidates (alpha-engine PR #77). Simulate mode has no real held
positions, so it can safely drop every signal referencing a missing ticker.
"""

from __future__ import annotations

from backtest import _filter_signals_to_universe


UNIVERSE = {"AAPL", "MSFT", "SPY", "NVDA"}


def _signals_with_missing_tickers():
    """Build a signals envelope that includes a mix of in-universe and
    out-of-universe tickers across every ticker-carrying field."""
    return {
        "date": "2026-03-15",
        "market_regime": "neutral",
        "universe": [
            {"ticker": "AAPL", "signal": "HOLD"},
            {"ticker": "TSM", "signal": "HOLD"},    # out
            {"ticker": "MSFT", "signal": "HOLD"},
        ],
        "buy_candidates": [
            {"ticker": "NVDA", "score": 70},
            {"ticker": "ASML", "score": 65},        # out
        ],
        "enter": [
            {"ticker": "NVDA", "signal": "ENTER"},
            {"ticker": "TSM", "signal": "ENTER"},   # out — the one that crashes load_daily_vwap
        ],
        "exit": [
            {"ticker": "ASML", "signal": "EXIT"},   # out
        ],
        "reduce": [],
        "hold": [
            {"ticker": "AAPL", "signal": "HOLD"},
        ],
    }


def test_filter_drops_out_of_universe_tickers_across_all_fields():
    signals = _signals_with_missing_tickers()
    rejected: dict[str, int] = {}
    filtered = _filter_signals_to_universe(signals, UNIVERSE, rejected)

    # Every surviving ticker must be in UNIVERSE.
    for field in ("universe", "buy_candidates", "enter", "exit", "reduce", "hold"):
        for entry in filtered.get(field, []):
            assert entry["ticker"] in UNIVERSE, (
                f"{field} leaked {entry['ticker']} which is not in universe"
            )

    # TSM appears in 2 lists (universe + enter) — both get dropped.
    assert rejected.get("TSM") == 2
    # ASML appears in 2 lists (buy_candidates + exit) — both get dropped.
    assert rejected.get("ASML") == 2
    # AAPL/MSFT/NVDA/SPY were all in-universe → zero rejects for them.
    for kept in ("AAPL", "MSFT", "NVDA", "SPY"):
        assert kept not in rejected


def test_filter_preserves_non_list_fields_unchanged():
    signals = _signals_with_missing_tickers()
    filtered = _filter_signals_to_universe(signals, UNIVERSE, None)
    assert filtered["date"] == signals["date"]
    assert filtered["market_regime"] == signals["market_regime"]


def test_filter_handles_missing_lists_gracefully():
    """Some signal envelopes omit optional lists (e.g. no buy_candidates on
    a day with no fresh nominations). The filter must not crash or create
    empty lists where none existed."""
    sparse = {
        "date": "2026-03-15",
        "market_regime": "bull",
        "enter": [{"ticker": "AAPL"}],
        # no universe/buy_candidates/exit/reduce/hold
    }
    filtered = _filter_signals_to_universe(sparse, UNIVERSE, None)
    # enter survives with AAPL only
    assert filtered["enter"] == [{"ticker": "AAPL"}]
    # missing fields stay missing — don't manufacture empty lists
    for absent in ("universe", "buy_candidates", "exit", "reduce", "hold"):
        assert absent not in filtered


def test_filter_handles_missing_ticker_field_without_crash():
    """Malformed entries (no ticker) are dropped, not counted as rejects."""
    weird = {"enter": [{"ticker": "AAPL"}, {"signal": "ENTER"}, {"ticker": ""}]}
    rejected: dict[str, int] = {}
    filtered = _filter_signals_to_universe(weird, UNIVERSE, rejected)
    assert filtered["enter"] == [{"ticker": "AAPL"}]
    # The malformed entries weren't counted as rejects (no ticker to name).
    assert rejected == {}


def test_filter_is_case_insensitive_on_ticker_match():
    signals = {"enter": [{"ticker": "aapl"}, {"ticker": "tsm"}]}
    rejected: dict[str, int] = {}
    filtered = _filter_signals_to_universe(signals, UNIVERSE, rejected)
    assert filtered["enter"] == [{"ticker": "aapl"}]
    assert rejected.get("TSM") == 1
