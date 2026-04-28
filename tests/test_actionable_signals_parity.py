"""Parity test: backtester's `_build_actionable_signals_local` must
produce byte-equivalent output to alpha-engine's
`executor.signal_reader.get_actionable_signals`.

Background: 2026-04-28 Tier 4 Layer 3 v14 caught a bug where the
vectorized sweep produced 0 orders because it bypassed the actionable
transformation. The fix vendored a local copy of `get_actionable_
signals` into the backtester (so unit tests don't require the
alpha-engine repo to be importable in CI), but two copies create
drift risk. This test catches drift when the executor repo IS on
sys.path (developer laptop, spot instance) — skipping cleanly when
it isn't (GitHub Actions CI).

If this test fails, reconcile the two implementations BEFORE merging
or the vectorized sweep will diverge from scalar.
"""
from __future__ import annotations

import pytest

from backtest import _build_actionable_signals_local


def _try_import_executor():
    """Return executor's get_actionable_signals if importable, else None."""
    try:
        from executor.signal_reader import get_actionable_signals
        return get_actionable_signals
    except ImportError:
        return None


_executor_fn = _try_import_executor()


_FIXTURES = [
    pytest.param(
        {
            "date": "2026-04-25",
            "market_regime": "neutral",
            "sector_ratings": {"Technology": {"rating": "market_weight"}},
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER", "score": 85,
                 "sector": "Technology"},
                {"ticker": "MSFT", "signal": "ENTER", "score": 78,
                 "sector": "Technology"},
            ],
            "universe": [
                {"ticker": "JPM", "signal": "HOLD", "score": 50,
                 "sector": "Financial"},
                {"ticker": "BAC", "signal": "EXIT", "score": 25,
                 "sector": "Financial"},
                {"ticker": "WFC", "signal": "REDUCE", "score": 30,
                 "sector": "Financial"},
            ],
        },
        id="canonical_synthetic_envelope",
    ),
    pytest.param(
        # Empty envelope.
        {"date": "2026-04-25", "buy_candidates": [], "universe": []},
        id="empty",
    ),
    pytest.param(
        # Ticker present in BOTH buy_candidates and universe — candidates
        # take precedence per `get_actionable_signals` semantics.
        {
            "buy_candidates": [
                {"ticker": "AAPL", "signal": "ENTER", "score": 85},
            ],
            "universe": [
                {"ticker": "AAPL", "signal": "HOLD", "score": 50},  # dropped (dedup)
                {"ticker": "MSFT", "signal": "HOLD", "score": 60},
            ],
        },
        id="dedup_candidates_precedence",
    ),
    pytest.param(
        # Missing optional envelope fields — must default cleanly.
        {"buy_candidates": [{"ticker": "AAPL", "signal": "ENTER"}]},
        id="missing_optional_fields",
    ),
    pytest.param(
        # All four signal types represented.
        {
            "market_regime": "bear",
            "sector_ratings": {"Energy": {"rating": "underweight"}},
            "buy_candidates": [
                {"ticker": "T1", "signal": "ENTER"},
            ],
            "universe": [
                {"ticker": "T2", "signal": "EXIT"},
                {"ticker": "T3", "signal": "REDUCE"},
                {"ticker": "T4", "signal": "HOLD"},
                {"ticker": "T5", "signal": "HOLD"},
            ],
        },
        id="all_signal_types",
    ),
]


@pytest.mark.skipif(
    _executor_fn is None,
    reason=(
        "alpha-engine repo not on sys.path — parity test requires the "
        "executor module to be importable. Skip-on-CI is intentional; "
        "this test runs on developer laptops + spot instances where "
        "the executor repo IS available."
    ),
)
@pytest.mark.parametrize("envelope", _FIXTURES)
def test_local_matches_executor(envelope: dict):
    """Backtester's vendored copy must equal alpha-engine's canonical."""
    local_out = _build_actionable_signals_local(envelope)
    executor_out = _executor_fn(envelope)

    # Exact key set match.
    assert set(local_out) == set(executor_out), (
        f"Key drift: local has {set(local_out) - set(executor_out)} extra, "
        f"executor has {set(executor_out) - set(local_out)} extra"
    )

    # List values: same tickers, same order. Per-entry dicts are the
    # same object references in both copies (both implementations
    # build new lists pointing at the same input dicts), so equality
    # comparison is meaningful.
    for key in ("enter", "exit", "reduce", "hold"):
        local_tickers = [s["ticker"] for s in local_out[key]]
        executor_tickers = [s["ticker"] for s in executor_out[key]]
        assert local_tickers == executor_tickers, (
            f"Drift on {key!r}: local={local_tickers}, "
            f"executor={executor_tickers}"
        )

    # Scalar fields propagate identically.
    assert local_out["market_regime"] == executor_out["market_regime"]
    assert local_out["sector_ratings"] == executor_out["sector_ratings"]


def test_local_implementation_segments_canonical_envelope():
    """Smoke: the local implementation must work standalone (no executor
    dependency). Pinned for CI where the executor repo isn't checked out.
    """
    envelope = {
        "buy_candidates": [
            {"ticker": "AAPL", "signal": "ENTER", "score": 85},
        ],
        "universe": [
            {"ticker": "JPM", "signal": "HOLD", "score": 50},
            {"ticker": "BAC", "signal": "EXIT", "score": 25},
        ],
    }
    out = _build_actionable_signals_local(envelope)
    assert {s["ticker"] for s in out["enter"]} == {"AAPL"}
    assert {s["ticker"] for s in out["exit"]} == {"BAC"}
    assert {s["ticker"] for s in out["hold"]} == {"JPM"}
    assert out["reduce"] == []


def test_local_skips_non_dict_entries():
    """Defensive: non-dict garbage in input lists must not crash.
    Matches the existing `_build_signal_lookup` test contract; the
    executor's version does NOT have this guard, but our vendored copy
    does (input may come from `signals_raw_filtered` with no
    universe-filter sanitization)."""
    envelope = {
        "buy_candidates": [
            {"ticker": "AAPL", "signal": "ENTER"},
            "not-a-dict",
            None,
        ],
        "universe": [
            42,
            {"ticker": "MSFT", "signal": "HOLD"},
        ],
    }
    out = _build_actionable_signals_local(envelope)
    assert {s["ticker"] for s in out["enter"]} == {"AAPL"}
    assert {s["ticker"] for s in out["hold"]} == {"MSFT"}
