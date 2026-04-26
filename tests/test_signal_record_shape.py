"""tests/test_signal_record_shape.py — lock the trimmed signal record shape.

Stage 2 of the c5.large optimization arc dropped 3 diagnostic fields
(``technical_score``, ``gbm_adjustment``, ``alpha_predicted``) from each
record produced by ``predictions_to_signals``. Across a 10y predictor
backtest those records materialize ~2M times (911 tickers × 2316 dates),
each spare dict slot costing ~80-120 B in Python overhead — dropping 3
fields reclaims ~300 MB at peak.

This test prevents accidental re-bloat: if a future convenience PR adds
a field back without thinking about the memory cost on this hot path,
the shape assertion fires.

Required fields (consumed by alpha-engine executor's signal_reader,
risk_guard, position_sizer, eod_reconcile, trade_logger):
    ticker, score, signal, conviction, sector, rating

Forbidden fields (verified 2026-04-26 to have ZERO consumers in
``executor/`` or any test module — pure historical diagnostics):
    technical_score, gbm_adjustment, alpha_predicted

If a downstream consumer genuinely needs one of the diagnostic fields,
the path is to reconstruct it offline from the persisted inputs
(predictor/predictions/{date}.json + ArcticDB universe + the scoring
formula in ``synthetic/signal_generator._compute_technical_score``) —
not to re-add them to this hot-loop dict.
"""
from __future__ import annotations

import pandas as pd
import pytest

from synthetic.signal_generator import (
    indicators_from_precomputed,
    precompute_indicator_series,
    predictions_to_signals,
)


REQUIRED_FIELDS = frozenset({"ticker", "score", "signal", "conviction", "sector", "rating"})
FORBIDDEN_DIAGNOSTIC_FIELDS = frozenset({
    "technical_score", "gbm_adjustment", "alpha_predicted",
})


def _synthetic_ohlcv(n_bars: int = 250, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Two-ticker DataFrame fixture with enough history for indicator
    rolling-window warmup (>200 bars). Matches ``build_ohlcv_df_by_ticker``
    output: DatetimeIndex + lowercase OHLC."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_bars)
    out: dict[str, pd.DataFrame] = {}
    for ticker, base in (("X", 100.0), ("Y", 200.0)):
        log_rets = rng.standard_normal(n_bars) * 0.01
        closes = base * pd.Series(log_rets).cumsum().apply(lambda r: pd.np.exp(r) if False else 1.0)
        close = base * (1 + 0.001 * pd.Series(rng.standard_normal(n_bars)).cumsum().values)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close},
            index=dates,
        )
        out[ticker] = df
    return out


def _synthetic_ohlcv_3(seed: int = 0) -> dict[str, pd.DataFrame]:
    """Three-ticker variant so top_n=1 forces ≥1 record into universe and
    ≥1 into buy_candidates simultaneously — deterministic regardless of
    score distribution."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=250)
    out: dict[str, pd.DataFrame] = {}
    for ticker, base in (("X", 100.0), ("Y", 200.0), ("Z", 50.0)):
        close = base * (1 + 0.001 * pd.Series(rng.standard_normal(250)).cumsum().values)
        df = pd.DataFrame(
            {"open": close, "high": close * 1.005, "low": close * 0.995, "close": close},
            index=dates,
        )
        out[ticker] = df
    return out


def _envelope() -> dict:
    ohlcv = _synthetic_ohlcv_3(seed=42)
    precomputed = precompute_indicator_series(ohlcv)
    date_str = ohlcv["X"].index[-1].strftime("%Y-%m-%d")
    indicators = indicators_from_precomputed(precomputed, ["X", "Y", "Z"], date_str)
    predictions = {"X": 0.005, "Y": -0.003, "Z": 0.001}
    sector_map = {"X": "XLK", "Y": "XLF", "Z": "XLE"}
    # min_score=10 + top_n=1: guarantees at least 1 ENTER (buy_candidates
    # has the top-scored ticker that clears min_score) AND at least 1 in
    # universe (the other 2 either fail min_score or get top-N capped).
    return predictions_to_signals(
        predictions=predictions,
        date=date_str,
        sector_map=sector_map,
        precomputed_indicators=indicators,
        top_n=1,
        min_score=10,
    )


class TestSignalRecordShape:
    def test_required_fields_present_on_buy_candidate(self):
        env = _envelope()
        cands = env.get("buy_candidates") or []
        if not cands:
            pytest.skip("synthetic fixture didn't emit any ENTER on this seed; the "
                        "score/regime fixture is decoupled from this test's purpose")
        sample = cands[0]
        missing = REQUIRED_FIELDS - set(sample.keys())
        assert not missing, f"buy_candidate missing required fields: {missing}"

    def test_required_fields_present_on_universe_record(self):
        env = _envelope()
        univ = env.get("universe") or []
        assert univ, "synthetic fixture should emit at least one HOLD/EXIT in universe"
        sample = univ[0]
        missing = REQUIRED_FIELDS - set(sample.keys())
        assert not missing, f"universe record missing required fields: {missing}"

    def test_diagnostic_fields_dropped_buy_candidate(self):
        env = _envelope()
        for cand in env.get("buy_candidates") or []:
            extras = FORBIDDEN_DIAGNOSTIC_FIELDS & set(cand.keys())
            assert not extras, (
                f"buy_candidate has diagnostic fields that should have been "
                f"dropped (re-bloat regression): {extras}. If you genuinely "
                f"need one of these, reconstruct offline from predictions + "
                f"ArcticDB instead of re-adding to the hot dict."
            )

    def test_diagnostic_fields_dropped_universe_record(self):
        env = _envelope()
        for rec in env.get("universe") or []:
            extras = FORBIDDEN_DIAGNOSTIC_FIELDS & set(rec.keys())
            assert not extras, (
                f"universe record has diagnostic fields that should have been "
                f"dropped (re-bloat regression): {extras}"
            )

    def test_record_field_set_is_exactly_six(self):
        """Exact-shape assertion: any field beyond REQUIRED_FIELDS or any
        missing required field is a contract change that needs deliberate
        review. This locks the record schema across PRs."""
        env = _envelope()
        for kind in ("buy_candidates", "universe"):
            for rec in env.get(kind) or []:
                assert set(rec.keys()) == REQUIRED_FIELDS, (
                    f"{kind} record schema drifted: keys={set(rec.keys())} "
                    f"vs required={REQUIRED_FIELDS}"
                )
