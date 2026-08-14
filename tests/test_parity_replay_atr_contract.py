"""The ATR-map wiring contract on the replay path, and the verdict fields
that make a broken parity report visible (alpha-engine-config-I7222).

WHAT BROKE
----------
`backtest/{date}/parity_report.json` is the backtester's only real grade. Its
last report carrying `data_state: ok` is **2026-04-26**. The nine weekly
reports that follow (2026-05-29 … 2026-07-17) are ~1.5 KB each, every metric
0.0, `n_backtester_orders_total: 0`, and each carries the note:

    Backtester replay raised RuntimeError: atr_map missing EOG at
    decide_entries — load_atr_14_pct contract violated.

That message reads like a data gap in EOG. It is not one: EOG is in the
ArcticDB universe with a present, positive, non-NaN `atr_14_pct` on every day
of the replay window. The real cause is one day older than the first bad
report — `crucible-executor#110` (2026-04-27) moved the backtester off the
`executor_run(simulate=True, ...)` shell, which loaded ATR from ArcticDB
itself, onto direct `decide_entries` calls taking an INJECTED `atr_map`.
`_run_simulation_loop` and `run_param_sweep` were wired to load and pass the
feature maps. **Both replay entry points were not.** `atr_by_ticker=None`
collapses to `{}` at `_simulate_single_date`'s default, so every replayed date
died on whichever ENTER candidate reached the ATR line first — EOG being
merely the alphabetically-unlucky first name in the 2026-05-13 cohort.

WHY THE TESTS BELOW HAVE THIS SHAPE
-----------------------------------
The defect lived in a KWARG THAT WAS NOT PASSED. No unit test of
`_simulate_single_date` could see it, because every such test passes the maps
explicitly — the fixtures were correct and production was not. So the guard
has to be structural: assert at the call sites that both replay paths hand the
maps down. That is the only assertion that would have failed on 2026-04-27.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]

# Ensure executor on sys.path before importing backtester functions
# (mirrors tests/test_simulate_via_deciders.py).
_EXECUTOR_ROOT = os.path.expanduser("~/Development/alpha-engine")
if os.path.isdir(_EXECUTOR_ROOT) and _EXECUTOR_ROOT not in sys.path:
    sys.path.insert(0, _EXECUTOR_ROOT)


# ── Structural guard: the maps reach _simulate_single_date ──────────────────

#: Every function that calls ``_simulate_single_date``. A new one added
#: without the feature maps reproduces the 109-day outage, so the list is
#: derived from the source rather than hardcoded.
_REQUIRED_KWARGS = ("atr_by_ticker", "vwap_series_by_ticker", "coverage_by_ticker")


def _simulate_call_sites() -> list[tuple[str, set[str]]]:
    """Return (enclosing function name, kwarg names) for every
    ``_simulate_single_date(...)`` call in backtest.py."""
    tree = ast.parse((_REPO / "backtest.py").read_text())
    out: list[tuple[str, set[str]]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_simulate_single_date"
            ):
                out.append((fn.name, {kw.arg for kw in node.keywords if kw.arg}))
    return out


def test_every_simulate_call_site_passes_the_feature_maps():
    """THE regression test for config-I7222.

    A call site that omits ``atr_by_ticker`` hands ``decide_entries`` an empty
    map and every ENTER on that date aborts. This assertion is what fails on
    the 2026-04-27 tree and passes on the fix.
    """
    sites = _simulate_call_sites()
    assert sites, "no _simulate_single_date call sites found — re-point this test"
    offenders = {
        caller: sorted(set(_REQUIRED_KWARGS) - kwargs)
        for caller, kwargs in sites
        if set(_REQUIRED_KWARGS) - kwargs
    }
    assert not offenders, (
        f"these _simulate_single_date call sites do not pass the feature maps: "
        f"{offenders}. Without atr_by_ticker the map collapses to {{}} and every "
        f"ENTER aborts with 'atr_map missing <TICKER>' — the defect that left "
        f"parity_report.json ungraded for 109 days (config-I7222)."
    )


def test_replay_for_dates_loads_the_feature_maps():
    """Passing the kwarg is only half of it — the maps have to be LOADED.
    Both replay paths are fed from this one bulk read."""
    import backtest as _bt

    src = inspect.getsource(_bt.replay_for_dates)
    assert "load_precomputed_feature_maps" in src, (
        "replay_for_dates no longer loads the feature maps — its callees will "
        "receive None and every ENTER will abort (config-I7222)"
    )


def test_an_empty_atr_map_from_the_feature_store_is_refused():
    """An empty bulk read is never a legitimate replay input — it produces a
    report with zero backtester orders that LOOKS like a comparison. Fail loud
    instead of shipping the shape that hid for nine weeks."""
    import backtest as _bt

    src = inspect.getsource(_bt.replay_for_dates)
    tree = ast.parse(src.lstrip())
    raises_on_empty = any(
        isinstance(node, ast.Raise) for node in ast.walk(tree)
    )
    assert raises_on_empty and "atr_by_ticker" in src


# ── Behavioural guards on the bounding rule ────────────────────────────────

def _df_history(n_bars: int = 100, base: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": [base + i * 0.1 for i in range(n_bars)],
            "high": [base + i * 0.1 + 0.5 for i in range(n_bars)],
            "low": [base + i * 0.1 - 0.5 for i in range(n_bars)],
            "close": [base + i * 0.1 + 0.2 for i in range(n_bars)],
        },
        index=pd.bdate_range("2024-01-01", periods=n_bars),
    )


def _signals(tickers: list[str], date_str: str = "2026-05-13") -> dict:
    enter = [
        {
            "ticker": t,
            "signal": "ENTER",
            "score": 80,
            "conviction": "rising",
            "sector": "Technology",
            "rating": "BUY",
            "price_target_upside": 0.15,
            "thesis_summary": "test",
        }
        for t in tickers
    ]
    return {
        "date": date_str,
        "market_regime": "neutral",
        "sector_ratings": {"Technology": {"rating": "market_weight"}},
        "enter": enter,
        "exit": [],
        "reduce": [],
        "hold": [],
        "universe": enter,
        "buy_candidates": enter,
    }


def _config() -> dict:
    return {
        "init_cash": 1_000_000.0,
        "signals_bucket": "alpha-engine-research",
        "min_score_to_enter": 70,
        "min_conviction_to_enter": ["rising", "stable"],
        "max_position_pct": 0.05,
        "bear_max_position_pct": 0.025,
        "max_sector_pct": 0.25,
        "max_equity_pct": 0.90,
        "drawdown_circuit_breaker": 0.08,
        "earnings_proximity_warning_days": 2,
        "momentum_gate_enabled": True,
        "momentum_gate_threshold": -50.0,
        "atr_sizing_enabled": True,
        "correlation_block_enabled": False,
        "coverage_sizing_enabled": False,
        "reduce_fraction": 0.50,
        "strategy": {
            "graduated_drawdown": {"enabled": False},
            "exit_manager": {
                "atr_trailing_enabled": False,
                "fallback_stop_enabled": False,
                "profit_take_enabled": False,
                "momentum_exit_enabled": False,
                "time_decay_enabled": False,
                "sector_relative_veto_enabled": False,
            },
        },
    }


_SECTOR_ETFS = ["SPY", "XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLU",
                "XLRE", "XLB", "XLI", "XLC"]


def _fixture(tickers: list[str], date_str: str = "2026-05-13"):
    ts = pd.Timestamp(date_str)
    cols = tickers + _SECTOR_ETFS
    price_matrix = pd.DataFrame({t: [100.0] for t in cols}, index=[ts])
    ohlcv = {t: _df_history(base=100 + i) for i, t in enumerate(cols)}
    return price_matrix, ohlcv, cols


@pytest.mark.skipif(
    not os.path.isdir(_EXECUTOR_ROOT),
    reason="alpha-engine sibling repo not present at ~/Development/alpha-engine",
)
class TestAtrCoverageBounding:
    """The abort itself is CORRECT for the live executor — shipping an entry on
    a bogus zero ATR is worse than not trading — and is not weakened here. What
    changes is the blast radius inside a replay."""

    def test_the_real_failing_shape_names_the_wiring_not_the_ticker(self):
        """Reproduces production exactly: cohort tickers from the 2026-05-13
        signals payload, no atr_by_ticker passed.

        Before the fix this raised ``atr_map missing EOG at decide_entries``,
        which sent nine weeks of investigation at EOG's market data. It must
        now name the actual defect — the caller not passing the map."""
        from executor.ibkr import SimulatedIBKRClient
        from backtest import _build_merged_simulate_config, _simulate_single_date

        # The first five buy_candidates of the real 2026-05-13 signals.json,
        # in payload order — EOG is fifth, and was the first to reach the ATR
        # line once the earlier names were filtered by other gates.
        cohort = ["COST", "HD", "WING", "CAH", "EOG"]
        price_matrix, ohlcv, _ = _fixture(cohort)
        merged_config, strategy_config = _build_merged_simulate_config(_config())

        with pytest.raises(RuntimeError) as exc:
            _simulate_single_date(
                sim_client=SimulatedIBKRClient(prices={}, nav=1_000_000.0),
                signal_date="2026-05-13",
                price_matrix=price_matrix,
                ohlcv_by_ticker=ohlcv,
                bucket="test-bucket",
                merged_config=merged_config,
                strategy_config=strategy_config,
                signals_override=_signals(cohort),
                atr_by_ticker=None,          # ← the production defect
                vwap_series_by_ticker=None,
                coverage_by_ticker=None,
            )

        msg = str(exc.value)
        assert "atr_map is empty" in msg, msg
        assert "atr_by_ticker" in msg, msg
        # The old message blamed a ticker for a wiring bug. Never again.
        assert "atr_map missing EOG" not in msg, msg

    def test_one_uncovered_ticker_costs_that_name_not_the_run(self):
        """The issue's deliverable 2: a name whose ATR cannot be loaded is
        excluded from that cohort and COUNTED — the replay's product is a
        population comparison, and losing one name is a coverage fact while
        losing the run is an outage."""
        from executor.ibkr import SimulatedIBKRClient
        from backtest import _build_merged_simulate_config, _simulate_single_date

        cohort = ["COST", "HD", "EOG"]
        price_matrix, ohlcv, cols = _fixture(cohort)
        merged_config, strategy_config = _build_merged_simulate_config(_config())

        # EOG absent from the map — exactly what feature_maps does to a name
        # whose latest atr_14_pct is NaN / non-positive / column-absent.
        atr = {t: 0.02 for t in cols if t != "EOG"}
        coverage = {t: 1.0 for t in cols}
        excluded: dict[str, int] = {}

        orders, skip = _simulate_single_date(
            sim_client=SimulatedIBKRClient(prices={}, nav=1_000_000.0),
            signal_date="2026-05-13",
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv,
            bucket="test-bucket",
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=_signals(cohort),
            atr_by_ticker=atr,
            vwap_series_by_ticker=None,
            coverage_by_ticker=coverage,
            atr_excluded_counter=excluded,
        )

        # The run survived …
        assert skip is None
        assert isinstance(orders, list)
        # … the uncovered name is gone from the cohort …
        assert "EOG" not in {o["ticker"] for o in orders}
        # … and the exclusion is a COUNTED fact, not a silent shrink.
        assert excluded == {"EOG": 1}

    def test_a_fully_covered_cohort_excludes_nothing(self):
        """The bounding rule must not quietly drop names when coverage is
        complete — otherwise it becomes its own silent shrink."""
        from executor.ibkr import SimulatedIBKRClient
        from backtest import _build_merged_simulate_config, _simulate_single_date

        cohort = ["COST", "HD", "EOG"]
        price_matrix, ohlcv, cols = _fixture(cohort)
        merged_config, strategy_config = _build_merged_simulate_config(_config())
        excluded: dict[str, int] = {}

        orders, skip = _simulate_single_date(
            sim_client=SimulatedIBKRClient(prices={}, nav=1_000_000.0),
            signal_date="2026-05-13",
            price_matrix=price_matrix,
            ohlcv_by_ticker=ohlcv,
            bucket="test-bucket",
            merged_config=merged_config,
            strategy_config=strategy_config,
            signals_override=_signals(cohort),
            atr_by_ticker={t: 0.02 for t in cols},
            vwap_series_by_ticker=None,
            coverage_by_ticker={t: 1.0 for t in cols},
            atr_excluded_counter=excluded,
        )

        assert skip is None
        assert excluded == {}
        assert orders  # a covered cohort produces orders — the point of the fix


# ── The detector: a degraded report must be RED, not quiet ─────────────────

class TestParityReportVerdictFields:
    """Deliverable 3. Nine reports declared `backtester_replay_error` with all
    metrics 0.0 and nothing anywhere turned red, because the only consumer
    (`crucible-evaluator/grading/tiles/backtester.py`) mapped a non-ok
    `data_state` onto `input_present=False` — i.e. rendered a DECLARED FAILURE
    identically to an ABSENT FILE. Both halves are fixed: the producer stamps
    an explicit verdict (here), and the evaluator tile reads it
    (`crucible-evaluator` companion PR).
    """

    def _emit(self, monkeypatch, tmp_path, data_state: str):
        from tests.test_parity_replay import _emit_degraded_parity_result

        monkeypatch.setenv("PARITY_REPORT_DIR", str(tmp_path))
        monkeypatch.setenv("PARITY_SKIP_METRICS_WRITE", "1")
        _emit_degraded_parity_result(
            data_state=data_state,
            n_live_trades_total=120,
            n_excluded=0,
            bucket="alpha-engine-research",
            note=(
                "Backtester replay raised RuntimeError: atr_map missing EOG at "
                "decide_entries — load_atr_14_pct contract violated. "
                "Window=2026-05-13..2026-07-10 (10 cohort dates)."
            ),
        )
        return json.loads((tmp_path / "parity_report.json").read_text())

    def test_the_exact_payload_that_hid_for_nine_weeks_is_a_fail(
        self, monkeypatch, tmp_path,
    ):
        """Proof the detector fires: the byte-shape of the 2026-05-29 …
        2026-07-17 reports, asserted RED."""
        report = self._emit(monkeypatch, tmp_path, "backtester_replay_error")

        assert report["data_state"] == "backtester_replay_error"
        assert report["metrics"]["capture_rate"] == 0.0
        # The fields the surface reads — absent on every one of the nine.
        assert report["status"] == "failed"
        assert report["verdict"] == "FAIL"
        assert report["schema"] == "parity_report-0.0.0"
        assert "backtester_replay_error" in report["verdict_reason"]

    @pytest.mark.parametrize(
        "data_state",
        ["backtester_replay_error", "empty_trades_db", "insufficient_cohort_dates"],
    )
    def test_every_degraded_state_carries_the_fail_verdict(
        self, monkeypatch, tmp_path, data_state,
    ):
        """Not just the ATR one. Any state that is not a comparison is a FAIL —
        a per-state allowlist is how the next unhandled value goes quiet."""
        report = self._emit(monkeypatch, tmp_path, data_state)
        assert report["status"] == "failed"
        assert report["verdict"] == "FAIL"

    def test_the_verdict_schema_matches_the_launchers_absence_marker(self):
        """One predicate must cover both ways this artifact fails to be a
        grade. `spot_parity_replay.sh` writes the absence marker; the degraded
        emitter writes the present-but-empty one. If their field names drift,
        the consumer needs two rules and will grow only one."""
        import re

        body = (_REPO / "infrastructure/spot_parity_replay.sh").read_text()
        m = re.search(r"printf '(\{\"schema\":\"parity_report-0\.0\.0\".*?\})", body)
        assert m, "the absence-marker printf moved — re-point this test"
        marker = json.loads(m.group(1) % ("2026-08-15", 1, 1))

        producer_src = (_REPO / "tests/test_parity_replay.py").read_text()
        for field in ("schema", "status", "verdict", "verdict_reason"):
            assert f'"{field}"' in producer_src, (
                f"the degraded parity report does not carry {field!r}, which the "
                f"absence marker does — the consumer would need two rules"
            )
        assert marker["status"] == "failed" and marker["verdict"] == "FAIL"
