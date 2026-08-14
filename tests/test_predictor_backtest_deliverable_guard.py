"""predictor-backtest fails when its own deliverable is an error record.

`classify_simulation_outcome` guards portfolio_stats + sweep_df for the
simulate/param-sweep/all modes and exempts predictor-backtest, which produces
neither. That exemption left predictor-backtest with **no** deliverable guard,
and its own product is `predictor_stats.json`.

2026-08-13 (identically 2026-06-26): `_run_predictor_pipeline` caught
`ValueError: cannot convert float NaN to integer`, stored
`{"status": "error", ...}`, and the stage printed "Predictor-backtest
complete", exited 0, terminated its spot instance. Downstream, that read as
success:

- the SF marked `PredictorBacktest` Success, so `weekly_sf_rerun.py` emitted
  `skip_predictor_backtest` — every recovery rerun reused the error artifact;
- pit_parity's walk-forward reuse refused the `status != ok` artifact and fell
  back to the full predictor pipeline, which blew its 2700s ceiling. That
  week's contamination verdict was UNKNOWN as a result.

The guard's *scope* is the load-bearing half and gets as much coverage as its
polarity: `no_orders` and low-coverage are degeneracies of a run that
completed, and failing on those would crash the stage on a legitimate
no-admissible-entry week (config#7252).
"""

from __future__ import annotations

import pytest

from backtest import assert_predictor_backtest_deliverable as _guard


def test_error_status_fails_the_stage() -> None:
    with pytest.raises(RuntimeError) as exc:
        _guard(
            {
                "status": "error",
                "error": "cannot convert float NaN to integer",
                "error_class": "ValueError",
            },
            "2026-08-13",
        )
    assert "predictor-backtest produced no usable predictor_stats" in str(exc.value)


def test_failure_message_names_the_artifact_and_the_cause() -> None:
    """The operator must be able to act from the failure cause alone —
    obligation 3 of the 2026-08-13 resource-kill ruling, applied to this
    stage's own failure surface."""
    with pytest.raises(RuntimeError) as exc:
        _guard(
            {
                "status": "error",
                "error": "cannot convert float NaN to integer",
                "error_class": "ValueError",
            },
            "2026-08-13",
        )
    msg = str(exc.value)
    assert "backtest/2026-08-13/predictor_stats.json" in msg
    assert "ValueError" in msg
    assert "cannot convert float NaN to integer" in msg


def test_ok_status_passes() -> None:
    _guard({"status": "ok", "total_trades": 12}, "2026-08-13")


def test_absent_status_passes() -> None:
    """A stats dict with no `status` key is the legacy success shape."""
    _guard({"total_trades": 12}, "2026-08-13")


def test_none_stats_passes() -> None:
    """predictor_stats is None when the pipeline was skipped, not failed —
    the skip/resume path (L4527) must not be turned into a failure."""
    _guard(None, "2026-08-13")


@pytest.mark.parametrize("status", ["no_orders", "insufficient_coverage"])
def test_degenerate_but_completed_statuses_do_not_fail(status) -> None:
    """SCOPE guard. These mean the run completed and produced nothing
    admissible — the same shape as the EMPTY-sweep no-op, which is
    deliberately loud-but-not-fatal. Widening the check to `!= "ok"` would
    crash the stage on a legitimate no-entry week (config#7252)."""
    _guard({"status": status, "dates_simulated": 100}, "2026-08-13")


def test_degenerate_status_is_still_logged_loud(monkeypatch) -> None:
    """Not failing is not the same as being silent — the reuse path will
    refuse this artifact and re-run the full pipeline, which the operator
    needs to see."""
    import backtest

    seen: list[str] = []
    monkeypatch.setattr(
        backtest.logger, "warning", lambda msg, *a, **kw: seen.append(msg % a if a else msg)
    )
    _guard({"status": "no_orders"}, "2026-08-13")
    assert seen, "a degenerate predictor_stats status produced no warning"
    assert "no_orders" in seen[0]


def test_ok_status_is_not_warned_about(monkeypatch) -> None:
    import backtest

    seen: list[str] = []
    monkeypatch.setattr(
        backtest.logger, "warning", lambda msg, *a, **kw: seen.append(msg)
    )
    _guard({"status": "ok"}, "2026-08-13")
    assert not seen
