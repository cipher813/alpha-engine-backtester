"""inputs.py — the LIVE loader for the contribution-lift harness.

The two existing replay precedents in this repo — ``analysis/factor_blend_
counterfactual_replay.py`` and ``analysis/sector_constraint_replay.py`` — are
both default-OFF and take their cycles/price matrix as INJECTED config. Nothing
in the live weekly path ever injects them, so ``evaluate.py::_run_factor_blend_
counterfactual_replay`` is a guaranteed ``skipped`` on every Saturday run and
``sector_constraint_replay`` has zero live callers. A replay that never runs
measures nothing; the report card cannot be built on one.

This loader is therefore the deliberate departure: it assembles its OWN inputs
from the live archive, so ``contribution_lift`` runs on every weekly cycle
without an operator injecting anything.

Sources (all live as of 2026-08-17):

* ``s3://{bucket}/signals/{date}/signals.json``     — the selection actually made
* ``s3://{bucket}/predictor/predictions/{date}.json`` — the per-name alpha hats
* ``s3://{bucket}/factors/profiles/{date}/by_ticker.json`` — sub-factor pillars
* ArcticDB (via ``store.arctic_reader.load_universe_from_arctic``) — prices + SPY

Explicitly NOT an input: the retired six-team + CIO research-graph tables
(``team_candidates`` / ``cio_evaluations``, retired 2026-07-12 per
``analysis/end_to_end.RESEARCH_GRAPH_RETIRED_DATE``). A replay reading a
retired producer measures a system that no longer exists.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from analysis.contribution_lift.harness import HORIZON_DAYS, ReplayInputs

logger = logging.getLogger(__name__)

#: Default cycle window. 90 signal dates comfortably clears the 60-cycle
#: n_floor even after the horizon-tail cycles are dropped.
DEFAULT_LOOKBACK_CYCLES: int = 90

_SPY = "SPY"
_CASH = "CASH"


def _cost_model(config: dict) -> tuple[float, float, float]:
    """Read the SAME cost keys ``backtest._run_simulation_loop`` reads.

    If the objective is to be net-of-cost under the production cost model,
    it has to be the production cost model — not a second set of defaults
    that drifts away from it silently.
    """
    fees = float(config.get("simulation_fees", 0.001))
    slippage_bps = float((config.get("simulation") or {}).get("slippage_bps", 0))
    init_cash = float(config.get("init_cash", 1_000_000.0))
    return fees, slippage_bps, init_cash


def _empty_matrix() -> pd.DataFrame:
    return pd.DataFrame(index=pd.DatetimeIndex([], name="date"))


def _skipped(
    run_date: str, bucket: str, config: dict, reason: str, s3_client: Any
) -> ReplayInputs:
    fees, slippage_bps, init_cash = _cost_model(config)
    return ReplayInputs(
        run_date=run_date,
        dates=[],
        signals_by_date={},
        predictions_by_date={},
        pillar_profiles_by_date={},
        price_matrix=_empty_matrix(),
        spy_prices=pd.Series(dtype=float),
        bucket=bucket,
        fees=fees,
        slippage_bps=slippage_bps,
        init_cash=init_cash,
        s3_client=s3_client,
        status="skipped",
        reason=reason,
    )


def load_replay_inputs(
    config: dict,
    *,
    run_date: str | None = None,
    lookback_cycles: int | None = None,
    s3_client: Any = None,
) -> ReplayInputs:
    """Assemble the live replay inputs shared by every T5 spec.

    Window selection: the most recent ``lookback_cycles`` signal dates that
    still have at least ``HORIZON_DAYS`` price rows AFTER them, so every cycle
    the harness returns has a fully-defined objective. Cycles inside the
    horizon tail are excluded here rather than truncated later — a short
    window is a different measurement, not a noisier one.

    Fails LOUD when ArcticDB is unreachable or returns nothing for a non-empty
    cohort: an empty price matrix would make every arm's return 0.0 and every
    lift exactly 0.0, which is indistinguishable on the artifact from "this
    component contributes nothing". Degrades to ``status="skipped"`` ONLY when
    there are zero signal dates at all — there is genuinely nothing to replay,
    and the report is still emitted (always-emit contract).
    """
    import boto3

    from synthetic.predictor_backtest import _extract_close, build_price_matrix
    from synthetic.production_signal_backtest import _list_date_partitions, _load_json

    bucket = config.get("signals_bucket", "alpha-engine-research")
    s3 = s3_client or boto3.client("s3")
    lookback = int(
        lookback_cycles
        if lookback_cycles is not None
        else (config.get("contribution_lift") or {}).get(
            "lookback_cycles", DEFAULT_LOOKBACK_CYCLES
        )
    )
    run_date = run_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    signal_dates = _list_date_partitions(s3, bucket, "signals/", "/")
    if not signal_dates:
        return _skipped(
            run_date, bucket, config,
            f"no signals/{{date}}/signals.json partitions under s3://{bucket}/signals/",
            s3,
        )
    candidate_dates = signal_dates[-max(lookback * 2, lookback):]

    pred_dates = set(
        _list_date_partitions(s3, bucket, "predictor/predictions/", ".json")
    )

    signals_by_date: dict[str, dict] = {}
    predictions_by_date: dict[str, dict] = {}
    cohort: set[str] = set()
    for d in candidate_dates:
        sig = _load_json(s3, bucket, f"signals/{d}/signals.json")
        if not isinstance(sig, dict):
            continue
        signals_by_date[d] = sig
        signals = sig.get("signals") or {}
        rows = list(signals.values()) if isinstance(signals, dict) else list(signals)
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("ticker"), str):
                cohort.add(row["ticker"])
        if d in pred_dates:
            prd = _load_json(s3, bucket, f"predictor/predictions/{d}.json")
            if isinstance(prd, dict):
                by_ticker = {
                    p["ticker"]: p
                    for p in (prd.get("predictions") or [])
                    if isinstance(p, dict) and isinstance(p.get("ticker"), str)
                }
                predictions_by_date[d] = by_ticker
                cohort.update(by_ticker)
    cohort -= {_SPY, _CASH}

    if not signals_by_date:
        return _skipped(
            run_date, bucket, config,
            f"{len(candidate_dates)} signal partitions listed but none parsed "
            "into a usable signals.json body",
            s3,
        )

    from store.arctic_reader import (  # type: ignore[import-not-found]
        load_universe_from_arctic,
    )

    # Fail loud: no try/except. An ArcticDB outage must surface as an errored
    # module in the completeness tracker, never as a silently empty replay.
    price_data, _features = load_universe_from_arctic(
        bucket=bucket, tickers_allowlist=cohort | {_SPY},
    )
    if not price_data:
        raise RuntimeError(
            f"ArcticDB returned no price data for {len(cohort)} contribution-lift "
            "cohort tickers — every arm would simulate to a flat 0.0 return and "
            "every measured lift would be exactly 0.0, which is indistinguishable "
            "from 'this component contributes nothing'."
        )

    all_dates: set[pd.Timestamp] = set()
    for df in price_data.values():
        all_dates.update(pd.to_datetime(df.index))
    trading_dates = [d.strftime("%Y-%m-%d") for d in sorted(all_dates)]
    price_matrix = build_price_matrix(price_data, trading_dates)

    spy_prices = _extract_close(price_data, _SPY)
    if spy_prices is None or spy_prices.dropna().empty:
        raise RuntimeError(
            "SPY close series absent from ArcticDB price_data — the objective "
            "is benchmark-relative by definition and cannot be computed without it"
        )

    axis = pd.DatetimeIndex(price_matrix.index).sort_values()
    dates = _select_window(sorted(signals_by_date), axis, lookback)
    if not dates:
        return _skipped(
            run_date, bucket, config,
            f"{len(signals_by_date)} signal dates loaded but none has "
            f"{HORIZON_DAYS} price rows after it — every cycle's objective "
            "would be undefined",
            s3,
        )

    pillar_profiles_by_date = _load_pillar_profiles(bucket, dates)

    source_paths = [
        f"s3://{bucket}/signals/{{date}}/signals.json",
        f"s3://{bucket}/predictor/predictions/{{date}}.json",
        f"s3://{bucket}/factors/profiles/{{date}}/by_ticker.json",
        f"arcticdb://{bucket} (store.arctic_reader.load_universe_from_arctic)",
    ]

    fees, slippage_bps, init_cash = _cost_model(config)
    logger.info(
        "contribution_lift inputs: %d cycles (%s -> %s), cohort=%d, "
        "price_matrix=%dx%d, fees=%.4f slippage_bps=%.1f",
        len(dates), dates[0], dates[-1], len(cohort),
        len(price_matrix), len(price_matrix.columns), fees, slippage_bps,
    )
    return ReplayInputs(
        run_date=run_date,
        dates=dates,
        signals_by_date={d: signals_by_date[d] for d in dates},
        predictions_by_date={
            d: predictions_by_date[d] for d in dates if d in predictions_by_date
        },
        pillar_profiles_by_date=pillar_profiles_by_date,
        price_matrix=price_matrix,
        spy_prices=spy_prices,
        bucket=bucket,
        fees=fees,
        slippage_bps=slippage_bps,
        init_cash=init_cash,
        trades_db_path=config.get("_trades_db") or config.get("trades_db"),
        research_db_path=config.get("research_db"),
        s3_client=s3,
        status="ok",
        source_paths=source_paths,
    )


def _select_window(
    signal_dates: list[str], axis: pd.DatetimeIndex, lookback: int
) -> list[str]:
    """Most recent ``lookback`` signal dates with a FULL horizon of prices after."""
    usable: list[str] = []
    for d in signal_dates:
        pos = int(axis.searchsorted(pd.Timestamp(d), side="right"))
        if pos == 0 or pos + HORIZON_DAYS > len(axis):
            continue
        usable.append(d)
    return usable[-lookback:]


def _load_pillar_profiles(bucket: str, dates: list[str]) -> dict[str, dict]:
    """``{date: {ticker: {pillar: score}}}`` from ``factors/profiles/``.

    Delegates to ``analysis.end_to_end.load_historical_pillar_profiles`` (the
    single live reader of that artifact) and re-keys its ``(date, ticker)``
    output by date. That function is fail-soft by design; a date with no
    profile is simply absent, and a spec that needs pillars for a cycle it
    does not have emits ``N/A-MISSING-INPUT`` naming the artifact.
    """
    from analysis.end_to_end import load_historical_pillar_profiles

    flat = load_historical_pillar_profiles(bucket, dates)
    by_date: dict[str, dict] = {}
    for key, pillars in (flat or {}).items():
        try:
            date_str, ticker = key
        except (TypeError, ValueError):
            continue
        by_date.setdefault(str(date_str), {})[str(ticker)] = pillars
    return by_date
