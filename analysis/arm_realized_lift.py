"""Realized 21-day lift for one entry-selection arm, at a fixed selection size.

alpha-engine-config-I8757 / -I8755.

## Why this module exists

The entry-selection slot's scoring lived inside
``analysis/end_to_end.py::_scanner_then_predictor_topN``, which count-matched
each cycle to the number of names the agentic CIO stage marked ADVANCE and
``continue``-d any cycle with none. The CIO stage was deprecated and its table
stopped at 2026-07-10, so every cycle after that was skipped and the champion's
weekly score froze — identical on three consecutive gate runs, carried with
confidence ``ok``.

Brian, 2026-08-27: *"the cio no longer exists, that was deprecated long ago."*
So this is a repair of a reference to a deleted component, not a redesign.

## The selection size

Every arm in this slot emits ``champion_top_n_default`` = 10 entries
(``crucible-executor/executor/champion.py``). Scoring every arm at that same
fixed N satisfies ``champion-challenger-policy.md`` §4's count-matching rule
directly, and does it without depending on any other component being alive —
which is precisely how the old basis failed. **The size the arms are compared
at is the size they actually trade at.**

A fixed N is also what makes a THIRD arm free: nothing in this module knows how
many arms exist.

## What an arm has to supply

One thing: ``{eval_date: [tickers, best first]}`` — its ranked picks per cohort
date, before truncation. This module truncates to N, joins realized returns,
sector-neutralises within the cycle, and reports. An arm's ranking rule is its
own business; how it is *scored* is not.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# The size every arm in the entry-selection slot trades at, and therefore the
# size they are compared at. Mirrors crucible-executor's
# `champion_top_n_default`; held as a literal for the same reason that repo
# holds S3 keys as literals — the value is a shared contract, not shared code.
DEFAULT_SELECTION_N = 10

# A cycle needs at least this many ranked names to be a selection rather than
# "everything it had". Below it, taking the top 10 of 10 measures the pool, not
# the rule, and would flatter a narrow arm on days it happened to be short.
MIN_RANKED_FOR_SELECTION = 2


@dataclass(frozen=True)
class ArmLift:
    """One arm's realized lift over the cohort dates it actually produced on."""

    arm: str
    n_cycles: int
    n_picks: int
    mean_alpha_21d: float | None
    sector_neutral_mean_alpha_21d: float | None
    selection_n: int
    n_cycles_skipped_thin: int
    first_date: str | None
    last_date: str | None

    def as_dict(self) -> dict:
        return {
            "arm": self.arm,
            "n_cycles": self.n_cycles,
            "n_picks": self.n_picks,
            "mean_alpha_21d": self.mean_alpha_21d,
            "sector_neutral_mean_alpha_21d": self.sector_neutral_mean_alpha_21d,
            "selection_n": self.selection_n,
            "n_cycles_skipped_thin": self.n_cycles_skipped_thin,
            "first_date": self.first_date,
            "last_date": self.last_date,
        }


def load_realized_alpha(conn) -> pd.DataFrame:
    """``ticker, eval_date, sector, alpha21`` — realized 21d log return minus
    SPY's over the same window, i.e. already benchmark-relative at the source.

    Reads ``universe_returns`` only. That table has a LIVE producer (max
    ``eval_date`` 2026-08-14 measured 2026-08-27), unlike the two tables the
    previous scorer joined.
    """
    return pd.read_sql_query(
        "SELECT ticker, eval_date, sector, "
        "(log_return_21d - log_spy_return_21d) AS alpha21 "
        "FROM universe_returns "
        "WHERE log_return_21d IS NOT NULL AND log_spy_return_21d IS NOT NULL",
        conn,
    )


def _sector_neutral(frame: pd.DataFrame) -> pd.Series:
    """Alpha minus its per-sector mean WITHIN the cycle.

    Falls back to raw alpha when the cycle carries no sector at all — a
    demeaned-against-nothing residual is not more neutral than the raw number,
    it is the same number with a false claim attached.
    """
    if frame["sector"].notna().any():
        return frame["alpha21"] - frame.groupby("sector")["alpha21"].transform("mean")
    return frame["alpha21"]


def score_arm(
    arm: str,
    ranked_picks_by_date: dict[str, list[str]],
    realized: pd.DataFrame,
    *,
    selection_n: int = DEFAULT_SELECTION_N,
) -> ArmLift:
    """Score one arm's ranked picks at a fixed selection size.

    ``ranked_picks_by_date`` maps a cohort date to that date's ranked tickers,
    best first, BEFORE truncation — this module truncates, so every arm is cut
    at the same place whatever its own N happened to be.

    A cycle contributes only when it has matured realized returns AND at least
    ``MIN_RANKED_FOR_SELECTION`` ranked names. Thin cycles are COUNTED
    (``n_cycles_skipped_thin``), never silently dropped: an arm that was thin
    for half the window and an arm that was whole are different arms, and a
    bare mean cannot tell them apart.
    """
    if realized.empty:
        return ArmLift(arm, 0, 0, None, None, selection_n, 0, None, None)

    by_date = {d: g for d, g in realized.groupby("eval_date")}
    picked: list[float] = []
    picked_sn: list[float] = []
    n_cycles = 0
    n_thin = 0
    used_dates: list[str] = []

    for eval_date in sorted(ranked_picks_by_date):
        ranked = [str(t).upper() for t in ranked_picks_by_date[eval_date]]
        if len(ranked) < MIN_RANKED_FOR_SELECTION:
            n_thin += 1
            continue
        cycle = by_date.get(eval_date)
        if cycle is None or cycle.empty:
            continue
        cycle = cycle[cycle["alpha21"].notna()].copy()
        if cycle.empty:
            continue
        cycle["alpha_sn"] = _sector_neutral(cycle)
        lookup = cycle.set_index(cycle["ticker"].str.upper())

        rows = [lookup.loc[t] for t in ranked[:selection_n] if t in lookup.index]
        if not rows:
            continue
        n_cycles += 1
        used_dates.append(eval_date)
        picked.extend(float(r["alpha21"]) for r in rows)
        picked_sn.extend(float(r["alpha_sn"]) for r in rows)

    if n_thin:
        logger.info(
            "[arm_realized_lift] %s: %d cycle(s) had fewer than %d ranked names "
            "and were not scored — the top-%d of a shorter list measures the "
            "pool, not the rule",
            arm, n_thin, MIN_RANKED_FOR_SELECTION, selection_n,
        )

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 5) if xs else None

    return ArmLift(
        arm=arm,
        n_cycles=n_cycles,
        n_picks=len(picked),
        mean_alpha_21d=_mean(picked),
        sector_neutral_mean_alpha_21d=_mean(picked_sn),
        selection_n=selection_n,
        n_cycles_skipped_thin=n_thin,
        first_date=used_dates[0] if used_dates else None,
        last_date=used_dates[-1] if used_dates else None,
    )


def research_free_ranked_picks(conn) -> dict[str, list[str]]:
    """``scanner_predictor_direct``'s ranked picks per cohort date.

    Its pool is the research-free parquet's own cohort — which contains ONLY
    scanner-passing names by construction, because its producer
    (``analysis/scanner_predictor_research_free_backfill.py``) builds its work
    list from ``quant_filter_pass`` rows. So the previous join against
    ``scanner_evaluations`` was redundant even before that table's writer was
    retired: it re-filtered a set that was already filtered, and in doing so
    made a live measurement depend on a dead table.

    Ranking is by ``predicted_alpha`` descending, matching
    ``crucible-executor/executor/champion.py::_apply_scanner_predictor_direct``.
    """
    df = pd.read_sql_query(
        "SELECT ticker, prediction_date, predicted_alpha "
        "FROM predictor_outcomes_research_free "
        "WHERE predicted_alpha IS NOT NULL",
        conn,
    )
    out: dict[str, list[str]] = {}
    for d, g in df.groupby("prediction_date"):
        out[str(d)] = (
            g.sort_values("predicted_alpha", ascending=False)["ticker"]
            .astype(str).str.upper().tolist()
        )
    return out


# `watchlist_source`, stamped on each row of `predictor/predictions/{date}.json`
# by crucible-predictor's `resolve_universe_from_membership`, IS the cut's own
# name (`sources = {t: cut_name for t in tickers}`), or `held` for a name
# unioned in only because the book holds it, or `both` for a name that is in
# the cut AND held.
#
# Provenance true by construction (champion-challenger-policy.md §7.5): the cut
# each name came from is recorded ON the row by the producer, so this needs no
# membership join and cannot drift when the cut pointer moves.
HELD_ONLY_SOURCE = "held"
IN_CUT_AND_HELD_SOURCE = "both"

#: The cut this arm is defined over. A date qualifies ONLY if its predictions
#: carry this cut's name — see `predictor_cut_ranked_picks` for why that is a
#: refusal and not a filter.
DEFAULT_ARM_CUT_NAMES: tuple[str, ...] = ("attractiveness_top_20",)


def predictor_cut_ranked_picks(
    predictions_by_date: dict[str, list[dict]],
    *,
    cut_names: tuple[str, ...] = DEFAULT_ARM_CUT_NAMES,
) -> dict[str, list[str]]:
    """``scanner_top20_predictor``'s ranked picks per cohort date.

    ``predictions_by_date`` maps a date to that date's
    ``predictor/predictions/{date}.json`` ``predictions`` list, verbatim.

    **A date qualifies only when its rows name one of ``cut_names``.** This is
    a refusal, not a filter, and it is the load-bearing rule here.
    ``predictor_universe_cut`` has moved: it was ``scanner_candidates`` (60
    names) until ~2026-08-07 and ``attractiveness_top_20`` after. Scoring the
    earlier dates would score "top-10 of whatever the predictor's cut happened
    to be that month" — a different rule per era wearing one arm's name, and a
    near-clone of the incumbent on exactly the dates where both drew from the
    scanner's 60. Measured 2026-08-27: without this refusal the arm scored 51
    cycles back to 2026-05-01, 47 of them from the pre-cut era.

    Within a qualifying date, a candidate is a row whose source is the cut's
    name or ``both``. ``held``-only names are excluded: the predictor unions
    holdings into its scoring universe so EXITS can be decided, and this arm
    proposes ENTRIES — scoring a held name as if the arm had selected it would
    credit the arm for a position it never chose.

    Ranking is by ``predicted_alpha`` descending, matching
    ``crucible-executor/executor/champion.py::_apply_scanner_top20_predictor``.
    """
    wanted = set(cut_names)
    out: dict[str, list[str]] = {}
    for eval_date, rows in predictions_by_date.items():
        rows = rows or []
        if not any(r.get("watchlist_source") in wanted for r in rows):
            # The cut this arm is defined over was not what the predictor
            # resolved from on this date. Not scorable — see the docstring.
            continue
        candidates: list[tuple[str, float]] = []
        for row in rows:
            ticker = row.get("ticker")
            source = row.get("watchlist_source")
            alpha = row.get("predicted_alpha")
            if not ticker or ticker == "SPY" or alpha is None:
                continue
            if source not in wanted and source != IN_CUT_AND_HELD_SOURCE:
                continue
            try:
                candidates.append((str(ticker).upper(), float(alpha)))
            except (TypeError, ValueError):
                continue
        if not candidates:
            continue
        candidates.sort(key=lambda row: row[1], reverse=True)
        out[str(eval_date)] = [t for t, _ in candidates]
    return out


# ── reading the predictor's dated output (alpha-engine-config-I8756) ─────

PREDICTIONS_PREFIX = "predictor/predictions/"

#: How far back to read dated predictions when scoring the cut arm. The arm
#: only qualifies from ~2026-07-30 (before that `predictor_universe_cut` named a
#: different cut), so a wide window costs reads that can never contribute — but
#: it must comfortably exceed the 21-day maturation lag plus the evidence floor,
#: or the arm would be starved of history by its own reader.
PREDICTIONS_LOOKBACK_DAYS = 180

_DATED_PREDICTIONS_RE = None


def load_dated_predictions(
    bucket: str,
    *,
    s3_client=None,
    lookback_days: int = PREDICTIONS_LOOKBACK_DAYS,
    today: str | None = None,
) -> dict[str, list[dict]]:
    """``{date: predictions list}`` from ``predictor/predictions/{date}.json``.

    The dated artifacts, never ``latest.json``: ``latest`` is a mutable pointer
    and scoring history through it would attribute every past cohort to today's
    cut. The dated objects are what the arm actually consumed on the day.

    Fail-soft PER DATE — one unreadable object costs that cohort, not the run —
    but never silently: each miss is logged, and the caller sees a short dict
    rather than an exception it would have to distinguish from "no history".
    """
    import re
    from datetime import date as _date
    from datetime import timedelta as _td

    global _DATED_PREDICTIONS_RE
    if _DATED_PREDICTIONS_RE is None:
        _DATED_PREDICTIONS_RE = re.compile(r"predictions/(\d{4}-\d{2}-\d{2})\.json$")

    import boto3

    s3 = s3_client or boto3.client("s3")
    horizon = (
        _date.fromisoformat(today) if today else _date.today()
    ) - _td(days=lookback_days)

    keys: list[str] = []
    token = None
    while True:
        kwargs = {"Bucket": bucket, "Prefix": PREDICTIONS_PREFIX}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents") or []:
            m = _DATED_PREDICTIONS_RE.search(obj["Key"])
            if not m:
                continue
            try:
                if _date.fromisoformat(m.group(1)) < horizon:
                    continue
            except ValueError:
                continue
            keys.append(obj["Key"])
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")

    import json as _json

    out: dict[str, list[dict]] = {}
    n_unreadable = 0
    for key in sorted(keys):
        eval_date = _DATED_PREDICTIONS_RE.search(key).group(1)
        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            out[eval_date] = _json.loads(body).get("predictions") or []
        except Exception as exc:  # noqa: BLE001 — one bad object costs one cohort
            n_unreadable += 1
            logger.warning(
                "[arm_realized_lift] s3://%s/%s unreadable (%s) — that cohort "
                "date will not contribute", bucket, key, exc,
            )
    logger.info(
        "[arm_realized_lift] loaded %d dated predictions file(s) from the last "
        "%dd (%d unreadable)", len(out), lookback_days, n_unreadable,
    )
    return out
