"""Purged & embargoed walk-forward fold splitter for point-in-time backtesting.

Second pure building block of PIT discipline (ROADMAP L2349 / Backtester
Phase 2, P1). Plan: ``alpha-engine-docs/private/pit-discipline-260515.md``.

Institutional basis (López de Prado, *Advances in Financial ML* Ch. 7):
walk-forward folds with a **purge** gap between train-end and test-start (so the
test window's label-formation period cannot leak into training) and an
**embargo** after each test window (so serially-correlated features just after a
test fold cannot leak into the next train set).

Consistency-with-production note (grounded against current code 2026-05-15):
the predictor's own walk-forward (``meta_trainer.py`` ~1058-1080) is
**expanding-train + purge** (``train_mask = d <= train_end_date`` — no lower
bound), advancing the fold start by one test window, with
``WF_TEST_WINDOW_DAYS`` / ``WF_MIN_TRAIN_DAYS`` / ``WF_PURGE_DAYS``. This module
mirrors that index logic exactly and *adds* the embargo (the predictor has purge
but no embargo). ``train_mode`` defaults to ``"expanding"`` to genuinely match
the predictor (the plan doc's "rolling matches predictor" wording is a
documentation error — predictor is expanding; the plan's *intent* of
predictor-consistency is honored here, and the doc should be corrected). A
``"rolling"`` mode is provided for the sweepable-variant the plan anticipated.

Pure: operates on an ordered list of unique trading dates, no I/O, no S3, no
sweep wiring — unit-tested in isolation before anything consumes it.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass


@dataclass(frozen=True)
class Fold:
    """One walk-forward fold over an ordered unique-date axis.

    Index fields are positions into the ``unique_dates`` list passed to
    :func:`build_walk_forward_folds`. ``train_end_date`` precedes
    ``test_start_date`` by at least ``purge`` trading days; the next fold's train
    set excludes the ``embargo`` days immediately after ``test_end_date``.
    """

    train_start_idx: int
    train_end_idx: int
    test_start_idx: int
    test_end_idx: int
    train_start_date: _dt.date
    train_end_date: _dt.date
    test_start_date: _dt.date
    test_end_date: _dt.date


def build_walk_forward_folds(
    unique_dates: list[_dt.date],
    *,
    test_window: int,
    min_train: int,
    purge: int,
    embargo: int,
    train_mode: str = "expanding",
) -> list[Fold]:
    """Build purged + embargoed walk-forward folds.

    Parameters mirror the predictor's WF config so the two stay consistent:
      - ``test_window``  -> WF_TEST_WINDOW_DAYS (e.g. ~21 trading days = 1mo)
      - ``min_train``    -> WF_MIN_TRAIN_DAYS (first fold's minimum train length)
      - ``purge``        -> WF_PURGE_DAYS; default per plan = canonical label
        horizon = 21 trading days (the test label window cannot touch train)
      - ``embargo``      -> trading days after each test window excluded from the
        *next* fold's train set (plan default = 2; LdP lower bound)
      - ``train_mode``   -> "expanding" (default, matches predictor:
        train = all dates <= train_end) or "rolling" (train =
        ``test_window``-bounded lookback ending at train_end)

    Returns folds in chronological order. A fold is emitted only if a valid
    train block of length >= ``min_train`` // 2 exists after purging — matching
    the predictor's guard so degenerate early folds are skipped rather than
    silently producing a tiny train set.

    Raises ValueError on non-positive sizing params or an unknown train_mode so
    a misconfigured sweep fails loud rather than producing leaky folds.
    """
    if test_window <= 0 or min_train <= 0:
        raise ValueError("test_window and min_train must be positive")
    if purge < 0 or embargo < 0:
        raise ValueError("purge and embargo must be non-negative")
    if train_mode not in ("expanding", "rolling"):
        raise ValueError(f"unknown train_mode {train_mode!r}")

    n = len(unique_dates)
    folds: list[Fold] = []
    fold_start_idx = min_train
    while fold_start_idx < n:
        remaining = n - fold_start_idx
        # Predictor guard: stop once less than half a test window remains.
        if remaining < test_window // 2:
            break

        test_start_idx = fold_start_idx
        test_end_idx = min(fold_start_idx + test_window - 1, n - 1)

        # Purge: train ends `purge` trading days before the test window opens.
        train_end_idx = fold_start_idx - purge
        if train_end_idx < min_train // 2:
            fold_start_idx += test_window
            continue

        if train_mode == "expanding":
            train_start_idx = 0
        else:  # rolling: bounded lookback of one test_window ending at train_end
            train_start_idx = max(0, train_end_idx - test_window + 1)

        if train_end_idx < train_start_idx:
            fold_start_idx += test_window
            continue

        folds.append(
            Fold(
                train_start_idx=train_start_idx,
                train_end_idx=train_end_idx,
                test_start_idx=test_start_idx,
                test_end_idx=test_end_idx,
                train_start_date=unique_dates[train_start_idx],
                train_end_date=unique_dates[train_end_idx],
                test_start_date=unique_dates[test_start_idx],
                test_end_date=unique_dates[test_end_idx],
            )
        )

        # Embargo: the next fold's train set must not include the `embargo`
        # trading days immediately after this test window. We advance the fold
        # cursor by one test window (predictor cadence) and the embargo is
        # enforced structurally because the next fold's train_end_idx =
        # next_fold_start - purge, and purge >= embargo in the canonical config
        # (purge=21, embargo=2) so the post-test embargo region is already
        # outside the next train block. When embargo > purge we additionally
        # push the cursor so the gap is at least `embargo`.
        advance = test_window
        if embargo > purge:
            advance = max(test_window, test_window + (embargo - purge))
        fold_start_idx += advance

    return folds


def min_trading_dates_for_one_fold(
    *,
    test_window: int,
    min_train: int,
    purge: int,
    embargo: int,
    train_mode: str = "expanding",
    search_margin: int = 8,
    max_probe_length: int = 20_000,
) -> int | None:
    """Smallest date-axis length for which :func:`build_walk_forward_folds`
    emits at least one fold under these parameters — ``None`` if no length
    within the search window does.

    Derived BY CONSTRUCTION (the builder is called, not re-implemented) so it
    can never drift from the fold scheme it describes. That matters because it
    is the number two things depend on:

      * ``run_walk_forward_inference``'s zero-fold abort message, so an
        operator is told the axis length actually required rather than left to
        rederive it; and
      * the merge-blocking assertion that the ``smoke-pit-parity`` fixture's
        ``max_trading_days`` is arithmetically capable of building a fold at
        all (alpha-engine-config-I7309).

    The second is the one with history. The smoke fixture caps the axis at 60
    trading days while the canonical ``min_train`` is 504, so the fold loop's
    ``while fold_start_idx < n`` never executed: ZERO folds, empty predictions,
    zero ENTER signals, and a ``pit_status: no_orders`` that was read for three
    weeks as the walk-forward pass declining to trade. Any future edit to
    either number has to keep them compatible, and a hardcoded threshold would
    not notice.

    ``search_margin`` bounds the probe at ``min_train + search_margin *
    max(test_window, 1) + purge``, and ``max_probe_length`` bounds it
    absolutely: a scheme whose first fold needs a longer date axis than any
    real market history could supply is reported as ``None`` rather than
    probed, so a nonsense ``min_train`` cannot turn this into an
    out-of-memory.
    """
    if test_window <= 0 or min_train <= 0 or purge < 0 or embargo < 0:
        return None
    ceiling = min_train + search_margin * max(test_window, 1) + purge
    if ceiling > max_probe_length:
        return None
    base = _dt.date(2000, 1, 3)  # arbitrary; only the axis LENGTH matters
    axis = [base + _dt.timedelta(days=i) for i in range(ceiling)]
    # The builder's cursor starts at ``min_train`` and requires
    # ``fold_start_idx < n``, so no axis of length <= min_train can ever emit a
    # fold. Start the probe there rather than at 1 — the search is otherwise
    # quadratic in ``min_train`` (504 canonically, and unbounded when a caller
    # misconfigures it).
    for n in range(min_train + 1, ceiling + 1):
        folds = build_walk_forward_folds(
            axis[:n],
            test_window=test_window,
            min_train=min_train,
            purge=purge,
            embargo=embargo,
            train_mode=train_mode,
        )
        if folds:
            return n
    return None
