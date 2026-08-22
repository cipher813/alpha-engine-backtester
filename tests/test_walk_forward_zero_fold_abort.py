"""A walk-forward pass that builds ZERO folds has measured nothing, and must
say so (alpha-engine-config-I7309).

Sibling to `tests/test_pit_parity_smoke_fold_arithmetic.py`, which stops the
fixture from getting into this state. This file covers the state itself: the
fold count is a function of a config the operator can change, so the code has
to stay honest for schemes nobody has thought of yet.

Before the fix the zero-fold path was the ONE path through
`run_walk_forward_inference` that produced no signal of any kind:

    if folds and n_test_dates_scored == 0:   # <- gated on `folds`
        logger.error("... ZERO test dates scored ...")

With `folds == []` the guard is skipped, `{}` is returned, and every consumer
downstream degrades quietly into `no_orders` — an outcome sentence about the
RISK RULES, produced by a run that never reached them. Per the fleet fail-loud
rule the default is RAISE: an input degeneracy is not a product.
"""

from __future__ import annotations

import re

import pytest

from synthetic import predictor_backtest as pb


def _dates(n: int) -> list[str]:
    import datetime as dt
    base = dt.date(2024, 1, 2)
    return [(base + dt.timedelta(days=i)).isoformat() for i in range(n)]


def _run(monkeypatch, *, n_dates, wf_params=None):
    """Drive `run_walk_forward_inference` on a REAL date axis of `n_dates`.

    Nothing about the fold builder is stubbed — the whole point is that the
    fold count is genuine arithmetic over the axis the caller supplied, which
    is what production got wrong. Only the two things the guard sits in front
    of are stubbed: the predictor's momentum scorer (an import from a sibling
    checkout) and the inference tensor (a feature store), so the test stays
    hermetic and a NON-degenerate axis can still reach a clean return.
    """
    import sys
    import types
    mod = types.ModuleType("model.momentum_scorer")
    mod.predict_array = lambda X, names: [0.0] * len(X)
    pkg = types.ModuleType("model")
    pkg.momentum_scorer = mod
    monkeypatch.setitem(sys.modules, "model", pkg)
    monkeypatch.setitem(sys.modules, "model.momentum_scorer", mod)

    class _EmptyTensor:
        shape = (0, 0, 0)

    monkeypatch.setattr(
        pb, "build_inference_tensor",
        lambda features, names: (_EmptyTensor(), [], {}),
    )
    return pb.run_walk_forward_inference(
        {}, _dates(n_dates), "/nonexistent/predictor",
        bucket="test-bucket", wf_params=wf_params or {},
    )


def test_zero_folds_raises_instead_of_returning_empty_predictions(monkeypatch):
    """THE regression test. On the pre-fix tree this returns `({}, stats)` and
    the caller reports `no_orders`."""
    with pytest.raises(RuntimeError) as exc:
        _run(monkeypatch, n_dates=40)
    msg = str(exc.value)
    assert "ZERO folds" in msg
    # The message has to carry BOTH sides of the arithmetic, or the operator
    # is left rederiving the thing that took three weeks to see.
    assert "40 trading date(s)" in msg
    assert "min_train=504" in msg
    assert re.search(r"At least \d+ trading date\(s\) are required", msg), msg
    # And it must name the two knobs that fix it.
    assert "max_trading_days" in msg and "min_train" in msg


def test_the_abort_names_an_unusable_scheme_differently(monkeypatch):
    """"Needs a longer axis" and "these parameters can never work" are
    different operator actions, so they are different sentences. A `min_train`
    beyond any real market history is the second."""
    with pytest.raises(RuntimeError) as exc:
        _run(monkeypatch, n_dates=40, wf_params={"min_train": 500_000})
    msg = str(exc.value)
    assert "ZERO folds" in msg
    assert "unusable for any realistic date axis" in msg
    assert "At least" not in msg


def test_an_axis_long_enough_to_build_folds_does_not_raise(monkeypatch):
    """The guard must bind on the degenerate case ONLY. A short slice with a
    fold scheme scaled to it — exactly what the smoke fixture now carries —
    runs straight through."""
    preds, stats = _run(
        monkeypatch, n_dates=40,
        wf_params={"test_window": 5, "min_train": 10, "purge": 2, "embargo": 1},
    )
    assert preds == {}
    assert stats["n_folds"] >= 1
