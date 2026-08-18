"""pillar_weight_optimizer is retired — alpha-engine-config-I7637.

It scores from `{pillar}_quant` / `{pillar}_qual` and `legacy_blend_score`,
which came from `investment_thesis.composite_breakdown` — emitted by the
six-team + CIO research graph retired 2026-07-12 (config#1580).

Measured against live S3 on 2026-08-18, not inferred from the code:

  signals/latest.json                        investment_thesis on 0 of 903
  per-name shape                             {quant_score, qual_score,
                                              sub_scores: {quant, qual}}
  config/scoring_weights_shadow_history/     empty; latest.json 404s

So option (a) of I7637 — wire the columns — has no producer to wire. This is
option (b), declared and dated the way team_metrics was retired under #7616.
"""
from __future__ import annotations

import pandas as pd
import pytest

from optimizer import pillar_weight_optimizer as pwo


def _df_that_would_have_scored():
    """A frame carrying everything `recommend` used to need. Retirement must
    not depend on the input being absent — the module refuses even when it
    could compute, so a producer quietly reappearing does not silently rearm
    a path nobody has validated."""
    from nousergon_lib.pillars import PILLARS

    row = {"score_date": "2026-08-14", "legacy_blend_score": 50.0}
    for pillar in PILLARS:
        row[f"{pillar}_quant"] = 60.0
        row[f"{pillar}_qual"] = 40.0
    return pd.DataFrame([row, {**row, "score_date": "2026-08-07"}])


def test_recommend_returns_retired_with_a_date_and_a_reason():
    res = pwo.recommend(_df_that_would_have_scored())
    assert res["status"] == "retired"
    assert res["retired_on"] == "2026-08-18"
    assert "config#1580" in res["retired_reason"]
    assert "0 of 903" in res["retired_reason"]
    assert "I7637" in res["retired_reason"]


def test_retirement_is_not_a_silent_early_return():
    """The pre-existing shape returned early on missing columns and said
    nothing, which is why an optimizer with no inputs read as a dormant feature
    for two months. The status must be legible from the artifact."""
    res = pwo.recommend(pd.DataFrame())
    assert res["status"] == "retired"
    assert res["status"] not in ("insufficient_data", "no_subscores", "ok")
    assert res["note"].startswith("pillar_weight_optimizer was retired")


def test_apply_writes_nothing_for_a_retired_result():
    res = pwo.recommend(_df_that_would_have_scored())
    out = pwo.apply(res, "alpha-engine-research")
    assert out["applied"] is False
    assert "retired" in out["reason"]


def test_the_live_scoring_weights_key_is_still_never_written():
    """The pre-existing hard invariant, re-asserted: retiring the module must
    not be the change that loosens it.

    Checked on the AST rather than the text, so the docstring naming the key it
    must never write does not itself trip the assertion."""
    import ast
    import pathlib

    tree = ast.parse(pathlib.Path(pwo.__file__).read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "apply")
    names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
    assert "S3_WEIGHTS_KEY" not in names
    assert "S3_SHADOW_WEIGHTS_PREFIX" in names


@pytest.mark.parametrize("attr", ["RETIRED_ON", "RETIRED_REASON"])
def test_the_retirement_record_is_importable(attr):
    """A reader — or a console driver — can ask the module when and why,
    instead of inferring it from an empty S3 prefix."""
    assert getattr(pwo, attr)
