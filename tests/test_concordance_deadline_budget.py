"""config#6920 — the concordance replay stops early instead of being killed.

`ne-weekly-freshness-pipeline` execution `watch-rerun-2026-08-10-5`
(2026-08-11): `ReplayConcordance` ran for the full 900s and died at
``Status: timeout``. Nothing was persisted — the aggregate and the summary PUT
happen only after the replay loop finishes — so fifteen minutes of paid LLM
calls produced no artifact, and the stage degraded silently.

The cap that was supposed to prevent this read:

    150 artifacts × ~3-5 sec/replay ≈ 450-750 sec

Measured per-item latencies that day were **6s to 137s**. The cap was sized on
an assumed constant that reality had invalidated by an order of magnitude, and
nothing checked the clock at runtime — so the guard could never bind.

These tests pin the replacement: the loop measures its own item latencies,
stops while there is still time to write, and labels the result incomplete so a
truncated sweep can never be read as a full one.
"""

from __future__ import annotations

import pytest

from replay.batch import (
    CONCORDANCE_MIN_ITEM_ESTIMATE_S,
    CONCORDANCE_WRITE_RESERVE_S,
    _next_item_affordable,
)


class _Clock:
    """Seconds remaining; decrements by `per_call` on every read."""

    def __init__(self, seconds: float, per_call: float = 0.0) -> None:
        self.seconds = seconds
        self.per_call = per_call

    def __call__(self) -> float:
        value = self.seconds
        self.seconds -= self.per_call
        return value


def test_no_deadline_means_never_stop():
    """Spot runs and CLI runs have no wall — behaviour is unchanged."""
    affordable, needed = _next_item_affordable(None, [])
    assert affordable is True
    assert needed == 0.0


def test_a_fresh_run_uses_the_floor_estimate():
    """Before any item completes there is nothing to measure."""
    affordable, needed = _next_item_affordable(_Clock(900), [])
    assert affordable is True
    assert needed == CONCORDANCE_MIN_ITEM_ESTIMATE_S + CONCORDANCE_WRITE_RESERVE_S


def test_the_estimate_tracks_this_run_s_own_latencies():
    """The workload measures itself rather than trusting a literal.

    The 2026-08-11 distribution: mostly 6-20s with a 137s outlier. p90 must
    reflect the tail, not the median — sizing on the median is precisely the
    mistake the old 3-5s constant made.
    """
    latencies = [6276, 10227, 10713, 14182, 20673, 36280, 136538]
    _, needed = _next_item_affordable(_Clock(900), latencies)
    assert needed > 36.0 + CONCORDANCE_WRITE_RESERVE_S, (
        "the estimate ignored the slow tail — a p50-sized budget starts an "
        "item the deadline cannot cover"
    )


def test_an_item_is_declined_when_the_write_reserve_would_be_eaten():
    """Stopping with time to persist beats being killed with nothing."""
    remaining = CONCORDANCE_MIN_ITEM_ESTIMATE_S + CONCORDANCE_WRITE_RESERVE_S - 1
    affordable, _ = _next_item_affordable(_Clock(remaining), [])
    assert affordable is False


def test_the_reserve_is_actually_held_back():
    """A budget covering the item but not the write must still decline."""
    latencies = [10_000]  # 10s items
    remaining = CONCORDANCE_MIN_ITEM_ESTIMATE_S + 1  # room for an item, not the write
    affordable, needed = _next_item_affordable(_Clock(remaining), latencies)
    assert affordable is False
    assert needed >= CONCORDANCE_WRITE_RESERVE_S


# ---------------------------------------------------------------------------
# The loop and the summary
# ---------------------------------------------------------------------------

def _stub_replay(latency_ms: int = 5_000):
    class _Replay:
        replay_output_kind = "structured"
        replay_error = None
        replay_cost = {"input_tokens": 1, "output_tokens": 1}
        replay_latency_ms = latency_ms
        comparison = {"agent_id_base": "sector_quant", "agreement_score": 0.9}

    return _Replay()


def _run(monkeypatch, *, keys: int, clock):
    import replay.batch as B

    calls = {"n": 0}

    def _fake_replay_artifact(**kwargs):
        calls["n"] += 1
        return _stub_replay()

    monkeypatch.setattr(B, "replay_artifact", _fake_replay_artifact)
    monkeypatch.setattr(
        B, "_list_artifact_keys_in_window",
        lambda *a, **k: [f"decision_artifacts/2026/08/10/sector_quant:x{i}.json" for i in range(keys)],
    )
    monkeypatch.setattr(B, "_persist_batch_summary", lambda *a, **k: "s3://summary.json")

    summary = B.compute_and_emit_concordance(
        target_models=["deepseek/deepseek-v4-flash"],
        bucket="bucket",
        s3_client=object(),
        cloudwatch_client=None,
        emit_metrics=False,
        remaining_s=clock,
    )
    return summary["per_target_model"][0], calls


def test_a_starved_run_replays_nothing_and_says_so(monkeypatch):
    target, calls = _run(monkeypatch, keys=50, clock=_Clock(1))

    assert calls["n"] == 0, "an item was started with no time to finish it"
    assert target["complete"] is False
    assert target["budget_stopped"] is True
    assert target["n_artifacts_skipped_for_budget"] == 50
    assert target["n_artifacts_candidate"] == 50


def test_a_run_that_runs_out_partway_keeps_what_it_did(monkeypatch):
    """The whole point: a partial aggregate reaches S3 instead of nothing."""
    target, calls = _run(monkeypatch, keys=40, clock=_Clock(400, per_call=60))

    assert 0 < calls["n"] < 40
    assert target["complete"] is False
    assert target["n_artifacts_replayed"] == calls["n"]
    assert target["n_artifacts_skipped_for_budget"] == 40 - calls["n"]
    assert target["n_artifacts_replayed"] + target["n_artifacts_skipped_for_budget"] == 40


def test_a_healthy_run_is_marked_complete(monkeypatch):
    target, calls = _run(monkeypatch, keys=5, clock=_Clock(3600))

    assert calls["n"] == 5
    assert target["complete"] is True
    assert target["budget_stopped"] is False
    assert target["n_artifacts_skipped_for_budget"] == 0


def test_an_undeadlined_run_is_complete(monkeypatch):
    target, calls = _run(monkeypatch, keys=5, clock=None)

    assert calls["n"] == 5
    assert target["complete"] is True


# ---------------------------------------------------------------------------
# The handler must not call a truncated sweep OK
# ---------------------------------------------------------------------------

def test_budget_stopped_downgrades_the_handler_status():
    import inspect

    from lambda_concordance import handler as H

    # `_run`, not `handler`: config-I7423 split the entry point into a thin
    # `handler` wrapper whose only job is the cost-sink flush in a `finally`,
    # and `_run`, which carries the status logic this test pins.
    source = inspect.getsource(H._run)
    assert 'budget_stopped' in source, (
        "the handler ignores budget_stopped, so a truncated sweep returns OK "
        "and the SF treats a partial corpus as a full one"
    )


def test_remaining_seconds_is_none_without_a_lambda_context():
    from lambda_concordance.handler import _remaining_seconds

    assert _remaining_seconds(None) is None
    assert _remaining_seconds(object()) is None


def test_remaining_seconds_converts_millis():
    from lambda_concordance.handler import _remaining_seconds

    class _Ctx:
        @staticmethod
        def get_remaining_time_in_millis():
            return 120_000

    fn = _remaining_seconds(_Ctx())
    assert fn() == pytest.approx(120.0)
