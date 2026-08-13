"""config#7209 / config#7214 — `sizing_ab.json` must exist on every run.

`analysis/sizing_ab.py::run_sizing_ab` shipped with tests and its output had
never been persisted for any date, so there was no observation distinguishing
"the comparison was not wanted" from "the producer died". These tests pin the
three properties that make the artifact a usable coverage signal:

1. it is written to the PUBLIC `backtest/{date}/` namespace, not the internal
   `.phases/` resume namespace;
2. it is ALWAYS-EMIT — a degraded or errored producer body still writes;
3. an S3 failure is fail-soft and loud, never an abort of the backtest.
"""

import json

import pytest

from backtest import _persist_sizing_ab


class _FakeS3:
    def __init__(self, raises: Exception | None = None):
        self.raises = raises
        self.puts: list[dict] = []

    def put_object(self, **kw):
        if self.raises is not None:
            raise self.raises
        self.puts.append(kw)
        return {}


def test_key_is_the_public_backtest_namespace():
    s3 = _FakeS3()
    key = _persist_sizing_ab("b", "2026-08-15", {"status": "ok"}, s3_client=s3)
    assert key == "backtest/2026-08-15/sizing_ab.json"
    assert s3.puts[0]["Key"] == key
    assert ".phases/" not in key, "must not land in the internal resume namespace"
    assert s3.puts[0]["Bucket"] == "b"
    assert s3.puts[0]["ContentType"] == "application/json"


@pytest.mark.parametrize(
    "result",
    [
        {"status": "ok", "sharpe_diff": 0.2, "assessment": "sizing_helps"},
        {"status": "insufficient_data", "trades_a": 3, "trades_b": 4},
        {"status": "error", "error": "boom"},
        {"status": "not_run", "reason": "sizing A/B stage did not execute"},
    ],
)
def test_always_emits_whatever_the_status(result):
    """ALWAYS-EMIT: absence of the key means the stage never ran, and nothing
    else. A status-gated write makes a degraded producer indistinguishable
    from a dead one — the defect this artifact was created to close."""
    s3 = _FakeS3()
    _persist_sizing_ab("b", "2026-08-15", result, s3_client=s3)
    assert len(s3.puts) == 1
    body = json.loads(s3.puts[0]["Body"].decode())
    assert body["status"] == result["status"]
    assert body["run_date"] == "2026-08-15"


def test_run_date_is_stamped_into_the_body():
    """A reader must be able to tell which run wrote this without parsing the
    key — the key can be copied, the body cannot be copied by accident."""
    s3 = _FakeS3()
    _persist_sizing_ab("b", "2026-08-15", {"status": "ok"}, s3_client=s3)
    assert json.loads(s3.puts[0]["Body"].decode())["run_date"] == "2026-08-15"


def test_s3_failure_is_fail_soft_and_logged(caplog):
    """The backtest's primary deliverables are already computed by this point;
    a sizing-comparison persist failure must not abort them. It must still be
    LOUD — the freshness row for this key is the recording surface."""
    s3 = _FakeS3(raises=RuntimeError("s3 down"))
    key = _persist_sizing_ab("b", "2026-08-15", {"status": "ok"}, s3_client=s3)
    assert key == "backtest/2026-08-15/sizing_ab.json"
    assert any(
        r.levelname == "ERROR" and "sizing_ab" in r.getMessage()
        for r in caplog.records
    ), "a swallowed persist failure must still be recorded at ERROR"


def test_producer_is_reachable_and_unchanged():
    """The producer exists — this arc is wiring, not authorship. If this import
    breaks, the artifact above is writing a shape nothing produces."""
    from analysis.sizing_ab import run_sizing_ab

    calls: list[dict] = []

    def sim_fn(cfg):
        calls.append(cfg)
        return {"total_trades": 100, "sharpe_ratio": 1.0, "total_return": 0.1,
                "alpha": 0.02}

    result = run_sizing_ab(sim_fn, {"atr_sizing_enabled": True}, min_trades=1)
    assert len(calls) == 2, "A/B must run exactly two simulations"
    assert calls[0]["atr_sizing_enabled"] is True
    assert calls[1]["atr_sizing_enabled"] is False
    assert result["status"] == "ok"
