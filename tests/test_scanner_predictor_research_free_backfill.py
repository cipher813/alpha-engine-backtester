"""Tests for the research-free meta-ensemble backfill producer (config#1405,
build items 1+2; S3-sourced pending-universe + freshness assertion added by
config#3053) — ``analysis/scanner_predictor_research_free_backfill.py``.

A synthetic sqlite fixture (no ArcticDB needed) plus a fake S3 client
exercise the producer's logic — idempotency (skip-if-cached), the
(ticker, eval_date) pending-universe query against S3
``candidates/{date}/candidates.json::scanner_eval_log`` (config#3053 — NOT
the retired ``scanner_evaluations`` sqlite table), table creation / schema,
the champion-feed freshness assertion, and the research-free feature-zeroing
contract (``_assemble_research_free_features``). The ArcticDB-backed feature
computation (``run_backfill`` end-to-end) is exercised in the PR description
against the LIVE production store instead — not reproducible hermetically
here, per the issue's own testing section ("the meta-ensemble backfill
validates on the spot run").
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from botocore.exceptions import ClientError

from analysis.scanner_predictor_research_free_backfill import (
    ARTIFACT_KEY,
    RESEARCH_META_FEATURES,
    TABLE_NAME,
    NoCandidatesArtifactError,
    StaleChampionFeedError,
    _assemble_research_free_features,
    _ensure_table,
    _existing_keys,
    _list_recent_candidate_dates,
    _pending_universe,
    _predictor_cut_members,
    assert_champion_feed_fresh,
)


# ── Fake S3: download_file/upload_file (parquet) + list_objects_v2/get_object
# (candidates.json) backed by an in-memory dict of key -> bytes/local path.
# Mirrors tests/test_reporter.py's injected s3_client idiom.


class _FakeS3:
    def __init__(self, tmp_path):
        self._tmp = tmp_path
        self._objects: dict[str, str] = {}   # key -> local filepath (parquet)
        self._bodies: dict[str, bytes] = {}   # key -> raw bytes (candidates.json)
        self.upload_calls: list[tuple[str, str, str]] = []

    def put_local(self, key: str, local_path: str) -> None:
        self._objects[key] = local_path

    def put_candidates(self, date: str, artifact: dict) -> None:
        key = f"candidates/{date}/candidates.json"
        self._bodies[key] = json.dumps(artifact).encode("utf-8")

    def put_membership(self, date: str, artifact: dict) -> None:
        key = f"universe_membership/{date}/membership.json"
        self._bodies[key] = json.dumps(artifact).encode("utf-8")

    def download_file(self, bucket, key, dest):
        import shutil

        if key not in self._objects:
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
            )
        shutil.copyfile(self._objects[key], dest)

    def upload_file(self, src, bucket, key):
        import shutil

        stored = str(self._tmp / f"stored_{key.replace('/', '_')}")
        shutil.copyfile(src, stored)
        self._objects[key] = stored
        self.upload_calls.append((src, bucket, key))

    def get_object(self, Bucket, Key):
        if Key in self._bodies:
            return {"Body": _Body(self._bodies[Key])}
        if Key in self._objects:
            with open(self._objects[Key], "rb") as fh:
                return {"Body": _Body(fh.read())}
        raise ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject"
        )

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, ContinuationToken=None):
        dates = sorted({
            key[len(Prefix):].split("/")[0]
            for key in self._bodies
            if key.startswith(Prefix)
        })
        return {
            "CommonPrefixes": [{"Prefix": f"{Prefix}{d}/"} for d in dates],
            "IsTruncated": False,
        }


class _Body:
    def __init__(self, data: bytes):
        self._data = data

    def read(self):
        return self._data


# Fixture dates are RELATIVE, not literal (config#6920 sweep finding).
#
# This module used hardcoded 2026-04-12 / 2026-04-20 while `_pending_universe`
# filters on `lookback_days=120` counted from *today*. On 2026-08-11 the older
# date turned 121 days old and silently fell out of the window: three tests
# started failing on main with `assert 3 == 6`, with nothing having changed in
# the code. A fixed date inside a relative window is a time bomb with a
# knowable fuse — these are anchored to now so the fuse cannot be lit.
_LOOKBACK_DAYS = 120  # mirrors _pending_universe's default; see the guard test
_D_OLDEST = (date.today() - timedelta(days=110)).isoformat()
_D_OLD = (date.today() - timedelta(days=100)).isoformat()
_D_RECENT = (date.today() - timedelta(days=90)).isoformat()


def _candidates_artifact(run_date: str, eval_log: list[dict]) -> dict:
    return {"run_date": run_date, "scanner_eval_log": eval_log}


def _eval_rows(passing: list[str], failing: list[str]) -> list[dict]:
    return [
        {"ticker": t, "quant_filter_pass": 1} for t in passing
    ] + [
        {"ticker": t, "quant_filter_pass": 0} for t in failing
    ]


def _seeded_s3(tmp_path, *, dates=(_D_OLD, _D_RECENT)) -> _FakeS3:
    """A fake S3 carrying candidates.json for each of ``dates`` with 3
    passing (T0/T1/T2) + 2 failing (T3/T4) tickers — the S3 analog of the
    old ``_scanner_db`` sqlite fixture this module used before config#3053."""
    s3 = _FakeS3(tmp_path)
    for d in dates:
        s3.put_candidates(d, _candidates_artifact(d, _eval_rows(["T0", "T1", "T2"], ["T3", "T4"])))
    return s3


def _prefill(conn, rows):
    _ensure_table(conn)
    for ticker, d, alpha in rows:
        conn.execute(f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)", (ticker, d, alpha, 4))
    conn.commit()


# ── _ensure_table / schema ───────────────────────────────────────────────────


def test_ensure_table_creates_expected_schema(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    cols = {r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")}
    assert cols == {"ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"}, cols


def test_ensure_table_is_idempotent_call(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    _ensure_table(conn)  # must not raise on a second call
    n = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0]
    assert n == 0


def test_table_matches_consumer_contract_from_end_to_end_test():
    """The schema this producer writes must exactly match the frozen contract
    ``tests/test_scanner_then_predictor.py`` builds and
    ``analysis/end_to_end.py::_scanner_then_predictor_topN`` reads — same
    table name, same column set, same join-key semantics
    (``prediction_date`` == the scanner's ``eval_date``)."""
    import sqlite3 as _sq

    conn = _sq.connect(":memory:")
    _ensure_table(conn)
    cols = [r[1] for r in conn.execute(f"PRAGMA table_info({TABLE_NAME})")]
    assert TABLE_NAME == "predictor_outcomes_research_free"
    assert cols == ["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"]


# ── _pending_universe / idempotency (skip-if-cached) — config#3053: sourced
# from S3 candidates.json::scanner_eval_log, not the retired
# scanner_evaluations sqlite table. ───────────────────────────────────────


def test_pending_universe_returns_all_passing_rows_when_nothing_cached(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _seeded_s3(tmp_path)
    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)
    # 3 passing tickers x 2 dates = 6 rows, none cached yet
    assert len(pending) == 6, pending
    assert set(pending["ticker"]) == {"T0", "T1", "T2"}
    assert set(pending["eval_date"]) == {_D_OLD, _D_RECENT}


def test_pending_universe_excludes_already_cached_rows(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    _prefill(conn, [("T0", _D_OLD, 0.01), ("T1", _D_OLD, -0.02)])
    s3 = _seeded_s3(tmp_path)
    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)
    # 6 total - 2 cached = 4 remaining
    assert len(pending) == 4, pending
    pairs = set(zip(pending["ticker"], pending["eval_date"]))
    assert ("T0", _D_OLD) not in pairs
    assert ("T1", _D_OLD) not in pairs
    assert ("T2", _D_OLD) in pairs
    assert ("T0", _D_RECENT) in pairs


def test_pending_universe_empty_when_fully_cached(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    _prefill(conn, [
        (f"T{i}", d, 0.0)
        for d in (_D_OLD, _D_RECENT)
        for i in range(3)
    ])
    s3 = _seeded_s3(tmp_path)
    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)
    assert pending.empty, pending


def test_pending_universe_raises_when_no_candidates_artifact_in_window(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "bad.db"))
    _ensure_table(conn)
    s3 = _FakeS3(tmp_path)  # no candidates.json written at all
    try:
        _pending_universe(conn, bucket="any-bucket", s3_client=s3)
        assert False, "expected NoCandidatesArtifactError"
    except NoCandidatesArtifactError as e:
        assert "candidates.json" in str(e)


def test_pending_universe_skips_week_with_empty_eval_log_but_uses_others(tmp_path):
    """config#3053 item (c): a scan week producing zero eval rows is a
    Scanner-side contract violation, logged and skipped — not silently
    treated as identical to 'nothing to backfill' for the WHOLE window when
    another week in the same window has real data."""
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _FakeS3(tmp_path)
    s3.put_candidates(_D_OLD, _candidates_artifact(_D_OLD, []))
    s3.put_candidates(_D_RECENT, _candidates_artifact(
        _D_RECENT, _eval_rows(["T0"], ["T1"]),
    ))
    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)
    assert set(zip(pending["ticker"], pending["eval_date"])) == {("T0", _D_RECENT)}


def test_pending_universe_respects_lookback_window(tmp_path):
    """A candidates.json older than lookback_days is not considered."""
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _FakeS3(tmp_path)
    s3.put_candidates("2020-01-04", _candidates_artifact(
        "2020-01-04", _eval_rows(["ANCIENT"], []),
    ))
    try:
        _pending_universe(conn, bucket="any-bucket", s3_client=s3, lookback_days=120)
        assert False, "expected NoCandidatesArtifactError"
    except NoCandidatesArtifactError:
        pass


def test_existing_keys_reads_ticker_prediction_date_pairs(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _prefill(conn, [("T0", _D_OLD, 0.01)])
    keys = _existing_keys(conn)
    assert keys == {("T0", _D_OLD)}


def test_existing_keys_empty_when_table_absent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    assert _existing_keys(conn) == set()


# ── _list_recent_candidate_dates ─────────────────────────────────────────────


def test_list_recent_candidate_dates_sorted_and_filtered(tmp_path):
    s3 = _seeded_s3(tmp_path, dates=(_D_OLD, _D_RECENT, _D_OLDEST))
    dates = _list_recent_candidate_dates("any-bucket", s3_client=s3, lookback_days=365)
    assert dates == [_D_OLDEST, _D_OLD, _D_RECENT]


def test_list_recent_candidate_dates_empty_when_no_objects(tmp_path):
    s3 = _FakeS3(tmp_path)
    assert _list_recent_candidate_dates("any-bucket", s3_client=s3) == []


# ── assert_champion_feed_fresh (config#3053) ─────────────────────────────────


def _write_parquet_artifact(s3: _FakeS3, tmp_path, rows) -> None:
    import pandas as pd

    df = pd.DataFrame(rows, columns=["ticker", "prediction_date", "predicted_alpha", "n_research_features_missing"])
    local = tmp_path / "artifact.parquet"
    df.to_parquet(local, index=False)
    s3.put_local(ARTIFACT_KEY, str(local))


def test_assert_champion_feed_fresh_passes_within_window(tmp_path):
    s3 = _FakeS3(tmp_path)
    _write_parquet_artifact(s3, tmp_path, [("T0", "2026-07-17", 0.01, 4)])
    assert_champion_feed_fresh("any-bucket", run_date="2026-07-20", max_days=8, s3_client=s3)


def test_assert_champion_feed_fresh_raises_when_stale(tmp_path):
    s3 = _FakeS3(tmp_path)
    _write_parquet_artifact(s3, tmp_path, [("T0", "2026-07-10", 0.01, 4)])
    try:
        assert_champion_feed_fresh("any-bucket", run_date="2026-07-20", max_days=8, s3_client=s3)
        assert False, "expected StaleChampionFeedError"
    except StaleChampionFeedError as e:
        assert "stale" in str(e)


def test_assert_champion_feed_fresh_raises_when_artifact_missing(tmp_path):
    s3 = _FakeS3(tmp_path)  # no parquet uploaded
    try:
        assert_champion_feed_fresh("any-bucket", run_date="2026-07-20", max_days=8, s3_client=s3)
        assert False, "expected StaleChampionFeedError"
    except StaleChampionFeedError as e:
        assert "unreadable" in str(e)


def test_assert_champion_feed_fresh_raises_when_empty(tmp_path):
    s3 = _FakeS3(tmp_path)
    _write_parquet_artifact(s3, tmp_path, [])
    try:
        assert_champion_feed_fresh("any-bucket", run_date="2026-07-20", max_days=8, s3_client=s3)
        assert False, "expected StaleChampionFeedError"
    except StaleChampionFeedError as e:
        assert "empty" in str(e)


# ── _assemble_research_free_features — the research-free contract ───────────


def test_research_features_always_zeroed_even_if_available():
    """The 4 research meta-features are zeroed unconditionally — the whole
    point of the arm is "what if research never ran," not "zero only when
    missing." A ticker with a value sitting in momentum/macro dicts for a
    research-feature NAME must still be zeroed (defense: no legacy scorer
    dict accidentally supplies a 'research_*' key that leaks through)."""
    feat_names = [
        "research_calibrator_prob", "momentum_score", "expected_move",
        "research_composite_score", "research_conviction", "sector_macro_modifier",
        "macro_spy_20d_return", "regime_intensity_z",
    ]
    feats = _assemble_research_free_features(
        "AAPL", feat_names,
        momentum_scores={"AAPL": 0.4},
        resid_scores={},
        expected_moves={"AAPL": 0.02},
        macro_row={"macro_spy_20d_return": 0.03, "regime_intensity_z": 0.5},
    )
    for f in RESEARCH_META_FEATURES:
        if f in feat_names:
            assert feats[f] == 0.0, (f, feats)
    assert feats["momentum_score"] == 0.4
    assert feats["expected_move"] == 0.02
    assert feats["macro_spy_20d_return"] == 0.03
    assert feats["regime_intensity_z"] == 0.5


def test_residual_momentum_variant_schema():
    """The live deployed model may use residual_momentum_score instead of
    momentum_score/expected_move (observed in production — see module
    docstring); the assembler must read from ``resid_scores`` for that name
    without requiring momentum_score/expected_move to also be present."""
    feat_names = ["research_calibrator_prob", "residual_momentum_score", "macro_vix_level"]
    feats = _assemble_research_free_features(
        "MSFT", feat_names,
        momentum_scores={},
        resid_scores={"MSFT": -0.15},
        expected_moves={},
        macro_row={"macro_vix_level": 1.2},
    )
    assert feats["research_calibrator_prob"] == 0.0
    assert feats["residual_momentum_score"] == -0.15
    assert feats["macro_vix_level"] == 1.2


def test_unknown_feature_name_degrades_to_zero_not_crash():
    """A feature name this producer has no computer registered for (future
    model-schema drift) degrades to 0.0 rather than raising — matches
    MetaModel.predict_single's own .get(f, 0.0) missing-key contract."""
    feat_names = ["momentum_score", "some_future_feature_v9"]
    feats = _assemble_research_free_features(
        "GOOG", feat_names,
        momentum_scores={"GOOG": 0.1},
        resid_scores={},
        expected_moves={},
        macro_row={},
    )
    assert feats["some_future_feature_v9"] == 0.0
    assert feats["momentum_score"] == 0.1


def test_missing_ticker_in_component_dicts_degrades_to_zero():
    """A ticker absent from a component dict (e.g. the vol scorer failed for
    just this ticker) degrades that single feature to 0.0 rather than
    KeyError — the per-ticker graceful-degrade contract."""
    feat_names = ["momentum_score", "expected_move"]
    feats = _assemble_research_free_features(
        "ZZZZ", feat_names,
        momentum_scores={},  # ZZZZ absent
        resid_scores={},
        expected_moves={},  # ZZZZ absent
        macro_row={},
    )
    assert feats == {"momentum_score": 0.0, "expected_move": 0.0}


def test_n_research_features_missing_is_a_count_of_four_by_construction():
    """The RESEARCH_META_FEATURES set is exactly the 4 the issue names —
    a guard against silent drift in this module's constant."""
    assert RESEARCH_META_FEATURES == {
        "research_calibrator_prob",
        "research_composite_score",
        "research_conviction",
        "sector_macro_modifier",
    }
    assert len(RESEARCH_META_FEATURES) == 4


# ── Idempotent insert semantics (INSERT OR REPLACE on the PK) ───────────────


def test_insert_or_replace_on_primary_key_is_idempotent(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "e.db"))
    _ensure_table(conn)
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?,?,?,?)",
        ("T0", _D_OLD, 0.01, 4),
    )
    conn.execute(
        f"INSERT OR REPLACE INTO {TABLE_NAME} VALUES (?,?,?,?)",
        ("T0", _D_OLD, 0.05, 4),  # re-run with a different value
    )
    conn.commit()
    rows = conn.execute(f"SELECT * FROM {TABLE_NAME}").fetchall()
    assert len(rows) == 1, rows  # PK collision replaced, not duplicated
    assert rows[0][2] == 0.05


# ── S3 artifact seam: materialize_from_s3 / _export_artifact ────────────────
#
# The producer (PredictorBacktest box) and consumer (Evaluator box) each pull
# their OWN throwaway research.db copy from S3 and nothing pushes it back —
# the parquet at ARTIFACT_KEY is the only wire between them. These tests
# exercise both directions of that seam hermetically via the fake s3 client
# above (mirrors tests/test_reporter.py's injected s3_client idiom).


def test_materialize_from_s3_returns_zero_when_artifact_absent(tmp_path):
    from analysis.scanner_predictor_research_free_backfill import materialize_from_s3

    conn = sqlite3.connect(str(tmp_path / "m.db"))
    n = materialize_from_s3(conn, "any-bucket", s3_client=_FakeS3(tmp_path))
    assert n == 0
    # honest empty state: table exists (or is creatable) with zero rows
    _ensure_table(conn)
    assert conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 0


def test_export_then_materialize_roundtrip(tmp_path):
    """Producer-side export -> consumer-side materialize must reproduce the
    exact table contents on a second, independent connection (the two-box
    seam in miniature)."""
    from analysis.scanner_predictor_research_free_backfill import (
        _export_artifact,
        materialize_from_s3,
    )

    s3 = _FakeS3(tmp_path)
    producer = sqlite3.connect(str(tmp_path / "producer.db"))
    _ensure_table(producer)
    rows = [("T0", _D_OLD, 0.013, 4), ("T1", _D_RECENT, -0.021, 4)]
    producer.executemany(f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)", rows)
    producer.commit()

    key = _export_artifact(producer, "any-bucket", s3_client=s3)
    assert key == ARTIFACT_KEY
    assert [c[2] for c in s3.upload_calls] == [ARTIFACT_KEY]

    consumer = sqlite3.connect(str(tmp_path / "consumer.db"))
    n = materialize_from_s3(consumer, "any-bucket", s3_client=s3)
    assert n == 2
    got = sorted(consumer.execute(f"SELECT * FROM {TABLE_NAME}").fetchall())
    assert got == sorted(rows)

    # re-materializing is idempotent (INSERT OR REPLACE on the PK)
    n2 = materialize_from_s3(consumer, "any-bucket", s3_client=s3)
    assert n2 == 2
    assert consumer.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}").fetchone()[0] == 2


def test_materialize_seeds_pending_universe_idempotency(tmp_path):
    """run_backfill's idempotency depends on seeding the fresh local pull from
    the artifact — after materializing, _pending_universe must exclude the
    already-computed keys."""
    from analysis.scanner_predictor_research_free_backfill import (
        _export_artifact,
        materialize_from_s3,
    )

    s3 = _seeded_s3(tmp_path)
    prior = sqlite3.connect(str(tmp_path / "prior.db"))
    _ensure_table(prior)
    prior.executemany(
        f"INSERT INTO {TABLE_NAME} VALUES (?,?,?,?)",
        [("T0", _D_OLD, 0.01, 4), ("T1", _D_OLD, 0.02, 4)],
    )
    prior.commit()
    _export_artifact(prior, "any-bucket", s3_client=s3)

    fresh = sqlite3.connect(str(tmp_path / "fresh.db"))  # a brand-new pull: no backfill table at all
    materialize_from_s3(fresh, "any-bucket", s3_client=s3)
    pending = _pending_universe(fresh, bucket="any-bucket", s3_client=s3)
    pairs = set(zip(pending["ticker"], pending["eval_date"]))
    assert ("T0", _D_OLD) not in pairs
    assert ("T1", _D_OLD) not in pairs
    assert len(pending) == 4  # 6 passing - 2 seeded


def test_materialize_raises_on_non_404_download_error(tmp_path):
    """A corrupt/unreachable artifact must raise, never silently demote the
    counterfactual back to 'skipped' (fail-loud doctrine)."""
    from analysis.scanner_predictor_research_free_backfill import materialize_from_s3

    class _Denied(_FakeS3):
        def download_file(self, bucket, key, dest):
            raise ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "no"}}, "GetObject"
            )

    conn = sqlite3.connect(str(tmp_path / "d.db"))
    try:
        materialize_from_s3(conn, "any-bucket", s3_client=_Denied(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "AccessDenied" in str(e)


def test_fixture_dates_are_inside_the_lookback_window():
    """The window is relative; the fixtures must stay inside it forever.

    Directly pins the 2026-08-11 breakage: a fixture date that ages past
    `_pending_universe(lookback_days=...)` makes unrelated assertions fail with
    a row-count mismatch that looks like a logic bug and is not one.
    """
    import inspect

    from analysis import scanner_predictor_research_free_backfill as M

    default = inspect.signature(M._pending_universe).parameters["lookback_days"].default
    assert default == _LOOKBACK_DAYS, (
        f"_pending_universe's lookback default moved to {default}d; update "
        f"_LOOKBACK_DAYS and re-check every fixture date against it"
    )
    today = date.today()
    for label, value in (("_D_OLDEST", _D_OLDEST), ("_D_OLD", _D_OLD), ("_D_RECENT", _D_RECENT)):
        age = (today - date.fromisoformat(value)).days
        assert 0 < age < _LOOKBACK_DAYS, (
            f"{label} is {age}d old, outside the {_LOOKBACK_DAYS}d window "
            f"_pending_universe filters on"
        )


# ── The predictor-cut union (alpha-engine-config-I8755) ──────────────────────
#
# The entry-selection slot runs two arms over ONE parquet: the champion selects
# from the scanner-passing pool, the challenger from the cut the predictor
# resolves from. Both read the same `predicted_alpha` per (ticker, date), so
# the pool is the only treatment — which only holds if every member of BOTH
# pools has a row.
#
# Every test below is RED without the union: the pre-change `_pending_universe`
# returned scanner-passing rows only, so the cut-only names are simply absent
# and `.attrs` carries nothing (champion-challenger-policy.md §7.4 — a guard
# that cannot fail is worse than no guard).


def _membership(cut_name: str, tickers: list[str], run_date: str) -> dict:
    return {
        "run_date": run_date,
        "predictor_universe_cut": cut_name,
        "cuts": {
            cut_name: {"tickers": tickers},
            "some_other_cut": {"tickers": ["ZZ0", "ZZ1"]},
        },
    }


def test_pending_universe_unions_the_predictor_cut(tmp_path):
    """Cut members absent from the scanner-passing set still get a work row."""
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _seeded_s3(tmp_path, dates=(_D_RECENT,))  # passing T0/T1/T2
    # C0/C1 are cut-only; T0 overlaps the scanner pool.
    s3.put_membership(_D_RECENT, _membership("attractiveness_top_20", ["T0", "C0", "C1"], _D_RECENT))

    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)

    assert set(pending["ticker"]) == {"T0", "T1", "T2", "C0", "C1"}, pending
    # The overlap is not duplicated.
    assert len(pending) == 5, pending
    assert set(pending["eval_date"]) == {_D_RECENT}


def test_pending_universe_does_not_union_a_cut_the_artifact_does_not_name(tmp_path):
    """The cut NAME comes from the artifact, never from a literal here.

    `predictor_universe_cut` points at `some_other_cut`, so THAT is the pool —
    a hardcoded "attractiveness_top_20" would pick the wrong one and go stale
    the moment crucible-research moves the pointer (policy §7.5).
    """
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _seeded_s3(tmp_path, dates=(_D_RECENT,))
    doc = _membership("attractiveness_top_20", ["C0", "C1"], _D_RECENT)
    doc["predictor_universe_cut"] = "some_other_cut"  # -> ZZ0/ZZ1
    s3.put_membership(_D_RECENT, doc)

    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)

    assert set(pending["ticker"]) == {"T0", "T1", "T2", "ZZ0", "ZZ1"}, pending
    assert "C0" not in set(pending["ticker"])


def test_pending_universe_reports_coverage_per_cohort_date(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _seeded_s3(tmp_path, dates=(_D_RECENT,))
    s3.put_membership(_D_RECENT, _membership("attractiveness_top_20", ["T0", "C0", "C1"], _D_RECENT))

    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)
    coverage = pending.attrs["predictor_cut_coverage"]

    assert len(coverage) == 1
    row = coverage[0]
    assert row["run_date"] == _D_RECENT
    assert row["cut_name"] == "attractiveness_top_20"
    assert row["resolved"] is True
    assert row["n_members"] == 3
    assert row["n_already_scanner_passing"] == 1   # T0
    assert row["n_added_beyond_scanner_pool"] == 2  # C0, C1


def test_missing_membership_degrades_to_the_scanner_pool_and_says_so(tmp_path):
    """Fail-soft, never silent.

    The union only WIDENS the work list, so an unreadable membership must yield
    a narrower-but-valid backfill rather than no backfill. But a short pool is
    indistinguishable from an arm that selected badly, so it is REPORTED.
    """
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    _ensure_table(conn)
    s3 = _seeded_s3(tmp_path, dates=(_D_RECENT,))  # no membership written

    pending = _pending_universe(conn, bucket="any-bucket", s3_client=s3)

    assert set(pending["ticker"]) == {"T0", "T1", "T2"}
    row = pending.attrs["predictor_cut_coverage"][0]
    assert row["resolved"] is False
    assert row["n_members"] == 0
    assert "unreadable" in row["reason"]


def test_predictor_cut_members_reports_an_empty_named_cut(tmp_path):
    """An artifact naming a cut it does not contain is reported, not silently
    treated as 'no members'."""
    s3 = _FakeS3(tmp_path)
    s3.put_membership(
        _D_RECENT,
        {
            "run_date": _D_RECENT,
            "predictor_universe_cut": "attractiveness_top_20",
            "cuts": {"attractiveness_top_20": {"tickers": []}},
        },
    )

    tickers, coverage = _predictor_cut_members("any-bucket", _D_RECENT, s3_client=s3)

    assert tickers == set()
    assert coverage["resolved"] is False
    assert coverage["cut_name"] == "attractiveness_top_20"
    assert "empty or absent" in coverage["reason"]


def test_predictor_cut_members_uppercases_and_drops_blanks(tmp_path):
    s3 = _FakeS3(tmp_path)
    s3.put_membership(
        _D_RECENT,
        {
            "run_date": _D_RECENT,
            "predictor_universe_cut": "c",
            "cuts": {"c": {"tickers": ["aapl", "MSFT", "", None]}},
        },
    )

    tickers, coverage = _predictor_cut_members("any-bucket", _D_RECENT, s3_client=s3)

    assert tickers == {"AAPL", "MSFT"}
    assert coverage["resolved"] is True
    assert coverage["n_members"] == 2
