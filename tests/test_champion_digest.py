"""Regression tests for the weekly research champion/challenger verdict digest.

Every test here fails against the pre-fix tree, where
``optimizer/champion_digest.py`` did not exist and eleven consecutive weekly
verdicts (2026-07-13 → 2026-08-28) reached no operator surface at all.

The fixture record is the REAL 2026-08-28 audit artifact read from
``s3://alpha-engine-research/config/apply_audit/producer_champion/2026-08-28.json``
on 2026-08-29, with ``arm_scores`` added — see
``test_arm_scores_reaches_the_durable_audit_record`` in
``test_champion_promotion.py`` for why the live artifact lacks that field.
"""

from __future__ import annotations

import pytest

from optimizer import champion_digest


# The live 2026-08-28 verdict: no_contest, pointer held, BOTH challengers
# unscored on thin evidence, so ``challenger`` is null and the pair-shaped
# champion_score/challenger_score fields name almost nothing.
AUDIT_2026_08_28 = {
    "schema_version": 2,
    "date": "2026-08-28",
    "generated_at": "2026-08-29T13:59:36.979123+00:00",
    "outcome": "no_contest",
    "champion_before": "scanner_predictor_direct",
    "champion_after": "scanner_predictor_direct",
    "champion_score": 0.00257,
    "challenger_score": None,
    "blocked_by": [
        "scanner_top20_predictor_thin_evidence",
        "thinktank_coverage_thin_evidence",
    ],
    "challenger": None,
    "freeze": False,
    "leaderboard_date_used": "2026-08-28",
    "feed_dependencies": ["research_free_backfill"],
    "counterfactual_winner": None,
    "arm_confidence": {
        "scanner_predictor_direct": "ok",
        "scanner_top20_predictor": "thin",
        "thinktank_coverage": "thin",
    },
    "arm_scores": {
        "scanner_top20_predictor": None,
        "scanner_predictor_direct": 0.00257,
        "thinktank_coverage": None,
    },
}


class _Recorder:
    """Stand-in for ``krepis.email_sender.send_email``."""

    def __init__(self, result=True):
        self.result = result
        self.calls: list[tuple] = []

    def __call__(self, subject, body, **kwargs):
        self.calls.append((subject, body, kwargs))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_a_no_contest_week_is_still_delivered():
    """The dominant outcome — nine of eleven verdicts to date — must send.

    A digest that only fires on a promotion is indistinguishable from a loop
    that stopped running, which is exactly the state this slot was in.
    """
    send = _Recorder()
    assert champion_digest.send_verdict_digest(
        AUDIT_2026_08_28, send_email_fn=send, alert_fn=lambda *a, **k: None,
    ) is True
    assert len(send.calls) == 1


def test_subject_names_the_cycle_the_arm_count_and_that_nothing_promoted():
    subject = champion_digest.build_subject(AUDIT_2026_08_28)
    assert "2026-08-28" in subject
    assert "3 arms" in subject
    assert "promoted: none" in subject


def test_subject_names_the_arm_on_an_actual_promotion():
    audit = dict(AUDIT_2026_08_28)
    audit["outcome"] = "promoted"
    audit["champion_after"] = "thinktank_coverage"
    assert "promoted: thinktank_coverage" in champion_digest.build_subject(audit)


def test_body_lists_every_arm_including_the_unscored_ones():
    """champion-challenger-policy.md §3 — an arm that produced nothing is
    recorded as a miss, never omitted. The 2026-08-28 record's pair-shaped
    fields name only ``scanner_predictor_direct``; both other arms exist only
    in the N-arm maps, and dropping them from the digest would reproduce the
    exact collapse the artifact fix exists to undo."""
    body = champion_digest.build_body_md(AUDIT_2026_08_28)
    for arm in ("scanner_top20_predictor", "scanner_predictor_direct", "thinktank_coverage"):
        assert arm in body, f"{arm} missing from the digest body"


def test_an_unscored_arm_never_renders_as_a_zero():
    """§3: silent absence and a genuine zero must not render identically."""
    body = champion_digest.build_body_md(AUDIT_2026_08_28)
    row = next(line for line in body.splitlines() if "`thinktank_coverage`" in line)
    assert "—" in row
    assert "0.00000" not in row


def test_body_names_every_blocker_and_the_unchanged_pointer():
    body = champion_digest.build_body_md(AUDIT_2026_08_28)
    assert "thinktank_coverage_thin_evidence" in body
    assert "scanner_top20_predictor_thin_evidence" in body
    assert "unchanged" in body


def test_body_reports_a_shadow_only_hold_as_a_win_that_did_not_move_the_pointer():
    """``held_shadow_only`` is the one outcome where the winner is NOT the
    pointer's destination. A digest that printed only ``champion_after`` would
    silently report a challenger win as a defended incumbency."""
    audit = dict(AUDIT_2026_08_28)
    audit["outcome"] = "held_shadow_only"
    audit["counterfactual_winner"] = "thinktank_coverage"
    audit["blocked_by"] = ["shadow_only_arm"]
    body = champion_digest.build_body_md(audit)
    assert "Would have won on score" in body
    assert "thinktank_coverage" in body


def test_body_deep_links_the_console_at_the_pinned_slug():
    body = champion_digest.build_body_md(
        AUDIT_2026_08_28, console_base_url="https://console.example",
    )
    assert "https://console.example/experiments?date=2026-08-28" in body


def test_the_send_is_deduped_per_cycle():
    send = _Recorder()
    champion_digest.send_verdict_digest(
        AUDIT_2026_08_28, send_email_fn=send, alert_fn=lambda *a, **k: None,
    )
    kwargs = send.calls[0][2]
    assert kwargs["dedup_key"] == "research-champion-verdict:2026-08-28"


def test_a_send_that_does_not_land_escalates_rather_than_passing_silently():
    """``send_email`` returns False and never raises. Treating that as success
    would layer a second silence on the one this module exists to retire."""
    alerts: list[dict] = []
    sent = champion_digest.send_verdict_digest(
        AUDIT_2026_08_28,
        send_email_fn=_Recorder(result=False),
        alert_fn=lambda msg, **kw: alerts.append({"msg": msg, **kw}),
    )
    assert sent is False
    assert len(alerts) == 1
    assert alerts[0]["severity"] == "error"
    assert "2026-08-28" in alerts[0]["msg"]


def test_a_raising_sender_escalates_and_does_not_propagate():
    """A notification must never take down the weekly evaluate run."""
    alerts: list[dict] = []
    sent = champion_digest.send_verdict_digest(
        AUDIT_2026_08_28,
        send_email_fn=_Recorder(result=RuntimeError("SMTP down")),
        alert_fn=lambda msg, **kw: alerts.append({"msg": msg, **kw}),
    )
    assert sent is False
    assert len(alerts) == 1


def test_an_error_outcome_is_delivered_too():
    """The gate failing is the single most important week to be told about."""
    send = _Recorder()
    error_audit = {
        "schema_version": 2,
        "date": "2026-09-04",
        "outcome": "error",
        "champion_before": "scanner_predictor_direct",
        "champion_after": None,
        "champion_score": None,
        "challenger_score": None,
        "blocked_by": ["unclassified_error"],
        "freeze": False,
        "detail": "gate evaluation raised",
        "arm_confidence": None,
        "arm_scores": None,
    }
    assert champion_digest.send_verdict_digest(
        error_audit, send_email_fn=send, alert_fn=lambda *a, **k: None,
    ) is True
    assert "GATE ERROR" in send.calls[0][0]
    assert "gate evaluation raised" in send.calls[0][1]


@pytest.mark.parametrize(
    "outcome",
    ["promoted", "no_contest", "unchanged_winner_already_champion",
     "held_shadow_only", "error"],
)
def test_every_declared_gate_outcome_has_a_subject_label(outcome):
    """A new outcome slug added to ``champion_promotion.OUTCOMES`` without a
    label here would ship a subject reading ``outcome: <slug>``. That is
    legible, but this test makes the omission visible at the time it is made."""
    from optimizer import champion_promotion

    assert outcome in champion_promotion.OUTCOMES
    assert outcome in champion_digest._OUTCOME_LABEL
