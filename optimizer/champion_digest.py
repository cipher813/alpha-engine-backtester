"""champion_digest.py — deliver the weekly RESEARCH champion/challenger verdict.

The selection-producer slot (``scanner_top20_predictor`` /
``scanner_predictor_direct`` / ``thinktank_coverage``) has run a weekly
winner-take-all evaluation since 2026-07-13 (Brian's ruling, epic
alpha-engine-config-I2364; engine: ``optimizer/champion_promotion.py``). Every
one of those verdicts was written to S3 and to **nothing else**: no email, no
SNS, no Telegram. The only surface carrying it is the pull-only console
Ablations page. Eleven verdicts, zero deliveries — a component emitting nothing
is unobserved, not healthy (``principles.md`` §2.7 Measurability;
``champion-challenger-policy.md`` §7.2, "an unmeasurable result must fail LOUD,
not render as an empty success").

This module closes that gap by mirroring the pattern the **model-zoo rotation**
already proves in production — an SSM-run stage builds a subject plus an
HTML/markdown body, sends it through the fleet's single SES/SMTP chokepoint
``krepis.email_sender.send_email``, and deep-links to a console page for the
detail (``Alpha Engine | Model-Zoo Rotation 2026-08-28 | 4 challengers |
promoted: none``). Mirroring rather than lifting is deliberate per
``shared-code-policy.md``: this is the SECOND adoption of the digest SHAPE, but
the shape is already a library call (``send_email``) and only the
slot-specific body building is duplicated. A third slot adopting it is the
trigger to lift a generic ``verdict_digest`` builder into ``nousergon-lib``.

**Non-promotion is the normal case and it is still news.** Nine of the eleven
verdicts to date are ``no_contest``. The digest fires on EVERY outcome —
promoted, no_contest, unchanged_winner_already_champion, held_shadow_only,
error — because a loop that only writes when it promotes is indistinguishable
from a loop that is dead. The model-zoo digest makes the same choice: it says
``promoted: none`` out loud rather than staying quiet.

**Fail-loud on silence.** ``send_email`` is fire-and-forget by contract: it
returns ``False`` and never raises. A ``False`` here means the verdict reached
nobody, which is the exact defect this module exists to retire, so it is
escalated as an ops alert rather than swallowed (``AGENTS.md``: no silent
degrade on a producer). The send itself is still non-fatal to the weekly
evaluate run — a notification must never red the pipeline it reports on.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Cross-repo contract with crucible-dashboard's ``app.py`` (``url_path``),
# guarded there by ``tests/test_experiments_page.py``. Kept caller-side for the
# same reason ``emailer.ANALYSIS_SLUG`` is: the slug is this email's contract
# with that page, not a krepis concern.
EXPERIMENTS_SLUG = "experiments"

# One send per cycle. The Saturday SF's Backtester state is the only caller
# today, but ``evaluate.py`` is re-invocable by hand and by the watch-rerun
# path, and three near-identical digests is how the backtester digest got its
# own dedup key (config#2291).
DEDUP_KEY_PREFIX = "research-champion-verdict"
DEDUP_WINDOW_MIN = 1440  # 24h — covers same-day reruns, never the next cycle.

# Human-readable subject fragment per gate outcome. Deliberately mirrors the
# model-zoo digest's ``promoted: none`` phrasing so the two weekly champion
# emails read as one family in an inbox.
_OUTCOME_LABEL = {
    "promoted": "promoted: {champion_after}",
    "unchanged_winner_already_champion": "promoted: none (incumbent defended)",
    "no_contest": "promoted: none (no contest)",
    "held_shadow_only": "promoted: none (shadow-only arm won)",
    "error": "promoted: none (GATE ERROR)",
}


def experiments_url(run_date: str, console_base_url: str | None = None) -> str:
    """Deep-link to the console Ablations page for ``run_date``.

    Thin wrapper over the ``krepis.console.console_url`` chokepoint, mirroring
    ``emailer.analysis_report_url``.
    """
    from krepis.console import console_url

    return console_url(EXPERIMENTS_SLUG, date=run_date, base=console_base_url)


def _fmt_score(value: Any) -> str:
    """A score cell. ``None`` renders as an explicit em dash, never as ``0`` —
    "the arm produced no comparable evidence" and "the arm scored zero" are
    different facts and must never render identically
    (``champion-challenger-policy.md`` §3)."""
    if value is None:
        return "—"
    try:
        return f"{float(value):+.5f}"
    except (TypeError, ValueError):
        return str(value)


def build_subject(audit: dict) -> str:
    """``Alpha Engine | Research Champion/Challenger 2026-08-28 | 3 arms | promoted: none (no contest)``."""
    outcome = audit.get("outcome") or "unknown"
    arm_scores = audit.get("arm_scores") or {}
    arm_confidence = audit.get("arm_confidence") or {}
    # ``arm_scores`` is the N-arm view; fall back to ``arm_confidence`` (which
    # predates it) so a record written by an older engine still gets a count.
    n_arms = len(arm_scores) or len(arm_confidence)
    label = _OUTCOME_LABEL.get(outcome, f"outcome: {outcome}").format(
        champion_after=audit.get("champion_after") or "?",
    )
    arms = f"{n_arms} arm{'' if n_arms == 1 else 's'}"
    return (
        f"Alpha Engine | Research Champion/Challenger {audit.get('date')} "
        f"| {arms} | {label}"
    )


def build_body_md(audit: dict, *, console_base_url: str | None = None) -> str:
    """The verdict as markdown: what the gate decided, on what evidence, and
    what is blocking it — enough that a reader never has to open S3.

    Every arm registered in the slot gets a row, including the ones that scored
    nothing: an arm silently missing from the table is exactly the
    "well-formed artifact containing nothing" class §7.2 names.
    """
    outcome = audit.get("outcome") or "unknown"
    date = audit.get("date")
    before = audit.get("champion_before")
    after = audit.get("champion_after")
    arm_scores = audit.get("arm_scores") or {}
    arm_confidence = audit.get("arm_confidence") or {}
    blocked_by = audit.get("blocked_by") or []
    counterfactual = audit.get("counterfactual_winner")

    moved = bool(after) and after != before
    lines = [
        f"## Research champion/challenger — {date}",
        "",
        f"**Outcome:** `{outcome}`",
        (
            f"**Live pointer:** `{before}` → `{after}` (MOVED)"
            if moved
            else f"**Live pointer:** `{before}` (unchanged)"
        ),
    ]
    if counterfactual and counterfactual != after:
        lines.append(
            f"**Would have won on score:** `{counterfactual}` — the pointer was "
            f"held anyway; see blockers below."
        )
    if audit.get("freeze"):
        lines.append("**Freeze flag was set** — a promotion this week would have been suppressed.")
    if audit.get("leaderboard_date_used"):
        lines.append(f"**Evidence vintage:** `{audit['leaderboard_date_used']}`")
    if audit.get("detail"):
        lines.append(f"**Detail:** {audit['detail']}")

    lines += ["", "### Arms", "", "| arm | role | score | confidence |", "|---|---|---|---|"]
    # Union so an arm present in only one of the two maps is still shown.
    names = list(arm_scores) + [a for a in arm_confidence if a not in arm_scores]
    if not names:
        lines.append("| _no arm-level detail in this record_ | — | — | — |")
    for name in names:
        role = "champion" if name == before else "challenger"
        if name == after and moved:
            role = "champion (promoted)"
        lines.append(
            f"| `{name}` | {role} | {_fmt_score(arm_scores.get(name))} "
            f"| {arm_confidence.get(name, '—')} |"
        )

    lines += ["", "### Blocked by", ""]
    if blocked_by:
        lines += [f"- `{slug}`" for slug in blocked_by]
    else:
        lines.append("- _nothing — the gate reached a decision on the evidence._")

    lines += [
        "",
        f"Full history: `s3://alpha-engine-research/config/apply_audit/producer_champion/{date}.json`",
        f"Live pointer: `s3://alpha-engine-research/config/producer_champion.json`",
        "",
        f"[View the champion/challenger ledgers on the console]({experiments_url(str(date), console_base_url)})",
    ]
    return "\n".join(lines)


def send_verdict_digest(
    audit: dict,
    *,
    console_base_url: str | None = None,
    send_email_fn: Any = None,
    alert_fn: Any = None,
) -> bool:
    """Deliver ``audit`` as the weekly verdict email. Returns whether it landed.

    Fires on EVERY outcome — see the module docstring. Returns ``False`` when
    the send did not land, having first escalated an ops alert, because an
    undelivered verdict is the defect this module exists to retire and must not
    become a second silence layered on the first.

    ``send_email_fn`` / ``alert_fn`` are seams for tests only; production
    resolves ``krepis.email_sender.send_email`` (the fleet's single SES/SMTP
    chokepoint) and ``ops_alerts.publish_ops_alert`` lazily so importing this
    module never requires either.
    """
    date = audit.get("date")
    subject = build_subject(audit)
    body = build_body_md(audit, console_base_url=console_base_url)

    if send_email_fn is None:
        from krepis.email_sender import send_email as send_email_fn  # noqa: PLC0415

    sent = False
    failure = None
    try:
        sent = bool(
            send_email_fn(
                subject,
                body,
                dedup_key=f"{DEDUP_KEY_PREFIX}:{date}",
                dedup_window_min=DEDUP_WINDOW_MIN,
            )
        )
    except Exception as exc:  # noqa: BLE001
        # send_email's contract is "never raises", but a contract is not a
        # guarantee and a raise here would take down a weekly evaluate run over
        # a notification. Recorded, escalated below, never swallowed silently.
        logger.exception("[champion_digest] verdict email raised")
        failure = repr(exc)

    if sent:
        logger.info("[champion_digest] verdict email sent for %s: %s", date, subject)
        return True

    reason = failure or "send_email returned False (missing config, auth, or network)"
    logger.error(
        "[champion_digest] verdict email for %s did NOT land (%s) — the weekly "
        "research champion/challenger verdict reached no operator surface",
        date, reason,
    )
    _escalate_undelivered(audit, reason, alert_fn=alert_fn)
    return False


def _escalate_undelivered(audit: dict, reason: str, *, alert_fn: Any = None) -> None:
    """An undelivered verdict is an ops alert, not a log line.

    Mirrors ``champion_promotion._publish_gate_error_alert``. Best-effort by
    necessity — this is already the failure path of the notification layer —
    but it never passes silently: an alerting failure is logged with its
    traceback.
    """
    if alert_fn is None:
        try:
            from ops_alerts import publish_ops_alert as alert_fn  # noqa: PLC0415
        except ImportError as exc:
            logger.error(
                "[champion_digest] undelivered-verdict alert skipped — "
                "ops_alerts unavailable: %s", exc,
            )
            return
    date = audit.get("date")
    try:
        alert_fn(
            f"Research champion/challenger verdict for {date} was computed "
            f"(outcome={audit.get('outcome')}, pointer={audit.get('champion_after')}) "
            f"but the digest email did not land: {reason}. The verdict exists only "
            f"in s3://alpha-engine-research/config/apply_audit/producer_champion/"
            f"{date}.json and on the console.",
            severity="error",
            source="alpha-engine-backtester/optimizer/champion_digest.py::send_verdict_digest",
            dedup_key=f"champion_verdict_undelivered_{date}",
            dedup_window_min=720,
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the run
        logger.exception(
            "[champion_digest] undelivered-verdict alert publish ALSO failed",
        )
