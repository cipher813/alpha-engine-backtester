"""champion_promotion.py — weekly winner-take-all champion/challenger gate
(config#2364 / config#2367 origin; redesigned alpha-engine-config-I2518 /
epic I2515, 2026-07-14 ruling; scoring redesigned to direct per-arm lift,
no shared comparator, alpha-engine-config-I2998, 2026-07-20 ruling).

Writes the live pointer ``config/producer_champion.json`` that the
alpha-engine executor's ``executor/champion.py::load_champion_pointer``
reads at planner start to decide which entry-candidate producer arm is
LIVE. Statistical correctness matters here in a way it does not for the
advisory optimizers elsewhere in this module: a wrong pointer move silently
changes which strategy trades real (paper) capital.

**2026-07-14 seat swap (Brian's ruling, config-I2518, binding on this
issue):** the ``agentic`` seat retires with the multi-agent Research graph
(epic config-I2515) and is replaced by ``thinktank_coverage`` — the Think
Tank challenger arm (scanner top-~60 -> Think Tank full-coverage -> its
own top-~20 by independent TT rating). ``scanner_predictor_direct`` is the
new BASE-CASE champion (already live since 2026-07-13T22:07 UTC,
config-I2364)::

    VALID_CHAMPIONS = ("scanner_predictor_direct", "thinktank_coverage")

``agentic`` is READ-TOLERATED (a historical pointer/audit value must never
crash this engine — ``_normalize_champion_before`` below WARNs and treats
it as ``scanner_predictor_direct``) but WRITE-FORBIDDEN
(``write_champion_pointer`` raises on any value outside ``VALID_CHAMPIONS``,
which no longer includes it). No real ``config/producer_champion.json``
object was ever actually written with ``champion="agentic"`` — it was only
ever an implicit pre-bootstrap default — but the 2026-07-13 bootstrap DID
write a real audit record with ``champion_before="agentic"``
(``config/apply_audit/producer_champion/2026-07-13.json``), so the
read-tolerance is not purely defensive.

**Weekly winner-take-all policy (supersedes the entire HAC-significance /
2-week-hysteresis / 2-week-cooldown gated engine this module shipped with
under config#2367 — INCLUDING the standing ``cooldown_until:
2026-07-27`` carried in the prior ``latest.json``, which this policy no
longer reads or honors):**

    Each weekly evaluation compares the two arms' realized top-N alpha lift
    for the trailing week and flips the pointer to whichever arm scores
    higher, if that arm is not already champion. No significance test, no
    consecutive-week hysteresis, no cooldown — "whichever performs best in
    a given week is promoted at that time" (Brian's ruling, verbatim).

**SHADOW-ONLY ARMS (Brian's ruling, 2026-08-20, recorded on
alpha-engine-config-I2515) — NARROWLY supersedes the 2026-07-14 ruling
above:**

    "research should now be think tank in shadow mode only with the main
    research process skipped by passing a scanner top 20 to predictor"
    (Brian, verbatim)

``thinktank_coverage`` is MEASURED, not promotable. It keeps scoring every
week, keeps its leaderboard row, and this gate keeps recording who WOULD
have won — but it may never take the live pointer by winning. Promoting it
requires its own separate ruling, which is a ONE-LINE data change here
(remove it from ``SHADOW_ONLY_ARMS`` below) plus that ruling.

This supersedes 2026-07-14 on exactly ONE question — whether a shadow-only
arm may take the live pointer. Everything else the 2026-07-14 ruling
established is unchanged: both arms are scored weekly on the same yardstick
(champion-challenger-policy.md §3, measurement is unconditional), the
leaderboard is unchanged, ties still favour the incumbent, and a NON-shadow
challenger that outscores the champion still promotes immediately with no
significance test, hysteresis or cooldown.

Shadow-only-ness is a PROPERTY OF AN ARM, declared once in
``SHADOW_ONLY_ARMS`` — never a hard-coded string at the veto site — so a
future arm added in shadow mode inherits the protection automatically. It
is enforced at TWO layers: ``evaluate_gates`` (the POLICY — degrades a
shadow-only arm's win to ``outcome="held_shadow_only"`` with
``blocked_by=["shadow_only_arm"]``, never reaching the writer) and
``write_champion_pointer`` (the INVARIANT — raises, so a future caller that
bypasses the gate entirely still cannot flip the pointer onto a shadow arm).

**Validity guards (definitional NO-CONTEST, not a statistical gate) —
``evaluate_gates`` below:** a week where either arm's realized-lift score
is unavailable (no valid ``thinktank_coverage`` selections this week, no
resolved/matured outcomes yet, the evidence artifact itself missing or
stale) is a NO-CONTEST: the pointer is left unchanged and the outcome
record says so explicitly via a machine-readable ``blocked_by`` slug. A
no-contest NEVER defaults a win to either side.

**Evidence sourcing — DIRECT per-arm realized lift, NO shared comparator
(alpha-engine-config-I2998, 2026-07-20 ruling — supersedes the
Bucher-style indirect/common-comparator design below this module shipped
with under I2518):**

  The pre-I2998 design scored both arms as "lift vs the live
  ``agentic_sector_teams``/CIO-ADVANCE baseline" on the premise that
  Research kept running its full agentic pipeline weekly regardless of
  which arm the executor traded. config-I2993 (2026-07-19/20) found that
  premise false: ``agentic_sector_teams`` retired 2026-07-12 with no
  successor ``kind=="champion"`` producer registered, so BOTH arms'
  "vs agentic" scores could go simultaneously no-contest — a materially
  worse failure than either arm alone going stale, since a no-contest week
  is a legitimate, non-alerting outcome by design (freezing
  ``config/producer_champion.json`` silently). I2998's fix removes the
  shared-comparator dependency entirely: each arm now scores its OWN
  realized lift against a FIXED, always-available neutral baseline, so
  neither arm's score can ever depend on whether Research's agentic
  pipeline (or any future comparator) happens to be live that week.

  - ``scanner_predictor_direct``'s weekly score is this run's
    ``analysis.end_to_end.compute_lift_metrics()['scanner_then_predictor_counterfactual']
    ['methods']['scanner_then_predictor_topN']['sector_neutral_mean_alpha_21d']``
    — the arm's own realized, sector-neutral 21d alpha, ALREADY benchmark
    -relative (realized log return minus the log SPY return over the same
    window, at the source, ``analysis/end_to_end.py::_scanner_then_predictor_topN``) —
    i.e. lift vs the SPY zero-line, not vs any live comparator arm. A
    backtester-internal counterfactual (research.db-derived) already
    computed earlier in the same ``evaluate.py`` run, extracted via
    ``leaderboard_entry_from_e2e_lift`` (this module's OWN
    ``research/producer_leaderboard_champion_gate/{date}.json`` history
    artifact is STILL maintained for observability and to keep
    config#2452's in-flight live-verification intact — see
    ``update_leaderboard_and_get_gate_inputs`` — but its accumulated
    ``weekly_points`` series is no longer consumed by the gate itself,
    since winner-take-all needs only THIS week's point, not a multi-week
    HAC-adjusted series). The retired ``sn_lift_vs_agentic_cio`` field is
    still carried on the leaderboard-history entry for observability but
    is no longer the gate's score source.
  - ``thinktank_coverage``'s weekly score is read from crucible-research's
    real champion/challenger producer leaderboard,
    ``research/producer_leaderboard/{date}.json``
    (``scoring/leaderboard_producers.py::build_producer_leaderboard`` +
    ``scoring/leaderboard_scoring.py::score_leaderboard``, config#1221/
    #1223, made champion-optional under I2998) — verified schema
    (2026-07-20, read from the crucible-research checkout, NOT guessed):
    ``{"champion": <research producer champion name> | None,
    "horizon_days": 21, "top_n": 50, "benchmark_ticker": "SPY", "n_dates":
    int, "specs": [{"name", "kind", "realized_rank_ic",
    "topn_alpha_vs_champion": {...} | None,
    "topn_alpha_vs_benchmark": {"mean","se","t_stat","n_dates"} | None,
    "n_dates_scored", "confidence"}, ...],
    "horizons_days": [21, 126, 252], "min_dates_for_inference": 5,
    "horizons": [{"horizon_days", "status", "reason", "n_dates", "specs"},
    ...]}``. We read the ``specs`` row named
    ``"thinktank_coverage"`` and take its ``topn_alpha_vs_benchmark.mean``
    — the SAME kind of statistic as ``scanner_predictor_direct``'s score
    (a mean top-N realized return lift vs the SPY benchmark, date
    -clustered), so the two scores remain apples-to-apples comparable
    under winner-take-all's direct "higher wins" rule. This field is
    computed champion-free (``score_leaderboard`` degrades to
    champion-free metrics for every spec when no producer is registered
    ``kind=="champion"`` — see I2998) — unlike the retired
    ``topn_alpha_vs_champion``, it is available even while config-I2993's
    "no successor champion registered" state persists. ``coverage_complete``
    validity (the full current-scan top-60 rule, Brian's ruling
    config#1580) is enforced UPSTREAM at the artifact boundary —
    crucible-research PR427 writes ``signals_shadow/thinktank_coverage/
    {trading_day}/signals.json`` (the input this leaderboard scores) ONLY
    when ``coverage_complete`` — so any date this spec contributed to
    ``n_dates_scored`` was necessarily a full-coverage day; no separate
    coverage_complete check is needed on this side of the boundary.

    **LATEST-AVAILABLE read (alpha-engine-config-I2544, 2026-07-14 ruling,
    same-session follow-up to I2518):** ``research/producer_leaderboard/
    {date}.json`` is now written by an ASYNC advisory child Step Function
    (config-I2518's persistent-dash rearchitecture) that may not have
    finished — or may have failed outright — by the time this Evaluator
    -stage gate runs in the MAIN weekly SF. An exact same-day key read is
    therefore no longer a safe assumption. This module instead lists the
    ``research/producer_leaderboard/`` prefix
    (``find_latest_research_producer_leaderboard_date``) and reads the
    LATEST artifact dated <= ``run_date`` (``read_latest_research_producer
    _leaderboard``). This is the semantically CORRECT read, not a
    compromise: the gate scores REALIZED (matured) outcomes of PRIOR
    weeks' ``thinktank_coverage`` selections — a same-day leaderboard could
    not contain resolved outcomes for same-day picks even if it existed on
    time. An honest staleness bound still applies: a selected leaderboard
    more than ``LEADERBOARD_STALENESS_DAYS`` (8) calendar days older than
    ``run_date`` is treated as unavailable (``leaderboard_stale_gt_8d``,
    a no-contest) rather than silently scored against stale evidence. The
    date actually used is threaded through to the audit record as
    ``leaderboard_date_used`` (additive, nousergon_lib.contracts's
    producer_champion_audit schema) so the audit trail always shows which week's evidence
    decided (or declined to decide) a flip.

    **EVIDENCE-CONFIDENCE gate (alpha-engine-config-I7549, 2026-08-17):**
    until this change the gate accepted a leaderboard row on
    ``n_dates_scored`` being TRUTHY. ``n_dates_scored == 1`` is truthy, so
    a one-date mean — carrying ``se: null`` and ``t_stat: null`` because
    neither can be computed at n=1 — could move the live champion pointer.
    That was not hypothetical: ``research/producer_leaderboard/
    2026-08-14.json`` carried ``thinktank_coverage`` with
    ``n_dates_scored: 1``, ``topn_alpha_vs_benchmark.mean: -0.060751``,
    ``se: null``, ``t_stat: null``.

    crucible-research PR643 (alpha-engine-config-I7542) now stamps every
    spec row with an explicit ``confidence`` —
    ``insufficient`` (nothing scored) / ``thin`` (scored on fewer than the
    artifact's ``min_dates_for_inference`` dates) / ``ok`` — produced by
    ``scoring/leaderboard_scoring.py::confidence_for`` against the slot's
    registered evidence floor (``LEADERBOARD_SLOTS``). This module CONSUMES
    that field; it deliberately does not reimplement the thinness test, so
    the floor stays a single per-slot fact owned by the producer
    (champion-challenger-policy.md §10).

    Only a ``confidence == "ok"`` row is scored. A ``thin`` row yields
    ``thinktank_coverage_thin_evidence``; an ``insufficient`` row keeps the
    existing ``thinktank_coverage_no_resolved_outcomes`` — deliberately
    DIFFERENT slugs, because they call for different operator responses
    (wait for the cohort to mature vs go find out why nothing scored).

    This does NOT weaken §5's fast path. §5 permits promoting on
    DIRECTIONAL evidence without a full statistical gate because the
    platform is paper and the decision is reversible; it says nothing that
    makes n=1 evidence. A thin row is not a weak directional signal, it is
    a number whose own dispersion is undefined. Excluding it removes
    non-evidence from the fast path rather than adding a statistical gate
    to it: the promotion rule remains "strictly higher score wins", with
    no significance test, PSR/DSR bar, or cohort-count requirement.

    Nor does it weaken §5.2 hysteresis. This module's hysteresis under the
    I2518 winner-take-all ruling is "the pointer never moves on a null or
    an exactly-equal signal — ties favour the incumbent". Every path added
    here produces a ``None`` score, i.e. strictly MORE reasons to leave the
    pointer where it is, in BOTH directions (a thin champion row cannot
    demote the incumbent either — a no_contest holds the pointer). Nothing
    here can cause a flip that would not have happened before.

    **Backwards compatibility is fail-STATIC, never fail-``ok``.** A
    pre-I7542 artifact carries no ``confidence`` key. Treating its absence
    as ``ok`` would be the very defect being fixed, so this module derives
    the verdict from the artifact's OWN declared floor
    (``min_dates_for_inference``) against ``n_dates_scored``; when the
    artifact declares neither, the row is unavailable under its own slug
    (``thinktank_coverage_confidence_unknown``) rather than trusted. Same
    slug for a ``confidence`` value outside the known vocabulary — a
    producer that changed its vocabulary is a reason to stop, not to
    guess. The audit record's ``arm_confidence`` distinguishes the two
    (``unknown`` vs ``unrecognized``).

    **The CHAMPION side of the same comparison (alpha-engine-config-I7549,
    champion-side half).** Everything above gates the arm scored off the
    producer leaderboard. ``scanner_predictor_direct`` is scored off this
    repo's own ``scanner_then_predictor_topN`` counterfactual, which returns a
    result at ``n_cycles >= 1`` — so the identical one-observation defect
    survived on the other input to the same gate. A two-arm gate is not fixed
    while either arm can be a single draw: the quantity the gate acts on is the
    DIFFERENCE, and a thin champion makes that difference noise exactly as a
    thin challenger does, while the audit record reads identically either way.
    ``MIN_CYCLES_FOR_INFERENCE`` (5 cycles, this repo's registry because this
    repo owns that arm's measurement — §10) is that arm's floor, reported in
    the same ``ok``/``thin``/``insufficient``/``unknown`` vocabulary and
    blocking under its own per-arm slugs
    (``scanner_predictor_direct_thin_evidence`` /
    ``scanner_predictor_direct_confidence_unknown``), because the operator
    response differs by arm: one waits on crucible-research's cohort, the other
    on this repo's own research.db cycles. Measured 2026-08-17: live runs carry
    ``n_cycles: 15``, so the floor bounds the degenerate case without binding
    on today's evidence.

    **HORIZON: the gate reads the artifact's PRIMARY (21-session) block,
    and now says so (alpha-engine-config-I7540/I7549).** The leaderboard
    now scores every arm at 21, 126 and 252 sessions and emits one block
    per horizon under ``horizons``; the PRIMARY horizon's block is also
    spread across the artifact's top level, which is what ``specs`` here
    is. The gate stays on 21 for two binding reasons, not by inertia:
    (1) §4 requires the same horizon across every arm in a slot, and the
    other arm — ``scanner_predictor_direct`` — is scored from this repo's
    own ``sector_neutral_mean_alpha_21d`` counterfactual, a 21-session
    statistic with no 126/252 equivalent; ranking a 252-session
    thinktank_coverage number against a 21-session scanner number is not a
    comparison. (2) §3 requires a promoted arm's series to stay
    continuous; the live pointer's whole history is 21-session, and
    silently rebasing the gate onto a different horizon would reset it.
    Moving this gate to a longer horizon is a real and arguably correct
    future change — the scanner's stated objective is a ~1-year view — but
    it requires a 126/252-session score for BOTH arms first, and it is an
    explicit decision, not a side effect of the producer growing a field.
    ``GATE_HORIZON_DAYS`` names the choice, and a leaderboard whose
    top-level ``horizon_days`` disagrees with it is refused
    (``leaderboard_horizon_mismatch``) rather than scored — so an upstream
    change of primary horizon surfaces as a loud no-contest instead of
    silently rescoring the champion gate.

  **config-I2993 item 2 (windowing ``end_to_end.py``'s
  ``sn_lift_vs_agentic_cio`` aggregation) is NO LONGER a dependency for
  this gate's correctness** — that field is retired as this gate's score
  source under I2998 (still computed and carried for observability/other
  consumers, e.g. the evaluator tile, but this module reads
  ``sector_neutral_mean_alpha_21d`` instead). It may still be worth doing
  for the evaluator tile's own accuracy, independent of this gate.

  **REGISTRATION — CLOSED (was alpha-engine-config-I2519; corrected here
  2026-08-20):** this docstring previously carried I2519 as an open
  "KNOWN, TRACKED GAP" — ``thinktank_coverage`` not yet registered in
  crucible-research's ``producers/registry.py::RESEARCH_PRODUCERS`` /
  ``challenger_producers()``, so its leaderboard row could never exist and
  ``_score_thinktank_coverage`` would return
  ``blocked_by=["thinktank_coverage_not_in_leaderboard"]`` forever. That
  claim is STALE and has been removed rather than tolerated: the arm has
  been registered and scoring since 2026-08-14, whose evaluation was the
  first week with a real ``thinktank_coverage`` score
  (``scanner_predictor_direct`` -0.00203 vs ``thinktank_coverage``
  -0.060751, outcome ``unchanged_winner_already_champion``). The
  ``thinktank_coverage_not_in_leaderboard`` slug is RETAINED and still
  correct — it now means a genuine regression (the row disappeared), not
  an expected steady state, and is a NO-CONTEST either way.

``hac_significance`` (Newey-West/HAC overlap-aware significance) is
RETAINED below, unchanged and still independently unit-tested — it is no
longer wired into the promotion decision (winner-take-all has no
significance gate) but remains available as a possible future diagnostic
(e.g. reporting whether a winning margin looks like signal or noise
alongside the decision) without having to re-derive it.

**Single writer function, dual caller (no parallel writer implementations):**
``write_champion_pointer`` is the ONLY code path that may write
``config/producer_champion.json``. The gate engine calls it with
``promotion_source="gate_engine"``; the one-shot 2026-07-13 operator
bootstrap (``bootstrap_champion_promotion.py``) called it with
``promotion_source="operator_bootstrap"`` — never a hand-edited S3 object.

**Liveness (config#2054 lesson, binding):** ``config/producer_champion.json``
mtime alone cannot prove this engine is alive — a correctly-held
(no-contest) week does not touch it. ``run_weekly_evaluation`` writes
``config/apply_audit/producer_champion/{date}.json`` (+ ``latest.json``)
UNCONDITIONALLY, including on ``outcome="error"``.

**Promotion-time feed-dependency liveness (alpha-engine-config-I3165,
2026-07-23):** the config#3053 2026-07-20 no-trade-morning incident's root
cause was that ``scanner_predictor_direct``'s live-trade feed chain
(``research_free_backfill``) was never declared anywhere in the promotion
record, and nothing at promotion time checked its producer was alive —
config#1580 orphaned that chain's ultimate upstream one day after the
2026-07-13 bootstrap, invisibly, for 10 days. ``ARM_FEED_DEPENDENCIES``
below is the static arm -> feed source of truth this module previously
lacked; ``check_feed_dependencies_live`` probes each declared dependency
(reusing ``analysis/scanner_predictor_research_free_backfill.py::
assert_champion_feed_fresh`` for ``research_free_backfill``) and, wired
into ``evaluate_gates`` via ``run_weekly_evaluation``, degrades a would-be
promotion onto a dead/orphaned feed to ``no_contest`` with
``blocked_by=["feed_producer_dead"]`` — never crashes the run, never
silently promotes through it, exactly the same validity-guard posture as
``leaderboard_stale_gt_8d``. This is the PROMOTION-TIME complement to the
config-I3086 ``critical_while_champion_arm`` ONGOING-monitoring mechanism
(``alpha-engine-config/private-docs/ARTIFACT_REGISTRY.yaml``), which
catches a feed dying AFTER promotion; this gate catches it AT the moment
of promotion. ``build_champion_audit`` records the promoted arm's
declared dependencies as ``feed_dependencies`` on every outcome.
"""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import date, datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1          # config/producer_champion.json pointer — unchanged shape
AUDIT_SCHEMA_VERSION = 2    # config/apply_audit/producer_champion/{date}.json — v2 shape (I2518)
POINTER_KEY = "config/producer_champion.json"
AUDIT_PREFIX = "config/apply_audit/producer_champion"

VALID_CHAMPIONS = ("scanner_predictor_direct", "thinktank_coverage")

# Retired seat(s) — READ-TOLERATED (a historical pointer/audit artifact using
# this value must never crash the engine) but WRITE-FORBIDDEN (excluded from
# VALID_CHAMPIONS, so write_champion_pointer raises on it).
_LEGACY_CHAMPIONS = ("agentic",)

# ── Shadow-only arms — the single source of truth for "measured, never
# promoted" (Brian's ruling, 2026-08-20, recorded on
# alpha-engine-config-I2515) ──────────────────────────────────────────────
#
#     "research should now be think tank in shadow mode only with the main
#     research process skipped by passing a scanner top 20 to predictor"
#                                                     — Brian, 2026-08-20
#
# An arm listed here is scored every week exactly like any other arm
# (champion-challenger-policy.md §3: measurement is unconditional and is
# NOT what promotion governs), keeps its leaderboard row, and this gate
# keeps recording that it WOULD have won — but it may never take the live
# `config/producer_champion.json` pointer by winning on score. Promoting a
# shadow-only arm requires its own separate ruling from Brian; executing
# that ruling is a ONE-LINE data change (remove the arm from this
# frozenset) and nothing else.
#
# Shadow-only-ness is deliberately a PROPERTY OF AN ARM declared once here,
# not a string literal at the veto site: a future arm introduced in shadow
# mode inherits the protection by being added to this set, at both
# enforcement layers (evaluate_gates, the policy; write_champion_pointer,
# the invariant) simultaneously. Membership here is INDEPENDENT of
# VALID_CHAMPIONS — a shadow-only arm is a legal pointer VALUE to read
# (the executor has a live consumer for it, executor/champion.py::
# _apply_thinktank_coverage) and a legal arm to score; it is simply not a
# legal arm for this engine to promote.
SHADOW_ONLY_ARMS: frozenset[str] = frozenset({"thinktank_coverage"})


def is_shadow_only(arm: str | None) -> bool:
    """True when ``arm`` is declared shadow-only (measured, never promoted).
    The ONLY way any code in this module asks that question — see
    ``SHADOW_ONLY_ARMS`` for the ruling and the one-line unblock."""
    return arm in SHADOW_ONLY_ARMS


OUTCOMES = (
    "promoted",
    "no_contest",
    "unchanged_winner_already_champion",
    # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the winner
    # on score alone was a shadow-only arm, so the pointer was deliberately
    # held. Deliberately NOT reused from the existing vocabulary:
    # "no_contest" asserts the week produced no comparable evidence, which
    # would be false — there WAS a contest and the shadow arm won it; and
    # "unchanged_winner_already_champion" asserts the incumbent scored
    # highest, which would also be false. Both would erase the very
    # counterfactual shadow mode exists to measure. The record carries
    # `counterfactual_winner` naming who actually won.
    "held_shadow_only",
    "error",
)

# blocked_by slugs — union of the current winner-take-all vocabulary and two
# retired vocabularies kept for read-tolerance of historical audit records:
# the pre-I2518 HAC/hysteresis/cooldown engine, and the pre-I2544
# exact-date-only leaderboard read (superseded same-day by the
# latest-available read below; no code path in this module writes either
# retired group again). Slug vocabulary is unchanged by the I2998 direct
# -lift rescoring — only the underlying score SOURCE field changed per arm,
# not the failure-mode taxonomy.
_BLOCKED_BY_SLUGS = (
    # current (winner-take-all + latest-available leaderboard read + direct
    # per-arm lift scoring, I2518/I2544/I2998)
    "no_valid_scanner_predictor_direct_selections",
    "no_valid_thinktank_coverage_selections",
    "scanner_predictor_direct_counterfactual_unavailable",
    "thinktank_coverage_not_in_leaderboard",
    "thinktank_coverage_no_resolved_outcomes",
    "thinktank_coverage_thin_evidence",
    "thinktank_coverage_confidence_unknown",
    # The same two verdicts on the champion side of the comparison
    # (alpha-engine-config-I7549 champion-side half). Separate slugs per arm
    # because the operator response differs by arm: a thin leaderboard row
    # waits on crucible-research's cohort, a thin counterfactual waits on this
    # repo's own research.db cycles.
    "scanner_predictor_direct_thin_evidence",
    "scanner_predictor_direct_confidence_unknown",
    "leaderboard_unavailable",
    "leaderboard_stale_gt_8d",
    "leaderboard_horizon_mismatch",
    "arm_score_unavailable",
    "feed_producer_dead",
    # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the arm
    # that won on score is declared in SHADOW_ONLY_ARMS — measured, never
    # promoted. Paired with outcome="held_shadow_only" and the record's
    # `counterfactual_winner` field.
    "shadow_only_arm",
    "frozen",
    "unclassified_error",
    # retired (pre-I2518 HAC/hysteresis/cooldown engine) — historical read-only
    "insufficient_matured_cohorts",
    "cooldown_active",
    "not_significant_hac_adjusted",
    "hysteresis_not_satisfied",
    # retired (pre-I2544 exact-date-only leaderboard read) — historical
    # read-only: this slug fired when the artifact's self-reported "date"
    # field disagreed with an exact-match run_date key read; the
    # latest-available read no longer requires an exact match, so this
    # condition can no longer occur (superseded by leaderboard_stale_gt_8d
    # for the age-bound case).
    "leaderboard_stale",
)

# Honest staleness bound (alpha-engine-config-I2544, 2026-07-14): a selected
# research/producer_leaderboard/{date}.json artifact older than this many
# calendar days relative to run_date is treated as unavailable rather than
# scored — see find_latest_research_producer_leaderboard_date /
# _score_thinktank_coverage.
LEADERBOARD_STALENESS_DAYS = 8

# ── Evidence-confidence contract with crucible-research (alpha-engine-config
# -I7549 consuming I7542) ─────────────────────────────────────────────────
#
# The per-spec confidence vocabulary PRODUCED by crucible-research
# scoring/leaderboard_scoring.py::confidence_for. This module consumes it and
# deliberately does NOT reimplement the thinness test: the evidence floor is a
# per-slot fact owned by the producer's LEADERBOARD_SLOTS registry
# (champion-challenger-policy.md §10), and a second copy here would drift.
CONFIDENCE_OK = "ok"
CONFIDENCE_THIN = "thin"
CONFIDENCE_INSUFFICIENT = "insufficient"
LEADERBOARD_CONFIDENCE_VOCABULARY = (
    CONFIDENCE_OK, CONFIDENCE_THIN, CONFIDENCE_INSUFFICIENT,
)

# Verdicts this module records for itself when the ARTIFACT did not supply a
# usable one. Both are refusals, never a pass — see the module docstring's
# backwards-compatibility note.
CONFIDENCE_UNKNOWN = "unknown"            # pre-I7542 artifact, no floor to derive from
CONFIDENCE_UNRECOGNISED = "unrecognised"  # confidence present, outside the vocabulary
# RETIRED as an emitted value (alpha-engine-config-I7549 champion-side half,
# same tracker): between #688 and this change, scanner_predictor_direct carried
# ``not_leaderboard_scored`` because it is scored from this repo's own
# counterfactual rather than the producer leaderboard. That was true and is
# still true — but it named the SOURCE of the evidence where the field's job is
# to state how much of it there is, so the arm the gate compares AGAINST the
# gated one had no confidence verdict at all. It now carries a real verdict
# against this repo's own declared floor (MIN_CYCLES_FOR_INFERENCE below). The
# constant stays defined for read-tolerance of any audit record written in that
# window; nothing emits it.
CONFIDENCE_NOT_LEADERBOARD_SCORED = "not_leaderboard_scored"

# ── The champion side's evidence floor (alpha-engine-config-I7549) ─────────
#
# #688 closed the challenger side: a producer-leaderboard row is admitted only
# at ``confidence == "ok"``. The other arm of the same two-arm comparison was
# left unfloored, and it is reachable at exactly one observation:
# ``analysis/end_to_end.py::_scanner_then_predictor_topN`` returns a result at
# ``if n_cycles < 1``. So the identical defect — a mean with no computable
# dispersion moving the live pointer — survived on the champion's side of the
# comparison #688 fixed. A gate is not fixed while either input to it can be
# one observation; whichever arm is thin, the DIFFERENCE the gate acts on is
# noise, and the audit record reads the same either way.
#
# The floor lives here, not on the artifact, because THIS repo owns that arm's
# measurement (champion-challenger-policy.md §10: every slot names its evidence
# floor in the registry that owns it — the leaderboard-scored arm's floor is
# crucible-research's ``min_dates_for_inference`` and is read off the artifact,
# never reimplemented; this arm has no producer to read it from).
#
# 5 mirrors crucible-research's MIN_DATES_FOR_INFERENCE for the same reason:
# fewer than five clusters cannot carry a clustered statistic. Measured
# 2026-08-17 from ``research/producer_leaderboard_champion_gate/2026-08-14.json``:
# the live weekly runs carry ``n_cycles: 15`` (``n_picks: 119``), so this floor
# bounds the degenerate case without binding on today's evidence.
MIN_CYCLES_FOR_INFERENCE = 5

# The horizon this gate decides on, in trading sessions. See the module
# docstring's HORIZON section: 21 is the leaderboard's PRIMARY horizon (whose
# block is spread across the artifact's top level) and the only horizon at
# which BOTH arms are scored. Named here so the choice is an assertion rather
# than an accident of which block happens to sit at the top level.
GATE_HORIZON_DAYS = 21

RESEARCH_PRODUCER_LEADERBOARD_PREFIX = "research/producer_leaderboard/"
_RESEARCH_PRODUCER_LEADERBOARD_KEY_RE = re.compile(
    r"^research/producer_leaderboard/(\d{4}-\d{2}-\d{2})\.json$"
)

# ── Promotion-time feed-dependency liveness gate (alpha-engine-config-I3165,
# 2026-07-23) ─────────────────────────────────────────────────────────────
#
# config#3053 root cause: scanner_predictor_direct was promoted 2026-07-13
# with its live-trade feed chain (research_free_backfill, itself sourced
# from scanner_evaluations) undeclared anywhere in the promotion record —
# config#1580 orphaned that chain's ultimate upstream producer the very
# next day, and nothing at promotion time (or afterward, until the
# freshness monitor's config-I3086 critical_while_champion_arm mechanism)
# checked it. That mechanism is ONGOING monitoring, wired to the ARTIFACT
# _REGISTRY row's static severity being coerced dynamic while a listed arm
# is champion; THIS gate is the complementary PROMOTION-TIME check, run
# once per weekly evaluation, at the moment a challenger would newly become
# champion.
#
# ARM_FEED_DEPENDENCIES is the small, static, source-of-truth mapping this
# repo previously lacked entirely (no arm->feed mapping existed anywhere —
# scanner_predictor_direct's chain was documented only in this module's own
# sibling analysis/scanner_predictor_research_free_backfill.py docstring).
# Each value is a list of ARTIFACT_REGISTRY.yaml artifact_ids (alpha-engine
# -config/private-docs/ARTIFACT_REGISTRY.yaml) — deliberately the DIRECT
# feed the arm's live trading reads, not its deeper transitive upstreams
# (research_free_backfill's own upstream, scanner_evaluations, has no
# ARTIFACT_REGISTRY row at all as of this writing — registering it is a
# separate, out-of-scope gap). thinktank_coverage carries no entry (and
# none is required): its evidence chain is the producer leaderboard, already
# gated above by leaderboard_date_used/leaderboard_stale_gt_8d — it names no
# live-trade feed artifact of its own.
ARM_FEED_DEPENDENCIES: dict[str, list[str]] = {
    "scanner_predictor_direct": ["research_free_backfill"],
}

# artifact_id -> liveness prober. Each prober takes (bucket, run_date,
# s3_client) and RAISES on a dead/stale/missing/unreadable producer, returns
# None (no exception) when live — the same shape
# assert_champion_feed_fresh already uses. Deliberately reuses that
# existing, already-tested producer-side check (config#3053) rather than
# hand-rolling a second freshness reader for the same artifact: it already
# encodes the correct content-derived staleness rule for
# research_free_backfill (newest prediction_date vs run_date, not S3
# LastModified, which a no-op rewrite would falsely refresh). New feed
# dependencies added to ARM_FEED_DEPENDENCIES in the future need a prober
# registered here (or, if a cheap presence/HEAD check suffices, a lighter
# adapter) -- an arm whose feed has no registered prober here is simply not
# checked (fails open on THIS gate; the config-I3086 ongoing monitor still
# covers it once promoted) rather than crashing the run.
def _check_research_free_backfill_live(bucket: str, run_date: str, s3_client) -> None:
    from analysis.scanner_predictor_research_free_backfill import assert_champion_feed_fresh

    assert_champion_feed_fresh(bucket, run_date=run_date, s3_client=s3_client)


_FEED_LIVENESS_PROBES = {
    "research_free_backfill": _check_research_free_backfill_live,
}


def check_feed_dependencies_live(
    arm: str, *, bucket: str, run_date: str, s3_client=None,
) -> str | None:
    """Probe every feed artifact ``arm`` declares in
    ``ARM_FEED_DEPENDENCIES`` for producer liveness. Returns
    ``"feed_producer_dead"`` (the ``blocked_by`` slug) the first time a
    declared dependency's registered prober raises anything at all;
    returns ``None`` when ``arm`` declares no dependencies, every declared
    dependency has no registered prober, or every registered prober passed.

    Never raises — a probe failure (dead feed, unreadable artifact, an
    unexpected exception in the prober itself) must degrade the gate to a
    no-contest, exactly like every other validity guard in this module
    (module docstring's binding config#2884 lesson: an error here must
    never silently default to a promotion, and must never crash the weekly
    evaluation either).
    """
    for feed_id in ARM_FEED_DEPENDENCIES.get(arm, []):
        probe = _FEED_LIVENESS_PROBES.get(feed_id)
        if probe is None:
            logger.warning(
                "champion_promotion: %r declares feed dependency %r with no "
                "registered liveness prober — skipping (not checked by this "
                "gate; add a _FEED_LIVENESS_PROBES entry to cover it)",
                arm, feed_id,
            )
            continue
        try:
            probe(bucket, run_date, s3_client)
        except Exception as e:  # noqa: BLE001 — a dead/unreadable feed (or any
            # unexpected prober failure) degrades this promotion to
            # no_contest; it must never crash the weekly evaluation and
            # must never be silently swallowed into a promotion either.
            logger.warning(
                "champion_promotion: feed dependency %r for arm %r looks "
                "dead/orphaned at promotion time (%s) — blocking this "
                "promotion (feed_producer_dead)", feed_id, arm, e,
            )
            return "feed_producer_dead"
    return None


# HAC lag helper constants — still consulted by hac_significance() below,
# which is retained as an independent, tested utility (see module docstring)
# even though it no longer gates the promotion decision.
_HORIZON_DAYS = 21              # 21d forward-alpha horizon (config#1405 basis)
_CADENCE_DAYS = 7               # weekly evaluation cadence

_cfg: dict = {}


def init_config(config: dict) -> None:
    """Called unconditionally by evaluate.py at optimizer-stage start. The
    winner-take-all engine currently defines no configurable thresholds
    (no significance level, no hysteresis/cooldown weeks) — this is kept as
    a stable entry point for evaluate.py's wiring and for hac_significance's
    still-configurable horizon/cadence (lag = round(horizon/cadence))."""
    global _cfg
    _cfg = config.get("champion_promotion", {})


def _hac_lag() -> int:
    """Bartlett-kernel lag = round(horizon_days / cadence_days). Feeds
    ``hac_significance`` only — not load-bearing for the winner-take-all
    decision itself."""
    horizon = int(_cfg.get("horizon_days", _HORIZON_DAYS))
    cadence = int(_cfg.get("cadence_days", _CADENCE_DAYS))
    return max(0, round(horizon / cadence))


# ── Retained utility: HAC/Newey-West-adjusted overlap-aware significance ───
# Not wired into evaluate_gates() under the winner-take-all policy (see
# module docstring) — kept as an independently unit-tested, available
# diagnostic.


def hac_significance(
    weekly_sn_lift: list[float], *, alpha: float = 0.05,
) -> dict:
    """Overlap-aware two-sided significance test of whether a weekly
    sector-neutral lift series is significantly different from zero, using
    the Newey-West (1994) HAC standard error of the mean (Bartlett kernel;
    lag = ``_hac_lag()``) in place of the naive i.i.d. ``s/sqrt(n)`` standard
    error. See the module history (git log) for the full derivation this
    docstring previously carried in-line; unchanged behavior, retained as an
    available utility (not currently gate-connected — see module docstring).

    Uses the vendored, independently-unit-tested
    ``nousergon_lib.quant.stats.intervals.newey_west_se``.

    Returns a dict:
      ``{"status": "ok", "n": int, "mean": float, "se": float, "lags": int,
         "t_stat": float, "p_value": float, "significant": bool}``
      or ``{"status": "insufficient_data", "n": int}`` when fewer than 2
      finite observations are available.
    """
    from nousergon_lib.quant.stats.intervals import newey_west_se

    clean = [float(x) for x in weekly_sn_lift if x is not None and not math.isnan(x)]
    n = len(clean)
    if n < 2:
        return {"status": "insufficient_data", "n": n}

    nw = newey_west_se(clean, max_lags=_hac_lag())
    if nw.get("status") != "ok":
        return {"status": "insufficient_data", "n": n}

    mean = nw["estimate"]
    se = nw["se"]
    if se <= 0:
        return {
            "status": "ok", "n": n, "mean": mean, "se": 0.0, "lags": nw["lags"],
            "t_stat": math.inf if mean != 0 else 0.0,
            "p_value": 0.0 if mean != 0 else 1.0,
            "significant": bool(mean != 0),
        }

    from scipy.stats import t as _student_t

    t_stat = mean / se
    dof = max(n - 1, 1)
    p_value = float(2.0 * _student_t.sf(abs(t_stat), dof))
    return {
        "status": "ok",
        "n": n,
        "mean": mean,
        "se": se,
        "lags": nw["lags"],
        "t_stat": float(t_stat),
        "p_value": p_value,
        "significant": bool(p_value < alpha),
    }


# ── Gate engine (weekly winner-take-all) ────────────────────────────────────


def _other_champion(champion: str) -> str:
    others = [c for c in VALID_CHAMPIONS if c != champion]
    if len(others) != 1:
        raise ValueError(
            f"_other_champion expects exactly 2 VALID_CHAMPIONS, got {VALID_CHAMPIONS!r}"
        )
    return others[0]


def _normalize_champion_before(champion: str) -> str:
    """Normalize a pointer/default champion value for GATE purposes only —
    never mutates the pointer itself (a held/no-contest week must never
    write). ``agentic`` (the retired seat, config-I2518 seat swap) WARNs and
    is treated as ``scanner_predictor_direct`` — belt-and-braces: the live
    pointer has been ``scanner_predictor_direct`` since 2026-07-13, so this
    path is not expected to fire in practice, only to guarantee a stale or
    hand-inspected historical pointer can never crash this engine. Any
    other unrecognized value is treated the same way (WARN + default to the
    base-case arm)."""
    if champion in VALID_CHAMPIONS:
        return champion
    if champion in _LEGACY_CHAMPIONS:
        logger.warning(
            "Champion pointer had legacy champion=%r (retired seat, "
            "alpha-engine-config-I2518 seat swap) — normalizing to %r for "
            "gate purposes only; the pointer itself is left untouched unless "
            "this week's gates clear a move.",
            champion, VALID_CHAMPIONS[0],
        )
        return VALID_CHAMPIONS[0]
    logger.warning(
        "Champion pointer had unrecognized champion=%r — treating as %r for "
        "gate purposes only (the pointer itself is left untouched unless "
        "gates clear a move)", champion, VALID_CHAMPIONS[0],
    )
    return VALID_CHAMPIONS[0]


def evaluate_gates(
    *,
    champion_before: str,
    arm_scores: dict,
    freeze: bool,
    feed_blocked_slug: str | None = None,
) -> dict:
    """Weekly winner-take-all decision (Brian's ruling, alpha-engine-config
    -I2518, 2026-07-14) — supersedes the HAC-significance / 2-week hysteresis
    / 2-week cooldown gates this module previously enforced, INCLUDING the
    standing ``cooldown_until: 2026-07-27`` carried in a prior audit record
    (no longer read or honored).

    ``arm_scores`` is the return of ``build_weekly_arm_scores`` below:
    ``{"scores": {"scanner_predictor_direct": float|None,
    "thinktank_coverage": float|None}, "unavailable_reasons": {arm: slug,
    ...}, "leaderboard_date_used": str|None}``. A ``None`` score means no
    valid evidence exists for that arm THIS week — a definitional
    NO-CONTEST (validity guard), never a statistical gate and never a
    default win for either side. ``leaderboard_date_used`` (alpha-engine
    -config-I2544) is the date of the ``research/producer_leaderboard/
    {date}.json`` artifact actually selected (latest available <=
    run_date) — carried through into every outcome record (promoted,
    no_contest, and unchanged) so the audit trail always shows which
    week's evidence decided (or declined to decide) a flip.

    ``feed_blocked_slug`` (alpha-engine-config-I3165, 2026-07-23) is the
    precomputed result of ``check_feed_dependencies_live`` for the
    CHALLENGER arm — the caller (``run_weekly_evaluation``) does that I/O
    up front and passes only the verdict in, keeping this function pure.
    When non-None (currently only ever ``"feed_producer_dead"``) AND the
    challenger would otherwise win this week, the would-be promotion is
    degraded to a no_contest instead — a challenger whose declared feed
    producer looks dead/orphaned must never become champion, exactly
    parallel to the ``leaderboard_stale_gt_8d`` validity guard. Checked
    only on the win path (irrelevant to a no-contest or a defended
    incumbency, since the pointer would not move either way).

    Decision: whichever arm has the strictly higher score this week wins.
    A tie (or either side missing) never flips the pointer — ties favor the
    incumbent (``champion_before``) so the pointer never moves on a null or
    exactly-equal signal.

    Pure function — no I/O — independently unit-testable against synthetic
    score fixtures.

    ``arm_confidence`` (additive, alpha-engine-config-I7549) is threaded
    straight through from ``build_weekly_arm_scores`` onto every outcome
    record. Under I7549 a ``None`` score can now also mean "the arm WAS
    scored, on evidence too thin to compare" (``thinktank_coverage_thin
    _evidence``) — indistinguishable from every other no-contest by
    ``blocked_by`` alone once a second thin-adjacent slug exists, so the
    verdict itself is recorded. This strengthens rather than weakens the
    §5.2 hysteresis posture: every I7549 path produces a ``None`` score,
    which is already a definitional no-contest holding the pointer in BOTH
    directions, so no flip becomes possible that was not possible before.

    **Shadow-only veto (Brian's ruling, 2026-08-20, alpha-engine-config
    -I2515)** — symmetric with ``feed_blocked_slug`` above and applied on
    the same win path: when the arm that won on score is declared in
    ``SHADOW_ONLY_ARMS`` and is not already the champion, the would-be
    promotion is degraded to ``outcome="held_shadow_only"`` with
    ``blocked_by=["shadow_only_arm"]`` and ``champion_after`` left at
    ``champion_before`` — ``run_weekly_evaluation`` therefore never reaches
    ``write_champion_pointer``. The MEASUREMENT is untouched: both scores
    are still computed, still recorded, and ``counterfactual_winner`` names
    the arm that would have taken the pointer, so a shadow arm's wins
    remain visible in the durable weekly audit trail rather than being
    erased into a hold that looks like a defended incumbency. Checked
    BEFORE ``feed_blocked_slug`` and before ``--freeze``: a shadow-only
    hold is a standing policy decision, not a per-week validity guard or a
    suppression, so it is the true and stable reason the pointer did not
    move (and the feed probe is I/O the caller can skip entirely for a
    shadow-only challenger).

    ``counterfactual_winner`` (additive, alpha-engine-config-I2515) is on
    EVERY outcome: the arm with the strictly higher score this week, or
    ``None`` when no comparison was possible (a no-contest). On an ordinary
    promotion it equals ``champion_after``; on a defended incumbency it
    equals ``champion_before``; on ``held_shadow_only`` it is the shadow
    arm — the one case where it differs from the pointer's destination, and
    the reason the field exists.

    Returns a dict with keys: outcome, champion_before, champion_after,
    challenger, champion_score, challenger_score, blocked_by,
    leaderboard_date_used, arm_confidence, counterfactual_winner.
    """
    challenger = _other_champion(champion_before)
    scores = arm_scores.get("scores", {})
    reasons = arm_scores.get("unavailable_reasons", {})
    champ_score = scores.get(champion_before)
    chall_score = scores.get(challenger)

    record: dict[str, Any] = {
        "champion_before": champion_before,
        "champion_after": champion_before,
        "challenger": challenger,
        "champion_score": champ_score,
        "challenger_score": chall_score,
        "blocked_by": None,
        "leaderboard_date_used": arm_scores.get("leaderboard_date_used"),
        # alpha-engine-config-I7549 — carried on EVERY outcome, so the audit
        # trail names the evidence that decided, or declined to decide.
        "arm_confidence": arm_scores.get("arm_confidence") or None,
        # alpha-engine-config-I2515 (2026-08-20) — who WOULD have taken the
        # pointer on score alone. Populated below once both scores are
        # known; stays None on a no-contest, where no comparison happened.
        "counterfactual_winner": None,
    }

    if champ_score is None or chall_score is None:
        blocked: list[str] = []
        if champ_score is None:
            blocked.append(reasons.get(champion_before, "arm_score_unavailable"))
        if chall_score is None:
            blocked.append(reasons.get(challenger, "arm_score_unavailable"))
        record["outcome"] = "no_contest"
        record["blocked_by"] = blocked
        return record

    winner = challenger if chall_score > champ_score else champion_before
    record["counterfactual_winner"] = winner

    if winner == champion_before:
        record["outcome"] = "unchanged_winner_already_champion"
        return record

    # ── Shadow-only veto (Brian's ruling 2026-08-20, alpha-engine-config
    # -I2515) ────────────────────────────────────────────────────────────
    # The challenger won on score, but a SHADOW-ONLY arm is measured, never
    # promoted: it may not take the live pointer by winning. Checked first
    # on the win path — ahead of the feed probe and --freeze — because it
    # is a standing policy decision rather than a per-week validity guard
    # or a suppression, so it is the TRUE and stable reason the pointer did
    # not move, and the audit record must say that rather than attribute
    # the hold to a transient feed or a freeze flag.
    #
    # The measurement is untouched: champion_score, challenger_score,
    # arm_confidence and counterfactual_winner are all already on the
    # record above, so this week's "the shadow arm would have won" is
    # durable in config/apply_audit/producer_champion/{date}.json. Removing
    # the arm from SHADOW_ONLY_ARMS (with Brian's ruling) restores the
    # ordinary promotion path with no other change here.
    if is_shadow_only(winner):
        record["outcome"] = "held_shadow_only"
        record["blocked_by"] = ["shadow_only_arm"]
        logger.warning(
            "champion_promotion: %r outscored the champion %r this week "
            "(%s > %s) but is declared SHADOW-ONLY (alpha-engine-config"
            "-I2515, Brian's 2026-08-20 ruling) — the live pointer is HELD "
            "at %r. This is the designed outcome, not a defect: the win is "
            "recorded as counterfactual_winner in the weekly audit record. "
            "Promoting this arm needs its own ruling plus removing it from "
            "SHADOW_ONLY_ARMS.",
            winner, champion_before, chall_score, champ_score, champion_before,
        )
        return record

    # Challenger wins this week on score alone — but a promotion-time feed
    # -liveness guard (alpha-engine-config-I3165) can still veto it: the
    # challenger's declared feed_dependencies must have a live producer
    # before the pointer is allowed to move onto it. Checked before
    # --freeze so the audit record always shows the TRUE reason a
    # would-be promotion didn't happen (feed_producer_dead is a validity
    # guard, not a suppression like frozen — the two are mutually
    # exclusive outcomes of the same win, and the feed check is the more
    # fundamental one: freeze only suppresses a promotion this gate has
    # already decided is otherwise valid).
    if feed_blocked_slug is not None:
        record["outcome"] = "no_contest"
        record["blocked_by"] = [feed_blocked_slug]
        return record

    # Challenger wins this week — a promotion, subject only to --freeze.
    if freeze:
        record["outcome"] = "promoted"
        record["blocked_by"] = ["frozen"]
        # champion_after is NOT advanced under freeze — the write is
        # suppressed, so the carry-forward state must reflect reality.
        return record

    record["outcome"] = "promoted"
    record["champion_after"] = winner
    return record


# ── config/producer_champion.json writer (single writer, dual caller) ──────


def write_champion_pointer(
    bucket: str,
    champion: str,
    *,
    promotion_source: str,
    upload: bool,
    s3_client=None,
) -> dict:
    """THE single writer for ``config/producer_champion.json``. Both the
    gate engine (``promotion_source="gate_engine"``) and the one-shot
    2026-07-13 operator bootstrap (``promotion_source="operator_bootstrap"``)
    call this function — never write the pointer directly.

    ``champion`` MUST be in ``VALID_CHAMPIONS`` — this is the write-forbidden
    half of the read-tolerated/write-forbidden posture for retired seats
    (e.g. ``agentic``): raises ValueError for anything else, including every
    ``_LEGACY_CHAMPIONS`` value.

    ``champion`` MUST ALSO NOT be in ``SHADOW_ONLY_ARMS`` (Brian's ruling,
    2026-08-20, alpha-engine-config-I2515) — raises ValueError. This is
    DEFENCE IN DEPTH, deliberately duplicating the ``evaluate_gates`` veto
    one layer down: the gate is the POLICY (it decides, and records why the
    pointer was held), this writer is the INVARIANT (nothing that reaches
    S3 can violate it). A future caller that bypasses ``evaluate_gates``
    entirely — a new operator bootstrap, a backfill, a repair script —
    still cannot flip the live pointer onto a shadow-only arm. Note the
    asymmetry with reads: ``read_champion_pointer`` and
    ``_normalize_champion_before`` remain fully tolerant of a shadow-only
    value, since the executor has a live consumer for one
    (``executor/champion.py::_apply_thinktank_coverage``) and a historical
    or hand-inspected pointer must never crash this engine.

    Idempotent / bidirectional-safe: callers only invoke this when a gate
    decision has already determined the pointer SHOULD move (a no-contest or
    unchanged week must never call this).

    Raises on S3 write failure when ``upload=True`` — a swallowed failure
    here would silently leave the live executor trading the wrong arm.
    """
    if champion not in VALID_CHAMPIONS:
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} not in {VALID_CHAMPIONS}"
        )
    if is_shadow_only(champion):
        # Fail LOUD (module posture: no silent swallows on a writer). A
        # caller reaching here has already bypassed the evaluate_gates
        # veto, so degrading quietly would reproduce exactly the defect
        # this guard exists to prevent — a live pointer moved onto an arm
        # Brian ruled measure-only.
        raise ValueError(
            f"write_champion_pointer: champion={champion!r} is declared "
            f"SHADOW-ONLY ({sorted(SHADOW_ONLY_ARMS)}) — measured, never "
            "promoted (Brian's ruling 2026-08-20, alpha-engine-config"
            "-I2515). The live config/producer_champion.json pointer may "
            "not be moved onto it. Promoting it requires its own ruling "
            "plus removing it from SHADOW_ONLY_ARMS."
        )
    pointer = {
        "schema_version": SCHEMA_VERSION,
        "champion": champion,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "promotion_source": promotion_source,
    }
    if upload:
        s3 = s3_client or boto3.client("s3")
        body = json.dumps(pointer, indent=2, allow_nan=False).encode("utf-8")
        s3.put_object(
            Bucket=bucket, Key=POINTER_KEY, Body=body, ContentType="application/json",
        )
        logger.info(
            "Champion pointer written: s3://%s/%s champion=%s source=%s",
            bucket, POINTER_KEY, champion, promotion_source,
        )
    else:
        logger.info(
            "Champion pointer write skipped (upload=False) — would have set "
            "champion=%s source=%s", champion, promotion_source,
        )
    return pointer


def read_champion_pointer(bucket: str, s3_client=None) -> dict | None:
    """Read the current pointer. Returns None on 404/NoSuchKey (pre-bootstrap
    — no promotion has ever been written; callers should treat champion as
    the base-case arm, ``VALID_CHAMPIONS[0]`` == 'scanner_predictor_direct',
    mirroring executor/champion.py's own default post-I2515). Any other
    error is NOT swallowed here (this is the producer side, not the
    fail-loud executor consumer) but is logged and returns None so a
    transient read hiccup degrades to the base-case default rather than
    crashing the whole weekly evaluate run — the outcome is recorded as
    ``error`` by the caller either way, never a silent pointer write."""
    s3 = s3_client or boto3.client("s3")
    try:
        obj = s3.get_object(Bucket=bucket, Key=POINTER_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No champion pointer at s3://%s/%s (pre-bootstrap)", bucket, POINTER_KEY)
        else:
            logger.warning(
                "Champion pointer read failed (%s) — treating as pre-bootstrap "
                "base-case default", e,
            )
        return None
    except Exception as e:  # noqa: BLE001 — degraded-read carve-out, see docstring.
        logger.warning(
            "Champion pointer read failed (%s) — treating as pre-bootstrap "
            "base-case default", e,
        )
        return None


# ── config/apply_audit/producer_champion/{date}.json writer ────────────────


def load_prior_audit(bucket: str, s3_client=None) -> dict | None:
    """Read the prior weekly audit record (latest.json). Retained for API
    parity with the pre-I2518 engine (some callers/tests may still probe
    prior-run state for observability) — no longer consulted by
    evaluate_gates itself (winner-take-all carries no state forward:
    no hysteresis counter, no cooldown date). Absent artifact (first-ever
    run) -> None; any other read failure logs WARN and returns None."""
    s3 = s3_client or boto3.client("s3")
    key = f"{AUDIT_PREFIX}/latest.json"
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info("No prior producer_champion audit at s3://%s/%s (first run)", bucket, key)
        else:
            logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("Prior producer_champion audit read failed (%s) — state restarts", e)
        return None


def build_champion_audit(
    as_of: str,
    gate_result: dict | None,
    *,
    freeze: bool,
    error: str | None = None,
) -> dict:
    """Build the weekly audit record (schema v2, published in
    ``nousergon_lib.contracts`` as ``producer_champion_audit`` — moved out of
    this repo's local ``contracts/`` dir under alpha-engine-config-I7605 so
    the dashboard consumer reads the same resource this producer does,
    instead of walking this repo's working tree. Bumped from v1 under
    alpha-engine-config-I2518: the HAC/hysteresis/cooldown fields
    (``challenger_matured_cohorts``, ``sn_lift_vs_champion``,
    ``consecutive_wins``, ``cooldown_until``) are retired in favor of
    ``champion_score``/``challenger_score`` — no live consumer outside this
    repo reads the audit record's fields (verified 2026-07-14), so no
    cross-repo coordination was required for the bump; v1 historical
    records remain valid documents under the frozen v1 shape and are not
    revalidated against v2). Written every week regardless of outcome —
    this IS the liveness proxy (config#2054).

    ``leaderboard_date_used`` (additive, alpha-engine-config-I2544,
    2026-07-14) is the date of the ``research/producer_leaderboard/
    {date}.json`` artifact actually consulted this run (the latest
    available <= ``as_of``, or None when no leaderboard was available at
    all / evaluation aborted before scoring) — always present (nullable),
    on every outcome including ``error``, so the audit trail is never
    silent about which week's evidence decided a flip.

    ``counterfactual_winner`` (additive, alpha-engine-config-I2515,
    2026-08-20) is the arm with the strictly higher weekly score — who
    WOULD have taken the pointer on score alone — or None when no
    comparison was possible (no_contest / error). It equals
    ``champion_after`` on an ordinary promotion and ``champion_before`` on
    a defended incumbency; the case it exists for is
    ``outcome="held_shadow_only"``, where it names the shadow-only arm that
    won and was deliberately not promoted. This is what keeps shadow-mode
    measurement legible in the durable weekly record.

    ``feed_dependencies`` (additive, alpha-engine-config-I3165, 2026-07-23)
    is ``ARM_FEED_DEPENDENCIES.get(champion_after)`` — the live-trade feed
    artifact_id(s) the record's ``champion_after`` arm declares, or ``None``
    when ``champion_after`` declares none (``thinktank_coverage``) or is
    itself ``None`` (``outcome="error"``, evaluation aborted before a
    champion could be read). Always derived from ``champion_after``, never
    ``champion_before`` — this field names what the LIVE pointer now
    depends on, which is unchanged from before this run on every
    non-promoted outcome and newly the challenger's feed on a promotion.

    ``arm_confidence`` (additive, alpha-engine-config-I7549, 2026-08-17) is
    the per-arm evidence verdict that decided — or declined to decide — this
    week: ``{"scanner_predictor_direct": "ok"|"thin"|"insufficient"|"unknown",
    "thinktank_coverage": "ok"|"thin"|"insufficient"|"unknown"|
    "unrecognised"}`` — one vocabulary, both arms (alpha-engine-config-I7549
    champion-side half; ``not_leaderboard_scored`` was emitted for the champion
    arm between #688 and that change and remains read-tolerated). Null only on ``outcome="error"`` (evaluation aborted
    before scoring). Without it, a week held at ``blocked_by=
    ["thinktank_coverage_thin_evidence"]`` records THAT the gate declined but
    not on what — and the whole point of the I7549 change is that a thin row
    is a different fact from an absent one (champion-challenger-policy.md
    §7.2)."""
    if error is not None or gate_result is None:
        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "date": as_of,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "outcome": "error",
            "champion_before": None,
            "champion_after": None,
            "champion_score": None,
            "challenger_score": None,
            "blocked_by": ["leaderboard_unavailable" if error else "unclassified_error"],
            "freeze": freeze,
            "detail": error or "gate evaluation did not run",
            "leaderboard_date_used": None,
            "feed_dependencies": None,
            "counterfactual_winner": None,
            "arm_confidence": None,
        }
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "date": as_of,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "outcome": gate_result["outcome"],
        "champion_before": gate_result["champion_before"],
        "champion_after": gate_result["champion_after"],
        "champion_score": gate_result["champion_score"],
        "challenger_score": gate_result["challenger_score"],
        "blocked_by": gate_result["blocked_by"],
        "challenger": gate_result["challenger"],
        "freeze": freeze,
        "leaderboard_date_used": gate_result.get("leaderboard_date_used"),
        "feed_dependencies": ARM_FEED_DEPENDENCIES.get(gate_result["champion_after"]) or None,
        # alpha-engine-config-I2515 (2026-08-20 shadow-only ruling): the arm
        # that won on score this week, which on outcome="held_shadow_only"
        # is NOT champion_after. Without it the audit record could not
        # express "the shadow arm would have won", and a shadow arm that
        # silently stops being visible in the record defeats the whole
        # point of shadow mode (champion-challenger-policy.md §3).
        "counterfactual_winner": gate_result.get("counterfactual_winner"),
        "arm_confidence": gate_result.get("arm_confidence"),
    }


def write_champion_audit(bucket: str, run_date: str, audit: dict, s3_client=None) -> str:
    """Write dated + latest audit artifacts. RAISES on failure — mirrors
    ``optimizer/apply_audit.write_audit``'s fail-loud posture: this record
    is the load-bearing liveness proxy, a swallowed write failure would
    recreate the exact invisible-silence defect config#2054 exists to
    retire."""
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(audit, indent=2, allow_nan=False).encode("utf-8")
    dated_key = f"{AUDIT_PREFIX}/{run_date}.json"
    latest_key = f"{AUDIT_PREFIX}/latest.json"
    s3.put_object(Bucket=bucket, Key=dated_key, Body=body, ContentType="application/json")
    s3.put_object(Bucket=bucket, Key=latest_key, Body=body, ContentType="application/json")
    logger.info("Champion audit written: s3://%s/%s (+ latest.json)", bucket, dated_key)
    return dated_key


# ── Weekly arm-score sourcing ────────────────────────────────────────────────


def _score_scanner_predictor_direct(
    e2e_lift: dict | None,
) -> tuple[float | None, str | None, str]:
    """scanner_predictor_direct's weekly score (alpha-engine-config-I2998):
    this run's backtester-internal ``scanner_then_predictor_topN``
    counterfactual's OWN realized sector-neutral 21d alpha —
    ``sector_neutral_mean_alpha_21d`` — which is already benchmark-relative
    at the source (realized log return minus the log SPY return over the
    same window, see ``analysis/end_to_end.py::_scanner_then_predictor_topN``),
    i.e. a direct lift vs the SPY zero-line, not vs any live comparator arm. See
    ``leaderboard_entry_from_e2e_lift``.

    Returns ``(score, unavailable_slug, confidence_verdict)`` — the same
    three-element shape as ``_score_thinktank_coverage``, and for the same
    reason (alpha-engine-config-I7549 champion-side half): both arms of a
    two-arm gate must state how much evidence stands behind them, in one
    vocabulary, or the audit record cannot be read as a comparison.

    The evidence floor is ``MIN_CYCLES_FOR_INFERENCE`` cycles. The
    counterfactual admits a result at one cycle, so before this the champion
    side of the comparison could be a single observation while the challenger
    side was floored at five — the arm that #688 protected was being compared
    against one that nothing protected."""
    entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    if entry is None:
        # Nothing scored at all — the pre-existing slug, unchanged meaning.
        return (
            None,
            "scanner_predictor_direct_counterfactual_unavailable",
            CONFIDENCE_INSUFFICIENT,
        )
    n_cycles = entry.get("n_cycles")
    if isinstance(n_cycles, bool) or not isinstance(n_cycles, int) or n_cycles < 0:
        # A mean with no honest count behind it. Refused, never trusted —
        # exactly the posture leaderboard_row_confidence takes toward an
        # artifact that cannot supply a usable verdict.
        logger.warning(
            "[champion_promotion] scanner_predictor_direct counterfactual "
            "reported no usable n_cycles (%r) — refusing to score it, "
            "no-contest (alpha-engine-config-I7549)", n_cycles,
        )
        return (
            None, "scanner_predictor_direct_confidence_unknown", CONFIDENCE_UNKNOWN,
        )
    if n_cycles == 0:
        return (
            None,
            "scanner_predictor_direct_counterfactual_unavailable",
            CONFIDENCE_INSUFFICIENT,
        )
    if n_cycles < MIN_CYCLES_FOR_INFERENCE:
        logger.warning(
            "[champion_promotion] scanner_predictor_direct NOT scored this "
            "week: n_cycles=%d < MIN_CYCLES_FOR_INFERENCE=%d -> "
            "blocked_by=scanner_predictor_direct_thin_evidence. The champion "
            "pointer is unchanged; a thin arm is not evidence on the "
            "champion's side of the comparison either "
            "(champion-challenger-policy.md §4, alpha-engine-config-I7549).",
            n_cycles, MIN_CYCLES_FOR_INFERENCE,
        )
        return None, "scanner_predictor_direct_thin_evidence", CONFIDENCE_THIN
    return entry["sector_neutral_mean_alpha_21d"], None, CONFIDENCE_OK


def leaderboard_row_confidence(row: dict, leaderboard: dict) -> str:
    """How much evidence stands behind one producer-leaderboard spec ``row``.

    Returns one of ``LEADERBOARD_CONFIDENCE_VOCABULARY`` (the producer's own
    verdict, read straight off the row — alpha-engine-config-I7542), or one of
    this module's two refusal verdicts (``CONFIDENCE_UNKNOWN`` /
    ``CONFIDENCE_UNRECOGNISED``) when the artifact did not supply a usable one.

    NEVER returns ``ok`` on inference — ``ok`` is only ever the producer's own
    word, or derived from the artifact's OWN declared
    ``min_dates_for_inference`` floor for a pre-I7542 artifact. An old artifact
    read as confident is the same class of defect as the one I7549 fixes.

    Shared, arm-agnostic, and applied by name: any FUTURE arm scored off this
    artifact routes through here rather than growing a second thinness test
    (issue I7549 deliverable 2).
    """
    declared = row.get("confidence")
    if isinstance(declared, str):
        if declared in LEADERBOARD_CONFIDENCE_VOCABULARY:
            return declared
        # A producer that changed its vocabulary is a reason to stop, not to
        # guess which side of the floor an unknown word falls on.
        logger.warning(
            "[champion_promotion] leaderboard row %r carries an unrecognised "
            "confidence %r (known: %s) — refusing to score it",
            row.get("name"), declared, LEADERBOARD_CONFIDENCE_VOCABULARY,
        )
        return CONFIDENCE_UNRECOGNISED
    # Pre-I7542 artifact: derive from the artifact's own declared floor.
    floor = leaderboard.get("min_dates_for_inference")
    n_scored = row.get("n_dates_scored")
    if isinstance(floor, bool) or not isinstance(floor, int) or floor < 1:
        return CONFIDENCE_UNKNOWN
    if isinstance(n_scored, bool) or not isinstance(n_scored, int) or n_scored < 0:
        return CONFIDENCE_UNKNOWN
    if n_scored == 0:
        return CONFIDENCE_INSUFFICIENT
    if n_scored < floor:
        return CONFIDENCE_THIN
    return CONFIDENCE_OK


# Confidence verdict -> the blocked_by slug the gate reports when it declines
# to score a row carrying that verdict. ``ok`` is the only verdict absent from
# this map: it is the only one that scores.
_CONFIDENCE_BLOCK_SLUGS: dict[str, str] = {
    # Nothing scored at all — go look at the producer.
    CONFIDENCE_INSUFFICIENT: "thinktank_coverage_no_resolved_outcomes",
    # Scored, but below the slot's evidence floor — wait for the cohort.
    CONFIDENCE_THIN: "thinktank_coverage_thin_evidence",
    # The artifact could not tell us, either way.
    CONFIDENCE_UNKNOWN: "thinktank_coverage_confidence_unknown",
    CONFIDENCE_UNRECOGNISED: "thinktank_coverage_confidence_unknown",
}


def _score_thinktank_coverage(
    tt_leaderboard: dict | None, run_date: str, leaderboard_date_used: str | None,
) -> tuple[float | None, str | None, str]:
    """thinktank_coverage's weekly score: read from crucible-research's
    ``research/producer_leaderboard/{date}.json`` (see module docstring for
    the verified schema and the coverage_complete-enforced-upstream
    reasoning).

    ``leaderboard_date_used`` (alpha-engine-config-I2544) is the date of
    the ``tt_leaderboard`` artifact actually selected by
    ``find_latest_research_producer_leaderboard_date`` — the latest
    available <= ``run_date``, never the same-day exact match this
    function required before I2544. An honest staleness bound still
    applies: more than ``LEADERBOARD_STALENESS_DAYS`` calendar days older
    than ``run_date`` is treated as unavailable (this IS the semantically
    correct behavior, not a compromise — the gate scores realized outcomes
    of PRIOR weeks' selections, which a same-day leaderboard could not
    contain anyway; see module docstring).

    Returns ``(score, unavailable_slug, confidence_verdict)``. The third
    element (alpha-engine-config-I7549) is ALWAYS populated — including on
    every refusal path — so the audit record can name the evidence that
    declined to decide, not merely that something did
    (champion-challenger-policy.md §7.2). Paths that fail before a row is even
    reached report ``CONFIDENCE_UNKNOWN``: no row means no verdict, and that
    is itself the honest answer."""
    if not isinstance(tt_leaderboard, dict) or leaderboard_date_used is None:
        return None, "leaderboard_unavailable", CONFIDENCE_UNKNOWN
    age_days = (
        date.fromisoformat(run_date) - date.fromisoformat(leaderboard_date_used)
    ).days
    if age_days < 0:
        # Defensive: find_latest_research_producer_leaderboard_date never
        # selects a date > run_date, but a caller passing
        # leaderboard_date_used directly (bypassing that selection) must
        # never have a "future" artifact trusted as this week's evidence.
        return None, "leaderboard_unavailable", CONFIDENCE_UNKNOWN
    if age_days > LEADERBOARD_STALENESS_DAYS:
        return None, "leaderboard_stale_gt_8d", CONFIDENCE_UNKNOWN
    # alpha-engine-config-I7549: assert the horizon the gate decides on rather
    # than inheriting whichever block the producer spreads across the top
    # level. See the module docstring's HORIZON section. Tolerant of the field
    # being absent (a pre-multi-horizon artifact always meant 21) but never of
    # it DISAGREEING — an upstream primary-horizon change must surface as a
    # loud no-contest, not silently rescore the live champion gate.
    declared_horizon = tt_leaderboard.get("horizon_days")
    if declared_horizon is not None and declared_horizon != GATE_HORIZON_DAYS:
        logger.warning(
            "[champion_promotion] producer leaderboard %s declares "
            "horizon_days=%r but this gate decides at %d sessions "
            "(GATE_HORIZON_DAYS) — refusing to score, no-contest",
            leaderboard_date_used, declared_horizon, GATE_HORIZON_DAYS,
        )
        return None, "leaderboard_horizon_mismatch", CONFIDENCE_UNKNOWN
    specs = tt_leaderboard.get("specs")
    if not isinstance(specs, list):
        return None, "leaderboard_unavailable", CONFIDENCE_UNKNOWN
    row = next(
        (s for s in specs if isinstance(s, dict) and s.get("name") == "thinktank_coverage"),
        None,
    )
    if row is None:
        # Expected until crucible-research registers thinktank_coverage in
        # producers/registry.py::challenger_producers() — see module
        # docstring "KNOWN, TRACKED GAP" / alpha-engine-config-I2519. This
        # condition is now fully independent of whether a champion producer
        # is registered (alpha-engine-config-I2998 decoupled the two
        # concerns — score_leaderboard writes this row champion-free).
        return None, "thinktank_coverage_not_in_leaderboard", CONFIDENCE_UNKNOWN
    # alpha-engine-config-I7549: the evidence gate. Replaces the pre-fix
    # truthiness check on n_dates_scored, under which n_dates_scored == 1 --
    # a mean with a null SE and a null t_stat -- could move the live champion
    # pointer. Only the producer's own "ok" scores; every other verdict is a
    # no-contest that holds the pointer in BOTH directions.
    confidence = leaderboard_row_confidence(row, tt_leaderboard)
    if confidence != CONFIDENCE_OK:
        slug = _CONFIDENCE_BLOCK_SLUGS[confidence]
        # §7.2: a gate that declines to decide must SAY SO. The blocked_by
        # slug and arm_confidence land in the durable weekly audit record;
        # this log is the same fact on the run's own surface. Deliberately
        # NOT an ops alert: `thin` is a WAIT state that resolves as the
        # cohort matures, and a weekly page for a self-resolving condition
        # is the noise §7.2 exists to prevent. The dead/absent cases that
        # DO warrant a page are already covered by _publish_gate_error_alert
        # and the ARTIFACT_REGISTRY freshness monitor.
        logger.warning(
            "[champion_promotion] thinktank_coverage NOT scored this week: "
            "confidence=%s (n_dates_scored=%r, artifact "
            "min_dates_for_inference=%r, leaderboard=%s) -> blocked_by=%s. "
            "The champion pointer is unchanged; a thin arm is not evidence "
            "(champion-challenger-policy.md §5, alpha-engine-config-I7549).",
            confidence, row.get("n_dates_scored"),
            tt_leaderboard.get("min_dates_for_inference"),
            leaderboard_date_used, slug,
        )
        return None, slug, confidence
    # alpha-engine-config-I2998: direct lift vs the SPY benchmark, computed
    # champion-free — the SAME kind of statistic as
    # scanner_predictor_direct's score (mean top-N realized return lift vs
    # SPY, date-clustered), replacing the retired topn_alpha_vs_champion
    # (which required a live comparator producer and went permanently
    # unavailable once config-I2993 retired agentic_sector_teams with no
    # successor champion registered).
    alpha = row.get("topn_alpha_vs_benchmark")
    if not isinstance(alpha, dict) or alpha.get("mean") is None:
        # confidence said "ok" but the primary metric is absent — an
        # internally inconsistent row. Fail static, and keep the confidence
        # the row actually claimed so the audit shows the contradiction.
        return None, "thinktank_coverage_no_resolved_outcomes", confidence
    return float(alpha["mean"]), None, confidence


def build_weekly_arm_scores(
    e2e_lift: dict | None,
    tt_leaderboard: dict | None,
    *,
    run_date: str,
    leaderboard_date_used: str | None = None,
) -> dict:
    """Reduce this run's two evidence sources to the shape ``evaluate_gates``
    expects: ``{"scores": {"scanner_predictor_direct": float|None,
    "thinktank_coverage": float|None}, "unavailable_reasons": {arm: slug},
    "leaderboard_date_used": str|None}``. Both scores are lift-vs-the-shared
    -agentic-baseline (see module docstring's common-comparator reasoning)
    — comparable directly, no combined-variance step needed since
    winner-take-all performs no significance test.

    ``leaderboard_date_used`` (alpha-engine-config-I2544) MUST be supplied
    by the caller as the date actually selected via
    ``find_latest_research_producer_leaderboard_date`` /
    ``read_latest_research_producer_leaderboard`` — it is threaded straight
    through into the returned dict (and from there into every
    ``evaluate_gates`` outcome record) so the audit trail always shows
    which week's evidence decided (or declined to decide) a flip.

    ``arm_confidence`` (additive, alpha-engine-config-I7549) names, per arm,
    how much evidence stood behind its score this week —
    ``ok``/``thin``/``insufficient`` from the producer leaderboard row itself,
    ``unknown``/``unrecognised`` when the artifact could not say, and
    and the SAME vocabulary for ``scanner_predictor_direct``, measured against
    this repo's own ``MIN_CYCLES_FOR_INFERENCE`` floor rather than the
    leaderboard's (alpha-engine-config-I7549 champion-side half — it reads this
    repo's counterfactual, which has its own registry, not no registry).
    Threaded
    through ``evaluate_gates`` into the weekly audit record so the audit trail
    shows WHICH EVIDENCE declined to decide, not merely that something did."""
    spd_score, spd_reason, spd_confidence = _score_scanner_predictor_direct(e2e_lift)
    tt_score, tt_reason, tt_confidence = _score_thinktank_coverage(
        tt_leaderboard, run_date, leaderboard_date_used,
    )
    reasons: dict[str, str] = {}
    if spd_reason is not None:
        reasons["scanner_predictor_direct"] = spd_reason
    if tt_reason is not None:
        reasons["thinktank_coverage"] = tt_reason
    return {
        "scores": {
            "scanner_predictor_direct": spd_score,
            "thinktank_coverage": tt_score,
        },
        "unavailable_reasons": reasons,
        "arm_confidence": {
            "scanner_predictor_direct": spd_confidence,
            "thinktank_coverage": tt_confidence,
        },
        "leaderboard_date_used": leaderboard_date_used,
    }


def _publish_gate_error_alert(run_date: str, error: str) -> None:
    """Best-effort active alert when ``run_weekly_evaluation`` catches an
    exception during gate evaluation (config#2884). The weekly winner-take
    -all gate is correctly fail-STATIC on an internal error -- the pointer
    is never moved on a bad read -- but until now the ONLY liveness signal
    was the ARTIFACT_REGISTRY's file-PRESENCE SLA on the audit JSON, which
    a weekly ``outcome="error"`` write satisfies indefinitely. Without an
    active alert, a persistent bug could freeze ``config/producer_champion
    .json`` on a stale/losing arm for an unbounded number of weeks,
    discoverable only by manually reading each week's audit record.
    Mirrors the ``_publish_executor_opt_rejection_alert`` pattern in
    ``evaluate.py``. Never raises -- alerting must not crash the run this
    gate error already threatened to interrupt.
    """
    try:
        from ops_alerts import publish_ops_alert
    except ImportError as e:
        logger.warning(
            "[champion_promotion] gate-error alert skipped — ops_alerts "
            "unavailable: %s", e,
        )
        return
    message = (
        f"champion_promotion gate evaluation raised on {run_date}: {error}. "
        f"config/producer_champion.json was NOT re-evaluated this week "
        f"(fail-static — pointer unchanged). See "
        f"config/apply_audit/producer_champion/{run_date}.json (outcome=error)."
    )
    try:
        publish_ops_alert(
            message,
            severity="error",
            source="alpha-engine-backtester/optimizer/champion_promotion.py::run_weekly_evaluation",
            dedup_key=f"champion_promotion_gate_error_{run_date}",
            dedup_window_min=720,  # one alert per Saturday cycle, mirrors pit_parity.py
        )
    except Exception:  # noqa: BLE001 — alerting must never crash the run
        logger.exception(
            "[champion_promotion] gate-error alert publish failed (best-effort, swallowed)",
        )


def run_weekly_evaluation(
    *,
    bucket: str,
    run_date: str,
    e2e_lift: dict | None,
    tt_leaderboard: dict | None,
    tt_leaderboard_date_used: str | None = None,
    freeze: bool,
    upload: bool,
    s3_client=None,
) -> dict:
    """Top-level entry point wired into evaluate.py. Runs the weekly
    winner-take-all decision and writes both artifacts:

      1. The weekly audit record (config/apply_audit/producer_champion/
         {date}.json + latest.json) — ALWAYS written, any outcome.
      2. The champion pointer (config/producer_champion.json) — written ONLY
         on outcome="promoted" AND not freeze. A no-contest,
         unchanged-winner-already-champion, held_shadow_only, or frozen run
         never touches the pointer — idempotent, bidirectional-safe.
         ``held_shadow_only`` (alpha-engine-config-I2515, Brian's 2026-08-20
         ruling) is a shadow-only arm winning on score: the pointer is held
         and the win is recorded as ``counterfactual_winner``.

    ``e2e_lift`` is the ``diagnostics["e2e_lift"]`` dict already computed
    earlier in the same evaluate.py run (scanner_predictor_direct's
    evidence). ``tt_leaderboard`` is the parsed
    ``research/producer_leaderboard/{date}.json`` artifact for
    ``tt_leaderboard_date_used`` (thinktank_coverage's evidence), read from
    crucible-research via ``read_latest_research_producer_leaderboard`` —
    the LATEST artifact available <= ``run_date`` (alpha-engine-config
    -I2544: the writer is now an async advisory child SF that may not have
    finished/may have failed by the time this gate runs; a same-day exact
    match is no longer assumed). ``tt_leaderboard_date_used`` is the date
    of that selected artifact (None if none was found <= run_date) —
    threaded through to the audit record's ``leaderboard_date_used`` field
    so the audit trail always shows which week's evidence decided a flip.
    Either evidence source being unavailable, or the selected leaderboard
    being more than ``LEADERBOARD_STALENESS_DAYS`` days stale, degrades to
    a no-contest week for that arm (never an ``error`` outcome by itself)
    via ``build_weekly_arm_scores``; only an exception raised during
    evaluation itself produces ``outcome="error"``.

    Returns the audit record that was built (and, for callers that want it,
    it also carries the pointer dict under ``_pointer_write`` when a write
    happened — internal to evaluate.py wiring, not part of the frozen
    audit schema).
    """
    pointer = read_champion_pointer(bucket, s3_client=s3_client)
    champion_before = _normalize_champion_before(
        (pointer or {}).get("champion", VALID_CHAMPIONS[0])
    )

    gate_result = None
    error = None
    try:
        arm_scores = build_weekly_arm_scores(
            e2e_lift, tt_leaderboard, run_date=run_date,
            leaderboard_date_used=tt_leaderboard_date_used,
        )
        # alpha-engine-config-I3165: probe the CHALLENGER's declared feed
        # dependencies for producer liveness before deciding the gate — a
        # challenger that would otherwise win must not be promoted onto a
        # dead/orphaned feed (config#3053). check_feed_dependencies_live
        # never raises (any probe failure degrades to the
        # "feed_producer_dead" slug), so this call cannot itself turn a
        # normal week into an error outcome. Only probed when a promotion
        # is even POSSIBLE this week (both scores present and the
        # challenger's is strictly higher) — mirrors evaluate_gates' own
        # win condition so a no-contest/defended-incumbent week never pays
        # for an S3 read + parquet parse whose result couldn't change the
        # outcome either way.
        challenger = _other_champion(champion_before)
        scores = arm_scores.get("scores", {})
        challenger_would_win = (
            scores.get(champion_before) is not None
            and scores.get(challenger) is not None
            and scores[challenger] > scores[champion_before]
        )
        # alpha-engine-config-I2515: a shadow-only challenger's win is
        # vetoed by evaluate_gates regardless of feed liveness, so the
        # probe's S3 read + parquet parse could not change the outcome —
        # skipped for the same reason a non-winning challenger's is.
        feed_blocked_slug = (
            check_feed_dependencies_live(
                challenger, bucket=bucket, run_date=run_date, s3_client=s3_client,
            )
            if challenger_would_win and not is_shadow_only(challenger) else None
        )
        gate_result = evaluate_gates(
            champion_before=champion_before,
            arm_scores=arm_scores,
            freeze=freeze,
            feed_blocked_slug=feed_blocked_slug,
        )
    except Exception as e:  # noqa: BLE001 — gate evaluation must never
        # crash the weekly evaluate run; record as an error outcome
        # (still written, per the liveness posture) and move on.
        logger.exception("Champion-promotion gate evaluation raised")
        error = str(e)
        _publish_gate_error_alert(run_date, error)

    audit = build_champion_audit(run_date, gate_result, freeze=freeze, error=error)

    pointer_written = None
    if gate_result is not None and gate_result["outcome"] == "promoted" and not freeze:
        pointer_written = write_champion_pointer(
            bucket, gate_result["champion_after"],
            promotion_source="gate_engine", upload=upload, s3_client=s3_client,
        )

    logger.info(
        "producer_champion evaluation: outcome=%s champion_before=%s champion_after=%s "
        "champion_score=%s challenger_score=%s blocked_by=%s leaderboard_date_used=%s",
        audit["outcome"], audit["champion_before"], audit["champion_after"],
        audit.get("champion_score"), audit.get("challenger_score"), audit["blocked_by"],
        audit.get("leaderboard_date_used"),
    )

    if upload:
        try:
            write_champion_audit(bucket, run_date, audit, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_champion audit S3 write failed — this is the "
                "liveness proxy, surfacing loudly",
            )
            raise
    else:
        logger.info("producer_champion audit S3 write skipped (upload=%s) — logged only", upload)

    result = dict(audit)
    if pointer_written is not None:
        result["_pointer_write"] = pointer_written
    return result


# ── research/producer_leaderboard_champion_gate/{date}.json ─────────────────
#
# This module's OWN observability artifact (config#2367) — a per-run
# snapshot of scanner_predictor_direct's realized lift vs the agentic
# baseline, appended to a running history. Under the pre-I2518 HAC engine
# this fed the significance/hysteresis gate directly; under winner-take-all
# it is NO LONGER consumed by the gate (which only needs THIS week's point,
# taken straight from ``e2e_lift`` via ``_score_scanner_predictor_direct``
# above) but is STILL MAINTAINED here for observability/history continuity
# and because config#2452 (the key-collision fix that gave this artifact its
# own distinct key, distinct from crucible-research's
# research/producer_leaderboard/{date}.json) has an open live-verification
# tail expecting this artifact to keep being written every Saturday.
#
# config#2452 (found 2026-07-13, same day as first live run post-merge): this
# key was originally `research/producer_leaderboard/{date}.json` — the SAME
# key crucible-research's `scoring/leaderboard_producers.py` already writes,
# with an incompatible schema. Renamed before any collision occurred.


LEADERBOARD_KEY_TMPL = "research/producer_leaderboard_champion_gate/{date}.json"
_LEADERBOARD_HISTORY_KEEP_WEEKS = 26  # ~6 months of weekly points

RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL = "research/producer_leaderboard/{date}.json"


def leaderboard_entry_from_e2e_lift(e2e_lift: dict | None) -> dict | None:
    """Extract this week's sector-neutral alpha point (scanner_then_predictor's
    OWN realized 21d alpha, already SPY-relative) from the e2e_lift
    diagnostic already computed earlier in the same evaluate run. Returns
    None when the counterfactual is unavailable this week
    (skipped/insufficient_data/error/missing) — an honest "no new point this
    week" rather than fabricating one.

    ``sector_neutral_mean_alpha_21d`` (alpha-engine-config-I2998) is the
    current gate's scanner_predictor_direct score source (see
    ``_score_scanner_predictor_direct``) — the arm's direct lift vs the SPY
    zero-line, gated on THIS field's presence rather than the retired
    ``sn_lift_vs_agentic_cio`` (still carried for observability, may be
    None if the agentic-CIO comparator itself is unavailable that week —
    that no longer blocks this entry from being usable).
    """
    if not isinstance(e2e_lift, dict):
        return None
    cf = e2e_lift.get("scanner_then_predictor_counterfactual")
    if not isinstance(cf, dict) or cf.get("status") != "ok":
        return None
    methods = cf.get("methods", {})
    pred = methods.get("scanner_then_predictor_topN")
    if not isinstance(pred, dict):
        return None
    sn_alpha = pred.get("sector_neutral_mean_alpha_21d")
    if sn_alpha is None:
        return None
    sn_lift = pred.get("sn_lift_vs_agentic_cio")
    return {
        "sector_neutral_mean_alpha_21d": float(sn_alpha),
        "sn_lift_vs_agentic_cio": float(sn_lift) if sn_lift is not None else None,
        "n_picks": pred.get("n_picks"),
        "n_cycles": cf.get("n_cycles"),
    }


def build_leaderboard_artifact(run_date: str, history: list[dict], new_entry: dict | None) -> dict:
    """Append ``new_entry`` (if any) to ``history`` (oldest-first list of
    ``{"date": ..., "sector_neutral_mean_alpha_21d": ..., "sn_lift_vs_agentic_cio": ...,
    "n_picks": ..., "n_cycles": ...}``), trim to the retention window, and
    return the full artifact to write to
    ``research/producer_leaderboard_champion_gate/{run_date}.json``.
    """
    points = list(history)
    if new_entry is not None:
        points = [p for p in points if p.get("date") != run_date]
        points.append({"date": run_date, **new_entry})
    points = points[-_LEADERBOARD_HISTORY_KEEP_WEEKS:]
    return {
        "schema_version": 1,
        "as_of": run_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weekly_points": points,
    }


def leaderboard_gate_inputs(artifact: dict) -> dict:
    """Reduce a leaderboard artifact to matured cohort count + weekly SN-lift
    series. Retained for API parity / observability (e.g. a future
    diagnostic reusing hac_significance) — no longer consumed by
    evaluate_gates under winner-take-all."""
    points = artifact.get("weekly_points", []) if isinstance(artifact, dict) else []
    lifts = [p["sn_lift_vs_agentic_cio"] for p in points if p.get("sn_lift_vs_agentic_cio") is not None]
    return {
        "challenger_matured_cohorts": len(lifts),
        "challenger_weekly_sn_lift": lifts,
    }


def read_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    s3 = s3_client or boto3.client("s3")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    except Exception:
        return None


def read_prior_leaderboard_history(bucket: str, run_date: str, s3_client=None) -> list[dict]:
    """Read the most recent leaderboard artifact available (by scanning
    backward from ``run_date`` — NOT wall-clock today, so a ``--date``
    backfill run seeds history relative to the backfilled trading day, not
    the day the backfill happens to execute) to seed ``weekly_points``
    history."""
    s3 = s3_client or boto3.client("s3")
    from datetime import date as _date, timedelta

    anchor = _date.fromisoformat(run_date)
    for back in range(1, 15):
        probe_date = (anchor - timedelta(days=back)).isoformat()
        key = LEADERBOARD_KEY_TMPL.format(date=probe_date)
        try:
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read())
            return list(data.get("weekly_points", []))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                continue
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
        except Exception as e:  # noqa: BLE001
            logger.warning("Leaderboard history probe failed at %s: %s", key, e)
            return []
    return []


def write_leaderboard(bucket: str, run_date: str, artifact: dict, s3_client=None) -> str:
    s3 = s3_client or boto3.client("s3")
    body = json.dumps(artifact, indent=2, allow_nan=False).encode("utf-8")
    key = LEADERBOARD_KEY_TMPL.format(date=run_date)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    logger.info("producer_leaderboard written: s3://%s/%s (%d weekly points)", bucket, key, len(artifact.get("weekly_points", [])))
    return key


def update_leaderboard_and_get_gate_inputs(
    bucket: str, run_date: str, e2e_lift: dict | None, *, upload: bool, s3_client=None,
) -> dict:
    """Full leaderboard maintenance step: read prior history, append this
    week's point (if the counterfactual matured this run), write the
    updated artifact, and return the gate-ready reduction. Called once per
    evaluate run, BEFORE ``run_weekly_evaluation`` — maintained for
    observability / config#2452 continuity (see module docstring); its
    return value is no longer fed into the gate decision.
    """
    history = read_prior_leaderboard_history(bucket, run_date, s3_client=s3_client) if upload else []
    new_entry = leaderboard_entry_from_e2e_lift(e2e_lift)
    artifact = build_leaderboard_artifact(run_date, history, new_entry)
    if upload:
        try:
            write_leaderboard(bucket, run_date, artifact, s3_client=s3_client)
        except Exception:
            logger.exception(
                "producer_leaderboard write failed — this observability "
                "artifact will be missing this week (non-fatal to the gate, "
                "which no longer depends on it)",
            )
    return leaderboard_gate_inputs(artifact)


def read_research_producer_leaderboard(bucket: str, run_date: str, s3_client=None) -> dict | None:
    """Read crucible-research's REAL champion/challenger producer leaderboard
    (``scoring/leaderboard_producers.py::build_producer_leaderboard``,
    config#1221/#1223) for ``run_date`` — the evidence source for
    thinktank_coverage's weekly score (see module docstring). Distinct key
    from this module's OWN ``research/producer_leaderboard_champion_gate/
    {date}.json`` (config#2452 collision fix) — this function only READS
    the crucible-research-owned artifact, never writes it.

    Returns None on 404/NoSuchKey (not yet written this week — e.g. before
    the Saturday eval_rolling_mean Lambda step runs, or any week
    crucible-research's build fails) or any other read/parse failure
    (logged) — a missing/malformed leaderboard degrades to a no-contest week
    for thinktank_coverage (``_score_thinktank_coverage``), never a crash
    and never a fabricated score."""
    s3 = s3_client or boto3.client("s3")
    key = RESEARCH_PRODUCER_LEADERBOARD_KEY_TMPL.format(date=run_date)
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            logger.info(
                "No crucible-research producer_leaderboard at s3://%s/%s "
                "(not yet written this week)", bucket, key,
            )
        else:
            logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None
    except Exception as e:  # noqa: BLE001
        logger.warning("crucible-research producer_leaderboard read failed (%s)", e)
        return None


def find_latest_research_producer_leaderboard_date(
    bucket: str, run_date: str, s3_client=None,
) -> str | None:
    """List the ``research/producer_leaderboard/`` prefix (crucible-research
    -owned, config#1221/#1223) and return the latest well-formed date <=
    ``run_date`` found among its dated keys, or None if none exist at or
    before ``run_date``.

    alpha-engine-config-I2544 (2026-07-14 ruling): the ASYNC advisory child
    SF that now writes this artifact may not have finished — or may have
    failed outright — by the time this Evaluator-stage gate runs in the
    MAIN weekly SF, so an exact same-day key read is no longer a safe
    assumption. Reading the LATEST AVAILABLE leaderboard <= ``run_date`` is
    the semantically CORRECT read, not a compromise: the gate scores
    realized (matured) outcomes of PRIOR weeks' ``thinktank_coverage``
    selections — a same-day leaderboard could not contain resolved
    outcomes for same-day picks even if it existed on time.

    A single ``list_objects_v2`` call (no pagination) is sufficient: this
    prefix is written at most weekly, so even several years of history
    stays far under the 1000-key single-page ceiling — mirrors
    ``factor_blend_optimizer._read_recent_shadow_archives``'s identical
    single-call reasoning for its own weekly-cadence prefix. Any key under
    the prefix that doesn't match the ``{date}.json`` shape (e.g. a future
    ``latest.json`` sidecar some other consumer adds) is silently skipped,
    never crashes the scan. A list failure (ClientError or otherwise) is
    logged and treated as "nothing available" — degrades to a no-contest
    week downstream, never a crash.
    """
    s3 = s3_client or boto3.client("s3")
    try:
        resp = s3.list_objects_v2(Bucket=bucket, Prefix=RESEARCH_PRODUCER_LEADERBOARD_PREFIX)
    except ClientError as e:
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None
    except Exception as e:  # noqa: BLE001 — list must never crash the gate
        logger.warning(
            "research/producer_leaderboard/ list failed (%s) — treating as "
            "no leaderboard available", e,
        )
        return None

    anchor = date.fromisoformat(run_date)
    best: date | None = None
    for obj in resp.get("Contents") or []:
        m = _RESEARCH_PRODUCER_LEADERBOARD_KEY_RE.match(obj.get("Key", ""))
        if not m:
            continue
        candidate = date.fromisoformat(m.group(1))
        if candidate <= anchor and (best is None or candidate > best):
            best = candidate
    if best is None:
        logger.info(
            "No research/producer_leaderboard/ artifact <= %s found under "
            "s3://%s/%s", run_date, bucket, RESEARCH_PRODUCER_LEADERBOARD_PREFIX,
        )
        return None
    return best.isoformat()


def read_latest_research_producer_leaderboard(
    bucket: str, run_date: str, s3_client=None,
) -> tuple[dict | None, str | None]:
    """Combined list-then-read: find the latest
    ``research/producer_leaderboard/{date}.json`` <= ``run_date``
    (``find_latest_research_producer_leaderboard_date``) and read it
    (``read_research_producer_leaderboard``, reused unchanged — it is
    still the correct exact-date-read primitive once the date to read has
    been selected).

    THE production entry point for thinktank_coverage's evidence as of
    alpha-engine-config-I2544 — supersedes calling
    ``read_research_producer_leaderboard`` directly with ``run_date`` (an
    exact-match read that assumed same-day availability the async advisory
    child SF can no longer guarantee).

    Returns ``(leaderboard_dict, leaderboard_date_used)``: ``(None, None)``
    when no artifact <= ``run_date`` exists yet (or the list itself
    failed); ``(None, None)`` also when a date was found but the S3 read
    then failed (``read_research_producer_leaderboard`` already logs the
    specifics) — never a partial/inconsistent pairing of a leaderboard
    with the wrong date or a date with no leaderboard.
    """
    latest_date = find_latest_research_producer_leaderboard_date(
        bucket, run_date, s3_client=s3_client,
    )
    if latest_date is None:
        return None, None
    leaderboard = read_research_producer_leaderboard(bucket, latest_date, s3_client=s3_client)
    if leaderboard is None:
        return None, None
    return leaderboard, latest_date
