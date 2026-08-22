"""
Stage-coverage `--run-date` contract guard.

alpha-engine-config-I8155: `krepis.stage_coverage` groups verdicts under
`s3://alpha-engine-research/_stage_coverage/<run_date>/<Stage>.json`, keyed
solely by `run_date`. Before this fix, every `krepis.stage_coverage assert`
call site in this repo's `infrastructure/*.sh` launchers relied on the
CLI's own `os.environ.get("RUN_DATE", "")` argparse default — which reads
`RUN_DATE` AFTER `infrastructure/_spot_common.sh::spot_common_normalize_run_date`
has rewritten it from the SF execution's calendar run_date to the NYSE
trading day. On the 2026-08-22 weekly run this produced a one-day skew:
eight verdicts landed under `_stage_coverage/2026-08-21/` (the normalized
trading day) while the SF's own execution `run_date` was 2026-08-22,
making the coverage read for the actual execution day silently incomplete.

The fix: every stage-coverage call site now passes an explicit
`--run-date "$EXECUTION_RUN_DATE"` — a carrier exported by the Step
Functions definition (nousergon-data/infrastructure/step_function.json)
that `_spot_common.sh` deliberately never normalizes (see the comment at
`spot_common_normalize_run_date` and the `EXECUTION_RUN_DATE` default in
`spot_common_init_defaults`). `RUN_DATE` itself stays normalized — it is
still load-bearing for artifact keys (`backtest/{trading_day}/...`,
`parity/$RUN_DATE/...`) — this test only forbids `RUN_DATE` from reaching
the stage-coverage CLI, which is the exact carrier the regression used.

This test enumerates `infrastructure/*.sh` rather than hardcoding the
known 8 launcher scripts, so a new launcher script is covered
automatically without a test edit.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INFRA_DIR = REPO_ROOT / "infrastructure"

# Matches an actual `-m krepis.stage_coverage assert ...` CODE invocation
# (these are all single-line invocations, preceded by `"$LIB_PYTHON" -m`
# and followed by the `|| echo "WARNING: ..."` observe-mode guard). Requires
# the `-m` module-invocation flag immediately before the module name so
# that prose mentioning "krepis.stage_coverage assert" inside a `#` comment
# (e.g. explaining the contract, as this file's own header and
# _spot_common.sh's comments do) is never mistaken for a call site.
_STAGE_COVERAGE_ASSERT_RE = re.compile(
    r"-m\s+krepis\.stage_coverage\s+assert\b[^\n]*"
)


def _enumerate_stage_coverage_invocations() -> dict[str, list[str]]:
    """Return {relative_path: [invocation lines]} for every
    `krepis.stage_coverage assert` call site under infrastructure/*.sh.
    Comment lines (stripped line starts with `#`) are excluded so prose
    referencing the CLI in passing is never counted as a call site."""
    invocations: dict[str, list[str]] = {}
    for script in sorted(INFRA_DIR.glob("*.sh")):
        text = script.read_text(encoding="utf-8", errors="ignore")
        code_lines = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("#")
        )
        matches = _STAGE_COVERAGE_ASSERT_RE.findall(code_lines)
        if matches:
            invocations[str(script.relative_to(REPO_ROOT))] = matches
    return invocations


def test_scan_finds_stage_coverage_invocations():
    """Guard-the-guard: fail loud if the scan itself finds nothing — a scan
    that finds zero invocations would make every assertion below pass
    vacuously and silently stop covering this contract."""
    invocations = _enumerate_stage_coverage_invocations()
    total = sum(len(v) for v in invocations.values())
    assert total > 0, (
        "Scan for `krepis.stage_coverage assert` invocations under "
        f"{INFRA_DIR} found none. Either the scan regex/glob is broken, or "
        "every call site was removed — either way this test can no longer "
        "verify the --run-date contract and must be investigated before "
        "being trusted again."
    )
    # Measured fact at authorship time (alpha-engine-config-I8155): 8
    # launcher scripts each carry exactly 1 call site. Not asserted as an
    # exact count here (new launchers are expected to add more) — only
    # that the scan is finding a plausible number, not silently truncating.
    assert total >= 8, (
        f"Expected at least 8 `krepis.stage_coverage assert` call sites "
        f"(measured baseline), found {total}. Investigate before trusting "
        "this test's other assertions."
    )


def test_every_stage_coverage_invocation_passes_explicit_run_date():
    invocations = _enumerate_stage_coverage_invocations()
    missing = []
    for path, lines in sorted(invocations.items()):
        for line in lines:
            if "--run-date" not in line:
                missing.append(f"  - {path}: {line.strip()}")
    assert not missing, (
        "krepis.stage_coverage assert call site(s) missing an explicit "
        "--run-date flag:\n" + "\n".join(missing) + "\n\n"
        "Resolution: add --run-date \"$EXECUTION_RUN_DATE\" to each call "
        "site. Without it, the CLI falls back to its own RUN_DATE "
        "environment default, which infrastructure/_spot_common.sh "
        "normalizes to the NYSE trading day — a different value than the "
        "SF execution's own run_date (alpha-engine-config-I8155)."
    )


def test_no_stage_coverage_invocation_passes_run_date_variable():
    """Regression guard: `$RUN_DATE` (bare, or as `${RUN_DATE}`) must never
    be the value passed to --run-date. RUN_DATE is normalized to the NYSE
    trading day by infrastructure/_spot_common.sh::spot_common_normalize_run_date
    (deliberately, for artifact-key purposes) — passing it to
    stage_coverage reintroduces the exact skew this test exists to catch.
    The correct carrier is $EXECUTION_RUN_DATE, which _spot_common.sh
    never normalizes."""
    invocations = _enumerate_stage_coverage_invocations()
    offenders = []
    run_date_var_re = re.compile(r"--run-date[= ]\"?\$\{?RUN_DATE\}?\"?")
    for path, lines in sorted(invocations.items()):
        for line in lines:
            if run_date_var_re.search(line):
                offenders.append(f"  - {path}: {line.strip()}")
    assert not offenders, (
        "krepis.stage_coverage assert call site(s) pass the NORMALIZED "
        "$RUN_DATE instead of $EXECUTION_RUN_DATE:\n"
        + "\n".join(offenders)
        + "\n\n"
        "RUN_DATE is rewritten to the NYSE trading day by "
        "infrastructure/_spot_common.sh::spot_common_normalize_run_date "
        "and must stay that way for artifact-key purposes — but that makes "
        "it the wrong value for stage-coverage grouping, which must match "
        "the SF execution's own (unnormalized) run_date: $EXECUTION_RUN_DATE."
    )
