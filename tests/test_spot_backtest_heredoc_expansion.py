"""Heredoc-local variables must not be expanded by the *dispatching* shell.

**The outage this exists to prevent (2026-08-01).** The weekly SF reached
Backtester for the first time in over a week and died immediately::

    infrastructure/spot_backtest.sh: line 1459: _BACKTEST_WAS_SKIPPED: unbound variable

`run_ssm "backtest" ... <<BACKTEST` (line 1459) opens an **unquoted** heredoc,
so the local shell performs parameter expansion on the whole body before the
text is ever sent to the box. Line 1716 referenced ``"$_BACKTEST_WAS_SKIPPED"``
unescaped. That variable is assigned only *inside* the heredoc (lines 1501,
1512) — i.e. on the remote box — so locally it is unset, and the dispatcher
dies under ``set -u``.

Two things make this expensive to diagnose by eye:

1. **Bash reports heredoc-expansion errors at the line where the heredoc
   OPENS**, not where the offending reference sits. The error said 1459; the
   bug was at 1716, 257 lines away.
2. **A previous fix attempt sits at line 1463, inside the heredoc**::

       # MUST precede set -euo pipefail — any code path ... that references
       # this variable before its main init ... will trigger an
       # unbound-variable fatal exit under set -u.
       export _BACKTEST_WAS_SKIPPED=false

   The comment identifies the class correctly and the fix is in the **wrong
   shell**: it runs on the box, and cannot prevent an expansion that already
   happened locally.

The distinction this test encodes: a heredoc-local variable is one **assigned
inside the body and not in the outer script**. Those must be escaped
(``\\$VAR``) so they evaluate remotely. Variables set in the outer script and
referenced unescaped are *intentional pass-through* — that is how the
dispatcher injects ``$RUN_DATE``, ``$SKIP_STAGES``, ``$PIT_PARITY_ENABLED``.
Flagging those too would make the test noise and it would be disabled.

Derived, not enumerated: a hand-maintained list of "variables to escape" would
miss whichever one is added next, which is exactly how this shipped.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_INFRA = pathlib.Path(__file__).resolve().parent.parent / "infrastructure"

# Every launcher that dispatches an unquoted heredoc to a box, not just the
# monolith this test was written against. The 2026-05-31 L4472 phase-split
# copied `spot_backtest.sh`'s heredoc shape into four per-stage launchers and
# this guard stayed pinned to the original file, so the same class could ship
# again in any of them undetected (config-I7399 sweep).
_SCRIPTS = sorted(
    p for p in _INFRA.glob("spot_*.sh") if "<<" in p.read_text()
)

_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
# An unescaped $VAR / ${VAR}. The negative lookbehind is the whole point:
# `\$VAR` is correct and must not be flagged.
_UNESCAPED_REF = re.compile(r"(?<!\\)\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
# `run_ssm <slug> <timeout> <<DELIM` — unquoted delimiter, so the body expands
# locally. A quoted delimiter (<<'DELIM') suppresses expansion entirely and is
# not affected.
_HEREDOC_OPEN = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)\s*$")


def _lines(script: pathlib.Path) -> list[str]:
    return script.read_text().split("\n")


def _sourced_assignments() -> set[str]:
    """Names assigned in the shared library every launcher sources.

    Without this the detector reports each launcher's deliberate
    ``RUN_DATE="${RUN_DATE}"`` pass-through as heredoc-local, because
    ``RUN_DATE`` is assigned by ``spot_common_normalize_run_date`` in
    ``_spot_common.sh`` rather than in the launcher's own text. Those are
    intentional dispatcher-side injections, which the module docstring
    already exempts — the exemption just has to see the whole outer shell,
    not one file of it.
    """
    common = _INFRA / "_spot_common.sh"
    if not common.is_file():
        return set()
    return {
        m.group(1)
        for line in common.read_text().split("\n")
        for m in [_ASSIGN.match(line)]
        if m
    }


def _unquoted_heredocs(lines: list[str]) -> list[tuple[str, int, int]]:
    """[(delimiter, open_lineno, terminator_lineno)] — 1-indexed."""
    out: list[tuple[str, int, int]] = []
    i = 0
    while i < len(lines):
        m = _HEREDOC_OPEN.search(lines[i])
        if m:
            delim = m.group(1)
            for j in range(i + 1, len(lines)):
                if lines[j].strip() == delim:
                    out.append((delim, i + 1, j + 1))
                    i = j
                    break
        i += 1
    return out


def test_the_launcher_set_is_discoverable():
    """A glob that matched nothing would make every assertion below vacuous."""
    assert _SCRIPTS, f"no spot_*.sh launchers with heredocs found under {_INFRA}"
    names = {p.name for p in _SCRIPTS}
    for required in (
        "spot_backtest.sh",
        "spot_backtester.sh",
        "spot_predictor_backtest.sh",
        "spot_portfolio_optimizer_backtest.sh",
        "spot_evaluator.sh",
    ):
        assert required in names, f"{required} dropped out of the guarded set"


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_the_script_and_its_heredocs_are_discoverable(script):
    """A regex that matched nothing would make every assertion below vacuous."""
    assert script.is_file(), f"{script} missing"
    heredocs = _unquoted_heredocs(_lines(script))
    assert heredocs, f"no unquoted heredocs found in {script.name} — the parser is broken, not the script"


@pytest.mark.parametrize("script", _SCRIPTS, ids=lambda p: p.name)
def test_no_heredoc_local_variable_is_expanded_by_the_dispatching_shell(script):
    """THE REGRESSION (2026-08-01).

    Un-escaping ``\\$_BACKTEST_WAS_SKIPPED`` at line 1716 makes this fail with
    that variable named — the exact condition that killed the weekly run.
    """
    lines = _lines(script)
    outer_assigned = _sourced_assignments()
    heredocs = _unquoted_heredocs(lines)
    body_spans = [(o, t) for _, o, t in heredocs]

    def _in_a_body(n: int) -> bool:
        return any(o < n < t for o, t in body_spans)

    for n, line in enumerate(lines, start=1):
        if not _in_a_body(n):
            m = _ASSIGN.match(line)
            if m:
                outer_assigned.add(m.group(1))

    offenders: list[str] = []
    for delim, open_no, term_no in heredocs:
        body = lines[open_no : term_no - 1]
        local_assigned = {
            m.group(1) for l in body for m in [_ASSIGN.match(l)] if m
        }
        for offset, line in enumerate(body):
            lineno = open_no + 1 + offset
            for ref in _UNESCAPED_REF.finditer(line):
                var = ref.group(1)
                if var in local_assigned and var not in outer_assigned:
                    offenders.append(
                        f"  <<{delim} (opens line {open_no}): ${var} referenced "
                        f"unescaped at line {lineno}, but assigned ONLY inside "
                        f"the heredoc"
                    )

    assert not offenders, (
        f"{script.name}: heredoc-local variable(s) expanded by the DISPATCHING shell — they are "
        "unset locally and abort the dispatcher under `set -u`, with bash "
        "reporting the error at the heredoc's OPENING line:\n"
        + "\n".join(offenders)
        + "\n\nEscape them as \\$VAR so they evaluate on the box. Setting them "
        "inside the heredoc does NOT help — that runs remotely, after the "
        "local expansion has already failed."
    )
