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

_SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "infrastructure" / "spot_backtest.sh"

_ASSIGN = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=")
# An unescaped $VAR / ${VAR}. The negative lookbehind is the whole point:
# `\$VAR` is correct and must not be flagged.
_UNESCAPED_REF = re.compile(r"(?<!\\)\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")
# `run_ssm <slug> <timeout> <<DELIM` — unquoted delimiter, so the body expands
# locally. A quoted delimiter (<<'DELIM') suppresses expansion entirely and is
# not affected.
_HEREDOC_OPEN = re.compile(r"<<([A-Za-z_][A-Za-z0-9_]*)\s*$")


def _lines() -> list[str]:
    return _SCRIPT.read_text().split("\n")


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


def test_the_script_and_its_heredocs_are_discoverable():
    """A regex that matched nothing would make every assertion below vacuous."""
    assert _SCRIPT.is_file(), f"{_SCRIPT} missing"
    heredocs = _unquoted_heredocs(_lines())
    assert heredocs, "no unquoted heredocs found — the parser is broken, not the script"


def test_no_heredoc_local_variable_is_expanded_by_the_dispatching_shell():
    """THE REGRESSION (2026-08-01).

    Un-escaping ``\\$_BACKTEST_WAS_SKIPPED`` at line 1716 makes this fail with
    that variable named — the exact condition that killed the weekly run.
    """
    lines = _lines()
    outer_assigned = set()
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
        "Heredoc-local variable(s) expanded by the DISPATCHING shell — they are "
        "unset locally and abort the dispatcher under `set -u`, with bash "
        "reporting the error at the heredoc's OPENING line:\n"
        + "\n".join(offenders)
        + "\n\nEscape them as \\$VAR so they evaluate on the box. Setting them "
        "inside the heredoc does NOT help — that runs remotely, after the "
        "local expansion has already failed."
    )
