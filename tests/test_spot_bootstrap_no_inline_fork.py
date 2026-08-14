"""No shell file in this repo may bootstrap a spot instance inline.

## Why this test exists

This repo carried the fleet's MOST diverged copy of the spot bootstrap
(alpha-engine-config-I7372, I6922). Every function in
``infrastructure/_spot_common.sh`` is prefixed ``spot_common_*``, so no other
repo's function names transfer, and the fleet's original sweep for the fork —
``grep -rl ec2-spot-watchdog`` — structurally could not find it: this fork
never adopted the unit that grep anchored on. It carried the hard runtime cap
(``systemd-run --on-active``) and NO SSM-liveness watchdog at all, the exact
inverse of ``nousergon-data`` / ``crucible-predictor``, which carried the
watchdog and no cap. Each fork was uncovered against the other's failure mode,
and a name-based search could see neither.

So this test is **derived, never enumerated**. It names no file, no function
and no shell literal. It asks the canonical renderer's own classifier what the
tree contains, which is the only form of the question a renamed, relocated or
re-prefixed copy cannot dodge.

## The three assertions

1. ``scan_for_inline_bootstraps`` finds nothing — the behavioural classifier
   from ``krepis.spot_bootstrap``, the same module the launchers call at
   runtime.
2. No shell file selects its interpreter with a silent fallback
   (``command -v python3.12 ... || PYTHON_BIN=python3``). ``requirements.txt``
   is resolved against 3.12; the AMI's ``python3`` resolves different wheels,
   and the divergence surfaces as an ImportError deep inside a workload rather
   than at the step that chose wrong. This is assertion (1)'s blind spot: the
   scanner needs two signature CATEGORIES to fire, and a lone interpreter
   selection is one.
3. No shell file clones into ``/home/ec2-user/`` directly. This is assertion
   (1)'s OTHER blind spot, and the sharper one:
   ``scan_for_inline_bootstraps`` clears an ENTIRE FILE on a single
   ``-m krepis.spot_bootstrap`` match, before evaluating any signature
   (``if _DELEGATES.search(code): continue``). ``_spot_common.sh`` is ~900
   lines and now contains exactly such a delegate call, so a second inline
   bootstrap appended to the same file would scan clean forever. Tracked as a
   krepis fix in ``alpha-engine-config-I7378`` (regional rather than
   file-level clearing); until it lands, assertion (3) is what actually holds
   the line in THIS repo, whose two near-identical twin launchers are the most
   likely source of exactly that regression.

Assertion (1) alone would therefore be a detector reporting green while
measuring less than it appears to. Saying so here is cheaper than discovering
it from a weekly run.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Same subdirs and suffixes the canonical scanner walks, so assertions (2) and
# (3) cover exactly the surface assertion (1) does — no more, no less.
_SUBDIRS = ("infrastructure", "scripts", "bin")
_SUFFIXES = (".sh", ".bash")

# `|| <NAME>=python3` (optionally quoted, optionally `python3 -m pip`), but NOT
# `python3.12` and NOT `|| { echo ...; exit 1; }`. The negative lookahead on
# `[.\d]` is what separates the fallback from the strict form.
_SILENT_FALLBACK = re.compile(
    r"\|\|\s*[A-Za-z_][A-Za-z0-9_]*=\"?python3(?![.\d])"
)

# A clone landing anywhere under the spot's home directory. The renderer emits
# these from launcher-side literals; a shell file emitting one itself is an
# inline checkout by definition.
_HOME_CLONE = re.compile(r"\bgit\s+clone\b[^\n]*/home/ec2-user/")


def _shell_files() -> list[Path]:
    files: list[Path] = []
    for subdir in _SUBDIRS:
        base = REPO_ROOT / subdir
        if not base.is_dir():
            continue
        files.extend(
            p
            for p in sorted(base.rglob("*"))
            if p.is_file() and p.suffix in _SUFFIXES
        )
    return files


def _uncommented(text: str) -> list[tuple[int, str]]:
    """Non-comment, non-blank lines with their 1-based numbers.

    A shell `#` comment is stripped only when it starts the line (after
    whitespace). An inline `#` inside a quoted string is not a comment and
    guessing at one would silence a real finding.
    """
    out: list[tuple[int, str]] = []
    for n, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append((n, line))
    return out


def test_no_inline_spot_bootstrap_anywhere_in_the_tree():
    """Assertion (1): the canonical classifier finds no inline bootstrap.

    Behavioural, not name-based: a copy under a new filename, in a new
    directory, behind a new function prefix still matches, because what it
    DOES is unchanged. That is the property the `grep -rl ec2-spot-watchdog`
    sweep did not have, and the reason this repo's fork survived it.
    """
    scan = pytest.importorskip(
        "krepis.spot_bootstrap",
        reason="krepis>=0.59.6 provides the canonical inline-bootstrap scanner",
    )
    if not hasattr(scan, "scan_for_inline_bootstraps"):
        pytest.fail(
            "krepis.spot_bootstrap has no scan_for_inline_bootstraps — the "
            "installed krepis is below the >=0.59.6 floor requirements.txt "
            "declares. An older krepis makes this test silently unable to "
            "measure, which is not the same as finding nothing."
        )

    findings = scan.scan_for_inline_bootstraps(REPO_ROOT)
    assert not findings, (
        "Inline spot bootstrap(s) reappeared — collapse them onto "
        "`python -m krepis.spot_bootstrap render` (alpha-engine-config-I7372):\n"
        + "\n".join(f"  {f}" for f in findings)
    )


def test_no_silent_interpreter_fallback_in_any_shell_file():
    """Assertion (2): no `command -v python3.12 ... || VAR=python3`.

    The fallback was never confined to the bootstrap. In this repo it also sat
    inside the single-quoted one-line ``ENV_SOURCE`` that every downstream SSM
    step interpolates, so removing it from the bootstrap alone would have left
    the whole run free to resolve the AMI python3 against wheels installed for
    3.12.
    """
    offenders: list[str] = []
    for path in _shell_files():
        for n, line in _uncommented(path.read_text(encoding="utf-8")):
            if "python3.12" in line and _SILENT_FALLBACK.search(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"  {rel}:{n}: {line.strip()[:160]}")
    assert not offenders, (
        "Silent interpreter fallback to the AMI python3 reappeared "
        "(alpha-engine-config-I7372). requirements.txt resolves against "
        "python3.12; python3 resolves different wheels. Resolve strictly — "
        "`command -v python3.12 >/dev/null || { echo FATAL >&2; exit 1; }` — "
        "or let `krepis.spot_bootstrap` render it:\n" + "\n".join(offenders)
    )


def test_no_shell_file_clones_into_the_spot_home_directory():
    """Assertion (3): no `git clone ... /home/ec2-user/...` in a shell file.

    Deliberately independent of assertion (1), which cannot see this once a
    file contains any `-m krepis.spot_bootstrap` call: the scanner clears the
    whole file on that one match (alpha-engine-config-I7378). Every legitimate
    checkout on a spot is now a `--checkout` or `--extra-clone` argument to the
    renderer, so a raw clone line is always a regression.
    """
    offenders: list[str] = []
    for path in _shell_files():
        for n, line in _uncommented(path.read_text(encoding="utf-8")):
            if _HOME_CLONE.search(line):
                rel = path.relative_to(REPO_ROOT)
                offenders.append(f"  {rel}:{n}: {line.strip()[:160]}")
    assert not offenders, (
        "A raw `git clone` into /home/ec2-user/ reappeared. Spot checkouts "
        "belong in `krepis.spot_bootstrap render --checkout/--extra-clone`, "
        "where the URL and branch are launcher-side literals rather than "
        "shell variables the remote may expand to an empty string "
        "(crucible-predictor#463):\n" + "\n".join(offenders)
    )
