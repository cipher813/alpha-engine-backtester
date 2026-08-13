"""Every Lambda entry point in this repo carries flow-doctor crash capture.

``nousergon_lib.logging.monitor_handler`` reports an unhandled exception to
flow-doctor and re-raises. It only does that for the function it decorates.

config#6920 inserted a ``_remaining_seconds`` helper into
``lambda_concordance/handler.py`` immediately above ``handler``, and the
existing ``@monitor_handler`` line stayed attached to the line below it —
which was now the helper. From 2026-08-11 the concordance Lambda's real
entry point ran unwrapped: an unhandled exception reached Lambda without
ever reaching ``fd.report``, and the decorator instead guarded a
three-line getattr that cannot raise.

Nothing failed. That is the point: this defect is invisible to every test
that exercises the handler's return value, and visible only on the day
something crashes and no flow-doctor record exists to explain it. So it
gets a structural guard over every handler in the repo, not a fix to the
one file where it was found.

Asserted by AST rather than by import: the handler modules pull in the
Lambda task layout (``sys.path`` inserts, flow-doctor YAML discovery) and
importing all three here would make this test about the environment.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent

_HANDLER_MODULES = sorted(
    p for p in _REPO_ROOT.glob("lambda_*/handler.py")
)


def test_handler_modules_were_actually_found():
    """A glob that matches nothing makes every test below vacuously pass —
    the shape this repo has already shipped once elsewhere."""
    assert len(_HANDLER_MODULES) >= 3


@pytest.mark.parametrize(
    "path", _HANDLER_MODULES, ids=lambda p: p.parent.name,
)
def test_the_entry_point_named_handler_is_the_decorated_one(path):
    tree = ast.parse(path.read_text())
    decorated = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and any(
            (d.id if isinstance(d, ast.Name) else getattr(d, "attr", None))
            == "monitor_handler"
            for d in node.decorator_list
        )
    }
    assert "handler" in decorated, (
        f"{path.relative_to(_REPO_ROOT)}: @monitor_handler is not on "
        f"`handler` (found on {sorted(decorated) or 'nothing'}). The "
        f"entry point is running without flow-doctor crash capture."
    )


@pytest.mark.parametrize(
    "path", _HANDLER_MODULES, ids=lambda p: p.parent.name,
)
def test_monitor_handler_is_not_on_a_helper(path):
    """The decorator on a helper is the tell that it slid off the entry
    point — and a helper reporting to flow-doctor misattributes the frame."""
    tree = ast.parse(path.read_text())
    misplaced = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        and node.name != "handler"
        and any(
            (d.id if isinstance(d, ast.Name) else getattr(d, "attr", None))
            == "monitor_handler"
            for d in node.decorator_list
        )
    ]
    assert misplaced == [], (
        f"{path.relative_to(_REPO_ROOT)}: @monitor_handler on {misplaced}"
    )
