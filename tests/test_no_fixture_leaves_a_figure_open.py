"""A fixture that outlives its tests must close the figures it opened.

THE INVARIANT, in the user's terms: a test that counts the figures on screen
counts the ones it drew. Nothing another file drew half an hour earlier is
still hanging around in the count.

``plt.subplots`` and ``plt.figure`` register the figure with pyplot's
process-global manager -- ``plt.get_fignums()`` is a session-wide list, not a
per-test one. A function-scoped fixture is usually harmless because the next
module's ``plt.close("all")`` sweeps up after it. A **module**- or
**session**-scoped one is not: it outlives the file that declared it, so its
figure stays in the global count for the rest of the run.

That is a real, diagnosed failure, not a hypothetical. ``_pdf_assets`` in
``tests/qt/test_figure_queue.py`` built one 3x3 figure per module and returned
it without closing it. Three hundred tests later,
``test_cov_object_organelle_sam.py::test_plot_true_renders_a_real_figure``
asserted ``len(plt.get_fignums()) == 1`` and got 2 -- reported against a file
that has nothing to do with figure queues, and passing whenever it happened to
run first.

Checked by reading the source rather than by running every module in its own
session: the leak is a property of how the fixture is written -- returning
instead of yielding-and-closing -- and reading it takes a second where running
them takes minutes.
"""
from __future__ import annotations

import ast
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent

#: Scopes that outlive the file that declared them.
LONG_SCOPES = {"module", "package", "session"}

#: The pyplot calls that put a figure into the global manager. ``Figure()``
#: constructed directly is deliberately NOT here: it is never registered, so
#: it never shows up in ``get_fignums`` and never needs closing.
REGISTERING_CALLS = {"figure", "subplots", "subplot_mosaic"}


def _fixture_scope(decorator) -> str | None:
    """The scope of a ``@pytest.fixture`` decorator, or None if not one."""
    target = decorator.func if isinstance(decorator, ast.Call) else decorator
    name = getattr(target, "attr", getattr(target, "id", None))
    if name != "fixture":
        return None
    if not isinstance(decorator, ast.Call):
        return "function"
    for keyword in decorator.keywords:
        if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
            return str(keyword.value.value)
    return "function"


def _registers_a_figure(node) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr in REGISTERING_CALLS
        and getattr(child.func.value, "id", "") in ("plt", "pyplot", "plot")
        for child in ast.walk(node))


def _closes_a_figure(node) -> bool:
    return any(
        isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "close"
        for child in ast.walk(node))


def _yields(node) -> bool:
    return any(isinstance(child, (ast.Yield, ast.YieldFrom))
               for child in ast.walk(node))


def _offending_fixtures():
    """``(path, fixture name, why)`` for every long-lived figure leak."""
    offenders = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if not (path.name.startswith("test_") or path.name == "conftest.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                       # not ours to police
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            scopes = {_fixture_scope(d) for d in node.decorator_list}
            if not scopes & LONG_SCOPES:
                continue
            if not _registers_a_figure(node):
                continue
            if not _yields(node):
                offenders.append((path, node.name,
                                  "returns the figure instead of yielding it, "
                                  "so it never gets a teardown"))
            elif not _closes_a_figure(node):
                offenders.append((path, node.name,
                                  "yields but never calls plt.close"))
    return offenders


def test_the_scan_can_see_the_fixtures_it_is_meant_to_police():
    """A guard that matches nothing guards nothing.

    There is at least one long-lived fixture building a pyplot figure in this
    suite. If that stops being true the check above has gone blind -- most
    likely because the fixtures moved, or because ``plt`` got imported under
    another name -- and it should be noticed here rather than by the next
    figure-counting test to draw the short straw.
    """
    seen = []
    for path in sorted(TESTS_ROOT.rglob("*.py")):
        if not (path.name.startswith("test_") or path.name == "conftest.py"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and {_fixture_scope(d) for d in node.decorator_list} \
                    & LONG_SCOPES and _registers_a_figure(node):
                seen.append(f"{path.name}::{node.name}")
    assert seen, (
        "no module- or session-scoped fixture in this suite builds a pyplot "
        "figure any more; either that is genuinely true, or this scan can no "
        "longer recognise one")


def test_no_long_lived_fixture_leaves_its_figure_in_the_global_count():
    """Module- and session-scoped figure fixtures close what they opened."""
    offenders = _offending_fixtures()
    assert not offenders, (
        "these fixtures outlive their file and leave a figure registered with "
        "pyplot for the rest of the session, so every later test that counts "
        "plt.get_fignums() counts it too:\n"
        + "\n".join(f"  {path.relative_to(TESTS_ROOT.parent)}::{name} -- {why}"
                    for path, name, why in offenders))
