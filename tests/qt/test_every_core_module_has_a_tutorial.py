"""Every Core module the application ships should have a tutorial.

DERIVED FROM THE RUNTIME REGISTRY, NOT FROM A LIST. Instruction 358 asks for
exactly this and says why: a second hard-coded inventory drifts from the
first, and the failure it produces is a tutorial set that looks complete and
teaches a module structure the application no longer has.

So the Core inventory comes from `spacr.qt.app.APPS` filtered to
`SECTION_CORE` -- the same rows Home groups -- and the tutorial inventory
comes from the `_build_*_steps` functions that actually exist. Neither is
retyped here.

WHAT A TUTORIAL "COVERING" A MODULE MEANS: it navigates to it. A tutorial
that mentions a module in prose without ever opening it does not show the
reader where anything is, and the measurement that matters is the one a
reader would make by following along.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytest.importorskip("PySide6")

SCRIPTS = (pathlib.Path(__file__).resolve().parents[2]
           / "spacr" / "qt" / "tutorial" / "scripts.py")

#: Core modules with no tutorial that opens them, as measured on 2026-09-04.
#:
#: A RATCHET, NOT AN EXCUSE. It may only shrink: an entry is deleted when its
#: tutorial is written, and a Core module that appears WITHOUT a tutorial and
#: without being listed here fails the test. That is the property instruction
#: 358 asks for -- "fails when a new Core module appears without one".
#:
#: `classify_merged` WAS the interesting one and is now fixed. There was a
#: tutorial called `classify` that never opened Classify: it walked Annotate,
#: said "both open in the consolidated Classify module", and stopped at the
#: Train button. That is the stale module boundary 358 was filed about, not a
#: missing file. It opens Classify and names the five folded modules now.
#:
#: NOTE ON `train_compare`: Classify's masthead has a button that opens it,
#: and the tutorial points at that button. Pointing is not opening -- 358
#: asks that a folded action be NAMED AND LOCATED, and separately that every
#: Core module have a tutorial. Training Runs is both, so it stays here until
#: it has one of its own.
UNCOVERED_CORE_MODULES = {
    "map_barcodes",
    "regression",
    "train_compare",
    "profiler",
    "investigate_hit",
}


def _core_modules() -> dict:
    """The Core modules the running application registers."""
    from spacr.qt.app import APPS, SECTION_CORE

    return {row[0]: row[1] for row in APPS if row[3] == SECTION_CORE}


def _modules_each_tutorial_opens() -> dict:
    """``{tutorial: {module keys it navigates to}}``, read from the source.

    Read statically rather than by running the tutorials: a step's target is
    a literal in the script, and building every tutorial against a live
    window to answer "which modules does it visit" would be slow and would
    need a demo dataset present.
    """
    tree = ast.parse(SCRIPTS.read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef)
                and node.name.startswith("_build_")
                and node.name.endswith("_steps")):
            continue
        visited = set()
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Name)
                    and inner.func.id in ("_nav_to", "_sidebar_button")):
                visited.update(
                    a.value for a in inner.args
                    if isinstance(a, ast.Constant) and isinstance(a.value, str)
                )
        out[node.name[len("_build_"):-len("_steps")]] = visited
    return out


def test_the_core_inventory_is_read_from_the_registry():
    """The premise: there are Core modules, and they come from the app."""
    core = _core_modules()

    assert core, "no Core modules found; the registry moved"
    assert "mask" in core and "measure" in core


def test_every_core_module_is_opened_by_some_tutorial():
    """The inventory check instruction 358 asks for.

    Fails two ways, both wanted: a NEW Core module with no tutorial, and a
    module listed as uncovered that has quietly been covered (so the ratchet
    cannot rot).
    """
    core = set(_core_modules())
    opened = {m for visits in _modules_each_tutorial_opens().values()
              for m in visits}
    uncovered = core - opened

    unexpected = uncovered - UNCOVERED_CORE_MODULES
    assert not unexpected, (
        f"Core modules with no tutorial that opens them: {sorted(unexpected)}. "
        "Write one, or add it to UNCOVERED_CORE_MODULES with a reason.")

    stale = UNCOVERED_CORE_MODULES - uncovered
    assert not stale, (
        f"{sorted(stale)} are covered now; delete them from "
        "UNCOVERED_CORE_MODULES so the ratchet keeps shrinking.")


def test_no_tutorial_opens_a_module_that_does_not_exist():
    """A tutorial pointing at a removed module teaches a dead path.

    `__home__` is the Home pseudo-key rather than a module, so it is allowed
    without being in the registry.
    """
    from spacr.qt.app import APPS

    known = {row[0] for row in APPS} | {"__home__"}
    broken = {
        f"{tutorial} -> {module}"
        for tutorial, visits in _modules_each_tutorial_opens().items()
        for module in visits
        if module not in known
    }

    assert not broken, f"tutorials navigating to unknown modules: {sorted(broken)}"


def test_every_tutorial_demo_names_a_registered_demo():
    """A demo key that no longer exists loads nothing and says nothing."""
    from spacr.qt.app import DEMO_LABELS

    tree = ast.parse(SCRIPTS.read_text(encoding="utf-8"))
    asked = {
        a.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "_load_demo"
        for a in node.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    }
    missing = sorted(asked - set(DEMO_LABELS))

    assert not missing, f"tutorials load demos that are not registered: {missing}"
