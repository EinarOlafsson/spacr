"""Every Core module the application ships should have a tutorial.

DERIVED FROM THE RUNTIME REGISTRY, NOT FROM A LIST. Instruction 358 asks for
exactly this and says why: a second hard-coded inventory drifts from the
first, and the failure it produces is a tutorial set that looks complete and
teaches a module structure the application no longer has.

So the Core inventory comes from `spacr.qt.app.APPS` filtered to
`SECTION_CORE` -- the same rows Home groups -- and the tutorial inventory is
the lesson catalog the published library, the in-app Tutorial links and
`spacr-tutorial` all read (`spacr.qt.tutorials.lessons`). Neither is retyped
here.

WHAT A TUTORIAL "COVERING" A MODULE MEANS: a lesson names it as the screen it
covers, so the module's own Tutorial link and `spacr-tutorial <module>` both
open that lesson.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

#: Core modules with no lesson, as measured on 2026-10-12.
#:
#: A RATCHET, NOT AN EXCUSE. It may only shrink: an entry is deleted when its
#: lesson is published, and a Core module that appears WITHOUT a lesson and
#: without being listed here fails the test. That is the property instruction
#: 358 asks for -- "fails when a new Core module appears without one".
UNCOVERED_CORE_MODULES: set = set()


def _core_modules() -> dict:
    """The Core modules the running application registers."""
    from spacr.qt.app import APPS, SECTION_CORE

    return {row[0]: row[1] for row in APPS if row[3] == SECTION_CORE}


def test_the_core_inventory_is_read_from_the_registry():
    """The premise: there are Core modules, and they come from the app."""
    core = _core_modules()

    assert core, "no Core modules found; the registry moved"
    assert "mask" in core and "measure" in core


def test_every_core_module_has_a_lesson():
    """The inventory check instruction 358 asks for.

    Fails two ways, both wanted: a NEW Core module with no lesson, and a
    module listed as uncovered that has quietly been covered (so the ratchet
    cannot rot).
    """
    from spacr.qt.tutorials import find_lesson, lessons

    covered = {lesson["app_key"] for lesson in lessons()}
    uncovered = set(_core_modules()) - covered

    unexpected = uncovered - UNCOVERED_CORE_MODULES
    assert not unexpected, (
        f"Core modules with no lesson: {sorted(unexpected)}. Publish one, or "
        "add it to UNCOVERED_CORE_MODULES with a reason.")

    stale = UNCOVERED_CORE_MODULES - uncovered
    assert not stale, (
        f"{sorted(stale)} are covered now; delete them from "
        "UNCOVERED_CORE_MODULES so the ratchet keeps shrinking.")

    for key in set(_core_modules()) - uncovered:
        lesson, _ = find_lesson(key)
        assert lesson is not None and lesson["app_key"] == key, key
