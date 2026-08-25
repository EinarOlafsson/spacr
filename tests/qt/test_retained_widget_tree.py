"""What one test drops, the next test must not still be paying for.

A Qt widget with bound-method connections is part of a reference cycle, so
dropping the last reference to it frees nothing: the C++ widget stays alive,
and stays in ``QApplication.allWidgets()``, until the CYCLE collector runs.
Every global restyle visits all of them, which is why a suite that never
collects gets slower as it goes and the tests that pay are the ones that set a
stylesheet — ``test_field_fade.py`` costs 2.7 s on its own, 11.8 s with 5,000
spare widgets alive and 39.2 s with 20,000.

So the suite collects at a test boundary once the tree is big enough to
matter. What is pinned here is that the rule actually fires and actually
frees, measured through the real application rather than by trusting the
fixture to have been written correctly.

Automatic collection is off for this module on purpose. Python would collect
these widgets eventually on its own, and a test that cannot tell the suite's
explicit collect from Python's own timing proves nothing about either.
"""
from __future__ import annotations

import gc

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget

from tests.qt.conftest import COLLECT_ABOVE_WIDGETS

#: Comfortably over the threshold, so one test crossing it is unambiguous.
LEAKED = COLLECT_ABOVE_WIDGETS + 500


class _Cyclic(QWidget):
    """A widget that holds itself, which is all a reference cycle is.

    Nothing exotic: a bound method stored on the instance, a lambda closing
    over ``self``, or a child that keeps a handle on its parent does the same
    thing, and every screen in this application has several.
    """

    def __init__(self):
        super().__init__()
        self._cycle = self


@pytest.fixture(scope="module", autouse=True)
def _automatic_collection_is_off():
    """Leave only the deliberate collect, so what passes is attributable."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def _live_widgets():
    return len(QApplication.instance().allWidgets())


def test_widgets_dropped_inside_a_test_are_still_alive_when_it_ends():
    """The hazard: dropping the reference is not the same as freeing it."""
    before = _live_widgets()
    leaked = [_Cyclic() for _ in range(LEAKED)]
    assert _live_widgets() >= before + LEAKED

    del leaked
    assert _live_widgets() >= before + LEAKED, (
        "the cycle collector is the only thing that can free these, and it "
        "is switched off for this module")


def test_the_next_test_does_not_inherit_them():
    """Runs straight after the leak above, which is the whole point.

    No fixture is requested and nothing is cleaned up here. If this passes,
    the collecting happened between the two tests -- which is where it has to
    happen for the rest of the suite to stop paying for it.
    """
    assert _live_widgets() < COLLECT_ABOVE_WIDGETS


@pytest.mark.parametrize("round_number", range(3))
def test_the_tree_stays_bounded_however_many_tests_leak(round_number):
    """One leaky test is a blip; the suite has hundreds.

    Each round leaks the same amount again and finds the tree back under the
    threshold at its own start, so the ceiling is a ceiling rather than a
    slower climb.
    """
    assert _live_widgets() < COLLECT_ABOVE_WIDGETS
    leaked = [_Cyclic() for _ in range(LEAKED)]
    assert _live_widgets() >= LEAKED
    del leaked
