"""The automatic walkthrough must not re-navigate to the module it is on.

Instruction 314. ``QStackedWidget.currentChanged`` is emitted AFTER
``setCurrentWidget`` has made the screen current, and the walkthrough listens
to it. Before the guard, ``show_walkthrough`` navigated unconditionally, so
every automatic showing re-entered ``_on_nav_selected`` for the module already
on screen. Eleven modules took that second trip on every recorded startup --
the fallback and shared-settings screens -- and the nested call ran before the
outer one installed its readiness watcher, so the outer call then replaced the
nested probe.

The menu route is the reason this is a condition rather than a deletion: from
a menu a DIFFERENT module is current, and that navigation must still happen.
"""

import pytest

from spacr.qt import walkthrough


class _Screen:
    def __init__(self, app_key):
        self.app_key = app_key


class _Stack:
    def __init__(self, current):
        self._current = current

    def currentWidget(self):
        return self._current


class _Window:
    """Enough main window for ``show_walkthrough``'s navigation decision."""

    def __init__(self, current_key, keys):
        self._screens = {k: _Screen(k) for k in keys}
        self._stack = _Stack(self._screens.get(current_key))
        self.navigated = []

    def _on_nav_selected(self, app_key):
        self.navigated.append(app_key)
        self._stack._current = self._screens.get(app_key)


@pytest.fixture
def no_overlay(monkeypatch):
    """Stop before the overlay: this is about the navigation decision.

    ``build_steps`` returning nothing makes ``show_walkthrough`` return early
    AFTER the navigation block, which is the part under test.
    """
    monkeypatch.setattr(walkthrough, "build_steps", lambda app_key: [])


def test_the_module_already_on_screen_is_not_navigated_to_again(no_overlay):
    window = _Window("regression", ["regression", "mask"])

    walkthrough.show_walkthrough(window, "regression")

    assert window.navigated == []


def test_a_different_module_is_navigated_to_exactly_once(no_overlay):
    window = _Window("mask", ["regression", "mask"])

    walkthrough.show_walkthrough(window, "regression")

    assert window.navigated == ["regression"]


def test_an_unreadable_stack_navigates_rather_than_skipping(no_overlay):
    """The guard fails toward navigating, which is the cheaper mistake."""

    class _Broken(_Window):
        @property
        def _stack(self):
            raise RuntimeError("no stack here")

        @_stack.setter
        def _stack(self, value):
            pass

    window = _Broken("regression", ["regression"])

    walkthrough.show_walkthrough(window, "regression")

    assert window.navigated == ["regression"]


def test_the_guard_is_what_makes_the_first_test_pass(no_overlay, monkeypatch):
    """Mutation check, kept in the suite rather than done once by hand.

    With ``_already_current`` forced to ``False`` -- which is exactly the
    behaviour before the guard -- the already-current case navigates again.
    If this test ever goes green while the first one does too, the first one
    has stopped measuring the guard.
    """
    monkeypatch.setattr(walkthrough, "_already_current", lambda w, k: False)
    window = _Window("regression", ["regression", "mask"])

    walkthrough.show_walkthrough(window, "regression")

    assert window.navigated == ["regression"]
