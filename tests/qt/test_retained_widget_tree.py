"""What one test asks Qt to delete, the next test must not inherit.

A Qt wrapper may be part of a reference cycle while its C++ object is still a
live widget. Explicitly running Python's cycle collector over that heap is not
safe: an unreachable QThread wrapper can still own a running C++ thread, and
CI has segfaulted inside exactly that setup-boundary ``gc.collect`` call.

The safe ownership contract is narrower. A test registers a widget with
``qtbot`` or calls ``deleteLater`` itself; the next boundary delivers those
deferred deletes. What is pinned here is that thousands of requested deletes
really are delivered, keeping ``QApplication.allWidgets()`` bounded without
ever forcing Python collection over live Qt wrappers.
"""
from __future__ import annotations

import ast
import inspect

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
from tests import conftest as root_conftest
from tests.qt import conftest as qt_conftest

#: Large enough that failing to deliver the queued deletes is unmistakable.
RETAINED_WIDGET_CEILING = 2000
LEAKED = RETAINED_WIDGET_CEILING + 500


class _Cyclic(QWidget):
    """A widget that holds itself, which is all a reference cycle is.

    Nothing exotic: a bound method stored on the instance, a lambda closing
    over ``self``, or a child that keeps a handle on its parent does the same
    thing, and every screen in this application has several.
    """

    def __init__(self):
        super().__init__()
        self._cycle = self


def _live_widgets():
    return len(QApplication.instance().allWidgets())


def test_owner_requested_deletions_are_pending_when_the_test_ends():
    """``deleteLater`` queues ownership cleanup; it is not synchronous."""
    before = _live_widgets()
    leaked = [_Cyclic() for _ in range(LEAKED)]
    assert _live_widgets() >= before + LEAKED

    for widget in leaked:
        widget.deleteLater()
    del leaked
    assert _live_widgets() >= before + LEAKED, (
        "deleteLater must stay deferred until Qt reaches a safe boundary")


def test_the_next_test_does_not_inherit_them():
    """Runs straight after the leak above, which is the whole point.

    No fixture is requested and nothing is cleaned up here. If this passes,
    the deferred deletes were delivered between the two tests -- which is
    where that has to happen for the rest of the suite to stop paying for it.
    """
    assert _live_widgets() < RETAINED_WIDGET_CEILING


@pytest.mark.parametrize("round_number", range(3))
def test_the_tree_stays_bounded_over_repeated_owned_deletions(round_number):
    """One deletion batch is a blip; the suite has hundreds.

    Each round leaks the same amount again and finds the tree back under the
    threshold at its own start, so the ceiling is a ceiling rather than a
    slower climb.
    """
    assert _live_widgets() < RETAINED_WIDGET_CEILING
    leaked = [_Cyclic() for _ in range(LEAKED)]
    assert _live_widgets() >= LEAKED
    for widget in leaked:
        widget.deleteLater()
    del leaked


def test_no_boundary_fixture_forces_python_collection_over_live_qt_wrappers():
    """The CI segfault guard: ownership cleanup must stay on Qt's side."""
    sources = (
        inspect.getsource(
            root_conftest._the_widget_tree_does_not_outgrow_the_session),
        inspect.getsource(qt_conftest.deferred_deletions_flushed),
    )

    for source in sources:
        calls = [
            node.lineno for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "collect"
        ]
        assert not calls, f"boundary fixture calls collect on lines {calls}"


def test_menu_cleanup_does_not_load_pyqtgraph_for_a_plain_screen(
        qtbot, monkeypatch):
    """Closing a plot-free module must not cold-load optional plotting code."""
    import builtins

    owner = QWidget()
    qtbot.addWidget(owner)
    imported = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "pyqtgraph" or name.startswith("pyqtgraph."):
            raise AssertionError("cleanup imported pyqtgraph")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert retire_pyqtgraph_menus(owner) == 0


def test_closing_a_plot_screen_retires_its_parentless_menu_tree(qapp):
    """One regression screen used to leave about 650 live menu widgets."""
    before = _live_widgets()
    screen = AppScreen("regression")

    built = _live_widgets()
    assert built > before + 1000, "the real pyqtgraph-heavy screen was not built"

    retired = retire_pyqtgraph_menus(screen)
    screen.close()
    screen.deleteLater()
    from PySide6.QtCore import QCoreApplication, QEvent
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    assert retired >= 10, "the ownership walk stopped finding plot menus"
    assert _live_widgets() < before + 100, (
        "closing one screen left its parentless pyqtgraph menu tree alive")
