"""``retire_pyqtgraph_menus``: the three failures it swallows on the way out.

The sweep runs during teardown, where an exception has nowhere useful to go,
so every failure it meets has to cost it only the thing that failed -- never
the menus it has not reached yet. ``tests/qt/test_the_menu_sweep_only_touches_
what_it_owns.py`` pins the refusals that return 0. This file pins the three
that do not:

  * PySide6 itself cannot be imported -- the helper is a no-op, not an error;
  * a menu whose C++ half is already gone -- the *next* menu is still retired;
  * a build where ``PlotItem.ctrlMenu`` will not be reassigned -- the menu is
    still destroyed, only the courtesy reference survives.

Each one is asserted against the same sweep run with the failure absent, so
"nothing happened" cannot pass for "the failure was handled".
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

import shiboken6                                                    # noqa: E402
from PySide6.QtCore import QEvent                                   # noqa: E402
from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.widget_cleanup import retire_pyqtgraph_menus          # noqa: E402

pytestmark = pytest.mark.qt


def _plot_owner(qtbot):
    """A widget owning one real pyqtgraph plot: one PlotItem, one ViewBox.

    Returns ``(owner, plot_item, view_box)``. The two scene items are reached
    through ``scene().items()`` because that is exactly what the sweep walks,
    and the tests below have to name the same objects it will find.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1, 2], [0, 1, 4])
    qtbot.addWidget(plot)

    items = list(plot.scene().items())
    plot_item = next(i for i in items if isinstance(i, pg.PlotItem))
    view_box = next(i for i in items if isinstance(i, pg.ViewBox))
    return owner, plot_item, view_box


def _drain_deleteLater(qapp):
    """Turn the queued ``deleteLater`` calls into actual destruction.

    Until they are delivered, a retired menu's wrapper is still valid and
    ``shiboken6.isValid`` cannot tell a retired menu from a live one.
    """
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)


def test_a_build_without_pyside6_sweeps_nothing_instead_of_raising(qtbot,
                                                                   monkeypatch):
    """The Qt import is the first thing the helper does, and it is optional.

    spaCR is importable without a working PySide6 -- the CLI paths never touch
    it -- so a teardown hook that reached this helper on such a build would
    raise ImportError out of a close. The same owner is swept twice, once with
    the import broken and once with it working, so the 0 is a refusal and not
    an empty owner.
    """
    owner, _plot_item, _view_box = _plot_owner(qtbot)

    # A None entry in sys.modules is how CPython spells "this import is
    # halted"; `from PySide6.QtWidgets import ...` then raises ImportError.
    monkeypatch.setitem(sys.modules, "PySide6.QtWidgets", None)
    assert retire_pyqtgraph_menus(owner) == 0

    monkeypatch.undo()
    assert retire_pyqtgraph_menus(owner) == 2, (
        "the same owner retired nothing with PySide6 available, so the 0 "
        "above proved nothing about the import guard"
    )


def test_a_menu_already_freed_does_not_take_the_next_one_with_it(qtbot, qapp):
    """Qt can free a menu between the scene walk and the ``close()``.

    The PlotItem's control menu is visited first; if its RuntimeError escaped,
    the ViewBox menu -- a top-level window that every later palette change
    would have to walk -- would survive the screen. Both menus are still
    counted, and the survivor is the one the assertion is about.
    """
    owner, plot_item, view_box = _plot_owner(qtbot)
    view_box_menu = view_box.menu
    assert shiboken6.isValid(view_box_menu), "no live ViewBox menu to lose"

    # Free the C++ half only; the Python wrapper stays in ``plot_item`` and is
    # what the sweep will pick up, exactly as it would after Qt destroyed the
    # menu's parent chain.
    shiboken6.delete(plot_item.ctrlMenu)

    retired = retire_pyqtgraph_menus(owner)

    assert retired == 2, "the dead menu aborted the walk"
    assert plot_item.ctrlMenu is None
    assert view_box.menu is None

    _drain_deleteLater(qapp)
    assert not shiboken6.isValid(view_box_menu), (
        "the ViewBox menu outlived the sweep because a dead sibling raised"
    )


def test_a_ctrl_menu_reference_that_will_not_clear_is_still_destroyed(qtbot,
                                                                      qapp,
                                                                      monkeypatch):
    """Clearing ``ctrlMenu`` is a courtesy; destroying the menu is the job.

    Some pyqtgraph builds expose ``ctrlMenu`` as a read-only property. If the
    refused assignment escaped, the ViewBox loop below it would never run and
    that menu would leak -- so the refusal has to cost nothing but itself.
    """
    import pyqtgraph as pg

    owner, plot_item, view_box = _plot_owner(qtbot)
    ctrl_menu = plot_item.ctrlMenu
    view_box_menu = view_box.menu
    assert shiboken6.isValid(ctrl_menu) and shiboken6.isValid(view_box_menu)

    # A class-level property shadows the instance attribute, so the getter has
    # to hand back what pyqtgraph already stored there. Patched on the class
    # because that is where a build would seal it.
    monkeypatch.setattr(
        pg.PlotItem,
        "ctrlMenu",
        property(
            lambda self: self.__dict__.get("ctrlMenu"),
            lambda self, _value: (_ for _ in ()).throw(
                AttributeError("ctrlMenu is read-only on this build")
            ),
        ),
        raising=False,
    )

    retired = retire_pyqtgraph_menus(owner)

    assert retired == 2, "the refused assignment cost the ViewBox its menu"
    assert "ctrlMenu" in plot_item.__dict__, (
        "the reference was cleared after all, so nothing was refused"
    )
    assert view_box.menu is None

    _drain_deleteLater(qapp)
    assert not shiboken6.isValid(ctrl_menu), "the sealed menu was never retired"
    assert not shiboken6.isValid(view_box_menu)
