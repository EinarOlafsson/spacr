"""``retire_pyqtgraph_menus``: what it refuses to touch, and why that matters.

pyqtgraph creates ``PlotItem`` and ``ViewBox`` context menus as TOP-LEVEL
windows and keeps them alive through Python references, so closing a screen
used to leave hundreds of live menu widgets behind -- every one of which a
later palette change then had to visit.

The module's own docstring states the constraint the fix has to satisfy: it
walks only graphics scenes owned by the widget being closed, never sweeps
``QApplication`` and never invokes the cycle collector, *so it cannot collect
an unrelated live QThread wrapper*. Every refusal below is that constraint
being honoured, and each returns 0 rather than raising -- this runs during
teardown, where an exception has nowhere useful to go.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

from PySide6.QtWidgets import QWidget                               # noqa: E402

from spacr.qt.widget_cleanup import retire_pyqtgraph_menus          # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# owners it cannot or must not walk
# ---------------------------------------------------------------------------

def test_an_owner_that_is_not_a_widget_retires_nothing():
    """Called from teardown paths that may hold anything.

    ``findChildren`` is the first thing asked for, and an object without it is
    answered with 0 rather than an AttributeError travelling out of a close.
    """
    assert retire_pyqtgraph_menus(object()) == 0


def test_an_owner_whose_c_plus_plus_half_is_gone_retires_nothing():
    """Qt frees the widget before Python lets go of the wrapper.

    That is the ordinary end of a screen's life, so it is the case this
    helper meets most often.
    """
    class _Freed:
        def findChildren(self, *args, **kwargs):
            raise RuntimeError("Internal C++ object already deleted.")

    assert retire_pyqtgraph_menus(_Freed()) == 0


def test_a_widget_with_no_graphics_views_is_left_entirely_alone(qtbot):
    """Most screens have none, and the sweep must cost them nothing."""
    owner = QWidget()
    qtbot.addWidget(owner)
    QWidget(owner)

    assert retire_pyqtgraph_menus(owner) == 0


def test_the_sweep_does_not_import_pyqtgraph_to_do_its_work(qtbot,
                                                            monkeypatch):
    """Cleanup must not be the event that loads an optional plotting stack.

    A real pyqtgraph view can only exist after the package is imported, so a
    QGraphicsView with pyqtgraph absent from sys.modules has nothing this
    helper can own -- and importing it to find that out would make closing a
    screen pull in a plotting library the run never used.
    """
    from PySide6.QtWidgets import QGraphicsView

    owner = QWidget()
    qtbot.addWidget(owner)
    view = QGraphicsView(owner)
    assert view is not None

    monkeypatch.delitem(sys.modules, "pyqtgraph", raising=False)

    assert retire_pyqtgraph_menus(owner) == 0
    assert "pyqtgraph" not in sys.modules, "the sweep imported pyqtgraph"


# ---------------------------------------------------------------------------
# what it does retire
# ---------------------------------------------------------------------------

def test_a_real_plot_s_menus_are_retired_and_counted(qtbot):
    """The path everything above is defined against.

    Without it, "returns 0" would pass on a helper that never retires
    anything at all.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1, 2], [0, 1, 4])
    qtbot.addWidget(plot)

    retired = retire_pyqtgraph_menus(owner)

    assert retired > 0, "a real pyqtgraph plot left no menus to retire"


def test_sweeping_twice_retires_nothing_the_second_time(qtbot):
    """The menu references are cleared, so a second close is cheap.

    Leaving them in place would make every subsequent palette change walk
    menus that have already been deleted.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1], [0, 1])
    qtbot.addWidget(plot)

    first = retire_pyqtgraph_menus(owner)
    second = retire_pyqtgraph_menus(owner)

    assert first > 0
    assert second == 0, "the menus were retired twice"


def test_a_view_whose_scene_is_gone_is_skipped_not_fatal(qtbot,
                                                         monkeypatch):
    """A scene can be freed between the findChildren and the walk.

    Skipping that view and carrying on is what lets the rest of a screen's
    menus still be retired; raising would abandon them all.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1], [0, 1])
    qtbot.addWidget(plot)

    def gone(self):
        raise RuntimeError("Internal C++ object already deleted.")

    monkeypatch.setattr(type(plot), "scene", gone, raising=False)

    assert retire_pyqtgraph_menus(owner) == 0


def test_a_viewbox_that_will_not_close_still_has_its_menu_retired(qtbot,
                                                                  monkeypatch):
    """``ViewBox.close`` raises KeyError when pyqtgraph has already forgotten it.

    That is a bookkeeping failure inside the plotting library, and it must not
    cost the sweep the menu it came for -- the menu is the top-level window
    that would otherwise survive the screen and be walked by every later
    palette change.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1], [0, 1])
    qtbot.addWidget(plot)

    def forgotten(self):
        raise KeyError("this ViewBox is not in the registry")

    monkeypatch.setattr(pg.ViewBox, "close", forgotten, raising=False)

    assert retire_pyqtgraph_menus(owner) > 0


def test_a_menu_reference_that_cannot_be_cleared_is_not_fatal(qtbot,
                                                              monkeypatch):
    """Some pyqtgraph builds make ``menu`` a read-only property.

    The sweep's job is to retire the widget; clearing the reference is a
    courtesy that keeps a second pass cheap. Failing the whole cleanup because
    the courtesy was refused would leave every LATER menu alive too.
    """
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1], [0, 1])
    qtbot.addWidget(plot)

    class _Sealed(pg.ViewBox):
        pass

    def refuse(self, _value):
        raise AttributeError("menu is read-only on this build")

    monkeypatch.setattr(pg.ViewBox, "menu",
                        property(lambda self: getattr(self, "_menu_stub", None),
                                 refuse),
                        raising=False)

    # Nothing may escape; the count is whatever the build allows.
    assert retire_pyqtgraph_menus(owner) >= 0
