"""``retire_pyqtgraph_menus``: how many menu roots a closing screen retires.

A real pyqtgraph plot contributes two parentless menus (the PlotItem's
control menu and the ViewBox menu). One table drives the owners and
environments the sweep meets during teardown, asserting the exact count it
returns and the state it leaves the plot items and menus in: a missing
PySide6 or pyqtgraph, an owner that is gone or is not a widget, a menu whose
C++ half is already freed, a ``ctrlMenu`` that will not be cleared, and a
menu Qt already owns.
"""
from __future__ import annotations

import sys

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

import shiboken6  # noqa: E402
from PySide6.QtCore import QEvent  # noqa: E402
from PySide6.QtWidgets import QWidget  # noqa: E402

from spacr.qt.widget_cleanup import retire_pyqtgraph_menus  # noqa: E402

pytestmark = pytest.mark.qt


def _plot_owner(qtbot):
    """A widget owning one real plot; returns ``(owner, plot_item, view_box)``."""
    import pyqtgraph as pg

    owner = QWidget()
    qtbot.addWidget(owner)
    plot = pg.PlotWidget(parent=owner)
    plot.plot([0, 1, 2], [0, 1, 4])
    items = list(plot.scene().items())
    plot_item = next(i for i in items if isinstance(i, pg.PlotItem))
    view_box = next(i for i in items if isinstance(i, pg.ViewBox))
    return owner, plot_item, view_box


def _drain_deferred_deletes(qapp):
    """Deliver queued ``deleteLater`` calls so retired menus really die."""
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    qapp.sendPostedEvents(None, QEvent.DeferredDelete)


def test_retiring_pyqtgraph_menus_counts_each_root_once_and_survives_teardown(
        qtbot, qapp, monkeypatch, subtests):
    """Exact count and left-over state for every owner and environment the sweep meets."""
    import pyqtgraph as pg

    class Dead:
        def findChildren(self, *_args, **_kwargs):  # noqa: N802 - Qt spelling
            raise RuntimeError("Internal C++ object already deleted.")

    class NotAWidget:
        pass

    refusals = [
        ("PySide6 cannot be imported", "PySide6.QtWidgets", None),
        ("pyqtgraph was never imported", "pyqtgraph", None),
    ]
    for case, module, replacement in refusals:
        with subtests.test(msg=case):
            owner, plot_item, view_box = _plot_owner(qtbot)
            with monkeypatch.context() as patch:
                patch.setitem(sys.modules, module, replacement)
                assert retire_pyqtgraph_menus(owner) == 0
            assert plot_item.ctrlMenu is not None
            assert view_box.menu is not None
            assert retire_pyqtgraph_menus(owner) == 2

    plain = QWidget()
    qtbot.addWidget(plain)
    for case, owner in (("an owner with no plots", plain),
                        ("an owner whose C++ half is gone", Dead()),
                        ("an owner that is not a widget", NotAWidget())):
        with subtests.test(msg=case):
            assert retire_pyqtgraph_menus(owner) == 0

    with subtests.test(msg="both menus are retired, and only once"):
        owner, plot_item, view_box = _plot_owner(qtbot)
        ctrl_menu, view_menu = plot_item.ctrlMenu, view_box.menu
        assert retire_pyqtgraph_menus(owner) == 2
        assert (plot_item.ctrlMenu, view_box.menu) == (None, None)
        assert retire_pyqtgraph_menus(owner) == 0
        _drain_deferred_deletes(qapp)
        assert not shiboken6.isValid(ctrl_menu)
        assert not shiboken6.isValid(view_menu)

    with subtests.test(msg="a freed menu does not take the next one along"):
        owner, plot_item, view_box = _plot_owner(qtbot)
        view_menu = view_box.menu
        shiboken6.delete(plot_item.ctrlMenu)
        assert retire_pyqtgraph_menus(owner) == 2
        assert (plot_item.ctrlMenu, view_box.menu) == (None, None)
        _drain_deferred_deletes(qapp)
        assert not shiboken6.isValid(view_menu)

    with subtests.test(msg="a ctrlMenu that will not clear is still destroyed"):
        owner, plot_item, view_box = _plot_owner(qtbot)
        ctrl_menu, view_menu = plot_item.ctrlMenu, view_box.menu
        with monkeypatch.context() as patch:
            patch.setattr(pg.PlotItem, "ctrlMenu", property(
                lambda self: self.__dict__.get("ctrlMenu"),
                lambda self, _value: (_ for _ in ()).throw(
                    AttributeError("ctrlMenu is read-only on this build"))),
                raising=False)
            assert retire_pyqtgraph_menus(owner) == 2
            assert plot_item.__dict__["ctrlMenu"] is ctrl_menu
        assert view_box.menu is None
        _drain_deferred_deletes(qapp)
        assert not shiboken6.isValid(ctrl_menu)
        assert not shiboken6.isValid(view_menu)

    with subtests.test(msg="a menu Qt already owns is not reparented"):
        owner, _plot_item, _view_box = _plot_owner(qtbot)
        transfers = []
        with monkeypatch.context() as patch:
            patch.setattr(QWidget, "parentWidget", lambda self: owner)
            patch.setattr(QWidget, "setParent",
                          lambda self, parent: transfers.append(
                              type(self).__name__))
            assert retire_pyqtgraph_menus(owner) == 2
        assert transfers == []

    with subtests.test(msg="a parentless menu is adopted by the owner"):
        owner, _plot_item, _view_box = _plot_owner(qtbot)
        adopted = []
        with monkeypatch.context() as patch:
            patch.setattr(QWidget, "parentWidget", lambda self: None)
            patch.setattr(QWidget, "setParent",
                          lambda self, parent: adopted.append(parent))
            assert retire_pyqtgraph_menus(owner) == 2
        assert adopted[0] is owner
