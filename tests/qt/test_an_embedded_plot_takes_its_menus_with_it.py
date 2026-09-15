"""A plot destroyed inside another widget takes its pyqtgraph menus with it.

pyqtgraph builds each ``ViewBox`` menu and each ``PlotItem`` control menu as a
parentless, top-level window and keeps it only through Python references.
``FastPlot.closeEvent`` retires them, but a plot EMBEDDED in a panel never gets
a close event: it is destroyed with its parent, or detached and deleted when
the panel redraws (``MeasurementComparePanel._draw``). Its menus were then
left live, owned by nothing but a reference cycle, until Python's cycle
collector freed them at whatever allocation it happened to run.

That was not harmless. Found on 2026-09-15 while testing the CI crash family
(item 43): ``tests/qt/test_cov_r5_cell_montage_view.py`` followed by
``tests/qt/test_one_close_mark.py`` segfaulted inside ``QApplication.setStyleSheet``
in every run with automatic collection on, and never with it off. The
``GroupedPlot`` of the montage's Compare tab was built in one test; the
collector freed its two ``ViewBoxMenu`` twelve tests later; the next
application-wide restyle crashed in Qt. Keeping those menus alive, or deleting
them at a fixed point, removed the crash; nothing else did.

So the contract is ownership, tested without the collector: once the plot is
gone and the event loop has turned, its menus are gone too.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")
pytest.importorskip("pyqtgraph")

import shiboken6  # noqa: E402
from PySide6.QtWidgets import QVBoxLayout, QWidget  # noqa: E402

pytestmark = pytest.mark.qt


def _menus(plot):
    item = plot.plot.plotItem
    menus = [item.vb.menu, item.ctrlMenu]
    assert all(m is not None and shiboken6.isValid(m) for m in menus)
    return menus


def _embedded_plot():
    from spacr.qt.widgets.fast_plots import FastPlot

    host = QWidget()
    layout = QVBoxLayout(host)
    plot = FastPlot(title="embedded", parent=host)
    layout.addWidget(plot)
    return host, plot


def test_the_menus_start_as_windows_of_their_own():
    """The premise: nothing in Qt owns these menus, so nothing in Qt would
    delete them. If pyqtgraph ever parents them, this file can go."""
    host, plot = _embedded_plot()
    try:
        assert all(m.parentWidget() is None for m in _menus(plot))
    finally:
        shiboken6.delete(host)


def test_a_plot_destroyed_with_its_parent_takes_its_menus(qtbot):
    host, plot = _embedded_plot()
    menus = _menus(plot)

    host.deleteLater()
    qtbot.waitUntil(lambda: not shiboken6.isValid(host))
    qtbot.waitUntil(lambda: not any(shiboken6.isValid(m) for m in menus),
                    timeout=2000)


def test_a_plot_detached_and_deleted_takes_its_menus(qtbot):
    """The redraw path of the comparison panel: take the old canvas out of
    its layout, drop its parent, delete it later."""
    host, plot = _embedded_plot()
    qtbot.addWidget(host)
    menus = _menus(plot)

    plot.setParent(None)
    plot.deleteLater()
    qtbot.waitUntil(lambda: not shiboken6.isValid(plot))
    qtbot.waitUntil(lambda: not any(shiboken6.isValid(m) for m in menus),
                    timeout=2000)


def test_a_closed_plot_still_retires_its_menus_once(qtbot):
    """closeEvent already retired the menus; the plot's destruction must then
    find nothing to do rather than raise on a deleted menu."""
    host, plot = _embedded_plot()
    menus = _menus(plot)
    plot.close()
    host.deleteLater()
    with qtbot.capture_exceptions() as raised:
        qtbot.waitUntil(lambda: not shiboken6.isValid(host))
        qtbot.waitUntil(
            lambda: not any(shiboken6.isValid(m) for m in menus),
            timeout=2000)
    assert not raised, [f"{kind.__name__}: {value}"
                        for kind, value, _tb in raised]
