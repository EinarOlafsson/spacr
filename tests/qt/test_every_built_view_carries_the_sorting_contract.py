"""Build the real screens and look at the tables Qt actually made.

The other sweep reads the source. This one builds every registered module
through the application's own screen factory and inspects the view objects
that come out, so a table wired up through a helper the source sweep cannot
follow is still caught -- and so the contract is measured on the widget
rather than on the call that was supposed to set it up.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QObject, Qt  # noqa: E402
from PySide6.QtWidgets import (QTableView, QTableWidget, QTreeView,  # noqa: E402
                               QTreeWidget)

from spacr.qt.app import APPS, MainWindow  # noqa: E402

pytestmark = pytest.mark.qt

#: Views that must not take Qt's model sort, with the reason.
#: The database browser sorts in SQL over the whole table; Qt's sort would
#: reorder only the rows fetched so far and call that the table's order.
EXEMPT_OBJECTS = {"DbBrowserPreview"}


class _FactoryHost(QObject):
    """Stand-in ``self`` for the unbound ``MainWindow._build_screen``."""

    def _snapshot_current_screen_settings(self):
        return "mask", {}

    def _build_screen_timed(self, key):
        """Follow the production wrapper into the real screen factory."""
        return MainWindow._build_screen_timed(self, key)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        if not hasattr(MainWindow, name):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


def _views(widget):
    for view in widget.findChildren(QTableWidget):
        yield view
    for view in widget.findChildren(QTreeWidget):
        yield view
    for view in widget.findChildren(QTableView):
        if not isinstance(view, QTableWidget):
            yield view
    for view in widget.findChildren(QTreeView):
        if not isinstance(view, QTreeWidget):
            yield view


def _header(view):
    if isinstance(view, (QTreeView, QTreeWidget)):
        return view.header()
    return view.horizontalHeader()


def _describe(view) -> str:
    return f"{type(view).__name__}({view.objectName() or 'unnamed'})"


#: Modules that must own a table, so a sweep that inspected nothing cannot
#: pass for free. Regression is the module the request named.
MUST_HAVE_TABLES = {"regression": 2, "data_manager": 1}


@pytest.mark.parametrize("app_key", [key for key, *_rest in APPS])
def test_every_view_a_module_builds_sorts_descending_first(
        qtbot, qt_theme_applied, app_key):
    """Measured on the widget: sorting on, and a fresh column starts down."""
    host = _FactoryHost()
    try:
        screen = MainWindow._build_screen(host, app_key)
    except Exception as error:                               # noqa: BLE001
        # Whether every module can be BUILT is
        # ``test_all_module_smoke``'s question, and answering it here as
        # well would report someone else's breakage as a sorting failure.
        pytest.skip(f"{app_key} does not build: {error}")
    qtbot.addWidget(screen)
    screen.resize(1200, 720)
    screen.show()
    qt_theme_applied.processEvents()

    seen = 0
    for view in _views(screen):
        if view.objectName() in EXEMPT_OBJECTS:
            continue
        seen += 1
        where = f"{app_key}: {_describe(view)}"
        assert view.isSortingEnabled(), f"{where} cannot sort at all"
        header = _header(view)
        assert header.sectionsClickable(), f"{where} has a dead header"
        assert header.isSortIndicatorShown(), f"{where} shows no indicator"
        model = view.model()
        for column in range(model.columnCount()):
            initial = model.headerData(column, Qt.Horizontal,
                                       Qt.InitialSortOrderRole)
            assert initial == Qt.DescendingOrder.value, (
                f"{where} column {column} would sort ascending first")

    # Guards the guard. Most modules build no table and nothing above them
    # runs; these do, so a sweep that has stopped finding views says so here
    # rather than passing silently.
    expected = MUST_HAVE_TABLES.get(app_key)
    if expected is not None:
        assert seen >= expected, (
            f"{app_key} built {seen} views; the sweep expected at least "
            f"{expected} and is no longer looking in the right place")
