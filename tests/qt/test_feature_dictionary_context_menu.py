"""The Feature Dictionary's hooks into tables, menus and windows it does not own.

Right-clicking a column is help bolted onto grids the dictionary did not build,
through an application-wide event filter that sees every event in the process.
It must name a column only where a column is really under the pointer, stay
silent on a row header, past the last column, below the last row, on a grid
that names no columns and on a table that claimed its own menu, and never take
a right-click -- or the window -- down with it.  The Help-menu entry goes above
the separator, once, and each window hook survives the other one failing.

One general test per unit, each walking a case table with pytest's ``subtests``
fixture.  The context-menu runner is a recorder installed by an autouse
fixture, so no ``QMenu.exec`` can reach a headless run.
"""
from __future__ import annotations

import types

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import (  # noqa: E402
    QAbstractTableModel,
    QEvent,
    QModelIndex,
    QObject,
    QPoint,
    Qt,
)
from PySide6.QtGui import QContextMenuEvent  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QLabel,
    QMainWindow,
    QMenu,
    QTableView,
    QTableWidget,
    QWidget,
)

from spacr.qt.widgets import feature_dictionary as fd  # noqa: E402

pytestmark = pytest.mark.qt

MEASUREMENT_HEADERS = [
    "object_label",
    "cell_area",
    "cell_channel_1_percentile_75",
    "nucleus_periphery_mean",
]


#: Tables a case hands only a part of (a header, a viewport) back from; the
#: view itself is held here so Python does not collect it mid-case.
_ALIVE: list = []


@pytest.fixture(autouse=True)
def menus(qapp):
    """Record context menus instead of opening one, and leave nothing behind."""
    shown: list = []
    fd.set_menu_runner(lambda menu, pos: shown.append(menu))
    yield shown
    _ALIVE.clear()
    fd.close_feature_dictionary()
    fd.remove_context_menu_filter(qapp)
    fd.set_menu_runner(None)


class _HeaderModel(QAbstractTableModel):
    """A two-row model whose column names are exactly what it is handed.

    ``None`` means the model does not name that column at all, which neither
    ``QTableWidget`` nor ``QStandardItemModel`` can express.
    """

    def __init__(self, headers):
        super().__init__()
        self._headers = list(headers)

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else 2

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._headers)

    def data(self, index, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and index.isValid():
            return "1"
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or orientation != Qt.Horizontal:
            return None
        return self._headers[section]


class _TableWithNoHeaderAccessor(QTableWidget):
    """A part-destructed view: its header accessor raises."""

    def horizontalHeader(self):
        raise RuntimeError("horizontalHeader is gone")


def _table(qtbot, headers=None, cls=QTableWidget):
    headers = list(headers or MEASUREMENT_HEADERS)
    table = cls(2, len(headers))
    table.setHorizontalHeaderLabels(headers)
    qtbot.addWidget(table)
    table.resize(800, 200)
    _ALIVE.append(table)
    return table


def _model_table(qtbot, headers):
    table = QTableView()
    model = _HeaderModel(headers)
    table.setModel(model)
    table._keep_model = model          # the view does not own it
    qtbot.addWidget(table)
    table.resize(800, 200)
    _ALIVE.append(table)
    return table


def _cell(table, column):
    return table.visualRect(table.model().index(0, column)).center()


def _event(widget, pos):
    return QContextMenuEvent(QContextMenuEvent.Mouse, pos,
                             widget.mapToGlobal(pos))


def _right_click(qapp, widget, pos):
    """Send a real context-menu event, the way a right-click does."""
    qapp.notify(widget, _event(widget, pos))


def _texts(menus):
    return [[action.text() for action in menu.actions()] for menu in menus]


def test_column_name_at_names_only_a_column_under_the_pointer(qtbot, subtests):
    """Header sections and cells answer; everything else answers ``None``."""
    def stray():
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget, QPoint(2, 2)

    def no_model():
        view = QTableView()
        qtbot.addWidget(view)
        view.resize(400, 200)
        _ALIVE.append(view)
        return view.viewport(), QPoint(20, 20)

    def bare_view():
        view = QTableView()
        qtbot.addWidget(view)
        return view, QPoint(2, 2)

    def row_header():
        return _table(qtbot).verticalHeader(), QPoint(4, 4)

    def past_last_column():
        header = _table(qtbot, ["cell_area"]).horizontalHeader()
        return header, QPoint(header.width() + 500, 2)

    def below_last_row():
        table = _table(qtbot)
        return table.viewport(), QPoint(20, table.viewport().height() - 4)

    def unnamed_column():
        table = _model_table(qtbot, [None, None, None])
        return table.viewport(), _cell(table, 1)

    def header_section():
        header = _table(qtbot).horizontalHeader()
        return header, QPoint(header.sectionPosition(2) + 4, 4)

    def a_cell():
        table = _table(qtbot)
        return table.viewport(), _cell(table, 1)

    def the_view_itself():
        table = _table(qtbot)
        pos = table.viewport().mapTo(table, _cell(table, 3))
        return table, pos

    def a_named_column_of_a_foreign_grid():
        table = _model_table(qtbot, [None, "Source", "Plate", "Filename"])
        return table.viewport(), _cell(table, 1)

    cases = [
        ("a widget that is not part of a table", stray, None),
        ("a table that has no model yet", no_model, None),
        ("a view with no model", bare_view, None),
        ("a row header", row_header, None),
        ("past the last column", past_last_column, None),
        ("below the last row", below_last_row, None),
        ("a column the model does not name", unnamed_column, None),
        ("a header section", header_section, "cell_channel_1_percentile_75"),
        ("a cell", a_cell, "cell_area"),
        ("the view itself, in view coordinates", the_view_itself,
         "nucleus_periphery_mean"),
        ("a named column of a foreign grid",
         a_named_column_of_a_foreign_grid, "Source"),
    ]
    for case_id, build, expected in cases:
        with subtests.test(msg=case_id):
            widget, pos = build()
            assert fd.column_name_at(widget, pos) == expected

    with subtests.test(msg="a widget inside a cell answers through the table"):
        table = _table(qtbot)
        label = QLabel("42")
        table.setCellWidget(0, 1, label)
        table.visualRect(table.model().index(0, 1))   # place the cell widget
        assert fd.column_name_at(label, QPoint(2, 2)) is None
        assert fd.column_name_at(
            table.viewport(),
            label.mapTo(table.viewport(), QPoint(2, 2))) == "cell_area"

    with subtests.test(msg="the model behind a part of a table"):
        table = _table(qtbot, ["cell_area"])
        widget, _pos = stray()
        assert fd._model_of(table.viewport()) is table.model()
        assert fd._model_of(table.horizontalHeader()) is table.model()
        assert fd._model_of(table) is table.model()
        assert fd._model_of(widget) is None


def test_the_installed_filter_offers_help_only_on_measurement_columns(
        qtbot, qapp, menus, subtests):
    """A real right-click opens "What is this?" exactly where it means something."""
    def header_section():
        header = _table(qtbot).horizontalHeader()
        return header, QPoint(header.sectionPosition(2) + 4, 4)

    def row_header():
        return _table(qtbot).verticalHeader(), QPoint(4, 4)

    def past_last_column():
        header = _table(qtbot).horizontalHeader()
        return header, QPoint(header.length() + 60, 4)

    def below_last_row():
        table = _table(qtbot)
        return table.viewport(), QPoint(20, table.viewport().height() - 4)

    def no_model_yet():
        view = QTableView()
        qtbot.addWidget(view)
        view.resize(400, 200)
        _ALIVE.append(view)
        return view.viewport(), QPoint(20, 20)

    def own_actions_menu():
        table = _table(qtbot)
        table.setContextMenuPolicy(Qt.ContextMenuPolicy.ActionsContextMenu)
        return table.viewport(), _cell(table, 1)

    def own_custom_menu():
        table = _table(qtbot)
        table.horizontalHeader().setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu)
        return table.viewport(), _cell(table, 1)

    def nameless_grid():
        table = _model_table(qtbot, [None, None, None])
        return table.viewport(), _cell(table, 1)

    def foreign_grid():
        table = _model_table(qtbot, [None, "Source", "Plate", "Filename"])
        return table.viewport(), _cell(table, 1)

    def unrecognised_cell_in_measurements():
        table = _table(qtbot, MEASUREMENT_HEADERS + ["someones_custom_column"])
        return table.viewport(), _cell(table, 4)

    def header_that_will_not_be_handed_over():
        table = _table(qtbot, cls=_TableWithNoHeaderAccessor)
        return table.viewport(), _cell(table, 1)

    def widget_inside_a_cell():
        table = _table(qtbot)
        label = QLabel("42")
        table.setCellWidget(0, 1, label)
        table.visualRect(table.model().index(0, 1))
        return label, QPoint(2, 2)

    cases = [
        # id, build, column the menu explains (None: no menu)
        ("a measurement header section", header_section,
         "cell_channel_1_percentile_75"),
        ("a row header", row_header, None),
        ("past the last column", past_last_column, None),
        ("below the last row", below_last_row, None),
        ("a table with no model yet", no_model_yet, None),
        ("a table carrying its own actions", own_actions_menu, None),
        ("a header with its own custom menu", own_custom_menu, None),
        ("a grid that names no columns", nameless_grid, None),
        ("a grid with unnamed columns is not measurements", foreign_grid,
         None),
        ("an unrecognised cell in a measurements table",
         unrecognised_cell_in_measurements, "someones_custom_column"),
        ("a view that will not hand over its header",
         header_that_will_not_be_handed_over, "cell_area"),
        ("a widget inside a cell", widget_inside_a_cell, "cell_area"),
    ]
    filt = fd.install_context_menu_filter(qapp)
    assert fd.install_context_menu_filter(qapp) is filt
    for case_id, build, column in cases:
        with subtests.test(msg=case_id):
            menus.clear()
            widget, pos = build()
            _right_click(qapp, widget, pos)
            if column is None:
                assert menus == []
            else:
                assert _texts(menus) == [[fd.CONTEXT_ACTION_TEXT]]
                assert menus[0].actions()[0].statusTip() == (
                    f"Explain the column “{column}”")

    with subtests.test(msg="triggering the entry opens the dictionary on it"):
        menus.clear()
        widget, pos = unrecognised_cell_in_measurements()
        _right_click(qapp, widget, pos)
        menus[0].actions()[0].trigger()
        assert "Not in the dictionary" in fd._DIALOG.panel.detail_text()

    with subtests.test(msg="objects that are not widgets are ignored"):
        menus.clear()
        event = QContextMenuEvent(QContextMenuEvent.Mouse, QPoint(1, 1),
                                  QPoint(1, 1))
        assert filt.eventFilter(QObject(), event) is False
        assert filt.eventFilter(QObject(), QEvent(QEvent.Type.User)) is False
        assert menus == []


def test_feature_help_filter_refuses_whatever_is_not_a_column_gesture(
        qtbot, menus, monkeypatch, subtests):
    """``eventFilter`` answers True only for a live right-click on a column."""
    columns = ["cell_area", "cell_perimeter", "cell_eccentricity"]
    filt = fd.FeatureHelpFilter()

    class _DyingEvent:
        def type(self):
            raise RuntimeError("Internal C++ object already deleted.")

    def header_click():
        header = _table(qtbot, columns).horizontalHeader()
        return header, _event(header, QPoint(5, 5))

    def row_header_through_the_table():
        table = _table(qtbot, columns)
        point = table.verticalHeader().mapTo(table, QPoint(2, 20))
        return table, _event(table, point)

    def plain_object():
        widget = QWidget()
        qtbot.addWidget(widget)
        return QObject(), _event(widget, QPoint(1, 1))

    cases = [
        # id, build (obj, event), patch column_name_at to raise, handled
        ("a measurement column offers an explanation", header_click, False,
         True),
        ("a row header is not a column gesture", row_header_through_the_table,
         False, False),
        ("a plain object is not a table", plain_object, False, False),
        ("an event dying before its type is read",
         lambda: (object(), _DyingEvent()), False, False),
        ("a missing event", lambda: (object(), None), False, False),
        ("a missing object",
         lambda: (None, QEvent(QEvent.Type.ContextMenu)), False, False),
        ("a resolver that raises lets the click through", header_click, True,
         False),
    ]
    for case_id, build, broken, handled in cases:
        with subtests.test(msg=case_id):
            menus.clear()
            obj, event = build()
            with monkeypatch.context() as patch:
                if broken:
                    def refuse(*_args, **_kwargs):
                        raise RuntimeError("the column parser blew up")
                    patch.setattr(fd, "column_name_at", refuse)
                assert filt.eventFilter(obj, event) is handled
            assert _texts(menus) == ([[fd.CONTEXT_ACTION_TEXT]] if handled
                                     else [])


def test_install_and_remove_context_menu_filter_hold_one_filter(
        qapp, monkeypatch, subtests):
    """One filter at a time; removal reports whether there was one to drop."""
    with subtests.test(msg="removed once and only once"):
        fd.install_context_menu_filter(qapp)
        assert fd.remove_context_menu_filter(qapp) is True
        assert fd._FILTER is None
        assert fd.remove_context_menu_filter(qapp) is False
        assert fd._FILTER is None

    with subtests.test(msg="a fresh install is possible after a removal"):
        first = fd.install_context_menu_filter(qapp)
        assert first is not None
        fd.remove_context_menu_filter(qapp)
        second = fd.install_context_menu_filter(qapp)
        assert isinstance(second, fd.FeatureHelpFilter)
        assert second is not first
        fd.remove_context_menu_filter(qapp)

    with subtests.test(msg="with no application left the filter is dropped"):
        installed = fd.install_context_menu_filter(qapp)
        assert installed is not None
        with monkeypatch.context() as patch:
            patch.setattr(QApplication, "instance", staticmethod(lambda: None))
            assert fd.remove_context_menu_filter() is True
            assert fd._FILTER is None
            assert fd.remove_context_menu_filter() is False
            patch.setattr(fd, "_FILTER", object())
            assert fd.remove_context_menu_filter() is True
            assert fd._FILTER is None

    with subtests.test(msg="with no application nothing is installed"):
        with monkeypatch.context() as patch:
            patch.setattr(fd, "QApplication",
                          types.SimpleNamespace(instance=lambda: None))
            assert fd.install_context_menu_filter() is None
            assert fd._FILTER is None


def _help_window(qtbot, *, separator=True):
    window = QMainWindow()
    qtbot.addWidget(window)
    menu = window.menuBar().addMenu("&Help")
    menu.addAction("About")
    if separator:
        menu.addSeparator()
        menu.addAction("Check for updates…")
    return window, menu


def test_install_help_action_puts_one_entry_above_the_separator(
        qtbot, monkeypatch, subtests):
    """The entry lands with the explanations, once, or the window is left alone."""
    with subtests.test(msg="above the first separator"):
        window, menu = _help_window(qtbot)
        action = fd.install_help_action(window)
        assert action.text() == fd.HELP_ACTION_TEXT
        texts = [a.text() for a in menu.actions()]
        assert texts == ["About", fd.HELP_ACTION_TEXT, "", "Check for updates…"]
        assert menu.actions()[2].isSeparator()
        assert fd.install_help_action(window) is None
        assert [a.text() for a in menu.actions()] == texts

    with subtests.test(msg="appended to a menu with no separator"):
        window, menu = _help_window(qtbot, separator=False)
        assert fd.install_help_action(window) is not None
        assert [a.text() for a in menu.actions()] == ["About",
                                                      fd.HELP_ACTION_TEXT]

    class _NoBar(QMainWindow):
        def menuBar(self):
            return None

    with subtests.test(msg="a window whose menu bar is not built yet"):
        window = _NoBar()
        qtbot.addWidget(window)
        assert fd.install_help_action(window) is None

    with subtests.test(msg="a menu bar with no Help menu"):
        window = QMainWindow()
        qtbot.addWidget(window)
        window.menuBar().addMenu("&File").addAction("Open")
        assert fd.install_help_action(window) is None

    with subtests.test(msg="an unreadable menu does not hide the Help menu"):
        class _Unreadable(QMenu):
            def title(self):
                raise RuntimeError(
                    "Internal C++ object (PySide6.QtWidgets.QMenu) already "
                    "deleted")

        window = QMainWindow()
        qtbot.addWidget(window)
        bar = window.menuBar()
        bar.addMenu(_Unreadable("&Tools", bar))
        help_menu = bar.addMenu("&Help")
        help_menu.addAction("About")
        assert fd.install_help_action(window) is not None
        assert [a.text() for a in help_menu.actions()] == [
            "About", fd.HELP_ACTION_TEXT]

    with subtests.test(msg="a menu whose C++ object is gone is stepped over"):
        window = QMainWindow()
        qtbot.addWidget(window)
        help_menu = window.menuBar().addMenu("&Help")

        class _Dead:
            def title(self):
                raise RuntimeError("Internal C++ object already deleted.")

        bar_type = type(window.menuBar())
        real = bar_type.findChildren
        with monkeypatch.context() as patch:
            patch.setattr(bar_type, "findChildren",
                          lambda self, *a, **k: [_Dead()] + list(
                              real(self, *a, **k)))
            assert fd._find_menu(window, "Help") is help_menu


def test_install_window_hooks_tries_each_hook_whatever_the_other_does(
        qtbot, qapp, monkeypatch, subtests):
    """Both hooks are attempted, every failure is swallowed, nothing returns."""
    def window_with_help():
        window, menu = _help_window(qtbot, separator=False)
        return window, menu

    def plain_widget():
        widget = QWidget()
        qtbot.addWidget(widget)
        return widget, None

    attempts: list = []

    def refuse_action(_window):
        attempts.append("action")
        raise RuntimeError("no menu bar yet")

    def refuse_filter(*_args, **_kwargs):
        attempts.append("filter")
        raise RuntimeError("no application object")

    cases = [
        # id, build, broken hooks, Help entries afterwards, filter installed
        ("both hooks work", window_with_help, (), 1, True),
        ("a window with no menu bar still gets the filter", plain_widget, (),
         None, True),
        ("a broken help hook keeps the filter", window_with_help, ("action",),
         0, True),
        ("a broken filter hook keeps the help entry", window_with_help,
         ("filter",), 1, False),
        ("both broken, both tried", window_with_help, ("action", "filter"),
         0, False),
    ]
    for case_id, build, broken, entries, filtered in cases:
        with subtests.test(msg=case_id):
            fd.remove_context_menu_filter(qapp)
            attempts.clear()
            window, menu = build()
            with monkeypatch.context() as patch:
                if "action" in broken:
                    patch.setattr(fd, "install_help_action", refuse_action)
                if "filter" in broken:
                    patch.setattr(fd, "install_context_menu_filter",
                                  refuse_filter)
                assert fd.install_window_hooks(window) is None
                assert fd.install_window_hooks(window) is None
            assert attempts == list(broken) * 2
            if menu is not None:
                texts = [a.text() for a in menu.actions()]
                assert texts.count(fd.HELP_ACTION_TEXT) == entries
            assert (fd._FILTER is not None) is filtered
