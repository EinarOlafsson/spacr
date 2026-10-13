"""The column text size (Ctrl + wheel, item 529) at its edges.

A column is registered once however often it is offered, and "nothing" is
never registered. A column whose widget has been deleted is neither
restyled nor refitted. A widget outside every column, or something that is
not a widget at all, belongs to no column. The sheet a column inherits is
built from every ancestor that carries one, and from the application's
only when the application has one. With no application running there is
nothing to hold the gesture, so nothing is installed and nothing is
registered.
"""
import shiboken6
from PySide6.QtWidgets import QApplication, QLabel, QWidget

from spacr.qt import live_zoom
from spacr.qt.live_zoom import ColumnTextScale


def _column(qtbot, sheet=""):
    host = QWidget()
    qtbot.addWidget(host)
    host.setStyleSheet(sheet)
    column = QWidget(host)
    QLabel("text", column)
    return host, column


def test_a_column_offered_twice_is_registered_once(qtbot):
    scale = ColumnTextScale()
    _host, column = _column(qtbot)

    scale.register(column)
    scale.register(column)
    scale.register(None)

    assert scale.roots() == [column]


def test_the_inherited_sheet_carries_every_ancestors_rules(qtbot):
    host, column = _column(qtbot, "QLabel { font-size: 13px; }")

    inherited = ColumnTextScale._inherited_sheet(column)

    assert "font-size: 13px" in inherited


def test_without_an_application_sheet_only_the_ancestors_count(
        qtbot, monkeypatch):
    host, column = _column(qtbot, "QLabel { font-size: 11px; }")

    class _NoSheet:
        @staticmethod
        def instance():
            return type("App", (), {"styleSheet": lambda self: ""})()

    monkeypatch.setattr(live_zoom, "QApplication", _NoSheet)

    assert ColumnTextScale._inherited_sheet(column) == (
        "QLabel { font-size: 11px; }")


def test_a_deleted_column_is_not_restyled_or_refitted(qtbot):
    scale = ColumnTextScale()
    _host, column = _column(qtbot)
    scale.register(column)
    shiboken6.delete(column)

    assert scale.restyle(column) is False
    ColumnTextScale._refit(column)
    assert scale.roots() == []


def test_only_a_widget_inside_a_column_belongs_to_one(qtbot):
    scale = ColumnTextScale()
    assert scale.root_of(QLabel()) is None
    host, column = _column(qtbot)
    scale.register(column)
    inside = column.findChild(QLabel)

    assert scale.root_of(inside) is column
    assert scale.root_of("not a widget") is None
    assert scale.root_of(host) is None


def test_with_no_application_nothing_is_installed_or_registered(
        qtbot, monkeypatch):
    _host, column = _column(qtbot)

    class _NoApp:
        @staticmethod
        def instance():
            return None

    monkeypatch.setattr(live_zoom, "QApplication", _NoApp)

    assert live_zoom.install_column_text_scale() is None
    assert live_zoom.register_text_column(column) is None
    assert QApplication.instance() is not None
