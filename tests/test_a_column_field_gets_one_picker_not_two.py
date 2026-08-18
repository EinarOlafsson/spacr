"""A column field offers ONE column list, from the file the setting reads.

Asked for on 2026-08-17: "for the filter column there is an SQL buton this
should be a csv buton that can read the input csvs, the dependent variable
should have a simmilr CSV version of this buton."

In the regression module `filter_column` names a column of the INPUT CSVs. The
SQL picker opens the run's measurements.db, which a regression run need not
even have. Two buttons on one row offering two different column lists for one
setting is worse than either alone -- and once the CSV field wrapped the
input, the SQL button could no longer write into it, so it was a control that
did nothing.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

pytestmark = pytest.mark.qt


def _sql_buttons_around(widget):
    from PySide6.QtWidgets import QPushButton, QToolButton

    parent = widget.parentWidget()
    if parent is None:
        return []
    buttons = parent.findChildren(QPushButton) + parent.findChildren(QToolButton)
    return [b for b in buttons if "sql" in (b.text() or "").lower()]


def _screen(qtbot, key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(key)
    qtbot.addWidget(screen)
    return screen


def test_the_regression_filter_column_reads_the_input_csvs(qtbot):
    from spacr.qt.screens.settings_model import _CsvColumnField

    screen = _screen(qtbot, "regression")
    widget = screen._settings_model._widgets["filter_column"]
    assert isinstance(widget, _CsvColumnField)


def test_and_therefore_has_no_sql_button_beside_it(qtbot):
    screen = _screen(qtbot, "regression")
    widget = screen._settings_model._widgets["filter_column"]
    assert _sql_buttons_around(widget) == []


def test_the_dependent_variable_reads_the_input_csvs_too(qtbot):
    """It names a column of the score CSV, which is why it was never in
    COLUMN_TABLES -- pointing a measurements.db picker at it would be worse
    than having no picker at all."""
    from spacr.qt.screens.app_screen import COLUMN_TABLES
    from spacr.qt.screens.settings_model import _CsvColumnField

    screen = _screen(qtbot, "regression")
    assert "dependent_variable" not in COLUMN_TABLES
    widget = screen._settings_model._widgets["dependent_variable"]
    assert isinstance(widget, _CsvColumnField)


@pytest.mark.parametrize("app_key, key", [
    ("classify", "annotation_column"),
    ("classify_merged", "exclude"),
    ("ml_analyze", "annotation_column"),
    ("umap", "color_by"),
])
def test_a_field_that_really_does_name_a_database_column_keeps_its_picker(
        qtbot, app_key, key):
    """The guard is narrow on purpose.

    These settings DO name columns of measurements.db, and taking their
    picker away would leave a user typing a column name from memory -- which
    is how a typo silently creates a second near-identical column, the exact
    failure the SQL button exists to prevent.
    """
    screen = _screen(qtbot, app_key)
    widget = screen._settings_model._widgets[key]
    assert len(_sql_buttons_around(widget)) == 1


def test_the_rule_is_read_off_the_widget_not_a_second_table():
    """Whoever gave the field a CSV picker has already decided which file the
    setting reads. A per-module table here would be a second place for that
    decision to be made differently."""
    import inspect

    from spacr.qt.screens.app_screen import AppScreen

    source = inspect.getsource(AppScreen._attach_column_picker)
    assert "isinstance(widget, _CsvColumnField)" in source
