"""Computed columns: what the panel says before it lets one be added.

A formula that does not parse, or that names a column the table does not
have, must be refused at the boxes rather than at the redraw. The panel's
whole job is turning each of those into a sentence under the two fields and
leaving Add disabled until there is something addable.
"""
from __future__ import annotations

import pandas as pd
import pytest
from PySide6.QtCore import Qt

from spacr.qt.widgets.formula import ColumnFormula, FormulaError
from spacr.qt.widgets.formula_editor import FormulaPanel


@pytest.fixture
def table():
    """A small numeric table with two columns to combine."""
    return pd.DataFrame({"area": [10.0, 20.0, 30.0],
                         "perimeter": [4.0, 8.0, 12.0]})


@pytest.fixture
def panel(qtbot):
    widget = FormulaPanel()
    qtbot.addWidget(widget)
    return widget


def test_nothing_is_computed_before_a_table_is_loaded(panel):
    """A host can call ``computed_frame`` unconditionally and get None."""
    assert panel.computed_frame() is None
    assert panel.results() == []


def test_a_formula_naming_a_column_the_table_lacks_is_refused_with_the_name(
        panel, table):
    """The refusal names the missing column, which is the whole diagnosis."""
    panel.set_frame(table)
    assert panel.add_formula(ColumnFormula("bad", "area / girth")) is False
    assert "girth" in panel.status()
    assert panel.computed_frame().equals(table)


def test_an_unparseable_expression_is_refused_at_the_boxes(panel, table):
    """Add stays disabled and the parse error is what is shown."""
    panel.set_frame(table)
    panel._name.setText("ratio")
    panel._expression.setText("area / /")
    panel._validate()
    assert panel._add.isEnabled() is False
    assert panel.status()
    assert panel.commit() is False


def test_a_formula_that_parses_but_cannot_be_applied_is_refused_on_commit(
        panel, table):
    """Parsing is not applying; the apply failure comes back from commit."""
    panel.set_frame(table)
    panel._name.setText("ratio")
    panel._expression.setText("area / girth")
    assert panel.commit() is False
    assert "girth" in panel.status()


def test_half_filled_boxes_are_not_a_formula(panel, table):
    """A name with no expression is a user still typing, not an error."""
    panel.set_frame(table)
    panel._name.setText("ratio")
    panel._expression.setText("")
    assert panel._current_formula() is None
    assert panel.commit() is False


def test_a_valid_formula_clears_the_boxes_and_announces_the_column(
        panel, table):
    """After a successful add the fields are ready for the next one."""
    panel.set_frame(table)
    panel._name.setText("compactness")
    panel._expression.setText("area / perimeter ** 2")
    assert panel.commit() is True
    assert panel._name.text() == "" and panel._expression.text() == ""
    assert panel._replace.isChecked() is False
    assert "added compactness" in panel.status()
    assert panel.computed_frame()["compactness"].iloc[0] == pytest.approx(
        10.0 / 16.0)


def test_a_formula_that_parses_with_no_table_says_it_needs_one(panel):
    """Add is enabled on a parseable name so the formula survives the load."""
    panel._name.setText("ratio")
    panel._expression.setText("area / perimeter")
    panel._validate()
    assert panel._add.isEnabled() is True
    assert panel.status() == "parses — load a table to see the values"


def test_removing_with_nothing_selected_does_nothing(panel, table):
    """The Remove button with an empty list is not an error."""
    panel.set_frame(table)
    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    panel._list.setCurrentItem(None)
    panel.remove_selected()
    assert panel.formulas().names == ("ratio",)


def test_removing_the_selected_formula_drops_its_column(panel, table, qtbot):
    """The selection carries the formula's name in its user role."""
    panel.set_frame(table)
    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    panel._list.setCurrentRow(0)
    assert panel._list.currentItem().data(Qt.UserRole) == "ratio"
    with qtbot.waitSignal(panel.formulas_changed):
        panel.remove_selected()
    assert panel.formulas().names == ()
    assert "ratio" not in panel.computed_frame().columns


def test_removing_a_formula_that_is_not_defined_changes_nothing(panel, table):
    """A name nobody defined is not a column to drop."""
    panel.set_frame(table)
    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    panel.remove("never_defined")
    assert panel.formulas().names == ("ratio",)


def test_clearing_an_empty_panel_announces_nothing(panel, table, qtbot):
    """No formulas means no change, so no listener is woken."""
    panel.set_frame(table)
    with qtbot.assertNotEmitted(panel.formulas_changed):
        panel.clear()

    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    with qtbot.waitSignal(panel.formulas_changed):
        panel.clear()
    assert panel.formulas().names == ()
    assert panel.results() == []


def test_a_preview_over_a_long_table_says_how_many_rows_it_read(panel):
    """The note has to say the values shown are not the whole column."""
    from spacr.qt.widgets.formula_editor import PREVIEW_ROWS

    long_table = pd.DataFrame({"area": range(PREVIEW_ROWS * 3),
                               "perimeter": range(1, PREVIEW_ROWS * 3 + 1)})
    panel.set_frame(long_table)
    panel._name.setText("ratio")
    panel._expression.setText("area / perimeter")
    panel._validate()
    assert f"previewed on the first {PREVIEW_ROWS:,} rows" in panel.status()


def test_a_whole_table_formula_says_it_moves_with_the_table(panel, table):
    """A column built from an aggregate changes when the table is filtered."""
    panel.set_frame(table)
    panel._name.setText("share")
    panel._expression.setText("area / sum(area)")
    panel._validate()
    assert "uses the whole table" in panel.status()


def test_an_invalid_name_is_reported_rather_than_stored(panel, table):
    """The formula never gets constructed, so it cannot reach a redraw."""
    panel.set_frame(table)
    with pytest.raises(FormulaError):
        ColumnFormula("2 bad name", "area")
    panel._name.setText("2 bad name")
    panel._expression.setText("area")
    panel._validate()
    assert panel._add.isEnabled() is False
    assert panel.status()


def test_pointing_the_panel_at_no_table_computes_nothing(panel, table):
    """Closing the table leaves the formulas defined and the values gone."""
    panel.set_frame(table)
    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    panel.set_frame(None)
    assert panel.computed_frame() is None
    assert panel.results() == []
    assert panel.formulas().names == ("ratio",)


def test_a_kept_formula_reports_the_column_a_new_table_does_not_have(panel,
                                                                    table):
    """Formulas survive a reload on purpose, so a mismatch has to be said."""
    panel.set_frame(table)
    panel.add_formula(ColumnFormula("ratio", "area / perimeter"))
    panel.set_frame(pd.DataFrame({"area": [1.0, 2.0]}))
    assert "perimeter" in panel.status()
    assert panel.results() == []
