"""Computed columns: what the panel says before it lets one be added.

A formula that does not parse, or that names a column the table does not
have, must be refused at the boxes rather than at the redraw. The panel's
whole job is turning each of those into a sentence under the two fields and
leaving Add disabled until there is something addable.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets.formula import ColumnFormula
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
