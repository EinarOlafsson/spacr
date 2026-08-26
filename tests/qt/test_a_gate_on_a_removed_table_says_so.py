"""A gate the working set can no longer answer keeps its row and says why.

Removing a table from the Gate Editor's working set takes its measurements
out of the frame. `GateSet.stats` is all-or-nothing -- it walks every gate
and the first one whose columns are absent raises -- so one nucleus gate
used to blank the counts of every CELL gate as well, which is the opposite
of "removing a chip removes that table's measurements": it removed everyone's
numbers.
"""

import pandas as pd
import pytest

from spacr.qt.widgets.gate_editor import GateTree
from spacr.qt.widgets.gate_spec import GateSet, RectGate


def _gates():
    return (GateSet()
            .add(RectGate(name="big_cell", x_column="cell_area",
                          y_column="cell_intensity",
                          x_low=0.0, x_high=1e9, y_low=0.0, y_high=1e9))
            .add(RectGate(name="bright_nucleus", x_column="nucleus_area",
                          y_column="nucleus_intensity",
                          x_low=0.0, x_high=1e9, y_low=0.0, y_high=1e9)))


@pytest.fixture
def tree(qt_theme_applied, qtbot):
    widget = GateTree()
    qtbot.addWidget(widget)
    return widget


def _rows(tree):
    root = tree.tree
    return {root.topLevelItem(i).text(0): root.topLevelItem(i)
            for i in range(root.topLevelItemCount())}


def test_a_gate_that_still_applies_keeps_its_count(tree):
    """The regression: the nucleus gate must not cost the cell gate its
    numbers."""
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                          "cell_intensity": [10.0, 20.0, 30.0]})
    tree.set_gates(_gates(), frame)
    rows = _rows(tree)
    assert rows["big_cell"].text(1) == "3", (
        "a gate on a table that is gone blanked the counts of one that is "
        "still here")


def test_the_gate_that_cannot_be_answered_says_so(tree):
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                          "cell_intensity": [10.0, 20.0, 30.0]})
    tree.set_gates(_gates(), frame)
    rows = _rows(tree)
    assert "bright_nucleus" in rows, "the gate vanished instead of saying so"
    assert rows["bright_nucleus"].text(1) == GateTree.UNAVAILABLE
    assert "nucleus_area" in rows["bright_nucleus"].toolTip(0)


def test_nothing_changes_when_every_gate_applies(tree):
    frame = pd.DataFrame({"cell_area": [1.0, 2.0, 3.0],
                          "cell_intensity": [10.0, 20.0, 30.0],
                          "nucleus_area": [1.0, 2.0, 3.0],
                          "nucleus_intensity": [1.0, 2.0, 3.0]})
    tree.set_gates(_gates(), frame)
    rows = _rows(tree)
    assert rows["big_cell"].text(1) == "3"
    assert rows["bright_nucleus"].text(1) == "3"
