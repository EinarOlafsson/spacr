"""One column on Rows and on Columns is refused in the panel's notice.

Each well rejects a duplicate within itself, but nothing stops the same
column landing on Rows and on Columns. Building that spec raises
`PivotError`; `recompute` runs from the debounce timer, so the refusal has
to become the panel's message rather than escape the slot.
"""
from __future__ import annotations

import pandas as pd
import pytest

from spacr.qt.widgets import pivot_builder as PB

pytestmark = pytest.mark.qt


@pytest.fixture()
def panel(qtbot):
    widget = PB.PivotPanel()
    qtbot.addWidget(widget)
    widget.set_frame(pd.DataFrame({
        "plateID": ["p1", "p1", "p2", "p2"],
        "rowID": ["r1", "r2", "r1", "r2"],
        "cell_area": [1.0, 2.0, 3.0, 4.0],
    }))
    return widget


def test_a_column_on_both_axes_becomes_the_notice(panel):
    panel.wells[PB.AXIS_ROWS].set_columns(("plateID",))
    panel.wells[PB.AXIS_VALUES].set_columns(("cell_area",))
    assert panel.recompute() is not None

    panel.wells[PB.AXIS_COLS].set_columns(("plateID",))

    assert panel.recompute() is None
    assert panel._result is None
    assert "on both the row and the column axis" in panel.notice.text()
