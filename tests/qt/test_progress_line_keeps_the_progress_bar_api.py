"""ProgressLine stands in for QProgressBar, so its bar API has to behave like one.

Callers that used to hold a ``QProgressBar`` set the ends one at a time,
reset it, and read the range back; a line built without a detail row is
still handed detail text by code that does not know which kind it got.
"""
from __future__ import annotations

from spacr.qt.widgets.eliding import SEPARATOR, ProgressLine


def test_the_ends_can_be_set_one_at_a_time_and_read_back(qtbot):
    line = ProgressLine()
    qtbot.addWidget(line)

    line.setMinimum(10)
    line.setMaximum(20)
    line.setValue(15)

    assert (line.minimum(), line.maximum(), line.value()) == (10, 20, 15)
    assert line.percent() == 50
    assert line.text() == "50%"
    assert line.count_text() == line.text()


def test_reset_empties_the_bar_and_drops_the_number(qtbot):
    line = ProgressLine()
    qtbot.addWidget(line)
    line.setRange(0, 4)
    line.setFormat("%v / %m files")
    line.setValue(3)
    assert line.text() == f"3 / 4 files{SEPARATOR}75%"

    line.reset()

    assert line.value() < line.minimum()
    assert line.percent() is None
    assert line.text() == "0 / 4 files"


def test_a_line_without_a_detail_row_ignores_detail_text(qtbot):
    line = ProgressLine(detail=False)
    qtbot.addWidget(line)
    line.setRange(0, 2)
    line.setValue(1)

    line.set_detail("plate1_A01.tif")

    assert line.detail is None
    assert line.detail_text() == ""
    assert line.displayed_text() == "50%"
