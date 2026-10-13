"""Download buttons survive header changes and table/layout lifecycles."""

import pytest
import shiboken6
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (
    QHBoxLayout, QPushButton, QTableWidget, QVBoxLayout, QWidget,
)

from spacr.qt.widgets.column_aligned_row import (
    ColumnAlignedRow, TRAILING_SPACING, align_row_to_columns,
)


@pytest.fixture
def strip_and_table(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    outer = QVBoxLayout(window)
    strip = QWidget(window)
    table = QTableWidget(2, 3, window)
    outer.addWidget(strip)
    outer.addWidget(table)
    header = table.horizontalHeader()
    header.setSectionsMovable(True)
    for column, width in enumerate((160, 190, 220)):
        table.setColumnWidth(column, width)
    window.resize(900, 300)
    window.show()
    qtbot.waitExposed(window)
    # pytest-qt keeps weak references; retain the owning window until teardown.
    yield strip, table, header


def test_replacing_an_existing_layout_preserves_buttons_and_header_widths(
        strip_and_table, qtbot):
    strip, table, header = strip_and_table
    old = QHBoxLayout(strip)
    score, count = QPushButton("Score"), QPushButton("Count")
    old.addWidget(score)
    old.addStretch()
    old.addWidget(count)
    widths = [header.sectionSize(i) for i in range(3)]
    row = align_row_to_columns(strip, header, [(score, 0), (count, 1)])
    qtbot.wait(10)
    assert not shiboken6.isValid(old)
    assert score.parentWidget() is strip and count.parentWidget() is strip
    assert row.count() == 2
    assert row.itemAt(0).widget() is score
    assert row.itemAt(-1) is None and row.itemAt(2) is None
    assert align_row_to_columns(strip, header, []) is row
    assert [header.sectionSize(i) for i in range(3)] == widths
    for button, column in ((score, 0), (count, 1)):
        section_left = header.viewport().mapToGlobal(
            QPoint(header.sectionViewportPosition(column), 0)).x()
        button_left = button.mapToGlobal(QPoint(0, 0)).x()
        assert abs(button_left + button.width() / 2
                   - section_left - header.sectionSize(column) / 2) <= 1


@pytest.mark.parametrize("unmapped", [None, -1, 7, 1])
def test_unavailable_columns_leave_their_buttons_in_the_trailing_run(
        strip_and_table, qtbot, unmapped):
    strip, table, header = strip_and_table
    fixed, trailing = QPushButton("Score"), QPushButton("Image crops")
    row = align_row_to_columns(strip, header, [(fixed, 0), (trailing, unmapped)])
    fixed.show()
    trailing.show()
    if unmapped == 1:
        table.hideColumn(1)
    qtbot.wait(10)
    row.setGeometry(strip.rect())
    right_edge = strip.mapFromGlobal(header.viewport().mapToGlobal(
        QPoint(header.sectionViewportPosition(0) + header.sectionSize(0), 0))).x()
    assert trailing.isVisible()
    assert trailing.x() == right_edge + TRAILING_SPACING
    assert fixed.geometry().right() < trailing.x()


def test_reordering_columns_moves_buttons_without_changing_widths(
        strip_and_table, qtbot):
    strip, table, header = strip_and_table
    buttons = [QPushButton(label) for label in ("Score", "Count", "Database")]
    row = align_row_to_columns(strip, header, list(zip(buttons, range(3))))
    for button in buttons:
        button.show()
    qtbot.wait(10)
    widths = [header.sectionSize(i) for i in range(3)]
    header.moveSection(2, 0)
    qtbot.wait(10)
    assert buttons[2].x() < buttons[0].x() < buttons[1].x()
    assert [header.sectionSize(i) for i in range(3)] == widths
    assert row.expandingDirections() == Qt.Horizontal
    assert row.minimumSize().width() == 0
    assert row.sizeHint().height() >= row.minimumSize().height()


def test_deleting_the_table_keeps_the_download_buttons_usable(
        strip_and_table, qtbot):
    strip, table, header = strip_and_table
    first, second = QPushButton("Score"), QPushButton("Count")
    row = align_row_to_columns(strip, header, [(first, 0), (second, 1)])
    first.show()
    second.show()
    qtbot.wait(10)
    shiboken6.delete(table)
    assert not shiboken6.isValid(header)
    row.setGeometry(QRect(0, 0, 700, 40))
    assert first.isEnabled() and second.isEnabled()
    assert first.x() == TRAILING_SPACING
    assert second.x() >= first.x() + first.width() + TRAILING_SPACING
    assert align_row_to_columns(strip, header, []) is None


def test_qt_can_remove_and_reinsert_a_trailing_button(strip_and_table, qtbot):
    strip, table, header = strip_and_table
    row = ColumnAlignedRow(header, strip)
    button = QPushButton("Image crops")
    row.addWidget(button)
    qtbot.wait(10)
    assert row.count() == 1
    item = row.takeAt(0)
    assert item.widget() is button
    assert row.count() == 0 and row.itemAt(0) is None
    assert row.takeAt(0) is None and row.takeAt(-1) is None
    assert shiboken6.isValid(button)
    row.addItem(item)
    row.activate()
    assert row.count() == 1 and row.itemAt(0).widget() is button


def test_failed_reentrant_caption_refit_does_not_disable_future_refits(
        strip_and_table, qtbot, caplog):
    strip, table, header = strip_and_table
    button = QPushButton("Count")
    row = align_row_to_columns(strip, header, [(button, 1)])
    button.show()
    qtbot.wait(10)
    calls = []

    def refit():
        calls.append(button.text())
        row.invalidate()  # A header refit can itself invalidate the strip.
        if len(calls) == 1:
            raise RuntimeError("the screen is still being laid out")

    row._on_invalidate = refit
    with caplog.at_level("DEBUG", logger="spacr.qt.widgets.column_aligned_row"):
        row.invalidate()
    assert calls == ["Count"]
    assert "column refit callback failed" in caplog.text
    button.setText("Contagem")
    row.invalidate()
    assert calls[-1] == "Contagem" and len(calls) >= 2
    assert not row._in_callback
