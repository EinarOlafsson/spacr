"""Montage provenance remains reachable and identifiable without a mouse."""

import numpy as np
import pandas as pd
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QAccessible, QCursor, QEnterEvent
from PySide6.QtWidgets import QApplication, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from spacr.qt.i18n import retranslate_widget_tree
from spacr.qt.widgets.cell_montage_view import _WellTab


@pytest.fixture
def montage(qtbot, activate_focus_window):
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    tab = _WellTab(("gene", "gene", "guide", "plate1_r1_c1"), "plate1_r1_c1")
    after = QPushButton("After")
    for widget in (before, tab, after):
        layout.addWidget(widget)
    rows = pd.DataFrame({"montage_well": ["plate1_r1_c1"] * 2,
                         "object_label": [4, 9], "pred": [0.1, 0.9],
                         "montage_candidate": [False, True]})
    crops = [np.full((16, 16, 3), value, dtype=np.uint8) for value in (35, 70)]
    host.resize(700, 650)
    qtbot.addWidget(host)
    host.show()
    activate_focus_window(host)
    qtbot.wait(40)
    tab.set_content(rows, crops, "Two synthetic measured objects.", columns=2)
    retranslate_widget_tree(host, "en")
    qtbot.wait(40)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    assert len(tab.thumbs()) == 2
    yield before, tab, after, rows, crops
    for dialog in tab._details:
        qtbot.addWidget(dialog)


def test_tab_reaches_both_montage_cells_and_leaves_the_panel(montage, qtbot):
    before, tab, after, _, _ = montage
    thumbs = set(tab.thumbs())
    reached = set()
    left_panel = False
    for _ in range(30):
        focused = QApplication.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key_Tab)
        focused = QApplication.focusWidget()
        if focused in thumbs:
            reached.add(focused)
        left_panel = left_panel or focused is after
        if focused is before:
            break
    assert reached == thumbs
    assert left_panel


@pytest.mark.parametrize("key,modifier", [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Space, Qt.NoModifier),
    pytest.param(Qt.Key_Enter, Qt.KeypadModifier, id="numeric-enter"),
])
def test_keyboard_opens_the_actual_cells_provenance_without_editing_data(
        montage, qtbot, key, modifier):
    _, tab, _, rows, crops = montage
    original_rows = rows.copy(deep=True)
    original_crops = [crop.copy() for crop in crops]
    thumb = tab.thumbs()[1]
    expected = thumb.toolTip()
    thumb.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(thumb, key, modifier)
    assert len(tab._details) == 1
    dialog = tab._details[0]
    assert dialog.isVisible()
    assert dialog.findChild(QPlainTextEdit).toPlainText() == expected
    assert "label 9" in expected
    pd.testing.assert_frame_equal(rows, original_rows)
    for actual, original in zip(tab.crops(), original_crops):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("index", [0, 1])
def test_accessibility_names_identify_the_actual_cell(montage, index):
    _, tab, _, _, _ = montage
    thumb = tab.thumbs()[index]
    interface = QAccessible.queryAccessibleInterface(thumb)
    assert interface.text(QAccessible.Name) == thumb.toolTip()
    assert interface.state().focusable


@pytest.mark.parametrize("modifier", [Qt.ControlModifier, Qt.AltModifier,
                                      Qt.ShiftModifier, Qt.MetaModifier,
                                      Qt.KeypadModifier | Qt.ControlModifier,
                                      Qt.KeypadModifier | Qt.AltModifier,
                                      Qt.KeypadModifier | Qt.ShiftModifier,
                                      Qt.KeypadModifier | Qt.MetaModifier])
def test_modified_activation_does_not_open_cell_details(montage, qtbot, modifier):
    _, tab, _, _, _ = montage
    thumb = tab.thumbs()[0]
    thumb.setFocus(Qt.TabFocusReason)
    for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
        qtbot.keyClick(thumb, key, modifier)
    assert tab._details == []


@pytest.mark.parametrize("key", [Qt.Key_Return, Qt.Key_Space])
def test_keypad_modifier_does_not_expand_other_provenance_keys(montage, qtbot, key):
    _, tab, _, _, _ = montage
    thumb = tab.thumbs()[0]
    thumb.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(thumb, key, Qt.KeypadModifier)
    assert tab._details == []


def test_focus_uses_the_existing_ring_independently_from_hover(montage):
    before, tab, after, _, _ = montage
    thumb = tab.thumbs()[1]
    highlight = thumb.highlight
    QCursor.setPos(QPoint(-500, -500))
    QApplication.sendEvent(thumb, QEvent(QEvent.Leave))
    before.setFocus()
    resting = thumb.grab().toImage()
    thumb.setFocus(Qt.TabFocusReason)
    assert thumb.hasFocus()
    focused = thumb.grab().toImage()
    assert focused != resting
    point = QPointF(5, 5)
    QApplication.sendEvent(thumb, QEnterEvent(point, point, point))
    QApplication.sendEvent(thumb, QEvent(QEvent.Leave))
    assert not thumb._hovered
    assert thumb.grab().toImage() == focused
    after.setFocus()
    assert thumb.grab().toImage() == resting
    QApplication.sendEvent(thumb, QEnterEvent(point, point, point))
    assert thumb.grab().toImage() == focused
    assert thumb.highlight == highlight
