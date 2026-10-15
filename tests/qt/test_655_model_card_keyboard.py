"""Model cards can be paged and enlarged without reaching for a mouse."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from spacr import model_cards as cards
from spacr.qt.widgets.model_card_viewer import ModelCardViewer

from .test_model_card_viewer import a_card, png_bytes


@pytest.fixture
def card_ui(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(cards, "_spacr_home", lambda: tmp_path / "home")
    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    viewer = ModelCardViewer(threaded=False, fetch=lambda _url: png_bytes(48, 36))
    after = QPushButton("After")
    for widget in (before, viewer, after):
        layout.addWidget(widget)
    viewer.set_card(a_card())
    host.resize(480, 360)
    host.show()
    host.activateWindow()
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield before, viewer, after


def test_tab_and_space_reach_both_page_buttons(card_ui, qtbot):
    before, viewer, _after = card_ui
    pages = []
    viewer.page_changed.connect(pages.append)
    QTest.keyClick(before, Qt.Key_Tab)
    qtbot.waitUntil(viewer.hasFocus)
    QTest.keyClick(viewer, Qt.Key_Tab)
    assert viewer._next.hasFocus(), "Next is absent from the keyboard order"
    QTest.keyClick(viewer._next, Qt.Key_Space)
    assert viewer.page() == 1
    QTest.keyClick(viewer._next, Qt.Key_Backtab)
    assert viewer._prev.hasFocus(), "Previous is absent from the keyboard order"
    QTest.keyClick(viewer._prev, Qt.Key_Space)
    assert viewer.page() == 0 and pages == [1, 0]
    assert viewer._dialog is None


@pytest.mark.parametrize("key,modifier", [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Space, Qt.NoModifier),
    pytest.param(Qt.Key_Enter, Qt.KeypadModifier, id="numeric-enter"),
])
def test_focused_card_opens_the_current_page_larger(card_ui, qtbot, key, modifier):
    before, viewer, _after = card_ui
    viewer.go_to(1)
    QTest.keyClick(before, Qt.Key_Tab)
    assert viewer.hasFocus()
    QTest.keyClick(viewer, key, modifier)
    assert viewer._dialog is not None, "keyboard activation did not enlarge the card"
    qtbot.addWidget(viewer._dialog)
    big = viewer._dialog.findChild(ModelCardViewer)
    assert big.card() is viewer.card() and big.page() == 1
    big.setFocus(Qt.TabFocusReason)
    QTest.keyClick(big, key, modifier)
    assert big._dialog is None, "an enlarged viewer opened another enlarged viewer"


@pytest.mark.parametrize("key,modifier", [
    (Qt.Key_Return, Qt.NoModifier), (Qt.Key_Enter, Qt.NoModifier),
    (Qt.Key_Space, Qt.NoModifier),
    pytest.param(Qt.Key_Enter, Qt.KeypadModifier, id="numeric-enter"),
])
def test_page_button_activation_never_opens_an_enlarged_dialog(card_ui, key, modifier):
    _before, viewer, _after = card_ui
    viewer._next.setFocus(Qt.TabFocusReason)
    assert viewer._next.hasFocus()
    QTest.keyClick(viewer._next, key, modifier)
    assert viewer._dialog is None


def test_modified_activation_and_an_empty_card_do_not_open_a_dialog(card_ui):
    before, viewer, _after = card_ui
    QTest.keyClick(before, Qt.Key_Tab)
    for modifier in (Qt.ControlModifier, Qt.AltModifier,
                     Qt.ShiftModifier, Qt.MetaModifier):
        for key in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            for extra in (Qt.NoModifier, Qt.KeypadModifier):
                QTest.keyClick(viewer, key, modifier | extra)
                assert viewer._dialog is None
    for key in (Qt.Key_Return, Qt.Key_Space):
        QTest.keyClick(viewer, key, Qt.KeypadModifier)
        assert viewer._dialog is None
    viewer.set_card(None)
    QTest.keyClick(viewer, Qt.Key_Return)
    assert viewer._dialog is None
    QTest.keyClick(viewer, Qt.Key_Enter, Qt.KeypadModifier)
    assert viewer._dialog is None


def test_focused_viewer_has_a_visible_border(card_ui):
    before, viewer, after = card_ui
    unfocused = viewer.grab().toImage()
    QTest.keyClick(before, Qt.Key_Tab)
    assert viewer.hasFocus()
    focused = viewer.grab().toImage()
    x = focused.width() // 2
    assert focused.pixelColor(x, 0) != unfocused.pixelColor(x, 0)
    after.setFocus(Qt.TabFocusReason)
    assert viewer.grab().toImage().pixelColor(x, 0) == unfocused.pixelColor(x, 0)


def test_accessible_card_name_follows_model_changes_and_clearing(card_ui):
    _before, viewer, _after = card_ui
    interface = QAccessible.queryAccessibleInterface(viewer)
    assert interface.text(QAccessible.Name) == viewer.card()["display_name"]
    viewer.set_card(a_card("toxoplasma_pv_v4"))
    assert interface.text(QAccessible.Name) == viewer.card()["display_name"]
    viewer.set_card(None)
    assert interface.text(QAccessible.Name) == ""
