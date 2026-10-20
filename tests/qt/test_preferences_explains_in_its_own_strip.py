"""Preferences explains a control in its strip, in paragraph form."""
from __future__ import annotations

import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QTabWidget,
    QTextEdit,
)

from spacr.qt import preferences as P
from spacr.qt.widgets.hint_bar import HintBar


def test_the_strip_sits_above_the_buttons(qtbot):
    """Asked for 2026-08-28: above Defaults / Close / Open, not below."""
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    layout = dlg.layout()
    bar = dlg.findChildren(HintBar)[0]
    box = dlg.findChildren(QDialogButtonBox)[0]
    assert layout.indexOf(bar) < layout.indexOf(box)


def test_the_standing_sentences_are_gone(qtbot):
    """Two paragraphs sat under the tabs on every visit."""
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    said = " ".join((l.text() or "") for l in dlg.findChildren(QLabel))
    assert "apply instantly" not in said
    assert "Colour-blind mode affects plot colours" not in said


def test_the_resource_buttons_explain_in_a_paragraph():
    """The bulleted promise belongs to the confirmation, not to a hover."""
    from spacr.qt import resource_cleanup

    for action in ("ram", "vram", "cpu", "disk"):
        short = resource_cleanup.summary_text(action)
        long = resource_cleanup.confirmation_text(action)
        assert "•" not in short, f"{action} still hovers a bulleted list"
        assert len(short) < len(long)
        # The limit is the part a user is uneasy about; it must survive.
        assert short.strip().endswith(".")
    # And the long form is untouched, because the confirmation still needs it.
    assert "•" in resource_cleanup.confirmation_text("ram")


def test_the_strip_cannot_grow_without_bound(qtbot):
    """A strip that resized made the dialog jump as the pointer moved."""
    bar = HintBar()
    qtbot.addWidget(bar)
    assert bar.maximumHeight() < 200
    tall = bar.maximumHeight()
    bar.setText("word " * 400)
    assert bar.maximumHeight() == tall


def test_translated_long_help_scrolls_without_moving_preferences(qtbot,
                                                                  monkeypatch):
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    dlg.show()
    QApplication.processEvents()
    bar = dlg.findChild(HintBar)
    view = bar.findChild(QTextEdit, "HintBarText")
    tabs = dlg.findChild(QTabWidget)
    buttons = dlg.findChild(QDialogButtonBox)
    short = QPushButton("Short", dlg)
    four = QPushButton("Four", dlg)
    long = QPushButton("Long", dlg)
    bar.explain(short, "First line.\nSecond line.")
    four_lines = "First line.\nSecond line.\nThird line.\nFourth line."
    bar.explain(four, four_lines)
    bar.explain(long, "long-help-source")
    translated = "\n".join(
        f"Aide détaillée, ligne {number}." for number in range(13))
    original_translate = bar._translated
    monkeypatch.setattr(
        bar, "_translated",
        lambda text: translated if text == "long-help-source"
        else original_translate(text))

    def geometry():
        return (dlg.size(), tabs.geometry(), bar.geometry(),
                buttons.geometry())

    initial = geometry()
    bar.eventFilter(short, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert view.toPlainText() == bar.text() == "First line.\nSecond line."
    assert view.verticalScrollBar().maximum() == 0
    assert geometry() == initial

    bar.eventFilter(four, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert view.toPlainText() == bar.text() == four_lines
    assert view.verticalScrollBar().maximum() == 0
    assert geometry() == initial

    bar.eventFilter(long, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert view.toPlainText() == bar.text() == translated
    assert bar.accessibleName() == translated
    assert view.verticalScrollBar().maximum() > 0
    view.setFocus()
    QTest.keyClick(view, Qt.Key_End)
    assert view.verticalScrollBar().value() == view.verticalScrollBar().maximum()
    assert geometry() == initial
    assert not long.toolTip()

    font = bar.font()
    font.setPointSize(font.pointSize() + 9)
    bar.setFont(font)
    QApplication.processEvents()
    assert view.fontInfo().pixelSize() == bar.fontInfo().pixelSize()
    margins = bar.contentsMargins()
    lines = max(bar.fontMetrics().lineSpacing() * 4, bar._four_painted_rows())
    assert bar.height() == max(
        lines + 12, lines + margins.top() + margins.bottom()
        + bar._resize_handle.height())
    scaled = geometry()
    bar.eventFilter(short, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert view.verticalScrollBar().maximum() == 0
    bar.eventFilter(long, QEvent(QEvent.Enter))
    QApplication.processEvents()
    assert view.toPlainText() == translated
    assert view.verticalScrollBar().maximum() > 0
    assert geometry() == scaled


def test_long_help_survives_moving_from_its_button_into_the_scroll_area(qtbot):
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    dlg.show()
    QApplication.processEvents()
    bar = dlg.findChild(HintBar)
    view = bar.findChild(QTextEdit, "HintBarText")
    button = dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Save)
    long_help = "\n".join(f"Instruction {number}" for number in range(13))
    bar.explain(button, long_help)

    bar.eventFilter(button, QEvent(QEvent.Enter))
    assert view.verticalScrollBar().maximum() > 0
    bar.eventFilter(button, QEvent(QEvent.Leave))
    QTest.mouseMove(view.viewport(), view.viewport().rect().center())
    QApplication.processEvents()
    assert bar.text() == long_help
    wheel_at = view.viewport().rect().center()
    wheel = QWheelEvent(QPointF(wheel_at),
                        QPointF(view.viewport().mapToGlobal(wheel_at)),
                        QPoint(), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                        Qt.NoScrollPhase, False)
    QApplication.sendEvent(view.viewport(), wheel)
    assert view.verticalScrollBar().value() > 0
    QTest.keyClick(view, Qt.Key_End)
    assert view.verticalScrollBar().value() == view.verticalScrollBar().maximum()

    bar.eventFilter(bar._resize_handle, QEvent(QEvent.Enter))
    assert bar.text() == long_help
    grip = bar._resize_handle
    grip_point = grip.rect().center()
    QTest.mousePress(grip, Qt.LeftButton, pos=grip_point)
    QTest.mouseMove(grip, QPoint(grip_point.x(), grip_point.y() - 20))
    QTest.mouseRelease(grip, Qt.LeftButton,
                       pos=QPoint(grip_point.x(), grip_point.y() - 20))
    assert bar.text() == long_help
    other = dlg.findChild(QDialogButtonBox).button(QDialogButtonBox.Cancel)
    bar.explain(other, "Other help")
    bar.eventFilter(other, QEvent(QEvent.Enter))
    assert bar.text() == "Other help"
    bar.eventFilter(other, QEvent(QEvent.Leave))
    assert bar.text() == bar._translated(bar._default)


def test_the_help_edge_resizes_and_remembers_without_hiding_buttons(
        qtbot, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    dlg.show()
    QApplication.processEvents()
    bar = dlg.findChild(HintBar)
    handle = bar._resize_handle
    tabs = dlg.findChild(QTabWidget)
    buttons = dlg.findChild(QDialogButtonBox)
    original_height = bar.height()
    original_tabs = tabs.height()
    button_geometry = buttons.geometry()
    point = handle.rect().center()

    QTest.mousePress(handle, Qt.LeftButton, pos=point)
    QTest.mouseMove(handle, QPoint(point.x(), point.y() - 70))
    QTest.mouseRelease(handle, Qt.LeftButton,
                       pos=QPoint(point.x(), point.y() - 70))
    QApplication.processEvents()
    assert bar.height() == original_height + 70
    assert tabs.height() == original_tabs - 70
    assert buttons.geometry() == button_geometry
    assert buttons.isVisibleTo(dlg)

    bar.setText("A very long translated explanation. " * 50)
    QApplication.processEvents()
    assert bar._view.verticalScrollBar().maximum() > 0
    assert bar.height() == original_height + 70
    assert buttons.geometry() == button_geometry

    original_font = bar.font()
    larger = bar.font()
    larger.setPointSize(larger.pointSize() + 15)
    bar.setFont(larger)
    QApplication.processEvents()
    margins = bar.contentsMargins()
    lines = max(bar.fontMetrics().lineSpacing() * 4, bar._four_painted_rows())
    assert bar.height() == max(
        original_height + 70, lines + 12,
        lines + margins.top() + margins.bottom() + handle.height())
    assert buttons.geometry() == button_geometry
    bar.setFont(original_font)
    QApplication.processEvents()
    assert bar.height() == original_height + 70

    dlg.close()
    reopened = P.PreferencesDialog(None)
    qtbot.addWidget(reopened)
    reopened.show()
    for _ in range(3):
        QApplication.processEvents()
    restored = reopened.findChild(HintBar)
    assert restored.height() == original_height + 70
    assert reopened.findChild(QDialogButtonBox).isVisibleTo(reopened)


@pytest.mark.parametrize("scheme", ("dark", "light"))
def test_four_styled_help_lines_fit_without_scrolling(qtbot, scheme):
    from spacr.qt import theme

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    dlg.setStyleSheet(theme.stylesheet(scheme))
    dlg.show()
    QApplication.processEvents()
    bar = dlg.findChild(HintBar)
    view = bar.findChild(QTextEdit, "HintBarText")
    bar.setText("First line.\nSecond line.\nThird line.\nFourth line.")
    QApplication.processEvents()
    margins = bar.contentsMargins()
    assert bar.height() >= (
        bar.fontMetrics().lineSpacing() * 4 + margins.top()
        + margins.bottom() + bar._resize_handle.height())
    assert view.fontInfo().pixelSize() == bar.fontInfo().pixelSize()
    assert view.verticalScrollBar().maximum() == 0


def test_the_help_is_justified(qtbot):
    """Ragged right is most visible in a narrow popup of prose."""
    bar = HintBar()
    qtbot.addWidget(bar)
    assert bar.alignment() & Qt.AlignJustify

    from spacr.qt.widgets.hover_tooltip import HoverTooltip

    tip = HoverTooltip()
    qtbot.addWidget(tip)
    assert tip._label.alignment() & Qt.AlignJustify


def test_nothing_in_preferences_pops_a_floating_tooltip(qtbot):
    """The strip is the answer, not a second one that the window covers."""
    from PySide6.QtWidgets import QWidget

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    bar = dlg.findChildren(HintBar)[0]

    still_popping = [
        w for w in dlg.findChildren(QWidget)
        if (w.toolTip() or "").strip() and not isinstance(w, HintBar)]
    assert still_popping == [], (
        f"{len(still_popping)} controls answer twice, e.g. "
        f"{[type(w).__name__ for w in still_popping[:5]]}")

    # And the help was MOVED, not merely deleted.
    assert len(getattr(bar, "_hints", {})) > 100
