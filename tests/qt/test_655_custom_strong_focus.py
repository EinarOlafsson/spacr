"""The documented stronger-focus preference reaches custom figure controls."""

import pytest
from PySide6.QtCore import QEvent, QPoint, QSettings, Qt
from PySide6.QtGui import QColor, QCursor, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from spacr.qt import preferences, theme
from spacr.qt.widgets.cell_montage_view import _Thumb
from spacr.qt.widgets.figure_grid_view import _FigureCell
from spacr.qt.widgets.figure_queue import _ClearFiguresLabel


@pytest.fixture
def control(qtbot, monkeypatch, tmp_path, request):
    kind, theme_name = request.param
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_theme_choice(theme_name)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#465767"))
    if kind == "grid":
        widget = _FigureCell(4, pixmap, "Plate five")
        widget.setFixedSize(180, 140)
    elif kind == "crop":
        widget = _Thumb(pixmap, "well A1, label 9", size=96, highlight="#009944")
    else:
        widget = _ClearFiguresLabel()
        widget.setFixedSize(180, 48)
    after = QPushButton("After")
    for child in (before, widget, after):
        layout.addWidget(child)
    qtbot.addWidget(host)
    host.show()
    host.activateWindow()
    QCursor.setPos(QPoint(-500, -500))
    if kind == "crop":
        QApplication.sendEvent(widget, QEvent(QEvent.Leave))
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield host, before, widget, after, theme_name
    theme._forget_painting_theme()


CASES = [(kind, palette) for kind in ("grid", "crop", "clear")
         for palette in ("dark", "light", "high_contrast")]


@pytest.mark.parametrize("control", CASES, indirect=True)
def test_stronger_focus_adds_the_three_pixel_theme_outline_and_reverts(control):
    host, before, widget, _, theme_name = control
    name = widget.accessibleName()
    unfocused = widget.grab().toImage()
    widget.setFocus(Qt.TabFocusReason)
    ordinary = widget.grab().toImage()
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert widget.hasFocus()
    stronger = widget.grab().toImage()
    assert stronger != ordinary
    expected = QColor(theme.focus_ring_colour(theme_name))
    x = stronger.width() // 2
    assert [stronger.pixelColor(x, y) for y in range(theme.FOCUS_RING_PX)] == (
        [expected] * theme.FOCUS_RING_PX)
    assert stronger.copy(8, 8, stronger.width() - 16, stronger.height() - 16) == (
        ordinary.copy(8, 8, ordinary.width() - 16, ordinary.height() - 16))
    before.setFocus(Qt.TabFocusReason)
    assert widget.grab().toImage() == unfocused
    widget.setFocus(Qt.TabFocusReason)
    preferences.set_strong_focus(False)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert widget.grab().toImage() == ordinary
    assert widget.accessibleName() == name


@pytest.mark.parametrize("control", [(kind, "dark") for kind in ("grid", "crop", "clear")],
                         indirect=True)
def test_painting_does_not_reread_the_stronger_focus_preference(control, monkeypatch):
    host, _, widget, _, theme_name = control
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    widget.setFocus(Qt.TabFocusReason)

    def forbidden():
        raise AssertionError("The paint path read the preference store")

    monkeypatch.setattr(preferences, "get_strong_focus", forbidden)
    assert not widget.grab().isNull()
