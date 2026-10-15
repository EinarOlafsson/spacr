"""Figure section fold controls expose their identity and keyboard focus."""

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAccessible, QColor, QPixmap
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout, QWidget

from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.figure_grid_view import FigureGridView


@pytest.fixture
def section(qtbot, monkeypatch, tmp_path, request):
    theme_name = getattr(request, "param", "dark")
    settings_path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(settings_path), QSettings.IniFormat))
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    preferences.set_theme_choice(theme_name)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    host = QWidget()
    layout = QVBoxLayout(host)
    before = QPushButton("Before")
    grid = FigureGridView()
    after = QPushButton("After")
    for child in (before, grid, after):
        layout.addWidget(child)
    image = QPixmap(120, 90)
    image.fill(QColor("#465767"))
    grid.set_figures([image], ["Plate A"], [("Run 2026-10-15", 0, 1)])
    qtbot.addWidget(host)
    host.resize(500, 380)
    host.show()
    host.activateWindow()
    qtbot.wait(50)
    before.setFocus(Qt.TabFocusReason)
    qtbot.waitUntil(before.hasFocus)
    yield host, before, grid, after, theme_name
    theme._forget_painting_theme()


def test_focusable_section_exposes_its_actual_run_name_after_language_pass(section):
    _, _, grid, _, _ = section
    i18n.retranslate_widget_tree(grid, "en")
    header = grid._headers[0]
    interface = QAccessible.queryAccessibleInterface(header)
    assert interface.state().focusable
    assert interface.text(QAccessible.Name) == header.text() == "Run 2026-10-15"


@pytest.mark.parametrize("section", ["dark", "light", "high_contrast"], indirect=True)
def test_focused_section_has_visible_outline_and_live_stronger_focus(section):
    host, before, grid, _, theme_name = section
    header = grid._headers[0]
    before.setFocus(Qt.TabFocusReason)
    unfocused = header.grab().toImage()
    header.setFocus(Qt.TabFocusReason)
    assert header.hasFocus()
    ordinary = header.grab().toImage()
    assert ordinary != unfocused
    preferences.set_strong_focus(True)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert grid._headers[0] is header
    assert header.hasFocus()
    stronger = header.grab().toImage()
    expected = QColor(theme.focus_ring_colour(theme_name))
    x = stronger.width() // 2
    assert [stronger.pixelColor(x, y) for y in range(theme.FOCUS_RING_PX)] == (
        [expected] * theme.FOCUS_RING_PX)
    assert stronger != ordinary
    preferences.set_strong_focus(False)
    host.setStyleSheet(theme._the_focus_ring(theme_name))
    QApplication.processEvents()
    assert grid._headers[0] is header
    assert header.hasFocus()
    assert header.grab().toImage() == ordinary
    before.setFocus(Qt.TabFocusReason)
    assert header.grab().toImage() == unfocused


def test_painting_the_focused_header_does_not_read_preferences(section, monkeypatch):
    _, _, grid, _, _ = section
    header = grid._headers[0]
    header.setFocus(Qt.TabFocusReason)

    def forbidden():
        raise AssertionError("The paint path read the preference store")

    monkeypatch.setattr(preferences, "get_strong_focus", forbidden)
    assert not header.grab().isNull()
