"""Keyboard focus remains visible on empty and selected plate-map wells."""

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAccessible, QColor
from PySide6.QtWidgets import QApplication

from spacr.qt import i18n, preferences, theme
from spacr.qt.widgets.plate_map_picker import PlateMapPicker, well_side


@pytest.fixture
def plate(qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(tmp_path / 'prefs.ini'), QSettings.IniFormat))
    monkeypatch.setattr(preferences, '_SAFE_MODE', False)
    preferences.set_strong_focus(False)
    theme._forget_painting_theme()
    widget = PlateMapPicker(layout=6)
    qtbot.addWidget(widget)
    i18n.retranslate_widget_tree(widget, 'en')
    widget.resize(500, 300)
    widget.show()
    widget.activateWindow()
    QApplication.processEvents()
    yield widget
    theme._forget_painting_theme()


@pytest.mark.parametrize('scheme', ['dark', 'light', 'high_contrast'])
@pytest.mark.parametrize('selected', [False, True])
def test_focus_changes_pixels_and_tracks_strong_preference_on_open_well(plate, scheme, selected):
    preferences.set_theme_choice(scheme)
    plate.setStyleSheet(theme.stylesheet())
    well = plate._wells[(1, 1)]
    well.setChecked(selected)
    plate.plate_button.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    original_geometry = well.geometry()
    plain = well.grab().toImage()
    center = plain.pixelColor(plain.width() // 2, plain.height() // 2)
    well.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    ordinary = well.grab().toImage()
    assert ordinary != plain
    assert ordinary.pixelColor(ordinary.width() // 2, ordinary.height() // 2) == center
    preferences.set_strong_focus(True)
    plate.setStyleSheet(theme.stylesheet())
    QApplication.processEvents()
    stronger = well.grab().toImage()
    assert stronger != ordinary
    colour = QColor(theme.focus_ring_colour(scheme))
    assert [stronger.pixelColor(stronger.width() // 2, y)
            for y in range(theme.FOCUS_RING_PX)] == [colour] * theme.FOCUS_RING_PX
    assert stronger.pixelColor(stronger.width() // 2, stronger.height() // 2) == center
    assert well.geometry() == original_geometry
    assert well.width() == well.height() == well_side()
    assert well.isChecked() is selected
    preferences.set_strong_focus(False)
    plate.setStyleSheet(theme.stylesheet())
    QApplication.processEvents()
    assert well.hasFocus() and well.grab().toImage() == ordinary
    plate.plate_button.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    assert well.grab().toImage() == plain


def test_real_tab_and_space_keep_well_identity_and_selection(plate, qtbot):
    well = plate._wells[(1, 1)]
    plate.plate_button.setFocus(Qt.TabFocusReason)
    for _ in range(24):
        qtbot.keyClick(QApplication.focusWidget(), Qt.Key_Tab)
        if well.hasFocus():
            break
    assert well.hasFocus()
    interface = QAccessible.queryAccessibleInterface(well)
    assert interface.text(QAccessible.Name) == 'A01'
    qtbot.keyClick(well, Qt.Key_Space)
    assert plate.selection() == {(1, 1)} and interface.state().checked
    qtbot.keyClick(well, Qt.Key_Space)
    assert plate.selection() == set() and not interface.state().checked
    qtbot.keyClick(well, Qt.Key_Tab)
    assert not well.hasFocus()
    for language in ('ko', 'en'):
        i18n.retranslate_widget_tree(plate, language)
        assert interface.text(QAccessible.Name) == 'A01'


def test_well_paint_uses_cached_focus_policy(plate, monkeypatch):
    well = plate._wells[(1, 1)]
    well.setFocus(Qt.TabFocusReason)
    QApplication.processEvents()
    from spacr.qt.widgets import figure_grid_view
    def no_preference_read():
        raise AssertionError('Painting must not reread stored focus preferences')
    monkeypatch.setattr(figure_grid_view, '_strong_focus_style', no_preference_read)
    assert not well.grab().isNull()
