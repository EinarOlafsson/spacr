"""Item 522: a preview's own scale stops at the windows it opens.

A dialog parented to a preview panel is a window of its own, but Qt
cascaded the panel's scaled sheet into it and the scaler's walk rescaled
its sizes: at 150 % Plaque's Paper, Settings and Overlay dialogs, and
Mask's Live Settings, had 59 px buttons, not the 40 px of every other
dialog (30 px at 75 %). Item 521 re-parented only the install dialog.
"""
import pytest
from PySide6.QtWidgets import (QApplication, QDialog, QDialogButtonBox,
                               QPushButton, QVBoxLayout)


def _settle():
    for _ in range(20):
        QApplication.processEvents()


def _screen(qtbot, name):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(name)
    qtbot.addWidget(screen)
    screen.resize(1366, 768)
    screen.show()
    _settle()
    return screen


@pytest.fixture
def plaque(qtbot, qt_theme_applied):
    screen = _screen(qtbot, "analyze_plaques")
    yield screen
    screen._live_preview.preview_scaler.set_scale(1.0, persist=False)


@pytest.fixture
def mask(qtbot, qt_theme_applied):
    screen = _screen(qtbot, "mask")
    yield screen
    screen._live_preview.preview_scaler.set_scale(1.0, persist=False)


def _normal(qtbot, window) -> int:
    dialog = QDialog(window)
    qtbot.addWidget(dialog)
    QVBoxLayout(dialog).addWidget(box := QDialogButtonBox(QDialogButtonBox.Ok))
    dialog.show()
    _settle()
    height = box.button(QDialogButtonBox.Ok).height()
    dialog.close()
    return height


def _heights(dialog):
    return [b.height() for b in dialog.findChildren(QPushButton)
            if b.isVisible()]


def _panel_button(panel) -> int:
    return max(b.sizeHint().height() for b in panel.findChildren(QPushButton)
               if b.window() is panel.window())


def _plaque_dialogs(panel):
    from spacr.qt.widgets import plaque_preview as pp

    return {"paper": lambda: pp.PaperDialog(panel, ""),
            "settings": lambda: pp.PlaqueSettingsDialog(panel),
            "overlay": lambda: pp.PlaqueOverlayDialog(panel._overlay_style,
                                                      panel)}


@pytest.mark.parametrize("scale", [1.5, 0.75])
@pytest.mark.parametrize("which", ["paper", "settings", "overlay"])
def test_plaque_dialogs_keep_the_ordinary_button_height(
        qtbot, plaque, scale, which):
    panel = plaque._live_preview
    normal = _normal(qtbot, plaque)
    unscaled = _panel_button(panel)
    panel.preview_scaler.set_scale(scale, persist=False)
    _settle()
    assert abs(_panel_button(panel) - unscaled * scale) <= 2
    dialog = _plaque_dialogs(panel)[which]()
    qtbot.addWidget(dialog)
    dialog.show()
    _settle()
    heights = _heights(dialog)
    assert heights and all(abs(h - normal) <= 2 for h in heights), (
        heights, normal)
    dialog.close()


@pytest.mark.parametrize("scale", [1.5, 0.75])
def test_mask_live_settings_keep_the_ordinary_button_height(
        qtbot, mask, scale):
    panel = mask._live_preview
    normal = _normal(qtbot, mask)
    panel.preview_scaler.set_scale(scale, persist=False)
    _settle()
    panel.open_live_settings()
    dialog = panel._live_settings_dialog
    _settle()
    heights = _heights(dialog)
    assert heights and all(abs(h - normal) <= 2 for h in heights), (
        heights, normal)
    dialog.close()


def test_an_open_dialog_follows_a_later_scale_and_gets_its_sheet_back(
        qtbot, plaque):
    panel = plaque._live_preview
    normal = _normal(qtbot, plaque)
    dialog = _plaque_dialogs(panel)["overlay"]()
    qtbot.addWidget(dialog)
    dialog.show()
    _settle()
    own = dialog.styleSheet()
    for scale in (1.5, 0.75):
        panel.preview_scaler.set_scale(scale, persist=False)
        _settle()
        assert all(abs(h - normal) <= 2 for h in _heights(dialog))
    panel.preview_scaler.set_scale(1.0, persist=False)
    _settle()
    assert dialog.styleSheet() == own
    assert all(abs(h - normal) <= 2 for h in _heights(dialog))
