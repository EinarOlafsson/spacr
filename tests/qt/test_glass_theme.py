"""The Glass preference and its translucent module material."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_qsettings(qt_theme_applied, tmp_path):
    from PySide6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.IniFormat)
    QSettings.setPath(
        QSettings.IniFormat, QSettings.UserScope, str(tmp_path))
    QSettings("spacr", "qt").clear()
    yield
    QSettings("spacr", "qt").clear()


def _rule(stylesheet, selector):
    start = stylesheet.index(selector)
    end = stylesheet.index("}", start)
    return stylesheet[start:end]


def test_glass_is_a_persisted_theme():
    from spacr.qt import preferences
    from spacr.qt import theme

    preferences.set_theme("glass")
    assert preferences.get_theme() == "glass"
    assert preferences.resolve_effective_theme() == "glass"
    assert "glass" in theme.THEMES
    assert "glass" in theme.IMAGE_THEMES


def test_preferences_dialog_offers_glass(qtbot):
    from PySide6.QtWidgets import QComboBox
    from spacr.qt.preferences import PreferencesDialog

    dialog = PreferencesDialog()
    qtbot.addWidget(dialog)
    combos = dialog.findChildren(QComboBox)
    values = {
        combo.itemData(index)
        for combo in combos
        for index in range(combo.count())
    }
    assert "glass" in values


def test_glass_scrims_are_translucent_and_accessible():
    from spacr.qt import theme

    assert theme.scrim_under("glass") == theme.GLASS_BACKDROP_UNDER
    assert theme.scrim_failures("glass") == []
    assert theme.contrast_failures("glass") == []
    for role in theme.SCRIM_ROLES:
        assert 0.0 < theme.scrim_alpha("glass", role) < 0.88


def test_glass_styles_every_module_box_with_rgba_material():
    from spacr.qt import theme

    qss = theme.stylesheet("glass")
    assert "qlineargradient" in qss
    for selector in (
        "QFrame#Card {",
        "QFrame#ConsoleBox {",
        "QFrame#SectionCard {",
        "QLineEdit, QSpinBox",
    ):
        rule = _rule(qss, selector)
        assert "rgba(" in rule, f"{selector} remained opaque"


def test_glass_popups_remain_opaque():
    from spacr.qt import theme

    assert theme.scrim_alpha("glass", "elevated") == 1.0
    popup = _rule(theme.stylesheet("glass"), "QMenu {")
    assert "rgba(" not in popup
