"""The saved-figure appearance preference reaches the production resolver."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QComboBox, QDialogButtonBox


def test_the_preferences_control_drives_saved_figure_appearance(
        monkeypatch, qtbot, qt_theme_applied):
    """Drive the real combo, Save it, and ask the renderer-facing call site.

    The environment assertion is part of the same path: Preferences is the
    durable answer, while an explicit command-line/notebook override remains
    the highest-priority answer for that process.
    """
    from spacr import figure_style
    from spacr.qt import preferences

    monkeypatch.delenv("SPACR_FIGURE_SAVE_MODE", raising=False)
    assert preferences.get_figure_save_mode() == "print"

    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    combo = dialog.findChild(QComboBox, "FigureSaveMode")

    assert combo is not None, "Preferences has no figure save-mode control"
    assert [combo.itemData(index) for index in range(combo.count())] == list(
        figure_style.SAVE_MODES)

    combo.setCurrentIndex(combo.findData("transparent"))
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.Save).click()

    assert preferences.get_figure_save_mode() == "transparent"
    assert figure_style.figure_save_mode() == "transparent"
    assert figure_style.saved_figure_appearance().mode == "transparent"

    reopened = preferences.PreferencesDialog()
    qtbot.addWidget(reopened)
    assert reopened.findChild(
        QComboBox, "FigureSaveMode").currentData() == "transparent"

    monkeypatch.setenv("SPACR_FIGURE_SAVE_MODE", "screen")
    assert figure_style.figure_save_mode() == "screen"
    assert figure_style.saved_figure_appearance().mode == "screen"
