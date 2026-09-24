"""Translated inversion help names real controls and survives widget translation."""
import json
from pathlib import Path

import pytest
from PySide6.QtGui import QTextDocument
from PySide6.QtWidgets import QPushButton

from spacr.qt import i18n
from spacr.qt.screens.make_masks import MakeMasksScreen


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
def test_inversion_help_and_control_names_agree(qtbot, monkeypatch, language):
    source = json.loads((Path(__file__).resolve().parents[2] /
        "tests/data/release_contracts/435_inversion_help_2026-09-22.json").read_text())["sources"]
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    try:
        i18n.retranslate_widget_tree(screen, language)
        warning = i18n.tr(source["warning"]["text"], language)
        tooltip = i18n.tr(source["tooltip"]["text"], language)
        assert warning != source["warning"]["text"]
        assert tooltip != source["tooltip"]["text"]
        assert screen._invert_warning.text() == warning
        rendered_help = screen._invert_display.toolTip()
        document = QTextDocument()
        document.setHtml(rendered_help)
        assert tooltip in document.toPlainText()
        assert 'href=' in rendered_help
        assert screen._invert_display.text() == i18n.tr("Invert image", language)
        assert screen._btn_filter.text() == i18n.tr("Filter", language)
        swap = i18n.tr("Swap object and background", language)
        assert swap != "Swap object and background"
        assert swap in tooltip
        assert swap in {button.text() for button in screen.findChildren(QPushButton)}
        assert "0..1" in tooltip
    finally:
        screen._magnifier.close()
        screen.close_folded()
