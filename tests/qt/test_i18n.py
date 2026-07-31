"""Localization catalogs, fallback behavior and runtime Qt retranslation."""
from __future__ import annotations

import pytest


def test_ten_requested_languages_are_bundled():
    from spacr.qt.i18n import VALID_LANGUAGE_CODES

    assert VALID_LANGUAGE_CODES == (
        "en", "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
    )


def test_every_non_english_catalog_covers_every_core_phrase():
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES, _ROWS

    expected = set(_ROWS)
    for code in VALID_LANGUAGE_CODES[1:]:
        assert set(CATALOGS[code]) == expected
        assert all(str(value).strip() for value in CATALOGS[code].values())


def test_every_registered_module_and_section_has_an_exact_translation():
    from spacr.qt.app import APPS
    from spacr.qt.i18n import CATALOGS, VALID_LANGUAGE_CODES

    phrases = {name for _key, name, _desc, _section in APPS}
    phrases.update(section for _key, _name, _desc, section in APPS)
    for code in VALID_LANGUAGE_CODES[1:]:
        missing = sorted(phrases - set(CATALOGS[code]))
        assert not missing, f"{code} missing registry phrases: {missing}"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pt-BR", "pt"),
        ("zh", "zh_CN"),
        ("ZH-cn", "zh_CN"),
        ("de_DE", "de"),
        ("unknown", "en"),
        (None, "en"),
    ],
)
def test_locale_shaped_codes_normalize_safely(raw, expected):
    from spacr.qt.i18n import normalize_language

    assert normalize_language(raw) == expected


def test_translation_formats_after_translation():
    from spacr.qt.i18n import tr

    assert tr("Opened {name}", "sv", name="Masker") == "Öppnade Masker"
    assert tr("Go to  {name}", "zh_CN", name="测量") == "前往测量"


def test_unknown_scientific_text_falls_back_to_english():
    from spacr.qt.i18n import tr

    source = "Leiden resolution epsilon"
    assert tr(source, "fr") == source


def test_common_settings_terms_translate_conservatively():
    from spacr.qt.i18n import tr

    assert tr("Input & Metadata", "de") == "Eingabe & Metadaten"
    assert tr("Cell Segmentation", "sv") == "Cell Segmentering"
    assert tr("/data/Cell_Segmentation.tif", "fr") == \
        "/data/Cell_Segmentation.tif"


def test_widget_tree_switches_languages_and_preserves_user_values(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QVBoxLayout
    from PySide6.QtWidgets import QWidget
    from spacr.qt.i18n import retranslate_widget_tree

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    title = QLabel("Home")
    run = QPushButton("Run")
    user_value = QLineEdit("/data/my plate")
    layout.addWidget(title)
    layout.addWidget(run)
    layout.addWidget(user_value)

    retranslate_widget_tree(root, "sv")
    assert title.text() == "Hem"
    assert run.text() == "Kör"
    assert user_value.text() == "/data/my plate"

    retranslate_widget_tree(root, "ko")
    assert title.text() == "홈"
    assert run.text() == "실행"
    assert user_value.text() == "/data/my plate"

    retranslate_widget_tree(root, "en")
    assert title.text() == "Home"
    assert run.text() == "Run"


def test_table_headers_retranslate_and_profile_names_do_not(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import (
        QComboBox, QTableWidget, QVBoxLayout, QWidget,
    )
    from spacr.qt.i18n import retranslate_widget_tree

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    table = QTableWidget(0, 2, root)
    table.setHorizontalHeaderLabels(["Job", "Status"])
    profile = QComboBox(root)
    profile.setProperty("i18nSkipItems", True)
    profile.addItem("Home")
    layout.addWidget(table)
    layout.addWidget(profile)

    retranslate_widget_tree(root, "sv")
    assert table.horizontalHeaderItem(0).text() == "Jobb"
    assert profile.itemText(0) == "Home"
    retranslate_widget_tree(root, "ko")
    assert table.horizontalHeaderItem(0).text() == "작업"
    assert profile.itemText(0) == "Home"


def test_environment_language_override(monkeypatch):
    from spacr.qt.i18n import current_language

    monkeypatch.setenv("SPACR_LANGUAGE", "is-IS")
    assert current_language() == "is"


def test_main_window_and_lazy_screen_follow_runtime_language(
    qtbot, qt_theme_applied, monkeypatch,
):
    from spacr.qt.app import MainWindow

    monkeypatch.setenv("SPACR_LANGUAGE", "sv")
    window = MainWindow()
    qtbot.addWidget(window)

    mask_button = next(
        button for button in window._sidebar._items
        if button.property("navKey") == "mask"
    )
    assert "Masker" in mask_button.text()
    assert window._app_actions["measure"].text() == "Mätning"

    window._on_nav_selected("measure")
    screen = window._screens["measure"]
    section_titles = [
        section._header.text() for section in screen._settings_sections
    ]
    assert any("MÄTNING" in title for title in section_titles)

    monkeypatch.setenv("SPACR_LANGUAGE", "ko")
    window.refresh_language()
    assert "마스크" in mask_button.text()
    assert window._app_actions["measure"].text() == "측정"
