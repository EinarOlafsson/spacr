"""Coverage for reviewed module-summary localization."""

from spacr.qt.i18n_module_summaries import MODULE_SUMMARIES, module_summary


NON_ENGLISH = {"sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"}


def test_every_supported_non_english_language_covers_all_modules():
    assert set(MODULE_SUMMARIES) == NON_ENGLISH
    key_sets = {frozenset(catalog) for catalog in MODULE_SUMMARIES.values()}
    assert len(key_sets) == 1
    assert len(next(iter(key_sets))) == 34
    assert all(text.strip() for catalog in MODULE_SUMMARIES.values()
               for text in catalog.values())


def test_module_summary_uses_reviewed_translation_and_safe_fallback():
    english = "Generate UMAP embeddings with image glyphs"
    assert module_summary("umap", english, "de") != english
    assert "UMAP" in module_summary("umap", english, "zh_CN")
    assert module_summary("future_plugin", "Plugin summary", "fr") == "Plugin summary"
    assert module_summary("umap", english, "en") == english


def test_sidebar_module_help_retranslates_semantically(
    qtbot, qt_theme_applied,
):
    from PySide6.QtWidgets import QPushButton
    from spacr.qt.app import APPS, Sidebar
    from spacr.qt.i18n import retranslate_widget_tree

    sidebar = Sidebar()
    qtbot.addWidget(sidebar)
    buttons = {
        button.property("navKey"): button
        for button in sidebar.findChildren(QPushButton)
    }
    mask = buttons["mask"]
    english = next(description for key, _name, description, _section in APPS
                   if key == "mask")

    retranslate_widget_tree(sidebar, "de")
    assert mask.accessibleName() == "Masken"
    assert module_summary("mask", english, "de") in mask.toolTip()
    assert mask.accessibleDescription() == module_summary("mask", english, "de")

    retranslate_widget_tree(sidebar, "en")
    assert mask.accessibleName() == "Mask"
    assert mask.toolTip() == f"Mask — {english}"
