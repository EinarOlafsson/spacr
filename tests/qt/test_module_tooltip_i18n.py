"""Coverage for reviewed module-summary localization."""

import hashlib

from spacr.qt.i18n_module_summaries import (
    MODULE_SUMMARIES,
    REVIEWED_SOURCE_HASHES,
    module_summary,
)


NON_ENGLISH = {"sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"}


def test_every_supported_non_english_language_covers_all_modules():
    assert set(MODULE_SUMMARIES) == NON_ENGLISH
    key_sets = {frozenset(catalog) for catalog in MODULE_SUMMARIES.values()}
    assert len(key_sets) == 1
    reviewed_keys = next(iter(key_sets))
    assert len(reviewed_keys) == 33
    assert set(REVIEWED_SOURCE_HASHES) == set(reviewed_keys)
    assert all(text.strip() for catalog in MODULE_SUMMARIES.values()
               for text in catalog.values())


def test_reviewed_summary_hashes_match_current_builtin_sources():
    from spacr.qt.app import APPS

    sources = {key: summary for key, _name, summary, _section in APPS}
    assert all(
        REVIEWED_SOURCE_HASHES[key]
        == hashlib.sha256(sources[key].encode("utf-8")).hexdigest()
        for key in REVIEWED_SOURCE_HASHES
    )


def test_module_summary_uses_reviewed_translation_and_safe_fallback():
    english = "Generate UMAP embeddings with image glyphs"
    assert module_summary("umap", english, "de") != english
    assert "UMAP" in module_summary("umap", english, "zh_CN")
    assert module_summary("future_plugin", "Plugin summary", "fr") == "Plugin summary"
    assert module_summary("umap", english, "en") == english


def test_stale_reviewed_summary_cannot_bypass_external_source_hash(
    monkeypatch,
):
    import spacr.qt.i18n_catalogs as external
    import spacr.qt.i18n_module_summaries as reviewed

    english = "Generate UMAP embeddings with image glyphs"
    monkeypatch.setitem(reviewed.REVIEWED_SOURCE_HASHES, "umap", "stale")
    monkeypatch.setattr(
        external,
        "module_summary",
        lambda key, source, language: "current hashed translation",
    )
    assert module_summary("umap", english, "de") == "current hashed translation"


def test_stale_make_masks_training_summary_was_removed():
    english = (
        "Correct a mask by hand: brush, flood fill, relabel, fill, "
        "remove small"
    )
    assert all("make_masks" not in catalog for catalog in MODULE_SUMMARIES.values())
    assert "Cellpose" not in module_summary("make_masks", english, "de")


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
