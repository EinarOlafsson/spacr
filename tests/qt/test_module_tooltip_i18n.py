"""Coverage for reviewed module-summary localization."""

import hashlib
from importlib import import_module

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
    # 30: it was 32 when Classify CV and Classify ML became one screen (their
    # two reviewed rows collapsed into `classify_merged`), and the merge that
    # made the Cellpose Workbench cost two more. Folding a module into a host
    # does NOT cost it a row here -- the sentence a reviewer read is still the
    # sentence its button carries. What did is the merge: `cellpose_masks`
    # became a TAB, which has a label and no summary, and `train_cellpose`'s
    # sentence was rewritten to cover both halves. Both sets of reviewed
    # translations were bound to English the app no longer says, and
    # `module_summary` had already stopped using them.
    assert len(reviewed_keys) == 30
    assert set(REVIEWED_SOURCE_HASHES) == set(reviewed_keys)
    assert all(text.strip() for catalog in MODULE_SUMMARIES.values()
               for text in catalog.values())


def _english_summaries():
    """The one-line summary of every module, tile or no tile.

    A reviewed translation is bound to the exact English sentence it was
    reviewed against. Folding a module into a host deletes its registry
    row but not its sentence — the sentence becomes the fold button's
    tooltip, kept in a ``FOLD_FALLBACK`` — so the binding is still
    checkable and still worth checking. Reading APPS alone would raise
    ``KeyError`` on the folded keys and reading it with a ``.get`` would
    silently stop checking them.

    ``folded_modules`` applies the same host-local fallback precedence as the
    buttons themselves.  Reading selected historical fallback tables here
    would bind reviewed prose to text that the current fold strip no longer
    renders.
    """
    from spacr.qt.app import APPS
    from spacr.qt.widgets.fold_strip import folded_modules

    sources = {key: summary for key, _name, summary, _section in APPS}
    sources.update({key: entry[1] for key, entry in folded_modules().items()})
    return sources


def test_reviewed_summary_hashes_match_current_builtin_sources():
    sources = _english_summaries()

    missing = sorted(set(REVIEWED_SOURCE_HASHES) - set(sources))
    assert not missing, (
        f"reviewed translations for modules with no English sentence "
        f"anywhere — neither a registry row nor a fold fallback: {missing}")
    assert all(
        REVIEWED_SOURCE_HASHES[key]
        == hashlib.sha256(sources[key].encode("utf-8")).hexdigest()
        for key in REVIEWED_SOURCE_HASHES
    )


def test_external_catalogs_use_the_reviewed_module_summaries():
    """Generated catalogs may not override reviewed scientific summaries."""
    sources = _english_summaries()
    for language, reviewed in MODULE_SUMMARIES.items():
        catalog = import_module(f"spacr.qt.i18n_catalogs.{language}")
        for key, target in reviewed.items():
            assert key in sources
            assert catalog.MODULE_SUMMARIES[key] == target


def test_module_summary_uses_reviewed_translation_and_safe_fallback():
    english = "Visualize UMAP embeddings with image glyphs"
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

    # NO TOOLTIP TO CHECK ANY MORE, and that is the point of the assertion
    # that replaces it. The module popup came off on 2026-09-03 ("remove the
    # popup window tooltip on the moduals"), so what has to retranslate is
    # the ACCESSIBLE text -- which is what a screen reader reads, and the
    # only place the sentence lives on the row itself now.
    #
    # `_refresh_module_help` is still the function under test: it is what
    # clears the tooltip AND writes the translated name and summary, so a
    # regression that put the popup back, or that stopped translating,
    # fails here either way.
    retranslate_widget_tree(sidebar, "de")
    assert mask.accessibleName() == "Masken"
    assert mask.accessibleDescription() == module_summary("mask", english, "de")
    assert mask.toolTip() == "", "the module popup is back"

    retranslate_widget_tree(sidebar, "en")
    assert mask.accessibleName() == "Mask"
    assert mask.accessibleDescription() == english
    assert mask.toolTip() == "", "the module popup is back"
