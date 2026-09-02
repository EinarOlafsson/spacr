"""The Mask live preview's dropdowns in a language that is not English.

Five entries in the panel -- ``auto``, ``Overlay``, ``Masks``, ``Flows`` and
``All channels`` -- were marked untranslatable because the handlers read them
back with ``currentText()``: translating the caption moved the value with it,
so every lookup missed. Two more dropdowns were NOT marked and did exactly
that, silently. On a Swedish screen the segmentation object box handed the
worker ``cellen``, and the intensity threshold wrote ``medelvärde`` into
``cell_intensity_threshold_method``, a setting that only accepts ``mean``.

Every entry now keeps its English value in the item's DATA, so the caption is
free to follow the language. These tests read the caption and the value apart
and drive a real language change over a built panel, because a caption that is
translated but never re-rendered looks exactly like one that was never
translated at all.
"""
from __future__ import annotations

import numpy as np
import pytest

from PySide6.QtWidgets import QComboBox

from spacr.qt.i18n import retranslate_widget_tree, set_translatable_items, tr
from spacr.qt.widgets import live_preview as LP
from spacr.qt.widgets.preview_controls import populate_channel_combo


LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


@pytest.fixture
def panel(qtbot, qapp):
    """A Mask live preview with a three-channel image set enumerated."""
    widget = LP.LivePreviewPanel()
    qtbot.addWidget(widget)
    populate_channel_combo(widget._channel_box, 3)
    widget._localise_channel_combo()
    return widget


def _captions(combo: QComboBox) -> list[str]:
    return [combo.itemText(i) for i in range(combo.count())]


def _values(combo: QComboBox) -> list[str]:
    return [combo.itemData(i) for i in range(combo.count())]


# ---------------------------------------------------------------------------
# The captions follow the language; the values never move
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", LANGUAGES)
def test_every_value_carrying_entry_is_translated_and_keeps_its_value(
        panel, language):
    """Caption in the chosen language, value in the one the code is written in."""
    retranslate_widget_tree(panel, language)

    for combo, sources in (
            (panel._object_box, LP.OBJECT_TYPES),
            (panel._outline_colour, LP.OUTLINE_CHOICES),
            (panel._view_mode, LP.VIEW_MODES),
    ):
        assert _values(combo) == list(sources)
        assert _captions(combo) == [tr(s, language) for s in sources]
        # Every one of these has a reviewed row, so nothing is left English
        # unless the language genuinely spells it the same way.
        still_english = [s for s, shown in zip(sources, _captions(combo))
                         if shown == s and tr(s, language) != s]
        assert not still_english

    assert _values(panel._channel_box) == [
        "All channels", "Ch 0", "Ch 1", "Ch 2"]
    assert panel._channel_box.itemText(0) == tr("All channels", language)
    # A plane's name is an identifier, not prose, in every language.
    assert _captions(panel._channel_box)[1:] == ["Ch 0", "Ch 1", "Ch 2"]


def test_the_captions_come_back_to_english(panel):
    """A language change re-renders in both directions, not just away from English."""
    retranslate_widget_tree(panel, "sv")
    assert _captions(panel._view_mode) != list(LP.VIEW_MODES)
    retranslate_widget_tree(panel, "en")
    assert _captions(panel._view_mode) == list(LP.VIEW_MODES)
    assert _captions(panel._object_box) == list(LP.OBJECT_TYPES)


def test_a_choice_survives_the_language_change(panel):
    """What the user picked is still picked afterwards, by value not by caption."""
    panel._object_box.setCurrentIndex(LP.OBJECT_TYPES.index("cell + nucleus"))
    panel._outline_colour.setCurrentIndex(
        LP.OUTLINE_CHOICES.index("magenta"))
    panel._view_mode.setCurrentIndex(LP.VIEW_MODES.index("Flows"))
    panel._channel_box.setCurrentIndex(2)

    retranslate_widget_tree(panel, "sv")

    assert panel._selected_object_types() == ("cell", "nucleus")
    assert panel._outline_choice() == "magenta"
    assert panel._outline_rgb() == LP.LivePreviewPanel.OUTLINE_COLOURS["magenta"]
    assert panel._view_mode_choice() == "Flows"
    assert panel.display_channel() == 1


# ---------------------------------------------------------------------------
# The two dropdowns that were quietly writing translated values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", LANGUAGES)
def test_the_worker_is_asked_for_a_compartment_it_knows(panel, language):
    """``cell``, never ``cellen``: the entry is the key every setting is spelled with."""
    retranslate_widget_tree(panel, language)
    panel._object_box.setCurrentIndex(LP.OBJECT_TYPES.index("cell + nucleus"))

    assert panel._selected_object_types() == ("cell", "nucleus")
    assert panel.current_params()["object_types"] == ("cell", "nucleus")

    settings = panel.settings_for_propagation()
    for compartment in ("cell", "nucleus"):
        assert f"remove_background_{compartment}" in settings
        assert f"{compartment}_signal_to_noise" in settings
    assert not [key for key in settings if key.startswith("cellen")]


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_threshold_method_propagates_as_a_value_spacr_accepts(
        panel, language):
    """``mean``/``percentile`` reach the settings dict whatever the caption reads."""
    retranslate_widget_tree(panel, language)
    methods = {key: value
               for key, value in panel._compartment_settings().items()
               if key.endswith("intensity_threshold_method")}

    assert methods
    assert set(methods.values()) == {"mean"}

    for widget in panel._compartment_widgets["cell"].values():
        if isinstance(widget, QComboBox):
            widget.setCurrentIndex(
                LP.INTENSITY_THRESHOLD_METHODS.index("percentile"))
    assert panel._compartment_settings()[
        "cell_intensity_threshold_method"] == "percentile"


@pytest.mark.parametrize("language", LANGUAGES)
def test_the_displayed_channel_is_read_from_the_entry_not_the_caption(
        panel, language):
    """A translated ``All channels`` still means "show it as stored"."""
    retranslate_widget_tree(panel, language)

    panel._channel_box.setCurrentIndex(0)
    assert panel.display_channel() is None
    for index in (1, 2, 3):
        panel._channel_box.setCurrentIndex(index)
        assert panel.display_channel() == index - 1


def test_a_reload_keeps_the_chosen_channel_across_the_language(panel):
    """Re-enumerating a folder re-selects by entry; the caption no longer matches."""
    retranslate_widget_tree(panel, "sv")
    panel._channel_box.setCurrentIndex(2)
    assert panel.display_channel() == 1

    canonical = panel._channel_box.currentData()
    populate_channel_combo(panel._channel_box, 4, keep=canonical)
    panel._localise_channel_combo()

    assert panel.display_channel() == 1
    assert panel._channel_box.itemText(0) == tr("All channels", "sv")


# ---------------------------------------------------------------------------
# The drawn pixels, which is where the old bug would have shown
# ---------------------------------------------------------------------------

def test_the_outline_colour_still_reaches_the_canvas_in_swedish(panel):
    """The colour a user picks on a Swedish screen is the colour that is drawn.

    Reading the entry back by text is what made these dropdowns
    untranslatable: a translated caption matched nothing in the colour table
    and every choice silently fell back to the compartment default.
    """
    image = np.zeros((20, 20), np.uint8)
    mask = np.zeros((20, 20), np.int32)
    mask[5:15, 5:15] = 1
    panel._image = image
    panel._masks = {"cell": mask}

    retranslate_widget_tree(panel, "sv")
    panel._outline_colour.setCurrentIndex(LP.OUTLINE_CHOICES.index("red"))
    assert panel._outline_colour.currentText() == "röd"

    panel._refresh_canvases()
    painted = LP.overlay_masks(image, {"cell": mask},
                               outline_rgb=panel._outline_rgb())
    assert tuple(painted[5, 5]) == LP.LivePreviewPanel.OUTLINE_COLOURS["red"]


def test_the_masks_view_is_chosen_by_value_in_swedish(panel):
    """``_refresh_canvases`` branches on the entry, so ``Masker`` selects Masks."""
    image = np.zeros((20, 20), np.uint8)
    mask = np.zeros((20, 20), np.int32)
    mask[5:15, 5:15] = 1
    panel._image = image
    panel._masks = {"cell": mask}
    retranslate_widget_tree(panel, "sv")
    panel._view_mode.setCurrentIndex(LP.VIEW_MODES.index("Masks"))

    assert panel._view_mode.currentText() == "Masker"
    assert panel._view_mode_choice() == "Masks"
    panel._refresh_canvases()
    assert panel._mask_view._pixmap_item is not None


# ---------------------------------------------------------------------------
# The helper itself
# ---------------------------------------------------------------------------

def test_set_translatable_items_refuses_a_mismatched_value_list(qapp):
    """One value per caption, or the dropdown would silently mis-map entries."""
    combo = QComboBox()
    with pytest.raises(ValueError):
        set_translatable_items(combo, ("a", "b"), values=("a",))


def test_set_translatable_items_overrides_a_class_level_skip(qapp):
    """A FlatComboBox marks every entry untranslatable; these entries are not."""
    combo = QComboBox()
    combo.setProperty("i18nSkipItems", True)
    set_translatable_items(combo, LP.VIEW_MODES, language="sv")

    assert combo.property("i18nSkipItems") is False
    assert combo.itemText(0) == tr("Overlay", "sv")
    assert combo.itemData(0) == "Overlay"


def test_a_dropdown_filled_any_other_way_still_reads_back_as_its_caption(qapp):
    """Model names and file names carry no data, and must keep working."""
    combo = QComboBox()
    combo.addItems(["cpsam", "cyto3"])
    assert LP._combo_value(combo) == "cpsam"


# ---------------------------------------------------------------------------
# Qt's own menu text, which no spaCR catalog can reach
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("language", ("hi", "is"))
def test_qt_menus_degrade_to_english_where_qt_ships_no_catalog(
        qapp, language):
    """``&Copy`` and ``Select All`` are Qt's words, from ``qtbase_<lang>.qm``.

    Qt ships seven of spaCR's nine languages. The two it does not must come
    out English rather than as a missing-file error, and the last language
    that DID load must not be left underneath answering for them.
    """
    from spacr.qt.i18n import install_qt_translations

    try:
        assert install_qt_translations(qapp, "sv") is True
        assert qapp.translate("QLineEdit", "&Copy") != "&Copy"

        assert install_qt_translations(qapp, language) is False
        assert qapp.translate("QLineEdit", "&Copy") == "&Copy"
        assert qapp.translate("QLineEdit", "Select All") == "Select All"
    finally:
        install_qt_translations(qapp, "en")


def test_qt_menus_follow_a_language_chosen_after_launch(qapp, panel):
    """The catalog is loaded by the language pass, not only once at startup."""
    from spacr.qt.i18n import install_qt_translations

    try:
        install_qt_translations(qapp, "en")
        assert qapp.translate("QLineEdit", "&Copy") == "&Copy"

        retranslate_widget_tree(panel, "de")
        assert qapp.translate("QLineEdit", "&Copy") == "&Kopieren"
    finally:
        install_qt_translations(qapp, "en")
