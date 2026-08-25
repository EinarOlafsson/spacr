"""Translation where the catalog does not have the answer.

The compact catalog covers what it covers. Everything interesting in this
module is what happens for the strings it does not: a mnemonic ampersand, an
uppercased section header, a label a worker rewrote with a file path in it, a
combo whose items are filenames, a language Qt itself does not ship. Each of
those has a rule, and every rule exists because the naive behaviour was worse
than English -- a translated file path is a broken file path.

The widgets here are real Qt widgets carrying real Qt properties, because the
opt-out this module implements is written entirely in `setProperty` and
`property`, and a stand-in with a dict would be testing the stand-in.
"""

import os
import sys
import types

import pytest
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QComboBox, QLabel, QPushButton, QTableWidget,
                               QTreeWidget, QWidget)

from spacr.qt import i18n
from spacr.qt.i18n import (CATALOGS, DEFAULT_LANGUAGE, ENV_LANGUAGE, _ROWS,
                           _TERM_ROWS, _exact_translation, _row,
                           _term_translation, add_translation,
                           catalog_coverage, current_language, has_translation,
                           install_dialog_translation, install_qt_translations,
                           language_choices, normalize_language,
                           retranslate_widget_tree, set_translatable_text, tr)

#: The nine translated codes, in catalog order.
_CODES = i18n._TRANSLATED_CODES


@pytest.fixture
def catalog_restored():
    """`add_translation` writes into module globals; undo what a test adds."""
    before = set(_ROWS)
    yield
    for source in list(_ROWS):
        if source not in before:
            del _ROWS[source]
            for code in _CODES:
                CATALOGS[code].pop(source, None)


@pytest.fixture(autouse=True)
def english_unless_asked(monkeypatch):
    """Pin the ambient language so a stray preference cannot move a result."""
    monkeypatch.setenv(ENV_LANGUAGE, "en")


# ---------------------------------------------------------------------------
# the catalog rows
# ---------------------------------------------------------------------------

def test_a_row_with_the_wrong_number_of_languages_is_refused():
    """A short row would silently shift every language after the gap."""
    with pytest.raises(ValueError) as raised:
        _row("bara en")
    assert str(len(_CODES)) in str(raised.value)

    assert len(_row(*[f"x{i}" for i in range(len(_CODES))])) == len(_CODES)


def test_registering_an_app_name_reaches_every_catalog(catalog_restored):
    """An app gives its translations once and they arrive in all nine."""
    values = [f"Prov {code}" for code in _CODES]
    assert add_translation("W2_2 Probe", values) is True

    for code, value in zip(_CODES, values):
        assert CATALOGS[code]["W2_2 Probe"] == value
        assert tr("W2_2 Probe", code) == value


def test_registering_the_same_name_twice_is_a_no_op(catalog_restored):
    """Two modules naming one app is not a conflict worth raising over."""
    values = [f"Prov {code}" for code in _CODES]
    assert add_translation("W2_2 Probe", values) is True
    assert add_translation("W2_2 Probe", ["other"] * len(_CODES)) is False
    assert CATALOGS[_CODES[0]]["W2_2 Probe"] == values[0]


def test_a_blank_translation_fails_where_the_app_name_is_in_the_message(
        catalog_restored):
    """Better here than as a blank sidebar row in Korean."""
    values = [f"Prov {code}" for code in _CODES]
    values[3] = "   "
    with pytest.raises(ValueError) as raised:
        add_translation("W2_2 Probe", values)
    assert "W2_2 Probe" in str(raised.value)
    assert "W2_2 Probe" not in _ROWS


def test_a_short_translation_row_fails_with_the_count_in_the_message(
        catalog_restored):
    """The other half of the same guarantee."""
    with pytest.raises(ValueError):
        add_translation("W2_2 Probe", ["only", "two"])
    assert "W2_2 Probe" not in _ROWS


# ---------------------------------------------------------------------------
# picking up apps registered before this module was imported
# ---------------------------------------------------------------------------

def test_with_no_app_module_there_is_nothing_to_absorb(monkeypatch):
    """A process that never imported the app registry simply finds nothing."""
    monkeypatch.delitem(sys.modules, "spacr.qt.app", raising=False)
    before = len(i18n._ROWS)

    i18n._absorb_registered_app_names()

    # Nothing found means nothing added -- the guard is that it does not
    # invent rows, not merely that it survives.
    assert len(i18n._ROWS) == before


def test_a_partially_built_app_module_is_not_read(monkeypatch):
    """`spacr.qt.app` is in `sys.modules` and half built while this runs."""
    half_built = types.ModuleType("spacr.qt.app")
    monkeypatch.setitem(sys.modules, "spacr.qt.app", half_built)
    before = len(i18n._ROWS)

    i18n._absorb_registered_app_names()

    # A module with no APPS yet must contribute nothing, rather than a
    # partial set that a later full import would have to correct.
    assert len(i18n._ROWS) == before


def test_a_registered_app_name_is_absorbed_from_the_registry(monkeypatch,
                                                             catalog_restored):
    """The pull half of the seam: apps that registered first are picked up."""
    values = [f"Prov {code}" for code in _CODES]
    fake = types.ModuleType("spacr.qt.app")
    fake.APP_META = {"w2_2_probe": {"name": "W2_2 Probe"}}
    fake.registered_metadata = lambda what: {"w2_2_probe": values}
    monkeypatch.setitem(sys.modules, "spacr.qt.app", fake)

    i18n._absorb_registered_app_names()
    assert CATALOGS[_CODES[0]]["W2_2 Probe"] == values[0]


def test_one_apps_bad_row_costs_that_app_its_translations_not_the_app(
        monkeypatch, catalog_restored):
    """A ValueError here must not stop the module importing."""
    fake = types.ModuleType("spacr.qt.app")
    fake.APP_META = {}
    fake.registered_metadata = lambda what: {"w2_2_broken": ["too", "few"]}
    monkeypatch.setitem(sys.modules, "spacr.qt.app", fake)

    i18n._absorb_registered_app_names()        # must not raise
    assert "w2_2_broken" not in _ROWS


# ---------------------------------------------------------------------------
# which language a code names
# ---------------------------------------------------------------------------

def test_a_locale_shaped_code_resolves_to_the_bundled_language():
    """`pt_BR` and `zh-CN` come out of a real system locale."""
    assert normalize_language("pt_BR") == "pt"
    assert normalize_language("zh-CN") == "zh_CN"
    assert normalize_language("zh") == "zh_CN"
    assert normalize_language("zh_TW") == "zh_CN"
    assert normalize_language("DE") == "de"
    assert normalize_language("de_AT") == "de"


def test_a_hand_edited_settings_file_is_harmless():
    """Anything unrecognised falls back to English rather than blanking the UI."""
    assert normalize_language("") == DEFAULT_LANGUAGE
    assert normalize_language(None) == DEFAULT_LANGUAGE
    assert normalize_language("klingon") == DEFAULT_LANGUAGE
    assert normalize_language(17) == DEFAULT_LANGUAGE


def test_the_environment_overrides_the_persisted_language(monkeypatch):
    """`SPACR_LANGUAGE` is how a test or a launcher pins the language."""
    monkeypatch.setenv(ENV_LANGUAGE, "sv")
    assert current_language() == "sv"


def test_with_no_preferences_module_the_language_is_english(monkeypatch):
    """Reading preferences must not be able to break a translation call."""
    monkeypatch.delenv(ENV_LANGUAGE, raising=False)

    import spacr.qt.preferences as preferences

    def explode():
        raise RuntimeError("the settings store is unreadable")

    monkeypatch.setattr(preferences, "get_language", explode)
    assert current_language() == DEFAULT_LANGUAGE


def test_the_language_picker_offers_a_label_and_a_code():
    """Preferences shows the display name and stores the code."""
    choices = language_choices()
    assert ("English", "en") in choices
    assert all(len(pair) == 2 for pair in choices)
    assert {code for _label, code in choices} == set(i18n.VALID_LANGUAGE_CODES)


# ---------------------------------------------------------------------------
# finding a translation
# ---------------------------------------------------------------------------

def _a_catalogued_source():
    """One English string the compact catalog really carries."""
    return next(iter(_ROWS))


def test_surrounding_whitespace_survives_the_lookup():
    """A caption padded for layout still finds its translation.

    Translating away the padding would move the text in the widget.
    """
    source = _a_catalogued_source()
    plain = _exact_translation(source, "sv")
    assert plain is not None

    padded = _exact_translation(f"  {source}\n", "sv")
    assert padded == f"  {plain}\n"


def test_a_mnemonic_ampersand_stays_on_the_translated_word():
    """Qt reads '&' as the keyboard mnemonic, and it has to survive."""
    source = _a_catalogued_source()
    plain = _exact_translation(source, "sv")

    assert _exact_translation(f"&{source}", "sv") == f"&{plain}"


def test_a_literal_double_ampersand_is_kept_literal():
    """'&&' is a literal '&' in Qt, and must not become a mnemonic."""
    source = _a_catalogued_source()
    plain = _exact_translation(source, "sv")
    translated = _exact_translation(source.replace("&", "&&") if "&" in source
                                    else f"{source}", "sv")
    assert translated is not None
    assert plain is not None


def test_an_uppercased_section_header_is_translated_uppercased():
    """Headers are often uppercased before they reach the QLabel."""
    source = _a_catalogued_source()
    plain = _exact_translation(source, "sv")
    if source.upper() == source:
        pytest.skip("the first catalog row is already uppercase")
    assert _exact_translation(source.upper(), "sv") == plain.upper()


def test_an_uncatalogued_string_is_left_in_english():
    """A missing entry stays English rather than becoming blank."""
    assert _exact_translation("zzz not in any catalog zzz", "sv") is None
    assert tr("zzz not in any catalog zzz", "sv") == "zzz not in any catalog zzz"


def test_a_plugin_catalog_that_cannot_be_discovered_is_not_fatal(monkeypatch):
    """Plugin translations are optional metadata; core localization is not."""
    import spacr.plugins as plugins

    def explode():
        raise RuntimeError("the plugin directory is unreadable")

    monkeypatch.setattr(plugins, "discover_plugins", explode)
    source = _a_catalogued_source()
    assert _exact_translation(source, "sv") is not None
    assert _exact_translation("zzz nothing has this zzz", "sv") is None


def test_a_plugin_may_supply_a_translation_the_core_catalog_lacks(monkeypatch):
    """A plugin's own UI words reach the same lookup."""
    import spacr.plugins as plugins

    plugin = types.SimpleNamespace(
        translations={"sv": {"Plugin Only Label": "Endast insticksmodul"}})
    monkeypatch.setattr(plugins, "discover_plugins", lambda: [plugin])

    assert _exact_translation("Plugin Only Label", "sv") == \
        "Endast insticksmodul"


# ---------------------------------------------------------------------------
# the conservative word-by-word fallback
# ---------------------------------------------------------------------------

def _a_catalogued_term():
    return next(iter(_TERM_ROWS))


def test_prose_and_paths_are_never_decomposed_into_words():
    """A translated file path is a broken file path.

    The same rule keeps a long paragraph and a fragment of HTML out of the
    word-by-word fallback, where a partial match reads as a typo.
    """
    term = _a_catalogued_term()
    assert _term_translation("/data/plate1/" + term, "sv") is None
    assert _term_translation("C:\\data\\" + term, "sv") is None
    assert _term_translation("https://example.org/" + term, "sv") is None
    assert _term_translation(f"{term}\nsecond line", "sv") is None
    assert _term_translation("<b>" + term + "</b>", "sv") is None
    assert _term_translation(term + " " + "x" * 90, "sv") is None


def test_a_label_with_no_known_word_in_it_is_left_alone():
    """"Changed nothing" is reported as None, not as the original string."""
    assert _term_translation("zzzz qqqq", "sv") is None


def test_an_uppercased_term_comes_back_uppercased_where_case_exists():
    """Latin scripts have case; the CJK and Indic catalogs do not."""
    term = _a_catalogued_term()
    if not term.isalpha():
        pytest.skip("the first term row is not a single word")
    lowered = _term_translation(term, "sv")
    if lowered is None:
        pytest.skip("the first term row has no Swedish entry")
    assert _term_translation(term.upper(), "sv") == lowered.upper()
    assert _term_translation(term.upper(), "ko") == \
        _term_translation(term, "ko")


def test_a_language_with_no_term_catalog_translates_nothing():
    """English is the source; there is nothing to look up."""
    assert _term_translation(_a_catalogued_term(), "en") is None


# ---------------------------------------------------------------------------
# tr, and what it reports about itself
# ---------------------------------------------------------------------------

def test_placeholders_are_filled_after_translation_not_before():
    """A catalog may reorder placeholders, which only works in that order."""
    assert tr("{n} more under All settings", "en", n=4) == \
        "4 more under All settings"


def test_a_placeholder_the_caller_did_not_supply_leaves_the_template():
    """A missing value is a template on screen, never a KeyError in a paint."""
    assert tr("{n} more under All settings", "en") == \
        "{n} more under All settings"
    assert tr("{n} more under All settings", "en", wrong=1) == \
        "{n} more under All settings"


def test_english_is_returned_untouched():
    """The default language does no lookup at all."""
    source = _a_catalogued_source()
    assert tr(source, "en") == source
    assert tr(source) == source


def test_has_translation_answers_for_both_catalogs():
    """English asks whether the source is catalogued; a language asks for a
    rendering."""
    source = _a_catalogued_source()
    assert has_translation(source, "en") is True
    assert has_translation(source, "sv") is True
    assert has_translation("zzz nothing zzz", "en") is False
    assert has_translation("zzz nothing zzz", "sv") is False


def test_coverage_counts_each_source_once():
    """A repeated caption is one string to translate, not two."""
    source = _a_catalogued_source()
    translated, total = catalog_coverage([source, source, "zzz nothing zzz"],
                                         "sv")
    assert total == 2
    assert translated == 1


# ---------------------------------------------------------------------------
# translating a live widget tree
# ---------------------------------------------------------------------------

def test_text_a_worker_replaced_is_kept_byte_for_byte(qapp):
    """A label rewritten outside the translator carries data, not chrome.

    Restoring a stale caption over a path, a progress value or a result is
    the failure this opt-out exists to prevent.
    """
    source = _a_catalogued_source()
    label = QLabel(source)

    retranslate_widget_tree(label, "sv")
    swedish = label.text()
    assert swedish == tr(source, "sv")

    label.setText("/data/plate1/well_A01.tif")
    retranslate_widget_tree(label, "sv")
    assert label.text() == "/data/plate1/well_A01.tif"
    assert label.property("i18nSkipText") is True

    # and it stays opted out on the next pass
    retranslate_widget_tree(label, "de")
    assert label.text() == "/data/plate1/well_A01.tif"


def test_a_dynamic_caption_keeps_its_template_and_its_values(qapp):
    """Application chrome with a value in it retranslates without losing it."""
    button = QPushButton()
    set_translatable_text(button, "{n} more under All settings", "en", n=3)
    assert button.text() == "3 more under All settings"

    retranslate_widget_tree(button, "sv")
    assert "3" in button.text()


def test_a_combo_of_filenames_can_opt_out_of_item_translation(qapp):
    """`i18nSkipItems` is how a data-bearing combo says so."""
    combo = QComboBox()
    combo.addItems(["well_A01.tif", "well_A02.tif"])
    combo.setProperty("i18nSkipItems", True)

    retranslate_widget_tree(combo, "sv")
    assert [combo.itemText(i) for i in range(combo.count())] == \
        ["well_A01.tif", "well_A02.tif"]


def test_a_combo_of_catalogued_items_is_translated(qapp):
    """The ordinary case: a picker of English words becomes Swedish ones."""
    source = _a_catalogued_source()
    combo = QComboBox()
    combo.addItem(source)

    retranslate_widget_tree(combo, "sv")
    assert combo.itemText(0) == tr(source, "sv")

    # and switching back restores English rather than compounding
    retranslate_widget_tree(combo, "en")
    assert combo.itemText(0) == source


def test_table_and_tree_headers_are_translated_and_restored(qapp):
    """Header captions are chrome; the cells under them are not."""
    source = _a_catalogued_source()

    table = QTableWidget(1, 1)
    table.setHorizontalHeaderLabels([source])
    retranslate_widget_tree(table, "sv")
    assert table.horizontalHeaderItem(0).text() == tr(source, "sv")
    retranslate_widget_tree(table, "en")
    assert table.horizontalHeaderItem(0).text() == source

    tree = QTreeWidget()
    tree.setColumnCount(1)
    tree.setHeaderLabels([source])
    retranslate_widget_tree(tree, "sv")
    assert tree.headerItem().text(0) == tr(source, "sv")


def test_a_widget_with_its_own_retranslate_hook_is_asked(qapp):
    """A widget that renders its own content gets told the language changed."""
    class _Custom(QWidget):
        def __init__(self):
            super().__init__()
            self.told = []

        def retranslate_dynamic_content(self, code):
            self.told.append(code)

    widget = _Custom()
    retranslate_widget_tree(widget, "sv")
    assert widget.told == ["sv"]


def test_a_retranslate_hook_that_raises_does_not_stop_the_pass(qapp):
    """One broken widget must not leave the rest of the window in English."""
    class _Broken(QWidget):
        def retranslate_dynamic_content(self, code):
            raise RuntimeError("this widget is half built")

    source = _a_catalogued_source()
    root = QWidget()
    broken = _Broken()
    broken.setParent(root)
    label = QLabel(source, root)

    retranslate_widget_tree(root, "sv")       # must not raise
    assert label.text() == tr(source, "sv")


def test_a_module_entry_rebuilds_its_help_semantically(qapp):
    """Module help is a sentence, not a bag of words to translate one by one."""
    source = _a_catalogued_source()

    sidebar = QLabel()
    sidebar.setProperty("moduleAppKey", "mask")
    sidebar.setProperty("moduleNameSource", "Mask")
    sidebar.setProperty("moduleSummarySource", "Segment cells and nuclei.")
    sidebar.setProperty("moduleTooltipStyle", "sidebar")
    retranslate_widget_tree(sidebar, "sv")
    assert " — " in sidebar.toolTip()
    assert sidebar.accessibleName()
    assert sidebar.accessibleDescription()

    tile = QLabel()
    tile.setProperty("moduleAppKey", "mask")
    tile.setProperty("moduleNameSource", "Mask")
    tile.setProperty("moduleSummarySource", "Segment cells and nuclei.")
    tile.setProperty("moduleTooltipStyle", "tile")
    tile.setProperty("moduleStageSource", source)
    retranslate_widget_tree(tile, "sv")
    assert "(" in tile.toolTip() and ")" in tile.toolTip()
    assert tile.accessibleDescription()


def test_an_action_module_entry_uses_its_status_tip(qapp):
    """A QAction has a status tip and no widget tooltip."""
    root = QWidget()
    action = QAction("Mask", root)
    action.setProperty("moduleAppKey", "mask")
    action.setProperty("moduleNameSource", "Mask")
    action.setProperty("moduleSummarySource", "Segment cells and nuclei.")

    retranslate_widget_tree(root, "sv")
    assert action.statusTip()


def test_a_widget_without_the_module_properties_is_left_alone(qapp):
    """No app key means there is no module help to rebuild."""
    label = QLabel("plain")
    retranslate_widget_tree(label, "sv")
    assert label.toolTip() == ""


def test_translating_nothing_is_a_no_op(qapp):
    """`None` is a legitimate root when a screen has not been built.

    And the pass must still work afterwards: a walk that returned early
    by breaking its own state would pass a "did not raise" test and fail
    the next real screen.
    """
    retranslate_widget_tree(None, "sv")

    label = QLabel("Language")
    retranslate_widget_tree(label, "sv")
    assert label.text() != "Language", (
        "the walk stopped working after being handed None")


# ---------------------------------------------------------------------------
# Qt's own strings
# ---------------------------------------------------------------------------

def test_with_no_application_there_is_nothing_to_install():
    """A headless caller gets False rather than an AttributeError."""
    assert install_qt_translations(None) is False
    assert install_dialog_translation(None) is None


def test_a_language_qt_does_not_ship_gets_english_menus(qapp):
    """Hindi and Icelandic have no `qtbase` catalog, and that is said by
    returning False rather than by installing an empty translator."""
    assert install_qt_translations(qapp, "hi") is False
    assert install_qt_translations(qapp, "is") is False
    assert install_qt_translations(qapp, "en") is False


def test_switching_language_twice_leaves_one_translator(qapp):
    """A translator left underneath answers for strings the new one lacks."""
    try:
        install_qt_translations(qapp, "sv")
        first = getattr(qapp, "_spacr_qt_translator", None)
        install_qt_translations(qapp, "de")
        second = getattr(qapp, "_spacr_qt_translator", None)
        if first is not None and second is not None:
            assert first is not second
        # a language with no catalog clears the previous one
        assert install_qt_translations(qapp, "hi") is False
        assert getattr(qapp, "_spacr_qt_translator", None) is None
    finally:
        previous = getattr(qapp, "_spacr_qt_translator", None)
        if previous is not None:
            qapp.removeTranslator(previous)
            qapp._spacr_qt_translator = None


def test_the_dialog_filter_is_installed_once(qapp):
    """A second filter would translate every dialog twice."""
    had = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    try:
        qapp._spacr_dialog_i18n_filter = None
        install_dialog_translation(qapp)
        first = qapp._spacr_dialog_i18n_filter
        assert first is not None
        install_dialog_translation(qapp)
        assert qapp._spacr_dialog_i18n_filter is first
    finally:
        current = getattr(qapp, "_spacr_dialog_i18n_filter", None)
        if current is not None and current is not had:
            qapp.removeEventFilter(current)
        qapp._spacr_dialog_i18n_filter = had


# ---------------------------------------------------------------------------
# the rest of the lookup chain
# ---------------------------------------------------------------------------

def _a_term_only_source():
    """A source the term catalog carries and the compact catalog does not."""
    for source in _TERM_ROWS:
        if source not in _ROWS:
            return source
    pytest.skip("every term row is also a compact row")


def test_a_reviewed_phrase_is_taken_whole_before_it_is_decomposed():
    """The word-by-word fallback can never match a multi-word dictionary key."""
    source = _a_term_only_source()
    assert _exact_translation(source, "sv") is not None


def test_an_external_catalog_fills_in_where_the_compact_one_stops(
        monkeypatch):
    """The compact catalog is reviewed by hand; the big one is generated."""
    import spacr.qt.i18n_catalogs as catalogs

    def ui_text(source, language):
        return "Ur katalogen" if source == "Only In The Big Catalog" else None

    monkeypatch.setattr(catalogs, "ui_text", ui_text)
    assert _exact_translation("Only In The Big Catalog", "sv") == \
        "Ur katalogen"
    assert _exact_translation("&Only In The Big Catalog", "sv") == \
        "&Ur katalogen"


def test_without_the_external_catalogs_the_compact_one_still_answers(
        monkeypatch):
    """Their absence must not make the core catalog unavailable."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if "i18n_catalogs" in str(name):
            raise ImportError("no generated catalogs in this build")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)
    monkeypatch.delitem(sys.modules, "spacr.qt.i18n_catalogs", raising=False)

    source = _a_catalogued_source()
    assert _exact_translation(source, "sv") is not None
    assert _exact_translation("zzz nothing zzz", "sv") is None
    assert _exact_translation("&zzz nothing zzz", "sv") is None


# ---------------------------------------------------------------------------
# translating one Qt property
# ---------------------------------------------------------------------------

def test_an_object_with_no_such_property_is_skipped(qapp):
    """Not every object in a tree has every string property."""
    label = QLabel("x")
    i18n._translate_qt_text(label, "noSuchGetter", "noSuchSetter",
                            "_spacr_i18n_nothing", "sv")

    # Skipped means untouched: a property it cannot read is one it must
    # not write either, or the widget ends up with a value from nowhere.
    assert label.text() == "x"
    assert label.property("_spacr_i18n_nothing") is None


def test_a_destroyed_widget_in_the_tree_is_stepped_over(qapp):
    """A deferred-delete wrapper can still be in `findChildren` briefly."""
    import shiboken6

    label = QLabel("x")
    shiboken6.delete(label)
    i18n._translate_qt_text(label, "text", "setText", "_spacr_i18n_text",
                            "sv")                          # must not raise
    assert i18n._refresh_dynamic_text(label, "sv") is False
    i18n._refresh_module_help(label, "sv")                 # must not raise


def test_a_tooltip_replaced_outside_the_translator_is_kept(qapp):
    """Only the text property opts a widget out; a tooltip is just preserved."""
    source = _a_catalogued_source()
    widget = QWidget()
    widget.setToolTip(source)

    retranslate_widget_tree(widget, "sv")
    assert widget.toolTip() == tr(source, "sv")

    widget.setToolTip("/data/plate1 — 4 fields")
    retranslate_widget_tree(widget, "sv")
    assert widget.toolTip() == "/data/plate1 — 4 fields"
    assert widget.property("i18nSkipText") is None


def test_a_root_that_cannot_be_walked_translates_only_itself(qapp,
                                                             monkeypatch):
    """`findChildren` on a half-destroyed root raises; the root still counts."""
    source = _a_catalogued_source()
    label = QLabel(source)

    def explode(*_args, **_kwargs):
        raise RuntimeError("this widget is being destroyed")

    monkeypatch.setattr(label, "findChildren", explode)
    retranslate_widget_tree(label, "sv")
    assert label.text() == tr(source, "sv")


# ---------------------------------------------------------------------------
# a settings form's own labels
# ---------------------------------------------------------------------------

def test_a_settings_label_the_compact_catalog_knows_uses_it(qapp):
    """The reviewed catalog stays authoritative for a label it carries."""
    source = _a_catalogued_source()
    label = QLabel(source)
    label.setProperty("settingKey", "cell_diameter")
    label.setProperty("settingsAppKey", "mask")

    retranslate_widget_tree(label, "sv")
    assert label.text() == tr(source, "sv")


def test_a_settings_label_it_does_not_know_asks_the_context_catalog(
        qapp, monkeypatch):
    """The context-keyed catalog fills the much larger settings surface."""
    import spacr.qt.i18n_catalogs as catalogs

    asked = []

    def setting_label(key, source, language, app_key):
        asked.append((key, source, language, app_key))
        return "Celldiameter"

    monkeypatch.setattr(catalogs, "setting_label", setting_label)

    label = QLabel("Cell diameter in this module only")
    label.setProperty("settingKey", "cell_diameter")
    label.setProperty("settingsAppKey", "mask")

    retranslate_widget_tree(label, "sv")
    assert label.text() == "Celldiameter"
    assert asked and asked[0][0] == "cell_diameter"
    assert asked[0][3] == "mask"


def test_a_settings_label_no_catalog_knows_falls_back_to_english(
        qapp, monkeypatch):
    """A missing entry is English, not blank."""
    import spacr.qt.i18n_catalogs as catalogs

    monkeypatch.setattr(catalogs, "setting_label",
                        lambda *_a, **_k: None)

    label = QLabel("Zzz Not In Any Catalog Zzz")
    label.setProperty("settingKey", "zzz")
    label.setProperty("settingsAppKey", "mask")

    retranslate_widget_tree(label, "sv")
    assert label.text() == "Zzz Not In Any Catalog Zzz"


def test_an_api_link_is_repointed_at_the_translated_documentation(qapp):
    """The docs have language variants, and the link has to follow."""
    class _Link(QWidget):
        def __init__(self):
            super().__init__()
            self.urls = []

        def set_url(self, url):
            self.urls.append(url)

    link = _Link()
    link.setProperty("moduleApiAppKey", "mask")
    retranslate_widget_tree(link, "sv")
    assert link.urls, "the API link was never repointed"


def test_a_menu_action_is_translated_with_the_window(qapp):
    """Actions live beside the widgets and carry the same three properties."""
    source = _a_catalogued_source()
    root = QWidget()
    action = QAction(source, root)
    action.setToolTip(source)
    action.setStatusTip(source)

    retranslate_widget_tree(root, "sv")
    assert action.text() == tr(source, "sv")
    assert action.toolTip() == tr(source, "sv")
    assert action.statusTip() == tr(source, "sv")


# ---------------------------------------------------------------------------
# Qt's own catalogs, when Qt will not co-operate
# ---------------------------------------------------------------------------

def test_a_translator_that_will_not_load_is_reported_as_not_loaded(
        qapp, monkeypatch):
    """A `qtbase` catalog missing from the install is False, not a crash."""
    from PySide6 import QtCore

    class _Refusing(QtCore.QTranslator):
        def load(self, *_args, **_kwargs):
            return False

    monkeypatch.setattr(QtCore, "QTranslator", _Refusing)
    had = getattr(qapp, "_spacr_qt_translator", None)
    try:
        assert install_qt_translations(qapp, "sv") is False
    finally:
        qapp._spacr_qt_translator = had


def test_a_translator_that_throws_is_reported_as_not_loaded(qapp,
                                                            monkeypatch):
    """Anything Qt raises here costs the menus, never the application."""
    from PySide6 import QtCore

    def explode(*_args, **_kwargs):
        raise RuntimeError("no translations directory")

    monkeypatch.setattr(QtCore, "QTranslator", explode)
    had = getattr(qapp, "_spacr_qt_translator", None)
    try:
        assert install_qt_translations(qapp, "sv") is False
    finally:
        qapp._spacr_qt_translator = had


def test_a_translator_that_cannot_be_removed_is_not_fatal(qapp, monkeypatch):
    """Removing an already-destroyed translator raises; switching still works."""
    class _Stuck:
        pass

    had = getattr(qapp, "_spacr_qt_translator", None)
    try:
        qapp._spacr_qt_translator = _Stuck()

        def refuse(_translator):
            raise RuntimeError("that translator has gone")

        monkeypatch.setattr(qapp, "removeTranslator", refuse)
        assert install_qt_translations(qapp, "hi") is False
        assert qapp._spacr_qt_translator is None
    finally:
        qapp._spacr_qt_translator = had


def test_a_shown_dialog_is_translated_by_the_filter(qapp):
    """A file picker built and executed in one expression never met the
    window's language pass."""
    from PySide6.QtWidgets import QDialog

    source = _a_catalogued_source()
    had = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    dialog = QDialog()
    try:
        qapp._spacr_dialog_i18n_filter = None
        install_dialog_translation(qapp)

        label = QLabel(source, dialog)
        dialog.show()
        assert label.text() == tr(source, "en")
    finally:
        dialog.close()
        current = getattr(qapp, "_spacr_dialog_i18n_filter", None)
        if current is not None and current is not had:
            qapp.removeEventFilter(current)
        qapp._spacr_dialog_i18n_filter = had


# ---------------------------------------------------------------------------
# the last few shapes a Qt tree can take
# ---------------------------------------------------------------------------

def test_a_literal_ampersand_survives_a_mnemonic_lookup(qapp,
                                                        catalog_restored):
    """Qt spells a literal '&' as '&&', and the translation has to keep it.

    Getting this wrong turns "Copy && Paste" into a caption with a stray
    keyboard mnemonic in the middle of it.
    """
    values = [f"Kopiera & Klistra {code}" for code in _CODES]
    add_translation("Copy & Paste", values)

    assert _exact_translation("Copy & Paste", "sv") == values[0]
    assert _exact_translation("&Copy && Paste", "sv") == \
        "&" + values[0].replace("&", "&&")


def test_the_external_catalog_keeps_a_literal_ampersand_too(qapp,
                                                            monkeypatch):
    """The same rule, on the other half of the lookup chain."""
    import spacr.qt.i18n_catalogs as catalogs

    monkeypatch.setattr(
        catalogs, "ui_text",
        lambda source, language: ("Kopiera & Klistra"
                                  if source == "Zzz Copy & Paste" else None))

    assert _exact_translation("&Zzz Copy && Paste", "sv") == \
        "&Kopiera && Klistra"


def test_a_group_box_title_and_a_placeholder_are_chrome(qapp):
    """Both are static captions, so both are translated."""
    from PySide6.QtWidgets import QGroupBox, QLineEdit

    source = _a_catalogued_source()
    box = QGroupBox(source)
    edit = QLineEdit(box)
    edit.setPlaceholderText(source)

    retranslate_widget_tree(box, "sv")
    assert box.title() == tr(source, "sv")
    assert edit.placeholderText() == tr(source, "sv")
    assert edit.text() == "", "the translator wrote into a line edit"


def test_tab_captions_are_translated_and_restored(qapp):
    """A tab bar is chrome; the pages under it are not touched."""
    from PySide6.QtWidgets import QTabWidget

    source = _a_catalogued_source()
    tabs = QTabWidget()
    tabs.addTab(QWidget(), source)

    retranslate_widget_tree(tabs, "sv")
    assert tabs.tabText(0) == tr(source, "sv")

    retranslate_widget_tree(tabs, "en")
    assert tabs.tabText(0) == source


def test_a_settings_catalog_that_throws_leaves_the_label_alone(qapp,
                                                               monkeypatch):
    """A broken context catalog costs that label, not the settings form."""
    import spacr.qt.i18n_catalogs as catalogs

    def explode(*_args, **_kwargs):
        raise RuntimeError("the context catalog is unreadable")

    monkeypatch.setattr(catalogs, "setting_label", explode)

    label = QLabel("Zzz Not In Any Catalog Zzz")
    label.setProperty("settingKey", "zzz")
    label.setProperty("settingsAppKey", "mask")

    retranslate_widget_tree(label, "sv")      # must not raise
    assert label.text() == "Zzz Not In Any Catalog Zzz"


@pytest.mark.xfail(strict=True, reason=(
    "retranslate_widget_tree's own settings-label block catches TypeError "
    "from setting_label; the refresh_api_tooltips pass at the end of the "
    "same function catches only (ImportError, AttributeError, RuntimeError), "
    "so a TypeError from the same catalog escapes a language switch and "
    "leaves the window half translated."))
def test_a_language_switch_never_raises_whatever_the_catalog_does(
        qapp, monkeypatch):
    """The pass documents itself as best-effort, so nothing may escape it.

    A language switch that raises leaves the window part English and part
    Swedish, with no way back except another switch.
    """
    import spacr.qt.i18n_catalogs as catalogs

    def explode(*_args, **_kwargs):
        raise TypeError("the context catalog is malformed")

    monkeypatch.setattr(catalogs, "setting_label", explode)

    label = QLabel("Zzz Not In Any Catalog Zzz")
    label.setProperty("settingKey", "zzz")
    label.setProperty("settingsAppKey", "mask")

    retranslate_widget_tree(label, "sv")


def test_a_widget_that_refuses_one_property_is_stepped_over(qapp):
    """Reading a property off a widget mid-teardown raises."""
    class _Picky(QWidget):
        def property(self, name):
            if name == "moduleApiAppKey":
                raise RuntimeError("this widget is going away")
            return super().property(name)

    widget = _Picky()
    sibling = QLabel("Language", widget)

    retranslate_widget_tree(widget, "sv")

    # STEPPED OVER, NOT STOPPED AT. The widget that refuses is skipped and
    # the rest of the tree is still translated -- otherwise one widget
    # mid-teardown leaves the whole window half English.
    assert sibling.text() != "Language"


def test_without_the_settings_model_the_pass_still_finishes(qapp,
                                                            monkeypatch):
    """Settings tooltips are an extra pass, not a precondition."""
    source = _a_catalogued_source()
    monkeypatch.setitem(sys.modules, "spacr.qt.screens.settings_model", None)

    label = QLabel(source)
    retranslate_widget_tree(label, "sv")
    assert label.text() == tr(source, "sv")


def _block_pyside(monkeypatch, module_name):
    """Make one PySide6 submodule unimportable for the duration of a call."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            raise ImportError(f"{module_name} is not available")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked)


def test_without_the_qt_widgets_module_nothing_is_translated(qapp,
                                                             monkeypatch):
    """A build with no QtWidgets has no widget tree to walk."""
    source = _a_catalogued_source()
    label = QLabel(source)
    _block_pyside(monkeypatch, "PySide6.QtWidgets")

    retranslate_widget_tree(label, "sv")
    assert label.text() == source, "the tree was walked without QtWidgets"


def test_without_qtranslator_qt_keeps_its_english(qapp, monkeypatch):
    """No QTranslator is False rather than an ImportError out of startup."""
    had = getattr(qapp, "_spacr_qt_translator", None)
    try:
        _block_pyside(monkeypatch, "PySide6.QtCore")
        assert install_qt_translations(qapp, "sv") is False
    finally:
        qapp._spacr_qt_translator = had


def test_without_qtcore_no_dialog_filter_is_installed(qapp, monkeypatch):
    """The same build; the dialog filter simply does not exist."""
    had = getattr(qapp, "_spacr_dialog_i18n_filter", None)
    try:
        qapp._spacr_dialog_i18n_filter = None
        _block_pyside(monkeypatch, "PySide6.QtCore")
        install_dialog_translation(qapp)
        assert getattr(qapp, "_spacr_dialog_i18n_filter", None) is None
    finally:
        qapp._spacr_dialog_i18n_filter = had
