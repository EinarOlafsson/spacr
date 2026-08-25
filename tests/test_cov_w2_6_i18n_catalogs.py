"""A translation is served only while the English it was made from stands.

The catalogs are real generated modules, so these read them: a label whose
canonical source has since been edited must fall back to current English
rather than show the stale translation that was correct last release.
"""
from __future__ import annotations

import pytest

from spacr.qt import i18n_catalogs as cat


@pytest.fixture(autouse=True)
def _forget_cached_modules():
    """`_module` is cached for the life of the process; a test that makes an
    import fail must not leave that verdict behind for the next one."""
    cat._module.cache_clear()
    yield
    cat._module.cache_clear()


def _one_real_setting_label():
    """A (key, English source) pair that at least one catalog translates."""
    english = getattr(cat._english(), "SETTING_LABELS", {})
    for language in cat.CATALOG_LANGUAGES:
        module = cat._module(language)
        for key, source in english.items():
            if cat._localized_value(module, "SETTING_LABELS", key, source):
                return language, key, source
    pytest.skip("no catalog translates any setting label")


# --------------------------------------------------------------------------
# which languages have a catalog at all
# --------------------------------------------------------------------------

def test_english_has_no_catalog_because_english_is_the_source():
    assert cat._module("en") is None
    assert cat._module("") is None
    assert cat._module(None) is None


def test_a_language_nobody_wrote_a_catalog_for_has_none():
    assert cat._module("xx_YY") is None


def test_every_declared_language_actually_has_a_module():
    for language in cat.CATALOG_LANGUAGES:
        assert cat._module(language) is not None, language


def test_a_catalog_that_is_not_installed_is_a_quiet_english_fallback(
        monkeypatch):
    """An optional catalog file that did not ship must cost that language its
    translations, not the app its start-up."""
    name = f"{cat.__name__}.sv"

    def _absent(module_name):
        raise ModuleNotFoundError(f"No module named {module_name!r}",
                                  name=module_name)

    monkeypatch.setattr(cat, "import_module", _absent)
    assert cat._module("sv") is None


def test_a_broken_import_inside_a_catalog_is_raised_not_swallowed(monkeypatch):
    """A catalog whose own imports fail is a real defect; hiding it behind the
    English fallback would make it invisible for a release."""
    def _catalog_is_broken(module_name):
        raise ModuleNotFoundError("No module named 'not_installed'",
                                  name="not_installed")

    monkeypatch.setattr(cat, "import_module", _catalog_is_broken)
    with pytest.raises(ModuleNotFoundError, match="not_installed"):
        cat._module("de")


# --------------------------------------------------------------------------
# every lookup falls back to English when there is no catalog
# --------------------------------------------------------------------------

def test_english_gets_no_translation_from_any_of_the_lookups():
    assert cat.ui_text("Run", "en") is None
    assert cat.setting_label("cell_diameter", "Cell diameter", "en") is None
    assert cat.setting_tooltip("cell_diameter", "anything", "en") is None
    assert cat.category_help("anything", "en") is None
    assert cat.module_summary("mask", "anything", "en") is None


def test_an_unknown_language_gets_no_translation_from_any_lookup():
    english = getattr(cat._english(), "SETTING_LABELS", {})
    key, source = next(iter(english.items()))
    assert cat.setting_label(key, source, "xx") is None
    tooltips = getattr(cat._english(), "SETTING_TOOLTIPS", {})
    if tooltips:
        t_key, t_source = next(iter(tooltips.items()))
        assert cat.setting_tooltip(t_key, t_source, "xx") is None
    categories = getattr(cat._english(), "CATEGORY_SOURCES", frozenset())
    if categories:
        assert cat.category_help(next(iter(categories)), "xx") is None
    summaries = getattr(cat._english(), "MODULE_SUMMARIES", {})
    if summaries:
        m_key, m_source = next(iter(summaries.items()))
        assert cat.module_summary(m_key, m_source, "xx") is None


# --------------------------------------------------------------------------
# a translation survives only while its English source does
# --------------------------------------------------------------------------

def test_a_label_whose_english_was_edited_falls_back_to_english():
    language, key, _source = _one_real_setting_label()
    assert cat.setting_label(key, "A label nobody wrote", language) is None


def test_a_label_whose_english_still_matches_is_translated():
    language, key, source = _one_real_setting_label()
    said = cat.setting_label(key, source, language)
    assert isinstance(said, str) and said.strip()


def test_an_app_scoped_key_falls_back_to_the_bare_key():
    """A setting shared by several apps is stored once under its bare name;
    asking for it under an app prefix must still find it."""
    language, key, source = _one_real_setting_label()
    if "." in key:
        pytest.skip("the chosen key is already app-scoped")
    assert cat.setting_label(key, source, language,
                             app_key="no_such_app") == \
        cat.setting_label(key, source, language)


def test_a_key_that_is_not_in_the_english_source_is_never_translated():
    for language in cat.CATALOG_LANGUAGES:
        assert cat.setting_label("no_such_setting", "x", language) is None
        assert cat.setting_tooltip("no_such_setting", "x", language) is None
        assert cat.category_help("prose nobody wrote", language) is None
        assert cat.module_summary("no_such_module", "x", language) is None
        break


def test_a_stale_stored_hash_is_refused_even_with_the_text_present():
    """The hash is what makes an edited English string retire its old
    translations, so a matching key alone must not be enough."""
    module = cat._module(cat.CATALOG_LANGUAGES[0])
    assert cat._localized_value(module, "UI", "Run", "some other source") \
        is None


def test_a_blank_translation_is_treated_as_no_translation():
    class Blank:
        SOURCE_HASHES = {}
        UI = {"Run": "   "}

    import hashlib
    digest = hashlib.sha256("Run".encode("utf-8")).hexdigest()
    Blank.SOURCE_HASHES = {("UI", "Run"): digest}
    assert cat._localized_value(Blank, "UI", "Run", "Run") is None


def _first_live(language, table, canonical):
    """The first entry of ``table`` in ``language`` whose stored hash is
    still current, as ``(key, english_source)``."""
    module = cat._module(language)
    for key, source in canonical:
        if cat._localized_value(module, table, key, source):
            return key, source
    pytest.skip(f"no live {table} entry in {language}")


def test_static_qt_text_is_translated_when_the_catalog_still_matches():
    """`ui_text` keys on the English string itself, so the hash check is the
    only thing standing between an edited button label and a stale one."""
    module = cat._module("sv")
    pairs = [(k, k) for k in getattr(module, "UI", {})]
    source, _ = _first_live("sv", "UI", pairs)
    said = cat.ui_text(source, "sv")
    assert isinstance(said, str) and said.strip()
    assert cat.ui_text(source + " (edited)", "sv") is None


def test_a_scientific_tooltip_is_translated_while_its_prose_matches():
    canonical = getattr(cat._english(), "SETTING_TOOLTIPS", {})
    key, source = _first_live("sv", "SETTING_TOOLTIPS", canonical.items())
    said = cat.setting_tooltip(key, source, "sv")
    assert isinstance(said, str) and said.strip()
    assert cat.setting_tooltip(key, source + " Extra sentence.", "sv") is None


def test_a_category_blurb_is_translated_only_for_prose_english_declares():
    sources = sorted(getattr(cat._english(), "CATEGORY_SOURCES", frozenset()))
    text, _ = _first_live("sv", "CATEGORY_HELP", [(s, s) for s in sources])
    said = cat.category_help(text, "sv")
    assert isinstance(said, str) and said.strip()
    assert cat.category_help(text + " Rewritten.", "sv") is None


def test_a_module_summary_is_translated_while_its_description_matches():
    canonical = getattr(cat._english(), "MODULE_SUMMARIES", {})
    key, source = _first_live("sv", "MODULE_SUMMARIES", canonical.items())
    said = cat.module_summary(key, source, "sv")
    assert isinstance(said, str) and said.strip()
    assert cat.module_summary(key, "A different description", "sv") is None
