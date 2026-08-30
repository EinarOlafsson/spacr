"""What an aliased organelle slot does when the machinery under it gives way.

Slots five through twenty-six are not materialised in the catalogs; they
borrow slot one's reviewed translation and renumber it. This file pins the
three ways that borrowing can fail and still has to answer in English:

* :mod:`spacr.organelle_types` refuses the key (the ``except`` arm that
  clears ``role``/``number``/``primary``),
* the language has no catalog at all (English and unknown codes),
* the catalog exists but holds no record for the primary key.

Every case is paired in the same test with the input that DOES produce a
translation, so an assertion of ``None`` cannot pass by accident.
"""
from __future__ import annotations

import pytest

from spacr.qt import i18n_catalogs as cat

#: Slot 5 -- above ``CATALOGUED_ORGANELLE_SLOTS`` (4), so it is served by the
#: alias path rather than by a record of its own.
ALIAS_KEY = "organellee_diameter"
PRIMARY_KEY = "organelle_diameter"


@pytest.fixture(autouse=True)
def _forget_cached_modules():
    """``_module`` is cached for the life of the process."""
    cat._module.cache_clear()
    yield
    cat._module.cache_clear()


def _alias_label_source() -> str:
    """The English label slot 5 renders, derived the way the app derives it."""
    primary = cat._english().SETTING_LABELS[PRIMARY_KEY]
    return primary.replace("Organelle 1", "Organelle 5", 1)


def _alias_tooltip_source() -> str:
    primary = cat._english().SETTING_TOOLTIPS[PRIMARY_KEY]
    return primary.replace("organelle_", "organellee_").replace(
        "organelle ", "organelle 5 "
    )


def _swedish_or_skip():
    module = cat._module("sv")
    if module is None:
        pytest.skip("the Swedish catalog did not ship")
    return module


# --------------------------------------------------------------------------
# the alias path answers at all
# --------------------------------------------------------------------------

def test_slot_five_borrows_slot_ones_label_and_renumbers_it():
    translated = cat.setting_label(ALIAS_KEY, _alias_label_source(), "sv")
    primary = cat.setting_label(PRIMARY_KEY, cat._english().SETTING_LABELS[
        PRIMARY_KEY], "sv")
    assert translated is not None
    assert primary is not None
    assert "5" in translated
    assert "1" not in translated
    # It really is slot one's prose, with the one digit moved on.
    assert translated == primary.replace("1", "5", 1)


def test_slot_five_borrows_slot_ones_tooltip_with_its_own_prefix():
    translated = cat.setting_tooltip(ALIAS_KEY, _alias_tooltip_source(), "sv")
    assert translated is not None
    assert "organellee_" in translated
    assert "organelle_" not in translated.replace("organellee_", "")


# --------------------------------------------------------------------------
# organelle_types refuses the key -- the `except` arm
# --------------------------------------------------------------------------

def test_a_label_alias_falls_back_to_english_when_the_slot_map_refuses(
        monkeypatch):
    """``organelle_role_of`` raising must cost the alias its translation and
    nothing else -- ``role`` is cleared, so the alias check fails closed."""
    source = _alias_label_source()
    assert cat.setting_label(ALIAS_KEY, source, "sv") is not None

    import spacr.organelle_types as organelle_types

    def _refuses(_key):
        raise ValueError("no slot map in this build")

    monkeypatch.setattr(organelle_types, "organelle_role_of", _refuses)
    assert cat.setting_label(ALIAS_KEY, source, "sv") is None
    # A key the alias path never reaches is unaffected.
    canonical = cat._english().SETTING_LABELS[PRIMARY_KEY]
    assert cat.setting_label(PRIMARY_KEY, canonical, "sv") is not None


def test_a_tooltip_alias_falls_back_to_english_when_the_slot_map_refuses(
        monkeypatch):
    source = _alias_tooltip_source()
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is not None

    import spacr.organelle_types as organelle_types

    def _refuses(_key):
        raise ValueError("no slot map in this build")

    monkeypatch.setattr(organelle_types, "organelle_role_of", _refuses)
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is None
    canonical = cat._english().SETTING_TOOLTIPS[PRIMARY_KEY]
    assert cat.setting_tooltip(PRIMARY_KEY, canonical, "sv") is not None


def test_an_import_error_inside_the_alias_path_is_an_english_fallback(
        monkeypatch):
    """``organelle_types`` is imported lazily inside the function; when it
    cannot be imported the alias must answer None rather than raise."""
    import builtins

    source = _alias_tooltip_source()
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is not None

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "spacr.organelle_types":
            raise ImportError("organelle_types is not available")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(__import__("sys").modules, "spacr.organelle_types",
                        raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is None


# --------------------------------------------------------------------------
# the alias check passes but the language has no catalog
# --------------------------------------------------------------------------

def test_english_gets_no_alias_translation_because_english_is_the_source():
    label_source = _alias_label_source()
    tooltip_source = _alias_tooltip_source()
    assert cat.setting_label(ALIAS_KEY, label_source, "sv") is not None
    assert cat.setting_tooltip(ALIAS_KEY, tooltip_source, "sv") is not None
    assert cat.setting_label(ALIAS_KEY, label_source, "en") is None
    assert cat.setting_tooltip(ALIAS_KEY, tooltip_source, "en") is None
    assert cat.setting_tooltip(ALIAS_KEY, tooltip_source, "xx_YY") is None


# --------------------------------------------------------------------------
# the catalog exists but has no record for the primary key
# --------------------------------------------------------------------------

def test_an_alias_of_an_untranslated_primary_is_english(monkeypatch):
    """The alias renumbers slot one's TRANSLATION. With no translation of
    slot one there is nothing to renumber, and English must win."""
    module = _swedish_or_skip()
    source = _alias_label_source()
    assert cat.setting_label(ALIAS_KEY, source, "sv") is not None

    labels = dict(getattr(module, "SETTING_LABELS", {}))
    labels.pop(PRIMARY_KEY, None)
    monkeypatch.setattr(module, "SETTING_LABELS", labels)
    assert cat.setting_label(ALIAS_KEY, source, "sv") is None


def test_an_alias_tooltip_of_an_untranslated_primary_is_english(monkeypatch):
    module = _swedish_or_skip()
    source = _alias_tooltip_source()
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is not None

    tooltips = dict(getattr(module, "SETTING_TOOLTIPS", {}))
    tooltips.pop(PRIMARY_KEY, None)
    monkeypatch.setattr(module, "SETTING_TOOLTIPS", tooltips)
    assert cat.setting_tooltip(ALIAS_KEY, source, "sv") is None
