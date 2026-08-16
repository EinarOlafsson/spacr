"""Mechanical correctness checks on the translation catalogs.

Instruction 112. These exist because the sweep that produced them will be
asked for again -- it already has been, for work that was closed -- and the
only durable answer to "is the translation correct" is a check that fails
when it stops being.

Every check here PASSES today. That is the point: they are ratchets, not
bug reports. What they buy is that the next catalog edit cannot quietly
break a placeholder, lose a column name, or add an untranslated string.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

#: `{name}` style. ``tr()`` applies ``str.format`` AFTER translation, so a
#: catalog that drops or renames one raises at format time -- in that locale
#: only, which means it ships.
PLACEHOLDER = re.compile(r"\{[^{}]*\}")

#: Identifiers that are spaCR's data model, not prose. A translated
#: ``columnID`` is a broken database, not a localised one.
PROTECTED_TERMS = (
    "columnID", "rowID", "plateID", "fieldID", "objectID",
    "prcfo", "prcf", "prc", "spaCR",
)

#: Strings that are DELIBERATELY identical to their English source, with the
#: reason. Anything identical and not listed here is an untranslated entry
#: masquerading as a translated one -- it passes a coverage count while
#: showing the user English.
#:
#: Three kinds, all legitimate:
#:   * proper nouns and product names -- "Toxoplasma", "spaCR AI", "AI";
#:   * loanwords that really are spelled the same in that language --
#:     "Navigation" in German and French, "Status" in Swedish, "Console" in
#:     French, "Regression" in German;
#:   * format strings and technical tokens whose content is not prose.
DELIBERATELY_UNTRANSLATED = {
    "AI": {"ko", "sv"},
    "API: {url}": {"de", "es", "hi", "is", "ko", "pt", "sv"},
    "Actions": {"fr"},
    "Activation": {"fr"},
    "Backend": {"de"},
    "Console": {"fr", "pt"},
    "Data": {"sv"},
    "Demos": {"de"},
    "Design": {"sv"},
    "Documentation (web)": {"fr"},
    "Live": {"de", "sv"},
    "Module": {"fr"},
    "Navigation": {"de", "fr"},
    "Pause": {"de", "fr"},
    "Personal Access Token (ghp_… / github_pat_…)": {"hi"},
    "Regression": {"de", "sv"},
    "Status": {"de", "pt", "sv"},
    "Toxoplasma": {"de", "es", "fr", "is", "pt", "sv"},
    "Tutorial (web)": {"es", "pt"},
    "optional": {"de"},
    "spaCR AI": {"ko", "sv"},
    "tuple": {"fr"},
}

INSTALLER_CATALOGS = pathlib.Path("packaging/i18n")


def _runtime_catalogs():
    from spacr.qt.i18n import CATALOGS

    return CATALOGS


# --------------------------------------------------------------------------- #
#  Runtime catalogs -- what the app itself shows
# --------------------------------------------------------------------------- #

def test_every_runtime_translation_keeps_its_placeholders():
    """A dropped ``{name}`` raises at format time, in one locale only."""
    broken = []
    for language, catalog in sorted(_runtime_catalogs().items()):
        for source, translated in catalog.items():
            if not isinstance(translated, str):
                continue
            want = sorted(PLACEHOLDER.findall(source))
            got = sorted(PLACEHOLDER.findall(translated))
            if want != got:
                broken.append(f"{language}: {source!r} {want} -> {got}")
    assert not broken, (
        "translations changed their placeholders:\n  " + "\n  ".join(broken))


def test_no_runtime_translation_loses_a_protected_term():
    """``columnID`` is a column name. Translating it breaks a join."""
    leaks = []
    for language, catalog in sorted(_runtime_catalogs().items()):
        for source, translated in catalog.items():
            if not isinstance(translated, str):
                continue
            for term in PROTECTED_TERMS:
                if term in source and term not in translated:
                    leaks.append(f"{language}: {term!r} lost from {source!r}")
    assert not leaks, "\n  ".join(leaks)


def test_an_untranslated_entry_is_declared_or_absent():
    """Identical to English is only acceptable on purpose.

    A catalog entry whose value is its own key passes a coverage count while
    showing the user English, which is the one failure mode a coverage count
    cannot see.
    """
    undeclared = []
    for language, catalog in sorted(_runtime_catalogs().items()):
        for source, translated in catalog.items():
            if not isinstance(translated, str) or translated != source:
                continue
            if language in DELIBERATELY_UNTRANSLATED.get(source, ()):
                continue
            undeclared.append(f"{language}: {source!r}")
    assert not undeclared, (
        "these entries are identical to their English source and not declared "
        "deliberate:\n  " + "\n  ".join(undeclared)
        + "\n\nEither translate them, or add them to "
          "DELIBERATELY_UNTRANSLATED with the reason.")


def test_the_allowlist_does_not_outlive_what_it_excuses():
    """A stale allowlist is how a ratchet becomes a rubber stamp.

    If an entry gets translated, its excuse must go, or the next
    untranslated string can hide behind it.
    """
    catalogs = _runtime_catalogs()
    stale = []
    for source, languages in DELIBERATELY_UNTRANSLATED.items():
        for language in languages:
            catalog = catalogs.get(language, {})
            if source in catalog and catalog[source] != source:
                stale.append(f"{language}: {source!r} is now translated")
    assert not stale, (
        "remove these from DELIBERATELY_UNTRANSLATED:\n  " + "\n  ".join(stale))


def test_every_language_carries_the_same_keys():
    """A key present in one catalog and missing from another is a string
    that silently falls back to English in that locale alone."""
    catalogs = _runtime_catalogs()
    sizes = {lang: len(cat) for lang, cat in catalogs.items()}
    assert len(set(sizes.values())) == 1, f"catalogs differ in size: {sizes}"


# --------------------------------------------------------------------------- #
#  Installer catalogs -- shipped as JSON beside the installer
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not INSTALLER_CATALOGS.is_dir(),
                    reason="installer catalogs not in this checkout")
def test_installer_catalogs_agree_with_english():
    """Same keys and same placeholders as ``en.json``, in every language."""
    english = json.loads((INSTALLER_CATALOGS / "en.json").read_text())
    problems = []
    for path in sorted(INSTALLER_CATALOGS.glob("*.json")):
        if path.stem == "en":
            continue
        catalog = json.loads(path.read_text())
        for key in english:
            if key not in catalog:
                problems.append(f"{path.stem}: missing key {key!r}")
        for key, value in catalog.items():
            if not isinstance(value, str) or not isinstance(
                    english.get(key), str):
                continue
            want = sorted(PLACEHOLDER.findall(english[key]))
            got = sorted(PLACEHOLDER.findall(value))
            if want != got:
                problems.append(
                    f"{path.stem}: {key!r} placeholders {want} -> {got}")
    assert not problems, "\n  ".join(problems)
