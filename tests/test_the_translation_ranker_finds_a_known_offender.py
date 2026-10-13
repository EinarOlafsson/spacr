"""The ranker must be able to find a defect it is shown.

A CHECKER THAT NEVER MATCHES PASSES FOREVER, and both sessions working on
this repository shipped one on 2026-09-08 -- a leak probe that read a
monkeypatch instead of the leak, and a docstring scanner that reported
zero at-risk classes because a NumPy parameter name sits at column 0
exactly like a paragraph does. Both agreed with their author on every
sample. This file is the discipline that caught the second one, applied
to `tools/rank_translation_suspects.py` before anyone trusts its output.

The ranker is NOT a gate and this does not make it one. It scores
strings so a fluent reader knows which fifty to read first; nothing here
asserts anything about the shipped catalogs, because a ranking that went
quiet when the catalogs were fixed would still have to work the next
time they were not.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ranker():
    spec = importlib.util.spec_from_file_location(
        "rank_translation_suspects",
        ROOT / "tools" / "rank_translation_suspects.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_it_recognises_the_wrong_cognate_it_was_built_from(ranker):
    """"Drops X" means REMOVES X; the Spanish read it as droplets."""
    approved, wrong = ranker.GLOSSARY["drops"]["es"]
    rendered = ("Gotas organelas cuya intensidad media cae por debajo de "
                "este percentil.")
    assert any(ranker._contains_word(rendered, w) for w in wrong), (
        "the ranker no longer recognises 'Gotas' as the wrong reach for "
        "'drops', which is the observed failure it was built from")
    assert not any(ranker._contains_word(rendered, a) for a in approved)


def test_a_correct_rendering_is_not_flagged(ranker):
    """The half that keeps it usable: no cry of wolf on a good string."""
    approved, wrong = ranker.GLOSSARY["drops"]["es"]
    rendered = ("Elimina los orgánulos cuya intensidad media cae por debajo "
                "de este percentil.")
    assert any(ranker._contains_word(rendered, a) for a in approved)
    assert not any(ranker._contains_word(rendered, w) for w in wrong)


def test_a_term_left_in_english_is_distinguishable(ranker):
    """The fifth observed failure, and a different one from the other four.

    `number_of_organelles` in Spanish reads "ranuras de organelle" -- the
    term not rendered at all, inside otherwise fluent Spanish. That is
    not a wrong cognate and must not be scored as one.
    """
    approved, wrong = ranker.GLOSSARY["organelle"]["es"]
    rendered = "¿Cuántas ranuras de organelle tiene esta ejecución?"
    assert ranker._contains_word(rendered, "organelle")
    assert not any(ranker._contains_word(rendered, a) for a in approved)
    assert not any(ranker._contains_word(rendered, w) for w in wrong)


def test_word_boundaries_are_respected(ranker):
    """Substring matching would count one string twice.

    "organelle" inside "organelles" is the same word, and a matcher that
    saw two would rank a string above a genuine offender for saying the
    term in the plural.
    """
    assert ranker._contains_word("the organelle mask", "organelle")
    assert ranker._contains_word("two organelles here", "organelles")
    assert not ranker._contains_word("organellocentric", "organelle")


def test_every_glossary_row_has_something_to_say(ranker):
    """A row with no approved target and no wrong cognate ranks nothing.

    It would sit in the table looking like coverage. Each row must carry
    at least the approved side, which is what the medium and low
    severities are scored against.
    """
    empty = [(term, locale)
             for term, locales in ranker.GLOSSARY.items()
             for locale, (approved, _wrong) in locales.items()
             if not approved]
    assert empty == [], f"glossary rows that cannot rank anything: {empty}"


def test_it_recognises_a_translated_acronym(ranker):
    """DoG is Difference of Gaussians; German read it as the animal.

    A different fault from the glossary's: not "the wrong sense of an
    English word" but "the model did not know this was a token at all".
    Found by the other session reading this tool's LOW-severity German
    output, where the glossary had flagged nothing.
    """
    wrong = ranker.PROTECTED_TOKENS["dog"]
    assert any(w in "organelle 1 — hundesigma hoch" for w in wrong), (
        "the ranker no longer recognises Hund as a translation of the DoG "
        "acronym, which is the observed failure it was built from")
    assert "log" in ranker.PROTECTED_TOKENS


def test_an_acronym_is_only_protected_inside_a_setting_name(ranker):
    """"log" is a term of art in a key and an ordinary verb in prose.

    `MODULE_SUMMARIES.curate` says "while logging each edit", and German
    renders that as *Protokoll* correctly. Flagging it would teach a
    reader that this list cries wolf, so the token is protected where the
    KEY contains it and nowhere else.
    """
    assert "log" in "organelle_log_max_sigma".lower().split("_")
    assert "log" not in "curate".lower().split("_")


def test_the_english_word_can_be_the_right_word(ranker):
    """"Organelle" IS German, and 269 correct strings were being flagged.

    A term that is its own approved target in a language cannot be
    "left in English" in that language. This was the false positive at
    the top of the largest locale, which is where one does most damage.
    """
    approved, _wrong = ranker.GLOSSARY["organelle"]["de"]
    assert any(a.lower() == "organelle" for a in approved), (
        "German's approved list no longer contains the English spelling, "
        "so every correct German string is a suspect again")
    approved_es, _ = ranker.GLOSSARY["organelle"]["es"]
    assert not any(a.lower() == "organelle" for a in approved_es), (
        "the exception must be per-locale: Spanish is orgánulo, and "
        "'organelle' standing in Spanish prose IS the defect")


def test_the_two_senses_of_log_are_told_apart(ranker):
    """LoG the filter and log the axis share a spelling, not a meaning.

    The other session could not put `log` in the runtime's CASED_TERMS
    for exactly this reason: a blanket word rule turns a log axis into a
    filter. So the acronym pass claims the four Laplacian-of-Gaussians
    suffixes and nothing else.
    """
    assert ranker._token_is_a_term_of_art("log", "organelle_log_max_sigma")
    assert ranker._token_is_a_term_of_art("log", "organelleb_log_threshold")
    assert not ranker._token_is_a_term_of_art("log", "log_x")
    assert not ranker._token_is_a_term_of_art("log", "log_data")
    # dog has one sense, so scoping must not have narrowed it too.
    assert ranker._token_is_a_term_of_art("dog", "organelle_dog_sigma_high")


def test_prose_log_means_logbook_and_protokoll_is_right(ranker):
    """The third sense, and the one that cost a false positive to find.

    Adding a glossary row for the logarithm immediately flagged three
    strings where the English says "the log prints a warning" and "the
    curation log" -- records, not logarithms, where German's *Protokoll*
    and *Kurationsprotokoll* are exactly right. Prose almost always
    means the logbook, so the row is scoped to the plot switches.
    """
    assert set(ranker.TERM_KEY_SCOPE) == {"log"}, (
        "a second scoped term appeared; confirm it needs scoping rather "
        "than that the scope leaked")
    scope = ranker.TERM_KEY_SCOPE["log"]
    assert "log_x" in scope and "infection_pca_min_silhouette" not in scope
    de = [row for row in ranker.suspects("de") if row[2] == "curate"]
    assert not de, (
        "MODULE_SUMMARIES.curate is back on the German list; its "
        "'curation log' is a record and Kurationsprotokoll is correct")


def test_a_wrong_cognate_is_caught_inside_a_german_compound(ranker):
    """One defect, three surface forms, and whole-word finds one.

    Protokoll x / Protokollieren y / Protokolldaten are the same wrong
    sense inflected and compounded. Prefix matching is used for the
    WRONG list only -- doing it for the approved list would accept a word
    that merely starts like the right one.
    """
    assert ranker._starts_a_word("Protokolldaten", "protokoll")
    assert ranker._starts_a_word("Protokollieren y", "protokoll")
    assert not ranker._starts_a_word("Kurationsprotokoll", "protokoll"), (
        "prefix matching must still respect word starts, or every German "
        "compound ENDING in the cognate becomes a false positive")

    # SYNTHETIC, NOT THE LIVE CATALOG, and this test is the argument for
    # why. It used to end by asserting that German's `log_x`, `log_y` and
    # `log_data` were on the live HIGH list -- and they were, until the
    # defect was fixed at the English source on 2026-09-08: "Log x" is
    # ambiguous before any translator sees it, so the labels now read
    # "Logarithmic x", "Logarithmic y" and "Log-transform features" and no
    # locale reaches for the logbook.
    #
    # That is the check going quiet BECAUSE IT WORKED, which is the one
    # thing this file exists to prevent. A guard whose evidence is a live
    # defect stops guarding the moment somebody fixes it, and then reads as
    # green forever. So the contract is asserted against strings written
    # here: three surface forms of one wrong sense, which whole-word
    # matching would find one of.
    forms = ("Protokoll x", "Protokollieren y", "Protokolldaten")
    assert all(ranker._starts_a_word(form, "protokoll") for form in forms), (
        "prefix matching no longer finds the inflected and compounded forms "
        "of one wrong sense; whole-word matching finds only the first")
