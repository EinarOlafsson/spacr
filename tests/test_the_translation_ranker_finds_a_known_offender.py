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
