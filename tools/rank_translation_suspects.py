#!/usr/bin/env python
"""Rank translated strings by how likely a domain term went wrong in them.

NOT A GATE, AND IT MUST NOT BECOME ONE. The failures this is built from are
single wrong WORDS inside fluent sentences, and no automatic check can decide
whether a fluent sentence means the right thing. What it can do is put the
fifty most suspicious strings in front of a person who reads the language, in
priority order, so a review that would otherwise start at string 1 of 5,226
starts where the evidence is.

WHY NOT ROUND-TRIP. Round-translating and comparing is the obvious lever and
it flatters exactly this defect: "Cochez ce qui n'a pas ete pietine" round-
trips to something plausible because it IS a fluent French sentence. It is
only wrong about what spaCR does.

THE FIVE OBSERVED FAILURES this is built from, all real, all shipped:

    fr  invert    -> "pietine"        (trampled)
    zh  invert    -> "点燃"   (ignited)
    es  Drops     -> "Gotas"          (droplets, not "removes")
    es  organelle -> "orgÃ¡ngeles"  (Los Angeles)
    es  organelle -> "organelle"      (left in English inside Spanish prose)

Four of the five are a term rendered as a homograph of the wrong SENSE, and
the fifth is the term not rendered at all. Both are visible without knowing
the language: one substitutes a word from a list of known wrong reaches, the
other leaves the English word standing.

USAGE
    python tools/rank_translation_suspects.py            # every locale
    python tools/rank_translation_suspects.py --language es --limit 30
"""
from __future__ import annotations

import argparse
import importlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

#: ``term -> {locale: (approved targets, cognates the model reaches for)}``.
#:
#: DELIBERATELY SMALL. Every row is a term whose correct target in that
#: language is something the maintainer or a reviewer can confirm in one
#: glance, and a wrong cognate that has actually been SEEN rather than
#: imagined. A guessed row makes the ranking worse than empty, because a
#: false positive at the top of the list is what stops anyone reading the
#: rest of it.
GLOSSARY: dict[str, dict[str, tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "organelle": {
        "es": (("orgánulo", "orgánulos"), ("orgángel", "orgángeles",
                                           "orgánicas", "orgálelas")),
        "fr": (("organite", "organites"), ()),
        # "Organelle" IS German, alongside "Organell". Without it here the
        # tool reported 269 German suspects for this row and every one of
        # them read correctly -- the largest locale, at the top of the
        # list, which is where a false positive costs the most.
        "de": (("Organell", "Organelle", "Organellen"), ()),
        "sv": (("organell", "organeller"), ()),
        "pt": (("organelo", "organelos", "organela", "organelas"), ()),
    },
    # "Drops X" in a spaCR tooltip always means REMOVES X. The Spanish
    # catalog rendered it as the noun -- droplets.
    "drops": {
        "es": (("elimina", "descarta", "suprime"), ("gotas", "gotitas")),
        "fr": (("supprime", "écarte", "retire"), ("gouttes", "gouttelettes")),
        "pt": (("remove", "descarta"), ("gotas",)),
    },
    # The OTHER `log`. Where the key is not one of LOG_IS_AN_ACRONYM the
    # word means logarithm, and *Protokoll* / *registro* is the logbook
    # sense -- a wrong sense rather than a lost acronym, so it belongs
    # here and not in PROTECTED_TOKENS.
    "log": {
        "de": (("log", "logarithmisch", "logarithmische",
                "logarithmischen"), ("protokoll", "logbuch")),
        "es": (("log", "logarítmico", "logarítmica"),
               ("registro", "diario", "bitácora")),
        "fr": (("log", "logarithmique"), ("journal", "registre")),
    },
    "invert": {
        "fr": (("inverse", "inverser"), ("piétiné", "pietine")),
        "zh_CN": (("反转", "反选"), ("点燃",)),
        "sv": (("invertera", "omvänd"), ()),
        "is": (("snúa", "umsnúa"), ()),
    },
}

#: Tables worth ranking: the ones a user reads while deciding something.
TABLES = ("SETTING_TOOLTIPS", "SETTING_LABELS", "MODULE_SUMMARIES")

HIGH, ACRONYM, MEDIUM, LOW = ("wrong-cognate", "translated-acronym",
                              "left-in-english", "no-known-target")

#: Tokens that must survive translation unchanged, and what the model
#: reaches for when it does not recognise them as tokens.
#:
#: A DIFFERENT FAULT FROM THE GLOSSARY ABOVE, and it wants its own table.
#: `GLOSSARY` catches "the model chose the wrong sense of an English
#: word". This catches "the model did not know a domain ACRONYM was an
#: acronym at all" -- German rendered ``dog`` as *Hund* and ``log`` as
#: *Protokoll*, so `organelle_dog_sigma_high` reads "Hundesigma hoch" and
#: `organelle_log_max_sigma` reads "Protokoll max sigma". DoG is
#: Difference of Gaussians and LoG is Laplacian of Gaussians, and every
#: organelle slot carries both, so neither is one string.
#:
#: Found by the other session reading this tool's LOW-severity output for
#: German, where it had flagged nothing: the acronyms were invisible to
#: the glossary because no wrong cognate had been recorded for them. They
#: are a severity of their own rather than glossary rows because the test
#: is "did this token change", which needs no per-locale target.
PROTECTED_TOKENS: dict[str, tuple[str, ...]] = {
    "dog": ("hund", "hunde", "chien", "perro", "cane", "cão"),
    "log": ("protokoll", "logbuch", "journal", "registro", "diario"),
    "mip": (),
    "sam": (),
    "clahe": (),
    "otsu": (),
}

#: The four suffixes where ``log`` is Laplacian of Gaussians.
#:
#: ONE SPELLING, TWO WORDS, AND ONLY ONE OF THEM IS AN ACRONYM. `log` is
#: LoG in `organelle_log_max_sigma` and an ordinary logarithm in `log_x`,
#: `log_y` and `log_data`, which are axis and transform switches on a
#: plot. Protecting the token everywhere would report a log axis as a
#: mistranslated filter -- the same shape as the *Protokoll* false
#: positive one level down, and the reason `log` could not simply join
#: `dog` in the runtime's CASED_TERMS.
LOG_IS_AN_ACRONYM = ("log_max_sigma", "log_min_sigma",
                     "log_num_sigma", "log_threshold")

#: Keys where GLOSSARY term ``log`` means logarithm. Everywhere else the
#: word means logbook and *Protokoll* is the RIGHT German.
#:
#: THE THIRD SENSE, found by running the second one. Adding a `log`
#: glossary row for the logarithm caught `log_x` -> "Protokoll x"
#: correctly and then flagged three strings where the English says "the
#: log prints a warning" and "the curation log" -- records, not
#: logarithms, and German's *Protokoll* / *Kurationsprotokoll* is exactly
#: right for them. Prose almost always means the logbook; only these
#: plot switches mean the function, so the row is scoped to them and not
#: the other way round.
LOG_IS_A_LOGARITHM = ("log_x", "log_y", "log_data")

#: ``GLOSSARY term -> the only keys it applies to``. Absent means "every
#: key", which is true of every term but this one.
TERM_KEY_SCOPE: dict[str, tuple[str, ...]] = {"log": LOG_IS_A_LOGARITHM}


def _token_is_a_term_of_art(token: str, key: str) -> bool:
    """Whether ``token`` carries its domain sense in this setting name.

    :param token: a key of :data:`PROTECTED_TOKENS`.
    :param key: the catalog key the string was found under.
    :returns: ``True`` where the token must survive translation.
    """
    if token not in str(key).lower().split("_"):
        return False
    if token == "log":
        return any(str(key).lower().endswith(s) for s in LOG_IS_AN_ACRONYM)
    return True


def _catalog(language: str):
    return importlib.import_module(f"spacr.qt.i18n_catalogs.{language}")


def _contains_word(haystack: str, needle: str) -> bool:
    """Word-boundary match, case-insensitive, on either side.

    Substring matching would report "organelle" inside "organelles" as a
    separate hit and count the same string twice.
    """
    return re.search(rf"(?i)\b{re.escape(needle)}\b", haystack) is not None


def _starts_a_word(haystack: str, needle: str) -> bool:
    """Whether any word in ``haystack`` BEGINS with ``needle``.

    FOR THE WRONG-COGNATE LIST ONLY, and because German compounds. The
    logbook sense of `log` reached the catalog three ways -- "Protokoll
    x", "Protokollieren y", "Protokolldaten" -- and a whole-word match
    finds one of the three. The other two are the same defect wearing an
    inflection and a compound.

    Deliberately not used for the approved list: matching a prefix there
    would accept a word that merely STARTS like the right one, which is
    how a checker starts passing things it should not.
    """
    return re.search(rf"(?i)\b{re.escape(needle)}", haystack) is not None


def suspects(language: str):
    """Ranked suspects for one locale, worst first.

    :param language: catalog language code.
    :returns: list of ``(severity, table, key, term, snippet)``.
    """
    english = _catalog("en")
    translated = _catalog(language)
    found = []
    for table in TABLES:
        source = getattr(english, table, {}) or {}
        target = getattr(translated, table, {}) or {}
        for key, text in source.items():
            rendered = target.get(key)
            if not rendered or rendered == text:
                continue                     # untranslated wholesale: not this
            for term, locales in GLOSSARY.items():
                if language not in locales:
                    continue
                if not _contains_word(text, term):
                    continue
                scope = TERM_KEY_SCOPE.get(term)
                if scope is not None and str(key).lower() not in scope:
                    continue                 # a different sense of the word
                approved, wrong = locales[language]
                hit = next((w for w in wrong
                            if _starts_a_word(rendered, w)), None)
                if hit is not None:
                    found.append((HIGH, table, key, f"{term} -> {hit}",
                                  rendered[:110]))
                elif any(_contains_word(rendered, a) for a in approved):
                    continue                 # rendered correctly
                elif _contains_word(rendered, term):
                    # THE ENGLISH WORD IS SOMETIMES THE RIGHT WORD.
                    # "Organelle" IS the German for organelle, and the
                    # `organelle` row produced 269 German suspects of
                    # which every one read correctly -- a false positive
                    # at the top of the largest locale, which is where it
                    # does the most damage. A term that is its own
                    # approved target in this language cannot be
                    # "left in English".
                    if any(a.lower() == term.lower() for a in approved):
                        continue
                    found.append((MEDIUM, table, key, term, rendered[:110]))
                else:
                    found.append((LOW, table, key, term, rendered[:110]))
            for token, wrong in PROTECTED_TOKENS.items():
                # THE KEY, NOT THE PROSE, and for `log` not even every
                # key -- see `_token_is_a_term_of_art`. "log" is an
                # acronym in `organelle_log_max_sigma`, a logarithm in
                # `log_x`, and an ordinary verb in "while logging each
                # edit", which German renders as *Protokoll* correctly.
                # Only the first is this pass's business; the second is a
                # GLOSSARY row and the third is nobody's.
                if not _token_is_a_term_of_art(token, key):
                    continue
                if not _contains_word(text, token):
                    continue
                if _contains_word(rendered, token):
                    continue                 # survived, which is the ask
                seen = next((w for w in wrong
                             if _contains_word(rendered, w)), None)
                found.append((ACRONYM, table, key,
                              f"{token} -> {seen or 'gone'}",
                              rendered[:110]))
    order = {HIGH: 0, ACRONYM: 1, MEDIUM: 2, LOW: 3}
    return sorted(found, key=lambda row: (order[row[0]], row[1], row[2]))


def renderings(language: str, term: str) -> dict:
    """How many distinct approved forms of ``term`` this locale uses.

    ONE TERM, NAMED FIVE WAYS, is a reading problem no other check we
    have can see. The Spanish catalog renders `organelle` as orgánulo,
    organelas, organela, orgánicos and organole across one panel -- every
    one of them arguably a translation, and a reader meeting all five
    cannot tell whether they are the same object.

    Counts only what the glossary already knows to be an approved target,
    so it makes no judgement about which spelling is right. Deciding that
    is a reader's job; this says there is a decision to make.

    WHAT IT DOES NOT YET CATCH, said plainly so nobody reads more into
    the output than is there: the five Spanish forms above are not all
    approved targets -- organelas, organela, orgánicos and organole are
    wrong, not variants -- so this counts one form and reports no spread.
    What it currently surfaces is INFLECTION: singular against plural,
    which is grammar rather than inconsistency. Catching the real thing
    needs a notion of "any rendering of this term", which is the hard
    half and is not attempted here.

    :param language: catalog language code.
    :param term: a key of :data:`GLOSSARY`.
    :returns: ``{approved form: how many records use it}``.
    """
    english = _catalog("en")
    translated = _catalog(language)
    approved = GLOSSARY.get(term, {}).get(language, ((), ()))[0]
    counts = {}
    for table in TABLES:
        source = getattr(english, table, {}) or {}
        target = getattr(translated, table, {}) or {}
        for key, text in source.items():
            rendered = target.get(key)
            if not rendered or not _contains_word(text, term):
                continue
            for form in approved:
                if _contains_word(rendered, form):
                    counts[form] = counts.get(form, 0) + 1
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--language", action="append", dest="languages")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    languages = args.languages or sorted(
        {loc for row in GLOSSARY.values() for loc in row})
    total = 0
    for language in languages:
        rows = suspects(language)
        total += sum(1 for r in rows if r[0] == HIGH)
        print(f"\n=== {language}: {len(rows)} suspect(s), "
              f"{sum(1 for r in rows if r[0] == HIGH)} of them wrong-cognate")
        for severity, table, key, term, snippet in rows[:args.limit]:
            print(f"  [{severity:15s}] {table}.{key}  ({term})")
            print(f"      {snippet}")
    print("\n=== how many ways each term is rendered")
    for language in languages:
        for term in sorted(GLOSSARY):
            if language not in GLOSSARY[term]:
                continue
            counts = renderings(language, term)
            if len(counts) > 1:
                spread = ", ".join(f"{form} x{n}"
                                   for form, n in sorted(counts.items(),
                                                         key=lambda kv: -kv[1]))
                print(f"  {language}/{term}: {len(counts)} forms -- {spread}")
    print(f"\n{total} wrong-cognate hit(s) across {len(languages)} locale(s). "
          f"This RANKS; it does not judge. A fluent reader decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
