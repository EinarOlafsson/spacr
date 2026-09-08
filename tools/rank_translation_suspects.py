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
        "de": (("Organell", "Organellen"), ()),
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
    "invert": {
        "fr": (("inverse", "inverser"), ("piétiné", "pietine")),
        "zh_CN": (("反转", "反选"), ("点燃",)),
        "sv": (("invertera", "omvänd"), ()),
        "is": (("snúa", "umsnúa"), ()),
    },
}

#: Tables worth ranking: the ones a user reads while deciding something.
TABLES = ("SETTING_TOOLTIPS", "SETTING_LABELS", "MODULE_SUMMARIES")

HIGH, MEDIUM, LOW = "wrong-cognate", "left-in-english", "no-known-target"


def _catalog(language: str):
    return importlib.import_module(f"spacr.qt.i18n_catalogs.{language}")


def _contains_word(haystack: str, needle: str) -> bool:
    """Word-boundary match, case-insensitive, on either side.

    Substring matching would report "organelle" inside "organelles" as a
    separate hit and count the same string twice.
    """
    return re.search(rf"(?i)\b{re.escape(needle)}\b", haystack) is not None


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
                approved, wrong = locales[language]
                hit = next((w for w in wrong if _contains_word(rendered, w)),
                           None)
                if hit is not None:
                    found.append((HIGH, table, key, f"{term} -> {hit}",
                                  rendered[:110]))
                elif any(_contains_word(rendered, a) for a in approved):
                    continue                 # rendered correctly
                elif _contains_word(rendered, term):
                    found.append((MEDIUM, table, key, term, rendered[:110]))
                else:
                    found.append((LOW, table, key, term, rendered[:110]))
    order = {HIGH: 0, MEDIUM: 1, LOW: 2}
    return sorted(found, key=lambda row: (order[row[0]], row[1], row[2]))


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
    print(f"\n{total} wrong-cognate hit(s) across {len(languages)} locale(s). "
          f"This RANKS; it does not judge. A fluent reader decides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
