"""A row with nothing to translate must come back exactly as it went in.

Some catalog rows are made entirely of protected literals: a bare identifier
like ``extra_performance``, a caption like ``Image UMAP…`` whose two words are
both product names, a format string like
``[{severity}] {object_type}: {flags}`` that is nothing but placeholders and
punctuation. There is no prose in them for a translator to change, so asking a
model for a translation can only produce damage.

It did. Before the rule these tests pin, a nine-language rebuild produced:

    'extra_performance'  ->  'extra_performance oder'                  (de)
    'extra_performance'  ->  'extra_performance에 해당되는 글 1건'      (ko)
    'Image UMAP…'        ->  'Image UMAP...'                           (de)
    '[{severity}] {object_type}: {flags}'
                         ->  '({severity}] {object_type}: {flags}'     (de)

The last one turns a matched bracket pair into a mismatched one in a
user-facing severity line. The German ``oder`` is the word "or". The Korean is
a scrape artifact meaning roughly "1 post matching extra_performance".

NONE OF IT WAS VISIBLE TO ANY EXISTING CHECK, and that is the reason this file
exists rather than a one-line fix. Every coverage and exact-English audit asks
whether the target DIFFERS from the English. All four of those differ, so all
four scored as successfully translated, and the damage raised the coverage
number. A check that passes for a reason unrelated to what it is checking is
the failure mode this repository keeps rediscovering.

``_IDENTITY_TEXT`` is the enumerated half of the rule and its own comment
records the failures it was written for (``viridis`` to Korean "virus",
``slurm`` to German "mud"). Enumeration cannot keep up: every new
identifier-shaped caption reintroduces the bug until a human notices it. These
tests pin the DERIVED half.
"""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    import sys
    tools = str(ROOT / "tools")
    sys.path.insert(0, tools)
    try:
        import build_i18n_catalogs
        return build_i18n_catalogs
    finally:
        sys.path.remove(tools)


@pytest.fixture(scope="module")
def catalogs():
    return {
        language: _load(
            f"_cat_{language}",
            ROOT / "spacr" / "qt" / "i18n_catalogs" / f"{language}.py",
        )
        for language in LANGUAGES
    }


# The three rows the rule was written for. Kept as literals rather than
# derived, so that a change to the predicate cannot quietly stop covering the
# exact strings that were damaged.
DAMAGED_ROWS = (
    "extra_performance",
    "Image UMAP…",
    "[{severity}] {object_type}: {flags}",
)


@pytest.mark.parametrize("source", DAMAGED_ROWS)
def test_the_predicate_finds_no_prose_in_these(builder, source):
    """These are the rows with nothing a translator could change."""
    assert not builder._has_prose_outside_protected_literals(source)


@pytest.mark.parametrize("language", LANGUAGES)
@pytest.mark.parametrize("source", DAMAGED_ROWS)
def test_every_locale_leaves_these_rows_exactly_alone(
    catalogs, language, source,
):
    """The row must be byte-identical to the English in all nine locales.

    This is the assertion the coverage audits could not make: they ask whether
    the value DIFFERS from English, and here differing IS the defect.
    """
    assert catalogs[language].UI[source] == source


# Which locale table answers each source table, and how the row is keyed.
# THE KEYING DIFFERS AND CONFLATING IT IS A REAL TRAP: `setting_labels` maps a
# setting NAME to its English label, so the source is the VALUE and the lookup
# key is the NAME; `ui` and `categories` are sequences whose element is both.
# A first version of this sweep looked every English value up as if it were a
# key, and `seg_qc` -- a setting name that also appears as a bare source --
# collided, reporting the German LABEL "Abschnitt QC" as damage to an
# identifier. It is not: they are two different rows that happen to share a
# string.
_TABLES = (
    ("setting_labels", "SETTING_LABELS", "mapping"),
    ("setting_tooltips", "SETTING_TOOLTIPS", "mapping"),
    ("module_summaries", "MODULE_SUMMARIES", "mapping"),
    ("ui", "UI", "sequence"),
    ("categories", "CATEGORY_HELP", "sequence"),
)


@pytest.mark.parametrize("language", LANGUAGES)
def test_no_all_protected_row_was_altered_anywhere(builder, catalogs, language):
    """The general rule, not just the three known rows.

    Sweeps every runtime source. A row with no prose outside its protected
    literals must survive untouched UNLESS a human wrote a reviewed record for
    it -- see the next test for why that exception has to exist.
    """
    reviewed = builder.reviewed_runtime_translations(language)
    catalog = catalogs[language]
    sources = builder.canonical_sources()
    altered = []
    for table_name, catalog_attr, shape in _TABLES:
        table = sources.get(table_name)
        if table is None:
            continue
        rows = (
            table.items() if shape == "mapping"
            else ((element, element) for element in table)
        )
        target = getattr(catalog, catalog_attr, {})
        for key, english in rows:
            english = str(english)
            if builder._has_prose_outside_protected_literals(english):
                continue
            if english in reviewed:
                continue
            value = target.get(key)
            if value is not None and str(value) != english:
                altered.append((catalog_attr, english, str(value)))
    assert altered == [], f"{language}: {altered[:5]}"


@pytest.mark.parametrize("language", LANGUAGES)
def test_a_reviewed_record_still_beats_the_identity_rule(catalogs, language):
    """`{location}: {path}` is all-protected AND deliberately translated.

    A reviewer added the word path/Pfad/chemin/sökväg to all nine locales to
    make the string readable. That is a human decision about a row the derived
    rule would otherwise flatten, which is exactly why the rule sits BELOW the
    reviewed branches rather than above them. If this test goes red, the
    identity rule has been moved above reviewed evidence and is now discarding
    hand-written translations.
    """
    value = catalogs[language].UI["{location}: {path}"]
    assert value != "{location}: {path}", (
        f"{language} lost its hand-written record for this row"
    )
    assert "{location}" in value and "{path}" in value


def test_the_reviewed_record_for_that_row_exists_in_every_locale():
    """Guards the test above from going vacuous.

    If the reviewed records were deleted, every locale would legitimately fall
    back to identity and `test_a_reviewed_record_still_beats_the_identity_rule`
    would start failing for a reason that has nothing to do with branch order.
    Pinning the evidence itself keeps that distinguishable.
    """
    found = set()
    for path in (ROOT / "docs" / "i18n" / "reviewed" / "runtime").rglob("*.json"):
        document = json.loads(path.read_text(encoding="utf-8"))
        records = document.get("records", [])
        if isinstance(records, dict):
            records = list(records.values())
        for row in records:
            if isinstance(row, dict) and row.get("source") == "{location}: {path}":
                found.add(path.parent.name)
    assert set(LANGUAGES) <= found, f"missing reviewed record in {set(LANGUAGES) - found}"
