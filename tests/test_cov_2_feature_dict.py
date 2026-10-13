"""Fallbacks and filters in the feature dictionary that nothing else drives.

The dictionary is what turns a `measurements.db` column name back into a
described measurement, and what a user searches when they do not know spaCR's
naming scheme. The paths here are the ones taken when the ordinary answer is
not available: a summary column with no canonical twin, a query that only
matches a title, a family filter that removes a doc, a duplicate unknown
column. Each of them is a wrong answer a user could not tell from a right one,
so each is pinned.
"""
from __future__ import annotations

import dataclasses

import pytest

from spacr import feature_dict as fd


# ---------------------------------------------------------------------------
# parsing an organelle summary column
# ---------------------------------------------------------------------------

def test_a_role_spelled_summary_key_resolves_to_itself(monkeypatch):
    """A curated key written under one organelle role keeps its own name.

    ``organelle_summary_*`` columns are normally curated once under the
    generic ``organelle`` role and every role's column is mapped onto that
    canonical key. A key curated for one role specifically has no canonical
    twin to map to, and must resolve under its own literal name rather than
    falling through to "not in the dictionary" -- which is what the user would
    be told about a column the dictionary does describe.

    The role-specific key is injected because the shipped table has none; the
    branch exists so that adding one does not silently break its own column.
    """
    name = "organelle_summary_organelleb_lysosome_ratio"
    assert name not in fd.KNOWN_PROPERTIES
    assert "organelle_summary_organelle_lysosome_ratio" not in fd.KNOWN_PROPERTIES

    info = fd.PropertyInfo(
        family="morphology",
        description="Share of the parent occupied by role-b organelles.",
        unit="fraction",
        computed_by="spacr.measure._summarize_organelles_per_parent",
        notes=None,
    )
    monkeypatch.setitem(fd.KNOWN_PROPERTIES, name, info)

    entry = fd.parse_column(name)

    assert entry.key == name, "resolved under its own name, not a canonical one"
    assert entry.family == "morphology"
    assert entry.unit == "fraction"
    assert entry.description == info.description

    # And without the injection the same column is honestly unexplained.
    monkeypatch.delitem(fd.KNOWN_PROPERTIES, name)
    assert fd.parse_column(name).family == "unknown"


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------

def test_coverage_lists_a_repeated_unknown_column_once():
    """A column repeated in the input must not be repeated in the report.

    ``coverage`` is what a "which of my columns can spaCR explain?" panel
    shows. Two tables can carry the same unexplained column name, and listing
    it twice would make a one-column gap read as a two-column one. The totals
    still count both, because they count columns, not distinct names.
    """
    result = fd.coverage(["cell_area", "zzz_not_a_column", "zzz_not_a_column"])

    assert result.total == 3
    assert result.explained == 1
    assert result.unknown == ("zzz_not_a_column",)


# ---------------------------------------------------------------------------
# the docs
# ---------------------------------------------------------------------------

def test_a_feature_doc_round_trips_through_a_plain_dict():
    """``to_dict`` is how a doc leaves Python -- JSON, a table, a template.

    Every field has to survive, because a docs page that quietly dropped
    ``unit`` or ``written_when`` would read as a feature that has neither.
    """
    doc = fd.doc_for("area")
    assert doc is not None

    as_dict = doc.to_dict()

    assert isinstance(as_dict, dict)
    assert set(as_dict) == {f.name for f in dataclasses.fields(fd.FeatureDoc)}
    assert as_dict["key"] == "area"
    assert as_dict["family"] == doc.family
    assert as_dict["title"] == doc.title
    assert as_dict["examples"] == doc.examples


def test_asking_for_a_key_that_is_not_curated_gets_nothing_back():
    """``doc_for`` answers ``None`` rather than raising or guessing.

    Callers pass keys read off a database that may predate the dictionary, and
    a near-miss substitute would document the wrong measurement.
    """
    assert fd.doc_for("not_a_curated_key") is None
    assert fd.doc_for("") is None
    assert fd.doc_for("area") is not None


def test_a_key_that_humanises_to_nothing_keeps_its_own_spelling():
    """A title is never empty, even for a key made only of separators.

    ``_title_for`` turns underscores into spaces, and a key that is all
    underscores would leave a blank title -- a row in the feature browser with
    no name on it. The key itself is shown instead, which is at least
    something the user can search for.
    """
    assert fd._title_for("__") == "__"
    assert fd._title_for("") == ""
    assert fd._title_for("area_filled") == "Area filled"


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

def test_a_key_outside_a_concepts_own_list_ranks_at_the_bottom():
    """Ranking within a concept is by that concept's list; anything else is 0.

    :func:`search_features` boosts a hit by how characteristic it is of the
    matched concept, read off the position of the key in the concept's own
    ordered list. A key that is not in that list has no position, and must
    score zero rather than propagate a ``ValueError`` out of a search box.
    """
    concept = "texture"
    keys = fd.CONCEPTS[concept].keys
    assert keys, "the concept must have an ordered list for this to mean anything"

    assert fd._concept_rank(concept, "not_a_curated_key") == 0.0
    # The list itself still ranks, most characteristic first.
    assert fd._concept_rank(concept, keys[0]) > fd._concept_rank(concept, keys[-1])


def test_a_query_that_only_matches_the_title_still_finds_the_feature():
    """Words separated by spaces must find keys separated by underscores.

    Nobody types ``major_axis_length``. The title is the humanised key, so
    "major axis" matches there and nowhere else -- and a search that only
    looked at the key would return nothing for the most natural phrasing.
    """
    hits = fd.search_features("major axis")

    keys = [hit.doc.key for hit in hits]
    assert "major_axis_length" in keys
    matched = next(h for h in hits if h.doc.key == "major_axis_length")
    assert "title contains the query" in matched.reason
    assert "major axis" not in matched.doc.key.lower(), (
        "the point is that the key does NOT contain the query")


def test_a_family_filter_removes_docs_from_another_family():
    """Filtering by family must drop the hits that would otherwise be there.

    "area" matches morphology features and the ``meta`` voxel-size columns.
    A filter that widened to everything -- or that was simply ignored -- would
    put metadata rows in a morphology search, which is the failure this
    asserts against by naming both families first.
    """
    unfiltered = fd.search_features("area")
    families = {hit.doc.family for hit in unfiltered}
    assert "meta" in families and "morphology" in families, (
        "this query must span two families or the filter proves nothing")

    filtered = fd.search_features("area", family="morphology")

    assert {hit.doc.family for hit in filtered} == {"morphology"}
    assert len(filtered) < len(unfiltered)
    assert "area" in [hit.doc.key for hit in filtered]
