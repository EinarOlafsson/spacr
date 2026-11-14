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

from spacr import feature_dict as fd


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
