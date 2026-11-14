"""Table-driven tests of column parsing and feature search in ``spacr.feature_dict``.

``parse_column`` turns a ``measurements.db`` column name back into a described
measurement; its merge-suffix and ``__dup<n>`` rules must back out when what
is left is not a real statistic, so an unknown column stays unknown.
``search_features`` is what a user types into when they do not know spaCR's
naming scheme: titles must match spaced words, a family filter must only
remove hits, and a query of stopwords must match nothing.  One general test
per unit, each walking a case table with pytest's ``subtests`` fixture.
"""
from __future__ import annotations

from spacr import feature_dict as fd

UNKNOWN = ("Not recognised by the spaCR feature dictionary. Its meaning was not "
           "determined from the spaCR source; check the code that wrote the table, "
           "or a user-supplied custom feature (see spacr.custom_features).")
MERGE_NOTE = ("The trailing '_nucleus' is a pandas merge suffix added when the object "
              "tables were joined (spacr.io._read_and_join_tables), not part of the "
              "feature name.")
DUP_NOTE = ("Occurrence {n} of a duplicated column name: spacr.utils._check_integrity "
            "suffixes every repeat after the first, so this is another copy of 'area'.")
SUMMARY_PREFIX_NOTE = (
    "Column carries the organelle_summary_ prefix written by "
    "spacr.measure._measure_crop_core, but the summary statistic is not in the "
    "dictionary. ")


def test_parse_column_strips_suffixes_only_when_a_real_stat_is_left(monkeypatch, subtests):
    """Merge and dup suffixes resolve to the stat beneath, or back out to 'unknown'."""
    area_notes = fd.parse_column("cell_area").notes
    cases = [
        # column, key, family, object_type, object_type_2, exact notes
        ("cell_area", "area", "morphology", "cell", None, area_notes),
        ("cell_area_nucleus", "area", "morphology", "cell", "nucleus",
         MERGE_NOTE + " " + area_notes),
        ("cell_zzznotastat_nucleus", None, "unknown", "cell", None, UNKNOWN),
        ("cell_area__dup2", "area", "morphology", "cell", None,
         DUP_NOTE.format(n=2) + " " + area_notes),
        ("cell_area__dup7", "area", "morphology", "cell", None,
         DUP_NOTE.format(n=7) + " " + area_notes),
        ("cell_zzznotastat__dup2", None, "unknown", "cell", None, UNKNOWN),
        ("organelle_summary_organelleb_lysosome_ratio", None, "unknown", "organelleb",
         None, SUMMARY_PREFIX_NOTE + UNKNOWN),
    ]
    for column, key, family, object_type, object_type_2, notes in cases:
        with subtests.test(msg=column):
            entry = fd.parse_column(column)
            assert entry.column == column
            assert (entry.key, entry.family, entry.object_type,
                    entry.object_type_2) == (key, family, object_type, object_type_2)
            assert entry.notes == notes
    assert area_notes.startswith("In 2-D, multiply by (um/px)^2")

    with subtests.test(msg="a role-spelled summary key resolves under its own name"):
        name = "organelle_summary_organelleb_lysosome_ratio"
        assert name not in fd.KNOWN_PROPERTIES
        assert "organelle_summary_organelle_lysosome_ratio" not in fd.KNOWN_PROPERTIES
        info = fd.PropertyInfo(
            family="morphology",
            description="Share of the parent occupied by role-b organelles.",
            unit="fraction",
            computed_by="spacr.measure._summarize_organelles_per_parent",
            notes=None)
        with monkeypatch.context() as patch:
            patch.setitem(fd.KNOWN_PROPERTIES, name, info)
            entry = fd.parse_column(name)
        assert (entry.key, entry.family, entry.unit, entry.description) == (
            name, "morphology", "fraction", info.description)
        assert fd.parse_column(name).family == "unknown"


def test_search_features_matches_titles_filters_families_and_ignores_stopwords(subtests):
    """Exact hits for spaced words, family filtering as a pure subset, stopwords as nothing."""
    everything = fd.search_features("")
    with subtests.test(msg="an empty query lists every doc"):
        assert [hit.doc.key for hit in everything] == [doc.key for doc in fd.feature_docs()]
        assert {hit.reason for hit in everything} == {"listed"}
        assert len(everything) > 100

    for query in ("of", "the", "of the", "and to with"):
        with subtests.test(msg=f"stopwords only: {query!r}"):
            assert fd.search_features(query) == []

    with subtests.test(msg="spaced words match the humanised title"):
        hits = fd.search_features("major axis")
        assert [(hit.doc.key, hit.reason) for hit in hits] == [
            ("major_axis_length", "title contains the query"),
            ("organelle_summary_organelle_mean_major_axis", "title contains the query")]
        assert all("major axis" not in hit.doc.key for hit in hits)

    with subtests.test(msg="one real term reaches the definitions"):
        hits = fd.search_features("homogeneity")
        assert (hits[0].doc.key, hits[0].reason) == (
            "homogeneity_distance_<d>", "name contains the query; concept: texture")
        assert any("mentioned in the definition" in hit.reason for hit in hits)
        assert 0 < len(hits) < len(everything)

    unfiltered = fd.search_features("area")
    with subtests.test(msg="'area' spans three families"):
        assert {hit.doc.family for hit in unfiltered} == {"meta", "morphology", "spatial"}
        assert unfiltered[0].doc.key == "area"
    for family in ("morphology", "meta", "spatial", "no_such_family"):
        with subtests.test(msg=f"family filter {family}"):
            filtered = fd.search_features("area", family=family)
            assert [(h.doc.key, h.reason) for h in filtered] == [
                (h.doc.key, h.reason) for h in unfiltered if h.doc.family == family]
            assert len(filtered) < len(unfiltered)
    assert fd.search_features("area", family="morphology")[0].doc.key == "area"
