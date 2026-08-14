"""The feature dictionary as a LOOKUP, not as an export.

:mod:`spacr.feature_dict` could describe a database into a markdown file
before this; what it could not do was answer "what is this column?" while you
are looking at one. These tests pin the three properties that turn it into an
answer:

* it explains essentially all of a REAL measurements table — the fixture in
  ``tests/data/real_measurement_columns.tsv`` is the column list of two
  shipped databases, so the coverage number is measured, not asserted;
* it resolves a name it has never seen, by composition, because the names are
  composed (object + channel + statistic + qualifier) and a dictionary that
  only matches literals would miss almost every real column;
* it says "unknown" — and nothing else — for a name it cannot explain.

Plus the search layer: a user who does not know the naming scheme has to be
able to find a feature by saying what they mean.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from spacr.feature_dict import (
    CHANNEL_NONE,
    CHANNEL_PAIR,
    CHANNEL_SINGLE,
    CONCEPTS,
    FEATURE_SCOPE,
    KNOWN_PROPERTIES,
    META_COLUMNS,
    OBJECT_TYPES,
    concept_of,
    concepts_for,
    coverage,
    doc_for,
    feature_docs,
    parse_column,
    scope_for,
    search_features,
)

REAL_COLUMNS_FILE = Path(__file__).parent / "data" / "real_measurement_columns.tsv"

#: The bar the resolver has to clear on a real measurements table.
#:
#: It is 100% today and the assertion is deliberately a little below that: the
#: file is a snapshot of two real databases, and a THIRD database with one
#: user-named annotation column in it would not be a regression. A drop to 97%
#: would be — that is a whole family of columns going dark.
REQUIRED_REAL_COVERAGE = 0.99


def _real_columns() -> list[tuple[str, str, str]]:
    """``(database, table, column)`` for every column of the real fixture."""
    rows = []
    for line in REAL_COLUMNS_FILE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        db, table, column = line.split("\t")
        rows.append((db, table, column))
    return rows


# --------------------------------------------------------------------------
# coverage of a real measurements table
# --------------------------------------------------------------------------

def test_the_fixture_is_a_real_table_not_a_toy():
    """Guard the guard: a shrunken fixture would make coverage meaningless."""
    rows = _real_columns()
    assert len(rows) > 1500, "the real-column fixture has been truncated"
    tables = {table for _db, table, _col in rows}
    assert {"cell", "nucleus", "pathogen", "cytoplasm"} <= tables
    columns = {col for _db, _t, col in rows}
    # The four families that make it a measurements table rather than a log.
    assert any(c.endswith("_area") for c in columns)
    assert any("_percentile_" in c for c in columns)
    assert any("_Pearson_correlation" in c for c in columns)
    assert any("_zernike_" in c for c in columns)


def test_resolver_explains_a_real_measurements_table():
    """The headline number: what share of a real table can be explained."""
    columns = [col for _db, _table, col in _real_columns()]
    result = coverage(columns)
    assert result.total == len(columns)
    assert result.fraction >= REQUIRED_REAL_COVERAGE, (
        f"only {result.fraction:.1%} of {result.total} real columns were "
        f"explained; unexplained: {result.unknown[:20]}")


def test_every_real_object_table_column_is_explained():
    """Per table, so a regression cannot hide behind the metadata tables."""
    by_table: dict[str, list[str]] = {}
    for _db, table, column in _real_columns():
        by_table.setdefault(table, []).append(column)
    for table in ("cell", "nucleus", "pathogen", "cytoplasm"):
        result = coverage(by_table[table])
        assert result.fraction == 1.0, (
            f"{table}: {result.unknown}")


def test_the_legacy_columns_that_used_to_be_unknown_are_explained():
    """The four families the gap analysis turned up, named individually.

    Each was a real column in a real database that the dictionary reported as
    unrecognised, and each is now described from what the code did, not from
    what the name suggests.
    """
    entropy = parse_column("cell_channel_1_shannon_entropy")
    assert entropy.family == "intensity"
    # The critical fact: it was measured on the whole field, not the object.
    assert "WHOLE FIELD" in (entropy.description or "")
    assert "identical value" in (entropy.notes or "")

    for name in ("row_name", "column_name"):
        entry = parse_column(name)
        assert entry.family == "meta"
        assert "legacy" in (entry.notes or "").lower()

    counts = parse_column("cell_before_filtration")
    assert counts.family == "meta"
    assert counts.object_type == "cell"
    assert "pivoted_counts" in (counts.notes or "")
    # A per-FIELD count that is spelled like a per-object feature is exactly
    # the sort of thing a dictionary exists to warn about.
    assert "not per object" in (counts.description or "").lower()


@pytest.mark.parametrize("path", [
    "/mnt/firecuda2/methods_paper/plate1/measurements/measurements.db",
    "/mnt/firecuda2/Claude/e2e_work/plate1/measurements/measurements.db",
])
def test_resolver_explains_a_live_database_when_one_is_present(path):
    """The same measurement against the live file, when this machine has it.

    The fixture above is what makes the check portable; this is what keeps the
    fixture honest.
    """
    db = Path(path)
    if not db.is_file():
        pytest.skip(f"no measurements database at {db}")
    columns: list[str] = []
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        for (table,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"):
            columns += [row[1] for row in
                        conn.execute(f'PRAGMA table_info("{table}")')]
    result = coverage(columns)
    assert result.fraction >= REQUIRED_REAL_COVERAGE, result.unknown[:20]


# --------------------------------------------------------------------------
# composition: names nobody has ever written down
# --------------------------------------------------------------------------

@pytest.mark.parametrize("column, key, obj, channel", [
    # A channel index no shipped database has.
    ("cytoplasm_channel_9_percentile_25", "percentile_<p>", "cytoplasm", 9),
    # A GLCM distance outside the default [8, 16, 32].
    ("pathogen_channel_5_homogeneity_distance_128",
     "homogeneity_distance_<d>", "pathogen", 5),
    # A Manders threshold nobody uses, on a channel pair nobody has.
    ("organelle_channel_6_channel_7_M2_correlation_42",
     "M2_correlation_<t>", "organelle", 6),
    # A Zernike index past degree 8.
    ("nucleus_zernike_48", "zernike_<i>", "nucleus", None),
    # A radial bin past the six the pipeline writes.
    ("pathogen_rad_dist_channel_3_bin_11",
     "rad_dist_channel_<c>_bin_<b>", "pathogen", 3),
    # A percentile the object interior does not emit (only the rings do).
    ("cell_channel_0_percentile_50", "percentile_<p>", "cell", 0),
])
def test_an_unseen_but_well_formed_name_resolves_by_composition(
        column, key, obj, channel):
    """None of these appears in any database, and all of them resolve.

    This is the property that matters: the names are COMPOSED, so a
    dictionary that only matched literals would answer "no entry" for most of
    a real table the first time somebody used a fifth channel.
    """
    assert column not in KNOWN_PROPERTIES
    entry = parse_column(column)
    assert entry.family != "unknown"
    assert entry.key == key
    assert entry.object_type == obj
    assert entry.channel == channel
    assert entry.description, f"{column} resolved with no definition"
    # The composed parameter reaches the prose, rather than the entry
    # describing a generic percentile.
    assert entry.column == column


def test_composition_fills_the_parameter_into_the_definition():
    """A composed name is described with ITS number, not the template's."""
    assert "25th percentile" in parse_column(
        "cytoplasm_channel_9_percentile_25").description
    assert "shell 11" in parse_column(
        "pathogen_rad_dist_channel_3_bin_11").description


@pytest.mark.parametrize("column", [
    "wibble",                      # no object prefix, no metadata match
    "cell_wibble_wobble",          # object prefix, unknown statistic
    "cell_channel_1_flurb",        # object + channel, unknown statistic
    "totally made up",             # spaces: not a column name at all
    "channel_0_mean_intensity",    # channel but no object prefix
    "",                            # nothing
])
def test_a_malformed_name_is_reported_unknown_not_invented(column):
    """The dictionary must never guess.

    Answering "probably an intensity" for ``cell_channel_1_flurb`` because it
    looks like one is worse than answering nothing: a user would believe it.
    """
    entry = parse_column(column)
    assert entry.family == "unknown"
    assert entry.description is None
    assert entry.unit is None
    assert entry.computed_by == "unknown"
    assert entry.key is None
    assert entry.object_types == ()
    assert "Not recognised" in (entry.notes or "")


def test_an_unknown_column_still_reports_what_it_could_parse():
    """Unknown is not the same as unparsed: the object prefix is still real."""
    entry = parse_column("cell_channel_1_flurb")
    assert entry.family == "unknown"
    assert entry.object_type == "cell"
    assert entry.channel == 1


# --------------------------------------------------------------------------
# search by concept
# --------------------------------------------------------------------------

@pytest.mark.parametrize("query, expected", [
    ("intensity", "mean_intensity"),
    ("texture", "homogeneity_distance_<d>"),
    ("shape", "eccentricity"),
    ("distance", "distance_to_nucleus"),
    ("size", "area"),
    ("colocalisation", "Pearson_correlation"),
    ("position", "centroid_weighted-0"),
])
def test_a_concept_search_leads_with_the_right_feature(query, expected):
    """The four concepts the task names, plus the three that came with them."""
    hits = search_features(query)
    assert hits, f"no hit for {query!r}"
    assert hits[0].doc.key == expected, [h.doc.key for h in hits[:5]]


@pytest.mark.parametrize("word, concept", [
    ("how big", "size"),
    ("roundness", "shape"),
    ("blurry", "texture"),
    ("brightness", "intensity"),
    ("proximity", "distance"),
    ("heterogeneity", "distribution"),
    ("manders", "colocalisation"),
    ("colocalization", "colocalisation"),   # the other spelling
])
def test_a_synonym_finds_the_concept(word, concept):
    """Nobody types 'equivalent_diameter_area'. They type 'how big'."""
    assert concept_of(word) == concept
    hits = search_features(word)
    assert hits
    assert concept in hits[0].doc.concepts, hits[0].doc.key


def test_texture_search_does_not_return_plain_brightness_first():
    """The concepts have to separate, or they are one concept."""
    top = [h.doc.key for h in search_features("texture", limit=3)]
    assert "mean_intensity" not in top
    assert "homogeneity_distance_<d>" in top


def test_a_pasted_column_name_leads_with_its_own_feature():
    """The commonest gesture: paste the column you are staring at."""
    for column, key in (
        ("cell_channel_1_percentile_75", "percentile_<p>"),
        ("nucleus_channel_0_channel_2_M1_correlation_85", "M1_correlation_<t>"),
        # Case matters in the emitted names, and lower-casing the query used
        # to make every colocalisation column unresolvable.
        ("cell_channel_0_Pearson_correlation", "Pearson_correlation"),
        ("pathogen_zernike_3", "zernike_<i>"),
    ):
        hits = search_features(column)
        assert hits[0].doc.key == key, (column, [h.doc.key for h in hits[:4]])


def test_search_by_substring_finds_the_family():
    keys = [h.doc.key for h in search_features("percentile")]
    assert "percentile_<p>" in keys
    assert "periphery_percentile_<p>" in keys
    assert "outside_percentile_<p>" in keys


def test_object_filter_uses_where_the_feature_is_actually_written():
    """The point of the scope table.

    ``periphery_mean`` exists for the nucleus and not for the cell, because
    ``_intensity_measurements`` guards the block with
    ``if ls[j] in ('nucleus', 'pathogen', 'organelle')``. Nothing in the name
    says so, so the panel has to be told.
    """
    nucleus = [h.doc.key for h in search_features("periphery",
                                                  object_type="nucleus")]
    assert "periphery_mean" in nucleus
    cell = [h.doc.key for h in search_features("periphery",
                                               object_type="cell")]
    assert "periphery_mean" not in cell
    # Zernike is written for four object types and not for the cytoplasm.
    assert "zernike_<i>" in [
        h.doc.key for h in search_features("zernike", object_type="cell")]
    assert "zernike_<i>" not in [
        h.doc.key for h in search_features("zernike", object_type="cytoplasm")]
    # distance_to_nucleus is the cell table's and nobody else's.
    assert "distance_to_nucleus" not in [
        h.doc.key for h in search_features("distance", object_type="nucleus")]


def test_an_unknown_concept_filter_returns_nothing_rather_than_everything():
    assert search_features("", concept="not-a-concept") == []


def test_an_empty_query_lists_everything():
    assert len(search_features("")) == len(feature_docs())


def test_junk_input_returns_nothing_rather_than_everything():
    """What junk *returns* is the point; "did not raise" pins nothing.

    Both directions are failure modes the panel would ship silently. A query
    of punctuation that widened to the whole catalogue looks like a working
    search returning 137 irrelevant rows; a whitespace-only query that stopped
    being stripped would return zero and look like "nothing matches" for what
    is really the empty query.
    """
    catalogue = len(feature_docs())

    # Whitespace is stripped, so "   " is the empty query: list everything.
    for blank in ("", "   ", "\t\n "):
        assert len(search_features(blank)) == catalogue

    # Punctuation and a 500-character run of one letter match nothing at all,
    # and must NOT fall back to the full listing.
    for junk in ("%%%", "<<>>", "a" * 500):
        assert search_features(junk) == [], f"{junk[:12]!r} widened the search"

    # A pasted SQL fragment tokenises to real words ("cell" is mentioned in
    # some definitions), so it finds a handful -- bounded, not the catalogue.
    sql = search_features("SELECT * FROM cell")
    assert 0 < len(sql) < catalogue
    assert all(h.score > 0 and h.reason for h in sql)
    # ...and every hit is a real doc, not a placeholder.
    assert all(h.doc.key for h in sql)


# --------------------------------------------------------------------------
# the scope table, checked against the parser
# --------------------------------------------------------------------------

def test_every_example_column_round_trips_to_its_own_feature():
    """The scope table and the parser cannot drift apart.

    Each doc's example column is BUILT from the scope row (object type,
    channel arity) and then parsed back. A wrong object type or a wrong
    channel arity produces a name that resolves to a different key, or to
    nothing, and this fails.
    """
    for doc in feature_docs():
        for example in doc.examples:
            entry = parse_column(example)
            assert entry.key == doc.key, (
                f"{doc.key}: example {example!r} parsed as {entry.key!r}")
            if doc.object_types:
                assert entry.object_type in doc.object_types


def test_only_the_uncalled_features_have_no_example():
    """A feature with no example column is a feature nothing writes."""
    without = {doc.key for doc in feature_docs() if not doc.examples}
    assert without == {"skeleton_length", "skeleton_branch_points"}
    for key in without:
        assert "not called" in (KNOWN_PROPERTIES[key].notes or "")


def test_every_measured_feature_has_a_scope():
    """No curated feature may be silent about where it is written."""
    missing = [key for key in KNOWN_PROPERTIES if key not in FEATURE_SCOPE]
    assert missing == [], missing


def test_scopes_name_only_real_object_types_and_channel_arities():
    for key, scope in FEATURE_SCOPE.items():
        assert set(scope.objects) <= set(OBJECT_TYPES), key
        assert scope.channels in (CHANNEL_NONE, CHANNEL_SINGLE, CHANNEL_PAIR)
        assert scope.module.startswith("spacr."), key


def test_the_ring_features_are_scoped_to_the_three_objects_that_have_them():
    from spacr.object_roles import ORGANELLE_ROLES
    for key in ("periphery_mean", "outside_mean", "periphery_percentile_<p>",
                "outside_percentile_<p>", "rad_dist_channel_<c>_bin_<b>"):
        assert scope_for(key).objects == (
            "nucleus", "pathogen", *ORGANELLE_ROLES)


def test_the_correlation_features_are_the_only_channel_pair_ones():
    pairs = {key for key, scope in FEATURE_SCOPE.items()
             if scope.channels == CHANNEL_PAIR}
    assert pairs == {"Pearson_correlation", "M1_correlation_<t>",
                     "M2_correlation_<t>"}


# --------------------------------------------------------------------------
# what the panel shows for each feature
# --------------------------------------------------------------------------

def test_a_doc_carries_everything_the_panel_has_to_show():
    """Definition, unit, object types, channel, module — for every feature."""
    for doc in feature_docs():
        assert doc.title
        assert doc.family
        assert doc.computed_by
        assert doc.module != "" and doc.module is not None
        if doc.kind == "feature" and doc.key not in (
                "skeleton_length", "skeleton_branch_points"):
            assert doc.description, doc.key
            assert doc.object_types, doc.key
        assert doc.channel_scope in (CHANNEL_NONE, CHANNEL_SINGLE, CHANNEL_PAIR)


def test_every_geometric_feature_states_its_unit_as_a_condition():
    """A 3-D run measures a volume under the same name; the doc must say so."""
    doc = doc_for("area")
    assert "measurement_units" in doc.unit
    assert "px^2" in doc.unit and "um^3" in doc.unit


def test_concepts_only_reference_keys_that_exist():
    known = set(KNOWN_PROPERTIES) | set(META_COLUMNS)
    for name, concept in CONCEPTS.items():
        unknown = [k for k in concept.keys if k not in known]
        # The identity concept sweeps in the link columns too, which live in
        # their own private table; everything else must be a real key.
        if name != "identity":
            assert unknown == [], (name, unknown)
        assert concept.gloss
        assert concept.synonyms


def test_the_four_named_concepts_all_exist_and_are_populated():
    """The task names these four by hand; none may be empty."""
    for name in ("intensity", "texture", "shape", "distance"):
        assert name in CONCEPTS
        assert len(CONCEPTS[name].keys) >= 3
        assert search_features(name, concept=name)


def test_a_feature_knows_its_concepts_both_ways():
    assert "size" in concepts_for("area")
    assert "area" in CONCEPTS["size"].keys
