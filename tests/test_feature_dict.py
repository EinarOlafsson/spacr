"""Tests for :mod:`spacr.feature_dict`, the measurements.db data dictionary.

Every column name asserted on here was taken verbatim from an actual run of
``spacr.measure._morphological_measurements`` / ``_intensity_measurements``
followed by ``spacr.utils._check_integrity``; the f-string in measure.py that
assembles each one is quoted in a comment next to the assertion.

CPU only, offline, no fixtures beyond a temporary SQLite file.
"""

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from spacr.feature_dict import (
    FEATURE_FAMILIES,
    KNOWN_PROPERTIES,
    META_COLUMNS,
    OBJECT_TYPES,
    FeatureEntry,
    describe_columns,
    describe_database,
    export_dictionary,
    parse_column,
)


# --------------------------------------------------------------------------
# parse_column on real column names
# --------------------------------------------------------------------------

# (column, object_type, channel, family)
# Each column below is produced by the measure.py f-string quoted beside it.
REAL_COLUMNS = [
    # measure.py:225  df.columns = [f'{ls[i]}_{col}' for col in df.columns]
    # with morphological_props from measure.py:163-164
    ("cell_area", "cell", None, "morphology"),
    ("cell_area_filled", "cell", None, "morphology"),
    ("cell_area_bbox", "cell", None, "morphology"),
    ("nucleus_convex_area", "nucleus", None, "morphology"),
    ("nucleus_major_axis_length", "nucleus", None, "morphology"),
    ("pathogen_minor_axis_length", "pathogen", None, "morphology"),
    ("pathogen_eccentricity", "pathogen", None, "morphology"),
    ("cytoplasm_solidity", "cytoplasm", None, "morphology"),
    ("cytoplasm_extent", "cytoplasm", None, "morphology"),
    ("cell_perimeter", "cell", None, "morphology"),
    ("cell_euler_number", "cell", None, "morphology"),
    ("cell_equivalent_diameter_area", "cell", None, "morphology"),
    ("organelle_feret_diameter_max", "organelle", None, "morphology"),

    # measure.py:77  columns=[f'zernike_{i}' for i in range(feature_length)]
    # then prefixed by measure.py:225
    ("cell_zernike_0", "cell", None, "moment"),
    ("nucleus_zernike_12", "nucleus", None, "moment"),
    ("pathogen_zernike_24", "pathogen", None, "moment"),

    # measure.py:395  [f'{ls[j]}_channel_{i}_{col}' ...] over intensity_props
    # from measure.py:360
    ("cell_channel_0_mean_intensity", "cell", 0, "intensity"),
    ("cell_channel_1_max_intensity", "cell", 1, "intensity"),
    ("nucleus_channel_2_min_intensity", "nucleus", 2, "intensity"),
    ("cell_channel_0_centroid_weighted-0", "cell", 0, "moment"),
    ("cell_channel_0_centroid_weighted-1", "cell", 0, "moment"),
    ("cell_channel_1_centroid_weighted_local-0", "cell", 1, "moment"),
    ("cell_channel_1_centroid_weighted_local-1", "cell", 1, "moment"),

    # measure.py:520-532, prefixed by measure.py:395
    ("cell_channel_0_integrated_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_std_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_median_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_skew_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_kurtosis_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_mode_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_range_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_iqr_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_cv_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_gini_intensity", "cell", 0, "intensity"),
    ("cell_channel_0_frac_high90", "cell", 0, "intensity"),
    ("cell_channel_0_frac_low10", "cell", 0, "intensity"),
    ("cell_channel_0_entropy_intensity", "cell", 0, "intensity"),

    # measure.py:536  df[f'percentile_{p}'] for p in [5, 10, 25, 75, 85, 95]
    ("cell_channel_1_percentile_75", "cell", 1, "intensity"),
    ("cytoplasm_channel_2_percentile_5", "cytoplasm", 2, "intensity"),

    # measure.py:556  columns = [f'homogeneity_distance_{d}' for d in distances]
    ("cell_channel_0_homogeneity_distance_8", "cell", 0, "texture"),
    ("nucleus_channel_1_homogeneity_distance_32", "nucleus", 1, "texture"),

    # measure.py:385  columns=[f'periphery_{stat}' for stat in col_lables]
    # with col_lables from measure.py:361
    ("nucleus_channel_0_periphery_mean", "nucleus", 0, "intensity"),
    ("nucleus_channel_2_periphery_75_percentile", "nucleus", 2, "intensity"),
    ("pathogen_channel_1_periphery_50_percentile", "pathogen", 1, "intensity"),

    # measure.py:390  columns=[f'outside_{stat}' for stat in col_lables]
    ("nucleus_channel_0_outside_mean", "nucleus", 0, "intensity"),
    ("organelle_channel_1_outside_95_percentile", "organelle", 1, "intensity"),

    # measure.py:444
    #   f'{object_type}_rad_dist_channel_{channel_index}_bin_{i}'
    ("nucleus_rad_dist_channel_0_bin_0", "nucleus", 0, "intensity"),
    ("pathogen_rad_dist_channel_2_bin_5", "pathogen", 2, "intensity"),
    ("organelle_rad_dist_channel_1_bin_3", "organelle", 1, "intensity"),

    # measure.py:787-788
    #   f'cell_channel_{ch}_distance_to_nucleus' / '..._to_pathogen'
    ("cell_channel_0_distance_to_nucleus", "cell", 0, "morphology"),
    ("cell_channel_2_distance_to_pathogen", "cell", 2, "morphology"),

    # measure.py:393 then measure.py:395 -> the prefix appears twice
    ("cell_channel_0_cell_channel_0_blur", "cell", 0, "texture"),
    ("nucleus_channel_1_nucleus_channel_1_blur", "nucleus", 1, "texture"),

    # measure.py:429
    #   f'{ls[m]}_channel_{i}_channel_{j}_{col}' over measure.py:693-706
    ("cell_channel_0_channel_1_Pearson_correlation", "cell", 0, "correlation"),
    ("pathogen_channel_0_channel_2_M1_correlation_85", "pathogen", 0, "correlation"),
    ("cytoplasm_channel_1_channel_2_M2_correlation_15", "cytoplasm", 1, "correlation"),
]


@pytest.mark.parametrize("column,object_type,channel,family", REAL_COLUMNS)
def test_parse_column_real_names(column, object_type, channel, family):
    entry = parse_column(column)
    assert entry.column == column
    assert entry.object_type == object_type
    assert entry.channel == channel
    assert entry.family == family
    assert entry.description, f"{column} has no description"
    assert entry.computed_by and entry.computed_by != "unknown"
    assert entry.family in FEATURE_FAMILIES


def test_parse_column_returns_frozen_dataclass():
    entry = parse_column("cell_area")
    assert isinstance(entry, FeatureEntry)
    with pytest.raises(Exception):
        entry.column = "something else"


def test_morphology_columns_have_no_channel_and_pixel_units():
    entry = parse_column("cell_area")
    assert entry.channel is None
    assert entry.unit is not None and "px" in entry.unit
    ecc = parse_column("nucleus_eccentricity")
    assert ecc.channel is None
    assert "dimensionless" in ecc.unit


def test_orientation_free_units_are_not_asserted_as_pixels():
    """Dimensionless shape ratios must not claim a pixel unit."""
    for column in ("cell_solidity", "cell_extent", "cell_eccentricity"):
        assert "px" not in parse_column(column).unit


def test_zernike_notes_record_the_radius_argument_quirk():
    entry = parse_column("nucleus_zernike_12")
    assert "12" in entry.description
    # measure.py:68 calls zernike_moments(region.image, degree); mahotas'
    # signature is (im, radius, degree=8), so degree lands in radius.
    assert "radius" in entry.notes


def test_percentile_description_names_the_percentile():
    assert "75th" in parse_column("cell_channel_1_percentile_75").description
    assert "5th" in parse_column("cell_channel_1_percentile_5").description


def test_homogeneity_description_names_the_offset():
    entry = parse_column("cell_channel_0_homogeneity_distance_32")
    assert "32" in entry.description
    assert entry.family == "texture"


# --------------------------------------------------------------------------
# correlation columns name two channels
# --------------------------------------------------------------------------

def test_correlation_column_captures_both_channels():
    # measure.py:429  f'{ls[m]}_channel_{i}_channel_{j}_{col}'
    entry = parse_column("pathogen_channel_0_channel_2_M1_correlation_85")
    assert entry.object_type == "pathogen"
    assert entry.channel == 0
    assert entry.channel_2 == 2
    assert entry.family == "correlation"
    assert "85" in entry.description


def test_pearson_correlation_captures_both_channels():
    entry = parse_column("cell_channel_1_channel_2_Pearson_correlation")
    assert (entry.channel, entry.channel_2) == (1, 2)
    assert entry.unit is not None and "-1" in entry.unit


def test_single_channel_column_has_no_second_channel():
    entry = parse_column("cell_channel_1_mean_intensity")
    assert entry.channel == 1
    assert entry.channel_2 is None


def test_manders_notes_say_it_is_not_classical_manders():
    entry = parse_column("cell_channel_0_channel_1_M1_correlation_15")
    assert "Manders" in entry.notes


# --------------------------------------------------------------------------
# periphery / outside / radial variants
# --------------------------------------------------------------------------

def test_periphery_and_outside_are_distinguished():
    peri = parse_column("nucleus_channel_0_periphery_25_percentile")
    outs = parse_column("nucleus_channel_0_outside_25_percentile")
    assert peri.description != outs.description
    assert "rim" in peri.description
    assert "outward" in outs.description
    assert "5 px" in outs.description


def test_radial_bin_zero_warning_is_present():
    entry = parse_column("nucleus_rad_dist_channel_0_bin_0")
    assert entry.channel == 0
    # bin 0 collects every pixel where the distance map is 0, which includes
    # everything outside the parent cell (measure.py:658).
    assert "bin_0" in entry.notes


def test_blur_column_notes_the_duplicated_prefix():
    entry = parse_column("cell_channel_0_cell_channel_0_blur")
    assert entry.channel == 0
    assert "TWICE" in entry.notes


def test_mode_intensity_is_flagged_as_always_nan():
    entry = parse_column("cell_channel_0_mode_intensity")
    assert "NaN" in entry.notes


# --------------------------------------------------------------------------
# metadata columns
# --------------------------------------------------------------------------

# Written by spacr.utils._merge_and_save_to_database (utils.py:1741-1758),
# spacr.utils.filepaths_to_database (utils.py:830-855),
# spacr.io._save_settings_to_db (io.py:2133-2146) and
# spacr.utils._save_object_counts_to_database (utils.py:2249).
META_NAMES = [
    "object_label", "cell_id", "plateID", "rowID", "columnID", "fieldID",
    "timeID", "time_id", "prcf", "prcfo", "prcft", "prc", "file_name",
    "path_name", "png_path", "label_list", "label_list_morphology",
    "label_list_intensity", "setting_key", "setting_value", "count_type",
    "object_count", "nucleus_id", "pathogen_id", "cytoplasm_id",
]


@pytest.mark.parametrize("column", META_NAMES)
def test_metadata_columns_are_meta(column):
    entry = parse_column(column)
    assert entry.family == "meta", f"{column} was mis-parsed as {entry.family}"
    assert entry.description
    assert entry.computed_by and entry.computed_by != "unknown"


def test_cell_id_is_meta_not_a_cell_feature():
    """`cell_id` must not be read as object 'cell' with a stat named 'id'."""
    entry = parse_column("cell_id")
    assert entry.family == "meta"
    assert entry.object_type is None
    assert entry.channel is None


def test_child_parent_link_columns_are_meta_with_object_type():
    # measure.py:183/194 merge the cell<->child mapping in, then measure.py:225
    # prefixes it, producing nucleus_cell_id / nucleus_nucleus.
    link = parse_column("nucleus_cell_id")
    assert link.family == "meta"
    assert link.object_type == "nucleus"
    dup = parse_column("pathogen_pathogen")
    assert dup.family == "meta"
    assert dup.object_type == "pathogen"
    # measure.py:207 _map_child_to_parent(child_name='organelle', parent_name='cell')
    org = parse_column("organelle_cell")
    assert org.family == "meta"
    assert org.object_type == "organelle"


def test_pathogen_annotation_column_is_meta_not_an_object():
    """The bare `pathogen` annotation column is experiment metadata."""
    entry = parse_column("pathogen")
    assert entry.family == "meta"
    assert "annotation" in entry.description.lower()


# --------------------------------------------------------------------------
# organelle summaries
# --------------------------------------------------------------------------

def test_organelle_summary_columns():
    # measure.py:1046  [f'organelle_summary_{col}' if col != 'label' else col ...]
    entry = parse_column("organelle_summary_organelle_count")
    assert entry.family == "morphology"
    assert entry.object_type == "organelle"
    assert entry.description


def test_organelle_summary_per_channel_columns():
    # measure.py:325  f'organelle_ch{ch}_mean_intensity_per_{parent_name}'
    # then measure.py:1046 prefixes it with organelle_summary_
    entry = parse_column("organelle_summary_organelle_ch1_mean_intensity_per_cell")
    assert entry.channel == 1
    assert entry.object_type == "organelle"
    assert entry.object_type_2 == "cell"
    assert entry.family == "intensity"


# --------------------------------------------------------------------------
# unknown columns
# --------------------------------------------------------------------------

UNKNOWN_NAMES = [
    "totally_made_up_column",
    "cell_frobnicator_9000",
    "nucleus_channel_3_wibble",
    "",
    "___",
    "channel_0_orphan",
]


@pytest.mark.parametrize("column", UNKNOWN_NAMES)
def test_unknown_column_round_trips_without_crashing(column):
    entry = parse_column(column)
    assert isinstance(entry, FeatureEntry)
    assert entry.column == column
    assert entry.family == "unknown"
    assert entry.description is None
    assert entry.unit is None
    assert entry.computed_by == "unknown"
    assert entry.notes


def test_unknown_column_still_reports_structure_it_could_parse():
    entry = parse_column("nucleus_channel_3_wibble")
    assert entry.object_type == "nucleus"
    assert entry.channel == 3
    assert entry.family == "unknown"


def test_parse_column_accepts_non_string_without_crashing():
    entry = parse_column(17)
    assert entry.column == "17"
    assert entry.family == "unknown"


def test_unknown_organelle_summary_statistic():
    entry = parse_column("organelle_summary_organelle_wibble")
    assert entry.family == "unknown"
    assert entry.object_type == "organelle"
    assert entry.description is None
    assert "organelle_summary_" in entry.notes


def test_single_prefix_blur_is_still_recognised():
    """If the duplicated-prefix defect is ever fixed, the plain name parses."""
    entry = parse_column("cell_channel_2_blur")
    assert entry.family == "texture"
    assert entry.channel == 2
    assert "TWICE" not in (entry.notes or "")


def test_mismatched_double_blur_prefix_is_flagged():
    entry = parse_column("cell_channel_0_nucleus_channel_1_blur")
    assert entry.family == "texture"
    assert "disagree" in entry.notes


def test_correlation_label_column_is_meta():
    # measure.py:693  corr_data[i] = {f'label_correlation': i, ...}
    entry = parse_column("cell_channel_0_channel_1_label_correlation")
    assert entry.family == "meta"
    assert entry.channel == 0
    assert entry.channel_2 == 1


def test_periphery_region_label_is_meta():
    # measure.py:361 col_lables starts with 'region_label'
    entry = parse_column("nucleus_channel_0_periphery_region_label")
    assert entry.family == "meta"
    assert entry.object_type == "nucleus"


def test_pandas_merge_suffix_is_recognised():
    """spacr.io._read_and_join_tables merges with suffixes=('', '_<entity>')."""
    entry = parse_column("cell_channel_0_mean_intensity_nucleus")
    assert entry.family == "intensity"
    assert entry.object_type == "cell"
    assert entry.object_type_2 == "nucleus"
    assert "merge suffix" in entry.notes


def test_duplicate_column_index_suffix_is_recognised():
    """spacr.utils._check_integrity appends the positional index on clashes."""
    entry = parse_column("cell_channel_0_mean_intensity_57")
    assert entry.family == "intensity"
    assert entry.channel == 0
    assert "_check_integrity" in entry.notes
    assert "57" in entry.notes


def test_describe_columns_preserves_order_and_length():
    columns = ["cell_area", "totally_made_up", "cell_channel_0_mean_intensity"]
    entries = describe_columns(columns)
    assert [e.column for e in entries] == columns
    assert len(entries) == len(columns)


# --------------------------------------------------------------------------
# curated tables
# --------------------------------------------------------------------------

def test_every_known_property_has_a_computed_by():
    for key, info in KNOWN_PROPERTIES.items():
        assert isinstance(info.computed_by, str)
        assert info.computed_by.strip(), f"{key} has an empty computed_by"
        assert len(info.computed_by) > 5, f"{key} has a stub computed_by"


def test_every_meta_column_has_a_computed_by():
    for key, info in META_COLUMNS.items():
        assert info.computed_by.strip(), f"{key} has an empty computed_by"
        assert info.family == "meta", f"{key} is not in the meta family"


def test_every_known_property_family_is_declared():
    for key, info in KNOWN_PROPERTIES.items():
        assert info.family in FEATURE_FAMILIES, f"{key} has family {info.family}"


def test_known_properties_have_descriptions():
    for key, info in KNOWN_PROPERTIES.items():
        assert info.description, f"{key} has no description"


def test_feature_families_cover_the_documented_set():
    for family in ("morphology", "intensity", "texture", "correlation",
                   "moment", "meta", "unknown"):
        assert family in FEATURE_FAMILIES
        assert FEATURE_FAMILIES[family]


def test_object_types_match_measure_py():
    # measure.py:363  ls = ['cell', 'nucleus', 'pathogen', 'organelle', 'cytoplasm']
    assert set(OBJECT_TYPES) == {
        "cell", "nucleus", "pathogen", "organelle", "cytoplasm"}


# --------------------------------------------------------------------------
# describe_database
# --------------------------------------------------------------------------

CELL_TABLE_COLUMNS = [
    "object_label", "plateID", "rowID", "columnID", "fieldID", "prcf",
    "file_name", "path_name", "cell_area", "cell_solidity", "cell_zernike_3",
    "cell_channel_0_mean_intensity", "cell_channel_0_percentile_95",
    "cell_channel_0_homogeneity_distance_8",
    "cell_channel_0_cell_channel_0_blur",
    "cell_channel_0_channel_1_Pearson_correlation",
    "cell_channel_0_distance_to_nucleus",
    "label_list_morphology", "label_list_intensity",
    "a_column_nobody_documented",
]

NUCLEUS_TABLE_COLUMNS = [
    "object_label", "cell_id", "plateID", "prcf",
    "nucleus_area", "nucleus_zernike_0",
    "nucleus_channel_1_periphery_mean",
    "nucleus_channel_1_outside_25_percentile",
    "nucleus_rad_dist_channel_1_bin_2",
]


def _make_db(path):
    """Create a small measurements-like database and return its column map."""
    tables = {
        "cell": CELL_TABLE_COLUMNS,
        "nucleus": NUCLEUS_TABLE_COLUMNS,
        "settings": ["setting_key", "setting_value"],
    }
    with sqlite3.connect(path) as conn:
        for table, columns in tables.items():
            cols = ", ".join(f'"{c}" REAL' for c in columns)
            conn.execute(f'CREATE TABLE "{table}" ({cols})')
    return tables


@pytest.fixture()
def measurements_db(tmp_path):
    path = tmp_path / "measurements.db"
    tables = _make_db(path)
    return path, tables


def test_describe_database_returns_one_row_per_column(measurements_db):
    path, tables = measurements_db
    df = describe_database(path)
    assert len(df) == sum(len(c) for c in tables.values())
    for table, columns in tables.items():
        got = df.loc[df["table"] == table, "column"].tolist()
        assert got == columns, f"{table} lost or reordered columns"


def test_describe_database_drops_nothing_unknown(measurements_db):
    path, _ = measurements_db
    df = describe_database(path)
    unknown = df[df["family"] == "unknown"]
    assert unknown["column"].tolist() == ["a_column_nobody_documented"]
    assert unknown["description"].isna().all()


def test_describe_database_single_table(measurements_db):
    path, tables = measurements_db
    df = describe_database(path, table="nucleus")
    assert set(df["table"]) == {"nucleus"}
    assert df["column"].tolist() == tables["nucleus"]


def test_describe_database_channel_column_is_integer(measurements_db):
    path, _ = measurements_db
    df = describe_database(path)
    row = df[df["column"] == "cell_channel_0_channel_1_Pearson_correlation"]
    assert row["channel"].iloc[0] == 0
    assert row["channel_2"].iloc[0] == 1
    assert str(df["channel"].dtype) == "Int64"


def test_describe_database_metadata_not_mis_parsed(measurements_db):
    path, _ = measurements_db
    df = describe_database(path)
    meta = df[df["column"].isin(
        ["object_label", "plateID", "rowID", "columnID", "fieldID", "prcf",
         "file_name", "path_name", "cell_id", "setting_key", "setting_value"])]
    assert (meta["family"] == "meta").all()


def test_describe_database_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        describe_database(tmp_path / "nope.db")


def test_describe_database_missing_table(measurements_db):
    path, _ = measurements_db
    with pytest.raises(ValueError, match="not found"):
        describe_database(path, table="does_not_exist")


def test_describe_database_ignores_sqlite_internal_tables(tmp_path):
    path = tmp_path / "m.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)')
        conn.execute("INSERT INTO t DEFAULT VALUES")
    df = describe_database(path)
    assert set(df["table"]) == {"t"}


def test_describe_database_empty_database(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    df = describe_database(path)
    assert df.empty
    assert "column" in df.columns


# --------------------------------------------------------------------------
# export_dictionary
# --------------------------------------------------------------------------

def test_export_csv(measurements_db, tmp_path):
    path, tables = measurements_db
    out = export_dictionary(path, tmp_path / "dict.csv", fmt="csv")
    assert out.is_file()
    df = pd.read_csv(out)
    assert len(df) == sum(len(c) for c in tables.values())
    assert {"table", "column", "family", "description", "unit",
            "computed_by"} <= set(df.columns)


def test_export_json(measurements_db, tmp_path):
    path, tables = measurements_db
    out = export_dictionary(path, tmp_path / "dict.json", fmt="json")
    payload = json.loads(out.read_text())
    assert payload["n_columns"] == sum(len(c) for c in tables.values())
    assert payload["n_unrecognised"] == 1
    assert len(payload["columns"]) == payload["n_columns"]
    assert payload["families"]["morphology"]
    entry = next(c for c in payload["columns"] if c["column"] == "cell_area")
    assert entry["family"] == "morphology"
    assert entry["channel"] is None


def test_export_markdown(measurements_db, tmp_path):
    path, tables = measurements_db
    out = export_dictionary(path, tmp_path / "dict.md", fmt="md")
    text = out.read_text()
    n_columns = sum(len(c) for c in tables.values())
    assert "# spaCR feature dictionary" in text
    assert str(path) in text
    assert f"Columns described: {n_columns}" in text
    assert "Unrecognised columns: 1" in text
    # grouped by object, then family
    assert "## cell" in text
    assert "## nucleus" in text
    assert "### morphology" in text
    assert "### intensity" in text
    # every column appears exactly once, in a table row
    for columns in tables.values():
        for column in columns:
            assert f"`{column}`" in text, f"{column} missing from markdown"


def test_export_markdown_row_count(measurements_db, tmp_path):
    path, tables = measurements_db
    out = export_dictionary(path, tmp_path / "dict.md", fmt="md")
    rows = [ln for ln in out.read_text().splitlines() if ln.startswith("| `")]
    assert len(rows) == sum(len(c) for c in tables.values())


def test_export_creates_parent_directory(measurements_db, tmp_path):
    path, _ = measurements_db
    out = export_dictionary(path, tmp_path / "deep" / "nested" / "d.csv")
    assert out.is_file()


def test_export_rejects_bad_format(measurements_db, tmp_path):
    path, _ = measurements_db
    with pytest.raises(ValueError, match="fmt must be"):
        export_dictionary(path, tmp_path / "d.xlsx", fmt="xlsx")


def test_helpers_tolerate_odd_values():
    """The render/serialise helpers must not choke on non-scalar cells."""
    from spacr.feature_dict import _fill, _is_missing, _jsonable

    # a malformed curated template must be returned verbatim, not raise
    assert _fill("holds {missing_key}", {"p": "5"}) == "holds {missing_key}"
    assert _fill(None, {"p": "5"}) is None
    assert _fill("plain", {}) == "plain"

    # pd.isna raises on array-likes; _is_missing must swallow that
    assert _is_missing(None) is True
    assert _is_missing(float("nan")) is True
    assert _is_missing(pd.NA) is True
    assert _is_missing([1, 2, 3]) is False
    assert _is_missing("text") is False

    assert _jsonable(pd.NA) is None
    assert _jsonable("text") == "text"
    assert _jsonable(True) is True
    assert _jsonable(3) == 3
    assert _jsonable(2.5) == 2.5
    assert _jsonable(object()) is not None


def test_export_markdown_escapes_pipes(tmp_path):
    path = tmp_path / "m.db"
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE t ("weird|name" REAL)')
    out = export_dictionary(path, tmp_path / "d.md", fmt="md")
    text = out.read_text()
    assert "weird\\|name" in text


# --------------------------------------------------------------------------
# dependency weight
# --------------------------------------------------------------------------

def test_import_pulls_no_heavy_dependencies():
    """feature_dict must be usable without torch / cellpose / skimage.

    Measured as a diff against the interpreter's state just before the import,
    so a sitecustomize or a conftest that pre-imports something heavy cannot
    mask (or fake) the result.
    """
    banned = ("torch", "cellpose", "skimage", "cv2", "mahotas", "matplotlib",
              "tensorflow", "scipy", "sklearn", "PIL")
    code = (
        "import sys\n"
        f"banned = {banned!r}\n"
        "before = set(sys.modules)\n"
        "import spacr.feature_dict as fd\n"
        "fd.parse_column('cell_area')\n"
        "fd.parse_column('nope')\n"
        "added = set(sys.modules) - before\n"
        "print(','.join(sorted(m for m in banned if m in added)))\n"
    )
    env = dict(os.environ)
    # Only the repo itself on the path: the shared scratchpad on PYTHONPATH may
    # hold a sitecustomize that pre-imports torch, which is not our doing.
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, timeout=300, env=env)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "", (
        f"spacr.feature_dict imported heavy modules: {proc.stdout.strip()}")


def test_module_source_declares_no_heavy_imports():
    """A cheap guard that survives even if the subprocess test is skipped."""
    source = (REPO_ROOT / "spacr" / "feature_dict.py").read_text()
    for banned in ("import torch", "import cellpose", "import skimage",
                   "from skimage", "import cv2", "import numpy"):
        assert banned not in source, f"feature_dict.py contains '{banned}'"
