"""The feature dictionary against a 3-D measure run.

``spacr.measure`` now measures a ``(Z, Y, X)`` mask in 3-D. That added five
per-row provenance columns, two explicitly-named volume columns and six
axis-named centroids, and — the part that matters more — it made the unit of
``<object>_area`` conditional: px^2 in 2-D, a cubic-xy-pixel volume in a 3-D
run configured with ``anisotropy``, and um^3 in one configured with
``voxel_size_z_um`` + ``voxel_size_xy_um``. The column name is identical in all
three cases, so the dictionary cannot state one unit any more.

The column names asserted on here are not copied from a brief: the
round-trip tests RUN ``spacr.measure`` and feed whatever it emits through
:func:`spacr.feature_dict.parse_column`, so a name that changes in measure.py
fails here rather than silently becoming ``family='unknown'``.

CPU only, offline, deterministic, no network.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import feature_dict as fd
from spacr.feature_dict import (
    ConditionalUnit,
    MEASUREMENT_STAMP_COLUMNS,
    MEASUREMENT_UNITS,
    UNITS_PX,
    UNITS_PX_XY,
    UNITS_UM,
    describe_database,
    export_dictionary,
    parse_column,
)


# --------------------------------------------------------------------------
# the vocabulary is the producer's, not this module's
# --------------------------------------------------------------------------

def test_units_vocabulary_matches_measure():
    """feature_dict duplicates measure's unit literals; they must agree.

    They are duplicated on purpose — feature_dict must stay importable without
    numpy or scikit-image — so something has to pin them together.
    """
    from spacr import measure as M

    assert (UNITS_PX, UNITS_PX_XY, UNITS_UM) == (
        M.UNITS_PX, M.UNITS_PX_XY, M.UNITS_UM)
    assert MEASUREMENT_UNITS == (M.UNITS_PX, M.UNITS_PX_XY, M.UNITS_UM)


def test_stamp_columns_match_measure_and_utils():
    from spacr import measure as M
    from spacr import utils as U

    assert MEASUREMENT_STAMP_COLUMNS == M.MEASUREMENT_STAMP_COLUMNS
    assert MEASUREMENT_STAMP_COLUMNS == U.MEASUREMENT_STAMP_COLUMNS


def test_every_stamp_column_is_documented():
    for column in MEASUREMENT_STAMP_COLUMNS:
        entry = parse_column(column)
        assert entry.family == "meta", f"{column} -> {entry.family}"
        assert entry.description
        assert entry.computed_by and entry.computed_by != "unknown"
        assert entry.notes


def test_legacy_units_agree_with_utils():
    """An unstamped row is 2-D/px here for the same reason it is in utils."""
    from spacr import utils as U

    assert fd._LEGACY_UNITS == U._LEGACY_STAMP[1]


# --------------------------------------------------------------------------
# the statement that became false
# --------------------------------------------------------------------------

def test_no_curated_entry_claims_spacr_never_calibrates():
    """The old blanket claim must not survive in anything the dictionary says.

    It may still be quoted in a docstring as the history of why the units are
    conditional now; what it may not do is come back out of parse_column.
    """
    claim = "never applies a physical pixel size"
    tables = (fd.KNOWN_PROPERTIES, fd.META_COLUMNS, fd._LINK_COLUMNS)
    for table in tables:
        for key, info in table.items():
            texts = [info.description, info.notes]
            if isinstance(info.unit, ConditionalUnit):
                texts += [info.unit.resolve(u) for u in MEASUREMENT_UNITS]
                texts.append(info.unit.resolve())
            else:
                texts.append(info.unit)
            for text in texts:
                assert claim not in (text or ""), key

    for column in ("cell_area", "cell_major_axis_length",
                   "cell_channel_0_distance_to_nucleus"):
        for units in (None,) + MEASUREMENT_UNITS:
            entry = parse_column(column, units)
            assert claim not in entry.unit
            assert claim not in (entry.notes or "")


def test_um_per_pixel_is_not_implied_to_reach_a_measurement():
    """um_per_pixel only sizes a scale bar; the dictionary must not blur that."""
    entry = parse_column("voxel_size_xy_um")
    assert "um_per_pixel" in entry.notes
    assert "scale bar" in entry.notes
    # and it is not documented as a measurement column of its own
    assert parse_column("um_per_pixel").family == "unknown"


# --------------------------------------------------------------------------
# the conditional unit
# --------------------------------------------------------------------------

CONDITIONAL_COLUMNS = [
    "cell_area", "cell_area_filled", "cell_area_bbox", "cell_convex_area",
    "cell_major_axis_length", "cell_minor_axis_length",
    "cell_equivalent_diameter_area", "cell_feret_diameter_max",
    "cell_channel_0_distance_to_nucleus",
    "organelle_summary_organelle_total_area",
    "organelle_summary_organelle_mean_major_axis",
]


@pytest.mark.parametrize("column", CONDITIONAL_COLUMNS)
@pytest.mark.parametrize("units", list(MEASUREMENT_UNITS))
def test_conditional_unit_resolves_for_every_stamp(column, units):
    entry = parse_column(column, units)
    assert entry.measurement_units == units
    assert entry.unit
    # a resolved unit is a statement, not a menu
    assert "depends on the row's measurement_units" not in entry.unit
    assert "when measurement_units=" not in entry.unit


def test_area_unit_is_px2_in_2d_and_a_volume_in_3d():
    """2-D is never ambiguous; 3-D is where the unit changes."""
    assert parse_column("cell_area", UNITS_PX).unit.startswith("px^2")
    px_xy = parse_column("cell_area", UNITS_PX_XY).unit
    um = parse_column("cell_area", UNITS_UM).unit
    assert "cubic xy pixels" in px_xy and "VOLUME" in px_xy
    assert um.startswith("um^3") and "VOLUME" in um


def test_length_unit_follows_the_same_three_way_split():
    assert parse_column("cell_major_axis_length", UNITS_PX).unit.startswith("px")
    assert "xy pixels" in parse_column(
        "cell_major_axis_length", UNITS_PX_XY).unit
    assert parse_column("cell_major_axis_length", UNITS_UM).unit.startswith("um")


def test_unknown_stamp_states_the_condition_instead_of_guessing():
    unresolved = parse_column("cell_area").unit
    assert "depends on the row's measurement_units" in unresolved
    for units in MEASUREMENT_UNITS:
        assert f"measurement_units='{units}'" in unresolved
    assert parse_column("cell_area").measurement_units is None
    # a value from outside the vocabulary is not silently accepted either
    assert parse_column("cell_area", "furlongs").unit == unresolved


def test_2d_only_columns_have_an_unconditional_unit():
    """resolve_measurement_spacing returns None in 2-D so these never move."""
    for column in ("cell_perimeter", "cell_channel_0_centroid_weighted-0",
                   "cell_channel_0_centroid_weighted_local-1"):
        for units in (None,) + MEASUREMENT_UNITS:
            entry = parse_column(column, units)
            assert entry.unit.startswith("px (pixels; 2-D only")
            assert entry.measurement_units is None


def test_3d_only_columns_say_so_under_the_2d_stamp():
    entry = parse_column("cell_channel_0_centroid_weighted_z", UNITS_PX)
    assert entry.unit == "not written when measurement_units='px'"


def test_duplicate_suffixed_columns_resolve_their_unit_too():
    """A de-duplicated copy is the same measurement, so it gets the same unit.

    Both spellings _check_integrity has used: the current '__dup<n>' and the
    legacy positional '_<n>'.
    """
    for column in ("cell_area__dup1", "cell_area_1"):
        entry = parse_column(column, UNITS_UM)
        assert entry.family == "morphology"
        assert entry.unit.startswith("um^3")
        assert entry.measurement_units == UNITS_UM
        assert "copy of 'area'" in entry.notes


def test_dimensionless_columns_never_gain_a_pixel_unit():
    for column in ("cell_solidity", "cell_extent", "cell_eccentricity",
                   "cell_channel_0_channel_1_Pearson_correlation"):
        for units in (None,) + MEASUREMENT_UNITS:
            assert "px" not in parse_column(column, units).unit


def test_integrated_intensity_is_not_converted_by_the_voxel_size():
    """The sum has no spacing factor, so um^3 would be a lie."""
    px = parse_column("cell_channel_0_integrated_intensity", UNITS_PX).unit
    um = parse_column("cell_channel_0_integrated_intensity", UNITS_UM).unit
    assert "px^2" in px
    assert "voxel" in um and "um^3" in um and "NOT" in um


def test_conditional_unit_helper_semantics():
    unit = ConditionalUnit(px="a", px_xy=None, um="c")
    assert unit.resolve(UNITS_PX) == "a"
    assert unit.resolve(UNITS_UM) == "c"
    assert unit.resolve(UNITS_PX_XY) == "not written when measurement_units='px_xy'"
    text = unit.resolve()
    assert text == unit.conditional_text()
    assert "a when measurement_units='px'" in text
    assert "not written when measurement_units='px_xy'" in text


def test_every_conditional_unit_defines_every_mode_it_claims():
    """No curated entry may resolve to an empty or duplicated-looking unit."""
    for key, info in fd.KNOWN_PROPERTIES.items():
        if not isinstance(info.unit, ConditionalUnit):
            continue
        for units in MEASUREMENT_UNITS:
            resolved = info.unit.resolve(units)
            assert resolved and resolved.strip(), key


# --------------------------------------------------------------------------
# the new columns
# --------------------------------------------------------------------------

NEW_3D_COLUMNS = [
    ("cell_volume_voxels", "cell", None, "morphology"),
    ("nucleus_volume_voxels", "nucleus", None, "morphology"),
    ("cell_volume_um3", "cell", None, "morphology"),
    ("pathogen_volume_um3", "pathogen", None, "morphology"),
    ("cell_channel_0_centroid_weighted_z", "cell", 0, "moment"),
    ("cell_channel_0_centroid_weighted_y", "cell", 0, "moment"),
    ("cell_channel_1_centroid_weighted_x", "cell", 1, "moment"),
    ("nucleus_channel_2_centroid_weighted_local_z", "nucleus", 2, "moment"),
    ("nucleus_channel_2_centroid_weighted_local_y", "nucleus", 2, "moment"),
    ("nucleus_channel_2_centroid_weighted_local_x", "nucleus", 2, "moment"),
]


@pytest.mark.parametrize("column,object_type,channel,family", NEW_3D_COLUMNS)
def test_new_3d_columns_are_described(column, object_type, channel, family):
    entry = parse_column(column, UNITS_UM)
    assert entry.family == family
    assert entry.object_type == object_type
    assert entry.channel == channel
    assert entry.description
    assert entry.unit
    assert entry.computed_by and entry.computed_by != "unknown"


def test_volume_columns_name_their_own_unit():
    voxels = parse_column("cell_volume_voxels", UNITS_UM)
    um3 = parse_column("cell_volume_um3", UNITS_UM)
    # named units, so they do not move with the stamp
    assert voxels.unit == parse_column("cell_volume_voxels", UNITS_PX_XY).unit
    assert "voxel" in voxels.unit
    assert um3.unit.startswith("um^3")
    assert "volume_stats" in voxels.notes and "volume_stats" in um3.notes


def test_centroid_axes_are_distinguished_from_the_2d_spelling():
    z = parse_column("cell_channel_0_centroid_weighted_z", UNITS_UM)
    y = parse_column("cell_channel_0_centroid_weighted_y", UNITS_UM)
    x = parse_column("cell_channel_0_centroid_weighted_x", UNITS_UM)
    assert "z" in z.description and "plane" in z.description.lower()
    assert "y" in y.description and "Row" in y.description
    assert "x" in x.description and "Column" in x.description
    # the 2-D names still describe the 2-D axes, and say they are 2-D
    two_d = parse_column("cell_channel_0_centroid_weighted-0")
    assert "Row (y)" in two_d.description
    assert "2-D" in two_d.notes


# --------------------------------------------------------------------------
# round-trip against the real producer
# --------------------------------------------------------------------------

def _settings(**kw):
    s = {
        "cell_mask_dim": 4,
        "nucleus_mask_dim": 5,
        "pathogen_mask_dim": 6,
        "organelle_mask_dim": None,
        "cytoplasm": False,
        "radial_dist": True,
        "calculate_correlation": True,
        "manders_thresholds": [15, 85],
        "homogeneity": True,
        "homogeneity_distances": [4],
        "distance_gaussian_sigma": 0,
        "strict_errors": False,
    }
    s.update(kw)
    return s


def _masks_2d(shape=(24, 24)):
    cell = np.zeros(shape, np.uint16)
    nucleus = np.zeros(shape, np.uint16)
    pathogen = np.zeros(shape, np.uint16)
    cell[3:13, 3:13] = 1
    cell[15:21, 15:21] = 2
    nucleus[6:10, 6:10] = 1
    nucleus[17:19, 17:19] = 2
    pathogen[4:6, 4:6] = 1
    return cell, nucleus, pathogen, np.zeros(shape, np.uint16)


def _masks_3d(shape=(4, 24, 24)):
    c2, n2, p2, _ = _masks_2d(shape[1:])
    cell = np.zeros(shape, np.uint16)
    nucleus = np.zeros(shape, np.uint16)
    pathogen = np.zeros(shape, np.uint16)
    for z in range(1, 3):
        cell[z] = c2
        nucleus[z] = n2
        pathogen[z] = p2
    return cell, nucleus, pathogen, np.zeros(shape, np.uint16)


def _measured_columns(ndim, settings):
    """Every column one field of a real measure pass would write, per table.

    Runs the actual emitters and reproduces the merge and the stamp that
    ``spacr.utils._merge_and_save_to_database`` performs, so the names come
    from measure.py rather than from this file.
    """
    from spacr import measure as M
    from spacr.utils import _check_integrity

    if ndim == 3:
        cell, nucleus, pathogen, zero = _masks_3d()
        channels = np.random.RandomState(0).rand(4, 24, 24, 2)
    else:
        cell, nucleus, pathogen, zero = _masks_2d()
        channels = np.random.RandomState(0).rand(24, 24, 2)

    morph = M._morphological_measurements(
        cell, nucleus, pathogen, zero, zero, settings)
    inten = M._intensity_measurements(
        cell, nucleus, pathogen, zero, zero, channels, settings,
        sizes=[1, 2], periphery=True, outside=True)

    out = {}
    for name, morph_df, inten_df in zip(
            ("cell", "nucleus", "pathogen"), morph[:3], inten[:3]):
        morph_df = _check_integrity(morph_df)
        inten_df = _check_integrity(inten_df)
        if len(morph_df) == 0:
            continue
        merged = pd.merge(morph_df, inten_df, on="object_label", how="outer")
        merged = merged.rename(columns={
            "label_list_x": "label_list_morphology",
            "label_list_y": "label_list_intensity"})
        columns = list(merged.columns) + list(MEASUREMENT_STAMP_COLUMNS)
        # _merge_and_save_to_database also stamps the well keys onto every row.
        columns += ["file_name", "path_name", "plateID", "rowID", "columnID",
                    "fieldID", "prcf"]
        out[name] = columns

    # the organelle roll-up, prefixed exactly as measure_crop prefixes it
    spacing = M.resolve_measurement_spacing(settings, ndim)[0]
    summary = M._summarize_organelles_per_parent(
        nucleus, cell, channels, parent_name="cell", spacing=spacing)
    out["cell_organelle_summary"] = [
        f"organelle_summary_{c}" if c != "label" else "object_label"
        for c in summary.columns]
    return out


@pytest.mark.parametrize("units,extra", [
    (UNITS_UM, {"voxel_size_z_um": 0.5, "voxel_size_xy_um": 0.1}),
    (UNITS_PX_XY, {"anisotropy": 5.0}),
])
def test_3d_run_columns_all_parse(units, extra):
    """Nothing a 3-D measure pass writes may come back as 'unknown'."""
    tables = _measured_columns(3, _settings(**extra))
    assert tables, "the producer wrote nothing"
    unknown = {}
    for table, columns in tables.items():
        for column in columns:
            entry = parse_column(column, units)
            if entry.family == "unknown":
                unknown.setdefault(table, []).append(column)
            else:
                assert entry.description, column
                assert entry.computed_by != "unknown", column
    assert not unknown, f"undocumented columns from a 3-D run: {unknown}"


def test_3d_run_writes_the_columns_this_dictionary_gained():
    """Guard the round-trip above against passing vacuously."""
    tables = _measured_columns(
        3, _settings(voxel_size_z_um=0.5, voxel_size_xy_um=0.1))
    cell = set(tables["cell"])
    assert {"cell_volume_voxels", "cell_volume_um3"} <= cell
    assert "cell_channel_0_centroid_weighted_z" in cell
    assert "cell_channel_0_centroid_weighted_local_x" in cell
    assert set(MEASUREMENT_STAMP_COLUMNS) <= cell
    # and the 2-D-only names really are absent
    assert "cell_channel_0_centroid_weighted-0" not in cell
    assert "cell_perimeter" not in cell
    assert not [c for c in cell if "zernike" in c]


def test_2d_run_columns_all_parse():
    """The 2-D path is unchanged, and unambiguous: px throughout."""
    tables = _measured_columns(2, _settings())
    unknown = {}
    for table, columns in tables.items():
        for column in columns:
            if parse_column(column, UNITS_PX).family == "unknown":
                unknown.setdefault(table, []).append(column)
    assert not unknown, f"undocumented columns from a 2-D run: {unknown}"
    cell = set(tables["cell"])
    assert "cell_channel_0_centroid_weighted-0" in cell
    assert "cell_perimeter" in cell
    assert not [c for c in cell if "volume" in c]


def test_anisotropy_only_run_writes_no_um3_column():
    tables = _measured_columns(3, _settings(anisotropy=5.0))
    cell = set(tables["cell"])
    assert "cell_volume_voxels" in cell
    assert "cell_volume_um3" not in cell


# --------------------------------------------------------------------------
# describe_database reads the units out of the database it documents
# --------------------------------------------------------------------------

OBJECT_COLUMNS = [
    "object_label", "plateID", "prcf", "file_name",
    "cell_area", "cell_major_axis_length", "cell_volume_voxels",
    "cell_volume_um3", "cell_channel_0_centroid_weighted_z",
    "cell_channel_0_mean_intensity",
]


def _write_db(path, units, rows=1, with_stamp=True):
    columns = list(OBJECT_COLUMNS)
    if with_stamp:
        columns += list(MEASUREMENT_STAMP_COLUMNS)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE "cell" ('
                     + ", ".join(f'"{c}" REAL' for c in columns) + ")")
        for i in range(rows):
            values = [0.0] * len(columns)
            if with_stamp:
                stamp = list(units)[i] if isinstance(units, list) else units
                values[columns.index("measurement_units")] = stamp
                values[columns.index("measurement_ndim")] = (
                    2 if stamp == UNITS_PX else 3)
            conn.execute(f'INSERT INTO "cell" VALUES '
                         f'({", ".join("?" * len(columns))})', values)
    return path


def _unit_of(df, column):
    return df.loc[df["column"] == column, "unit"].iloc[0]


@pytest.mark.parametrize("units,expect", [
    (UNITS_PX, "px^2"),
    (UNITS_PX_XY, "cubic xy pixels"),
    (UNITS_UM, "um^3"),
])
def test_describe_database_resolves_units_from_the_rows(tmp_path, units, expect):
    db = _write_db(tmp_path / "m.db", units)
    df = describe_database(db)
    assert _unit_of(df, "cell_area").startswith(expect)
    row = df[df["column"] == "cell_area"].iloc[0]
    assert row["measurement_units"] == units
    # a column whose unit does not depend on the stamp is not claimed to
    assert df[df["column"] == "cell_channel_0_mean_intensity"].iloc[0][
        "measurement_units"] is None


def test_describe_database_treats_an_unstamped_table_as_2d_pixels(tmp_path):
    """Not a guess: a pre-3-D release could not write anything else."""
    db = _write_db(tmp_path / "legacy.db", UNITS_PX, with_stamp=False)
    df = describe_database(db)
    assert _unit_of(df, "cell_area").startswith("px^2")
    assert df[df["column"] == "cell_area"].iloc[0]["measurement_units"] == UNITS_PX


def test_describe_database_leaves_an_empty_table_conditional(tmp_path):
    db = _write_db(tmp_path / "empty.db", UNITS_UM, rows=0)
    df = describe_database(db)
    unit = _unit_of(df, "cell_area")
    assert "depends on the row's measurement_units" in unit
    assert df[df["column"] == "cell_area"].iloc[0]["measurement_units"] is None


def test_describe_database_refuses_to_pick_one_of_mixed_units(tmp_path):
    db = _write_db(tmp_path / "mixed.db", [UNITS_PX, UNITS_UM], rows=2)
    df = describe_database(db)
    assert "depends on the row's measurement_units" in _unit_of(df, "cell_area")


def test_describe_database_reads_a_null_stamp_as_legacy(tmp_path):
    db = tmp_path / "null.db"
    columns = OBJECT_COLUMNS + list(MEASUREMENT_STAMP_COLUMNS)
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "cell" ('
                     + ", ".join(f'"{c}" REAL' for c in columns) + ")")
        conn.execute(f'INSERT INTO "cell" VALUES '
                     f'({", ".join("NULL" for _ in columns)})')
    df = describe_database(db)
    assert _unit_of(df, "cell_area").startswith("px^2")


def test_describe_database_ignores_an_unrecognised_stamp_value(tmp_path):
    db = _write_db(tmp_path / "weird.db", "cubits")
    df = describe_database(db)
    assert "depends on the row's measurement_units" in _unit_of(df, "cell_area")


def test_describe_database_units_can_be_forced(tmp_path):
    db = _write_db(tmp_path / "m.db", UNITS_PX)
    df = describe_database(db, measurement_units=UNITS_UM)
    assert _unit_of(df, "cell_area").startswith("um^3")


def test_describe_database_frame_carries_the_units_column(tmp_path):
    db = _write_db(tmp_path / "m.db", UNITS_UM)
    df = describe_database(db)
    assert list(df.columns).index("measurement_units") == \
        list(df.columns).index("unit") + 1


def test_table_measurement_units_survives_an_unreadable_table(tmp_path):
    """A view whose backing table is gone must not take the export down."""
    db = tmp_path / "view.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "cell" ("cell_area" REAL, '
                     '"measurement_units" TEXT)')
        conn.execute('CREATE VIEW "v" AS SELECT * FROM "cell"')
        conn.execute('DROP TABLE "cell"')
    units, why = fd._table_measurement_units(db, "v", ["measurement_units"])
    assert units is None
    assert "unreadable" in why


def test_quoting_survives_a_hostile_table_name(tmp_path):
    db = tmp_path / "quoted.db"
    with sqlite3.connect(db) as conn:
        conn.execute('CREATE TABLE "we""ird" ("cell_area" REAL)')
        conn.execute('INSERT INTO "we""ird" VALUES (1.0)')
    df = describe_database(db)
    assert _unit_of(df, "cell_area").startswith("px^2")


# --------------------------------------------------------------------------
# the generated dictionary
# --------------------------------------------------------------------------

def test_export_csv_carries_the_new_columns_and_their_units(tmp_path):
    db = _write_db(tmp_path / "m.db", UNITS_UM)
    out = export_dictionary(db, tmp_path / "d.csv", fmt="csv")
    frame = pd.read_csv(out)
    assert "measurement_units" in frame.columns
    by_column = dict(zip(frame["column"], frame["unit"]))
    assert by_column["cell_volume_um3"].startswith("um^3")
    assert "voxel" in by_column["cell_volume_voxels"]
    assert by_column["cell_area"].startswith("um^3")
    assert "um" in by_column["cell_channel_0_centroid_weighted_z"]
    for stamp in MEASUREMENT_STAMP_COLUMNS:
        assert stamp in by_column


def test_export_markdown_states_the_unit_basis(tmp_path):
    db = _write_db(tmp_path / "m.db", UNITS_UM)
    text = export_dictionary(db, tmp_path / "d.md", fmt="md").read_text()
    assert "## Units" in text
    assert "measurement_units = 'um'" in text
    assert "`cell_volume_um3`" in text
    assert "`measurement_units`" in text
    # the false blanket claim is gone, and um_per_pixel is disambiguated
    assert "all geometric features are in pixels" not in text
    assert "um_per_pixel" in text and "scale bar" in text


def test_export_markdown_says_when_the_basis_is_unknown(tmp_path):
    db = _write_db(tmp_path / "mixed.db", [UNITS_PX, UNITS_UM], rows=2)
    text = export_dictionary(db, tmp_path / "d.md", fmt="md").read_text()
    assert "MIXED measurement_units" in text
    assert "Not pinned to one value" in text


def test_export_markdown_notes_a_legacy_table(tmp_path):
    db = _write_db(tmp_path / "legacy.db", UNITS_PX, with_stamp=False)
    text = export_dictionary(db, tmp_path / "d.md", fmt="md").read_text()
    assert "no measurement_units column" in text
    assert "2-D pixel measurement" in text


def test_export_json_carries_the_per_table_units(tmp_path):
    db = _write_db(tmp_path / "m.db", UNITS_PX_XY)
    out = export_dictionary(db, tmp_path / "d.json", fmt="json")
    payload = json.loads(out.read_text())
    assert payload["measurement_units"]["cell"]["measurement_units"] == UNITS_PX_XY
    assert "VOLUME" in payload["measurement_units"]["cell"]["geometric_columns"]
    entry = next(c for c in payload["columns"] if c["column"] == "cell_area")
    assert entry["measurement_units"] == UNITS_PX_XY
    assert "cubic xy pixels" in entry["unit"]


def test_export_json_on_an_empty_database(tmp_path):
    db = tmp_path / "nothing.db"
    sqlite3.connect(db).close()
    payload = json.loads(
        export_dictionary(db, tmp_path / "d.json", fmt="json").read_text())
    assert payload["n_columns"] == 0
    assert payload["measurement_units"] == {}


def test_export_markdown_on_an_empty_database(tmp_path):
    db = tmp_path / "nothing.db"
    sqlite3.connect(db).close()
    text = export_dictionary(db, tmp_path / "d.md", fmt="md").read_text()
    assert "## Units" in text
    assert "Columns described: 0" in text
