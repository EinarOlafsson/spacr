"""
Foreign-data importer — somebody else's images, masks and measurements
turned into a spaCR project.

Everything here runs against synthetic third-party data built in a temp
folder: TIFF images with their own naming, label images with a ``_mask``
suffix, and a CellProfiler-shaped ``results.csv`` whose column names mean
whatever their author meant by them.

The properties pinned here are the ones that decide whether the module
produces a real project or a plausible-looking wrong one:

* a **full import** lands merged arrays, a ``measurements.db`` carrying
  spaCR's ``plateID``/``rowID``/``columnID``/``fieldID``/``object_label``,
  and their columns beside them;
* an **unmapped column is reported by name** and is imported anyway —
  silently dropping a column the user cared about is the failure mode;
* a **colliding name is refused or renamed**, never written over spaCR's;
* a **unit conversion** happens when the pixel size is known and the
  column is flagged uncalibrated when it is not — an unknown scale never
  becomes 1.0;
* **rows that match no mask object are counted**, not inner-joined away;
* an image with no mask and a mask with no image are reported **per file**;
* the mapping **round-trips** through save/load, and the import applies
  exactly what was saved;
* a **re-run** replaces rather than duplicates;
* the image half really is :mod:`spacr.convert` — asserted by spying on
  it, not assumed.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest
import tifffile

from spacr import convert as cv
from spacr import crops as cropping
from spacr import feature_dict as fdict
from spacr import foreign as fg
from spacr.errors import ConfigurationError


# ---------------------------------------------------------------------------
# Synthetic third-party data
# ---------------------------------------------------------------------------

SIZE = 24


def _label_image():
    """Two objects, labels 1 and 2, at fixed positions."""
    mask = np.zeros((SIZE, SIZE), np.uint16)
    mask[2:8, 2:8] = 1        # 36 px
    mask[12:20, 12:20] = 2    # 64 px
    return mask


def _write_image(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, np.full((SIZE, SIZE), value, np.uint16))
    return path


def _write_mask(path, mask=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, _label_image() if mask is None else mask)
    return path


class Theirs:
    """The folder tree and table a collaborator actually sends."""

    def __init__(self, root):
        self.root = str(root)
        self.images = os.path.join(self.root, "their_images")
        self.cell_masks = os.path.join(self.root, "their_cell_masks")
        self.nucleus_masks = os.path.join(self.root, "their_nucleus_masks")
        self.table = os.path.join(self.root, "results.csv")

    def masks(self, *types):
        return {t: getattr(self, f"{t}_masks") for t in (types or ("cell",))}


@pytest.fixture
def theirs(tmp_path):
    """Two fields, two channels, cell masks, and a measurement table.

    ``AreaShape_Area`` is in px² and needs nothing; ``AreaShape_Area_um2``
    is in µm² and needs a pixel size; ``Metadata_Treatment`` is a string;
    ``cell_area`` is deliberately named exactly like one of spaCR's.
    """
    data = Theirs(tmp_path)
    rows = []
    for field in (1, 2):
        for channel in (1, 2):
            _write_image(os.path.join(data.images,
                                      f"fov{field:02d}_C{channel}.tif"),
                         field * 10 + channel)
        _write_mask(os.path.join(data.cell_masks,
                                 f"fov{field:02d}_cell_mask.tif"))
        _write_mask(os.path.join(data.nucleus_masks,
                                 f"fov{field:02d}_nucleus_mask.tif"))
        for label, area in ((1, 36.0), (2, 64.0)):
            rows.append({
                "ImageNumber": f"fov{field:02d}_C1.tif",
                "ObjectNumber": label,
                "AreaShape_Area": area,
                "AreaShape_Area_um2": area * 0.25,
                "Metadata_Treatment": "wt" if field == 1 else "ko",
                "cell_area": 999.0,
            })
    pd.DataFrame(rows).to_csv(data.table, index=False)
    return data


def _plan(theirs, **kwargs):
    kwargs.setdefault("um_per_px", 0.5)
    return fg.plan_import(theirs.images, theirs.masks(), theirs.table, **kwargs)


def _tables(db_path):
    connection = sqlite3.connect(db_path)
    try:
        return sorted(str(r[0]) for r in connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')"))
    finally:
        connection.close()


def _read(db_path, table):
    connection = sqlite3.connect(db_path)
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# 1. A full import
# ---------------------------------------------------------------------------

def test_a_full_import_produces_a_working_spacr_project(theirs, tmp_path):
    """Merged arrays, a spaCR-shaped database, and their columns in it."""
    plan = _plan(theirs)
    assert plan.ok, fg.format_plan(plan)
    result = fg.run_import(plan, str(tmp_path / "imported"))

    # -- the merged arrays are spaCR's layout ---------------------------
    assert result.n_fields == 2
    for path in result.merged:
        array = np.load(path)
        # four intensity planes? no: two channels + one cell mask plane.
        assert array.shape == (SIZE, SIZE, 3)
        assert set(np.unique(array[:, :, 2])) == {0, 1, 2}
    assert plan.mask_dims == {"cell": 2}
    # …and the plane order is the one spacr.crops documents.
    assert cropping.MASK_PLANE_ORDER[0] == "cell"

    # -- the Yokogawa images and the map back --------------------------
    names = sorted(os.listdir(os.path.join(result.dst, fg.IMAGES_DIRNAME)))
    assert "plate1_A01_T0001F001L01A01Z01C01.tif" in names
    assert cv.MAP_FILENAME in names

    # -- the database ---------------------------------------------------
    assert os.path.isfile(result.db_path)
    tables = _tables(result.db_path)
    for name in ("cell", "foreign_cell", cv.CONVERSION_TABLE,
                 fg.FOREIGN_COLUMNS_TABLE, fg.IMPORT_TABLE):
        assert name in tables

    cells = _read(result.db_path, "cell")
    assert len(cells) == 4
    for key in ("object_label", "plateID", "rowID", "columnID", "fieldID",
                "prcf", "file_name", "path_name"):
        assert key in cells.columns
    assert sorted(cells["prcf"].unique()) == ["plate1_r1_c1_f1",
                                              "plate1_r1_c1_f2"]
    assert sorted(cells["object_label"].unique()) == [1, 2]
    # path_name points at a merged array that exists — that is what every
    # crop and every annotation follows.
    for path in cells["path_name"]:
        assert os.path.isfile(path)

    # -- their columns are there, under the foreign prefix --------------
    assert "foreign_areashape_area" in cells.columns
    assert "foreign_metadata_treatment" in cells.columns
    assert sorted(cells["foreign_areashape_area"].unique()) == [36.0, 64.0]
    assert set(cells["foreign_metadata_treatment"]) == {"wt", "ko"}


def test_the_import_says_which_columns_are_theirs_and_which_are_spacrs(
        theirs, tmp_path):
    """``foreign_columns`` answers it per column, not in prose."""
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"))
    provenance = _read(result.db_path, fg.FOREIGN_COLUMNS_TABLE)
    cell = provenance[provenance["table"] == "cell"]

    spacr = set(cell[cell["origin"] == "spacr"]["column"])
    foreign = set(cell[cell["origin"] == "foreign"]["column"])
    assert {"object_label", "plateID", "rowID", "columnID", "fieldID",
            "prcf"} <= spacr
    assert "foreign_areashape_area" in foreign
    assert not spacr & foreign
    # every foreign column records the column it came from
    assert set(cell[cell["origin"] == "foreign"]["source_column"]) >= {
        "AreaShape_Area", "Metadata_Treatment", "cell_area"}
    assert "NOT re-extracted" in result.summary()


def test_crops_are_cut_through_spacr_crops(theirs, tmp_path):
    """One PNG per object, via :mod:`spacr.crops`, not a second cropper."""
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                           crops=True)
    assert len(result.crops) == 4
    for path in result.crops:
        assert os.path.isfile(path)
    assert all(os.sep + "crops" + os.sep in p for p in result.crops)


def test_two_mask_classes_land_in_spacrs_plane_order(theirs, tmp_path):
    """cell then nucleus, which is what every mask_dim in spaCR assumes."""
    plan = fg.plan_import(theirs.images, theirs.masks("cell", "nucleus"),
                          theirs.table, um_per_px=0.5)
    assert plan.mask_dims == {"cell": 2, "nucleus": 3}
    result = fg.run_import(plan, str(tmp_path / "imported"))
    array = np.load(result.merged[0])
    assert array.shape == (SIZE, SIZE, 4)
    assert set(np.unique(array[:, :, 2])) == {0, 1, 2}
    assert set(np.unique(array[:, :, 3])) == {0, 1, 2}


# ---------------------------------------------------------------------------
# 2. Unmapped columns
# ---------------------------------------------------------------------------

def test_an_unmapped_column_is_reported_by_name_and_not_dropped(theirs,
                                                                tmp_path):
    """The user's own words: report it, do not silently drop it."""
    maps = [m for m in fg.infer_column_map(
        pd.read_csv(theirs.table), image_key="ImageNumber",
        label_key="ObjectNumber") if m.source != "Metadata_Treatment"]

    plan = _plan(theirs, column_maps=maps)
    # named in the plan …
    assert plan.unmapped == ["Metadata_Treatment"]
    # … in the rendered plan …
    text = fg.format_plan(plan)
    assert "COLUMNS WITH NO MAPPING (1)" in text
    assert "Metadata_Treatment" in text
    # … and in the warnings, by name.
    assert any("Metadata_Treatment" in w for w in plan.warnings)

    result = fg.run_import(plan, str(tmp_path / "imported"))
    # … and the values are still there.
    cells = _read(result.db_path, "cell")
    assert "foreign_metadata_treatment" in cells.columns
    assert set(cells["foreign_metadata_treatment"]) == {"wt", "ko"}
    assert "Metadata_Treatment" in result.summary()

    provenance = _read(result.db_path, fg.FOREIGN_COLUMNS_TABLE)
    row = provenance[(provenance["table"] == "cell")
                     & (provenance["column"] == "foreign_metadata_treatment")]
    assert row.iloc[0]["status"] == "unmapped"
    assert row.iloc[0]["source_column"] == "Metadata_Treatment"


def test_an_empty_target_is_an_unmapped_column_too(theirs):
    """Blanking the target in the map file means "I did not decide"."""
    maps = fg.infer_column_map(pd.read_csv(theirs.table),
                               image_key="ImageNumber",
                               label_key="ObjectNumber")
    maps = [m if m.source != "AreaShape_Area"
            else fg.ColumnMap(source=m.source, target="") for m in maps]
    plan = _plan(theirs, column_maps=maps)
    resolution = next(r for r in plan.resolved
                      if r.source == "AreaShape_Area")
    assert resolution.status == "unmapped"
    assert resolution.target == "foreign_areashape_area"
    assert "rather than dropped" in resolution.reason
    # "no row at all" and "a row with a blank target" are the same thing to
    # the user, so they are one list and one report.
    assert plan.unmapped == ["AreaShape_Area"]
    assert "COLUMNS WITH NO MAPPING (1)" in fg.format_plan(plan)


# ---------------------------------------------------------------------------
# 3. Name collisions
# ---------------------------------------------------------------------------

def test_a_target_that_is_a_spacr_name_is_refused(theirs):
    """Their Area is not spaCR's cell_area, and the default says no."""
    maps = [fg.ColumnMap(source="AreaShape_Area", target="cell_area")]
    plan = _plan(theirs, column_maps=maps)

    assert not plan.ok
    kinds = {c.kind for c in plan.conflicts}
    assert "spacr_name" in kinds
    conflict = next(c for c in plan.conflicts if c.kind == "spacr_name")
    assert conflict.blocking
    assert "cell_area" in str(conflict)
    # …and refusing means refusing, not warning and carrying on.
    with pytest.raises(ConfigurationError) as excinfo:
        fg.run_import(plan, str(theirs.root) + "/out")
    assert "cell_area" in str(excinfo.value)


def test_a_colliding_target_can_be_renamed_instead_of_overwriting(theirs,
                                                                  tmp_path):
    """``on_conflict='rename'`` keeps the values and keeps spaCR's name free."""
    maps = [fg.ColumnMap(source="AreaShape_Area", target="cell_area")]
    plan = _plan(theirs, column_maps=maps, on_conflict="rename")

    assert plan.ok
    resolution = next(r for r in plan.resolved if r.source == "AreaShape_Area")
    assert resolution.status == "renamed"
    assert resolution.target.startswith(fg.FOREIGN_PREFIX)

    result = fg.run_import(plan, str(tmp_path / "imported"))
    cells = _read(result.db_path, "cell")
    assert "cell_area" not in cells.columns
    assert resolution.target in cells.columns
    assert sorted(cells[resolution.target].unique()) == [36.0, 64.0]


def test_a_spacr_target_can_be_opted_into_explicitly(theirs):
    """A decision with a name on it is allowed; an accident is not."""
    maps = [fg.ColumnMap(source="AreaShape_Area", target="cell_area")]
    plan = _plan(theirs, column_maps=maps, allow_spacr_targets=True)
    assert plan.ok
    assert not [c for c in plan.conflicts if c.kind == "spacr_name"]
    assert plan.target_for("AreaShape_Area") == "cell_area"


def test_a_reserved_key_column_is_never_a_valid_target(theirs):
    """Corrupting ``prcf`` does not corrupt a measurement, it corrupts the index."""
    for reserved in ("object_label", "plateID", "prcf", "path_name"):
        maps = [fg.ColumnMap(source="AreaShape_Area", target=reserved)]
        plan = _plan(theirs, column_maps=maps, allow_spacr_targets=True)
        assert not plan.ok, reserved
        conflict = next(c for c in plan.conflicts if c.kind == "reserved")
        assert conflict.target == reserved
        assert plan.target_for("AreaShape_Area").startswith(fg.FOREIGN_PREFIX)


def test_two_columns_cannot_share_one_target(theirs):
    maps = [fg.ColumnMap(source="AreaShape_Area", target="foreign_a"),
            fg.ColumnMap(source="AreaShape_Area_um2", target="foreign_a")]
    plan = _plan(theirs, column_maps=maps)
    assert not plan.ok
    conflict = next(c for c in plan.conflicts if c.kind == "duplicate_target")
    assert conflict.target == "foreign_a"
    assert plan.target_for("AreaShape_Area_um2") != "foreign_a"


def test_a_column_named_like_spacrs_is_flagged_even_when_it_is_renamed(theirs):
    """Their table has a literal ``cell_area``; the user has to be told."""
    plan = _plan(theirs)
    assert plan.ok                       # nothing is overwritten …
    shadow = next(c for c in plan.conflicts if c.kind == "shadows_spacr")
    assert not shadow.blocking           # … so it is not blocking …
    assert shadow.source == "cell_area"
    assert shadow.target == "foreign_cell_area"
    assert "not the same measurement" in shadow.detail   # … but it is said.
    assert "shadows_spacr" in fg.format_plan(plan)


def test_is_spacr_name_defers_to_the_feature_dictionary():
    """One parser for spaCR's column grammar, not two."""
    assert fg.is_spacr_name("cell_area")
    assert fg.is_spacr_name("nucleus_channel_1_mean_intensity")
    assert fg.is_spacr_name("object_label")
    assert not fg.is_spacr_name("foreign_cell_area")
    assert not fg.is_spacr_name("AreaShape_Area")
    # and it really is feature_dict's answer
    assert fdict.parse_column("cell_area").family != "unknown"
    assert fdict.parse_column("AreaShape_Area").family == "unknown"


# ---------------------------------------------------------------------------
# 4. Units
# ---------------------------------------------------------------------------

def test_a_declared_unit_conversion_is_applied_when_the_scale_is_known(theirs,
                                                                      tmp_path):
    """9 µm² at 0.5 µm/px is 36 px². The factor is the pixel size squared."""
    plan = _plan(theirs, um_per_px=0.5)
    resolution = next(r for r in plan.resolved
                      if r.source == "AreaShape_Area_um2")
    assert resolution.mapping.transform == "area"
    assert resolution.mapping.normalised_unit_in == "um^2"
    assert resolution.calibrated
    assert resolution.factor == pytest.approx(4.0)
    assert resolution.unit == "px^2"

    result = fg.run_import(plan, str(tmp_path / "imported"))
    cells = _read(result.db_path, "cell")
    converted = cells[resolution.target]
    original = cells["foreign_areashape_area"]
    # the µm² column, converted, is now the px² column
    assert list(converted) == pytest.approx(list(original))


def test_an_unknown_pixel_size_is_reported_not_assumed_to_be_one(theirs,
                                                                tmp_path):
    """The failure this module exists to prevent, stated in three places."""
    plan = _plan(theirs, um_per_px=None)
    resolution = next(r for r in plan.resolved
                      if r.source == "AreaShape_Area_um2")

    assert not resolution.calibrated
    assert resolution.factor is None            # not 1.0
    assert resolution.status == "uncalibrated"
    assert resolution.unit == "um^2"            # stored in THEIR unit
    assert "no pixel size" in resolution.reason
    assert resolution.target.startswith(fg.FOREIGN_PREFIX)

    assert any("UNCALIBRATED" in w for w in plan.warnings)
    assert "UNCALIBRATED COLUMNS" in fg.format_plan(plan)
    assert resolution in plan.uncalibrated

    result = fg.run_import(plan, str(tmp_path / "imported"))
    cells = _read(result.db_path, "cell")
    # the values are untouched — 9 and 16 µm², not 36 and 64 px²
    assert sorted(cells[resolution.target].unique()) == [9.0, 16.0]
    provenance = _read(result.db_path, fg.FOREIGN_COLUMNS_TABLE)
    row = provenance[(provenance["table"] == "cell")
                     & (provenance["column"] == resolution.target)]
    assert int(row.iloc[0]["calibrated"]) == 0
    assert row.iloc[0]["unit"] == "um^2"
    assert "UNCALIBRATED" in result.summary()


def test_an_uncalibrated_column_never_lands_on_a_spacr_name(theirs):
    """µm² in a column labelled px² is exactly the corruption to avoid."""
    maps = [fg.ColumnMap(source="AreaShape_Area_um2", target="cell_area",
                         transform="area", unit_in="um^2", unit_out="px^2")]
    plan = _plan(theirs, column_maps=maps, um_per_px=None,
                 allow_spacr_targets=True)
    resolution = next(r for r in plan.resolved)
    assert resolution.target != "cell_area"
    assert resolution.target.startswith(fg.FOREIGN_PREFIX)
    assert not resolution.calibrated
    assert any("cell_area" in w and "UNCALIBRATED" in w for w in plan.warnings)


@pytest.mark.parametrize("transform,unit_in,unit_out,scale,expected", [
    ("identity", "", "", None, 1.0),
    ("length", "um", "px", 0.5, 2.0),
    ("area", "um^2", "px^2", 0.5, 4.0),
    ("volume", "um^3", "px^3", 0.5, 8.0),
    ("length", "px", "um", 0.5, 0.5),
    ("*2.5", "", "", None, 2.5),
    ("/4", "", "", None, 0.25),
])
def test_the_transform_vocabulary(transform, unit_in, unit_out, scale,
                                  expected):
    mapping = fg.ColumnMap(source="x", target="y", transform=transform,
                           unit_in=unit_in, unit_out=unit_out)
    factor, reason = mapping.resolve(scale)
    assert reason == ""
    assert factor == pytest.approx(expected)


@pytest.mark.parametrize("transform,unit_in,unit_out,scale,fragment", [
    ("area", "", "px^2", 1.0, "needs unit_in"),
    ("area", "um^2", "px^2", None, "no pixel size"),
    ("area", "um^2", "px^2", 0.0, "positive number"),
    ("area", "um^2", "px^2", "big", "not a number"),
    ("area", "counts", "px^2", 1.0, "not a length/area/volume unit"),
    ("area", "um^2", "um^2", 1.0, "nothing to convert"),
    ("area", "um", "px", 1.0, "power-2 conversion"),
    ("identity", "um^2", "px^2", 1.0, 'transform is "identity"'),
    ("wobble", "", "", 1.0, "unknown transform"),
    ("/0", "", "", 1.0, "unknown transform"),
])
def test_a_conversion_that_cannot_be_done_says_why(transform, unit_in,
                                                   unit_out, scale, fragment):
    mapping = fg.ColumnMap(source="x", target="y", transform=transform,
                           unit_in=unit_in, unit_out=unit_out)
    factor, reason = mapping.resolve(scale)
    assert factor is None
    assert fragment in reason


@pytest.mark.parametrize("text,expected", [
    ("um", "um"), ("µm", "um"), ("MICRONS", "um"), ("um^2", "um^2"),
    ("µm²", "um^2"), ("um2", "um^2"), ("px", "px"), ("Pixels", "px"),
    ("px^2", "px^2"), ("", ""), ("none", ""), ("furlongs", "furlongs"),
])
def test_unit_spellings_normalise_and_unknown_ones_survive(text, expected):
    """A unit we do not understand is recorded, never blanked."""
    assert fg._norm_unit(text) == expected


def test_a_non_numeric_column_is_never_multiplied(theirs):
    """Scaling a column of treatment names is a crash, not a conversion."""
    resolution = fg.ResolvedColumn(
        mapping=fg.ColumnMap(source="Metadata_Treatment", target="t"),
        target="t", factor=4.0, calibrated=True, unit="px^2",
        status="mapped")
    values = pd.Series(["wt", "ko"])
    assert list(resolution.apply(values)) == ["wt", "ko"]


# ---------------------------------------------------------------------------
# 5. The join
# ---------------------------------------------------------------------------

def test_the_join_key_is_stated(theirs):
    plan = _plan(theirs)
    assert plan.join.image_key == "ImageNumber"
    assert plan.join.label_key == "ObjectNumber"
    assert plan.join.object_type == "cell"
    assert "ImageNumber" in plan.join.key_description
    assert "ObjectNumber" in plan.join.key_description
    assert "Join key" in fg.format_plan(plan)
    assert plan.join.rows_total == 4
    assert plan.join.rows_matched == 4
    assert plan.join.match_rate == 1.0


def test_rows_matching_no_mask_object_are_counted_and_reported(theirs,
                                                              tmp_path):
    """40% of rows matching nothing is broken — with the number attached."""
    frame = pd.read_csv(theirs.table)
    extra = frame.iloc[:2].copy()
    extra["ObjectNumber"] = [98, 99]          # labels in no mask
    orphan = frame.iloc[:1].copy()
    orphan["ImageNumber"] = "fov77_C1.tif"    # an image in no import
    pd.concat([frame, extra, orphan], ignore_index=True).to_csv(
        theirs.table, index=False)

    plan = _plan(theirs)
    join = plan.join
    assert join.rows_total == 7
    assert join.rows_matched == 4
    assert join.rows_unmatched == 3
    assert join.n_no_object == 2
    assert join.n_unresolved == 1
    assert join.match_rate == pytest.approx(4 / 7)

    # counted per field, and named
    assert dict(join.rows_no_object)["plate1_A01_1"] == 2
    assert dict(join.unresolved_fields)["fov77_C1.tif"] == 1
    assert any("no object with label 98" in e for e in join.examples)
    assert any("match no object in the masks" in w for w in plan.warnings)

    text = fg.format_plan(plan)
    assert "4/7 measurement row(s) matched" in text
    assert "fov77_C1.tif" in text

    result = fg.run_import(plan, str(tmp_path / "imported"))
    # the resolvable rows survive with their label, unmatched or not
    cells = _read(result.db_path, "cell")
    assert len(cells) == 6                    # the orphan field is unknown
    assert {98, 99} <= set(cells["object_label"])
    run = _read(result.db_path, fg.IMPORT_TABLE).iloc[0]
    assert int(run["rows_total"]) == 7
    assert int(run["rows_matched"]) == 4
    assert int(run["rows_unmatched"]) == 3
    assert "matched no mask object" in result.summary()


def test_mask_objects_nobody_measured_are_counted_too(theirs):
    frame = pd.read_csv(theirs.table)
    frame = frame[frame["ObjectNumber"] == 1]      # drop every label-2 row
    frame.to_csv(theirs.table, index=False)

    plan = _plan(theirs)
    assert plan.join.n_objects_unmeasured == 2     # one per field
    assert dict(plan.join.objects_unmeasured)["plate1_A01_1"] == 1
    assert any("no row in the measurement table" in w for w in plan.warnings)
    assert "mask object(s) have no measurement row" in fg.format_plan(plan)


def test_a_table_with_no_label_column_is_refused(theirs):
    frame = pd.read_csv(theirs.table).drop(columns=["ObjectNumber"])
    frame.to_csv(theirs.table, index=False)
    plan = _plan(theirs, label_key=None)
    assert not plan.ok
    assert any("looks like an object label" in e for e in plan.errors)


def test_the_image_key_resolves_by_name_stem_or_field(theirs):
    """Their spelling of "which image" rarely matches ours exactly."""
    for spelling in ("fov01_C1.tif", "fov01_C2.tif", "fov01",
                     os.path.join("sub", "fov01_C1.tif")):
        frame = pd.read_csv(theirs.table)
        frame["ImageNumber"] = [spelling if i < 2 else "fov02_C1.tif"
                                for i in range(len(frame))]
        frame.to_csv(theirs.table, index=False)
        plan = _plan(theirs)
        assert plan.join.rows_matched == 4, spelling


# ---------------------------------------------------------------------------
# 6. Images and masks, file by file
# ---------------------------------------------------------------------------

def test_an_image_with_no_mask_is_reported_per_file(theirs, tmp_path):
    _write_image(os.path.join(theirs.images, "fov03_C1.tif"), 31)
    _write_image(os.path.join(theirs.images, "fov03_C2.tif"), 32)

    plan = _plan(theirs)
    orphans = {os.path.basename(p) for p, _t in plan.masks.images_without_masks}
    assert orphans == {"fov03_C1.tif", "fov03_C2.tif"}      # both files
    assert all(t == "cell" for _p, t in plan.masks.images_without_masks)
    assert "plate1_A01_3" in plan.masks.excluded
    assert not plan.masks.ok
    assert any("have no matching mask" in w for w in plan.warnings)

    text = fg.format_plan(plan)
    assert "image with no cell mask" in text
    assert "fov03_C1.tif" in text and "fov03_C2.tif" in text

    # the incomplete field is left out, and the rest still imports
    result = fg.run_import(plan, str(tmp_path / "imported"))
    assert result.n_fields == 2
    assert not any("plate1_A01_3" in p for p in result.merged)


def test_a_mask_with_no_image_is_reported_per_file(theirs):
    stray = _write_mask(os.path.join(theirs.cell_masks, "fov09_cell_mask.tif"))

    plan = _plan(theirs)
    assert [p for p, _t in plan.masks.masks_without_images] == [stray]
    assert plan.masks.masks_without_images[0][1] == "cell"
    assert any("match no image" in w for w in plan.warnings)
    assert "cell mask with no image" in fg.format_plan(plan)
    # everything that DID pair still imports
    assert plan.ok
    assert sorted(plan.masks.fields) == ["plate1_A01_1", "plate1_A01_2"]


def test_a_field_missing_one_of_two_mask_classes_is_excluded_and_named(theirs):
    os.remove(os.path.join(theirs.nucleus_masks, "fov02_nucleus_mask.tif"))
    plan = fg.plan_import(theirs.images, theirs.masks("cell", "nucleus"),
                          theirs.table, um_per_px=0.5)
    assert list(plan.masks.fields) == ["plate1_A01_1"]
    assert plan.masks.excluded == ["plate1_A01_2"]
    missing = {t for _p, t in plan.masks.images_without_masks}
    assert missing == {"nucleus"}


def test_two_masks_claiming_one_field_is_a_blocking_error(theirs):
    _write_mask(os.path.join(theirs.cell_masks, "fov01_masks.tif"))
    plan = _plan(theirs)
    assert not plan.ok
    assert any("both claim to be the cell mask" in e for e in plan.errors)


def test_no_mask_folder_at_all_is_refused(theirs):
    plan = fg.plan_import(theirs.images, {}, theirs.table)
    assert not plan.ok
    assert any("No mask folder was given" in e for e in plan.errors)


def test_an_unknown_mask_class_is_a_configuration_error(theirs):
    with pytest.raises(ConfigurationError, match="Unknown mask object type"):
        fg.plan_import(theirs.images, {"mitochondrion": theirs.cell_masks},
                       theirs.table)


@pytest.mark.parametrize("stem,expected", [
    ("fov01_cell_mask", "fov01"),
    ("fov01_masks", "fov01"),
    ("fov01_cp_masks", "fov01"),
    ("fov01-segmentation", "fov01"),
    ("fov01", "fov01"),
    ("fov01_treated", "fov01_treated"),      # not a mask token: kept whole
])
def test_mask_suffix_stripping_only_removes_mask_tokens(stem, expected):
    assert fg._normalise_mask_field(stem, "cell", fg.MASK_SUFFIXES) == expected


# ---------------------------------------------------------------------------
# 7. The mapping file
# ---------------------------------------------------------------------------

def test_the_mapping_round_trips_and_is_applied_exactly(theirs, tmp_path):
    """What you saved is what runs — not what the inference would redo."""
    plan = _plan(theirs)
    path = tmp_path / "column_map.csv"
    fg.save_column_map(plan, str(path))

    # a human edits it: a new target, a declared unit, a note
    edited = fg.load_column_map(str(path))
    assert [m.to_row() for m in edited] == [m.to_row()
                                            for m in plan.column_maps]
    edited = [fg.ColumnMap(source="AreaShape_Area", target="foreign_my_area",
                           transform="*10", note="checked by hand")
              if m.source == "AreaShape_Area" else m for m in edited]
    fg.save_column_map(edited, str(path))
    reloaded = fg.load_column_map(str(path))
    assert [m.to_row() for m in reloaded] == [m.to_row() for m in edited]

    applied = _plan(theirs, column_maps=reloaded)
    assert applied.target_for("AreaShape_Area") == "foreign_my_area"
    result = fg.run_import(applied, str(tmp_path / "imported"))

    cells = _read(result.db_path, "cell")
    assert "foreign_my_area" in cells.columns
    assert "foreign_areashape_area" not in cells.columns
    assert sorted(cells["foreign_my_area"].unique()) == [360.0, 640.0]

    # and the map that actually ran is written next to the project
    assert os.path.isfile(result.column_map_path)
    assert [m.to_row() for m in fg.load_column_map(result.column_map_path)] \
        == [m.to_row() for m in reloaded]


def test_the_map_file_carries_a_comment_header_that_survives_reading(tmp_path):
    maps = [fg.ColumnMap(source="A", target="foreign_a", note="hello, world")]
    path = tmp_path / "m.csv"
    fg.save_column_map(maps, str(path))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# spaCR foreign column map")
    assert "identity | length | area | volume" in text
    assert [m.to_row() for m in fg.load_column_map(str(path))] == \
        [m.to_row() for m in maps]


def test_a_bad_mapping_file_says_what_is_wrong(tmp_path):
    missing = tmp_path / "nope.csv"
    with pytest.raises(ConfigurationError, match="does not exist"):
        fg.load_column_map(str(missing))

    wrong = tmp_path / "wrong.csv"
    wrong.write_text("alpha,beta\n1,2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not a spaCR column map"):
        fg.load_column_map(str(wrong))

    empty = tmp_path / "empty.csv"
    empty.write_text("# only a comment\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no rows"):
        fg.load_column_map(str(empty))

    twice = tmp_path / "twice.csv"
    twice.write_text("source,target\nA,x\nA,y\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="twice"):
        fg.load_column_map(str(twice))

    nameless = tmp_path / "nameless.csv"
    nameless.write_text("source,target\n,x\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no \"source\""):
        fg.load_column_map(str(nameless))


def test_a_map_naming_a_column_that_does_not_exist_is_blocking(theirs):
    plan = _plan(theirs, column_maps=[fg.ColumnMap(source="Nonexistent",
                                                   target="foreign_x")])
    assert not plan.ok
    assert any("Nonexistent" in e for e in plan.errors)


def test_the_inferred_mapping_is_labelled_a_proposal(theirs):
    plan = _plan(theirs)
    assert plan.proposed
    assert any("INFERRED PROPOSAL" in n for n in plan.notes)
    assert "INFERRED PROPOSAL" in fg.format_plan(plan)
    # …and it never proposes a spaCR feature name
    assert all(not fg.is_spacr_name(m.target) for m in plan.column_maps)
    assert all(m.target.startswith(fg.FOREIGN_PREFIX)
               for m in plan.column_maps)
    # a mapping handed in is not a proposal
    assert not _plan(theirs, column_maps=plan.column_maps).proposed


def test_the_inference_reads_units_out_of_column_headers():
    frame = pd.DataFrame(columns=["ImageNumber", "ObjectNumber",
                                  "Area (µm²)", "Perimeter_um",
                                  "Volume_um3", "Diameter_px", "Blob"])
    maps = {m.source: m for m in fg.infer_column_map(frame)}
    assert "ImageNumber" not in maps and "ObjectNumber" not in maps

    assert maps["Area (µm²)"].transform == "area"
    assert maps["Area (µm²)"].normalised_unit_in == "um^2"
    assert maps["Area (µm²)"].normalised_unit_out == "px^2"
    assert maps["Perimeter_um"].transform == "length"
    assert maps["Volume_um3"].transform == "volume"
    assert maps["Diameter_px"].transform == "identity"
    assert maps["Diameter_px"].normalised_unit_in == "px"
    assert maps["Blob"].transform == "identity"
    assert maps["Blob"].normalised_unit_in == ""
    assert "declare unit_in" in maps["Blob"].note


def test_the_inference_never_collides_two_headers_onto_one_target():
    frame = pd.DataFrame(columns=["Area", "area", "AREA", "id", "label"])
    maps = fg.infer_column_map(frame, image_key="id", label_key="label")
    targets = [m.target for m in maps]
    assert len(targets) == len(set(targets)) == 3


def test_editing_a_mapping_re_resolves_without_touching_the_disk(theirs):
    """What the GUI does on every keystroke."""
    plan = _plan(theirs, um_per_px=None)
    assert len(plan.uncalibrated) == 1

    # supplying the scale calibrates it, with no rescan
    rescaled = plan.with_column_maps(plan.column_maps, um_per_px=0.5)
    assert rescaled.uncalibrated == []
    assert rescaled.target_for("AreaShape_Area_um2") \
        == plan.target_for("AreaShape_Area_um2")
    assert rescaled.join is plan.join
    assert rescaled.images is plan.images

    # dropping a mapping makes its column unmapped, still not dropped
    fewer = [m for m in plan.column_maps if m.source != "AreaShape_Area"]
    trimmed = plan.with_column_maps(fewer)
    assert trimmed.unmapped == ["AreaShape_Area"]
    assert trimmed.target_for("AreaShape_Area") == "foreign_areashape_area"


# ---------------------------------------------------------------------------
# 8. Re-running
# ---------------------------------------------------------------------------

def test_a_re_run_does_not_duplicate_rows(theirs, tmp_path):
    dst = str(tmp_path / "imported")
    plan = _plan(theirs)

    first = fg.run_import(plan, dst)
    before = {name: len(_read(first.db_path, name))
              for name in ("cell", "foreign_cell", cv.CONVERSION_TABLE,
                           fg.FOREIGN_COLUMNS_TABLE, fg.IMPORT_TABLE)}
    checksum = np.load(first.merged[0]).sum()

    second = fg.run_import(plan, dst)
    after = {name: len(_read(second.db_path, name)) for name in before}
    assert after == before
    assert second.n_fields == first.n_fields
    assert np.load(second.merged[0]).sum() == checksum
    # the second pass wrote no new images — the converter skipped them
    assert second.conversion.n_written == 0
    assert len(second.conversion.existing) == first.conversion.n_written


# ---------------------------------------------------------------------------
# 9. The image half really is spacr.convert
# ---------------------------------------------------------------------------

def test_the_image_half_calls_spacr_convert(theirs, tmp_path, monkeypatch):
    """Asserted, not assumed: no second naming scheme lives in this module."""
    calls = {}

    def _spy(name):
        original = getattr(cv, name)

        def wrapper(*args, **kwargs):
            calls.setdefault(name, 0)
            calls[name] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(cv, name, wrapper)

    for name in ("scan", "plan", "convert", "write_map", "populate_db_from_map",
                 "target_name", "_read_source"):
        _spy(name)

    plan = _plan(theirs)
    assert calls["scan"] == 2            # images, then the cell masks
    assert calls["plan"] == 1
    assert calls["target_name"] >= 4     # one per output TIFF
    assert calls["_read_source"] == 2    # one per mask, to verify the join

    result = fg.run_import(plan, str(tmp_path / "imported"))
    assert calls["convert"] == 1
    assert calls["write_map"] == 1
    assert calls["populate_db_from_map"] == 1

    # …and the conversion map really is convert.py's, read back by convert.py
    frame = cv.read_map(result.conversion.map_path)
    assert list(frame.columns) == list(cv.MAP_COLUMNS)
    assert set(frame["source_relpath"]) == {"fov01_C1.tif", "fov01_C2.tif",
                                            "fov02_C1.tif", "fov02_C2.tif"}
    stored = _read(result.db_path, cv.CONVERSION_TABLE)
    assert len(stored) == 4
    assert set(stored["prcf"]) == {"plate1_r1_c1_f1", "plate1_r1_c1_f2"}


def test_the_merged_arrays_are_built_by_spacr_io(theirs, tmp_path,
                                                 monkeypatch):
    """The merger is spaCR's own, so the plane layout cannot drift."""
    import spacr.io as sio

    seen = {}
    original = sio._load_and_concatenate_arrays

    def wrapper(src, channels, **kwargs):
        seen["src"] = src
        seen["channels"] = channels
        seen["kwargs"] = kwargs
        return original(src, channels, **kwargs)

    monkeypatch.setattr(sio, "_load_and_concatenate_arrays", wrapper)
    dst = str(tmp_path / "imported")
    fg.run_import(_plan(theirs), dst)
    assert seen["src"] == dst
    assert seen["channels"] is None
    assert seen["kwargs"]["resume"] is False


def test_the_destination_layout_is_the_one_spacr_expects(theirs, tmp_path):
    dst = str(tmp_path / "imported")
    result = fg.run_import(_plan(theirs), dst)
    for relative in (fg.IMAGES_DIRNAME, "stack", "merged",
                     os.path.join("masks", "cell_mask_stack"), "measurements"):
        assert os.path.isdir(os.path.join(dst, relative)), relative
    assert sorted(os.listdir(os.path.join(dst, "stack"))) == [
        "plate1_A01_1.npy", "plate1_A01_2.npy"]
    assert np.load(os.path.join(dst, "stack", "plate1_A01_1.npy")).shape \
        == (SIZE, SIZE, 2)
    assert os.path.isfile(os.path.join(dst, fg.COLUMN_MAP_FILENAME))
    assert result.stacks and result.mask_files


# ---------------------------------------------------------------------------
# Re-extraction: optional and separate
# ---------------------------------------------------------------------------

def test_re_extraction_is_optional_and_keeps_the_two_halves_apart(theirs,
                                                                  tmp_path):
    """spaCR's columns in ``cell``, theirs in ``foreign_cell``, joined by a view."""
    import spacr.measure as measure

    captured = {}

    def fake_measure_crop(settings):
        captured.update(settings)
        # stand in for the real thing: write a spaCR-shaped cell table
        connection = sqlite3.connect(
            os.path.join(settings["src"], "measurements", "measurements.db"))
        try:
            pd.DataFrame([
                {"object_label": label, "prcf": f"plate1_r1_c1_f{field}",
                 "cell_area": 36.0 if label == 1 else 64.0}
                for field in (1, 2) for label in (1, 2)
            ]).to_sql("cell", connection, if_exists="replace", index=False)
        finally:
            connection.close()

    original = measure.measure_crop
    measure.measure_crop = fake_measure_crop
    try:
        result = fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                               measure=True)
    finally:
        measure.measure_crop = original

    assert result.measured
    assert captured["src"] == result.dst
    assert captured["channels"] == [0, 1]
    assert captured["cell_mask_dim"] == 2
    assert captured["nucleus_mask_dim"] is None

    tables = _tables(result.db_path)
    assert "cell" in tables and "foreign_cell" in tables
    assert "cell_with_foreign" in tables

    spacr_cells = _read(result.db_path, "cell")
    assert "cell_area" in spacr_cells.columns          # spaCR's
    assert not [c for c in spacr_cells.columns
                if c.startswith(fg.FOREIGN_PREFIX)]    # and only spaCR's

    joined = _read(result.db_path, "cell_with_foreign")
    assert len(joined) == 4
    assert "cell_area" in joined.columns
    assert "foreign_areashape_area" in joined.columns
    assert "re-extracted into the standard object tables" in result.summary()


# ---------------------------------------------------------------------------
# Reading their table
# ---------------------------------------------------------------------------

def test_read_measurements_handles_the_formats_people_send(theirs, tmp_path):
    frame = pd.read_csv(theirs.table)
    assert len(fg.read_measurements(frame)) == len(frame)

    tsv = tmp_path / "t.tsv"
    frame.to_csv(tsv, sep="\t", index=False)
    assert list(fg.read_measurements(str(tsv)).columns) == list(frame.columns)

    db = tmp_path / "t.db"
    connection = sqlite3.connect(str(db))
    try:
        frame.to_sql("objects", connection, index=False)
    finally:
        connection.close()
    assert len(fg.read_measurements(str(db))) == len(frame)
    assert len(fg.read_measurements(str(db), table="objects")) == len(frame)


def test_read_measurements_refuses_what_it_cannot_open(tmp_path):
    with pytest.raises(ConfigurationError, match="does not exist"):
        fg.read_measurements(str(tmp_path / "nope.csv"))

    weird = tmp_path / "data.rds"
    weird.write_text("x", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="no reader"):
        fg.read_measurements(str(weird))

    two = tmp_path / "two.db"
    connection = sqlite3.connect(str(two))
    try:
        pd.DataFrame({"a": [1]}).to_sql("one", connection, index=False)
        pd.DataFrame({"a": [1]}).to_sql("other", connection, index=False)
    finally:
        connection.close()
    with pytest.raises(ConfigurationError, match="name the one to import"):
        fg.read_measurements(str(two))
    with pytest.raises(ConfigurationError, match="has no table"):
        fg.read_measurements(str(two), table="absent")

    broken = tmp_path / "broken.csv"
    broken.write_text("", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="could not be read"):
        fg.read_measurements(str(broken))


# ---------------------------------------------------------------------------
# Options and entry points
# ---------------------------------------------------------------------------

def test_keeping_every_z_plane_is_refused_because_merged_arrays_are_2d(
        theirs, tmp_path):
    stack = np.stack([_label_image()] * 3).astype(np.uint16)
    tifffile.imwrite(os.path.join(theirs.images, "fov01_C1.tif"), stack,
                     metadata={"axes": "ZYX"})
    plan = fg.plan_import(theirs.images, theirs.masks(), theirs.table,
                          z_handling=cv.Z_KEEP)
    assert not plan.ok
    assert any("no z axis" in e for e in plan.errors)
    assert any("more than one plane" in e for e in plan.errors)
    # …and the default, max-projection, is fine and says it is lossy.
    projected = fg.plan_import(theirs.images, theirs.masks(), theirs.table)
    assert projected.ok
    assert any("max-projects" in w for w in projected.images.warnings)


def test_a_ragged_channel_count_is_refused(theirs):
    os.remove(os.path.join(theirs.images, "fov02_C2.tif"))
    plan = _plan(theirs)
    assert not plan.ok
    assert any("channel(s) but the experiment has 2" in e for e in plan.errors)


def test_an_unknown_conflict_policy_is_a_configuration_error(theirs):
    with pytest.raises(ConfigurationError, match="Unknown on_conflict"):
        fg.plan_import(theirs.images, theirs.masks(), theirs.table,
                       on_conflict="shrug")


def test_import_project_previews_before_it_writes(theirs, tmp_path, capsys):
    settings = {"images": theirs.images, "masks": theirs.masks(),
                "measurements": theirs.table, "dst": str(tmp_path / "out"),
                "um_per_px": 0.5, "preview_only": True}
    result = fg.import_project(settings)
    printed = capsys.readouterr().out
    assert "spaCR foreign import — nothing has been written." in printed
    assert "Join key" in printed
    assert result.merged == []
    assert not os.path.exists(str(tmp_path / "out"))

    result = fg.import_project(settings, preview_only=False)
    assert result.n_fields == 2
    assert os.path.isfile(result.db_path)


def test_import_project_needs_all_three_inputs(theirs):
    for missing in ("images", "masks", "measurements"):
        settings = {"images": theirs.images, "masks": theirs.masks(),
                    "measurements": theirs.table}
        settings[missing] = None
        with pytest.raises(ConfigurationError, match=missing):
            fg.import_project(settings)


def test_import_project_refuses_a_blocking_plan(theirs, tmp_path):
    path = tmp_path / "map.csv"
    fg.save_column_map([fg.ColumnMap(source="AreaShape_Area",
                                     target="cell_area")], str(path))
    with pytest.raises(ConfigurationError, match="Import refused"):
        fg.import_project({"images": theirs.images, "masks": theirs.masks(),
                           "measurements": theirs.table,
                           "column_map": str(path)})


def test_default_settings_round_trips_overrides():
    assert fg.default_settings()["on_conflict"] == "refuse"
    assert fg.default_settings({"um_per_px": 0.65})["um_per_px"] == 0.65
    assert fg.default_settings()["z_handling"] == cv.Z_MAX


def test_progress_is_reported_step_by_step(theirs, tmp_path):
    seen = []
    fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                  progress=lambda done, total, message: seen.append(
                      (done, total, message)))
    assert [m for _d, _t, m in seen][:2] == ["converting images",
                                             "writing stacks"]
    assert all(total >= done for done, total, _m in seen)


# ---------------------------------------------------------------------------
# The edges: every branch that only fires when something is wrong
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("Area (µm²)", "area_um2"),
    ("!!!", "column"),
    ("3rd channel", "c3rd_channel"),
    ("Mean  Intensity", "mean_intensity"),
])
def test_column_names_are_sanitised_into_usable_identifiers(raw, expected):
    assert fg._sanitise_column(raw) == expected


def test_a_feature_dictionary_that_raises_does_not_take_the_import_down(
        monkeypatch):
    """``is_spacr_name`` guards a *question*; it must never be the failure."""
    def _boom(_name):
        raise RuntimeError("the dictionary exploded")

    monkeypatch.setattr(fdict, "parse_column", _boom)
    assert fg.is_spacr_name("cell_area") is False


def test_the_literal_transform_is_recognised_as_one():
    assert fg.ColumnMap(source="x", transform="*2").is_literal
    assert fg.ColumnMap(source="x", transform="/2").is_literal
    assert not fg.ColumnMap(source="x", transform="area").is_literal
    assert fg.ColumnMap(source="x", transform="x2").literal_factor == 2.0
    assert fg.ColumnMap(source="x", transform="/0").literal_factor is None


def test_two_units_this_module_has_no_opinion_about_are_still_compared():
    assert fg.ColumnMap(source="x", transform="identity",
                        unit_in="counts", unit_out="au").declares_conversion
    assert not fg.ColumnMap(source="x", transform="identity",
                            unit_in="counts", unit_out="counts"
                            ).declares_conversion


def test_a_unit_hint_that_is_not_a_length_is_not_read_as_one():
    assert fg._column_unit_hint("Signal_counts") == ""
    assert fg._column_unit_hint("Perimeter_um") == "um"
    assert fg._column_unit_hint("Blob") == ""


def test_a_metric_unit_with_no_family_hint_is_proposed_by_its_exponent():
    """``Thing_um`` says nothing about being a length; the unit does."""
    maps = {m.source: m for m in fg.infer_column_map(
        pd.DataFrame(columns=["id", "label", "Thing_um", "Thing_um2"]),
        image_key="id", label_key="label")}
    assert maps["Thing_um"].transform == "length"
    assert maps["Thing_um2"].transform == "area"


def test_the_key_hints_match_on_a_prefix_or_a_suffix_too():
    assert fg._pick_key(["Image_Name", "Obj_label"], fg._IMAGE_KEY_HINTS) \
        == "Image_Name"
    assert fg._pick_key(["Image_Name", "Obj_label"], fg._LABEL_KEY_HINTS) \
        == "Obj_label"
    assert fg._pick_key(["alpha", "beta"], fg._LABEL_KEY_HINTS) is None


def test_a_column_the_inference_skips_is_left_out_entirely():
    maps = fg.infer_column_map(pd.DataFrame(columns=["a", "b", "c"]),
                               image_key="a", label_key="b", skip=["c"])
    assert maps == []


def test_excel_and_parquet_tables_are_read_too(theirs, tmp_path):
    pytest.importorskip("openpyxl")
    pytest.importorskip("pyarrow")
    frame = pd.read_csv(theirs.table)

    excel = tmp_path / "t.xlsx"
    frame.to_excel(excel, index=False)
    assert len(fg.read_measurements(str(excel))) == len(frame)

    parquet = tmp_path / "t.parquet"
    frame.to_parquet(parquet, index=False)
    assert len(fg.read_measurements(str(parquet))) == len(frame)


def test_a_multi_plane_mask_uses_its_first_plane(theirs, tmp_path):
    path = os.path.join(theirs.cell_masks, "fov01_cell_mask.tif")
    stacked = np.stack([_label_image(), np.zeros((SIZE, SIZE), np.uint16)])
    tifffile.imwrite(path, stacked, metadata={"axes": "CYX"})
    assert set(fg._read_mask(path).ravel()) == {0, 1, 2}


def test_a_mask_that_is_not_a_2d_label_image_is_refused(theirs):
    tifffile.imwrite(os.path.join(theirs.cell_masks, "fov01_cell_mask.tif"),
                     np.zeros((1, 1), np.uint16))
    with pytest.raises(ConfigurationError, match="2-D label image"):
        _plan(theirs)


def test_an_unreadable_mask_is_reported_and_not_guessed_at(theirs):
    broken = os.path.join(theirs.cell_masks, "fov09_cell_mask.tif")
    with open(broken, "wb") as handle:
        handle.write(b"this is not a TIFF")
    plan = _plan(theirs)
    assert [p for p, _reason in plan.masks.unreadable_masks] == [broken]
    assert any("could not be read" in w for w in plan.warnings)
    assert "unreadable mask" in fg.format_plan(plan)
    assert not plan.masks.ok


def test_a_mask_that_fails_only_when_it_is_read_is_recorded_not_raised(
        theirs, monkeypatch):
    """The scan said yes and the read said no — one field lost, not a batch."""
    real = fg._read_mask

    def _flaky(source):
        path = source if isinstance(source, str) else source.path
        if "fov02" in str(path):
            raise OSError("input/output error")
        return real(source)

    monkeypatch.setattr(fg, "_read_mask", _flaky)
    plan = _plan(theirs)
    assert [os.path.basename(p) for p, _r in plan.masks.unreadable_masks] \
        == ["fov02_cell_mask.tif"]
    assert list(plan.masks.fields) == ["plate1_A01_1"]


def test_a_mask_named_exactly_like_the_image_field_matches_without_stripping(
        theirs):
    for field in (1, 2):
        os.remove(os.path.join(theirs.cell_masks,
                               f"fov{field:02d}_cell_mask.tif"))
        _write_mask(os.path.join(theirs.cell_masks, f"fov{field:02d}.tif"))
    plan = _plan(theirs)
    assert sorted(plan.masks.fields) == ["plate1_A01_1", "plate1_A01_2"]
    assert all(m["cell"].match == "exact" for m in plan.masks.fields.values())


def test_masks_that_pair_with_nothing_leave_nothing_to_import(theirs, tmp_path):
    lonely = tmp_path / "lonely_masks"
    lonely.mkdir()
    _write_mask(str(lonely / "zzz_cell_mask.tif"))
    plan = fg.plan_import(theirs.images, {"cell": str(lonely)}, theirs.table)
    assert not plan.ok
    assert any("nothing to import" in e for e in plan.errors)
    assert "BLOCKING PROBLEMS" in fg.format_plan(plan)


def test_a_single_field_needs_no_image_key_but_many_fields_do(theirs, tmp_path):
    frame = pd.read_csv(theirs.table).drop(columns=["ImageNumber"])
    frame.to_csv(theirs.table, index=False)
    plan = _plan(theirs)
    assert not plan.ok
    assert any("No column identifies which image" in e for e in plan.errors)

    # with one field it is unambiguous, and the join says so
    for name in os.listdir(theirs.images):
        if name.startswith("fov02"):
            os.remove(os.path.join(theirs.images, name))
    os.remove(os.path.join(theirs.cell_masks, "fov02_cell_mask.tif"))
    frame[frame["ObjectNumber"].notna()].iloc[:2].to_csv(theirs.table,
                                                         index=False)
    single = _plan(theirs)
    assert single.ok
    assert single.join.image_key == ""
    assert "single field" in single.join.key_description
    assert single.join.rows_matched == 2


def test_measuring_an_object_type_with_no_mask_is_refused(theirs):
    plan = _plan(theirs, measurement_object="nucleus")
    assert not plan.ok
    assert any("has no mask folder" in e for e in plan.errors)


def test_a_label_that_is_not_an_integer_is_counted_and_still_imported(
        theirs, tmp_path):
    frame = pd.read_csv(theirs.table)
    # The malformed value is intentional, but construct it without relying on
    # pandas' deprecated implicit int-to-object dtype conversion.
    frame["ObjectNumber"] = frame["ObjectNumber"].astype(object)
    frame.loc[0, "ObjectNumber"] = "not-a-label"
    frame.to_csv(theirs.table, index=False)

    plan = _plan(theirs)
    assert plan.join.n_no_object == 1
    assert any("is not an integer" in e for e in plan.join.examples)

    result = fg.run_import(plan, str(tmp_path / "imported"))
    cells = _read(result.db_path, "cell")
    assert len(cells) == 4
    assert 0 in set(cells["object_label"])      # unusable label, kept as 0


def test_a_renamed_duplicate_target_says_so(theirs):
    maps = [fg.ColumnMap(source="AreaShape_Area", target="foreign_a"),
            fg.ColumnMap(source="AreaShape_Area_um2", target="foreign_a")]
    plan = _plan(theirs, column_maps=maps, on_conflict="rename")
    assert plan.ok
    assert any("was renamed" in w and "foreign_a" in w for w in plan.warnings)
    assert plan.target_for("AreaShape_Area") == "foreign_a"
    assert plan.target_for("AreaShape_Area_um2") != "foreign_a"


def test_a_declared_unit_out_that_is_not_a_scale_unit_is_refused():
    mapping = fg.ColumnMap(source="x", target="y", transform="area",
                           unit_in="um^2", unit_out="counts")
    factor, reason = mapping.resolve(0.5)
    assert factor is None
    assert "needs unit_out to be a pixel or micrometre unit" in reason


def test_more_than_twenty_ragged_fields_are_summarised(tmp_path):
    images = tmp_path / "images"
    masks = tmp_path / "masks"
    for field in range(1, 24):
        _write_image(str(images / f"fov{field:02d}_C1.tif"), field)
        _write_mask(str(masks / f"fov{field:02d}_cell_mask.tif"))
    _write_image(str(images / "fov01_C2.tif"), 99)       # only field 1 has C2
    table = tmp_path / "r.csv"
    pd.DataFrame([{"ImageNumber": "fov01_C1.tif", "ObjectNumber": 1,
                   "Area": 1.0}]).to_csv(table, index=False)

    plan = fg.plan_import(str(images), {"cell": str(masks)}, str(table))
    assert not plan.ok
    ragged = [e for e in plan.errors if "channel(s) but the experiment" in e]
    assert len(ragged) == 20
    assert any("more field(s) with a different channel count" in e
               for e in plan.errors)


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def test_the_join_report_summarises_long_lists_rather_than_dumping_them():
    report = fg.JoinReport(
        image_key="Image", label_key="Object", rows_total=100, rows_matched=0,
        unresolved_fields=[(f"img{i}", 1) for i in range(15)],
        rows_no_object=[(f"stem{i}", 2) for i in range(15)],
        objects_unmeasured=[(f"stem{i}", 3) for i in range(15)],
        ambiguous_keys=[f"k{i}" for i in range(3)],
        examples=["one", "two"])
    text = report.summary()
    assert "and 5 more distinct value(s)" in text
    assert "and 5 more field(s)" in text
    assert "matched more than one field" in text
    assert "e.g. one" in text
    assert report.match_rate == 0.0
    assert fg.JoinReport().match_rate == 1.0        # nothing to match


def test_format_plan_truncates_long_file_lists(theirs):
    plan = _plan(theirs)
    plan.masks.images_without_masks = [(f"/img{i}.tif", "cell")
                                       for i in range(25)]
    plan.masks.masks_without_images = [(f"/msk{i}.tif", "cell")
                                       for i in range(25)]
    plan.resolved = list(plan.resolved) + [
        fg.ResolvedColumn(mapping=fg.ColumnMap(source=f"c{i}",
                                               target=f"foreign_c{i}"),
                          target=f"foreign_c{i}", factor=1.0, calibrated=True,
                          unit="", status="mapped") for i in range(45)]
    text = fg.format_plan(plan)
    assert "more image file(s)" in text
    assert "more mask file(s)" in text
    assert "and 9 more" in text                     # 45 + 4 mapped, 40 shown


def test_the_plan_lists_every_target_and_answers_for_unknown_sources(theirs):
    plan = _plan(theirs)
    assert plan.targets() == [r.target for r in plan.resolved]
    assert plan.target_for("no such column") == ""
    assert plan.stems == ["plate1_A01_1", "plate1_A01_2"]


def test_field_aliases_ignore_empty_values_and_resolve_nothing_for_blanks():
    aliases = fg._field_aliases({"source": "", "source_relpath": "",
                                 "plate": "", "well": "", "field": None})
    assert aliases == []
    assert fg._resolve_field(None, {}) is None
    assert fg._resolve_field("   ", {}) is None
    assert fg._resolve_field("nope", {"yes": "stem"}) is None


def test_mask_folders_accept_a_path_a_mapping_a_pair_list_or_nothing(tmp_path):
    assert fg._mask_folders(None) == {}
    assert fg._mask_folders(str(tmp_path)) == {"cell": str(tmp_path)}
    assert fg._mask_folders([("nucleus", "/n"), ("cell", "/c")]) == {
        "cell": "/c", "nucleus": "/n"}


def test_the_summary_reports_crops_and_any_late_warnings(theirs, tmp_path):
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                           crops=True)
    result.warnings.append("a disk went away halfway through")
    text = result.summary()
    assert "4 crop(s) cut." in text
    assert "WARNING: a disk went away halfway through" in text


# ---------------------------------------------------------------------------
# Failures during the run itself
# ---------------------------------------------------------------------------

def test_an_image_that_fails_to_convert_costs_its_field_and_says_so(
        theirs, tmp_path, monkeypatch):
    real = cv._imwrite

    def _flaky(path, array):
        if path.endswith(".tif") and "C02" in os.path.basename(path):
            pass
        return real(path, array)

    def _explode(path, array):
        raise OSError("no space left on device")

    plan = _plan(theirs)
    # fail the very first source: convert records it and carries on
    monkeypatch.setattr(cv, "_imwrite", _explode)
    result = fg.run_import(plan, str(tmp_path / "imported"))
    monkeypatch.setattr(cv, "_imwrite", real)

    assert not result.is_complete
    assert result.stacks == []
    assert any("channel image(s) were not converted" in w
               for w in result.warnings)
    assert "INCOMPLETE" in result.summary()


def test_a_field_with_no_merged_array_is_named(theirs, tmp_path, monkeypatch):
    import spacr.io as sio

    monkeypatch.setattr(sio, "_load_and_concatenate_arrays",
                        lambda *a, **k: None)
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"))
    assert result.merged == []
    assert sorted(w for w in result.warnings) == [
        "plate1_A01_1: no merged array was produced.",
        "plate1_A01_2: no merged array was produced."]


def test_the_join_view_is_only_created_when_both_halves_exist(tmp_path):
    db = str(tmp_path / "m.db")
    connection = sqlite3.connect(db)
    try:
        pd.DataFrame({"a": [1]}).to_sql("cell", connection, index=False)
    finally:
        connection.close()
    fg._write_view(db, "cell")               # no foreign_cell: nothing happens
    assert "cell_with_foreign" not in _tables(db)


def test_a_resolution_for_a_column_that_is_not_in_the_table_is_skipped(theirs):
    plan = _plan(theirs)
    plan.resolved = list(plan.resolved) + [fg.ResolvedColumn(
        mapping=fg.ColumnMap(source="ghost", target="foreign_ghost"),
        target="foreign_ghost", factor=1.0, calibrated=True, unit="",
        status="mapped")]
    frame = fg._foreign_frame(plan, plan.stems, "/merged")
    assert "foreign_ghost" not in frame.columns
    assert len(frame) == 4


def test_crops_skip_a_missing_array_and_a_label_in_no_mask(theirs, tmp_path):
    dst = str(tmp_path / "imported")
    plan = _plan(theirs)
    result = fg.run_import(plan, dst)

    frame = pd.DataFrame([
        {"file_name": "plate1_A01_1", "object_label": 1},
        {"file_name": "plate1_A01_1", "object_label": 999},   # in no mask
        {"file_name": "plate1_A01_9", "object_label": 1},     # no array
    ])
    written = fg._cut_crops(dst, plan, frame, "cell")
    assert len(written) == 1
    assert os.path.isfile(written[0])


def test_a_crop_limit_stops_early(theirs, tmp_path):
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"),
                           crops=True, crop_limit=1)
    assert len(result.crops) == 1


def test_a_missing_or_blank_cell_in_the_map_file_reads_as_empty(tmp_path):
    mapping = fg.ColumnMap.from_row({"source": " A ", "target": None,
                                     "unit_in": float("nan")})
    assert mapping.source == "A"
    assert mapping.target == ""
    assert mapping.unit_in == ""

    path = tmp_path / "m.csv"
    path.write_text("source,target,transform\n"
                    ",,,\n"                       # a blank row: skipped
                    "A,foreign_a\n",              # a short row: padded
                    encoding="utf-8")
    maps = fg.load_column_map(str(path))
    assert [m.source for m in maps] == ["A"]
    # a missing transform is the identity, not an empty transform
    assert maps[0].transform == "identity"


def test_a_mask_tree_shaped_like_the_image_tree_matches_exactly(tmp_path):
    """Same plate and well folders on both sides: no guessing needed."""
    images = tmp_path / "imgs" / "plateA" / "wellA"
    masks = tmp_path / "msks" / "plateA" / "wellA"
    for channel in (1, 2):
        _write_image(str(images / f"fov01_C{channel}.tif"), channel)
    _write_mask(str(masks / "fov01.tif"))
    table = tmp_path / "r.csv"
    pd.DataFrame([{"ImageNumber": "fov01_C1.tif", "ObjectNumber": 1,
                   "Area": 36.0}]).to_csv(table, index=False)

    plan = fg.plan_import(str(tmp_path / "imgs"),
                          {"cell": str(tmp_path / "msks")}, str(table),
                          layout="plate_well")
    assert plan.ok, fg.format_plan(plan)
    field = next(iter(plan.masks.fields.values()))["cell"]
    assert field.match == "exact"
    assert plan.join.rows_matched == 1


def test_a_blocking_problem_in_the_image_plan_is_shown_in_the_plan(theirs):
    """convert.py's own errors surface here, not just this module's."""
    tifffile.imwrite(os.path.join(theirs.images, "fov01_C1.tif"),
                     np.zeros((3, SIZE, SIZE), np.uint16),
                     metadata={"axes": "CYX"})
    plan = _plan(theirs)
    assert not plan.ok
    assert plan.images.errors
    text = fg.format_plan(plan)
    assert "BLOCKING PROBLEMS" in text
    assert "would be silently dropped" in text
    assert "Plan is NOT READY to run." in text


def test_the_result_counts_every_row_it_wrote(theirs, tmp_path):
    result = fg.run_import(_plan(theirs), str(tmp_path / "imported"))
    assert result.rows == {"cell": 4, "foreign_cell": 4}
    assert result.n_rows == 8
    assert result.n_fields == 2
