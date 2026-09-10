"""The identity layer's last untaken turns: filters, schema, crops, align.

Five modules that all answer the same question -- "which row is this object,
and what may be joined onto it" -- and the branches left in them are the ones
that only fire on a database or a frame that is missing a column.

What is pinned here:

* :mod:`spacr.filters` -- a relationships table with no ``parent_label``
  column (a crop is still attached to its own type), one with no
  ``object_label`` at all (nothing is attached, rather than matched blind),
  and a ``png_list`` whose crop type has no ``in_<type>`` flag on the
  bootstrap frame (the type-axis nulling cannot run, so a label-matched path
  survives).
* :mod:`spacr.metadata_resolution` -- pseudo-wells asked for only one of the
  two identity axes, in both directions.
* :mod:`spacr.schema` -- ``row_id`` on every one- and two-letter row label
  there is, which is the proof that its ``index is not None`` re-check can
  never fail.
* :mod:`spacr.crops` and :mod:`spacr.align` -- the two "the window might not
  overlap" guards, both of which sit after a computation that has already
  guaranteed an overlap. The invariant is asserted rather than the guard
  driven, because the guard cannot be driven; see the module comments.
"""

from __future__ import annotations

import os
import sqlite3
import string

import numpy as np
import pandas as pd
import pytest

from spacr import align, crops, filters, metadata_resolution, schema

BASE = dict(plateID="p1", rowID="r1", columnID="c1", fieldID="f1")


# ---------------------------------------------------------------------------
# filters: what a relationships table missing a column can still be joined on
# ---------------------------------------------------------------------------

def _database(tmp_path, *, crop_column="cell_id", relationships=None,
              name="measurements.db"):
    """A cell/nucleus database with one crop per cell.

    ``relationships`` -- when given -- is written as the STORED relationships
    table, which is what ``build_filters_from_relationships`` reads back
    verbatim. That is the only way to hand the joiner a frame whose columns
    it did not itself derive.
    """
    os.makedirs(str(tmp_path), exist_ok=True)
    path = str(tmp_path / name)
    con = sqlite3.connect(path)
    try:
        pd.DataFrame([{**BASE, "object_label": i, "cell_area": 10.0 * i}
                      for i in (1, 2)]).to_sql("cell", con, index=False)
        pd.DataFrame([{**BASE, "object_label": j, "cell_id": i,
                       "nucleus_area": 3.0 * j}
                      for i in (1, 2) for j in (1, 2)]
                     ).to_sql("nucleus", con, index=False)
        pd.DataFrame([{**BASE, crop_column: f"o{i}",
                       "png_path": f"/crops/{i}.png"} for i in (1, 2)]
                     ).to_sql("png_list", con, index=False)
        if relationships is not None:
            relationships.to_sql("relationships", con, index=False)
        con.commit()
    finally:
        con.close()
    return path


def test_a_relationships_table_with_no_parent_column_still_gets_its_own_crops(
        tmp_path):
    """No parent axis means no child join -- and the cells keep their crops.

    ``_attach_png_paths`` looks for ``parent_label``/``parent_type`` before it
    joins a child onto its parent's crop. A stored relationships table written
    by an older spaCR has neither, and the objects that ARE of the cropped
    type must still be given their own picture.
    """
    flat = pd.DataFrame([{**BASE, "object_label": i, "object_type": "cell"}
                         for i in (1, 2)]
                        + [{**BASE, "object_label": j, "object_type": "nucleus"}
                           for j in (1, 2)])
    without_parents = filters.build_filters_from_relationships(
        _database(tmp_path / "a", relationships=flat))

    assert list(without_parents.loc[
        without_parents["object_type"] == "cell", "png_path"]) == [
            "/crops/1.png", "/crops/2.png"]
    # No parent axis, so a nucleus cannot be told which cell it is in and
    # gets no picture at all rather than one matched on its bare label.
    assert without_parents.loc[
        without_parents["object_type"] == "nucleus", "png_path"].isna().all()

    # The same database with the parent columns present DOES carry the child
    # join -- so the assertion above is about the missing columns, not about
    # a join that never happens.
    with_parents = filters.build_filters_from_relationships(
        _database(tmp_path / "b"))
    nuclei = with_parents[with_parents["object_type"] == "nucleus"]
    assert sorted(nuclei["png_path"].dropna()) == [
        "/crops/1.png", "/crops/1.png", "/crops/2.png", "/crops/2.png"]


def test_a_relationships_table_with_no_object_label_carries_no_crop_paths(
        tmp_path):
    """Nothing shared to join on means no ``png_path`` column at all.

    ``key_columns`` reads column NAMES, so a relationships table that lost its
    ``object_label`` shares nothing with ``png_list``. Merging on the well
    alone would give every object in the field the same picture, so the
    column is not added.
    """
    keyless = pd.DataFrame([{"object_type": "cell", "cell_area": 1.0},
                            {"object_type": "cell", "cell_area": 2.0}])
    frame = filters.build_filters_from_relationships(
        _database(tmp_path / "a", relationships=keyless))

    assert "png_path" not in frame.columns
    assert list(frame["in_cell"]) == [1, 1]

    # Drive the present case on the same database shape: with the object
    # label back, the identical call does attach the paths.
    keyed = keyless.assign(object_label=[1, 2], **BASE)
    attached = filters.build_filters_from_relationships(
        _database(tmp_path / "b", relationships=keyed))
    assert list(attached["png_path"]) == ["/crops/1.png", "/crops/2.png"]


def test_a_crop_type_with_no_membership_flag_cannot_be_nulled_by_type(
        tmp_path, monkeypatch):
    """``in_pathogen`` is not on the frame, so the type filter cannot run.

    The bootstrap frame has no ``object_type`` axis; the ``in_<table>`` flag
    stands in for it. When ``png_list`` was cropped by an object type this
    database has no table for, there is no flag to test and the path matched
    on the bare label stays. Driven through the object-table fallback, which
    is the only route to that frame.
    """
    def _explode(_path):
        raise RuntimeError("relationships unavailable")

    monkeypatch.setattr(filters, "build_filters_from_relationships", _explode)

    # png_list keyed by pathogen_id, but this database has no pathogen table.
    unflagged = filters.build_filters_frame(
        _database(tmp_path / "a", crop_column="pathogen_id"))
    assert filters.png_crop_type(
        str(tmp_path / "a" / "measurements.db")) == "pathogen"
    assert "in_pathogen" not in unflagged.columns
    assert sorted(unflagged["png_path"].dropna()) == [
        "/crops/1.png", "/crops/2.png"]

    # Cropped by cell, and `in_cell` IS on the frame: object 1 and 2 appear in
    # both tables so they keep their path, and nothing else can claim one.
    flagged = filters.build_filters_frame(_database(tmp_path / "b"))
    assert "in_cell" in flagged.columns
    assert flagged.loc[flagged["in_cell"] != 1, "png_path"].isna().all()


# ---------------------------------------------------------------------------
# metadata_resolution: pseudo-wells for one axis at a time
# ---------------------------------------------------------------------------

def _resolve(frame):
    return metadata_resolution.resolve_metadata_columns(
        frame, [schema.ROW_KEY, schema.COLUMN_KEY],
        pseudo_source="sample", allow_pseudo=True)


def test_pseudo_wells_fill_only_the_identity_axis_that_is_missing():
    """A table that already has one of the two keeps the one it has.

    Both directions in one test: overwriting the axis the file supplied would
    silently remap the plate, and the audit trail must still name both
    coordinates for every distinct source value either way.
    """
    have_row = pd.DataFrame({"sample": ["a", "b", "a"],
                             schema.ROW_KEY: ["r9", "r9", "r9"]})
    result = _resolve(have_row)
    assert list(result.frame[schema.ROW_KEY]) == ["r9", "r9", "r9"]
    assert list(result.frame[schema.COLUMN_KEY]) == ["c1", "c2", "c1"]

    have_column = pd.DataFrame({"sample": ["a", "b", "a"],
                                schema.COLUMN_KEY: ["c7", "c7", "c7"]})
    mirrored = _resolve(have_column)
    assert list(mirrored.frame[schema.COLUMN_KEY]) == ["c7", "c7", "c7"]
    assert list(mirrored.frame[schema.ROW_KEY]) == ["r1", "r1", "r1"]

    # The audit is the record of what the pseudo-well map decided, and it
    # names both axes even for the one that was not written.
    assert [entry[schema.COLUMN_KEY] for entry in mirrored.pseudo_map] == \
        ["c1", "c2"]


# ---------------------------------------------------------------------------
# schema.row_id: the re-check that cannot fail
# ---------------------------------------------------------------------------

def test_every_row_label_row_id_accepts_parses_to_an_index():
    """UNREACHABLE ARC, PROVEN: ``row_id`` line 765's False side.

    ``row_id`` only calls ``row_index_from_letters`` behind
    ``_ROW_ONLY.match(letters)``, which is ``^[A-Za-z]{1,2}$``.
    ``row_index_from_letters`` returns None for a non-string, for an empty
    string, and for any character outside ``A``-``Z`` after ``.upper()`` --
    none of which a one- or two-letter ASCII string can be -- and its
    bijective base-26 total is at least 1, so ``total or None`` is an int.
    The ``if index is not None`` guard therefore always holds and the fall
    through to ``_prefixed_id`` is dead for these inputs.

    Asserted here over the whole domain the regex admits: 702 labels, every
    one of which round-trips.
    """
    letters = list(string.ascii_uppercase) + [
        a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]
    assert len(letters) == 702

    for label in letters:
        index = schema.row_index_from_letters(label)
        assert isinstance(index, int) and index >= 1, label
        assert schema.row_id(label) == f"r{index}"
        assert schema.row_id(label.lower()) == f"r{index}"
        assert schema.letters_from_row_index(index) == label

    # The fall-through is live for tokens the regex does NOT admit, which is
    # what keeps the guard from being the only thing standing between a bad
    # label and _prefixed_id.
    assert schema.row_index_from_letters("A1") is None
    assert schema.row_id("r12") == "r12"


def test_the_object_role_registry_validates_itself_at_import():
    """UNREACHABLE LINES, PROVEN: ``schema`` lines 285-289.

    The registry self-check runs once, at import, over the literal
    :data:`schema.OBJECT_TYPES` defined twenty lines above it in the same
    module. Its raise can only execute if that literal is edited to contain a
    digit, an empty string, or the key separator -- there is no input any
    caller can supply, and no reload can reach it because ``OBJECT_TYPES`` is
    built in the same file.

    What can be asserted is the property the check enforces, and that it is
    the property the id grammar actually needs: a digit in a role makes
    ``split_object_id`` ambiguous.
    """
    for role in schema.OBJECT_TYPES:
        assert role, "an empty role would make object_type_prefix ambiguous"
        assert schema.KEY_SEPARATOR not in role
        assert not any(character.isdigit() for character in role)

    # WHY the check exists: role + label is concatenated with no separator, so
    # every registered role has to be recoverable from the front of the id.
    for role in schema.OBJECT_TYPES:
        assert schema.split_object_id(f"{role}17") == (role, "17")


# ---------------------------------------------------------------------------
# crops: the region window that is always there
# ---------------------------------------------------------------------------

def _merged(tmp_path, mask, shape=(40, 40)):
    """A merged .npy: two intensity planes then a cell mask plane."""
    # A gradient, not a constant: the normaliser takes its percentiles over
    # the non-zero pixels, so a flat field would come back flat and hide
    # whichever pixels the region mask actually zeroed.
    intensity = (np.arange(shape[0] * shape[1], dtype=np.uint32)
                 .reshape(shape) * 7 + 500).astype(np.uint16)
    stack = np.stack([intensity, intensity // 2 + 1, mask.astype(np.uint16)],
                     axis=-1)
    path = str(tmp_path / "plate1_A01_F001.npy")
    np.save(path, stack)
    return path


def test_a_region_and_its_window_always_overlap(tmp_path):
    """UNREACHABLE ARCS, PROVEN: ``crops`` lines 1103 and 1107, False sides.

    ``_crop_from_field`` re-checks ``region is not None`` and then checks that
    the crop window overlaps the region bounds. Neither can fail:

    * ``_region_for`` has a single ``return`` and every path through it binds
      ``region`` to a boolean array -- ``np.ones`` on the bounding-box branch,
      ``window == label`` (which raises ``LabelMissing`` when it is empty) on
      the outline branch, and ``_binary_dilate``'s array after dilation. It
      never returns None, whatever its docstring allows for.
    * the window is centred on ``round(centroid)``, and the centroid is
      computed from pixels inside ``(ry0, ry1, rx0, rx1)`` on every branch, so
      ``ry0 <= round(cy) <= ry1 - 1`` and the window contains that pixel.
      ``max(wy0, ry0) <= round(cy) < min(wy1, ry1)``, so the overlap is at
      least one pixel wide on both axes.

    Both halves are asserted directly here, over the shapes most likely to
    break them: an object hard against the field edge, and a C-shape whose
    centroid is not on the object at all.
    """
    mask = np.zeros((40, 40), dtype=np.uint16)
    mask[0:6, 0:6] = 1                       # flush with the corner
    mask[20:34, 10:14] = 2                   # the two arms of a C ...
    mask[20:34, 24:28] = 2
    mask[30:34, 14:24] = 2                   # ... joined at the bottom
    path = _merged(tmp_path, mask)

    field = crops.open_merged_field(path, {"cell": 2})
    for label in (1, 2):
        for use_bbox in (False, True):
            spec = crops.CropSpec(merged_path=path, object_type="cell",
                                  label=label, channels=(0, 1), size=(8, 8),
                                  mask_dims={"cell": 2},
                                  use_bounding_box=use_bbox)
            centroid, (ry0, ry1, rx0, rx1), region = crops._region_for(
                field, spec)
            assert region is not None, (label, use_bbox)

            cy, cx = int(centroid[0]), int(centroid[1])
            assert ry0 <= cy < ry1 and rx0 <= cx < rx1, (label, use_bbox)
            wy0, wx0 = cy - 8 // 2, cx - 8 // 2
            assert min(wy0 + 8, ry1) > max(wy0, ry0)
            assert min(wx0 + 8, rx1) > max(wx0, rx0)

    # The C-shape's centroid really is off the object, which is the case that
    # would break the argument if the window were centred on a mask pixel.
    spec = crops.CropSpec(merged_path=path, object_type="cell", label=2,
                          channels=(0, 1), size=(8, 8), mask_dims={"cell": 2})
    centroid, _bounds, _region = crops._region_for(field, spec)
    assert mask[int(centroid[0]), int(centroid[1])] == 0

    # And the crop that comes out is a real one. The corner object sits at
    # (0..6, 0..6) with its centroid at (2, 2), so the 8x8 window starts at
    # (-2, -2): the first two rows and columns are off the field and come
    # back as zero padding, while the overlap the guard re-checks is what
    # carries the object's pixels.
    crop = crops.extract_crop(path, "cell", 1, channels=(0, 1), size=(8, 8),
                              mask_dims={"cell": 2})
    # two requested channels, padded to RGB with a zero third plane exactly
    # as the PNG path does
    assert crop.shape == (8, 8, 3)
    assert not crop[:2, :, 0].any() and not crop[:, :2, 0].any()
    # 35 of the 36 overlap pixels carry signal; the 36th is the dimmest, and
    # the 0-100 percentile stretch puts it at the bottom of the range.
    assert np.count_nonzero(crop[2:, 2:, 0]) == 35
    assert crop[2:, 2:, 0].max() == np.iinfo(np.uint16).max
    assert not crop[..., 2].any()


# ---------------------------------------------------------------------------
# align: the overlap that is already known to be positive
# ---------------------------------------------------------------------------

def test_every_overlap_window_align_returns_is_at_least_one_pixel():
    """UNREACHABLE ARC, PROVEN: ``align`` line 1773's False side.

    ``_feather_width`` skips a pair whose ``_overlap_windows`` is None and
    then re-checks ``span > 0``. ``_overlap_windows`` returns None exactly
    when ``ay1 <= ay0 or ax1 <= ax0``, so any window it does return has both
    extents strictly positive and ``span = min(...) >= 1``.

    Asserted over a sweep of offsets that covers touching, overlapping,
    disjoint and fully-contained tiles.
    """
    tile_a = align.Tile(path="a.tif", index=0, shape=(16, 12, 1))
    tile_b = align.Tile(path="b.tif", index=1, shape=(16, 12, 1))

    seen_none = seen_window = 0
    for dy in range(-20, 21, 2):
        for dx in range(-16, 17, 2):
            windows = align._overlap_windows(tile_a, tile_b, dy, dx)
            if windows is None:
                seen_none += 1
                continue
            seen_window += 1
            (ay0, ay1, ax0, ax1), (by0, by1, bx0, bx1) = windows
            assert min(ay1 - ay0, ax1 - ax0) >= 1, (dy, dx)
            # the two rectangles are the same size, which is what makes the
            # feather ramp comparable between the tiles
            assert (ay1 - ay0, ax1 - ax0) == (by1 - by0, bx1 - bx0)

    assert seen_none and seen_window, "the sweep must cover both answers"


def test_a_coordinate_table_written_to_a_bare_filename_lands_in_the_cwd(
        tmp_path, monkeypatch):
    """UNREACHABLE ARC, PROVEN: ``align`` line 2329's False side.

    ``parent = os.path.dirname(os.path.abspath(db_path))`` is empty only if
    ``abspath`` returned a path with no separator in it, which it never does:
    it always returns an absolute path, so ``dirname`` is at worst ``"/"``.
    The ``if parent:`` guard therefore always holds.

    A bare relative filename is the input that would fire it if anything
    could, so that is what this writes -- and the rows land beside the cwd.
    """
    tile = align.Tile(path="a.tif", index=0, plate="plate1", well="A01",
                      field=1, shape=(4, 4, 1))
    plan = align.AlignPlan(tiles=[tile],
                           placements=[align.Placement(tile=tile)],
                           canvas_shape=(4, 4, 1))

    monkeypatch.chdir(tmp_path)
    assert os.path.dirname(os.path.abspath("coords.db")) == str(tmp_path)

    written = align.save_coordinates(plan, "coords.db")
    assert written == 1
    assert os.path.isfile(tmp_path / "coords.db")
    con = sqlite3.connect(str(tmp_path / "coords.db"))
    try:
        stored = pd.read_sql_query(
            f'SELECT * FROM "{align.ALIGN_TABLE}"', con)
    finally:
        con.close()
    assert len(stored) == 1
    assert stored.loc[0, "field"] == 1
