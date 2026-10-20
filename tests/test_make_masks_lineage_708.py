"""Synthetic uint32 masks retain extensible lineage inside each TIFF."""

from __future__ import annotations

import copy
import json
import os
from uuid import UUID

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine
from spacr.tiff_io import write_tiff


def _field(tmp_path, name="frame.tif"):
    image = np.arange(144, dtype=np.uint16).reshape(12, 12)
    write_tiff(tmp_path / name, image)
    return str(tmp_path / name), image


def _mapped_cell(state, name, label, frame):
    state, identity = engine.new_lineage_cell(state, name, label, frame)
    return engine.assign_lineage_label(state, frame, label, identity), identity


def test_sparse_uint32_editing_lookup_filters_and_overlay_allocate_observed_ids_only():
    mask = np.zeros((12, 12), np.uint32)
    mask[2:7, 2:7] = 2**31 + 11
    mask[4, 4] = 0
    mask[9, 9] = engine.MAX_MASK_LABEL
    before = mask.copy()
    image = np.arange(mask.size, dtype=np.uint16).reshape(mask.shape)
    lookup = engine.ObjectLookup(mask, image, preserve_ids=True)
    assert lookup.at(3, 3).label == 2**31 + 11
    assert lookup.measure(engine.MAX_MASK_LABEL)[0] == 1
    assert len(lookup._boxes) == 2
    filled = engine.fill_holes(mask, preserve_ids=True)
    assert filled[4, 4] == 2**31 + 11
    shrunk = engine.shrink_objects(filled)
    assert shrunk[4, 4] == 2**31 + 11
    grown = engine.dilate_objects(mask)
    assert set(np.unique(grown)) == set(np.unique(mask))
    filtered, removed = engine.apply_filters(mask, image,
        [{"property": "area", "min": 2}], preserve_ids=True)
    assert [item.label for item in removed] == [engine.MAX_MASK_LABEL]
    assert filtered[9, 9] == 0
    assert filtered[3, 3] == 2**31 + 11
    overlay = engine.overlay_mask(image, mask)
    assert overlay.shape == (12, 12, 3)
    assert overlay.dtype == np.uint8
    np.testing.assert_array_equal(mask, before)


def test_division_display_name_reuse_is_not_a_self_parent_cycle(tmp_path):
    first, _ = _field(tmp_path)
    second = str(tmp_path / "next.tif")
    state, founder = _mapped_cell(None, "1", 1, first)
    divided, daughters = engine.divide_lineage_cell(state, founder, second,
                                                    ["1", "1d"], [1, 2**31 + 7])
    assert founder not in daughters and daughters[0] != daughters[1]
    assert divided["cells"][daughters[0]]["name"] == "1"
    assert all(divided["cells"][child]["parent_uuid"] == founder for child in daughters)
    assert divided["frames"][first]["1"] == founder
    assert divided["frames"][second] == {}
    assert state["cells"][founder]["children"] == []
    assert engine.validate_lineage(divided) == divided


def test_snapshot_merge_retains_new_name_and_new_division(tmp_path):
    frame, _ = _field(tmp_path)
    initial, founder = _mapped_cell(None, "1", 1, frame)
    divided, daughters = engine.divide_lineage_cell(initial, founder,
        str(tmp_path / "next.tif"), ["1", "1d"], [1, 2])
    renamed = engine.rename_lineage_cell(divided, daughters[0], "1a")
    merged = engine.merge_lineage_states(renamed, divided, frame=frame)
    assert merged["cells"][daughters[0]]["name"] == "1a"
    merged = engine.merge_lineage_states(merged, initial, frame=frame)
    assert merged["cells"][founder]["children"] == list(daughters)
    conflicting = copy.deepcopy(divided)
    conflicting["cells"][daughters[0]]["born_frame"] = "elsewhere"
    conflicting["cells"][daughters[0]].pop("born_frame_id")
    with pytest.raises(ValueError, match="ancestry"):
        engine.merge_lineage_states(merged, conflicting)


def test_mapping_removal_and_collision_are_copy_on_write(tmp_path):
    frame, _ = _field(tmp_path)
    state, one = _mapped_cell(None, "first", 1, frame)
    state, two = _mapped_cell(state, "second", 2, frame)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="already belongs"):
        engine.assign_lineage_label(state, frame, 1, two)
    assert state == before
    removed = engine.assign_lineage_label(state, frame, 0, old_label=1)
    assert removed["frames"][frame] == {"2": two}
    assert one in removed["cells"]


def test_tiff_is_self_contained_and_ordinary_resave_retains_width_and_lineage(tmp_path):
    frame, image = _field(tmp_path)
    state, founder = _mapped_cell(None, "1", 2**31 + 9, frame)
    state, unplaced = _mapped_cell(state, "unplaced", 2, frame)
    mask = np.zeros(image.shape, np.uint32)
    mask[2:6, 2:6] = 2**31 + 9
    mask[9, 9] = 2**31 + 9
    path = engine.save_mask(str(tmp_path), "frame.tif", mask, lineage=state)
    assert sorted(item.name for item in (tmp_path / "masks").iterdir()) == ["frame.tif"]
    metadata = engine.load_mask_metadata(str(tmp_path), "frame.tif")
    assert metadata["preserve_ids"]
    assert metadata["lineage"]["frames"][frame] == {str(2**31 + 9): founder}
    assert unplaced in metadata["lineage"]["cells"]
    _, reopened = engine.load_image_and_mask(str(tmp_path), "frame.tif")
    np.testing.assert_array_equal(reopened, mask)
    assert reopened.dtype == np.uint32
    engine.save_mask(str(tmp_path), "frame.tif", reopened)
    assert engine.load_mask_lineage(str(tmp_path), "frame.tif") == metadata["lineage"]
    with tifffile.TiffFile(path) as handle:
        standalone = json.loads(handle.pages[0].description)["spacr_mask"]
        assert standalone == engine.load_mask_metadata(str(tmp_path), "frame.tif")


def test_empty_time_graph_forces_uint32_without_requiring_any_cell(tmp_path):
    _, image = _field(tmp_path)
    path = engine.save_mask(str(tmp_path), "frame.tif", np.zeros(image.shape, np.uint16),
                            lineage=engine.validate_lineage())
    assert tifffile.imread(path).dtype == np.uint32
    assert engine.load_mask_metadata(str(tmp_path), "frame.tif")["preserve_ids"]


@pytest.mark.parametrize("mask_suffix", [".tif", ".png"])
@pytest.mark.parametrize("bigtiff", [False, True])
@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_png_source_finds_metadata_in_the_selected_tiff_content(
        tmp_path, mask_suffix, bigtiff, byteorder):
    masks = tmp_path / "masks"
    masks.mkdir()
    path = masks / ("frame" + mask_suffix)
    metadata = {"schema": "spacr.mask/v1", "preserve_ids": True}
    write_tiff(path, np.ones((4, 4), np.uint32), metadata=None,
               description=json.dumps({"spacr_mask": metadata}),
               bigtiff=bigtiff, byteorder=byteorder)
    before = path.read_bytes()
    assert engine.load_mask_metadata(str(tmp_path), "frame.png") == metadata
    assert path.read_bytes() == before


@pytest.mark.parametrize("mask_name, data", [
    ("frame.tif", b"not a TIFF"),
    ("frame.tif", b"II+\x00\x04\x00\x00\x00"),
    ("frame.png", b"II+\x00\x04\x00\x00\x00"),
])
def test_damaged_selected_tiff_metadata_is_not_treated_as_absent(
        tmp_path, mask_name, data):
    masks = tmp_path / "masks"
    masks.mkdir()
    path = masks / mask_name
    path.write_bytes(data)
    with pytest.raises(tifffile.TiffFileError):
        engine.load_mask_metadata(str(tmp_path), "frame.png")
    assert path.read_bytes() == data


def test_malformed_embedded_metadata_in_a_png_named_tiff_is_refused(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    path = masks / "frame.png"
    write_tiff(path, np.ones((4, 4), np.uint16), metadata=None,
               description='{"spacr_mask": broken}')
    with pytest.raises(ValueError, match="embedded mask metadata"):
        engine.load_mask_metadata(str(tmp_path), "frame.png")


def test_atomic_write_failure_and_overflow_keep_previous_tiff(tmp_path, monkeypatch):
    _, image = _field(tmp_path)
    mask = np.ones(image.shape, np.uint32)
    path = engine.save_mask(str(tmp_path), "frame.tif", mask, lineage=engine.validate_lineage())
    before = open(path, "rb").read()
    for invalid in (np.full(image.shape, -1, np.int64),
                    np.full(image.shape, 2**32, np.uint64)):
        with pytest.raises(ValueError):
            engine.save_mask(str(tmp_path), "frame.tif", invalid)
        assert open(path, "rb").read() == before
    def fail_after_open(destination, array, **kwargs):
        with open(destination, "wb") as handle:
            handle.write(b"partial TIFF")
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(engine, "write_tiff", fail_after_open)
    with pytest.raises(OSError, match="synthetic"):
        engine.save_mask(str(tmp_path), "frame.tif", mask)
    assert open(path, "rb").read() == before
    assert not list((tmp_path / "masks").glob(".spacr-mask-*"))


def test_large_population_and_deep_ancestry_roundtrip_with_tiny_pixels(tmp_path):
    frame, _ = _field(tmp_path)
    state = engine.validate_lineage()
    cells = state["cells"]
    for index in range(10000):
        identity = str(UUID(int=index + 1))
        cells[identity] = {"name": str(index + 1), "parent_uuid": None,
            "born_frame": frame, "division_frame": None, "children": [],
            "label": index + 1, "revision": 1}
    parent = str(UUID(int=1))
    for depth in range(256):
        children = [str(UUID(int=10001 + 2 * depth + side)) for side in range(2)]
        division_frame = str(tmp_path / f"frame-{depth + 1}.tif")
        cells[parent].update(children=children, division_frame=division_frame)
        for side, child in enumerate(children):
            cells[child] = {"name": f"branch-{depth}-{side}", "parent_uuid": parent,
                "born_frame": division_frame, "division_frame": None, "children": [],
                "label": 2**31 + 2 * depth + side, "revision": 2}
        parent = children[0]
    state["frames"][frame] = {str(engine.MAX_MASK_LABEL): parent}
    state["frame_revisions"][frame] = 3
    mask = np.array([[0, engine.MAX_MASK_LABEL], [0, 0]], np.uint32)
    write_tiff(tmp_path / "frame.tif", np.zeros(mask.shape, np.uint16))
    engine.save_mask(str(tmp_path), "frame.tif", mask, lineage=state)
    reopened = engine.load_mask_lineage(str(tmp_path), "frame.tif")
    assert reopened == engine.validate_lineage(state)
    assert len(reopened["cells"]) == 10512
    count = 0
    while reopened["cells"][parent]["parent_uuid"] is not None:
        parent = reopened["cells"][parent]["parent_uuid"]
        count += 1
    assert count == 256


def test_invalid_cycle_or_unknown_schema_is_refused(tmp_path):
    frame, _ = _field(tmp_path)
    state, founder = _mapped_cell(None, "1", 1, frame)
    state["cells"][founder]["parent_uuid"] = founder
    with pytest.raises(ValueError, match="metadata"):
        engine.validate_lineage(state)
    with pytest.raises(ValueError, match="metadata"):
        engine.validate_lineage({"schema": "future/99", "cells": {}, "frames": {}})


def test_tracking_permuted_segmentation_ids_preserves_both_objects_and_identity(tmp_path):
    first, image = _field(tmp_path)
    second, _ = _field(tmp_path, "next.tif")
    state, identity = _mapped_cell(None, "1", 1, first)
    mask = np.zeros(image.shape, np.uint16)
    mask[2:5, 2:5] = 1
    mask[7:10, 7:10] = 2
    before = mask.copy()
    tracked, state, old, local = engine.assign_tracked_object(
        mask, 8, 8, state, second, identity, preferred_label=1)
    assert (old, local) == (2, 2)
    np.testing.assert_array_equal(tracked, before)
    assert state["frames"][second] == {"2": identity}
    again, unchanged, _, _ = engine.assign_tracked_object(tracked, 8, 8, state, second, identity)
    assert unchanged == state
    np.testing.assert_array_equal(again, tracked)
    with pytest.raises(ValueError, match="already labels another object"):
        engine.assign_tracked_object(tracked, 3, 3, state, second, identity)
    engine.save_mask(str(tmp_path), "next.tif", tracked, lineage=state)
    reopened = engine.load_mask_lineage(str(tmp_path), "next.tif")
    assert reopened["frames"][first]["1"] == reopened["frames"][second]["2"]


def test_tracking_refuses_one_identity_on_two_objects_in_a_frame(tmp_path):
    frame, image = _field(tmp_path)
    state, first = _mapped_cell(None, "first", 1, frame)
    state, second = _mapped_cell(state, "second", 2, frame)
    mask = np.zeros(image.shape, np.uint32)
    mask[2:5, 2:5] = 1
    mask[7:10, 7:10] = 2
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="first already labels another object in this frame"):
        engine.assign_tracked_object(mask, 8, 8, state, frame, first)
    assert state == before
    assert state["frames"][frame]["2"] == second


def _continuation_movie(tmp_path):
    frames = [_field(tmp_path, f"t{index}.tif")[0] for index in range(4)]
    state, early = _mapped_cell(None, "107", 107, frames[0])
    state, mother = _mapped_cell(state, "61", 61, frames[1])
    state, (kept, sister) = engine.divide_lineage_cell(
        state, mother, frames[2], ["89", "90"], [89, 90])
    for frame in frames[2:]:
        state = engine.assign_lineage_label(state, frame, 89, kept)
        state = engine.assign_lineage_label(state, frame, 90, sister)
    masks = []
    for local in ((107,), (61,), (89, 90), (89, 90)):
        mask = np.zeros((12, 12), np.uint32)
        mask[1:5, 1:5] = local[0]
        if len(local) > 1:
            mask[7:11, 7:11] = local[1]
        masks.append(mask)
    return frames, masks, state, (early, mother, kept, sister)


def test_selected_mother_continues_through_a_hovered_daughter(tmp_path):
    frames, masks, state, (early, mother, kept, sister) = _continuation_movie(tmp_path)
    tracked, result, old, local = engine.assign_tracked_object(
        masks[2], 2, 2, state, frames[2], mother)
    assert (old, local) == (89, 61)
    assert int(tracked[2, 2]) == 61 and int(tracked[8, 8]) == 90
    cells = result["cells"]
    assert result["frames"][frames[2]] == {"61": kept, "90": sister}
    assert (cells[kept]["name"], cells[kept]["label"]) == ("61", 61)
    assert cells[mother]["children"] == [kept, sister]
    assert cells[sister]["name"] == "90" and cells[sister]["parent_uuid"] == mother
    following, result, old, local = engine.assign_tracked_object(
        masks[3], 2, 2, result, frames[3], kept)
    assert (old, local) == (89, 61)
    assert result["frames"][frames[3]] == {"61": kept, "90": sister}
    with pytest.raises(ValueError, match="61 already labels another object in this frame"):
        engine.assign_tracked_object(following, 8, 8, result, frames[3], kept)
    back, result, old, local = engine.assign_tracked_object(
        masks[1], 2, 2, result, frames[1], kept)
    assert (old, local) == (61, 61)
    assert result["frames"][frames[1]] == {"61": mother}
    engine.save_mask(str(tmp_path), "t2.tif", tracked, lineage=result)
    reopened = engine.load_mask_lineage(str(tmp_path), "t2.tif")
    assert reopened["cells"][mother]["children"] == [kept, sister]
    assert reopened["cells"][kept]["name"] == "61"
    assert reopened["frames"][frames[2]] == {"61": kept, "90": sister}


def test_a_second_daughter_cannot_also_continue_the_mother(tmp_path):
    frames, masks, state, (early, mother, kept, sister) = _continuation_movie(tmp_path)
    _, result, _, _ = engine.assign_tracked_object(masks[2], 2, 2, state, frames[2], mother)
    before = copy.deepcopy(result)
    with pytest.raises(ValueError, match="61 already continues through its other daughter"):
        engine.assign_tracked_object(masks[3], 8, 8, result, frames[3], mother)
    assert result == before
    with pytest.raises(ValueError, match="61 already continues through its other daughter"):
        engine.assign_tracked_object(masks[1], 2, 2, result, frames[1], sister)


def test_an_unrelated_cell_left_without_frames_is_merged_and_restored_by_undo(tmp_path):
    frames, masks, state, (early, mother, kept, sister) = _continuation_movie(tmp_path)
    tracked, result, old, local = engine.assign_tracked_object(
        masks[0], 2, 2, state, frames[0], mother)
    assert (old, local) == (107, 61) and int(tracked[2, 2]) == 61
    assert early not in result["cells"] and early in result["deleted_cells"]
    assert result["frames"][frames[0]] == {"61": mother}
    undone = engine.restore_lineage_state(state, result)
    assert undone["cells"][early]["name"] == "107"
    assert undone["frames"][frames[0]] == {"107": early}
    _, kept_cell, _, _ = engine.assign_tracked_object(masks[2], 8, 8, state, frames[2], early)
    assert sister in kept_cell["cells"] and kept_cell["frames"][frames[2]][str(107)] == early


def test_undo_division_and_rename_survive_saved_older_snapshots_and_redo(tmp_path):
    frame, image = _field(tmp_path)
    next_frame = str(tmp_path / "next.tif")
    initial, founder = _mapped_cell(None, "1", 1, frame)
    divided, daughters = engine.divide_lineage_cell(initial, founder, next_frame,
                                                    ["1", "1d"], [1, 2])
    undone = engine.restore_lineage_state(initial, divided)
    mask = np.ones(image.shape, np.uint32)
    engine.save_mask(str(tmp_path), "frame.tif", mask, lineage=undone)
    reopened = engine.load_mask_lineage(str(tmp_path), "frame.tif")
    merged = engine.merge_lineage_states(reopened, divided, frame=next_frame)
    assert merged["cells"][founder]["children"] == []
    assert not any(child in merged["cells"] for child in daughters)
    assert merged["frames"][next_frame] == {}
    redone = engine.restore_lineage_state(divided, merged)
    assert redone["cells"][founder]["children"] == list(daughters)
    assert all(child in redone["cells"] for child in daughters)
    renamed = engine.rename_lineage_cell(redone, daughters[0], "1a")
    name_undone = engine.restore_lineage_state(redone, renamed)
    merged = engine.merge_lineage_states(name_undone, renamed)
    assert merged["cells"][daughters[0]]["name"] == "1"


def test_new_cell_does_not_bind_an_unrelated_object_with_its_preferred_id(tmp_path):
    frame, image = _field(tmp_path)
    state, identity = engine.new_lineage_cell(None, "1", 1, frame)
    assert state["frames"].get(frame, {}) == {}
    mask = np.zeros(image.shape, np.uint32)
    mask[2:5, 2:5] = 1
    mask[7:10, 7:10] = 5
    changed, state, old, local = engine.assign_tracked_object(mask, 8, 8, state, frame, identity)
    assert (old, local) == (5, 5)
    np.testing.assert_array_equal(changed, mask)
    assert state["frames"][frame] == {"5": identity}
    divided, children = engine.divide_lineage_cell(state, identity, frame, ["1", "1d"], [1, 2])
    assert divided["frames"][frame] == {}
    assert all(child in divided["cells"] for child in children)


def test_preferred_numeric_edit_relabels_existing_track_but_repeated_hover_is_noop(tmp_path):
    frame, image = _field(tmp_path)
    state, identity = _mapped_cell(None, "name", 321, frame)
    mask = np.full(image.shape, 321, np.uint32)
    changed, state, old, local = engine.assign_tracked_object(
        mask, 2, 2, state, frame, identity, preferred_label=654)
    assert (old, local) == (321, 654)
    assert np.all(changed == 654)
    assert state["frames"][frame] == {"654": identity}
    _, unchanged, _, _ = engine.assign_tracked_object(changed, 2, 2, state, frame, identity)
    assert unchanged == state


def test_pruned_mapping_beats_an_older_tiff_snapshot(tmp_path):
    frame, image = _field(tmp_path)
    original, identity = _mapped_cell(None, "1", 1, frame)
    path = engine.save_mask(str(tmp_path), "frame.tif", np.zeros(image.shape, np.uint32),
                            lineage=original)
    pruned = engine.load_mask_lineage(str(tmp_path), "frame.tif")
    merged = engine.merge_lineage_states(pruned, original, frame=frame)
    assert merged["frames"][frame] == {}
    assert identity in merged["cells"]
    assert tifffile.imread(path).dtype == np.uint32


def test_directory_move_preserves_frame_uuid_mapping_and_immutable_ancestry(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    first, image = _field(original)
    second, _ = _field(original, "next.tif")
    state, parent = _mapped_cell(None, "1", 1, first)
    state, children = engine.divide_lineage_cell(state, parent, second, ["1", "1d"], [1, 2])
    state = engine.assign_lineage_label(state, second, 1, children[0])
    mask = np.ones(image.shape, np.uint32)
    for name in ("frame.tif", "next.tif"):
        engine.save_mask(str(original), name, mask, lineage=state)
    moved = tmp_path / "renamed sequence"
    original.rename(moved)
    assert not original.exists()
    reopened = engine.load_mask_lineage(str(moved), "frame.tif")
    next_state = engine.load_mask_lineage(str(moved), "next.tif")
    merged = engine.merge_lineage_states(reopened, next_state, frame=str(moved / "next.tif"))
    assert merged["frames"][str(moved / "frame.tif")]["1"] == parent
    assert merged["frames"][str(moved / "next.tif")]["1"] == children[0]
    assert merged["cells"][children[0]]["born_frame_id"] == state["cells"][children[0]]["born_frame_id"]
    assert merged["cells"][parent]["division_frame_id"] == state["cells"][parent]["division_frame_id"]
    assert all(str(original) not in frame for frame in merged["frame_ids"])
    _, pixels = engine.load_image_and_mask(str(moved), "next.tif")
    np.testing.assert_array_equal(pixels, mask)


def test_windows_frame_aliases_reopen_on_linux_without_cwd_prefix(tmp_path):
    frame, image = _field(tmp_path)
    state, identity = _mapped_cell(None, "1", 1, frame)
    windows = r"C:\acquisition\frame.tif"
    state = engine._remap_lineage_frames(state, {frame: windows})
    target = tmp_path / "masks/frame.tif"
    target.parent.mkdir()
    metadata = {"schema": "spacr.mask/v1", "preserve_ids": True,
                "frame": windows, "lineage": state}
    write_tiff(target, np.ones(image.shape, np.uint32), metadata=None,
               description=json.dumps({"spacr_mask": metadata}))
    reopened = engine.load_mask_lineage(str(tmp_path), "frame.tif")
    assert reopened["frames"] == {frame: {"1": identity}}
    assert reopened["cells"][identity]["born_frame"] == frame
    assert reopened["cells"][identity]["born_frame_id"] == state["cells"][identity]["born_frame_id"]
    # Nothing of the Windows alias survives, in any spelling or under a cwd
    # prefix. Named by its folder, because on Windows the real frame's own
    # path legitimately starts with a drive letter too.
    assert not any("acquisition" in path for path in reopened["frame_ids"])
    if os.name != "nt":
        assert not any("C:" in path for path in reopened["frame_ids"])


def test_copied_standalone_mask_recovers_names_and_ids_without_source_or_sidecar(tmp_path):
    import shutil

    original = tmp_path / "original"
    original.mkdir()
    frame, image = _field(original)
    state, identity = _mapped_cell(None, "1ad", 2**31 + 8, frame)
    mask = np.full(image.shape, 2**31 + 8, np.uint32)
    saved = engine.save_mask(str(original), "frame.tif", mask, lineage=state)
    standalone = tmp_path / "standalone.tif"
    shutil.copy2(saved, standalone)
    shutil.rmtree(original)
    metadata = engine.load_mask_metadata(str(tmp_path), "standalone.tif", masks_dir=str(tmp_path))
    assert metadata["lineage"]["cells"][identity]["name"] == "1ad"
    assert metadata["lineage"]["frames"][str(standalone)] == {str(2**31 + 8): identity}
    assert metadata["frame_id"] == state["frame_ids"][frame]
    np.testing.assert_array_equal(tifffile.imread(standalone), mask)
    assert list(tmp_path.iterdir()) == [standalone]


def test_earlier_machine_clock_cannot_make_a_new_edit_lose_on_merge(tmp_path, monkeypatch):
    import time

    frame, _ = _field(tmp_path)
    old, identity = _mapped_cell(None, "1", 1, frame)
    monkeypatch.setattr(time, "time_ns", lambda: 1)
    renamed = engine.rename_lineage_cell(old, identity, "1a")
    assert renamed["cells"][identity]["revision"] > old["cells"][identity]["revision"]
    assert engine.merge_lineage_states(renamed, old)["cells"][identity]["name"] == "1a"
    remapped = engine.assign_lineage_label(renamed, frame, 2, identity, old_label=1)
    assert remapped["frame_revisions"][frame] > renamed["cells"][identity]["revision"]


def test_boundary_ids_do_not_change_other_identity_colors_or_prevent_correction():
    ids = np.array([1, 65535, 65536, 2**31 + 7, engine.MAX_MASK_LABEL], np.uint32)
    colors = engine._label_colors(ids)
    for index, identity in enumerate(ids):
        np.testing.assert_array_equal(colors[index], engine._label_colors(np.array([identity]))[0])
    mask = np.zeros((20, 20), np.uint32)
    mask[2:18, 2:18] = engine.MAX_MASK_LABEL
    mask[0, 0] = 1
    assert engine.next_label(mask) == 2
    changed, splits = engine.divide_object(mask, (9, 0), (9, 19))
    assert splits == [(engine.MAX_MASK_LABEL, 2)]
    assert changed[0, 0] == 1
    assert engine.MAX_MASK_LABEL in changed
    drawn, label = engine.fill_polygon(mask, [(0, 18), (2, 18), (2, 19), (0, 19)])
    assert label == 2
    assert drawn.dtype == np.uint32


def _lone_daughter_movie(tmp_path):
    frames = [_field(tmp_path, f"lone{index}.tif")[0] for index in range(5)]
    state, mother = _mapped_cell(None, "61", 61, frames[0])
    state = engine.assign_lineage_label(state, frames[1], 61, mother)
    state, lone = _mapped_cell(state, "107", 107, frames[2])
    state = engine.assign_lineage_label(state, frames[3], 107, lone)
    state, daughters = engine.divide_lineage_cell(state, lone, frames[4], ["107a", "107b"], [89, 90])
    state = engine.assign_lineage_label(state, frames[4], 89, daughters[0])
    state = engine.assign_lineage_label(state, frames[4], 90, daughters[1])
    return frames, state, mother, lone, daughters


def test_a_lone_daughter_merges_into_its_mother_with_its_division(tmp_path):
    frames, state, mother, lone, daughters = _lone_daughter_movie(tmp_path)
    merged = engine._merge_continuation(state, mother, lone)
    assert lone not in merged["cells"] and lone in merged["deleted_cells"]
    assert merged["cells"][mother]["children"] == list(daughters)
    assert merged["cells"][mother]["division_frame"] == frames[4]
    assert all(merged["cells"][child]["parent_uuid"] == mother for child in daughters)
    assert [merged["frames"][frame] for frame in frames[2:4]] == [{"107": mother}] * 2
    older = copy.deepcopy(state)
    assert engine.merge_lineage_states(older, merged) == engine.merge_lineage_states(merged, older)
    assert engine.merge_lineage_states(older, merged)["cells"][daughters[0]]["parent_uuid"] == mother
    undone = engine.restore_lineage_state(state, merged)
    assert undone["cells"][daughters[0]]["parent_uuid"] == lone and lone in undone["cells"]
    assert engine.merge_lineage_states(merged, undone)["cells"][lone]["children"] == list(daughters)


def test_divided_or_overlapping_cells_never_merge(tmp_path):
    frames, state, mother, lone, daughters = _lone_daughter_movie(tmp_path)
    with pytest.raises(ValueError, match="cannot become part of"):
        engine._merge_continuation(state, lone, mother)
    with pytest.raises(ValueError, match="cannot become part of"):
        engine._merge_continuation(state, mother, daughters[0])
    overlapping = engine.assign_lineage_label(state, frames[2], 61, mother)
    with pytest.raises(ValueError, match="cannot become part of"):
        engine._merge_continuation(overlapping, mother, lone)


def test_equal_revisions_that_disagree_about_ancestry_are_still_refused(tmp_path):
    frames, state, mother, lone, daughters = _lone_daughter_movie(tmp_path)
    forged = copy.deepcopy(state)
    forged["cells"][daughters[0]]["parent_uuid"] = None
    forged["cells"][lone]["children"] = []
    forged["cells"][lone]["division_frame"] = None
    forged["cells"][lone].pop("division_frame_id")
    forged["cells"][daughters[1]]["parent_uuid"] = None
    with pytest.raises(ValueError, match="disagree"):
        engine.merge_lineage_states(state, forged)
