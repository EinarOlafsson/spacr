"""Synthetic uint32 masks retain extensible lineage inside each TIFF."""

from __future__ import annotations

import copy
import json
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
    state, founder = engine.new_lineage_cell(None, "1", 1, first)
    divided, daughters = engine.divide_lineage_cell(state, founder, second,
                                                    ["1", "1d"], [1, 2**31 + 7])
    assert founder not in daughters and daughters[0] != daughters[1]
    assert divided["cells"][daughters[0]]["name"] == "1"
    assert all(divided["cells"][child]["parent_uuid"] == founder for child in daughters)
    assert divided["frames"][first]["1"] == founder
    assert divided["frames"][second]["1"] == daughters[0]
    assert state["cells"][founder]["children"] == []
    assert engine.validate_lineage(divided) == divided


def test_snapshot_merge_retains_new_name_and_new_division(tmp_path):
    frame, _ = _field(tmp_path)
    initial, founder = engine.new_lineage_cell(None, "1", 1, frame)
    divided, daughters = engine.divide_lineage_cell(initial, founder,
        str(tmp_path / "next.tif"), ["1", "1d"], [1, 2])
    renamed = engine.rename_lineage_cell(divided, daughters[0], "1a")
    merged = engine.merge_lineage_states(renamed, divided, frame=frame)
    assert merged["cells"][daughters[0]]["name"] == "1a"
    merged = engine.merge_lineage_states(merged, initial, frame=frame)
    assert merged["cells"][founder]["children"] == list(daughters)
    conflicting = copy.deepcopy(divided)
    conflicting["cells"][daughters[0]]["born_frame"] = "elsewhere"
    with pytest.raises(ValueError, match="ancestry"):
        engine.merge_lineage_states(merged, conflicting)


def test_mapping_removal_and_collision_are_copy_on_write(tmp_path):
    frame, _ = _field(tmp_path)
    state, one = engine.new_lineage_cell(None, "first", 1, frame)
    state, two = engine.new_lineage_cell(state, "second", 2, frame)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="already belongs"):
        engine.assign_lineage_label(state, frame, 1, two)
    assert state == before
    removed = engine.assign_lineage_label(state, frame, 0, old_label=1)
    assert removed["frames"][frame] == {"2": two}
    assert one in removed["cells"]


def test_tiff_is_self_contained_and_ordinary_resave_retains_width_and_lineage(tmp_path):
    frame, image = _field(tmp_path)
    state, founder = engine.new_lineage_cell(None, "1", 2**31 + 9, frame)
    state, unplaced = engine.new_lineage_cell(state, "unplaced", 2, frame)
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
    engine.save_mask(str(tmp_path), "frame.tif", mask, lineage=state)
    reopened = engine.load_mask_lineage(str(tmp_path), "frame.tif")
    assert reopened == state
    assert len(reopened["cells"]) == 10512
    count = 0
    while reopened["cells"][parent]["parent_uuid"] is not None:
        parent = reopened["cells"][parent]["parent_uuid"]
        count += 1
    assert count == 256


def test_invalid_cycle_or_unknown_schema_is_refused(tmp_path):
    frame, _ = _field(tmp_path)
    state, founder = engine.new_lineage_cell(None, "1", 1, frame)
    state["cells"][founder]["parent_uuid"] = founder
    with pytest.raises(ValueError, match="metadata"):
        engine.validate_lineage(state)
    with pytest.raises(ValueError, match="metadata"):
        engine.validate_lineage({"schema": "future/99", "cells": {}, "frames": {}})


def test_tracking_permuted_segmentation_ids_preserves_both_objects_and_identity(tmp_path):
    first, image = _field(tmp_path)
    second, _ = _field(tmp_path, "next.tif")
    state, identity = engine.new_lineage_cell(None, "1", 1, first)
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


def test_tracking_refuses_to_steal_an_established_cell(tmp_path):
    frame, image = _field(tmp_path)
    state, first = engine.new_lineage_cell(None, "first", 1, frame)
    state, second = engine.new_lineage_cell(state, "second", 2, frame)
    mask = np.zeros(image.shape, np.uint32)
    mask[2:5, 2:5] = 1
    mask[7:10, 7:10] = 2
    before = copy.deepcopy(state)
    with pytest.raises(ValueError, match="already belongs to another"):
        engine.assign_tracked_object(mask, 8, 8, state, frame, first)
    assert state == before
    assert state["frames"][frame]["2"] == second


def test_undo_division_and_rename_survive_saved_older_snapshots_and_redo(tmp_path):
    frame, image = _field(tmp_path)
    next_frame = str(tmp_path / "next.tif")
    initial, founder = engine.new_lineage_cell(None, "1", 1, frame)
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
