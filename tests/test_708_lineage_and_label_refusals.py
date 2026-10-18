"""Item 708: embedded lineage and uint32 labels refuse every malformed or unsafe input.

A lineage graph read from a TIFF is accepted only when every cell, division,
frame identity, label and revision is consistent; each defect is named in
the refusal. Editing operations refuse unknown cells, non-integer pixels and
labels outside uint32, merging refuses snapshots that disagree, and label
allocation, combination and width fitting refuse to wrap past uint32.
"""
from __future__ import annotations

import copy
import json
import os
from uuid import uuid4

import numpy as np
import pytest
import tifffile

from spacr.qt import mask_engine as engine
from spacr.tiff_io import write_tiff


def _divided(tmp_path):
    first, second = str(tmp_path / "first.tif"), str(tmp_path / "second.tif")
    state, founder = engine.new_lineage_cell(None, "A", 1, first)
    state = engine.assign_lineage_label(state, first, 1, founder)
    state, daughters = engine.divide_lineage_cell(state, founder, second, ("A.1", "A.2"), (2, 3))
    return engine.validate_lineage(state), founder, daughters, first, second


def _breaks(reason, state):
    with pytest.raises(ValueError, match="lineage metadata: " + reason):
        engine.validate_lineage(state)


def test_an_empty_state_is_a_fresh_graph():
    assert engine.validate_lineage(None) == {
        "schema": engine.LINEAGE_SCHEMA, "cells": {}, "frames": {}, "frame_revisions": {},
        "deleted_cells": {}, "frame_ids": {}}


def _mutations(state, founder, daughters, first, second):
    a, b = daughters
    cases = {}

    def case(reason, change):
        broken = copy.deepcopy(state)
        change(broken)
        cases.setdefault(reason, []).append(broken)

    case("graph", lambda s: s.update(cells=[]))
    case("frame identities", lambda s: s.update(frame_ids=[]))
    case("frame path", lambda s: s["frames"].update({"": {}}))
    case("frame identity", lambda s: s["frame_ids"].update({first: s["frame_ids"][first].upper()}))
    case("revisions", lambda s: s.update(frame_revisions=[]))
    case("deleted cells", lambda s: s.update(deleted_cells=[]))
    case("deleted cell revision", lambda s: s["deleted_cells"].update({str(uuid4()): -1}))
    case("UUID", lambda s: s["cells"].update({str(uuid4()).upper(): dict(
        s["cells"][a], name="loose", parent_uuid=None)}))
    case("name", lambda s: s["cells"][founder].update(name="  "))
    loner = str(uuid4())
    case("parent", lambda s: s["cells"].update({loner: dict(
        s["cells"][a], name="loner", parent_uuid=loner)}))

    def daughter_first(s):
        s["cells"] = {key: s["cells"][key] for key in (a, b, founder)}
        s["cells"][founder]["children"] = [b, str(uuid4())]

    case("parent division", daughter_first)
    case("children", lambda s: s["cells"][founder].update(children=[a]))

    def stranger(s):
        other = str(uuid4())
        s["cells"][other] = dict(s["cells"][founder], name="X", parent_uuid=None)
    case("daughter", stranger)
    case("division", lambda s: s["cells"][founder].update(division_frame=None,
                                                           division_frame_id=None))
    case("ancestry frame identity", lambda s: s["cells"][a].update(born_frame_id=str(uuid4())))
    case("label", lambda s: s["cells"][a].update(label=0))
    case("label", lambda s: s["cells"][a].update(label=True))
    case("revision", lambda s: s["cells"][a].update(revision=-1))
    case("deleted cell", lambda s: s["deleted_cells"].update(
        {a: s["cells"][a].get("revision", 0)}))
    case("frame mapping", lambda s: s["frames"].update({first: []}))
    case("duplicate identity", lambda s: s["frames"].update({first: {"1": founder, "2": founder}}))
    case("frame label", lambda s: s["frames"].update({first: {"0": founder}}))
    case("frame label", lambda s: s["frames"].update({first: {"1": str(uuid4())}}))
    case("frame revision", lambda s: s["frame_revisions"].update({first: -1}))
    return cases


def test_every_lineage_defect_is_refused_by_name(tmp_path):
    state, founder, daughters, first, second = _divided(tmp_path)
    for reason, states in _mutations(state, founder, daughters, first, second).items():
        for broken in states:
            _breaks(reason, broken)
    assert engine.validate_lineage(state) == state


def test_a_parent_cycle_is_refused(tmp_path):
    frame = str(tmp_path / "f.tif")
    state = engine.validate_lineage(None)
    a, b, c, d = (str(uuid4()) for _ in range(4))
    common = {"born_frame": frame, "division_frame": None}
    state["cells"] = {
        a: dict(common, name="a", parent_uuid=b, children=[b, c], division_frame=frame),
        b: dict(common, name="b", parent_uuid=a, children=[a, d], division_frame=frame),
        c: dict(common, name="c", parent_uuid=a, children=[]),
        d: dict(common, name="d", parent_uuid=b, children=[]),
    }
    _breaks("cycle", state)


def test_editing_refuses_unknown_cells_bad_pixels_and_labels(tmp_path):
    state, founder, daughters, first, second = _divided(tmp_path)
    mask = np.zeros((6, 6), np.uint32)
    mask[1:3, 1:3] = 7
    with pytest.raises(ValueError, match="whole numbers"):
        engine.assign_lineage_label(state, first, -1, founder)
    with pytest.raises(ValueError, match="existing lineage cell"):
        engine.assign_lineage_label(state, first, 4, str(uuid4()))
    with pytest.raises(ValueError, match="needs a name"):
        engine.new_lineage_cell(state, " ", 4, first)
    with pytest.raises(ValueError, match="existing lineage cell"):
        engine.assign_tracked_object(mask, 1, 1, state, first, str(uuid4()))
    with pytest.raises(ValueError, match="integer pixels"):
        engine.assign_tracked_object(mask, 1.5, 1, state, first, founder)
    for x, y in ((9, 1), (5, 5)):
        labels, same, old, new = engine.assign_tracked_object(mask, x, y, state, first, founder)
        assert (old, new) == (0, 0) and same == state
        np.testing.assert_array_equal(labels, mask)
    with pytest.raises(ValueError, match="nonzero uint32 label"):
        engine.assign_tracked_object(mask, 1, 1, state, first, founder, preferred_label=0)
    with pytest.raises(ValueError, match="nonempty name"):
        engine.rename_lineage_cell(state, founder, "")
    with pytest.raises(ValueError, match="not already divided"):
        engine.divide_lineage_cell(state, founder, second, ("x", "y"), (5, 6))
    a = daughters[0]
    with pytest.raises(ValueError, match="distinct nonzero labels"):
        engine.divide_lineage_cell(state, a, second, ("x", "y"), (5, 5))
    with pytest.raises(ValueError, match="distinct nonzero labels"):
        engine.divide_lineage_cell(state, a, second, ("x", "y"), (0, 5))
    with pytest.raises(ValueError, match="integer pixels"):
        engine.assign_object_label(mask, 1, 1.0, 3)


def test_snapshots_that_disagree_about_a_frame_are_not_merged(tmp_path):
    state, *_rest, first, _second = _divided(tmp_path)
    other = copy.deepcopy(state)
    other["frame_ids"][first] = str(uuid4())
    other["cells"] = {}
    other["frames"] = {}
    with pytest.raises(ValueError, match="disagree about a frame's identity"):
        engine.merge_lineage_states(state, other)


def test_a_cell_deleted_in_the_newer_snapshot_leaves_no_label_behind(tmp_path):
    state, founder, daughters, first, second = _divided(tmp_path)
    state, loose = engine.new_lineage_cell(state, "B", 9, first)
    state = engine.assign_lineage_label(state, first, 9, loose)
    assert state["frames"][first]["9"] == loose
    incoming = {"schema": engine.LINEAGE_SCHEMA, "cells": {}, "frames": {},
                "frame_ids": dict(state["frame_ids"]), "frame_revisions": {},
                "deleted_cells": {loose: 4 * 10 ** 18}}
    merged = engine.merge_lineage_states(state, incoming)
    assert loose not in merged["cells"] and merged["deleted_cells"][loose] == 4 * 10 ** 18
    assert "9" not in merged["frames"][first]
    assert merged["cells"][founder]["children"] == list(daughters)


def test_moving_frames_onto_one_path_with_different_contents_is_refused(tmp_path):
    state, founder, daughters, first, second = _divided(tmp_path)
    with pytest.raises(ValueError, match="incompatible frame identity"):
        engine._remap_lineage_frames(state, {second: first})
    same_identity = copy.deepcopy(state)
    same_identity["frame_ids"][second] = same_identity["frame_ids"][first]
    same_identity["frames"][second] = {"9": daughters[0]}
    same_identity["cells"][daughters[0]]["born_frame_id"] = same_identity["frame_ids"][first]
    same_identity["cells"][daughters[1]]["born_frame_id"] = same_identity["frame_ids"][first]
    same_identity["cells"][founder]["division_frame_id"] = same_identity["frame_ids"][first]
    with pytest.raises(ValueError, match="incompatible frame identity"):
        engine._remap_lineage_frames(same_identity, {second: first})


def _mask_with_description(tmp_path, description, name="field.tif"):
    folder = tmp_path / "images"
    folder.mkdir(exist_ok=True)
    write_tiff(folder / name, np.zeros((4, 4), np.uint16))
    target = engine.mask_save_path(str(folder), name)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tifffile.imwrite(target, np.zeros((4, 4), np.uint32), description=description,
                     metadata=None)
    return str(folder), name


@pytest.mark.parametrize("description, error", [
    ('{"spacr_mask": ', "cannot be read"),
    (json.dumps({"spacr_mask": {"schema": "spacr.mask/v0"}}), "unsupported embedded mask"),
])
def test_unreadable_or_unknown_mask_metadata_is_refused(tmp_path, description, error):
    folder, name = _mask_with_description(tmp_path, description)
    with pytest.raises(ValueError, match=error):
        engine.load_mask_metadata(folder, name)


def test_mask_metadata_whose_frame_identity_does_not_match_its_lineage_is_refused(tmp_path):
    state, *_rest = _divided(tmp_path)
    folder, name = _mask_with_description(tmp_path, json.dumps({"spacr_mask": {
        "schema": "spacr.mask/v1", "lineage": state,
        "frame": str(tmp_path / "first.tif"), "frame_id": str(uuid4())}}))
    with pytest.raises(ValueError, match="unsupported embedded mask"):
        engine.load_mask_metadata(folder, name)
    folder, name = _mask_with_description(tmp_path, "plain description", name="other.tif")
    assert engine.load_mask_metadata(folder, name) == {}


def test_time_labels_need_a_tiff_and_bundles_refuse_32_bit_ids(tmp_path):
    with pytest.raises(ValueError, match="TIFF masks with embedded lineage"):
        engine.save_mask(str(tmp_path), "field_seg.npy", np.zeros((4, 4), np.uint16),
                         lineage=engine.validate_lineage(None))
    bundle = tmp_path / "field_seg.npy"
    np.save(bundle, {"masks": np.zeros((4, 4), np.uint16), "img": np.zeros((4, 4))},
            allow_pickle=True)
    big = np.zeros((4, 4), np.uint32)
    big[0, 0] = 70000
    with pytest.raises(ValueError, match="32-bit manual identities as TIFF"):
        engine.save_seg_bundle(str(bundle), big, preserve_ids=True)
    assert int(np.load(bundle, allow_pickle=True).item()["masks"].max()) == 0


def test_labels_past_uint32_are_refused_everywhere(monkeypatch):
    huge = np.array([[0, 2 ** 32]], dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds uint32 capacity"):
        engine._as_field_mask(huge, huge.shape, "/m.tif", "f.tif")
    with pytest.raises(ValueError, match="outside unsigned uint32"):
        engine._fit_label_width(huge, np.zeros((1, 2), np.uint8))
    with pytest.raises(ValueError, match="non-negative uint32"):
        next(engine._available_labels(np.array([[-1, 2]])))
    assert engine._label_boxes(np.zeros((0, 0), np.uint32)) == {}


def test_label_allocation_wraps_and_finally_runs_out(monkeypatch):
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 4)
    free = engine._available_labels(np.array([[0, 2, 4]]))
    assert [next(free), next(free)] == [1, 3]
    with pytest.raises(ValueError, match="No unused 32-bit label remains"):
        next(free)


def test_combining_near_the_ceiling_reuses_free_ids_or_refuses(monkeypatch):
    monkeypatch.setattr(engine, "MAX_MASK_LABEL", 6)
    old = np.array([[1, 0, 0, 0], [0, 0, 5, 0]], np.uint8)
    new = np.array([[0, 3, 0, 4], [0, 0, 0, 0]], np.uint8)
    combined = engine.combine_masks(old, new, mode="merge")
    assert combined[0, 0] == 1 and combined[1, 2] == 5
    assert {int(combined[0, 1]), int(combined[0, 3])} == {6, 2}
    with pytest.raises(ValueError, match="past uint32"):
        engine.combine_masks(old, np.array([[7, 0, 0, 0], [0, 0, 0, 0]], np.uint8))


def test_the_overlay_blend_cache_stays_small():
    engine._OVERLAY_BLEND_TABLES.clear()
    tables = [engine._overlay_blend_table(alpha / 20) for alpha in range(12)]
    assert len(engine._OVERLAY_BLEND_TABLES) <= 9
    assert tables[0][200, 10] == 200 and tables[10][0, 255] == 127


def test_the_wand_flood_stops_at_the_image_edge():
    image = np.full((5, 5), 10, np.uint8)
    out = engine.magic_wand(image, np.zeros((5, 5), np.uint8), 0, 0, tolerance=1)
    assert int(out.sum()) == 25 * 255


def test_detected_volume_labels_must_fit_the_session_dtype():
    session = engine.VolumeMaskSession(np.zeros((2, 3, 3), np.uint16), np.zeros((2, 3, 3), np.uint16))
    labels = np.zeros((2, 3, 3), np.uint32)
    labels[0, 0, 0] = 70000
    with pytest.raises(ValueError, match="No unused object identifier fits"):
        session.replace_labels(labels)
    assert not session.mask.any()


def test_a_tiff_with_too_many_channels_is_not_a_volume(tmp_path):
    path = tmp_path / "many.tif"
    tifffile.imwrite(path, np.zeros((17, 2, 4, 4), np.uint8), metadata={"axes": "CZYX"})
    with pytest.raises(ValueError, match="three spatial axes"):
        engine._read_tiff_volume(str(path))


def test_a_channel_spacing_entry_is_dropped_from_the_voxel_spacing(tmp_path):
    path = tmp_path / "two.tif"
    tifffile.imwrite(path, np.zeros((2, 3, 4, 4), np.uint8),
                     metadata={"axes": "CZYX", "voxel_spacing": [9.0, 2.0, 0.5, 0.25],
                               "unit": "um"})
    image, axes, spacing, unit = engine._read_tiff_volume(str(path))
    assert image.shape == (2, 3, 4, 4) and axes == "ZYX"
    assert tuple(spacing) == (2.0, 0.5, 0.25) and unit == "um"
