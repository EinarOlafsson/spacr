"""Malformed persisted identities are refused without changing valid edits."""

import copy
import json
from uuid import UUID

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.tiff_io import write_tiff


@pytest.fixture
def graph(tmp_path):
    frame, later = str(tmp_path / "first.tif"), str(tmp_path / "next.tif")
    state, parent = engine.new_lineage_cell(None, "parent", 1, frame)
    state = engine.assign_lineage_label(state, frame, 1, parent)
    state, other = engine.new_lineage_cell(state, "unrelated", 4, frame)
    state, children = engine.divide_lineage_cell(state, parent, later,
                                                ["first", "second"], [2, 3])
    return state, {"parent": parent, "first": children[0], "second": children[1],
                   "other": other, "frame": frame, "later": later}


def _resolve(value, names):
    if isinstance(value, str) and value.startswith("$"):
        return names[value[1:]]
    if isinstance(value, dict):
        return {_resolve(key, names): _resolve(item, names) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve(item, names) for item in value]
    return value


@pytest.mark.parametrize("path,value,reason", [
    (("cells",), [], "graph"),
    (("frame_ids",), [], "frame identities"),
    (("frames", ""), {}, "frame path"),
    (("frame_ids", "$frame"), "A0000000-0000-0000-0000-000000000001", "frame identity"),
    (("frame_revisions",), [], "revisions"),
    (("deleted_cells",), [], "deleted cells"),
    (("deleted_cells", "$other"), True, "deleted cell revision"),
    (("cells", "A0000000-0000-0000-0000-000000000001"), {}, "UUID"),
    (("cells", "$other", "name"), " ", "name"),
    (("cells", "$other", "parent_uuid"), "$parent", "parent division"),
    (("cells", "$parent", "children"), ["$first"], "children"),
    (("cells", "$parent", "children"), ["$first", "$other"], "daughter"),
    (("cells", "$parent", "division_frame"), None, "division"),
    (("cells", "$other", "born_frame_id"), str(UUID(int=999)), "ancestry frame identity"),
    (("cells", "$other", "label"), True, "label"),
    (("cells", "$other", "revision"), True, "revision"),
    (("deleted_cells", "$other"), 2**127, "deleted cell"),
    (("frames", "$later"), [], "frame mapping"),
    (("frames", "$later"), {"2": "$first", "3": "$first"}, "duplicate identity"),
    (("frames", "$later"), {"02": "$first"}, "frame label"),
    (("frame_revisions", "$later"), True, "frame revision"),
])
def test_corrupt_embedded_lineage_is_rejected_without_rewriting_the_tiff(
        graph, tmp_path, path, value, reason):
    state, names = graph
    corrupted = copy.deepcopy(state)
    target = corrupted
    keys = [_resolve(part, names) for part in path]
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = _resolve(value, names)
    before = copy.deepcopy(corrupted)
    metadata = {"schema": "spacr.mask/v1", "frame": names["frame"],
                "lineage": corrupted, "preserve_ids": True}
    destination = tmp_path / "masks" / "first.tif"
    destination.parent.mkdir()
    write_tiff(destination, np.ones((2, 3), np.uint32), metadata=None,
               description=json.dumps({"spacr_mask": metadata}))
    original_bytes = destination.read_bytes()

    with pytest.raises(ValueError, match=reason):
        engine.load_mask_lineage(str(tmp_path), "first.tif")

    assert destination.read_bytes() == original_bytes
    assert corrupted == before
    assert engine.validate_lineage(state) == state


def test_a_reciprocal_parent_cycle_is_rejected_even_with_valid_uuid_links(graph):
    state, names = graph
    parent, first, other = (names[key] for key in ("parent", "first", "other"))
    state["cells"][parent]["parent_uuid"] = first
    state["cells"][first].update(children=[parent, other], division_frame=names["later"],
                                  division_frame_id=state["frame_ids"][names["later"]])
    state["cells"][other]["parent_uuid"] = first
    original = copy.deepcopy(state)

    with pytest.raises(ValueError, match="cycle"):
        engine.validate_lineage(state)

    assert state == original


def test_embedded_mask_frame_id_must_agree_with_its_lineage(graph, tmp_path):
    state, names = graph
    metadata = {"schema": "spacr.mask/v1", "frame": names["frame"],
                "frame_id": str(UUID(int=999)), "lineage": state}
    destination = tmp_path / "masks" / "first.tif"
    destination.parent.mkdir()
    write_tiff(destination, np.ones((2, 3), np.uint32), metadata=None,
               description=json.dumps({"spacr_mask": metadata}))
    original = destination.read_bytes()

    with pytest.raises(ValueError, match="embedded mask metadata"):
        engine.load_mask_lineage(str(tmp_path), "first.tif")

    assert destination.read_bytes() == original


@pytest.mark.parametrize("description,invalid", [
    ("Legacy acquisition notes", False),
    (json.dumps({"spacr_mask": []}), True),
    (json.dumps({"spacr_mask": {"schema": "future/99"}}), True),
])
def test_legacy_descriptions_and_unsupported_mask_records_remain_distinct(
        tmp_path, description, invalid):
    destination = tmp_path / "masks" / "first.tif"
    destination.parent.mkdir()
    write_tiff(destination, np.ones((2, 3), np.uint32), metadata=None,
               description=description)
    original = destination.read_bytes()

    if invalid:
        with pytest.raises(ValueError, match="embedded mask metadata"):
            engine.load_mask_metadata(str(tmp_path), "first.tif")
    else:
        assert engine.load_mask_metadata(str(tmp_path), "first.tif") == {}

    assert destination.read_bytes() == original


@pytest.mark.parametrize("operation,reason", [
    (lambda s, n, m: engine.assign_lineage_label(s, n["frame"], True, n["other"]), "whole numbers"),
    (lambda s, n, m: engine.assign_lineage_label(s, n["frame"], 8, str(UUID(int=999))), "existing lineage"),
    (lambda s, n, m: engine.new_lineage_cell(s, " ", 8, n["frame"]), "name"),
    (lambda s, n, m: engine.assign_tracked_object(m, 0, 0, s, n["frame"], str(UUID(int=999))), "existing lineage"),
    (lambda s, n, m: engine.assign_tracked_object(m, .5, 0, s, n["frame"], n["other"]), "integer pixels"),
    (lambda s, n, m: engine.assign_tracked_object(m, 0, 0, s, n["frame"], n["other"], True), "nonzero uint32"),
    (lambda s, n, m: engine.rename_lineage_cell(s, n["other"], ""), "nonempty name"),
    (lambda s, n, m: engine.divide_lineage_cell(s, n["parent"], n["later"], ["a", "b"], [8, 9]), "already divided"),
    (lambda s, n, m: engine.divide_lineage_cell(s, n["other"], n["later"], ["a", "b"], [8, 8]), "distinct"),
    (lambda s, n, m: engine.divide_lineage_cell(s, n["other"], n["later"], ["a", " "], [8, 9]), "daughter names"),
])
def test_invalid_lineage_edits_leave_pixels_and_graph_untouched(graph, operation, reason):
    state, names = graph
    mask = np.array([[8, 0], [0, 0]], np.uint32)
    old_state, old_mask = copy.deepcopy(state), mask.copy()

    with pytest.raises(ValueError, match=reason):
        operation(state, names, mask)

    assert state == old_state
    np.testing.assert_array_equal(mask, old_mask)


@pytest.mark.parametrize("point", [(-1, 0), (1, 1)])
def test_tracking_outside_an_object_does_not_create_a_frame_mapping(graph, point):
    state, names = graph
    mask = np.array([[8, 0], [0, 0]], np.uint32)
    before = copy.deepcopy(state)

    result, unchanged, previous, assigned = engine.assign_tracked_object(
        mask, *point, state, names["later"], names["other"])

    np.testing.assert_array_equal(result, mask)
    assert unchanged == state == before
    assert (previous, assigned) == (0, 0)


def test_incompatible_frame_identities_cannot_be_merged_or_collapsed(graph):
    state, names = graph
    other = copy.deepcopy(state)
    other["frame_ids"][names["frame"]] = str(UUID(int=999))
    for cell in other["cells"].values():
        if cell["born_frame"] == names["frame"]:
            cell["born_frame_id"] = other["frame_ids"][names["frame"]]
    before = copy.deepcopy(state)

    with pytest.raises(ValueError, match="frame's identity"):
        engine.merge_lineage_states(state, other)
    with pytest.raises(ValueError, match="incompatible frame identity"):
        engine._remap_lineage_frames(state, {names["later"]: names["frame"]})

    assert state == before


def test_same_frame_uuid_does_not_allow_collapsing_different_pixel_mappings(graph):
    state, names = graph
    state["frame_ids"][names["later"]] = state["frame_ids"][names["frame"]]
    for cell in state["cells"].values():
        for field in ("born_frame", "division_frame"):
            cell[field + "_id"] = state["frame_ids"].get(cell[field])
    before = copy.deepcopy(state)

    with pytest.raises(ValueError, match="incompatible frame identity"):
        engine._remap_lineage_frames(state, {names["later"]: names["frame"]})

    assert state == before


def test_a_newer_tombstone_removes_an_older_frames_pixel_reference(graph):
    state, names = graph
    restored = engine.validate_lineage()
    restored["deleted_cells"] = {
        identity: cell["revision"] + 1 for identity, cell in state["cells"].items()}

    merged = engine.merge_lineage_states(state, restored)

    assert merged["cells"] == {}
    assert all(not mapping for mapping in merged["frames"].values())
    assert set(merged["deleted_cells"]) == set(state["cells"])
    assert state["frames"][names["frame"]] == {"1": names["parent"]}


def test_merging_at_uint32_capacity_reuses_free_ids_without_overwriting_old_objects():
    old = np.array([[engine.MAX_MASK_LABEL, 0, 0], [1, 0, 0]], np.uint32)
    incoming = np.array([[5, 5, 0], [0, 7, 7]], np.uint32)
    before = old.copy()

    merged = engine.combine_masks(old, incoming, "merge")

    np.testing.assert_array_equal(merged, [[engine.MAX_MASK_LABEL, 2, 0], [1, 3, 3]])
    np.testing.assert_array_equal(old, before)
    assert merged.dtype == np.uint32


@pytest.mark.parametrize("bad", [-1, 2**32])
def test_label_allocation_and_casting_reject_values_outside_uint32(bad):
    mask = np.array([[bad]], np.int64)
    with pytest.raises(ValueError, match="uint32"):
        next(engine._available_labels(mask))
    with pytest.raises(ValueError, match="uint32 capacity"):
        engine._fit_label_width(mask, np.zeros((1, 1), np.uint16))
    np.testing.assert_array_equal(mask, [[bad]])


def test_replacing_with_an_overwide_mask_refuses_wraparound():
    old = np.array([[9, 0]], np.uint16)
    with pytest.raises(ValueError, match="past uint32"):
        engine.combine_masks(old, np.array([[0, 2**32]], np.uint64), "replace")
    np.testing.assert_array_equal(old, [[9, 0]])


@pytest.mark.parametrize("with_lineage", [False, True])
def test_cellpose_bundle_refuses_unsupported_manual_identity_storage(
        tmp_path, with_lineage):
    destination = tmp_path / "field_seg.npy"
    original = np.array([[0, 9], [9, 0]], np.uint16)
    np.save(destination, {"masks": original, "img": np.zeros_like(original),
                          "curator_note": "retain this annotation"})
    saved = destination.read_bytes()
    attempted = np.full(original.shape, engine.MAX_MASK_LABEL, np.uint32)

    with pytest.raises(ValueError, match="TIFF"):
        if with_lineage:
            engine.save_mask(str(tmp_path), destination.name, attempted,
                             lineage=engine.validate_lineage())
        else:
            engine.save_seg_bundle(str(destination), attempted, preserve_ids=True)

    assert destination.read_bytes() == saved
    assert not list(tmp_path.glob("*.tmp"))
    np.testing.assert_array_equal(np.load(destination, allow_pickle=True).item()["masks"], original)
    np.testing.assert_array_equal(attempted, np.full(original.shape, engine.MAX_MASK_LABEL, np.uint32))


def test_opening_an_overwide_tiff_mask_cannot_wrap_its_persisted_ids(tmp_path):
    source = tmp_path / "field.tif"
    destination = tmp_path / "masks" / source.name
    destination.parent.mkdir()
    write_tiff(source, np.zeros((2, 3), np.uint16))
    labels = np.array([[0, 1, 0], [0, 2**32, 0]], np.uint64)
    write_tiff(destination, labels)
    source_bytes, mask_bytes = source.read_bytes(), destination.read_bytes()

    with pytest.raises(ValueError, match="exceeds uint32 capacity"):
        engine.load_image_and_mask(str(tmp_path), source.name)

    assert source.read_bytes() == source_bytes
    assert destination.read_bytes() == mask_bytes


def test_edit_label_rejects_boolean_coordinates_without_touching_pixels():
    mask = np.array([[9, 0], [0, 0]], np.uint32)
    with pytest.raises(ValueError, match="integer pixels"):
        engine.assign_object_label(mask, True, 0, 2)
    np.testing.assert_array_equal(mask, [[9, 0], [0, 0]])


def test_an_empty_field_has_no_object_readout_or_phantom_identity():
    mask = np.empty((0, 3), np.uint16)
    image = np.empty(mask.shape, np.uint16)
    lookup = engine.ObjectLookup(mask, image)

    assert lookup.labels.shape == mask.shape
    assert lookup._boxes == {}
    assert lookup.at(0, 0) is None
    assert lookup.measure(engine.MAX_MASK_LABEL) is None
    with pytest.raises(ValueError, match="nonempty"):
        engine.ObjectLookup(mask, image, preserve_ids=True)
