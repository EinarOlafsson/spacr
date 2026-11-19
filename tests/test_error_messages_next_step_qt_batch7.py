"""Lineage and mask-label errors explain safe inputs while preserving identities."""
import copy

import numpy as np
import pytest

from spacr.qt import mask_engine as engine

pytestmark = pytest.mark.qt


@pytest.mark.parametrize("label", [-1, 1.5, True, 4294967296])
def test_lineage_label_error_explains_the_valid_input(tmp_path, label):
    """An invalid frame label is refused before the lineage state can change."""
    state = engine.validate_lineage()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError) as caught:
        engine.assign_lineage_label(state, str(tmp_path / "frame.tif"), label)
    assert "Enter a whole-number label within this range" in str(caught.value)
    assert state == before


@pytest.mark.parametrize("name,label", [("", 1), ("cell", 0)])
def test_new_lineage_cell_error_explains_name_and_label(tmp_path, name, label):
    """An invalid founder remains uncreated and the error states both required inputs."""
    state = engine.validate_lineage()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError) as caught:
        engine.new_lineage_cell(state, name, label, str(tmp_path / "frame.tif"))
    assert "Enter a cell name and an integer label from 1 to 4294967295" in str(caught.value)
    assert state == before


def test_unassigned_objects_cannot_swap_lineage_identities(tmp_path):
    """A swap with no frame assignments remains refused and explains how to assign them."""
    state = engine.validate_lineage()
    before = copy.deepcopy(state)
    with pytest.raises(ValueError) as caught:
        engine._swap_frame_identities(state, str(tmp_path / "frame.tif"), 1, 2)
    assert "Assign each object to a different lineage cell" in str(caught.value)
    assert state == before


def test_occupied_frame_label_error_preserves_both_cells(tmp_path):
    """Reusing a frame label cannot merge two identities and points to an unused label."""
    frame = str(tmp_path / "frame.tif")
    state, first = engine.new_lineage_cell(None, "first", 1, frame)
    state, second = engine.new_lineage_cell(state, "second", 2, frame)
    state = engine.assign_lineage_label(state, frame, 1, first)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError) as caught:
        engine.assign_lineage_label(state, frame, 1, second)
    assert "Choose a label that is unused by other cells" in str(caught.value)
    assert state == before


def test_duplicate_daughter_labels_leave_the_parent_undivided(tmp_path):
    """A division cannot reuse one label and tells the user to enter different labels."""
    frame = str(tmp_path / "frame.tif")
    state, parent = engine.new_lineage_cell(None, "parent", 1, frame)
    before = copy.deepcopy(state)
    with pytest.raises(ValueError) as caught:
        engine.divide_lineage_cell(state, parent, frame, ("a", "b"), (2, 2))
    assert "a different positive integer label for each daughter cell" in str(caught.value)
    assert state == before


@pytest.mark.parametrize("label", [-1, 4294967296])
def test_mask_capacity_error_preserves_the_original_labels(label):
    """Out-of-range IDs remain rejected before any cast can wrap or change them."""
    mask = np.array([[label]], dtype=np.int64)
    before = mask.copy()
    with pytest.raises(ValueError) as caught:
        engine._fit_label_width(mask, np.zeros((1, 1), np.uint16))
    assert "Use integer labels from 0 to 4294967295" in str(caught.value)
    np.testing.assert_array_equal(mask, before)


def test_volume_geometry_error_names_the_axes_and_next_step():
    """Invalid geometry changes no mask and names axes as translatable prose."""
    session = engine.VolumeMaskSession(np.zeros((2, 3, 4)), np.zeros((2, 3, 4), np.uint16))
    with pytest.raises(ValueError) as caught:
        session.apply_geometry(0, None)
    message = str(caught.value)
    assert "physical X, Y and Z coordinates" in message
    assert "Select a valid object and edit its boundary" in message
    assert not session.mask.any() and session._undo == []
