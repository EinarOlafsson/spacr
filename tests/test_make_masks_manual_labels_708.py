"""Manual identities survive correction, history, and sidecar-free mask storage."""

from __future__ import annotations

import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.tiff_io import write_tiff


def _labels(dtype=np.uint16):
    labels = np.zeros((12, 16), dtype=dtype)
    labels[2:7, 2:6] = 7
    labels[3:9, 10:14] = 12
    return labels


@pytest.mark.parametrize("target", [1, 255, 256, 65535, np.uint16(900)])
def test_explicit_label_changes_only_the_selected_identity_and_keeps_input(target):
    source = _labels(np.uint8)
    before = source.copy()
    changed, old_id = engine.assign_object_label(source, 3, 4, target)
    assert old_id == 7
    assert changed.dtype == np.uint16
    np.testing.assert_array_equal(source, before)
    np.testing.assert_array_equal(changed == target, before == 7)
    np.testing.assert_array_equal(changed[before != 7], before[before != 7])


@pytest.mark.parametrize("target", [-1, 65536, 2**64, 1.5, "7", True, np.nan])
def test_invalid_selected_id_is_rejected_without_mutation(target):
    source = _labels()
    before = source.copy()
    with pytest.raises(ValueError, match="whole numbers"):
        engine.assign_object_label(source, 3, 4, target)
    np.testing.assert_array_equal(source, before)


def test_occupied_identity_is_refused_instead_of_silently_merging_objects():
    source = _labels()
    before = source.copy()
    with pytest.raises(ValueError, match="already belongs"):
        engine.assign_object_label(source, 3, 4, 12)
    np.testing.assert_array_equal(source, before)


def test_zero_removes_only_the_clicked_object():
    source = _labels()
    changed, old_id = engine.assign_object_label(source, 3, 4, 0)
    assert old_id == 7
    assert set(np.unique(changed)) == {0, 12}
    np.testing.assert_array_equal(changed == 12, source == 12)


@pytest.mark.parametrize("position,target,old_id", [
    ((3, 4), 7, 7), ((0, 0), 3, 0), ((-1, 4), 3, 0), ((16, 4), 3, 0),
])
def test_same_identity_background_and_outside_do_not_edit(position, target, old_id):
    source = _labels()
    changed, previous = engine.assign_object_label(source, *position, target)
    assert previous == old_id
    np.testing.assert_array_equal(changed, source)
    assert not np.shares_memory(changed, source)


def test_disconnected_pieces_of_an_existing_identity_stay_together():
    source = _labels()
    source[10, 3] = 7
    changed, _ = engine.assign_object_label(source, 3, 4, 900)
    assert changed[10, 3] == 900
    assert not np.any(changed == 7)


def test_history_and_segmentation_correction_keep_manual_identity():
    original = _labels()
    original[4, 3] = 0
    history = engine.MaskHistory()
    history.push(original)
    labelled, _ = engine.assign_object_label(original, 2, 3, 65535)
    history.push(labelled)
    filled = engine.fill_holes(labelled, preserve_ids=True)
    assert filled[4, 3] == 65535
    history.push(filled)
    np.testing.assert_array_equal(history.undo(), labelled)
    np.testing.assert_array_equal(history.undo(), original)
    np.testing.assert_array_equal(history.redo(), labelled)
    np.testing.assert_array_equal(history.redo(), filled)
    assert labelled[4, 3] == 0


def test_saved_manual_ids_reopen_exactly_without_external_files(tmp_path):
    image = np.arange(12 * 16, dtype=np.uint16).reshape(12, 16)
    write_tiff(tmp_path / "frame.tif", image)
    labels = np.zeros(image.shape, np.uint16)
    labels[2:5, 2:5] = 7
    labels[8:10, 10:13] = 7
    labelled, _ = engine.assign_object_label(labels, 3, 3, 65535)
    target = engine.save_mask(str(tmp_path), "frame.tif", labelled, preserve_ids=True)
    _, reopened = engine.load_image_and_mask(str(tmp_path), "frame.tif")
    np.testing.assert_array_equal(reopened, labelled)
    assert sorted(path.name for path in (tmp_path / "masks").iterdir()) == ["frame.tif"]
    with pytest.raises(ValueError, match="uint16"):
        engine.save_mask(str(tmp_path), "frame.tif", labelled.astype(np.uint32) + 1,
                         preserve_ids=True)
    _, after_refused_save = engine.load_image_and_mask(str(tmp_path), "frame.tif")
    np.testing.assert_array_equal(after_refused_save, labelled)
    assert target == str(tmp_path / "masks/frame.tif")
