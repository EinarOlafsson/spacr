"""3D assignment counts nonzero objects even when no background voxel remains."""
import numpy as np
import pytest

from spacr.qt.mask_engine import VolumeMaskSession


@pytest.mark.parametrize("before,expected", [
    ([[[0, 0, 0]]], [[[0, 0, 0]]]),
    ([[[1, 1, 1]]], [[[1, 1, 1]]]),
    ([[[1, 1, 2, 2]]], [[[1, 1, 2, 2]]]),
    ([[[1, 2, 1]]], [[[1, 2, 3]]]),
    ([[[1, 0, 1]]], [[[1, 0, 2]]]),
], ids=["empty", "full-single", "full-multiple", "full-split", "background-split"])
def test_assign_objects_reports_nonzero_ids_and_preserves_voxel_history(before, expected):
    original = np.asarray(before, dtype=np.uint16)
    wanted = np.asarray(expected, dtype=np.uint16)
    session = VolumeMaskSession(np.zeros(original.shape, dtype=np.uint8), original.copy())

    changed, objects = session.assign_objects()

    assert objects == len(set(wanted.ravel()) - {0})
    assert changed == int(np.count_nonzero(original != wanted))
    assert session.mask.shape == original.shape and session.mask.dtype == original.dtype
    np.testing.assert_array_equal(session.mask, wanted)
    np.testing.assert_array_equal(session.mask[original == 0], original[original == 0])
    if changed:
        assert session.undo()
        np.testing.assert_array_equal(session.mask, original)
        assert session.redo()
        np.testing.assert_array_equal(session.mask, wanted)
    else:
        assert not session.undo() and not session.redo()
