"""The fields whose merged array is missing, empty or unreadable."""

import numpy as np

from spacr.utils import _unmerged_fields


def test_every_complete_field_is_merged(tmp_path):
    np.save(tmp_path / "a.npy", np.ones((2, 2)))
    assert _unmerged_fields(str(tmp_path), {"a.npy"}, {"a.npy"}) == []


def test_missing_empty_and_unreadable_fields_are_listed(tmp_path):
    np.save(tmp_path / "empty.npy", np.zeros((0, 2)))
    (tmp_path / "broken.npy").write_bytes(b"not an array")
    np.save(tmp_path / "good.npy", np.ones(3))
    stack = {"missing.npy", "empty.npy", "broken.npy", "good.npy"}
    merged = {"empty.npy", "broken.npy", "good.npy"}
    assert _unmerged_fields(str(tmp_path), stack, merged) == [
        "broken.npy", "empty.npy", "missing.npy"]
