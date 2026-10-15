"""An invalid old Measure target never retires its intact transport array."""
import io

import numpy as np
import pytest

from spacr import example_archives as examples


@pytest.mark.parametrize("damage", [
    "empty", "header", "body", "npz-disguised", "object-pickle",
])
def test_invalid_existing_array_keeps_both_original_files_and_warns(
        tmp_path, caplog, damage):
    archive = tmp_path / "field.npz"
    np.savez_compressed(archive, image=np.arange(12, dtype=np.uint16))
    buffer = io.BytesIO()
    if damage == "npz-disguised":
        np.savez_compressed(buffer, image=np.arange(12))
    elif damage == "object-pickle":
        np.save(buffer, np.array([{"must": "never deserialize"}], dtype=object))
    else:
        np.save(buffer, np.arange(12, dtype=np.uint16))
    encoded = buffer.getvalue()
    damaged = {"empty": b"", "header": encoded[:8],
               "body": encoded[:-3]}.get(damage, encoded)
    target = archive.with_suffix(".npy")
    target.write_bytes(damaged)
    original_archive = archive.read_bytes()
    examples.expand_measure_arrays(tmp_path)
    assert archive.read_bytes() == original_archive
    assert target.read_bytes() == damaged
    assert "could not unpack" in caplog.text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npy", "field.npz"]


@pytest.mark.parametrize("existing", [
    np.arange(6, dtype=np.uint16).reshape(2, 3),
    np.empty((0,), dtype=np.float32),
    np.empty((2, 0, 4), dtype=np.uint8),
    np.array(5, dtype=np.int16),
    np.asfortranarray(np.arange(12).reshape(3, 4)),
    np.zeros((2,), dtype=[("a", "u2"), ("b", "f4")]),
], ids=["numeric", "empty-vector", "empty-axis", "scalar", "fortran", "structured"])
def test_existing_valid_numeric_array_keeps_winning_without_rewrite(
        tmp_path, existing):
    archive = tmp_path / "field.npz"
    np.savez_compressed(archive, image=np.array([999], dtype=np.int32))
    target = archive.with_suffix(".npy")
    np.save(target, existing)
    original = target.read_bytes()
    examples.expand_measure_arrays(tmp_path)
    assert not archive.exists()
    assert target.read_bytes() == original
    actual = np.load(target, allow_pickle=False)
    assert actual.shape == existing.shape
    assert actual.dtype == existing.dtype
    assert np.array_equal(actual, existing)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["field.npy"]
