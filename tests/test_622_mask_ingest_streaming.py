"""Raw plate preprocessing must write/release fields before reading more."""
import weakref

import numpy as np
import pytest
import tifffile

from spacr import io

PATTERN = (r"(?P<plateID>plate)_(?P<wellID>A01)_T(?P<timeID>\d+)"
           r"F(?P<fieldID>\d+)Z\d+C(?P<chanID>\d+)\.tif")


def _plate(folder, fields=8):
    for field in range(fields):
        for channel in (1, 2):
            for z in (1, 2, 3):
                name = f"plate_A01_T1F{field:03}Z{z}C{channel}.tif"
                tifffile.imwrite(folder / name, np.full(
                    (32, 40), 100 * field + 10 * channel + z, np.uint16))


def test_memory_is_field_bounded_even_with_large_batch_and_many_z_slices(
        tmp_path, monkeypatch):
    _plate(tmp_path)
    real_load = io.load_images_from_paths
    real_save = io._save_array_atomic
    references = []
    written = []

    def load(paths):
        # A regex without sliceID must not collect an entire z-series.
        assert sum(map(len, paths.values())) == 1
        key = next(iter(paths))
        field = int(key[2])
        assert len(written) == field, "next field read before previous was saved"
        result = real_load(paths)
        for arrays in result.values():
            references.extend(weakref.ref(array) for array in arrays)
        assert sum(ref() is not None for ref in references) <= 3
        return result

    def save(path, array):
        field = len(written)
        assert array.shape == (32, 40, 2)
        np.testing.assert_array_equal(array[0, 0], [100*field + 13, 100*field + 23])
        real_save(path, array)
        written.append(path)

    monkeypatch.setattr(io, "load_images_from_paths", load)
    monkeypatch.setattr(io, "_save_array_atomic", save)
    assert io._rename_and_organize_image_files(
        str(tmp_path), PATTERN, batch_size=10000, metadata_type="custom",
        save_original_images=True) == 2
    assert len(written) == 8
    assert not any(ref() is not None for ref in references)
    assert len(list((tmp_path / "orig").glob("*.tif"))) == 48


def test_interrupted_plate_preserves_raws_and_resumes_completed_fields(
        tmp_path, monkeypatch):
    _plate(tmp_path, fields=3)
    real_save = io._save_array_atomic
    completed = []

    def interrupted(path, array):
        if completed:
            raise RuntimeError("simulated interruption")
        real_save(path, array)
        completed.append(path)

    monkeypatch.setattr(io, "_save_array_atomic", interrupted)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        io._rename_and_organize_image_files(str(tmp_path), PATTERN,
                                           metadata_type="custom")
    assert len(list(tmp_path.glob("*.tif"))) == 18
    assert len(list((tmp_path / "stack").glob("*.npy"))) == 1
    first_bytes = (tmp_path / "stack" / "plate_A01_0_1.npy").read_bytes()
    monkeypatch.setattr(io, "_save_array_atomic", real_save)
    io._rename_and_organize_image_files(str(tmp_path), PATTERN,
                                       metadata_type="custom")
    assert len(list((tmp_path / "stack").glob("*.npy"))) == 3
    assert (tmp_path / "stack" / "plate_A01_0_1.npy").read_bytes() == first_bytes
