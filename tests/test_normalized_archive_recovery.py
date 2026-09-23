"""Archive recovery retains missing-field evidence until the data is rebuilt."""
from __future__ import annotations

import io
import struct
import zipfile

import numpy as np
import pytest

from spacr import io as storage


def _archive(path, names=("plate_A01_1_1.npy",), *, data=None):
    if data is None:
        data = np.ones((len(names), 5, 7, 2), np.float32)
    np.savez_compressed(path, data=data, filenames=np.asarray(names))
    return path


def _npy_bytes(array, version=(1, 0)):
    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, version=version, allow_pickle=False)
    return buffer.getvalue()


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_short_array_payload_is_rejected_even_with_a_complete_zip(tmp_path, compression, version):
    path = tmp_path / "stack_0_norm.npz"
    pixels = _npy_bytes(np.ones((2, 5, 7, 2), np.float32), version)
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        archive.writestr("data.npy", pixels[:-16])
        archive.writestr("filenames.npy", _npy_bytes(np.array(["a.npy", "b.npy"])))
    original = path.read_bytes()
    with zipfile.ZipFile(path) as archive:
        assert archive.testzip() is None
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and "truncated" in reason
    assert fields is None and planes is None
    assert path.read_bytes() == original


@pytest.mark.parametrize("count", [0, 1, 3])
def test_archive_refuses_a_different_number_of_field_names_and_images(tmp_path, count):
    path = _archive(tmp_path / "batch.npz", [f"field_{i}.npy" for i in range(count)],
                    data=np.ones((2, 5, 7, 2), np.float32))
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and "filenames" in reason
    assert fields is None and planes is None


@pytest.mark.parametrize("names", [np.array("a.npy"), np.array([["a.npy", "b.npy"]])])
def test_field_names_must_be_a_vector_not_a_scalar_or_matrix(tmp_path, names):
    path = tmp_path / "batch.npz"
    np.savez_compressed(path, data=np.ones((names.size, 5, 7, 2), np.float32), filenames=names)
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and "filenames" in reason
    assert fields is None and planes is None


def test_preprocessing_off_keeps_refusing_a_quarantined_archive_on_rerun(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    good = _archive(masks / "stack_0_norm.npz")
    good_before = good.read_bytes()
    damaged = masks / "stack_1_norm.npz"
    damaged.write_bytes(b"interrupted zip")
    for _ in range(2):
        with pytest.raises(FileNotFoundError, match="stack_1_norm.npz"):
            storage._check_archives_without_preprocessing(str(tmp_path))
    assert (masks / "stack_1_norm.npz.damaged").read_bytes() == b"interrupted zip"
    assert good.read_bytes() == good_before


def test_preprocessing_off_reports_all_earlier_quarantines_without_raw_data(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    for name in ("stack_0_norm.npz.damaged", "stack_1_norm.npz.damaged.2"):
        (masks / name).write_bytes(b"retained evidence")
    with pytest.raises(FileNotFoundError) as caught:
        storage._check_archives_without_preprocessing(str(tmp_path))
    message = str(caught.value)
    assert "stack_0_norm.npz" in message and "stack_1_norm.npz" in message
    assert "Neither stack/ nor the raw images" in message
    assert len(list(masks.iterdir())) == 2


@pytest.mark.parametrize("rebuilt_name", ["stack_0_norm.npz", "stack_5_norm.npz"])
def test_successfully_rebuilt_archives_allow_preprocessing_off_with_quarantine_retained(
        tmp_path, rebuilt_name):
    masks = tmp_path / "masks"
    masks.mkdir()
    stack = tmp_path / "stack"
    stack.mkdir()
    field = "plate_A01_1_1.npy"
    np.save(stack / field, np.ones((5, 7, 2), np.uint16))
    (masks / "stack_0_norm.npz.damaged").write_bytes(b"old evidence")
    _archive(masks / rebuilt_name, [field])
    assert storage._check_archives_without_preprocessing(str(tmp_path)) == [rebuilt_name]
    assert (masks / "stack_0_norm.npz.damaged").read_bytes() == b"old evidence"


def test_earlier_quarantine_is_not_resolved_by_only_some_rebuilt_fields(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    stack = tmp_path / "stack"
    stack.mkdir()
    for field in ("plate_A01_1_1.npy", "plate_A01_2_1.npy"):
        np.save(stack / field, np.ones((5, 7, 2), np.uint16))
    (masks / "stack_0_norm.npz.damaged").write_bytes(b"old evidence")
    _archive(masks / "stack_5_norm.npz", ["plate_A01_1_1.npy"])
    with pytest.raises(FileNotFoundError, match="Turn preprocess on"):
        storage._check_archives_without_preprocessing(str(tmp_path))


@pytest.mark.parametrize("version", [(1, 0), (2, 0)])
def test_complete_arrays_keep_exact_field_order_and_channel_count(tmp_path, version):
    path = tmp_path / "batch.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", _npy_bytes(np.ones((2, 5, 7, 3), np.float32), version))
        archive.writestr("filenames.npy", _npy_bytes(np.array(["b.npy", "a.npy"])))
    assert storage._inspect_normalized_archive(str(path)) == (True, "done", ("b", "a"), 3)


def test_object_pixels_are_refused_without_unpickling(tmp_path):
    path = _archive(tmp_path / "batch.npz", data=np.zeros((1, 5, 7, 2), object))
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and "unpickling" in reason
    assert fields is None and planes is None


def test_legacy_object_names_remain_unlisted_and_do_not_resolve_a_quarantine(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    stack = tmp_path / "stack"
    stack.mkdir()
    field = "plate_A01_1_1.npy"
    np.save(stack / field, np.ones((5, 7, 2), np.uint16))
    path = _archive(masks / "stack_5_norm.npz", np.array([field], dtype=object))
    (masks / "stack_0_norm.npz.damaged").write_bytes(b"old evidence")
    assert storage._inspect_normalized_archive(str(path)) == (True, "done", None, 2)
    with pytest.raises(FileNotFoundError, match="stack_0_norm.npz"):
        storage._check_archives_without_preprocessing(str(tmp_path))


def test_replacing_the_damaged_archive_itself_resolves_its_quarantine(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "stack_0_norm.npz.damaged").write_bytes(b"old evidence")
    _archive(masks / "stack_0_norm.npz")
    assert storage._check_archives_without_preprocessing(str(tmp_path)) == ["stack_0_norm.npz"]


@pytest.mark.parametrize("damage", ["short_pixels", "missing_names"])
def test_real_preprocessing_rebuilds_structurally_invalid_archives(
        yokogawa_cellvoyager_dir, damage):
    from tests.test_a_rerun_trusts_nothing_a_killed_run_left import (
        _archive_fields, _preprocess, _stack_names,
    )

    plate = yokogawa_cellvoyager_dir["src"]
    _preprocess(plate)
    masks = plate / "masks"
    broken = masks / "stack_2_norm.npz"
    intact = {p: p.read_bytes() for p in masks.glob("*.npz") if p != broken}
    with zipfile.ZipFile(broken) as archive:
        pixels = archive.read("data.npy")
        names = archive.read("filenames.npy")
    if damage == "short_pixels":
        pixels = pixels[:-16]
    else:
        names = _npy_bytes(np.array([], dtype="U1"))
    with zipfile.ZipFile(broken, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("data.npy", pixels)
        archive.writestr("filenames.npy", names)
    damaged_bytes = broken.read_bytes()
    _preprocess(plate)
    assert _archive_fields(masks) == _stack_names(plate / "stack")
    assert (masks / "stack_2_norm.npz.damaged").read_bytes() == damaged_bytes
    assert all(path.read_bytes() == before for path, before in intact.items())
    assert sorted(storage._check_archives_without_preprocessing(str(plate))) == sorted(
        path.name for path in masks.glob("*.npz"))


@pytest.mark.parametrize("damage", ["missing", "empty", "missing_data", "missing_filenames",
                                    "invalid_data", "invalid_names"])
def test_unreadable_archive_components_report_damage_without_modifying_files(tmp_path, damage):
    path = tmp_path / "batch.npz"
    if damage == "empty":
        path.touch()
    elif damage != "missing":
        with zipfile.ZipFile(path, "w") as archive:
            if damage != "missing_data":
                archive.writestr("data.npy", b"invalid header" if damage == "invalid_data"
                                 else _npy_bytes(np.ones((1, 5, 7, 2), np.float32)))
            if damage != "missing_filenames":
                archive.writestr("filenames.npy", b"invalid header" if damage == "invalid_names"
                                 else _npy_bytes(np.array(["a.npy"])))
    before = path.read_bytes() if path.exists() else None
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and reason and fields is None and planes is None
    assert (path.read_bytes() if path.exists() else None) == before


def test_zip_member_extent_beyond_the_file_is_reported_before_reading(tmp_path):
    path = _archive(tmp_path / "batch.npz")
    encoded = bytearray(path.read_bytes())
    directory = encoded.index(b"PK\x01\x02")
    struct.pack_into("<I", encoded, directory + 20, len(encoded) * 2)
    path.write_bytes(encoded)
    ok, reason, fields, planes = storage._inspect_normalized_archive(str(path))
    assert not ok and "member runs past the end" in reason
    assert fields is None and planes is None
    assert path.read_bytes() == bytes(encoded)


def test_raw_images_offer_rebuilding_when_only_quarantine_evidence_remains(tmp_path):
    import tifffile

    masks = tmp_path / "masks"
    masks.mkdir()
    (masks / "stack_0_norm.npz.damaged").write_bytes(b"old evidence")
    tifffile.imwrite(tmp_path / "raw.tif", np.zeros((5, 7), np.uint16))
    with pytest.raises(FileNotFoundError, match="built again from the raw images"):
        storage._check_archives_without_preprocessing(str(tmp_path))


def test_missing_field_without_quarantine_is_reported_without_claiming_it_was_processed(
        tmp_path, capsys):
    masks = tmp_path / "masks"
    masks.mkdir()
    stack = tmp_path / "stack"
    stack.mkdir()
    np.save(stack / "missing.npy", np.ones((5, 7, 2), np.uint16))
    _archive(masks / "stack_0_norm.npz")
    assert storage._check_archives_without_preprocessing(str(tmp_path)) == ["stack_0_norm.npz"]
    output = capsys.readouterr().out
    assert "missing" in output and "get no masks" in output
    assert "turn preprocess on" in output


def test_legacy_object_names_without_quarantine_still_allow_existing_archives(tmp_path):
    masks = tmp_path / "masks"
    masks.mkdir()
    _archive(masks / "stack_0_norm.npz", np.array(["a.npy"], dtype=object))
    assert storage._check_archives_without_preprocessing(str(tmp_path)) == ["stack_0_norm.npz"]
