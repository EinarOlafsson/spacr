"""Public image imports preserve scientific input bytes across filesystem recovery."""
from __future__ import annotations

import errno
import os
import re
import shutil

import numpy as np
import pytest
import tifffile

from spacr.image_import import apply_import, plan_import
from spacr.utils import _get_regex


@pytest.fixture
def acquisition(tmp_path):
    source = tmp_path / "acquired"
    source.mkdir()
    names = ["plate1_A01_T0001F001L01A01Z01C01.tif",
             "plate1_A01_T0001F002L01A01Z01C01.tif"]
    arrays = {name: np.arange(256, dtype=np.uint16).reshape(16, 16) + offset
              for name, offset in zip(names, [0, 1000])}
    for name, array in arrays.items():
        tifffile.imwrite(source / name, array, metadata={"axes": "YX"})
    original = {name: (source / name).read_bytes() for name in names}
    plan = plan_import(source)
    assert not plan.problems()
    assert {entry["field"] for entry in plan.files.values()} == {1, 2}
    parser = re.compile(_get_regex("cellvoyager", "tif"))
    return source, plan, arrays, original, parser


def test_existing_canonical_destinations_are_replaced_without_changing_sources(
        acquisition, tmp_path):
    source, plan, arrays, original, parser = acquisition
    destination = tmp_path / "imported"
    destination.mkdir()
    first = next(iter(arrays))
    (destination / first).write_bytes(b"an earlier import")
    untouched = destination / "operator-notes.txt"
    untouched.write_bytes(b"preserve unrelated files")

    result = apply_import(plan, destination)

    assert result.written == 2 and not result.skipped
    assert untouched.read_bytes() == b"preserve unrelated files"
    for name, pixels in arrays.items():
        assert (source / name).read_bytes() == original[name]
        assert (destination / name).read_bytes() == original[name]
        np.testing.assert_array_equal(tifffile.imread(destination / name), pixels)
        parsed = parser.fullmatch(name).groupdict()
        assert (parsed["wellID"], int(parsed["fieldID"]), int(parsed["chanID"])) == (
            "A01", list(arrays).index(name) + 1, 1)
    assert {p.name for p in destination.iterdir()} == set(arrays) | {untouched.name}


@pytest.mark.parametrize("fail_one_copy", [False, True])
def test_unavailable_links_copy_images_and_report_a_single_failed_source(
        acquisition, tmp_path, monkeypatch, fail_one_copy):
    source, plan, arrays, original, parser = acquisition
    destination = tmp_path / "imported"
    destination.mkdir()
    untouched = destination / "operator-notes.txt"
    untouched.write_bytes(b"preserve unrelated files")
    refused_name = next(iter(arrays)) if fail_one_copy else None
    real_copy = shutil.copy2
    attempts = []

    def no_symlink(*args, **kwargs):
        raise OSError(errno.ENOTSUP, "symbolic links unavailable")

    def no_hardlink(*args, **kwargs):
        raise OSError(errno.EXDEV, "cross-device hard link")

    def copy_or_report_full_disk(src, dst, *args, **kwargs):
        attempts.append(src.name)
        if src.name == refused_name:
            raise OSError(errno.ENOSPC, "destination disk full")
        return real_copy(src, dst, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(os, "symlink", no_symlink)
        patch.setattr(os, "link", no_hardlink)
        patch.setattr(shutil, "copy2", copy_or_report_full_disk)
        result = apply_import(plan, destination)

    expected = set(arrays) - ({refused_name} if refused_name else set())
    assert attempts == sorted(arrays)
    assert result.written == len(expected)
    assert result.linked == 0 and result.bytes_saved == 0
    assert set(result.skipped) == ({refused_name} if refused_name else set())
    if refused_name:
        assert "destination disk full" in result.skipped[refused_name]
        assert refused_name in result.summary()
        assert "1 file(s) NOT written" in result.summary()
        assert not (destination / refused_name).exists()
    assert untouched.read_bytes() == b"preserve unrelated files"
    assert {p.name for p in destination.iterdir()} == expected | {untouched.name}
    for name, pixels in arrays.items():
        assert (source / name).read_bytes() == original[name]
        if name in expected:
            output = destination / name
            assert output.is_file() and not output.is_symlink()
            assert not os.path.samefile(output, source / name)
            assert output.read_bytes() == original[name]
            np.testing.assert_array_equal(tifffile.imread(output), pixels)
            parsed = parser.fullmatch(output.name).groupdict()
            assert (parsed["wellID"], int(parsed["fieldID"]), int(parsed["chanID"])) == (
                "A01", list(arrays).index(name) + 1, 1)
