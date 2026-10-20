"""Converted organize inputs retain every original and its recorded location."""
import csv
import errno
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import channel_sorting as cs


def _image(path, value=31):
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.full((20, 30, 3), value, dtype=np.uint8),
                     photometric="rgb")
    return path


def test_duplicate_basename_originals_remain_byte_identical_and_manifested(tmp_path):
    paths = [_image(tmp_path / str(i) / "field.tif", value)
             for i, value in enumerate((31, 173))]
    expected = {str(path): path.read_bytes() for path in paths}
    sets = {(("set", f"{i + 1:06d}"),): {1: str(path)}
            for i, path in enumerate(paths)}
    plan = cs.build_plan(str(tmp_path), sets, convert=True, masks={})
    assert plan.ok, plan.problems
    result = cs.apply_plan(plan, merge=False, log=lambda text: None)
    assert result.moved == 2
    with open(result.manifest, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    original_rows = [row for row in rows if row["kind"] == "original"]
    assert len(original_rows) == 2
    assert len({row["new_path"] for row in original_rows}) == 2
    for row in original_rows:
        saved = Path(row["new_path"])
        assert saved.name == "field.tif"
        assert saved.parent.parent == Path(result.dest) / "originals"
        assert saved.read_bytes() == expected[row["original_path"]]
        assert row["status"] == "moved"
    assert len([row for row in rows if row["status"] == "converted rgb"]) == 2
    from spacr.original_filenames import _load_map
    map_kind, entries, image_count = _load_map(
        Path(result.manifest).read_bytes(), result.manifest)
    assert map_kind == "channel_sorting_manifest"
    assert image_count == 2
    assert len(entries) == 2
    assert {name for destination, key, names in entries for name in names} == set(expected)
    assert {destination for destination, key, names in entries} == {
        row.target_image for row in plan.rows}
    assert all(not path.exists() for path in paths)
    for row, value in zip(plan.rows, (31, 173)):
        converted = tifffile.imread(row.target_image)
        assert converted.shape == (20, 30)
        assert np.all(converted == value)


@pytest.mark.parametrize("kind", [
    "regular", pytest.param("symlink", marks=pytest.mark.needs_symlink)])
def test_existing_original_name_is_preserved(tmp_path, kind):
    source = _image(tmp_path / "input" / "field.tif")
    before = source.read_bytes()
    dest = tmp_path / "output"
    originals = dest / "originals"
    originals.mkdir(parents=True)
    existing = originals / source.name
    foreign = tmp_path / "foreign.tif"
    foreign.write_bytes(b"unrelated bytes")
    if kind == "symlink":
        existing.symlink_to(foreign)
    else:
        existing.write_bytes(b"unrelated bytes")
    saved = Path(cs._write_converted(str(source), str(dest / "converted.tif"),
                                    "rgb", str(dest)))
    assert saved != existing
    assert saved.read_bytes() == before
    assert existing.read_bytes() == b"unrelated bytes"
    assert existing.is_symlink() == (kind == "symlink")
    assert foreign.read_bytes() == b"unrelated bytes"


def test_failed_move_preserves_source_and_removes_only_empty_private_directory(tmp_path, monkeypatch):
    source = _image(tmp_path / "input" / "field.tif")
    before = source.read_bytes()
    dest = tmp_path / "output"
    dest.mkdir()
    def fail(source_path, target_path):
        raise OSError("interrupted before transfer")
    monkeypatch.setattr(cs.shutil, "move", fail)
    with pytest.raises(OSError, match="interrupted"):
        cs._write_converted(str(source), str(dest / "converted.tif"), "rgb", str(dest))
    assert source.read_bytes() == before
    assert list((dest / "originals").iterdir()) == []


def test_exception_after_completed_move_keeps_the_only_original(tmp_path, monkeypatch):
    source = _image(tmp_path / "input" / "field.tif")
    before = source.read_bytes()
    dest = tmp_path / "output"
    dest.mkdir()
    actual_move = cs.shutil.move
    def interrupted_after_move(source_path, target_path):
        actual_move(source_path, target_path)
        raise OSError("interrupted after transfer")
    monkeypatch.setattr(cs.shutil, "move", interrupted_after_move)
    with pytest.raises(OSError, match="interrupted"):
        cs._write_converted(str(source), str(dest / "converted.tif"), "rgb", str(dest))
    assert not source.exists()
    kept = list((dest / "originals").rglob("field.tif"))
    assert len(kept) == 1
    assert kept[0].read_bytes() == before


def test_cross_device_move_fallback_preserves_original_bytes_and_mode(tmp_path, monkeypatch):
    source = _image(tmp_path / "input" / "field.tif")
    source.chmod(0o640)
    before = source.read_bytes()
    dest = tmp_path / "output"
    dest.mkdir()
    actual_rename = cs.os.rename
    attempted = []
    def cross_device(source_path, target_path, *args, **kwargs):
        if str(source_path) == str(source):
            attempted.append((source_path, target_path))
            raise OSError(errno.EXDEV, "different filesystem")
        return actual_rename(source_path, target_path, *args, **kwargs)
    monkeypatch.setattr(cs.os, "rename", cross_device)
    saved = Path(cs._write_converted(str(source), str(dest / "converted.tif"),
                                    "rgb", str(dest)))
    assert len(attempted) == 1
    assert not source.exists()
    assert saved.read_bytes() == before
    if os.name == "posix":
        assert saved.stat().st_mode & 0o777 == 0o640


@pytest.mark.posix_only
def test_valid_unicode_source_basename_does_not_expand_private_directory_name(tmp_path):
    source = _image(tmp_path / "input" / ("🧬" * 62 + ".tif"))
    before = source.read_bytes()
    dest = tmp_path / "output"
    dest.mkdir()
    saved = Path(cs._write_converted(str(source), str(dest / "converted.tif"),
                                    "rgb", str(dest)))
    assert saved.name == source.name
    assert saved.parent.name.startswith("spacr-original-")
    assert saved.read_bytes() == before
