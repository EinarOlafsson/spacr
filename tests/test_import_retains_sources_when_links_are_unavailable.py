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


@pytest.mark.parametrize("relationship", ["same_root", "nested_input", "path_alias"])
def test_input_entry_alias_is_refused_before_filesystem_mutation(
        acquisition, tmp_path, monkeypatch, relationship):
    """Public plans must never replace their own raw directory entries."""
    from pathlib import Path

    source, plan, arrays, original, _parser = acquisition
    if relationship == "nested_input":
        plan = plan_import(tmp_path)
        assert not plan.problems()
    destination = source / ".." / source.name if relationship == "path_alias" else source
    before = {name: (path.resolve(), path.stat().st_dev, path.stat().st_ino,
                     path.stat().st_mode)
              for name in arrays for path in [source / name]}
    mutations = []

    def watching(name, real, path_index):
        """Record real filesystem calls under this generated acquisition only."""
        def call(*args, **kwargs):
            path = Path(args[path_index])
            if str(path).startswith(str(tmp_path)):
                mutations.append((name, str(path)))
            return real(*args, **kwargs)
        return call

    refusal = None
    with monkeypatch.context() as patch:
        patch.setattr(Path, "mkdir", watching("mkdir", Path.mkdir, 0))
        patch.setattr(Path, "unlink", watching("unlink", Path.unlink, 0))
        patch.setattr(os, "symlink", watching("symlink", os.symlink, 1))
        patch.setattr(os, "link", watching("hardlink", os.link, 1))
        patch.setattr(shutil, "copy2", watching("copy", shutil.copy2, 1))
        try:
            apply_import(plan, destination)
        except ValueError as exc:
            refusal = exc

    kinds = {name: "symlink" if (source / name).is_symlink() else "file"
             for name in arrays}
    assert refusal is not None, (
        f"import did not refuse {relationship}; operations={mutations}; raw kinds={kinds}")
    assert "input" in str(refusal).lower() or "source" in str(refusal).lower()
    assert str(destination / next(iter(arrays))) in str(refusal)
    assert mutations == []
    assert {p.name for p in source.iterdir()} == set(arrays)
    for name in arrays:
        path = source / name
        assert not path.is_symlink()
        assert path.read_bytes() == original[name]
        assert (path.resolve(), path.stat().st_dev, path.stat().st_ino,
                path.stat().st_mode) == before[name]


@pytest.mark.parametrize("placement", ["nested", "sibling"])
def test_non_aliasing_nested_import_and_linked_reimport_remain_supported(
        acquisition, placement):
    """Nested headless outputs and an existing output link do not replace inputs."""
    source, plan, arrays, original, parser = acquisition
    destination = (source / "derived_project" if placement == "nested"
                   else source.parent / "derived_project")
    first_counts = None
    for _repeat in range(2):
        result = apply_import(plan, destination)
        assert result.written == 2 and result.skipped == {}
        counts = (result.written, result.linked, result.bytes_saved, result.stitched)
        if first_counts is None:
            first_counts = counts
        else:
            assert counts == first_counts
        assert set(p.name for p in destination.iterdir()) == set(arrays)
        for name, pixels in arrays.items():
            assert (source / name).read_bytes() == original[name]
            assert (destination / name).read_bytes() == original[name]
            np.testing.assert_array_equal(tifffile.imread(destination / name), pixels)
            fields = parser.fullmatch(name).groupdict()
            assert fields["wellID"] == "A01" and int(fields["chanID"]) == 1


def _observe_import_writes(monkeypatch, tmp_path):
    """Observe real filesystem/pixel-writer calls without replacing their behavior."""
    from contextlib import contextmanager
    from pathlib import Path
    from spacr import tiff_io

    @contextmanager
    def observe():
        """Limit the mutation observation to the public import call."""
        calls = []

        def watch(label, real, position):
            """Record a generated test-path mutation and delegate the real operation."""
            def call(*args, **kwargs):
                path = Path(args[position])
                if str(path).startswith(str(tmp_path)):
                    calls.append((label, str(path)))
                return real(*args, **kwargs)
            return call

        with monkeypatch.context() as patch:
            patch.setattr(Path, "mkdir", watch("mkdir", Path.mkdir, 0))
            patch.setattr(Path, "unlink", watch("unlink", Path.unlink, 0))
            patch.setattr(os, "symlink", watch("symlink", os.symlink, 1))
            patch.setattr(os, "link", watch("hardlink", os.link, 1))
            patch.setattr(shutil, "copy2", watch("copy", shutil.copy2, 1))
            patch.setattr(tiff_io, "write_tiff", watch("write_tiff", tiff_io.write_tiff, 0))
            yield calls
    return observe()


@pytest.mark.parametrize("link", [False, True])
def test_cross_input_output_hardlink_is_detached_without_changing_inputs(
        acquisition, tmp_path, monkeypatch, link):
    """Both modes replace an output alias without writing through its old target."""
    source, plan, arrays, original, _parser = acquisition
    destination = tmp_path / "imported"
    destination.mkdir()
    first, second = list(arrays)
    os.link(source / second, destination / first)
    assert os.path.samefile(source / second, destination / first)
    with _observe_import_writes(monkeypatch, tmp_path) as calls:
        result = apply_import(plan, destination, link=link)
    assert result.written == 2 and result.skipped == {}
    assert result.linked == (2 if link else 0)
    if not link:
        assert result.bytes_saved == 0
    assert ("unlink", str(destination / first)) in calls
    if not link:
        assert calls.index(("unlink", str(destination / first))) < calls.index(
            ("copy", str(destination / first)))
    for name, pixels in arrays.items():
        assert (source / name).read_bytes() == original[name]
        assert (destination / name).read_bytes() == original[name]
        np.testing.assert_array_equal(tifffile.imread(destination / name), pixels)
        assert os.path.samefile(destination / name, source / name) == link


def test_copying_an_existing_own_source_link_retains_the_safe_skip(acquisition):
    """copy2's existing same-file no-op remains a per-file reported skip."""
    source, plan, arrays, original, _parser = acquisition
    result = apply_import(plan, source, link=False)
    assert result.written == 0 and result.linked == 0 and result.bytes_saved == 0
    assert set(result.skipped) == set(arrays)
    for name in arrays:
        assert not (source / name).is_symlink()
        assert (source / name).read_bytes() == original[name]


def test_failed_output_unlink_cannot_fall_back_through_another_input(
        acquisition, tmp_path, monkeypatch):
    """A permission failure retains the old output and imports the other field."""
    from pathlib import Path

    source, plan, arrays, original, _parser = acquisition
    first, second = list(arrays)
    destination = tmp_path / "imported"
    destination.mkdir()
    protected_output = destination / first
    os.link(source / second, protected_output)
    real_unlink = Path.unlink
    real_copy = shutil.copy2
    copies = []

    def refuse_one_unlink(path, *args, **kwargs):
        """Represent an actual unremovable output directory entry."""
        if path == protected_output:
            raise PermissionError(errno.EACCES, "cannot remove existing output")
        return real_unlink(path, *args, **kwargs)

    def observe_copy(src, dst, *args, **kwargs):
        """Delegate copies while checking that unsafe fallback was never attempted."""
        copies.append(str(dst))
        return real_copy(src, dst, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", refuse_one_unlink)
        patch.setattr(shutil, "copy2", observe_copy)
        result = apply_import(plan, destination)
    changed = [name for name in arrays if (source / name).read_bytes() != original[name]]
    assert changed == [], f"failed-unlink fallback changed raw inputs={changed}"
    assert copies == []
    assert result.written == 1 and set(result.skipped) == {first}
    assert "cannot remove existing output" in result.skipped[first]
    assert os.path.samefile(protected_output, source / second)
    assert (destination / second).read_bytes() == original[second]


def test_stitched_tiff_detaches_an_input_hardlink_but_keeps_unshared_output(
        tmp_path, monkeypatch):
    """Stitching preserves raw tiles and the legacy unshared-output write mode."""
    source = tmp_path / "tiled"
    names = ["plate1_A01_T0001F001L01A01Z01C01.tif",
             "plate1_A01_T0001F002L01A01Z01C01.tif"]
    for tile in (1, 2):
        folder = source / f"tile{tile}"
        folder.mkdir(parents=True)
        for field, name in enumerate(names):
            pixels = np.arange(256, dtype=np.uint16).reshape(16, 16) + tile * 1000 + field * 100
            tifffile.imwrite(folder / name, pixels, metadata={"axes": "YX"})
    plan = plan_import(source)
    assert not plan.problems()
    assert {entry["tile"] for entry in plan.files.values()} == {1, 2}
    original = {p: p.read_bytes() for p in source.rglob("*.tif")}
    destination = tmp_path / "imported"
    destination.mkdir()
    os.link(source / "tile1" / names[0], destination / names[0])
    regular = destination / names[1]
    tifffile.imwrite(regular, np.full((5, 7), 4321, dtype=np.uint16))
    regular_original = regular.read_bytes()
    regular_inode = (regular.stat().st_dev, regular.stat().st_ino)
    with _observe_import_writes(monkeypatch, tmp_path) as calls:
        result = apply_import(plan, destination)
    assert result.written == 2 and result.stitched == 2 and result.skipped == {}
    assert all(p.read_bytes() == data for p, data in original.items())
    assert not os.path.samefile(destination / names[0], source / "tile1" / names[0])
    assert ("unlink", str(destination / names[0])) in calls
    assert ("unlink", str(regular)) not in calls
    assert (regular.stat().st_dev, regular.stat().st_ino) == regular_inode
    assert regular.read_bytes() != regular_original
    assert tifffile.imread(regular).shape != (5, 7)
    assert {p.name for p in destination.iterdir()} == set(names)
    for name in names:
        image = tifffile.imread(destination / name)
        assert image.ndim == 2 and image.dtype == np.uint16
        assert ("write_tiff", str(destination / name)) in calls


def test_copy_reimport_detaches_links_to_an_older_acquisition(
        acquisition, tmp_path):
    """Replacing a managed project must not modify a previous run outside the plan."""
    source, plan, arrays, original, _parser = acquisition
    older = tmp_path / "older_acquisition"
    older.mkdir()
    for index, name in enumerate(arrays):
        tifffile.imwrite(older / name, np.full((9, 11), 9000 + index, dtype=np.uint16),
                         metadata={"axes": "YX"})
    old_bytes = {name: (older / name).read_bytes() for name in arrays}
    destination = tmp_path / "imported"
    previous = apply_import(plan_import(older), destination)
    assert previous.written == 2 and not previous.skipped
    assert all(os.path.samefile(destination / name, older / name) for name in arrays)

    result = apply_import(plan, destination, link=False)

    changed = [name for name in arrays if (older / name).read_bytes() != old_bytes[name]]
    assert changed == [], f"copy wrote through links into older acquisition={changed}"
    assert result.written == 2 and result.linked == 0 and result.bytes_saved == 0
    assert result.skipped == {}
    for name, pixels in arrays.items():
        assert (source / name).read_bytes() == original[name]
        assert (destination / name).read_bytes() == original[name]
        assert not (destination / name).is_symlink()
        assert not os.path.samefile(destination / name, older / name)
        assert not os.path.samefile(destination / name, source / name)
        np.testing.assert_array_equal(tifffile.imread(destination / name), pixels)
