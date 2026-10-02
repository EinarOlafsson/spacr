"""Rejected archive destinations must not modify source or prior package data."""
from pathlib import Path

import pytest

from spacr import report


@pytest.fixture
def source(tmp_path):
    folder = tmp_path / "screen"
    folder.mkdir()
    (folder / "plate1_A01_T0001F001.tif").write_bytes(b"source pixels")
    (folder / "README.txt").write_bytes(b"original experiment notes")
    return folder


def snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes()
            for path in folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize("location", ["same", "nested", "source_alias", "leaf_alias"])
def test_source_overlap_is_rejected_before_creating_or_overwriting_files(source, tmp_path, location):
    before = snapshot(source)
    if location == "same":
        out, title = source.parent, source.name
    elif location == "nested":
        out, title = source / "new" / "archives", "package"
    elif location == "source_alias":
        alias = tmp_path / "alias"
        alias.symlink_to(source, target_is_directory=True)
        out, title = alias, "package"
    else:
        out, title = tmp_path / "exports", "package"
        out.mkdir()
        (out / title).symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the source"):
        report._write_archive_package(source, out, {"title": title}, copy_images=True)
    assert snapshot(source) == before
    assert sorted(path.name for path in source.iterdir()) == sorted(before)


@pytest.mark.parametrize("existing_kind", ["folder", "file", "dangling_symlink"])
def test_existing_destination_is_preserved(source, tmp_path, existing_kind):
    out = tmp_path / "exports"
    out.mkdir()
    package = out / "package"
    if existing_kind == "folder":
        package.mkdir()
        (package / "README.txt").write_bytes(b"prior package notes")
        (package / "idr").symlink_to(source, target_is_directory=True)
    elif existing_kind == "file":
        package.write_bytes(b"existing user file")
    else:
        package.symlink_to(tmp_path / "uncreated", target_is_directory=True)
    before = snapshot(source)
    with pytest.raises(ValueError, match="already exists"):
        report._write_archive_package(source, out, {"title": "package"})
    assert snapshot(source) == before
    if existing_kind == "folder":
        assert (package / "README.txt").read_bytes() == b"prior package notes"
        assert sorted(path.name for path in package.iterdir()) == ["README.txt", "idr"]
    elif existing_kind == "file":
        assert package.read_bytes() == b"existing user file"
    else:
        assert package.is_symlink() and not package.exists()
        assert not (tmp_path / "uncreated").exists()


def test_fresh_sibling_package_still_builds_and_keeps_source_bytes(source, tmp_path):
    before = snapshot(source)
    package = report._write_archive_package(source, tmp_path / "exports", {
        "title": "package", "description": "test screen", "authors": "Doe Jane",
        "email": "jane@example.org", "affiliation": "Test institute"})
    assert package == Path(tmp_path / "exports/package")
    assert (package / "archive_manifest.json").is_file()
    assert snapshot(source) == before
