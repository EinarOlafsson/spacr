"""The folder walk survives entries and folders the operating system refuses."""

from __future__ import annotations

import os
import types

from spacr.qt import folder_metadata as fm


def test_the_walk_yields_images_in_every_subfolder_and_honours_the_cap(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "deep").mkdir()
    for rel in ("top.tif", "a/one.png", "a/deep/two.tif", "a/notes.txt"):
        (tmp_path / rel).write_bytes(b"x")
    found = sorted(os.path.basename(p) for p in fm._iter_image_paths(tmp_path))
    assert found == ["one.png", "top.tif", "two.tif"]
    assert len(list(fm._iter_image_paths(tmp_path, cap=2))) == 2
    assert len([p.name for p in fm.iter_image_files(tmp_path, cap=1)]) == 1


def test_an_entry_that_cannot_be_statted_is_skipped_not_fatal(monkeypatch,
                                                              tmp_path):
    class Bad:
        name = "bad.tif"
        path = str(tmp_path / "bad.tif")

        def is_file(self):
            raise OSError("stale handle")

    class Good:
        name = "good.tif"
        path = str(tmp_path / "good.tif")

        def is_file(self):
            return True

    class Dir:
        name = "dir.tif"
        path = str(tmp_path / "dir.tif")

        def is_file(self):
            return False

    monkeypatch.setattr(fm, "_listings_in_rglob_order",
                        lambda root: iter([[Bad(), Dir(), Good()]]))
    assert list(fm._iter_image_paths(tmp_path)) == [Good.path]


def test_a_folder_that_cannot_be_listed_is_an_empty_folder(tmp_path):
    missing = tmp_path / "gone"
    assert list(fm._listings_in_rglob_order(missing)) == [[]]


def test_an_unreadable_subfolder_and_an_unclassifiable_entry_are_skipped(
        monkeypatch, tmp_path):
    (tmp_path / "ok").mkdir()
    (tmp_path / "broken").mkdir()
    real_scandir = os.scandir
    seen = {}

    def scandir(folder):
        if os.path.basename(os.fspath(folder)) == "broken":
            raise PermissionError("no")
        return real_scandir(folder)

    monkeypatch.setattr(fm.os, "scandir", scandir)
    listings = list(fm._listings_in_rglob_order(tmp_path))
    names = sorted(sorted(e.name for e in entries) for entries in listings)
    assert [] in names and ["broken", "ok"] in names
    seen["n"] = len(listings)
    assert seen["n"] == 3

    def top(folder):
        entry = types.SimpleNamespace(
            name="weird", path=str(tmp_path / "weird"),
            is_dir=lambda follow_symlinks=False: (_ for _ in ()).throw(
                OSError("cannot tell")))
        return iter([entry]) if False else _Scan([entry])

    class _Scan(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(fm.os, "scandir", top)
    assert len(list(fm._listings_in_rglob_order(tmp_path))) == 1


def test_a_folder_listed_twice_is_listed_again_rather_than_lost(monkeypatch,
                                                                tmp_path):
    sub = str(tmp_path / "sub")

    def entry(name, path, is_dir):
        return types.SimpleNamespace(
            name=name, path=path,
            is_dir=lambda follow_symlinks=False: is_dir)

    class Scan(list):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def scandir(folder):
        if os.fspath(folder) == str(tmp_path):
            return Scan([entry("sub", sub, True), entry("sub", sub, True)])
        return Scan([entry("x.tif", sub + "/x.tif", False)])

    monkeypatch.setattr(fm.os, "scandir", scandir)
    listings = list(fm._listings_in_rglob_order(tmp_path))
    assert len(listings) == 3
    assert listings[0][0].name == "sub"
