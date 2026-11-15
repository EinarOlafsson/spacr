"""The folder walk survives entries and folders the operating system refuses."""

from __future__ import annotations

import os

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
