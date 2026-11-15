"""How a dropped folder is listed, and how missing well/field ids are minted.

``spacr.qt.folder_metadata`` walks a dropped plate folder once and renames its
images to the canonical ``<plate>_<well>_F<field>_C<chan>.tif``. Two units:

* ``_listings_in_rglob_order`` yields one listing per folder, top folder
  first, and survives a folder that cannot be listed (an empty listing), an
  entry whose type cannot be read (skipped) and a folder named twice (listed
  twice, never lost);
* ``assign_missing_fields`` mints wells and fields so no canonical name is
  minted twice: 999 fields fill a well and the thousandth starts the next,
  and a known field with no well makes every file its own well.
"""
from __future__ import annotations

import os
import types
from pathlib import Path

from spacr.qt import folder_metadata as fm


class _Scan(list):
    """A ``scandir`` result: a list that is also a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _entry(name, path, is_dir):
    """A directory entry whose ``is_dir`` answers ``is_dir`` or raises it."""
    def answer(follow_symlinks=False):
        assert follow_symlinks is False
        if isinstance(is_dir, Exception):
            raise is_dir
        return is_dir

    return types.SimpleNamespace(name=name, path=path, is_dir=answer)


def test_each_folder_is_listed_once_in_walk_order_whatever_refuses(
        subtests, monkeypatch, tmp_path):
    """Listings come top-down; unreadable folders are empty, odd entries skipped."""
    def names(listings):
        return [sorted(entry.name for entry in listing) for listing in listings]

    with subtests.test(msg="a missing folder is one empty listing"):
        assert list(fm._listings_in_rglob_order(tmp_path / "gone")) == [[]]

    with subtests.test(msg="a real tree is listed top-down, depth first"):
        root = tmp_path / "tree"
        for rel in ("a/deep", "b"):
            (root / rel).mkdir(parents=True)
        for rel in ("top.tif", "a/one.tif", "a/deep/two.tif", "b/three.tif"):
            (root / rel).write_bytes(b"x")
        listings = list(fm._listings_in_rglob_order(root))
        assert sorted(map(tuple, names(listings))) == sorted([
            ("a", "b", "top.tif"), ("deep", "one.tif"), ("three.tif",),
            ("two.tif",)])
        assert names(listings)[0] == ["a", "b", "top.tif"]
        assert len(listings) == 4

    with subtests.test(msg="an unreadable subfolder is an empty listing"), \
            monkeypatch.context() as patch:
        root = tmp_path / "perm"
        (root / "ok").mkdir(parents=True)
        (root / "broken").mkdir()
        (root / "ok" / "x.tif").write_bytes(b"x")
        real_scandir = os.scandir

        def scandir(folder):
            if os.path.basename(os.fspath(folder)) == "broken":
                raise PermissionError("no")
            return real_scandir(folder)

        patch.setattr(fm.os, "scandir", scandir)
        listings = names(fm._listings_in_rglob_order(root))
        assert listings[0] == ["broken", "ok"]
        assert sorted(listings[1:]) == [[], ["x.tif"]]

    with subtests.test(msg="an entry whose type cannot be read is skipped"), \
            monkeypatch.context() as patch:
        seen = []

        def scandir(folder):
            seen.append(os.fspath(folder))
            return _Scan([_entry("weird", str(tmp_path / "weird"),
                                 OSError("cannot tell"))])

        patch.setattr(fm.os, "scandir", scandir)
        listings = list(fm._listings_in_rglob_order(tmp_path))
        assert names(listings) == [["weird"]]
        assert seen == [str(tmp_path)]

    with subtests.test(msg="a folder named twice is listed twice"), \
            monkeypatch.context() as patch:
        sub = str(tmp_path / "sub")
        seen = []

        def scandir(folder):
            seen.append(os.fspath(folder))
            if os.fspath(folder) == str(tmp_path):
                return _Scan([_entry("sub", sub, True),
                              _entry("sub", sub, True)])
            return _Scan([_entry("x.tif", sub + "/x.tif", False)])

        patch.setattr(fm.os, "scandir", scandir)
        listings = list(fm._listings_in_rglob_order(tmp_path))
        assert names(listings) == [["sub", "sub"], ["x.tif"], ["x.tif"]]
        assert seen == [str(tmp_path), sub, sub, sub]


def test_minted_wells_and_fields_never_repeat_a_canonical_name(subtests):
    """Field counters roll over into the next well; known fields mint one well each."""
    with subtests.test(msg="999 fields fill a well, the 1000th starts the next"):
        names = [Path(f"img_{i:05d}.tif") for i in range(1002)]
        mappings = fm.assign_missing_fields(names, plate="plate1",
                                            have_well=False, have_field=False)
        assert len(mappings) == 1002
        picked = [(m.canonical, m.plate, m.well, m.field, m.channel)
                  for m in (mappings[0], mappings[1], mappings[998],
                            mappings[999], mappings[1001])]
        assert picked == [
            ("plate1_A01_F001_C01.tif", "plate1", "A01", 1, 1),
            ("plate1_A01_F002_C01.tif", "plate1", "A01", 2, 1),
            ("plate1_A01_F999_C01.tif", "plate1", "A01", 999, 1),
            ("plate1_A02_F001_C01.tif", "plate1", "A02", 1, 1),
            ("plate1_A02_F003_C01.tif", "plate1", "A02", 3, 1),
        ]
        assert {m.well for m in mappings[:999]} == {"A01"}
        assert {m.well for m in mappings[999:]} == {"A02"}
        assert len({m.canonical for m in mappings}) == 1002
        assert [m.original_path for m in mappings[:2]] == [
            "img_00000.tif", "img_00001.tif"]

    with subtests.test(msg="a known field makes every file its own well"):
        names = [Path(f"img_{i:03d}.tif") for i in range(3)]
        mappings = fm.assign_missing_fields(names, plate="plate1",
                                            have_well=False, have_field=True)
        assert [(m.canonical, m.well, m.field) for m in mappings] == [
            ("plate1_A01_F001_C01.tif", "A01", 1),
            ("plate1_A02_F001_C01.tif", "A02", 1),
            ("plate1_A03_F001_C01.tif", "A03", 1),
        ]

    with subtests.test(msg="known wells and fields come from the folders"):
        names = [Path("/d/H12/F002/img.tif"), Path("/d/b03/F007/img.tif")]
        mappings = fm.assign_missing_fields(names, plate="p9",
                                            have_well=True, have_field=True)
        assert [(m.canonical, m.well, m.field, m.original_path)
                for m in mappings] == [
            ("p9_H12_F002_C01.tif", "H12", 2, "/d/H12/F002/img.tif"),
            ("p9_B03_F007_C01.tif", "B03", 7, "/d/b03/F007/img.tif"),
        ]

    with subtests.test(msg="known wells with no fields count fields up"):
        names = [Path("/d/B03/img_a.tif"), Path("/d/B03/img_b.tif")]
        mappings = fm.assign_missing_fields(names, have_well=True,
                                            have_field=False)
        assert [m.canonical for m in mappings] == [
            "plate1_B03_F001_C01.tif", "plate1_B03_F002_C01.tif"]
