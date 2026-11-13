"""Table-driven tests of the raw-image organisers in ``spacr.io``.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture: ``_rename_and_organize_image_files`` (raw TIFFs ->
``stack/*.npy``), ``_move_to_chan_folder``, ``_merge_file`` /
``_merge_channels``, ``_concatenate_channel``, the per-channel movies,
``delete_empty_subdirectories`` and ``process_non_tif_non_2D_images``.
Every image is a tiny constant-valued TIFF so channel identity can be read
back from the pixels.
"""
from __future__ import annotations

import os
import random
import re
import types
from pathlib import Path

import numpy as np
import pytest
import tifffile
from PIL import Image

import spacr.io as IO
from tests.test_622_mask_ingest_streaming import PATTERN


@pytest.fixture(autouse=True)
def _close_figs():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


CV_REGEX = (r"(?P<plateID>[^_]+)_(?P<wellID>[A-Z]\d+)_T(?P<timeID>\d+)"
            r"F(?P<fieldID>\d+)L(?P<laserID>\d+)A(?P<AID>\d+)"
            r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")
CQ1_REGEX = (r"W(?P<wellID>\d+)F(?P<fieldID>\d+)T(?P<timeID>\d+)"
             r"Z(?P<sliceID>\d+)C(?P<chanID>\d+)\.tif")


def _write(path, value, shape=(6, 8), dtype=np.uint16):
    """Write a constant TIFF and return its array."""
    arr = np.full(shape, value, dtype=dtype)
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    tifffile.imwrite(str(path), arr)
    return arr


def _cv(well="A01", time="0001", field="001", chan="01", ext=".tif", z="01"):
    """A CellVoyager raw file name."""
    return f"plate1_{well}_T{time}F{field}L01A01Z{z}C{chan}{ext}"


def _names(folder):
    return sorted(p.name for p in Path(folder).iterdir())


def _channel_max(stack_file):
    """Per-channel maxima of a saved ``(H, W, C)`` stack."""
    arr = np.load(stack_file)
    return arr.shape, [int(arr[..., c].max()) for c in range(arr.shape[-1])]


# ---------------------------------------------------------------------------
# _rename_and_organize_image_files
# ---------------------------------------------------------------------------

def test_rename_and_organize_writes_one_stack_per_field_and_time(
        subtests, tmp_path, monkeypatch, capsys):
    """Raw channel TIFFs become ``<plate>_<well>_<field>_<time>.npy`` stacks."""
    real_listdir, real_remove = os.listdir, os.remove

    def corrupt(src):
        (src / _cv(well="A02", chan="01")).write_text("truncated", "utf-8")

    def stale_stack(src):
        (src / "stack").mkdir()
        np.save(src / "stack" / "plate1_A01_1_1.npy", np.zeros((2, 2), np.uint8))

    def old_backup(src):
        _write(src / "orig" / _cv(chan="01"), 999)

    cases = [
        # id, {raw name: value}, setup, kwargs, doubles,
        # {stack: (shape, channel maxima)}, returned channels,
        # raws left in src, files in orig/ (None = no orig/), printed
        ("two channels, originals archived",
         {_cv(chan="01"): 3, _cv(chan="02"): 4}, None,
         {"save_original_images": True}, None,
         {"plate1_A01_1_1.npy": ((6, 8, 2), [3, 4])}, 2, [],
         [_cv(chan="01"), _cv(chan="02")], None),
        ("a field missing a channel keeps what is there",
         {_cv(chan="01"): 101, _cv(chan="02"): 102,
          _cv(well="A02", chan="01"): 201}, None, {}, None,
         {"plate1_A01_1_1.npy": ((6, 8, 2), [101, 102]),
          "plate1_A02_1_1.npy": ((6, 8, 1), [201])}, 2, [], None,
         ["is missing channel 2", "plate1_A02_1_1.tif"]),
        ("a bare img_format string is one extension",
         {_cv(chan="01"): 5, _cv(chan="02"): 6,
          _cv(well="A02", chan="01", ext=".tiff"): 7}, None,
         {"img_format": ".tif", "save_original_images": True}, None,
         {"plate1_A01_1_1.npy": ((6, 8, 2), [5, 6])}, 2,
         [_cv(well="A02", chan="01", ext=".tiff")],
         [_cv(chan="01"), _cv(chan="02")], None),
        ("timelapse keeps one stack per timepoint",
         {_cv(time="0001", chan="01"): 11, _cv(time="0001", chan="02"): 12,
          _cv(time="0002", chan="01"): 21, _cv(time="0002", chan="02"): 22},
         None, {"timelapse": True}, None,
         {"plate1_A01_1_1.npy": ((6, 8, 2), [11, 12]),
          "plate1_A01_1_2.npy": ((6, 8, 2), [21, 22])}, 2, [], None, None),
        ("one corrupt raw does not stop the others",
         {_cv(chan="01"): 11, _cv(chan="02"): 12,
          _cv(well="A02", chan="02"): 22}, corrupt, {}, None,
         {"plate1_A01_1_1.npy": ((6, 8, 2), [11, 12])}, None, None, None,
         None),
        ("an orig/ name clash leaves that raw in place",
         {_cv(chan="01"): 31, _cv(chan="02"): 32}, old_backup,
         {"save_original_images": True}, None,
         # Channel maxima left out: the stale orig/ copy (999) is read as a
         # second plane of the same field and wins the max projection -- a
         # reported source defect, not behaviour to pin.
         {"plate1_A01_1_1.npy": ((6, 8, 2), None)}, 2, [_cv(chan="01")],
         [_cv(chan="01"), _cv(chan="02")],
         ["A file with the same name already exists"]),
        ("an undeletable raw is reported",
         {_cv(chan="01"): 41, _cv(chan="02"): 42}, None, {},
         {"remove": lambda p, *a, **k: (_ for _ in ()).throw(
             OSError(13, "Permission denied")) if str(p).endswith(".tif")
          else real_remove(p, *a, **k)},
         {"plate1_A01_1_1.npy": ((6, 8, 2), [41, 42])}, 2,
         [_cv(chan="01"), _cv(chan="02")], None,
         ["could not delete original image", "Permission denied"]),
        ("a stack already on disk is not overwritten",
         {_cv(chan="01"): 3, _cv(chan="02"): 4}, stale_stack,
         {"save_original_images": True}, {"listdir": "stack-empty-once"},
         {"plate1_A01_1_1.npy": ((2, 2), [0, 0])}, 2, [],
         [_cv(chan="01"), _cv(chan="02")],
         ["A file with the same name already exists"]),
        ("no usable MIP writes nothing",
         {_cv(chan="01"): 9}, None, {},
         {"loader": lambda paths: {key: [] for key in paths}}, None, 0,
         [_cv(chan="01")], None,
         ["is missing channel 1",
          "No valid channels to merge for file plate1_A01_1_1.tif"]),
    ]
    for i, (case_id, raws, setup, kwargs, doubles, stacks, n_channels, left,
            orig, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / "plate1"
            src.mkdir(parents=True)
            for name, value in raws.items():
                _write(src / name, value)
            if setup:
                setup(src)
            with monkeypatch.context() as m:
                for key, double in (doubles or {}).items():
                    if key == "remove":
                        m.setattr(os, "remove", double)
                    elif key == "loader":
                        m.setattr(IO, "load_images_from_paths", double)
                    else:
                        seen = []
                        stack_dir = os.path.abspath(str(src / "stack"))

                        def once_empty(path, _seen=seen, _dir=stack_dir):
                            if os.path.abspath(str(path)) == _dir:
                                _seen.append(1)
                                if len(_seen) == 1:
                                    return []
                            return real_listdir(path)

                        m.setattr(os, "listdir", once_empty)
                capsys.readouterr()
                got = IO._rename_and_organize_image_files(
                    str(src), CV_REGEX, **{"batch_size": 10,
                                           "metadata_type": "custom",
                                           "img_format": [".tif"],
                                           "save_original_images": False,
                                           **kwargs})
            out = capsys.readouterr().out
            if n_channels is not None:
                assert got == n_channels
            if stacks is None:
                assert not (src / "stack").exists()
            else:
                listed = _names(src / "stack")
                if left is None:
                    assert set(stacks) <= set(listed)
                else:
                    assert listed == sorted(stacks)
                for name, expected in stacks.items():
                    shape, maxima = expected
                    arr = np.load(src / "stack" / name)
                    assert arr.shape == shape
                    if arr.ndim == 3 and maxima is not None:
                        assert _channel_max(src / "stack" / name)[1] == maxima
            if left is not None:
                assert sorted(n for n in os.listdir(src)
                              if n.endswith((".tif", ".tiff"))) == sorted(left)
            if orig is None:
                assert not (src / "orig").exists() or setup is old_backup
            else:
                assert _names(src / "orig") == sorted(orig)
            for fragment in said or []:
                assert fragment in out
            if case_id.startswith("an undeletable"):
                assert out.count("could not delete original image") == 2
            if case_id.startswith("an orig/"):
                np.testing.assert_array_equal(
                    tifffile.imread(str(src / "orig" / _cv(chan="01"))),
                    np.full((6, 8), 999, np.uint16))

    with subtests.test(msg="timelapse stacks group into one time series"):
        stacks = _names(tmp_path / "c3" / "plate1" / "stack")
        assert IO._generate_time_lists(stacks) == [
            ["plate1_A01_1_1.npy", "plate1_A01_1_2.npy"]]

    with subtests.test(msg="z-planes of different dtypes project safely"):
        src = tmp_path / "zdtype"
        for z, (dtype, value) in enumerate(((np.uint8, 200),
                                            (np.uint16, 1000)), 1):
            _write(src / f"plate_A01_T1F000Z{z}C1.tif", value, (8, 8), dtype)
        IO._rename_and_organize_image_files(str(src), PATTERN,
                                            metadata_type="custom",
                                            save_original_images=False)
        assert int(np.load(next((src / "stack").glob("*.npy"))).max()) == 1000

    with subtests.test(msg="a populated stack folder is never rebuilt"):
        src = tmp_path / "done"
        (src / "stack").mkdir(parents=True)
        arr = np.ones((4, 4, 2), np.uint16)
        np.save(str(src / "stack" / "plate1_A01_1.npy"), arr)
        (src / "plate1_A01_T0001F001L01A01Z01C01.tif").write_bytes(b"junk")
        assert IO._rename_and_organize_image_files(
            str(src),
            r"(?P<plateID>.*)_(?P<wellID>[A-Z]\d{2})_(?P<chanID>C\d{2})") == 0
        assert _names(src / "stack") == ["plate1_A01_1.npy"]
        np.testing.assert_array_equal(np.load(src / "stack" / "plate1_A01_1.npy"),
                                      arr)


# ---------------------------------------------------------------------------
# _move_to_chan_folder
# ---------------------------------------------------------------------------

def test_move_to_chan_folder_copies_into_channel_folders_and_archives(
        subtests, tmp_path, capsys):
    """Each parsable raw is copied to ``<chan>/<plate>_<well>_<field>_.tif``."""
    custom = (r"(?P<plateID>[^-]+)-(?P<wellID>[^-]+)-(?P<fieldID>[^-]+)"
              r"-(?P<chanID>[^-]+)-(?P<timeID>[^.]+)")
    cases = [
        # id, folder name, {raw: value}, pre-existing orig/{name: value},
        # regex, metadata_type, {chan folder: [copies]}, orig/ listing,
        # raws left in src, printed fragments
        ("CQ1 ids converted, plate from folder", "cq1plate",
         {"W1F001T0001Z01C1.tif": 71, "W25F002T0001Z01C2.tif": 72}, {},
         CQ1_REGEX, "cq1",
         {"1": ["cq1plate_A01_1_.tif"], "2": ["cq1plate_B01_2_.tif"]},
         ["W1F001T0001Z01C1.tif", "W25F002T0001Z01C2.tif"], [],
         ["Converted Well ID: 1 to A01", "Converted Well ID: 25 to B01"]),
        ("two slices collapse to one copy", "plate1",
         {_cv(z="01"): 55, _cv(z="02"): 56}, {}, CV_REGEX, "cellvoyager",
         {"1": ["plate1_A01_1_.tif"]}, [_cv(z="01"), _cv(z="02")], [],
         ["A file with the same name already exists"]),
        ("an orig/ clash keeps both files", "plate1",
         {_cv(well="A01"): 91, _cv(well="A02"): 92}, {_cv(well="A01"): 500},
         CV_REGEX, "cellvoyager",
         {"1": ["plate1_A01_1_.tif", "plate1_A02_1_.tif"]},
         [_cv(well="A01"), _cv(well="A02")], [_cv(well="A01")],
         ["A file with the same name already exists"]),
        ("an unparsable name is reported and archived", "plate1",
         {_cv(): 81, "garbage_image.tif": 82}, {}, CV_REGEX, "cellvoyager",
         {"1": ["plate1_A01_1_.tif"]}, [_cv(), "garbage_image.tif"], [],
         ["Could not extract information from filename garbage_image.tif"]),
    ]
    for i, (case_id, folder, raws, orig, regex, mtype, copies, archived,
            left, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / folder
            src.mkdir(parents=True)
            values = {name: _write(src / name, v) for name, v in raws.items()}
            for name, v in orig.items():
                _write(src / "orig" / name, v)
            capsys.readouterr()
            IO._move_to_chan_folder(str(src), regex, timelapse=False,
                                    metadata_type=mtype)
            out = capsys.readouterr().out
            assert sorted(p.name for p in src.iterdir()
                          if p.is_dir() and p.name != "orig") == sorted(copies)
            for chan, names in copies.items():
                assert _names(src / chan) == names
            assert _names(src / "orig") == sorted(archived)
            assert sorted(n for n in os.listdir(src) if n.endswith(".tif")) \
                == left
            for fragment in said:
                assert fragment in out
            if folder == "cq1plate":
                np.testing.assert_array_equal(
                    tifffile.imread(str(src / "1" / "cq1plate_A01_1_.tif")),
                    values["W1F001T0001Z01C1.tif"])
            for name, v in orig.items():
                assert int(tifffile.imread(str(src / "orig" / name)).max()) == v

    with subtests.test(msg="an existing stack/ is a no-op"):
        src = tmp_path / "stacked"
        (src / "stack").mkdir(parents=True)
        _write(src / _cv(), 61)
        assert IO._move_to_chan_folder(str(src), CV_REGEX) is None
        assert _names(src) == [_cv(), "stack"]

    with subtests.test(msg="text ids pass through, other files ignored"):
        src = tmp_path / "text_ids"
        src.mkdir()
        (src / "plateA-wellB-fieldC-chanD-timeE.tif").write_bytes(b"x")
        (src / "readme.md").write_text("ignore me", encoding="utf-8")
        IO._move_to_chan_folder(str(src), custom, timelapse=False,
                                metadata_type="")
        moved = _names(src / "chanD")
        assert len(moved) == 1
        assert "wellB" in moved[0] and "fieldC" in moved[0]
        assert (src / "readme.md").read_text(encoding="utf-8") == "ignore me"


# ---------------------------------------------------------------------------
# _merge_file / _merge_channels
# ---------------------------------------------------------------------------

def test_merge_file_stacks_the_channels_that_could_be_read(
        subtests, tmp_path, capsys):
    """One field across channel folders -> ``stack/<field>.npy``, readable ones only."""
    cases = [
        # id, {chan dir: value or None (file absent)}, existing stack,
        # (shape, channel maxima) or None, "Failed to read" count, said
        ("three channels", {"c0": 10, "c1": 20, "c2": 30}, None,
         ((16, 16, 3), [10, 20, 30]), 0, None),
        ("one channel missing", {"c0": 10, "c1": None}, None,
         ((16, 16, 1), [10]), 1, None),
        ("no channel readable", {"c0": None, "c1": None}, None, None, 2,
         "No valid channels to merge for file f.tif"),
        ("an existing stack is a checkpoint", {"c0": 3},
         np.full((4, 4, 1), 99, np.uint16), ((4, 4, 1), [99]), 0, None),
    ]
    for i, (case_id, chans, existing, expected, failed, said) in \
            enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"c{i}"
            dirs = []
            for name, value in chans.items():
                (root / name).mkdir(parents=True)
                if value is not None:
                    _write(root / name / "f.tif", value, (16, 16))
                dirs.append(str(root / name))
            stack = root / "stack"
            if existing is not None:
                stack.mkdir()
                np.save(stack / "f.npy", existing)
            capsys.readouterr()
            assert IO._merge_file(dirs, str(stack), "f.tif") is None
            out = capsys.readouterr().out
            assert out.count("Warning: Failed to read image") == failed
            if expected is None:
                assert os.listdir(stack) == []
            else:
                assert _channel_max(stack / "f.npy") == expected
            if said:
                assert said in out


def _chan_dirs(root, n_chan, names=("f0", "f1")):
    """``root/<1..n>/<field>.tif`` with pixel value ``100 * chan + field``."""
    for c in range(1, n_chan + 1):
        for i, name in enumerate(names):
            _write(root / str(c) / f"{name}.tif", 100 * c + i, (6, 7))
    return root


def test_merge_channels_builds_stack_from_numbered_channel_folders(
        subtests, tmp_path, monkeypatch, capsys):
    """Returns the channel-folder count; stacks follow sorted folder order."""
    import spacr.plot as SP

    def add_subdir(src):
        (src / "1" / "not_an_image").mkdir()

    def add_stack(src):
        (src / "stack").mkdir()
        np.save(src / "stack" / "already_here.npy", np.zeros((2, 2, 1)))

    cases = [
        # id, n channels, setup, plot, returned, stack listing,
        # channel maxima of f0.npy, plotted
        ("three channels plotted", 3, None, True, 3, ["f0.npy", "f1.npy"],
         [100, 200, 300], True),
        ("non-file entries are skipped", 2, add_subdir, False, 2,
         ["f0.npy", "f1.npy"], [100, 200], False),
        ("a populated stack/ is left alone", 2, add_stack, False, 2,
         ["already_here.npy"], None, False),
    ]
    for i, (case_id, n_chan, setup, plot, returned, listing, maxima,
            plotted) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = _chan_dirs(tmp_path / f"c{i}", n_chan)
            if setup:
                setup(src)
            calls = []
            with monkeypatch.context() as m:
                m.setattr(SP, "plot_arrays", lambda p, *a, **k: calls.append(p))
                assert IO._merge_channels(str(src), plot=plot) == returned
            assert _names(src / "stack") == listing
            if maxima:
                assert _channel_max(src / "stack" / "f0.npy") == \
                    ((6, 7, n_chan), maxima)
            assert calls == ([str(src / "stack")] if plotted else [])

    empties = [
        ("loose image and a non-channel folder",
         ["img_A01_f1.tif", "notes/"], "stack/ will be built"),
        ("only a non-channel folder", ["notachannel/"],
         "No single-channel folders"),
    ]
    for case_id, entries, said in empties:
        with subtests.test(msg=case_id):
            src = tmp_path / case_id.replace(" ", "_")
            src.mkdir()
            for entry in entries:
                if entry.endswith("/"):
                    (src / entry[:-1]).mkdir()
                else:
                    (src / entry).write_bytes(b"")
            capsys.readouterr()
            assert IO._merge_channels(str(src), plot=False) == 0
            assert said in capsys.readouterr().out
            assert _names(src) == sorted(e.rstrip("/") for e in entries)


# ---------------------------------------------------------------------------
# _concatenate_channel
# ---------------------------------------------------------------------------

def _npz(path):
    with np.load(path) as z:
        return z["data"], [str(f) for f in z["filenames"]]


def test_concatenate_channel_batches_fields_into_channel_stack(
        subtests, tmp_path, monkeypatch, capsys):
    """Stacks become ``channel_stack/*.npz`` of the kept channels, padded."""
    def stacks(spec):
        def build(src):
            for name, (shape, fill) in spec.items():
                np.save(src / name, np.full(shape, fill, np.uint16))
        return build

    def reverse(seq):
        seq.sort(key=os.path.basename, reverse=True)

    six = {f"plate1_A01_{i}.npy": ((4, 5, 3), 10 + i) for i in range(6)}
    cases = [
        # id, files, kwargs, shuffle double, {npz: (shape, names, first px)}
        ("batches of two", {f"plate1_A01_{i}.npy": ((3, 3, 2), i)
                            for i in range(5)},
         {"channels": [0, 1], "batch_size": 2}, None,
         {"stack_0.npz": ((2, 3, 3, 2), None, None),
          "stack_1.npz": ((2, 3, 3, 2), None, None),
          "stack_2.npz": ((1, 3, 3, 2), None, None)}),
        ("mixed shapes are padded", {"small.npy": ((10, 12, 3), 4),
                                     "tall.npy": ((14, 8, 3), 9)},
         {"channels": [0, 2], "batch_size": 100}, None,
         {"stack_0.npz": ((2, 14, 12, 2), None, None)}),
        ("randomize shuffles, rows stay paired", six,
         {"channels": [0, 1, None], "batch_size": 100, "randomize": True},
         reverse,
         {"stack_0.npz": ((6, 4, 5, 2), sorted(six, reverse=True),
                          [15, 14, 13, 12, 11, 10])}),
        ("timelapse writes one npz per field",
         {"plate1_A01_1_2.npy": ((8, 9, 3), 20),
          "plate1_A01_1_1.npy": ((8, 9, 3), 10)},
         {"channels": [0, 1], "batch_size": 10, "timelapse": True}, None,
         {"plate1_A01_1.npz": ((2, 8, 9, 2),
                               ["plate1_A01_1_1.npy", "plate1_A01_1_2.npy"],
                               [10, 20])}),
        ("a single timelapse frame still returns the folder",
         {"plate1_A01_1_1.npy": ((6, 6, 2), 3)},
         {"channels": [0], "batch_size": 5, "timelapse": True}, None, None),
    ]
    for i, (case_id, files, kwargs, shuffle, expected) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / "stack"
            src.mkdir(parents=True)
            stacks(files)(src)
            (src / "ignored.txt").write_text("not an npy", encoding="utf-8")
            with monkeypatch.context() as m:
                if shuffle:
                    m.setattr(random, "shuffle", shuffle)
                out = IO._concatenate_channel(
                    str(src), **{"randomize": False, "timelapse": False,
                                 **kwargs})
            assert out == str(tmp_path / f"c{i}" / "channel_stack")
            assert os.path.isdir(out)
            if expected is None:
                continue
            assert _names(out) == sorted(expected)
            all_names = []
            for npz, (shape, names, first) in expected.items():
                data, got_names = _npz(Path(out) / npz)
                all_names += got_names
                assert data.shape == shape
                if names is not None:
                    assert got_names == names
                    assert [int(d[0, 0, 0]) for d in data] == first
            assert sorted(all_names) == sorted(files)
            if case_id == "mixed shapes are padded":
                data, names = _npz(Path(out) / "stack_0.npz")
                small, tall = data[names.index("small.npy")], \
                    data[names.index("tall.npy")]
                assert np.all(small[:10, :12] == 4) and np.all(small[10:] == 0)
                assert np.all(tall[:, :8] == 9) and np.all(tall[:, 8:] == 0)

    with subtests.test(msg="bad timelapse metadata is reported, not raised"):
        bad = tmp_path / "bad" / "stack"
        bad.mkdir(parents=True)
        for t in (1, 2):
            np.save(bad / f"plate1_A01_f1_{t}.npy", np.zeros((4, 4), np.float32))
        capsys.readouterr()
        IO._concatenate_channel(str(bad), channels=[0, 1], randomize=False,
                                timelapse=True, batch_size=2)
        out = capsys.readouterr().out
        assert "make sure filenames metadata is structured" in out
        assert "axis" in out.lower()
        assert list((tmp_path / "bad" / "channel_stack").glob("*.npz")) == []


# ---------------------------------------------------------------------------
# _create_movies_from_npy_per_channel
# ---------------------------------------------------------------------------

def test_create_movies_makes_one_movie_per_field_and_channel(
        subtests, tmp_path, monkeypatch, rng):
    """Every ``plate_well_field_time.npy`` group gets one movie per channel."""
    import spacr.timelapse as TL

    def fields(n_fields, n_times, channels, start=1):
        def build(src):
            for f in range(1, n_fields + 1):
                for t in range(start, start + n_times):
                    np.save(src / f"plate1_A01_f{f}_{t}.npy",
                            np.full((8, 8, channels), t, np.float32))
        return build

    def extras(src):
        np.save(src / "not_a_field_stem.npy", np.zeros((8, 8, 2), np.float32))
        (src / "plate1_A01_f3_1.txt").write_text("not an array", "utf-8")

    four = ["plate1_A01_f1_channel_0.mp4", "plate1_A01_f1_channel_1.mp4",
            "plate1_A01_f2_channel_0.mp4", "plate1_A01_f2_channel_1.mp4"]
    cases = [
        # id, builders, movies, frames per movie, frame time stamps
        ("two fields, two channels, foreign files ignored",
         [fields(2, 3, 2), extras], four, 3, [1, 2, 3]),
        ("time from zero", [fields(2, 2, 2, start=0)], four, 2, [0, 1]),
        ("nothing matches", [extras], [], None, None),
    ]
    for i, (case_id, builders, movies, n_frames, times) in enumerate(cases):
        with subtests.test(msg=case_id):
            src = tmp_path / f"c{i}" / "stack"
            src.mkdir(parents=True)
            for build in builders:
                build(src)
            calls = []
            with monkeypatch.context() as m:
                m.setattr(TL, "_npz_to_movie",
                          lambda arrays, names, path, fps: calls.append(
                              (os.path.basename(path), arrays, list(names), fps)))
                IO._create_movies_from_npy_per_channel(str(src), fps=7)
            assert sorted(c[0] for c in calls) == movies
            for _name, arrays, names, fps in calls:
                assert (len(arrays), fps) == (n_frames, 7)
                assert [int(re.search(r"_(\d+)\.npy$", n).group(1))
                        for n in names] == times
                assert all(a.dtype == np.uint8 and a.shape == (8, 8, 1)
                           for a in arrays)
            assert (tmp_path / f"c{i}" / "movies").is_dir()

    with subtests.test(msg="frames are rescaled on global percentiles"):
        src = tmp_path / "pct" / "stack"
        src.mkdir(parents=True)
        for t in range(2):
            arr = np.zeros((4, 4, 1), np.uint16)
            arr[..., 0] = np.linspace(0, 1000 * (t + 1), 16).reshape(4, 4)
            np.save(src / f"p_B02_g3_{t}.npy", arr)
        got = {}
        with monkeypatch.context() as m:
            m.setattr(TL, "_npz_to_movie",
                      lambda arrays, names, path, fps: got.update(
                          arrays=arrays, names=list(names), path=path))
            IO._create_movies_from_npy_per_channel(str(src), fps=7)
        assert got["path"].endswith("p_B02_g3_channel_0.mp4")
        assert got["names"] == ["p_B02_g3_0.npy", "p_B02_g3_1.npy"]
        assert [int(a.max()) < 255 for a in got["arrays"]] == [True, False]

    with subtests.test(msg="real writer: one mp4 per field and channel"):
        src = tmp_path / "real" / "stack"
        src.mkdir(parents=True)
        for f in ("f1", "f2"):
            for t in range(3):
                np.save(src / f"plate1_A01_{f}_{t}.npy",
                        rng.integers(0, 500, (16, 16, 2)).astype(np.uint16))
        np.save(src / "junk.npy", np.zeros((16, 16, 2), np.uint16))
        IO._create_movies_from_npy_per_channel(str(src), fps=2)
        assert _names(tmp_path / "real" / "movies") == four


# ---------------------------------------------------------------------------
# delete_empty_subdirectories
# ---------------------------------------------------------------------------

def test_delete_empty_subdirectories_prunes_bottom_up_and_tolerates_errors(
        subtests, tmp_path, monkeypatch):
    """Empty chains vanish, anything holding a file stays, rmdir errors pass."""
    real_rmdir = os.rmdir
    cases = [
        # id, dirs to make, files to make, locked dir, surviving top-level
        ("nested empties go, files stay",
         ["a/b/c", "keep/inner", "empty1", "empty2/nested_empty"],
         ["keep/inner/x.txt"], None, ["keep"]),
        ("a failing rmdir is skipped", ["locked", "gone"], [], "locked",
         ["locked"]),
    ]
    for i, (case_id, dirs, files, locked, kept) in enumerate(cases):
        with subtests.test(msg=case_id):
            root = tmp_path / f"c{i}"
            for d in dirs:
                (root / d).mkdir(parents=True)
            for f in files:
                (root / f).write_text("x", encoding="utf-8")
            with monkeypatch.context() as m:
                if locked:
                    def flaky(path, *a, **k):
                        if os.path.basename(str(path)) == locked:
                            raise PermissionError(13, "Permission denied")
                        return real_rmdir(path, *a, **k)
                    m.setattr(IO.os, "rmdir", flaky)
                IO.delete_empty_subdirectories(str(root))
            assert root.is_dir()
            assert _names(root) == kept
            for f in files:
                assert (root / f).exists()


# ---------------------------------------------------------------------------
# process_non_tif_non_2D_images
# ---------------------------------------------------------------------------

def _ramp(shape, dtype=np.uint16, mod=500):
    n = int(np.prod(shape))
    return ((np.arange(n) % mod) + 1).reshape(shape).astype(dtype)


def test_process_non_tif_non_2d_images_splits_every_plane_to_a_tiff(
        subtests, tmp_path, monkeypatch, capsys):
    """Multi-axis images become ``<stem>_C<c>[_Z<z>[_T<t>]].tif`` planes."""
    four = _ramp((8, 9, 2, 3))
    five = _ramp((6, 7, 2, 2, 2), np.uint8, 200)
    three = np.zeros((8, 6, 3), np.uint16)
    for c in range(3):
        three[..., c] = 100 * (c + 1)
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 10, 20, 30
    gray = (np.arange(64).reshape(8, 8) % 255).astype(np.uint8)
    czi = _ramp((6, 7, 3))
    nd2 = _ramp((5, 6, 2), np.uint8, 200)

    def tif(name, arr):
        return lambda d: tifffile.imwrite(str(d / name), arr)

    def png(name, arr, mode):
        return lambda d: Image.fromarray(arr, mode=mode).save(str(d / name))

    def raw(name, payload):
        return lambda d: (d / name).write_bytes(payload)

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def asarray(self):
            return czi

    class _ND2:
        def __init__(self, path):
            pass

        def __enter__(self):
            return nd2

        def __exit__(self, *exc):
            return False

    cases = [
        # id, builders, doubles, {written tif: (dtype, array)} or names only,
        # printed fragments
        ("4-D TIFF: channel x Z", [tif("vol.tif", four)], {},
         {f"vol_C{c}_Z{z}.tif": four[..., c - 1, z - 1]
          for c in (1, 2) for z in (1, 2, 3)}, []),
        ("5-D TIFF: channel x Z x T", [tif("movie.tif", five)], {},
         {f"movie_C{c}_Z{z}_T{t}.tif": five[..., c - 1, z - 1, t - 1]
          for c in (1, 2) for z in (1, 2) for t in (1, 2)}, []),
        ("3-D TIFF: one per channel, bit depth kept",
         [tif("plate1_A01.tif", three)], {},
         {f"plate1_A01_C{c}.tif": three[..., c - 1] for c in (1, 2, 3)}, []),
        ("RGB PNG", [png("img.png", rgb, "RGB")], {},
         {f"img_C{c}.tif": rgb[..., c - 1] for c in (1, 2, 3)}, []),
        ("grayscale PNG keeps 8 bits", [png("g.png", gray, "L")], {},
         {"g.tif": gray}, []),
        ("CZI through czifile", [raw("scan.czi", b"stub")],
         {"czifile": types.SimpleNamespace(CziFile=lambda p: _Ctx())},
         {f"scan_C{c}.tif": czi[..., c - 1] for c in (1, 2, 3)}, []),
        ("ND2 through ND2Reader", [raw("movie.nd2", b"stub")],
         {"ND2Reader": _ND2},
         {f"movie_C{c}.tif": nd2[..., c - 1] for c in (1, 2)}, []),
        ("2-D TIFF is left untouched", [tif("plain.tif", _ramp((12, 13)))], {},
         {}, ["already grayscale and in TIFF format"]),
        ("6-axis TIFF and foreign files are left alone",
         [tif("six.tif", np.arange(1440, dtype=np.uint16).reshape(
             2, 3, 4, 5, 6, 2)), tif("three.tif", _ramp((8, 8, 2))),
          lambda d: (d / "notes.txt").write_text("not an image", "utf-8")], {},
         {"three_C1.tif": _ramp((8, 8, 2))[..., 0],
          "three_C2.tif": _ramp((8, 8, 2))[..., 1]}, []),
        ("an unreadable file is reported, the next one split",
         [raw("bad.tif", b"definitely not a tiff"),
          tif("good.tif", _ramp((4, 5, 2)))], {},
         {"good_C1.tif": _ramp((4, 5, 2))[..., 0],
          "good_C2.tif": _ramp((4, 5, 2))[..., 1]},
         ["Error processing bad.tif"]),
    ]
    for i, (case_id, builders, doubles, written, said) in enumerate(cases):
        with subtests.test(msg=case_id):
            d = tmp_path / f"c{i}"
            d.mkdir()
            for build in builders:
                build(d)
            before = set(os.listdir(d))
            with monkeypatch.context() as m:
                for attr, double in doubles.items():
                    m.setattr(IO, attr, double)
                capsys.readouterr()
                ledger = IO.process_non_tif_non_2D_images(str(d))
            out = capsys.readouterr().out
            assert sorted(set(os.listdir(d)) - before) == sorted(written)
            for name, arr in written.items():
                back = tifffile.imread(str(d / name))
                assert back.dtype == arr.dtype, name
                np.testing.assert_array_equal(back, arr)
            for fragment in said:
                assert fragment in out
            assert ledger.is_complete is (not said or "already" in said[0])

    with subtests.test(msg="an extension load_image does not know"):
        d = tmp_path / "weird"
        d.mkdir()
        target = str(d / "weird.tif")
        tifffile.imwrite(target, _ramp((4, 4)))
        real_splitext = os.path.splitext
        seen = []

        def fake_splitext(p):
            if str(p) == target:
                seen.append(1)
                if len(seen) >= 2:
                    return (str(p)[:-4], ".xyz")
            return real_splitext(p)

        with monkeypatch.context() as m:
            m.setattr(os.path, "splitext", fake_splitext)
            capsys.readouterr()
            IO.process_non_tif_non_2D_images(str(d))
        out = capsys.readouterr().out
        assert "Unsupported file extension: .xyz" in out
        assert "Error processing weird.tif" in out
