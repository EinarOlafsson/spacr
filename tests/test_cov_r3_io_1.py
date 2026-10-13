"""Narrow coverage for failure-side branches of :mod:`spacr.io`.

Every test drives a path a user only meets when something has already gone
wrong: a notebook import that is mid-initialisation, a plate folder holding no
images, a preloader thread that will not die, a merge failure that is not a
duplicate key, an unreadable resume sidecar. Those paths rot silently, because
a green pipeline never walks them. All CPU-only and offline.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import queue
import sqlite3
import sys
import threading
import types

import numpy as np
import pandas as pd
import pytest

import spacr.io as IO


# module import: the IPython.display fallback

def test_importing_io_survives_a_half_initialised_ipython(monkeypatch):
    """spacr.io must import even when ``IPython.display`` cannot be imported.

    ``import IPython.display`` can fail while IPython is being initialised by
    another thread; the GUI imports spacr.io from a worker thread, so failing
    there takes the application down before any window appears. The module
    answers with a no-op ``display``. This loads the same file again with the
    import poisoned and checks the fallback is installed and callable.
    """
    monkeypatch.setitem(sys.modules, "IPython.display", None)

    name = "spacr._io_ipython_fallback_probe"
    spec = importlib.util.spec_from_file_location(name, IO.__file__)
    probe = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, probe)
    spec.loader.exec_module(probe)

    # The real IPython.display.display is not what we got.
    assert probe.display is not IO.display
    # ... and the stand-in swallows any call shape without raising.
    assert probe.display("anything", extra=1) is None
    # The rest of the module still loaded.
    assert callable(probe.preprocess_img_data)


# migrate_unescaped_plate_names

def test_migration_skips_a_dotted_name_that_is_not_a_field_stem(tmp_path):
    """A file whose name is not a parseable field stem is left alone.

    A plate folder collects hand-dropped arrays and sidecars beside the real
    field stacks. Renaming one on a guess loses it, so a name the key grammar
    rejects is skipped while the genuine underscored plate next to it is still
    planned — both live in the same folder so the skip cannot be confused with
    the migration simply finding nothing.
    """
    from spacr.io import migrate_unescaped_plate_names

    stack = tmp_path / "stack"
    stack.mkdir()
    # Five underscore-separated components, so the length guard lets it
    # through, but the dot makes the stem unparseable as plate_well_field_time.
    (stack / "a_b.c_d_e_f.npy").write_bytes(b"")
    # A genuine underscored plate name: five components, no dot.
    (stack / "exp_1_A01_1_1.npy").write_bytes(b"")

    planned = migrate_unescaped_plate_names(str(tmp_path), dry_run=True)

    moved = {os.path.basename(old): os.path.basename(new) for old, new in planned}
    assert moved == {"exp_1_A01_1_1.npy": "exp%5F1_A01_1_1.npy"}
    # Nothing was actually renamed in dry-run mode, and the odd file survives.
    assert (stack / "a_b.c_d_e_f.npy").exists()
    assert (stack / "exp_1_A01_1_1.npy").exists()


# spacrDataLoader.cleanup

def test_cleanup_reports_a_preloader_that_refuses_to_stop(monkeypatch, caplog):
    """A preloader thread still alive after the grace period is reported.

    ``cleanup`` abandons the daemon thread rather than blocking teardown for
    ever, which is right — but doing it in silence leaves a user whose run
    ends with a wedged decoder no reason for the leaked worker. The log line
    is the only evidence, so it has to be emitted.
    """
    from spacr.io import spacrDataLoader

    release = threading.Event()
    worker = threading.Thread(target=release.wait, daemon=True)
    worker.start()

    ticks = iter([0.0, 1.0, 1000.0, 1000.0, 1000.0])
    monkeypatch.setattr(IO.time, "monotonic", lambda: next(ticks, 1000.0))

    q = queue.Queue()
    q.put(object())
    fake = types.SimpleNamespace(
        _iteration_active=True, _stop_event=False,
        _stop_signal=threading.Event(), thread=worker, batch_queue=q)

    with caplog.at_level(logging.ERROR, logger="spacr.io"):
        spacrDataLoader.cleanup(fake)

    release.set()
    worker.join(timeout=5)

    assert "did not stop within five seconds" in caplog.text
    # The stop signal was raised and the queue drained before giving up.
    assert fake._stop_signal.is_set()
    assert fake._iteration_active is False
    assert q.empty()


# _merge_channels: the modern layout has no per-channel folders

def test_merge_channels_returns_zero_when_there_are_no_channel_folders(
        tmp_path, capsys):
    """A plate folder without numbered channel sub-folders is not an error.

    The modern ingest builds ``stack/`` straight from an in-memory channel
    dict, so this older two-step helper legitimately finds nothing to do.
    Indexing the empty list instead raised IndexError from a stage whose
    message named the plate, making an ordinary folder look corrupt.
    """
    from spacr.io import _merge_channels

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "img_A01_f1.tif").write_bytes(b"")
    (src / "notes").mkdir()          # a folder, but not a channel folder

    result = _merge_channels(str(src), plot=False)

    assert result == 0
    # The folder is left exactly as it was, so preprocess_img_data can fall
    # through to the branch that builds stack/ itself.
    assert sorted(os.listdir(str(src))) == ["img_A01_f1.tif", "notes"]
    assert "stack/ will be built" in capsys.readouterr().out


# _concatenate_channel: the timelapse branch reports its own failures

def test_timelapse_concatenation_reports_bad_metadata_instead_of_crashing(
        tmp_path, capsys):
    """A malformed timelapse array is reported, not raised, mid-batch.

    ``_concatenate_channel`` runs inside preprocessing; one unusable field
    must not abort the plate with a bare traceback. The message names the
    filename grammar the caller has to fix, which is the only clue a user
    gets, so it has to be printed.
    """
    from spacr.io import _concatenate_channel

    bad = tmp_path / "bad" / "stack"
    bad.mkdir(parents=True)
    # 2-D arrays: there is no axis 2 to take channels from.
    for t in (1, 2):
        np.save(bad / f"plate1_A01_f1_{t}.npy", np.zeros((4, 4), np.float32))

    _concatenate_channel(str(bad), channels=[0, 1], randomize=False,
                         timelapse=True, batch_size=2)

    captured = capsys.readouterr().out
    assert "make sure filenames metadata is structured" in captured
    assert "axis" in captured.lower()
    assert not list((tmp_path / "bad" / "channel_stack").glob("*.npz"))

    # The very same call over well-formed arrays does write its group, so the
    # emptiness above is the reported failure and not an inert code path.
    good = tmp_path / "good" / "stack"
    good.mkdir(parents=True)
    for t in (1, 2):
        np.save(good / f"plate1_A01_f1_{t}.npy",
                np.zeros((4, 4, 2), np.float32))
    _concatenate_channel(str(good), channels=[0, 1], randomize=False,
                         timelapse=True, batch_size=2)
    written = sorted((tmp_path / "good" / "channel_stack").glob("*.npz"))
    assert [f.name for f in written] == ["plate1_A01_f1.npz"]


# preprocess_img_data: the "nothing was produced" diagnosis

def _settings(src, **over):
    s = {"src": str(src), "metadata_type": "cellvoyager", "custom_regex": None,
         "channels": [0, 1], "nucleus_channel": 0, "cell_channel": 1,
         "pathogen_channel": None, "organelle_channel": None, "plot": False,
         "batch_size": 1, "test_mode": False, "timelapse": False,
         "normalize": True}
    s.update(over)
    return s


def test_a_folder_of_plates_is_named_as_the_likely_mistake(tmp_path, capsys,
                                                           monkeypatch):
    """Pointing src at a folder of plates must say so, not fail downstream.

    With no images directly in src nothing is organised and ``stack/`` stays
    empty; letting the run continue wrote an empty measurement set that looked
    like a result. The error names the sub-folders it found so the user can
    see they pointed one level too high.
    """
    from spacr.io import preprocess_img_data

    src = tmp_path / "experiment"
    src.mkdir()
    for plate in ("plate1", "plate2"):
        d = src / plate
        d.mkdir()
        (d / "keep.txt").write_text("not an image")   # keep it non-empty

    def _boom(*_a, **_k):
        raise RuntimeError("organiser found nothing to do")

    monkeypatch.setattr(IO, "_rename_and_organize_image_files", _boom)

    with pytest.raises(FileNotFoundError) as excinfo:
        preprocess_img_data(_settings(src))

    message = str(excinfo.value)
    assert "No image stacks were produced" in message
    assert "0 image file(s)" in message
    assert "plate1" in message and "plate2" in message
    assert "point src at one of them" in message
    # The organiser's own failure was reported rather than swallowed silently.
    assert "organiser found nothing to do" in capsys.readouterr().out


def test_a_source_folder_that_vanished_still_gets_a_named_error(tmp_path,
                                                               monkeypatch):
    """If src disappears mid-run the diagnosis still names src, not os.listdir.

    Scratch plates on shared storage do get swept mid-run. Re-listing src to
    build the hint must not replace the useful "nothing was produced from
    <src>" error with a raw OSError naming a path the user never typed.
    """
    from spacr.io import preprocess_img_data

    src = tmp_path / "plate1"
    src.mkdir()
    (src / "img_A01_f1.tif").write_bytes(b"")

    def _sweep(*_a, **_k):
        import shutil
        shutil.rmtree(str(src))
        raise RuntimeError("source folder was swept")

    monkeypatch.setattr(IO, "_rename_and_organize_image_files", _sweep)

    with pytest.raises(FileNotFoundError) as excinfo:
        preprocess_img_data(_settings(src))

    message = str(excinfo.value)
    assert "No image stacks were produced" in message
    assert str(src) in message
    assert "0 image file(s)" in message
    # The sweep really did happen inside the call.
    assert not src.exists()


# _merge_with_cardinality

def test_a_non_cardinality_merge_error_is_not_relabelled():
    """Merge failures that are not cardinality violations keep pandas' message.

    The wrapper turns "MergeError" into a sentence naming the two tables and
    the duplicated keys. Dressing up every MergeError that way would
    misdiagnose a mis-specified join as duplicated data and send the user off
    de-duplicating a table that is perfectly fine.
    """
    from spacr.io import _merge_with_cardinality, MergeCardinalityError

    left = pd.DataFrame({"k": [1, 2, 3], "v": [10.0, 20.0, 30.0]})
    right = pd.DataFrame({"j": [1, 2, 3], "w": [1.0, 2.0, 3.0]})

    # Nothing to join on at all: a MergeError with no duplicated key behind it.
    with pytest.raises(pd.errors.MergeError) as excinfo:
        _merge_with_cardinality(left, right, validate="one_to_one",
                                left_name="cell table",
                                right_name="nucleus table")

    assert not isinstance(excinfo.value, MergeCardinalityError)
    assert "No common columns" in str(excinfo.value)

    # The same wrapper WITH a real duplicate is relabelled, so the branch
    # above is a genuine discrimination and not simply a dead wrapper.
    dup = pd.DataFrame({"k": [1, 1], "w": [1.0, 2.0]})
    other = pd.DataFrame({"k": [1, 2], "z": [1.0, 2.0]})
    with pytest.raises(MergeCardinalityError) as dup_info:
        _merge_with_cardinality(dup, other, on="k", validate="one_to_one",
                                left_name="cell table",
                                right_name="nucleus table")
    assert "cell table has duplicated" in str(dup_info.value)


# _load_and_concatenate_arrays

def _merge_src(root, n_channels=2, name="fov.npy"):
    stack_dir = os.path.join(root, "stack")
    os.makedirs(stack_dir, exist_ok=True)
    img = np.stack([np.full((6, 6), float(c + 1), np.float32)
                    for c in range(n_channels)], axis=-1)
    np.save(os.path.join(stack_dir, name), img)
    return img


def test_resume_refuses_an_unreadable_plane_layout_sidecar(tmp_path):
    """A corrupt merged-layout sidecar stops the resume instead of guessing.

    The sidecar records which plane holds which biological role. Resuming
    without reading it would reuse merged arrays under an assumed layout:
    every plane still exists, so measurement succeeds and returns plausible
    numbers for the wrong objects.
    """
    from spacr.io import _load_and_concatenate_arrays
    from spacr.crops import MERGED_LAYOUT_SIDECAR

    root = str(tmp_path)
    _merge_src(root)
    merged = tmp_path / "merged"
    merged.mkdir()
    (merged / MERGED_LAYOUT_SIDECAR).write_text("{not json at all")

    with pytest.raises(ValueError) as excinfo:
        _load_and_concatenate_arrays(root, channels=[0, 1],
                                     cell_chann_dim=None,
                                     nucleus_chann_dim=None,
                                     pathogen_chann_dim=None,
                                     organelle_chann_dim=None,
                                     resume=True)

    message = str(excinfo.value)
    assert "Cannot resume" in message
    assert MERGED_LAYOUT_SIDECAR in message
    # The refusal came before any work: merged/ still holds only the sidecar,
    # still exactly as written.
    assert os.listdir(str(merged)) == [MERGED_LAYOUT_SIDECAR]
    assert (merged / MERGED_LAYOUT_SIDECAR).read_text() == "{not json at all"


def test_a_failed_sidecar_write_leaves_no_half_written_temporary(tmp_path,
                                                                monkeypatch):
    """A crash while writing the layout sidecar must not leave a temp file.

    The sidecar is written to a hidden temporary and renamed into place. If
    the write fails the temporary has to go, or the merged folder accumulates
    ``.spacr_plane_layout_*`` droppings the next resume scan steps over — and
    even when that cleanup itself fails the original error must still surface.
    """
    from spacr.io import _load_and_concatenate_arrays

    root = str(tmp_path)
    _merge_src(root)

    removed = []

    def _bad_dump(*_a, **_k):
        raise RuntimeError("disk full while writing layout")

    def _bad_remove(path, *a, **k):
        removed.append(path)
        raise OSError("cannot unlink the temporary either")

    monkeypatch.setattr(IO.json, "dump", _bad_dump)
    monkeypatch.setattr(IO.os, "remove", _bad_remove)

    with pytest.raises(RuntimeError, match="disk full while writing layout"):
        _load_and_concatenate_arrays(root, channels=[0, 1],
                                     cell_chann_dim=None,
                                     nucleus_chann_dim=None,
                                     pathogen_chann_dim=None,
                                     organelle_chann_dim=None)

    # The cleanup was attempted on the temporary this call created.
    assert len(removed) == 1
    assert os.path.basename(removed[0]).startswith(".spacr_plane_layout_")
    assert os.path.dirname(removed[0]) == os.path.join(root, "merged")


# _read_and_merge_data

def _meta(obj_key):
    r, c = f"r{(obj_key % 2) + 1}", f"c{(obj_key % 2) + 1}"
    return {"plateID": "plate1", "rowID": r, "columnID": c, "fieldID": "f1",
            "prcf": f"plate1_{r}_{c}_f1"}


def _cell_frame(n=6):
    return pd.DataFrame([
        dict(_meta(obj), object_label=obj, cell_area=100.0 + obj,
             cell_channel_0_mean_intensity=500.0 + obj)
        for obj in range(1, n + 1)])


def _write_db(path, tables):
    with sqlite3.connect(path) as con:
        for name, df in tables.items():
            df.to_sql(name, con, index=False, if_exists="replace")
    return str(path)


def test_png_crops_from_another_field_are_reported_not_merged(tmp_path, capsys):
    """Crops that key onto no measured object must say how many objects went.

    ``png_list`` joins inner: crops whose object ids match nothing delete
    every row of the merged table — a 100-cell plate becoming an empty one
    with no explanation. The shortfall must be reported by table, and the
    second (non-numeric) png block must still merge onto the now-empty frame.
    """
    png_rows = []
    for obj in range(1, 7):
        row = dict(_meta(obj))
        # A different field, so no prcfo can ever match a cell.
        row["fieldID"] = "f9"
        row["prcf"] = row["prcf"].replace("_f1", "_f9")
        row.update({"cell_id": f"o{obj}",
                    "png_path": f"/x/cell_png/o{obj}.png",
                    "file_name": f"o{obj}.png",
                    "test": obj % 2})
        png_rows.append(row)

    db = _write_db(tmp_path / "m.db",
                   {"cell": _cell_frame(), "png_list": pd.DataFrame(png_rows)})

    merged, obj_dfs = IO._read_and_merge_data([db], ["cell", "png_list"])

    out = capsys.readouterr().out
    assert "6 of 6 objects have no row in png_list" in out
    assert "gap in the database rather than a filter" in out
    assert len(merged) == 0
    # The raw cell table is still handed back untouched.
    assert len(obj_dfs) == 1
    assert len(obj_dfs[0]) == 6

    # Same tables, crops keyed onto the field they were cropped from: now the
    # crop columns really do arrive, so the emptiness above is the join
    # failing and not this test simply never producing them.
    matched = pd.DataFrame(png_rows).assign(
        fieldID="f1", prcf=lambda x: x["prcf"].str.replace("_f9", "_f1"))
    db2 = _write_db(tmp_path / "matched.db",
                    {"cell": _cell_frame(), "png_list": matched})
    merged2, _ = IO._read_and_merge_data([db2], ["cell", "png_list"])
    assert len(merged2) == 6
    assert "png_path" in merged2.columns


def test_an_organelle_only_database_anchors_the_merge_on_its_parent_cell(
        tmp_path, capsys):
    """An organelle table with no cell table still groups on the parent cell.

    Organelle slots were added after cell/nucleus/pathogen, so a database
    measured with only an organelle channel has no earlier role to merge onto.
    That table has to become the anchor, keyed on ``cell_id``, or the merge
    reads an unbound frame and the measurement set is lost.
    """
    rows = []
    for obj, cid in zip([1, 2, 3, 4], [1, 1, 2, 3]):
        row = dict(_meta(cid))
        row.update({"object_label": obj, "cell_id": cid,
                    "organelle_area": 10.0 + obj,
                    "organelle_channel_0_mean_intensity": 50.0 + obj})
        rows.append(row)
    db = _write_db(tmp_path / "m.db", {"organelle": pd.DataFrame(rows)})

    merged, _ = IO._read_and_merge_data([db], ["organelle"], verbose=True)

    # Three parent cells, not four organelles: the grouping is per cell.
    assert len(merged) == 3
    assert "organelle_area" in merged.columns
    counts = merged["organelle_prcfo_count"].to_dict()
    assert {k.rsplit("_", 1)[-1]: int(v) for k, v in counts.items()} == {
        "o1": 2, "o2": 1, "o3": 1}
    assert all(idx.endswith(("_o1", "_o2", "_o3")) for idx in merged.index)
    assert "organelle grouped" in capsys.readouterr().out
