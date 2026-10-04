"""The RAM guard's process list, its fallbacks and the input it samples."""
from __future__ import annotations

import types

import numpy as np
import pytest

from spacr import resource_log as rl


class _NoSuchProcess(Exception):
    pass


def _psutil(processes=(), me=None, username="me", memory=(8 << 30, 16 << 30),
            terminate=None):
    """A stand-in for psutil with just what the guard reads."""
    class _Proc:
        def __init__(self, pid=None):
            self.pid = pid

        def parents(self):
            if me == "broken":
                raise RuntimeError("no parents")
            return []

        def children(self, recursive=True):
            return []

        def username(self):
            if username is None:
                raise RuntimeError("no user")
            return username

        def terminate(self):
            if terminate is not None:
                terminate(self.pid)

    module = types.SimpleNamespace(
        NoSuchProcess=_NoSuchProcess,
        Process=lambda pid=None: _Proc(pid),
        process_iter=lambda fields: list(processes),
    )
    if memory is None:
        def broken():
            raise RuntimeError("no memory")
        module.virtual_memory = broken
    else:
        module.virtual_memory = lambda: types.SimpleNamespace(
            available=memory[0], total=memory[1])
    return module


def _proc(pid, name, user="me", rss=100, broken=False):
    class _P:
        @property
        def info(self):
            if broken:
                raise RuntimeError("vanished")
            return {"pid": pid, "name": name, "username": user,
                    "memory_info": types.SimpleNamespace(rss=rss)}
    return _P()


def test_spacr_ids_fall_back_to_this_process(monkeypatch):
    import os

    assert os.getpid() in rl._spacr_process_ids(_psutil(me="broken"))
    assert rl._current_username(_psutil(username=None)) is None


def test_without_psutil_nothing_is_listed_or_closed(monkeypatch):
    monkeypatch.setattr(rl, "_psutil", lambda: None)
    assert rl._closable_processes() == []
    assert rl._close_processes([123]) == {}
    assert rl._ram_snapshot() is None


def test_a_process_that_vanishes_while_listed_is_skipped():
    module = _psutil([_proc(500, "editor", broken=True),
                      _proc(501, "browser", rss=900)])
    assert [row["pid"] for row in rl._closable_processes(module)] == [501]


def test_closing_reports_gone_and_denied_processes():
    def terminate(pid):
        if pid == 501:
            raise _NoSuchProcess()
        raise PermissionError("not yours")

    module = _psutil([_proc(501, "browser"), _proc(502, "editor")],
                     terminate=terminate)
    assert rl._close_processes([501, 502], module) == {501: "gone",
                                                       502: "denied"}


def test_an_unreadable_memory_snapshot_gives_no_plan():
    assert rl._ram_snapshot(_psutil(memory=None)) is None
    assert rl._ram_plan(1 << 20, 4, psutil_module=_psutil(memory=None)) is None


def test_a_worker_of_no_size_has_no_safe_count():
    assert rl._max_safe_workers(1 << 30, 2 << 30, 0) is None


def test_the_sample_input_is_found_in_a_list_of_sources(tmp_path):
    folder = tmp_path / "plate"
    folder.mkdir()
    (folder / "a.npy").write_bytes(b"x")
    assert rl._sample_input_file([str(tmp_path / "none"), str(folder)],
                                 (".npy",)) == str(folder / "a.npy")
    assert rl._sample_input_file([], (".npy",)) is None
    assert rl._sample_input_file(str(folder), ()) is None
    assert rl._sample_input_file(str(folder / "a.npy"), (".npy",)) == str(
        folder / "a.npy")
    assert rl._sample_input_file(str(folder / "a.npy"), (".tif",)) is None


def test_a_huge_folder_is_not_walked_to_the_end(tmp_path, monkeypatch):
    for index in range(5):
        (tmp_path / f"f{index}.txt").write_text("x")
    monkeypatch.setattr(rl, "_SAMPLE_WALK_LIMIT", 2)
    assert rl._sample_input_file(str(tmp_path), (".npy",)) is None


def test_a_plan_that_cannot_be_estimated_keeps_n_jobs(monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("estimate failed")

    monkeypatch.setattr(rl, "_ram_plan", broken)
    assert rl._guard_workers("mask", 6, 1 << 20, settings={}) == 6


def test_unreadable_sizes_are_zero():
    class _Odd:
        @property
        def nbytes(self):
            raise RuntimeError("no size")

    assert rl._table_nbytes(_Odd()) == 0
    assert rl._loader_unit_bytes("many", 224) == 0
    assert rl._loader_unit_bytes(2, 8, channels=[0, 1]) == 2 * 8 * 8 * 2 * 4 * 2


def test_app_unit_sizes_follow_each_module(tmp_path):
    assert rl._app_unit_bytes("plot", {}) == 0
    assert rl._app_unit_bytes("map_barcodes", {"chunk_size": "lots"}) == (
        10000 * 2 * 1024)
    image = tmp_path / "a.png"
    np.save(tmp_path / "x.npy", np.zeros(1))
    from PIL import Image
    Image.fromarray(np.zeros((4, 4), np.uint8)).save(image)
    single = rl._app_unit_bytes("classify", {"src": str(tmp_path),
                                             "batch_size": "x"})
    double = rl._app_unit_bytes("classify", {"src": str(tmp_path),
                                             "batch_size": 2})
    assert double == 2 * single > 0
    assert rl._app_unit_bytes("train_cellpose", {"src": str(tmp_path / "none")}) == 0


def test_tiff_and_npz_inputs_are_sized_from_their_headers(tmp_path):
    import tifffile

    tifffile.imwrite(tmp_path / "a.tif", np.zeros((3, 4, 5), np.uint16))
    np.savez(tmp_path / "b.npz", one=np.zeros(4, np.float32),
             two=np.zeros(2, np.float64))
    assert rl._array_file_nbytes(str(tmp_path / "a.tif")) == 3 * 4 * 5 * 2
    assert rl._array_file_nbytes(str(tmp_path / "b.npz")) == 16 + 16


def test_a_data_frame_is_sized_from_its_columns():
    import pandas as pd

    frame = pd.DataFrame({"a": np.zeros(10)})
    assert rl._table_nbytes(frame) >= 80


def test_measure_samples_its_merged_field(tmp_path):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "plate1_A01_1.npy", np.zeros((4, 4), np.float32))
    assert rl._app_unit_bytes("measure", {"src": str(tmp_path)}) == 64


def test_a_scope_reads_the_run_choice():
    assert rl._ram_guard_scope({"ram_guard": False})._value is False
    assert rl._ram_guard_scope(None)._value is True
