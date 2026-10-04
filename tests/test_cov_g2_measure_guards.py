"""Measure's device choice, RAM sampling, backend copy and CellProfiler export edges."""
from __future__ import annotations

import builtins
import sys
import threading
import types

import numpy as np
import pytest

from spacr import measure as M


def test_measure_gpu_without_torch_or_cuda_measures_on_the_cpu(
        monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert M._measurement_device({"measure_gpu": True}) is None
    assert "no CUDA device" in capsys.readouterr().out
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert str(M._measurement_device({"measure_gpu": True})) == "cuda"

    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    assert M._measurement_device({"measure_gpu": True}) is None
    assert "PyTorch is not installed" in capsys.readouterr().out


def test_complex_or_half_precision_images_stay_on_the_cpu():
    labels = np.zeros((4, 4), np.int32)
    assert M._gpu_measurable(labels, np.zeros((4, 4), np.complex64)) is False
    assert M._gpu_measurable(labels, np.zeros((4, 4), np.float16)) is False
    assert M._gpu_measurable(labels, np.zeros((4, 4), np.float32)) is True


def test_psutil_missing_is_reported_as_none(monkeypatch):
    real_import = builtins.__import__

    def no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psutil)
    assert M._psutil_or_none() is None


def test_the_sample_field_comes_from_the_first_folder(tmp_path, monkeypatch):
    merged = tmp_path / "merged"
    merged.mkdir()
    np.save(merged / "b.npy", np.zeros(3))
    np.save(merged / "a.npy", np.zeros(5))
    assert M._sample_field_path([str(tmp_path)]).endswith("a.npy")
    assert M._sample_field_path(str(tmp_path / "empty_merged")) is None
    assert M._sample_field_path([]) is None
    assert M._sample_field_path(str(tmp_path / "none")) is None
    assert M._field_nbytes(str(tmp_path / "none.npy")) == 0
    assert M._ram_guard_plan(str(tmp_path / "none"), 4) is None

    def refuse(path):
        raise OSError("unreadable")

    monkeypatch.setattr(M.os, "listdir", refuse)
    assert M._sample_field_path(str(tmp_path)) is None


def test_child_memory_sampling_survives_vanishing_processes():
    stop_after = []

    class _Child:
        def __init__(self, rss=None):
            self.rss = rss

        def memory_info(self):
            if self.rss is None:
                raise RuntimeError("gone")
            return types.SimpleNamespace(rss=self.rss)

    class _Me:
        calls = 0

        def children(self, recursive=True):
            _Me.calls += 1
            if _Me.calls == 2:
                raise RuntimeError("table changed")
            if _Me.calls >= 3:
                stop_after.append(True)
                sampler._stop.set()
            return [_Child(), _Child(1234)]

    module = types.SimpleNamespace(Process=lambda: _Me())
    sampler = M._PeakChildMemory(psutil_module=module, interval=0)
    sampler._poll()
    assert sampler.peak == 1234 and stop_after

    def broken():
        raise RuntimeError("no process")

    quiet = M._PeakChildMemory(psutil_module=types.SimpleNamespace(
        Process=broken), interval=0)
    quiet._poll()
    assert quiet.peak == 0


def test_without_psutil_the_sampler_starts_no_thread(monkeypatch):
    monkeypatch.setattr(M, "_psutil_or_none", lambda: None)
    with M._PeakChildMemory() as sampler:
        assert sampler._thread is None


def test_waiting_for_ram_without_a_snapshot_returns_at_once(monkeypatch):
    monkeypatch.setattr(M, "_ram_snapshot", lambda module=None: None)
    assert M._wait_for_ram(1 << 30, lambda: True) == 0.0


def test_a_held_field_starts_alone_once_nothing_else_runs(monkeypatch, capsys):
    gib = 1 << 30
    monkeypatch.setattr(M, "_ram_snapshot", lambda module=None: (gib, 16 * gib))
    busy = iter([True, False])
    waited = M._wait_for_ram(4 * gib, lambda: next(busy), field="A01",
                             sleep=lambda s: None, poll=1)
    out = capsys.readouterr().out
    assert waited == 1
    assert "holding" in out and "starting A01 alone" in out


def test_a_failed_backend_copy_is_reported_not_raised(tmp_path, monkeypatch,
                                                      capsys):
    import spacr.tabular as tabular

    db = tmp_path / "measurements.db"
    db.write_bytes(b"")
    monkeypatch.setattr(M, "_measurement_backend_target",
                        lambda db_path, settings: str(tmp_path / "out.duckdb"))
    calls = iter([RuntimeError("unreadable"), ["cell"]])

    def tables(path):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    def migrate(source, target, tables):
        raise KeyError("odd")

    monkeypatch.setattr(tabular, "database_tables", tables)
    monkeypatch.setattr(tabular, "_migrate_database", migrate)
    assert M._copy_to_measurement_backend(str(db), {"measurement_backend": "duckdb"}) is None
    assert "copy failed (KeyError" in capsys.readouterr().out

    def bad_target(db_path, settings):
        raise ValueError("unknown backend")

    monkeypatch.setattr(M, "_measurement_backend_target", bad_target)
    assert M._copy_to_measurement_backend(str(db), {}) is None
    assert "copy skipped" in capsys.readouterr().out


def test_cellprofiler_export_skips_files_that_are_not_stacks(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    np.save(tmp_path / "flat.npy", np.zeros((4, 4)))
    assert M._cellprofiler_export(str(tmp_path), {}, str(tmp_path / "out")) == []


def test_cellprofiler_overlap_refuses_a_malformed_mask(tmp_path):
    assert M._cellprofiler_overlap_labels([], np.zeros((2, 2, 2))) == {}
    assert M._cellprofiler_overlap_labels([], np.full((2, 2), -1)) == {}
