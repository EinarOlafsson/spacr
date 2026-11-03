"""Opening a Time movie folds every frame's embedded lineage off the GUI thread.

Each mask of a Time movie embeds the whole movie's lineage graph, so the old
per-frame merge on the GUI thread grew with frames x objects and froze Make
Masks for tens of seconds on real movies. The fold now validates one copy of
each distinct embedded graph, skips revalidating a merge that changes
nothing, and runs on a worker. These tests pin that the merged graph is
byte-identical to the previous algorithm and that Qt keeps running.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid

import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QEventLoop, QTimer

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.tiff_io import write_tiff
from tests.qt.gui_timing import assert_within_budget
from tests.qt.test_gui_responsiveness import LoopWatchdog

FRAMES, CELLS = 90, 24


def _reference_merge(active, incoming, frame=None):
    """The merge exactly as it ran before the linear fold, kept as the oracle."""
    result, source = engine.validate_lineage(active), engine.validate_lineage(incoming)
    current = os.path.abspath(frame) if frame is not None else None
    known = {identity: path for path, identity in result["frame_ids"].items()}
    active_paths, source_paths = {}, {}
    for path, identity in source["frame_ids"].items():
        if identity in known:
            if path == current:
                active_paths[known[identity]] = path
            else:
                source_paths[path] = known[identity]
        elif path in result["frame_ids"] and result["frame_ids"][path] != identity:
            raise ValueError("frame identity")
    result = engine._remap_lineage_frames(result, active_paths)
    source = engine._remap_lineage_frames(source, source_paths)
    result["frame_ids"].update(source["frame_ids"])
    for identity, revision in source["deleted_cells"].items():
        result["deleted_cells"][identity] = max(
            revision, result["deleted_cells"].get(identity, 0))
    for identity, cell in source["cells"].items():
        old = result["cells"].get(identity)
        if old is None:
            result["cells"][identity] = cell
            continue
        if old.get("born_frame_id") != cell.get("born_frame_id") or (
                old.get("revision", 0) == cell.get("revision", 0) and any(
                    old.get(key) != cell.get(key) for key in (
                        "parent_uuid", "children", "division_frame_id"))):
            raise ValueError("ancestry")
        newer = cell if cell.get("revision", 0) > old.get("revision", 0) else old
        result["cells"][identity] = dict(newer)
    for identity, revision in result["deleted_cells"].items():
        if identity in result["cells"] and revision >= result["cells"][identity].get("revision", 0):
            del result["cells"][identity]
    for name, mapping in source["frames"].items():
        if name not in result["frames"] or ((current is None or name == current)
                and source["frame_revisions"].get(name, 0)
                > result["frame_revisions"].get(name, 0)):
            result["frames"][name] = mapping
            result["frame_revisions"][name] = source["frame_revisions"].get(name, 0)
    for mapping in result["frames"].values():
        for label, identity in list(mapping.items()):
            if identity not in result["cells"] and identity in result["deleted_cells"]:
                del mapping[label]
    return engine.validate_lineage(result)


def _reference_fold(state, folder, names, skip=None):
    """The old per-frame loop of ``_merge_sequence_lineage``."""
    skipped = 0
    for name in names:
        if name == skip:
            continue
        try:
            incoming = engine.load_mask_metadata(folder, name).get("lineage")
            if incoming is not None:
                state = _reference_merge(state, incoming,
                                         os.path.abspath(os.path.join(folder, name)))
        except engine.TIFF_READ_ERRORS:
            skipped += 1
    return state, skipped


def _graph(folder, frames=FRAMES, cells=CELLS, revision=10):
    """A whole-movie graph: ``cells`` founders, one dividing half way."""
    paths = [os.path.abspath(os.path.join(folder, f"t{index:03d}.tif"))
             for index in range(frames)]
    ids = [str(uuid.UUID(int=index + 1)) for index in range(cells + 2)]
    graph = {"schema": engine.LINEAGE_SCHEMA, "cells": {}, "frames": {},
             "frame_revisions": {}, "deleted_cells": {}}
    for index in range(cells):
        graph["cells"][ids[index]] = {
            "name": str(index + 1), "parent_uuid": None, "children": [],
            "born_frame": paths[0], "division_frame": None,
            "label": index + 1, "revision": revision}
    half = frames // 2
    mother = graph["cells"][ids[0]]
    mother.update(children=ids[cells:], division_frame=paths[half])
    for offset, child in enumerate(ids[cells:]):
        graph["cells"][child] = {
            "name": f"1{'ab'[offset]}", "parent_uuid": ids[0], "children": [],
            "born_frame": paths[half], "division_frame": None,
            "label": 100 + offset, "revision": revision}
    for index, path in enumerate(paths):
        mapping = {str(cell + 1): ids[cell] for cell in range(1, cells)}
        if index < half:
            mapping["1"] = ids[0]
        else:
            mapping.update({"100": ids[cells], "101": ids[cells + 1]})
        graph["frames"][path] = mapping
        graph["frame_revisions"][path] = revision
    return engine.validate_lineage(graph), ids


def _mask(cells=CELLS, daughters=False):
    mask = np.zeros((48, 64), np.uint32)
    for cell in range(cells):
        row, col = divmod(cell, 8)
        mask[row * 12 + 2:row * 12 + 9, col * 8 + 1:col * 8 + 7] = cell + 1
    if daughters:
        mask[mask == 1] = 0
        mask[2:5, 1:7] = 100
        mask[6:9, 1:7] = 101
    return mask


def _write_movie(folder, frames=FRAMES, graphs=None, bare=()):
    """Frames whose masks embed ``graphs[index]`` (the whole graph by default)."""
    os.makedirs(os.path.join(folder, "masks"), exist_ok=True)
    whole, ids = _graph(folder, frames)
    for index in range(frames):
        name = f"t{index:03d}.tif"
        write_tiff(os.path.join(folder, name), np.full((48, 64), index + 1, np.uint16))
        mask = _mask(daughters=index >= frames // 2)
        lineage = None if index in bare else (graphs or {}).get(index, whole)
        engine.save_mask(folder, name, mask, preserve_ids=True, lineage=lineage)
    return whole, ids


def _unreadable(path):
    """A mask whose embedded metadata names spaCR but cannot be parsed."""
    tifffile.imwrite(path, _mask(), description='{"spacr_mask": ', metadata=None)


def _names(folder):
    return sorted(name for name in os.listdir(folder) if name.endswith(".tif"))


def _dumps(state):
    return json.dumps(state)


def test_the_linear_fold_is_byte_identical_to_the_old_merge(tmp_path):
    folder = str(tmp_path / "movie")
    whole, ids = _write_movie(folder, frames=30)
    first = os.path.abspath(os.path.join(folder, "t000.tif"))
    renamed = engine.rename_lineage_cell(whole, ids[3], "renamed")
    older = engine.validate_lineage(whole)
    older["cells"][ids[5]]["name"] = "older"
    older["cells"][ids[5]]["revision"] = 1
    without = engine.validate_lineage(whole)
    del without["cells"][ids[7]]
    for mapping in without["frames"].values():
        mapping.pop("8", None)
    retired = engine.restore_lineage_state(without, whole)
    clash = engine.validate_lineage(whole)
    clash["cells"][ids[4]]["born_frame"] = os.path.join(folder, "t001.tif")
    clash["cells"][ids[4]].pop("born_frame_id")
    for index, graph in ((4, renamed), (9, older), (14, retired), (20, clash)):
        engine.save_mask(folder, f"t{index:03d}.tif", _mask(), preserve_ids=True,
                         lineage=graph)
    _unreadable(os.path.join(folder, "masks", "t025.tif"))
    names = _names(folder)
    for start in (None, whole):
        expected = _reference_fold(start, folder, names, skip="t000.tif")
        assert expected[1] == 2
        got = engine._merge_mask_lineages(start, folder, names, skip="t000.tif")
        assert _dumps(got[0]) == _dumps(expected[0]) and got[1] == expected[1]
    assert ids[7] not in got[0]["cells"] and ids[7] in got[0]["deleted_cells"]
    assert got[0]["cells"][ids[3]]["name"] == "renamed"
    moved = str(tmp_path / "moved")
    shutil.copytree(folder, moved)
    for start in (engine.load_mask_metadata(folder, "t000.tif")["lineage"], None):
        expected = _reference_fold(start, moved, names)
        got = engine._merge_mask_lineages(start, moved, names)
        assert _dumps(got[0]) == _dumps(expected[0]) and got[1] == expected[1]
    assert all(path.startswith(os.path.abspath(moved)) for path in got[0]["frames"])
    for active, incoming, frame in (
            (whole, retired, first), (retired, whole, None), (older, renamed, first),
            (engine.load_mask_metadata(moved, "t003.tif")["lineage"], whole,
             os.path.join(folder, "t003.tif"))):
        assert _dumps(engine.merge_lineage_states(active, incoming, frame)) == \
            _dumps(_reference_merge(active, incoming, frame))
    assert engine._merge_mask_lineages(whole, folder, names, interrupted=lambda: True) \
        == (None, 0)
    bare = engine.validate_lineage(renamed)
    for cell in bare["cells"].values():
        if cell["division_frame"] is None:
            del cell["division_frame"]
    assert _dumps(engine.merge_lineage_states(bare, whole, first)) == \
        _dumps(_reference_merge(bare, whole, first))
    moved_frame = engine.validate_lineage(whole)
    moved_frame["frame_ids"][os.path.abspath(os.path.join(folder, "t005.tif"))] = \
        str(uuid.UUID(int=999))
    for merge in (engine.merge_lineage_states, _reference_merge):
        with pytest.raises(ValueError, match="identity"):
            merge(whole, moved_frame)
    with pytest.raises(ValueError, match="embedded lineage"):
        engine.validate_lineage({"schema": engine.LINEAGE_SCHEMA, "bad": float("nan")})


def _spin(qtbot, condition, timeout=60.0):
    deadline = time.perf_counter() + timeout
    while not condition() and time.perf_counter() < deadline:
        loop = QEventLoop()
        QTimer.singleShot(5, loop.quit)
        loop.exec()
    assert condition()


@pytest.fixture(autouse=True)
def traced_scans(monkeypatch):
    """Let the lineage scan worker inherit the coverage trace, as the loaders do."""
    import sys
    from functools import wraps

    original = mm._LineageScanWorker.run

    @wraps(original)
    def traced_run(worker):
        """Run the scan under the trace the test thread was started with."""
        previous, inherited = sys.gettrace(), threading.gettrace()
        installed = previous is None and inherited is not None
        if installed:
            sys.settrace(inherited)
        try:
            return original(worker)
        finally:
            if installed:
                sys.settrace(previous)

    monkeypatch.setattr(mm._LineageScanWorker, "run", traced_run)


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    widget = mm.MakeMasksScreen()
    monkeypatch.setattr(widget, "_confirm", lambda *args: True)
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def _idle(screen):
    return (not screen._loading and screen._load_worker is None
            and not screen._lineage_scans and screen._canvas.mask is not None)


def test_opening_a_long_movie_keeps_qt_running_and_merges_identically(
        qtbot, screen, tmp_path):
    assert screen._sequence_frame_names() == []
    folder = str(tmp_path / "movie")
    _write_movie(folder)
    names = _names(folder)
    first = engine.load_mask_metadata(folder, names[0])["lineage"]
    expected, _ = _reference_fold(
        _reference_merge(None, first, os.path.join(folder, names[0])), folder, names,
        skip=names[0])
    warm = str(tmp_path / "warm")
    _write_movie(warm, frames=2)
    assert screen._open_folder(warm)
    _spin(qtbot, lambda: _idle(screen))
    watchdog = LoopWatchdog(interval_ms=1)
    watchdog.start()
    assert screen._open_folder(folder)
    _spin(qtbot, lambda: _idle(screen))
    watchdog.stop()
    assert _dumps(screen._lineage) == _dumps(expected)
    assert screen._lineage_scanned == folder and not screen._lineage_dirty
    assert watchdog.ticks > 0
    assert_within_budget(watchdog.worst, 0.5)
    assert _dumps(screen._merge_sequence_lineage(None, folder)) == \
        _dumps(_reference_fold(None, folder, names)[0])
    screen._btn_label_time.setChecked(True)
    assert screen._canvas.mode == mm.MODE_LABEL_TIME


def _gate(monkeypatch, fail=False):
    """Hold every scan until the test releases it; optionally make it raise."""
    released, started = threading.Event(), []
    real = engine._merge_mask_lineages

    def held(*args, **kwargs):
        """Wait for the release, then fold (or fail) as the real scan would."""
        started.append(True)
        released.wait(30)
        if fail:
            raise RuntimeError("scan exploded")
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_merge_mask_lineages", held)
    return released, started


def test_a_scan_outlived_by_navigation_or_a_lineage_change_is_not_installed(
        qtbot, screen, tmp_path, monkeypatch):
    folder = str(tmp_path / "movie")
    whole, _ = _write_movie(folder, frames=8)
    released, started = _gate(monkeypatch)
    assert screen._open_folder(folder)
    _spin(qtbot, lambda: started)
    assert screen._loading and screen._canvas.mask is None
    screen._on_next()
    _spin(qtbot, lambda: len(started) == 2)
    released.set()
    _spin(qtbot, lambda: _idle(screen))
    assert screen._current_index == 1 and screen._lineage_scanned == folder
    assert set(screen._lineage["cells"]) == set(whole["cells"])
    released.clear()
    started.clear()
    assert screen._open_folder(folder)
    _spin(qtbot, lambda: started)
    screen._lineage = engine.validate_lineage(None)
    released.set()
    _spin(qtbot, lambda: _idle(screen))
    assert len(started) == 2 and screen._current_index == 0
    assert set(screen._lineage["cells"]) == set(whole["cells"])


def test_a_failed_scan_reports_the_load_failure(qtbot, screen, tmp_path, monkeypatch):
    folder = str(tmp_path / "movie")
    _write_movie(folder, frames=4)
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append(text))
    released, started = _gate(monkeypatch, fail=True)
    released.set()
    assert screen._open_folder(folder)
    _spin(qtbot, lambda: started and not screen._lineage_scans and not screen._loading)
    assert warned and "scan exploded" in warned[-1]
    assert screen._canvas.mask is None and screen._lineage_scanned is None


def test_an_unreadable_mask_is_counted_and_closing_mid_scan_is_safe(
        qtbot, screen, tmp_path, monkeypatch):
    folder = str(tmp_path / "movie")
    _write_movie(folder, frames=6)
    _unreadable(os.path.join(folder, "masks", "t004.tif"))
    shown, real_text = [], screen._status_label.setText
    monkeypatch.setattr(screen._status_label, "setText",
                        lambda text: (shown.append(text), real_text(text)))
    assert screen._open_folder(folder)
    _spin(qtbot, lambda: _idle(screen))
    assert any("1 mask file(s) could not be read" in text for text in shown)
    released, started = _gate(monkeypatch)
    screen._lineage_scanned = None
    screen._on_next()
    _spin(qtbot, lambda: started)
    threading.Timer(0.2, released.set).start()
    screen.close()
    assert not screen._lineage_scans
