"""Item 593 on the Make Masks screen, offscreen.

A dropped folder whose images sit in subfolders is offered for
consolidation; the answer is monkeypatched because a headless ``_confirm``
always says no. The example-sets check is driven through its methods, and the
screen's off-thread copy and move are waited for.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import channel_sort_dialog as csd


def _tif(path: Path, array) -> Path:
    """Write ``array`` as a TIFF, making the folder.

    :param path: where.
    :param array: the pixels.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A Make Masks screen, closed down afterwards."""
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    worker = getattr(made, "_folder_job", None)
    if worker is not None:
        worker.wait(10000)
    made._magnifier.close()
    made.close_folded()


def _nested(root: Path) -> Path:
    """``exp/`` with one top image, two subfolders of images and a masks folder.

    The subfolders do not look like channels (item 600 opens "Organize for
    Measure" for those): their names are not channel names and their
    fields differ.

    :param root: the parent.
    """
    src = root / "exp"
    _tif(src / "top.tif", np.zeros((16, 16), np.uint16))
    _tif(src / "run1" / "a.tif", np.zeros((16, 16), np.uint16))
    _tif(src / "run2" / "b.tif", np.zeros((16, 16), np.uint16))
    _tif(src / "masks" / "top.tif", np.zeros((16, 16), np.uint16))
    return src


def test_the_organize_button_is_on_the_screen(screen):
    """Item 600 replaced 593's two buttons with one "Organize for Measure…"."""
    assert screen.findChild(type(screen._btn_open), "MakeMasksOrganizeButton")
    assert not screen.findChild(type(screen._btn_open),
                                "MakeMasksSortChannelsButton")


def test_a_headless_drop_opens_the_folder_as_it_is(screen, tmp_path,
                                                  monkeypatch, qtbot):
    """Nobody can answer the question, so nothing is copied."""
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    src = _nested(tmp_path)
    assert screen.open_paths([str(src)])
    qtbot.waitUntil(lambda: not screen._loading
                    and screen._reader_worker is None
                    and screen._load_worker is None)
    assert screen._folder == str(src)
    assert not (tmp_path / "exp_renamed").exists()


def test_a_folder_without_nested_images_is_not_offered(screen, tmp_path, monkeypatch):
    """Only a masks subfolder: no question at all."""
    folder = tmp_path / "flat"
    _tif(folder / "a.tif", np.zeros((8, 8), np.uint16))
    _tif(folder / "masks" / "a.tif", np.zeros((8, 8), np.uint16))
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda *a: asked.append(a) or True)
    assert screen.open_paths([str(folder)])
    assert not asked


def test_yes_consolidates_off_thread_and_opens_the_new_folder(
        screen, tmp_path, monkeypatch, qtbot):
    """The copy runs in a worker; the new folder opens when it ends."""
    src = _nested(tmp_path)
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda *a: asked.append(a) or True)
    assert screen.open_paths([str(src)])
    assert "2 image(s) in 2 subfolder(s)" in asked[0][1]
    output = tmp_path / "exp_renamed"
    qtbot.waitUntil(lambda: screen._folder == str(output), timeout=20000)
    assert sorted(screen._image_files) == ["exp.tif", "exp_run1.tif", "exp_run2.tif"]
    assert (src / "run1" / "a.tif").is_file()


def _drawn(root: Path) -> Path:
    """Nucleus and cell images of three fields, all masked, in one folder.

    :param root: the parent.
    """
    folder = root / "drawn"
    for i in range(1, 4):
        for stain, value in (("nuc", 10), ("cell", 100)):
            name = f"{stain}_img{i:02d}.tif"
            _tif(folder / name, np.full((12, 12), value + i, np.uint16))
            _tif(folder / "masks" / name, np.full((12, 12), i, np.uint16))
    return folder


def test_detected_sets_are_shown_three_at_a_time(qtbot, qt_theme_applied, tmp_path):
    """Detect sets pairs the images; the check shows three of them."""
    from spacr import channel_sorting as cs
    folder = _drawn(tmp_path)
    names = sorted(n for n in os.listdir(folder) if n.endswith(".tif"))
    channels = {n: 1 if n.startswith("nuc") else 2 for n in names}
    sets = cs.detect_sets(str(folder), channels).sets
    assert len(sets) == 3
    example = csd.ExampleSetsDialog(str(folder), sets)
    qtbot.addWidget(example)
    assert len(example.shown) == 3


def test_the_screen_moves_and_merges_off_thread(screen, tmp_path, qtbot, monkeypatch):
    """A confirmed plan is applied in a worker and the first channel opens."""
    from spacr import channel_sorting as cs

    folder = _drawn(tmp_path)
    header_read, release_header = threading.Event(), threading.Event()
    moved, release_move = threading.Event(), threading.Event()
    allow_move = threading.Event()
    pending_path, missing_reads, worker_threads = [], [], []
    main_thread = threading.get_ident()
    real_header = mm.engine._container_image_header
    real_move = cs.shutil.move
    real_load = mm.engine.load_image_and_mask

    def header(path):
        """Hold only the real original-folder header after its TIFF read."""
        worker_threads.append(threading.get_ident())
        result = real_header(path)
        if Path(path).parent == folder and not header_read.is_set():
            pending_path.append(Path(path))
            header_read.set()
            assert release_header.wait(10)
        return result

    def move(source, destination, *args, **kwargs):
        """Pause the actual sort only after its pending source has moved."""
        worker_threads.append(threading.get_ident())
        if Path(source) == pending_path[0]:
            assert allow_move.wait(10)
        result = real_move(source, destination, *args, **kwargs)
        if Path(source) == pending_path[0]:
            moved.set()
            assert release_move.wait(10)
        return result

    def load(folder, filename, *args, **kwargs):
        """Observe real decoding attempts without replacing their result."""
        path = Path(folder) / filename
        if path == pending_path[0]:
            allow_move.set()
            assert moved.wait(10)
        if not path.exists():
            missing_reads.append(path)
        return real_load(folder, filename, *args, **kwargs)

    monkeypatch.setattr(mm.engine, "_container_image_header", header)
    monkeypatch.setattr(cs.shutil, "move", move)
    monkeypatch.setattr(mm.engine, "load_image_and_mask", load)
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(header_read.is_set)
    pending_reader = screen._reader_worker
    assert pending_reader is not None and screen._btn_organize.isEnabled()
    names = cs.list_folder_images(str(folder))
    report = cs.check_sets(cs.parse_names(names, cs.infer_regex(names)))
    plan = cs.build_plan(str(folder), report.complete)
    originals = {target: Path(source).read_bytes()
                 for row in plan.rows
                 for source, target in ((row.source_image, row.target_image),
                                        (row.source_mask, row.target_mask))
                 if source and target}
    try:
        assert screen._start_channel_sort(plan)
        release_header.set()
        qtbot.waitUntil(lambda: pending_reader not in screen._reader_workers)
        allow_move.set()
        qtbot.waitUntil(moved.is_set)
        assert not pending_path[0].exists()
        assert missing_reads == []
    finally:
        release_header.set()
        allow_move.set()
        release_move.set()
    first = os.path.join(plan.dest, "C01")
    qtbot.waitUntil(lambda: screen._folder == first, timeout=60000)
    merged = os.listdir(os.path.join(plan.dest, "merged"))
    assert len([n for n in merged if n.endswith(".npy")]) == 3
    assert worker_threads and main_thread not in worker_threads
    assert {target: Path(target).read_bytes() for target in originals} == originals
