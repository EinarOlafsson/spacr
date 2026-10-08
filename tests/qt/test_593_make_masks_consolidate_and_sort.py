"""Item 593 on the Make Masks screen, offscreen.

A dropped folder whose images sit in subfolders is offered for
consolidation; the answer is monkeypatched because a headless ``_confirm``
always says no. The "Sort into channels…" dialog, its regex window and the
example-sets check are driven through their methods, and the screen's
off-thread copy and move are waited for.
"""
from __future__ import annotations

import os
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
                                                  monkeypatch):
    """Nobody can answer the question, so nothing is copied."""
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    src = _nested(tmp_path)
    assert screen.open_paths([str(src)])
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


def test_the_sort_dialog_lists_proposes_and_plans(qtbot, qt_theme_applied,
                                                  tmp_path, monkeypatch):
    """The regex is proposed on opening; the plan moves six images."""
    monkeypatch.setattr(csd, "_headless", lambda: True)
    folder = _drawn(tmp_path)
    dialog = csd.ChannelSortDialog(str(folder))
    qtbot.addWidget(dialog)
    assert dialog.list.count() == 6
    assert "chanID" in dialog.regex
    plan = dialog.prepare_plan()
    assert plan is not None and plan.ok
    assert plan.mask_roles == {1: "cell", 2: "nucleus"}
    dialog._on_apply()
    assert dialog.plan is plan or dialog.plan is not None
    dialog._stop_thumbs()


def test_selection_strategy_assigns_channels_and_bad_sets_ask_for_a_regex(
        qtbot, qt_theme_applied, tmp_path, monkeypatch):
    """Channels by hand; with no regex the sets are missing and the window opens."""
    monkeypatch.setattr(csd, "_headless", lambda: True)
    folder = _drawn(tmp_path)
    dialog = csd.ChannelSortDialog(str(folder))
    qtbot.addWidget(dialog)
    dialog.set_regex("")
    opened = []
    dialog.open_regex_window = lambda: opened.append(True) or False
    for i in range(dialog.list.count()):
        item = dialog.list.item(i)
        item.setSelected(item.data(0x0100).startswith("nuc"))
    dialog.assign_selected(1)
    dialog.list.selectAll()
    for i in range(dialog.list.count()):
        item = dialog.list.item(i)
        item.setSelected(item.data(0x0100).startswith("cell"))
    dialog.assign_selected(2)
    assert set(dialog.channels.values()) == {1, 2}
    assert dialog.prepare_plan() is None
    # Headless, the "change the regex?" question answers no.
    dialog._on_apply()
    assert dialog.plan is None
    dialog._stop_thumbs()


def test_the_regex_window_tests_infers_and_detects(qtbot, qt_theme_applied, tmp_path):
    """Live test table, Auto regex, and Detect sets with the check confirmed."""
    folder = _drawn(tmp_path)
    names = sorted(os.listdir(folder))
    names = [n for n in names if n.endswith(".tif")]
    channels = {n: 1 if n.startswith("nuc") else 2 for n in names}
    window = csd.RegexWindow(str(folder), names, channels, "nothing")
    qtbot.addWidget(window)
    report = window.run_test()
    assert window.table.rowCount() == 6 and not report.ok
    assert window.auto_regex()
    assert window.run_test().ok

    shown = []
    window.confirm_examples = lambda sets: shown.append(sets) or True
    sets = window.detect()
    assert sets and len(sets) == 3
    assert window.detected == sets
    example = csd.ExampleSetsDialog(str(folder), sets)
    qtbot.addWidget(example)
    assert len(example.shown) == 3


def test_the_screen_moves_and_merges_off_thread(screen, tmp_path, qtbot):
    """A confirmed plan is applied in a worker and the first channel opens."""
    from spacr import channel_sorting as cs

    folder = _drawn(tmp_path)
    assert screen._open_folder(str(folder))
    names = cs.list_folder_images(str(folder))
    report = cs.check_sets(cs.parse_names(names, cs.infer_regex(names)))
    plan = cs.build_plan(str(folder), report.complete)
    assert screen._start_channel_sort(plan)
    first = os.path.join(plan.dest, "C01")
    qtbot.waitUntil(lambda: screen._folder == first, timeout=60000)
    merged = os.listdir(os.path.join(plan.dest, "merged"))
    assert len([n for n in merged if n.endswith(".npy")]) == 3
