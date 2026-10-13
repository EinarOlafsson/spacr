"""Item 600: Make Masks understands what is dropped on it, offscreen.

``_confirm`` is headless-no, so tests that need a yes monkeypatch it; the
"Organize for Measure" popup is not run headless, so tests drive it through
``_run_organize``. The end-to-end test drops two folders, lets the popup
match them as two channels, applies, and reads the Yokogawa-named channel
folders and merged arrays the screen's worker wrote.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr.qt.dnd_handlers import MakeMasksDropHandler
from spacr.qt.screens import make_masks as mm

pytestmark = pytest.mark.qt


def _tif(path: Path, array=None) -> Path:
    """Write a TIFF, making the folder.

    :param path: where.
    :param array: the pixels; default a small noisy image.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if array is None:
        array = np.random.default_rng(len(str(path))).integers(
            100, 4000, (16, 16)).astype(np.uint16)
    tifffile.imwrite(str(path), np.asarray(array))
    return path


def _mask(i: int = 1) -> np.ndarray:
    """A label mask with one object.

    :param i: its label.
    """
    array = np.zeros((16, 16), np.uint16)
    array[3:8, 3:8] = i
    return array


def _tree(root: Path) -> Path:
    """``exp/DAPI`` and ``exp/GFP``, three masked fields each.

    :param root: the parent.
    """
    exp = root / "exp"
    for channel in ("DAPI", "GFP"):
        for i in range(1, 4):
            _tif(exp / channel / f"field{i}.tif")
            _tif(exp / channel / "masks" / f"field{i}.tif", _mask(i))
    return exp


@pytest.fixture
def screen(qtbot, qt_theme_applied, monkeypatch):
    """Keep unattended drop decisions headless, even with an Xvfb display."""
    monkeypatch.setattr(mm, "is_headless", lambda: True)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    worker = getattr(made, "_folder_job", None)
    if worker is not None:
        worker.wait(20000)
    made._magnifier.close()
    made.close_folded()


def _apply_popup(dialog) -> bool:
    """Answer the popup the way a user who presses Apply and Yes would.

    :param dialog: the "Organize for Measure" popup.
    :returns: whether it was accepted with a plan.
    """
    dialog.ask = lambda _title, _text: True
    dialog._on_apply()
    return dialog.plan is not None


def test_two_folders_dropped_become_two_channels_end_to_end(
        screen, tmp_path, qtbot, monkeypatch):
    exp = _tree(tmp_path)
    monkeypatch.setattr(screen, "_run_organize", _apply_popup)
    assert screen.open_paths([str(exp / "DAPI"), str(exp / "GFP")])
    dialog = screen._organize_dialog
    assert [c.kind for c in dialog.columns] == ["channel", "mask",
                                                "channel", "mask"]
    dest = exp / "sorted_channels"
    qtbot.waitUntil(lambda: screen._folder == str(dest / "C01"), timeout=60000)
    assert sorted(os.listdir(dest / "C01")) == [
        "masks", "plate1_A01_T0001F001L01A01Z01C01.tif",
        "plate1_A02_T0001F001L01A01Z01C01.tif",
        "plate1_A03_T0001F001L01A01Z01C01.tif"]
    assert sorted(os.listdir(dest / "C02" / "masks"))[0] == \
        "plate1_A01_T0001F001L01A01Z01C02.tif"
    merged = sorted(n for n in os.listdir(dest / "merged") if n.endswith(".npy"))
    assert len(merged) == 3
    assert np.load(dest / "merged" / merged[0]).shape == (16, 16, 4)
    assert (dest / "channel_sorting_manifest.csv").is_file()
    assert not list((exp / "DAPI").glob("*.tif"))


def test_cancelling_the_popup_opens_the_folders_as_a_queue(screen, tmp_path):
    exp = _tree(tmp_path)
    assert screen.open_paths([str(exp / "DAPI"), str(exp / "GFP")])
    assert screen._organize_dialog is not None
    assert len(screen._image_files) == 6
    assert screen._field_folders is not None


def test_channel_like_subfolders_open_the_popup_prefilled(screen, tmp_path,
                                                          monkeypatch):
    exp = _tree(tmp_path)
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda *a: asked.append(a) or True)
    seen = []
    monkeypatch.setattr(screen, "_run_organize",
                        lambda dialog: seen.append(dialog) or False)
    assert not screen.open_paths([str(exp)])
    # No question: the popup opened with the source and two channel columns.
    assert asked == []
    assert seen[0]._source() == str(exp)
    assert len(seen[0]._channel_columns()) == 2
    assert len(seen[0].rows) == 3


def test_other_nested_folders_still_get_the_consolidation_question(
        screen, tmp_path, monkeypatch):
    src = tmp_path / "exp"
    _tif(src / "monday" / "x1.tif")
    _tif(src / "tuesday" / "y7.tif")
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda *a: asked.append(a) or False)
    screen.open_paths([str(src)])
    assert asked and "Consolidate" in asked[0][0]
    assert getattr(screen, "_organize_dialog", None) is None


def test_images_dropped_with_their_masks_folder_open_with_them(screen,
                                                                tmp_path):
    for name in ("a.tif", "b.tif"):
        _tif(tmp_path / "imgs" / name)
        _tif(tmp_path / "drawn_masks" / name, _mask())
    assert screen.open_paths([str(tmp_path / "imgs" / "a.tif"),
                              str(tmp_path / "imgs" / "b.tif"),
                              str(tmp_path / "drawn_masks")])
    assert screen._folder == str(tmp_path / "imgs")
    assert screen._masks_dir == str(tmp_path / "drawn_masks")
    assert screen._image_files == ["a.tif", "b.tif"]


def test_masks_under_other_names_are_copied_beside_their_images(
        screen, tmp_path, monkeypatch):
    a = _tif(tmp_path / "imgs" / "a.tif")
    _tif(tmp_path / "imgs" / "masks" / "b.tif", _mask(5))
    b = _tif(tmp_path / "imgs" / "b.tif")
    ma = _tif(tmp_path / "out" / "a_cp_masks.tif", _mask(2))
    mb = _tif(tmp_path / "out" / "b_cp_masks.tif", _mask(3))
    stray = _tif(tmp_path / "out" / "zz_cp_masks.tif", _mask(4))
    asked = []
    monkeypatch.setattr(screen, "_confirm", lambda *a: asked.append(a) or True)
    assert screen.open_paths([str(a), str(b), str(ma), str(mb), str(stray)])
    assert "1 image(s) already have a mask" in asked[0][1]
    copied = tifffile.imread(str(tmp_path / "imgs" / "masks" / "a.tif"))
    assert copied.max() == 2
    kept = tifffile.imread(str(tmp_path / "imgs" / "masks" / "b.tif"))
    assert kept.max() == 5
    assert screen._image_files == ["a.tif", "b.tif"]
    assert "zz_cp_masks" in screen._masks_console.text()


def test_a_png_mask_is_written_as_a_tiff(screen, tmp_path, monkeypatch):
    from PIL import Image

    a = _tif(tmp_path / "imgs" / "a.tif")
    b = _tif(tmp_path / "imgs" / "b.tif")
    png = tmp_path / "labels_masks" / "a_mask.png"
    png.parent.mkdir()
    Image.fromarray(_mask(7).astype(np.uint8)).save(png)
    monkeypatch.setattr(screen, "_confirm", lambda *a: True)
    assert screen.open_paths([str(a), str(b), str(png)])
    written = tifffile.imread(str(tmp_path / "imgs" / "masks" / "a.tif"))
    assert written.max() == 7


def test_a_sorted_channels_folder_opens_its_first_channel(screen, tmp_path):
    dest = tmp_path / "sorted_channels"
    _tif(dest / "C01" / "plate1_A01_T0001F001L01A01Z01C01.tif")
    _tif(dest / "C02" / "plate1_A01_T0001F001L01A01Z01C02.tif")
    assert screen.open_paths([str(dest)])
    assert screen._folder == str(dest / "C01")
    assert "sorted by spaCR" in screen._masks_console.text()


def test_merged_arrays_alone_open_nothing_and_say_why(screen, tmp_path):
    (tmp_path / "plate" / "merged").mkdir(parents=True)
    np.save(tmp_path / "plate" / "merged" / "f.npy", np.zeros((4, 4, 2)))
    assert not screen.open_paths([str(tmp_path / "plate")])
    assert "Measure" in screen._masks_console.text()
    assert not screen._image_files


def test_unrecognised_items_are_listed_in_the_console(screen, tmp_path):
    a = _tif(tmp_path / "a.tif")
    npy = tmp_path / "x.npy"
    np.save(npy, np.zeros(3))
    assert screen.open_paths([str(a), str(npy)])
    assert screen._image_files == ["a.tif"]
    assert "Not used from this drop" in screen._masks_console.text()
    assert "x.npy" in screen._masks_console.text()
    note = tmp_path / "n.txt"
    note.write_text("x")
    assert not screen.open_paths([str(note)])
    assert "n.txt" in screen._masks_console.text()


def test_the_organize_button_opens_the_popup_on_the_open_folder(
        screen, tmp_path, monkeypatch):
    _tif(tmp_path / "flat" / "a.tif")
    assert screen._open_folder(str(tmp_path / "flat"))
    seen = []
    monkeypatch.setattr(screen, "_run_organize",
                        lambda dialog: seen.append(dialog) or False)
    screen._btn_organize.click()
    assert seen[0]._source() == str(tmp_path / "flat")


def test_the_drop_handler_takes_what_the_screen_can_explain(tmp_path):
    exp = _tree(tmp_path)
    npy = tmp_path / "x.npy"
    np.save(npy, np.zeros(3))
    handler = MakeMasksDropHandler()
    assert handler.can_accept(exp)
    assert handler.can_accept(npy)
    (tmp_path / "empty").mkdir()
    assert not handler.can_accept(tmp_path / "empty")
