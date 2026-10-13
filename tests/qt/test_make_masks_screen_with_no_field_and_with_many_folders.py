"""Make Masks with no field open, and with a queue dropped from many folders.

Pins what the screen does at the two ends of what it can be holding:

* with nothing open, every button that needs a field is a quiet no-op --
  curating, measuring, growing, shrinking, removing an object, opening the
  histogram, the levels or the comparison -- and the one that can explain
  itself says a field is needed;
* a verdict that cannot be written says so and leaves the buttons showing
  what is really on disk; a verdict that cannot be read shows none;
* a drop of files from several folders is one queue: duplicates are listed
  once, non-images are ignored, a drop with no image says so, the folder
  mask button refuses a queue spanning folders, and a recrop is queued and
  retired inside that queue at the field's own folder;
* a curation session with nothing left, or whose folder cannot be opened,
  is not opened; a missing terminal handover is simply no handover;
* the contribution dialog leaves Cellpose bundles out and opens on screen.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

N = 96


def _field_and_mask():
    image = np.full((N, N), 100, dtype=np.uint16)
    mask = np.zeros((N, N), dtype=np.uint16)
    image[20:40, 20:40] = 30000
    mask[20:40, 20:40] = 1
    image[60:80, 60:80] = 30000
    mask[60:80, 60:80] = 2
    return image, mask


def _folder(root: Path, name: str, fields=("a.tif", "b.tif")) -> Path:
    folder = root / name
    (folder / "masks").mkdir(parents=True)
    image, mask = _field_and_mask()
    for field in fields:
        imageio.imwrite(folder / field, image)
        imageio.imwrite(folder / "masks" / field, mask)
    return folder


@pytest.fixture
def blank(qtbot, qt_theme_applied):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made._magnifier.close()
    made.close_folded()


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path):
    folder = _folder(tmp_path, "one", ("a.tif", "b.tif", "c.tif"))
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    yield made, folder
    made._magnifier.close()
    made.close_folded()


# ---------------------------------------------------------------------------
# Nothing open
# ---------------------------------------------------------------------------

def test_with_no_field_every_field_button_is_a_quiet_no_op(blank):
    screen = blank
    screen._status_label.setText("untouched")
    assert screen._on_curate(True) is None
    assert not screen._btn_keep.isChecked()
    assert not screen._btn_discard.isChecked()
    screen._refresh_curation_buttons()
    assert screen._current_image_paths() == []
    assert screen._detector_image() is None
    assert screen._objects_now() == 0
    screen._on_dilate()
    screen._on_shrink()
    assert screen._remove_magnifier_object(3, 3) == 0
    screen._on_levels()
    screen._on_invert_display(True)
    screen._on_compare_enhanced()
    assert screen._levels_dialog is None
    assert getattr(screen, "_compare_dialog", None) is None
    assert screen._contribution_sources() == (None, None)
    assert screen._status_label.text() == "untouched"


def test_the_histogram_asks_for_a_field_first(blank):
    blank._on_show_otsu_histogram()
    assert blank._status_label.text() == (
        "Open a field first — a histogram needs an image.")
    assert blank._otsu_histogram_dialog is None


def test_a_field_index_past_the_list_has_no_paths(opened):
    screen, _folder_path = opened
    screen._current_index = 17
    assert screen._curation_paths() == (None, None)
    screen._current_index = None
    assert screen._curation_paths() == (None, None)
    screen._current_index = 1
    image, mask = screen._curation_paths()
    assert image.endswith("b.tif")


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def test_the_open_field_is_what_infer_reads(opened):
    screen, folder = opened
    assert screen._current_image_paths() == [str(folder / "a.tif")]


def test_a_verdict_that_cannot_be_written_says_so(opened, monkeypatch):
    screen, folder = opened

    def refuse(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(engine, "record_curation", refuse)
    assert screen._on_curate(True) is None
    assert "Verdict not recorded" in screen._status_label.text()
    assert "disk full" in screen._status_label.text()
    assert not screen._btn_keep.isChecked()
    assert screen._current_index == 0


def test_a_verdict_that_cannot_be_read_shows_none(opened, monkeypatch):
    screen, _folder_path = opened
    screen._btn_keep.setChecked(True)

    def unreadable(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(engine, "curation_verdict", unreadable)
    screen._refresh_curation_buttons()
    assert not screen._btn_keep.isChecked()
    assert not screen._btn_discard.isChecked()


# ---------------------------------------------------------------------------
# A drop from many folders
# ---------------------------------------------------------------------------

@pytest.fixture
def spread(qtbot, qt_theme_applied, tmp_path):
    first = _folder(tmp_path, "first", ("a.tif",))
    second = _folder(tmp_path, "second", ("b.tif",))
    (tmp_path / "notes.txt").write_text("not an image")
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made.open_paths([first / "a.tif", tmp_path / "notes.txt",
                            second / "b.tif", first / "a.tif"])
    yield made, first, second
    made._magnifier.close()
    made.close_folded()


def test_a_drop_from_two_folders_is_one_queue_listed_once(spread):
    screen, first, second = spread
    assert screen._image_files == ["a.tif", "b.tif"]
    assert screen._field_folders == [str(first), str(second)]
    assert "2 images from 2 folders" in screen._src_label.text()


def test_a_drop_with_no_image_says_so(blank, tmp_path):
    (tmp_path / "notes.txt").write_text("no pixels here")
    assert blank.open_paths([tmp_path / "notes.txt",
                             tmp_path / "missing.tif"]) is False
    assert "None of the dropped items" in blank._status_label.text()
    assert not blank._image_files


def test_a_drop_whose_folder_will_not_open_is_not_a_queue(
        blank, tmp_path, monkeypatch):
    first = _folder(tmp_path, "first", ("a.tif",))
    monkeypatch.setattr(blank, "_open_folder", lambda *a, **k: False)
    assert blank.open_paths([first / "a.tif"]) is False


def test_the_folder_mask_button_refuses_a_queue_spanning_folders(spread):
    screen, _first, _second = spread
    assert screen.mask_whole_folder() is False
    assert "dropped from several" in screen._status_label.text()


def test_a_recrop_is_queued_and_retired_at_its_own_folder(spread):
    screen, first, _second = spread
    written = screen.recrop(10, 10, 50, 50)
    assert written is not None
    assert screen._image_files == ["a.tif", written, "b.tif"]
    assert screen._field_folders[:2] == [str(first), str(first)]
    assert (first / written).is_file()

    assert screen.finish_recrop() is True
    assert screen._image_files == [written, "b.tif"]
    assert len(screen._field_folders) == 2
    assert (first / engine.RECROP_ARCHIVE_DIRNAME / "a.tif").is_file()


def test_a_paired_secondary_field_cannot_be_recropped(opened):
    screen, folder = opened
    screen._canvas.preserve_ids = True
    assert screen.recrop(10, 10, 50, 50) is None
    assert "Recrop requires a matching crop" in screen._status_label.text()
    assert screen._image_files == ["a.tif", "b.tif", "c.tif"]


# ---------------------------------------------------------------------------
# Curation sessions
# ---------------------------------------------------------------------------

def _queue(folder: Path, names):
    from spacr.curation_queue import LAYOUT_NESTED

    layout = SimpleNamespace(kind=LAYOUT_NESTED, images_dir=folder,
                             folder=folder)
    items = [SimpleNamespace(image=folder / name, bundle=None)
             for name in names]
    return SimpleNamespace(layout=layout, items=items, folder=folder)


def test_a_session_with_nothing_left_is_not_opened(blank, tmp_path):
    folder = _folder(tmp_path, "done", ("a.tif",))
    assert blank.open_queue(_queue(folder, [])) is False
    assert blank._queue is None
    assert not blank._image_files


def test_a_session_whose_folder_will_not_open_is_not_opened(
        blank, tmp_path, monkeypatch):
    folder = _folder(tmp_path, "todo", ("a.tif",))
    monkeypatch.setattr(blank, "_open_folder", lambda *a, **k: False)
    assert blank.open_queue(_queue(folder, ["a.tif"])) is False
    assert blank._queue is None


def test_no_terminal_handover_module_is_no_handover(blank, monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.cli_make_masks", None)
    assert blank._take_any_terminal_queue() is False
    assert blank._queue is None


# ---------------------------------------------------------------------------
# Contributions
# ---------------------------------------------------------------------------

def test_a_contribution_leaves_cellpose_bundles_out(opened):
    screen, folder = opened
    screen._image_files.append("d_seg.npy")
    current, curated = screen._contribution_sources()
    assert current["name"] == "a.tif"
    names = [item["name"] for item in curated]
    assert "d_seg.npy" not in names
    assert names == ["a.tif", "b.tif", "c.tif"]


def test_the_contribution_dialog_opens_on_screen(opened):
    screen, _folder_path = opened
    dialog = screen.contribute_images_and_masks(show=True,
                                                upload=lambda *a: "")
    try:
        assert dialog.isVisible()
        assert screen._contribute_dialog is dialog
    finally:
        dialog.close()
