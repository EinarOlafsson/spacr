"""N690: Make Masks on transmission electron micrographs.

The decision recorded in features/new/690: drawing on TEM needs no screen of
its own. What it needs is here -- TEM files open in the queue and save at
full size with their pixel size, a one-step contrast preset, a reader
missing its optional package says how to install it, a field of thousands
of pixels redraws at the widget's resolution rather than its own, and the
TEM test data sits on the "Load test data" menu. No network: the download is
replaced through the ``ask`` seam.
"""
from __future__ import annotations

import builtins
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import io as sio
from spacr.qt import make_masks_demo as demo
from spacr.qt import mask_engine as engine
from spacr.qt.hf_download import DownloadResult
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen, _MaskCanvas
from tests.tem_files import write_dm


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    """A Make Masks screen whose warnings are collected, not shown."""
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    widget.warnings = []
    widget._warn = lambda title, text: widget.warnings.append((title, text))
    return widget


def _field(shape=(96, 128)) -> np.ndarray:
    """A TEM-like uint16 field: bright cytoplasm, dark round organelles."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    image = np.full(shape, 40000, np.uint16)
    image[(yy - 30) ** 2 + (xx - 40) ** 2 < 15 ** 2] = 9000
    return image


def test_a_dm4_field_opens_and_saves_with_its_pixel_size(screen, tmp_path, qtbot):
    write_dm(tmp_path / "cell.dm4", _field(), scale=1.6, unit="nm")
    screen._open_folder(str(tmp_path))
    qtbot.waitUntil(lambda: screen._canvas.image is not None, timeout=10000)
    assert screen._canvas.image.shape == (96, 128)
    screen._canvas.mask[20:40, 30:50] = 255
    screen._on_save()
    saved = tmp_path / "masks" / "cell.tif"
    with tifffile.TiffFile(saved) as tif:
        assert tif.pages[0].shape == (96, 128)
    assert sio.tiff_pixel_size_nm(saved) == pytest.approx(1.6)


def test_the_contrast_preset_sets_the_display_in_one_step(screen):
    box = screen._contrast_preset
    assert [box.itemData(i) for i in range(box.count())] == [
        "custom", "tem", "tem_inverted"]
    box.setCurrentIndex(box.findData("tem"))
    assert screen._norm_lo.value() == pytest.approx(0.5)
    assert screen._norm_hi.value() == pytest.approx(99.5)
    assert screen._enh_clahe.isChecked() and screen._btn_apply.isChecked()
    assert screen._enh_clahe_tile.value() == 128
    assert not screen._invert_display.isChecked()
    assert "Electron microscopy" in screen._status_label.text()
    box.setCurrentIndex(box.findData("tem_inverted"))
    assert screen._invert_display.isChecked()
    screen._norm_lo.setValue(3.0)
    box.setCurrentIndex(box.findData("custom"))
    assert screen._norm_lo.value() == pytest.approx(3.0)
    assert screen._invert_display.isChecked()


def test_a_missing_mrc_reader_says_how_to_install_it(screen, tmp_path, qtbot,
                                                      monkeypatch):
    (tmp_path / "section.mrc").write_bytes(b"\x00" * 1024)
    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "mrcfile":
            raise ImportError("No module named 'mrcfile'")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "mrcfile", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)
    screen._open_folder(str(tmp_path))
    qtbot.waitUntil(lambda: bool(screen.warnings), timeout=10000)
    title, text = screen.warnings[-1]
    assert 'pip install "spacr[tem]"' in text and "mrcfile" in text
    assert ".mrc" in text


def test_other_load_failures_keep_their_own_message(screen):
    screen._handle_load_failure(ValueError("broken field"))
    assert screen.warnings[-1] == ("Load failed", "broken field")


def test_a_large_field_is_composed_at_the_widget_resolution(qtbot, qt_theme_applied,
                                                            monkeypatch):
    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(600, 400)
    seen = []
    real_overlay = engine.overlay_mask

    def spy(image, mask, alpha=0.5):
        seen.append(image.shape)
        return real_overlay(image, mask, alpha=alpha)

    monkeypatch.setattr(mm.engine, "overlay_mask", spy)
    monkeypatch.setattr(mm, "device_ratio", lambda _w: 1.0)
    image = np.tile(_field((256, 256)), (12, 12))
    canvas.set_image_and_mask(image, np.zeros(image.shape, np.uint8))
    step = engine._display_step(3072, 3072, 600, 400)
    assert step == 5
    assert seen[-1] == image[::step, ::step].shape
    cached = canvas._bounds_cache
    assert cached[0] is image
    canvas.refresh()
    assert canvas._bounds_cache is cached
    point = canvas._canvas_to_image(300, 200)
    assert point is not None
    assert abs(point[0] - 1536) <= step and abs(point[1] - 1536) <= step
    canvas.norm_hi = 99.0
    canvas.refresh()
    assert canvas._bounds_cache[1] == (canvas.norm_lo, 99.0)
    small = _field()
    canvas.set_image_and_mask(small, np.zeros(small.shape, np.uint8))
    assert canvas._bounds_cache is None
    assert seen[-1] == small.shape


def test_an_empty_field_draws_nothing(qtbot, qt_theme_applied):
    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.image = np.zeros((0, 0), np.uint16)
    canvas.mask = np.zeros((0, 0), np.uint8)
    canvas.refresh()
    assert canvas.pixmap() is None or canvas.pixmap().isNull()


def _write_tem_set(folder: Path) -> None:
    """The TEM test data as its archive unpacks, written out of name order."""
    (folder / "ground_truth_masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for stem in ("tem_02_b", "tem_01_a"):
        tifffile.imwrite(folder / f"{stem}.tif", _field().astype(np.uint8))
        tifffile.imwrite(folder / "ground_truth_masks" / f"{stem}.tif",
                         np.ones((96, 128), np.uint16))
        rows.append({"name": stem, "image": f"{stem}.tif",
                     "ground_truth_mask": f"ground_truth_masks/{stem}.tif"})
    with (folder / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_the_menu_downloads_the_tem_test_data_once_and_opens_it(screen, tmp_path,
                                                                 qtbot, monkeypatch):
    calls = []

    def fake_download(_screen, dest, on_done):
        calls.append(dest)
        assert not screen._btn_test_data.isEnabled()
        _write_tem_set(Path(dest))
        on_done(DownloadResult(dataset_path=Path(dest),
                               settings_path=Path(dest)), "")

    folder = tmp_path / "make_masks_tem"
    monkeypatch.setattr(demo, "tem_example_folder", lambda: folder)
    monkeypatch.setattr(demo, "download_tem_example", fake_download)
    action = screen._test_data_actions["tem"]
    assert action.objectName() == "MakeMasksTemTestData"
    action.trigger()
    qtbot.waitUntil(lambda: screen._canvas.image is not None, timeout=10000)
    assert calls == [folder]
    assert screen._image_files[screen._current_index] == "tem_01_a.tif"
    assert "ground_truth_masks" not in screen._image_files
    assert screen._btn_test_data.isEnabled()
    assert demo.load_the_tem_test_data(screen) is True
    assert calls == [folder]


def test_a_failed_tem_download_says_why(screen, tmp_path):
    def failing(_screen, _dest, on_done):
        on_done(None, "no route to host")

    assert demo.load_the_tem_test_data(screen, ask=failing,
                                       folder=tmp_path / "x") is False
    assert "no route to host" in screen.warnings[-1][1]


def test_the_tem_worker_fetches_its_own_archive(tmp_path):
    assert demo._TemTarWorker.repo == "einarolafsson/spacr-example-tem"
    assert demo._TemTarWorker.archive == demo.TEM_EXAMPLE_ARCHIVE
    assert demo.tem_example_folder().name == "make_masks_tem"


def test_download_tem_example_uses_the_shared_progress_dialog(monkeypatch, tmp_path):
    seen = {}

    def fake(parent, dest, on_done, *, worker_factory, title):
        seen.update(dest=dest, worker=worker_factory, title=title)

    monkeypatch.setattr(demo, "download_toxo_mito_demo", fake)
    demo.download_tem_example(None, tmp_path, lambda *_a: None)
    assert seen["worker"] is demo._TemTarWorker
    assert seen["dest"] == tmp_path and "TEM" in seen["title"]
