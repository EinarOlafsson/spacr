"""Make Masks' Model box, its downloads, and results that arrive late.

Pins the model-zoo rows and the download around them, and what the screen
does with work that finishes after the user has moved on:

* a zoo model listed twice, or a backend model already in the box, is one
  row; a chosen row that disappears on refresh selects the first row and
  says so with a change signal; a pending row with nowhere to go back to
  falls back to the first row;
* a download while another runs, an unverified model, a declined question
  and a folder that cannot be made each stop before downloading, with the
  reason on screen; a download with no size keeps the bar busy; a cancelled
  one says nothing; a finished file the box does not list is added and
  selected;
* a detection result for a request no longer current is dropped, and one
  that cannot be applied is reported;
* choosing an uninstalled backend while the box is already on it falls back
  to Otsu and offers the install;
* a tile count with no time estimate shows a percentage;
* closing the screen cancels a download on its way;
* Skip in a session writes its record and moves on, or says why it could not;
* the Filter button reports a list it cannot read or run;
* the U-Net checkpoint chooser keeps the path chosen and ignores a cancel;
* an install's streamed output is kept out of the corner;
* Send to Mask builds the Mask module when the window has none yet.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog, QMainWindow

from spacr import model_zoo
from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm

N = 64


def _write_folder(root: Path, names=("a.tif", "b.tif")) -> Path:
    folder = root / "fields"
    (folder / "masks").mkdir(parents=True)
    image = np.full((N, N), 100, dtype=np.uint16)
    image[10:30, 10:30] = 30000
    mask = np.zeros((N, N), dtype=np.uint16)
    mask[10:30, 10:30] = 1
    for name in names:
        imageio.imwrite(folder / name, image)
        imageio.imwrite(folder / "masks" / name, mask)
    return folder


def _entry(tmp_path: Path, key="zoo_one", sha="0" * 64):
    return model_zoo.ModelEntry(
        key=key, name=key + ".cp_model", kind="cellpose", source="remote",
        uri=(tmp_path / "nowhere").as_uri(), sha256=sha, size_bytes=2048)


@pytest.fixture
def listing(monkeypatch):
    rows = {"cellpose": [], "dino": [], "prefixed": []}
    monkeypatch.setattr(mm, "_zoo_cellpose_models", lambda: list(rows["cellpose"]))
    monkeypatch.setattr(mm, "_zoo_cellpose_dino_models", lambda: list(rows["dino"]))
    monkeypatch.setattr(mm, "_zoo_prefixed_models", lambda: list(rows["prefixed"]))
    return rows


@pytest.fixture
def screen(qtbot, qt_theme_applied, listing):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    yield made
    made._magnifier.close()
    made.close_folded()


@pytest.fixture
def opened(qtbot, qt_theme_applied, tmp_path, listing):
    folder = _write_folder(tmp_path)
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    assert made._open_folder(str(folder))
    yield made, folder
    made._magnifier.close()
    made.close_folded()


def _rows_with(combo, value):
    return [i for i in range(combo.count()) if combo.itemData(i) == value]


# ---------------------------------------------------------------------------
# The Model box
# ---------------------------------------------------------------------------

def test_a_model_listed_twice_is_one_row(screen, listing, tmp_path):
    weights = tmp_path / "zoo_one.cp_model"
    weights.write_bytes(b"w")
    entry = _entry(tmp_path)
    listing["cellpose"] = [("zoo_one", str(weights), entry),
                           ("zoo_one", str(weights), entry)]
    listing["dino"] = [("cellpose_dino:/m.pt", "Cellpose-DINO · m.pt"),
                       ("cellpose_dino:/m.pt", "Cellpose-DINO · m.pt")]
    screen._fill_zoo_models()
    combo = screen._cp_model
    assert len(_rows_with(combo, str(weights))) == 1
    assert len(_rows_with(combo, "cellpose_dino:/m.pt")) == 1


def test_a_chosen_row_that_disappears_selects_the_first(screen, listing,
                                                        tmp_path):
    weights = tmp_path / "gone.cp_model"
    weights.write_bytes(b"w")
    listing["cellpose"] = [("gone", str(weights), _entry(tmp_path, "gone"))]
    screen._fill_zoo_models()
    combo = screen._cp_model
    combo.setCurrentIndex(_rows_with(combo, str(weights))[0])
    changed = []
    combo.currentIndexChanged.connect(changed.append)
    listing["cellpose"] = []
    screen._fill_zoo_models()
    assert combo.currentIndex() == 0
    assert changed == [0]
    assert _rows_with(combo, str(weights)) == []


def test_a_pending_row_with_nowhere_to_go_back_to_falls_to_the_first(
        screen, listing, tmp_path):
    listing["cellpose"] = [("later", None, _entry(tmp_path, "later"))]
    screen._fill_zoo_models()
    combo = screen._cp_model
    pending = next(i for i in range(combo.count())
                   if combo.itemData(i, mm._ZOO_PENDING_ROLE) is not None)
    screen._cp_last_model = combo.count() + 5
    combo.setCurrentIndex(pending)
    assert combo.currentIndex() == 0
    started = []
    screen.download_zoo_model = started.append
    screen._on_model_activated(0)
    assert started == []


# ---------------------------------------------------------------------------
# Downloads that do not start
# ---------------------------------------------------------------------------

def test_a_second_download_waits_for_the_first(screen, tmp_path):
    screen._cp_download = SimpleNamespace(is_running=lambda: True)
    assert screen.download_zoo_model(_entry(tmp_path)) is False
    assert screen._status_label.text() == (
        "A model is already downloading; wait for it to finish.")
    screen._cp_download = None


def test_an_unverified_model_is_named_as_such_and_a_no_stops_it(
        screen, tmp_path, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    monkeypatch.setattr(model_zoo_picker, "remembered_model_dir",
                        lambda: str(tmp_path / "models"))
    asked = []
    monkeypatch.setattr(screen, "_confirm",
                        lambda title, text: asked.append(text) or False)
    assert screen.download_zoo_model(_entry(tmp_path, sha="")) is False
    assert "publishes no checksum" in asked[0]
    assert screen._cp_download is None
    assert not (tmp_path / "models").exists()


def test_a_download_folder_that_cannot_be_made_says_so(
        screen, tmp_path, monkeypatch):
    from spacr.qt.widgets import model_zoo_picker

    blocker = tmp_path / "a_file"
    blocker.write_text("in the way")
    monkeypatch.setattr(model_zoo_picker, "remembered_model_dir",
                        lambda: str(blocker / "models"))
    monkeypatch.setattr(screen, "_confirm", lambda *a: True)
    assert screen.download_zoo_model(_entry(tmp_path)) is False
    assert screen._status_label.text().startswith("Download failed:")
    assert screen._cp_download is None


# ---------------------------------------------------------------------------
# Downloads that end
# ---------------------------------------------------------------------------

def test_a_download_with_no_size_keeps_the_bar_busy(screen):
    bar = screen._cp_download_bar
    bar.setRange(0, 0)
    screen._on_model_download_progress(4096, 0)
    assert (bar.minimum(), bar.maximum()) == (0, 0)
    screen._on_model_download_progress(512, 1024)
    assert bar.maximum() == 1000 and bar.value() == 500


def test_a_cancelled_download_says_nothing(screen):
    screen._status_label.setText("before")
    screen._cp_download = SimpleNamespace(entry=SimpleNamespace(key="m"))
    screen._on_model_downloaded(False, "cancelled")
    assert screen._status_label.text() == "before"
    assert screen._cp_download is None
    assert screen._cp_download_bar.isHidden()


def test_a_finished_file_the_box_does_not_list_is_added_and_chosen(
        screen, tmp_path):
    landed = tmp_path / "private.cp_model"
    landed.write_bytes(b"w")
    screen._cp_download = None
    screen._on_model_downloaded(True, str(landed))
    combo = screen._cp_model
    assert combo.currentData() == str(landed)
    assert combo.currentText() == "private.cp_model"
    assert screen._status_label.text() == (
        f"{landed} is downloaded and selected.")


def test_closing_the_screen_cancels_a_download_on_its_way(qtbot,
                                                          qt_theme_applied,
                                                          listing):
    made = mm.MakeMasksScreen()
    qtbot.addWidget(made)
    cancelled = []
    made._cp_download = SimpleNamespace(cancel=lambda: cancelled.append(1),
                                        is_running=lambda: True)
    made.close()
    assert cancelled == [1]
    assert made._cp_download is None
    made.close_folded()


# ---------------------------------------------------------------------------
# Late results
# ---------------------------------------------------------------------------

def test_a_detection_for_an_old_request_is_dropped(opened):
    screen, _folder = opened
    before = screen._canvas.mask.copy()
    screen._detection_request = {"token": "current"}
    screen._take_detection(({"token": "old"}, np.ones((N, N)), None))
    np.testing.assert_array_equal(screen._canvas.mask, before)
    assert screen._detection_request == {"token": "current"}


def test_a_detection_that_cannot_be_applied_is_reported(opened, monkeypatch):
    screen, _folder = opened
    canvas = screen._canvas
    request = {"token": screen._load_token, "image_reference": canvas.image,
               "image": canvas.image.copy(), "mask": canvas.mask.copy(),
               "mode": "cellpose", "details": {}}
    screen._detection_request = request

    def broken(*args):
        raise RuntimeError("labels out of range")

    monkeypatch.setattr(screen, "_apply_detection", broken)
    screen._take_detection((request, np.zeros((N, N), np.int32), None))
    assert screen._detection_request is None
    assert screen._btn_cellpose.isEnabled()
    assert screen._status_label.text() == (
        "Object detection failed: labels out of range")


def test_an_uninstalled_backend_already_chosen_falls_back_to_otsu(
        screen, monkeypatch):
    monkeypatch.setattr(mm, "_backend_ready", lambda mode: False)
    offered = []
    monkeypatch.setattr(screen, "_offer_backend_install", offered.append)
    box = screen._mag_mode
    index = box.findData("dinocell")
    assert index >= 0
    screen._magnifier.mode = "dinocell"
    screen._on_magnifier_mode_activated(index)
    assert box.currentData() == "otsu"
    assert offered == ["dinocell"]


def test_a_tile_count_with_no_estimate_shows_a_percentage(screen,
                                                          monkeypatch):
    monkeypatch.setattr(screen._magnifier, "image_progress",
                        lambda: (3, 9, None))
    screen._mag_progress.setFormat("something else")
    screen._tick_magnifier_eta()
    assert screen._mag_progress.format() == "%p%"
    assert screen._mag_progress.maximum() == 9
    assert screen._mag_progress.value() == 2


# ---------------------------------------------------------------------------
# Skip, Filter, the U-Net chooser, the console, Send to Mask
# ---------------------------------------------------------------------------

def test_skip_in_a_session_records_it_and_moves_on(opened, monkeypatch):
    from spacr import curation_queue

    screen, folder = opened
    marked = []
    monkeypatch.setattr(curation_queue, "mark_state",
                        lambda where, stem, state: marked.append((stem, state)))
    screen._queue = SimpleNamespace(folder=str(folder))
    screen._on_skip()
    assert marked == [("a", "skip")]
    assert screen._current_index == 1
    assert "a.tif skipped" in screen._status_label.text()
    assert "now on b.tif" in screen._status_label.text()


def test_a_skip_that_cannot_be_recorded_stays_on_the_field(opened,
                                                           monkeypatch):
    from spacr import curation_queue

    screen, folder = opened

    def refuse(*args):
        raise OSError("read-only")

    monkeypatch.setattr(curation_queue, "mark_state", refuse)
    screen._queue = SimpleNamespace(folder=str(folder))
    screen._on_skip()
    assert screen._current_index == 0
    assert screen._status_label.text() == "Skip failed: read-only"


def test_the_filter_reports_a_list_it_cannot_read(opened):
    screen, _folder = opened
    screen._filter_list.add_filter("area", 50, 10, notify=False)
    assert screen.apply_object_filter() == 0
    assert "has a minimum (50) above its maximum (10)" in (
        screen._status_label.text())


def test_the_filter_reports_a_list_it_cannot_run(opened, monkeypatch):
    screen, _folder = opened
    screen._filter_list.add_filter("area", 10, None, notify=False)

    def refuse(*args, **kwargs):
        raise ValueError("cannot measure area here")

    monkeypatch.setattr(engine, "apply_filters", refuse)
    before = screen._canvas.mask.copy()
    assert screen.apply_object_filter() == 0
    assert screen._status_label.text() == "cannot measure area here"
    np.testing.assert_array_equal(screen._canvas.mask, before)


def test_the_unet_chooser_keeps_the_path_and_ignores_a_cancel(screen,
                                                              monkeypatch):
    answers = iter([("/models/unet.pt", "Torch checkpoints (*.pt *.pth)"),
                    ("", "")])
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: next(answers)))
    screen._on_pick_unet_model()
    assert screen._unet_path_edit.text() == "/models/unet.pt"
    screen._on_pick_unet_model()
    assert screen._unet_path_edit.text() == "/models/unet.pt"


def test_an_installs_streamed_output_stays_out_of_the_corner(screen):
    screen._status_label.setText("Installing…")
    screen._report("Collecting torch", "stream")
    assert screen._status_label.text() == "Installing…"
    screen._report("Installed.", "info")
    assert screen._status_label.text() == "Installed."


class _AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._screens = {}
        self.built = []

    def rebuild_app_screen(self, key, settings):
        self.built.append((key, dict(settings)))


def test_send_to_mask_builds_the_mask_module_when_there_is_none(
        qtbot, screen):
    window = _AppWindow()
    qtbot.addWidget(window)
    window.setCentralWidget(screen)
    screen._send_chain_to_mask()
    assert [key for key, _settings in window.built] == ["mask"]
    assert window.built[0][1] == screen.mask_settings()
    assert "Enhancement chain written to the Mask settings" in (
        screen._status_label.text())
    screen.setParent(None)
