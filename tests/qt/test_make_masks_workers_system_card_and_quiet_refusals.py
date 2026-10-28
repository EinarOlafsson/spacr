"""Make Masks: the canvas workers, the System card, and the screen's quiet refusals.

* The Wand and the object readout work off the GUI thread. A click reuses
  the worker it already has, a worker that was closed gives the cursor
  back, a flood that fails says why, a worker that answers after its canvas
  is gone does nothing, and a readout lookup is asked for once per state and
  dropped when the mask moved on.
* The System card beside the console: one bar per logical core on demand,
  no sample while one is out, stale and empty samples are ignored.
* The screen declines, with a status line and nothing written, what it
  cannot do: no folder for the vvvv export, Hide judged fields with no
  folder or every field judged, Open a file that is cancelled, Measure 3D
  with no volume or while busy, a 3D measurement for an image no longer open.

No GPU and no network.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import tifffile
from PySide6.QtCore import QPointF, Qt
from PySide6.QtWidgets import QFileDialog

from spacr.qt import mask_engine as engine
from spacr.qt.screens import make_masks as mm
from spacr.qt.screens.make_masks import MakeMasksScreen


# ---------------------------------------------------------------------------
# The Wand worker
# ---------------------------------------------------------------------------

@pytest.fixture
def wand_canvas(qtbot, qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    image = np.full((40, 40), 100, np.uint16)
    image[10:20, 10:20] = 5000
    canvas.set_image_and_mask(image, np.zeros(image.shape, np.uint16))
    canvas.wand_threaded = True
    canvas.wand_relative = False
    canvas.wand_tolerance = 50.0
    canvas.wand_max_pixels = 10_000
    yield canvas
    canvas.close_wand()


def test_a_second_threaded_wand_click_reuses_the_worker(qtbot, wand_canvas):
    canvas = wand_canvas
    canvas._wand_edit_at((12, 12), action="add")
    worker = canvas._wand_worker
    assert worker is not None
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=20000)
    first = int((canvas.mask > 0).sum())
    assert first == 100
    canvas._wand_edit_at((2, 2), action="add")
    assert canvas._wand_worker is worker
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=20000)
    assert int((canvas.mask > 0).sum()) > first


def test_a_closed_wand_worker_gives_the_cursor_back(qtbot, wand_canvas):
    canvas = wand_canvas
    canvas.setCursor(Qt.CrossCursor)
    canvas._wand_edit_at((12, 12), action="add")
    worker = canvas._wand_worker
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=20000)
    assert worker.close(timeout=2)
    before = canvas.mask.copy()
    canvas._wand_edit_at((2, 2), action="add")
    assert not canvas._wand_busy()
    assert canvas.cursor().shape() == Qt.CrossCursor
    np.testing.assert_array_equal(canvas.mask, before)


def test_a_wand_flood_that_fails_says_why_and_changes_nothing(qtbot, wand_canvas, monkeypatch):
    canvas = wand_canvas

    def broken(_request):
        raise MemoryError("no room for the flood")

    monkeypatch.setattr(mm, "_wand_for", broken)
    said = []
    canvas.status.connect(said.append)
    canvas.setCursor(Qt.CrossCursor)
    canvas._wand_edit_at((12, 12), action="add")
    qtbot.waitUntil(lambda: not canvas._wand_busy(), timeout=20000)
    assert said[-1] == "The Wand could not fill that region: no room for the flood"
    assert not canvas.mask.any()
    assert canvas.cursor().shape() == Qt.CrossCursor


@pytest.mark.parametrize("deliver", ["_wand_done", "_lookup_done"])
def test_a_worker_answering_a_deleted_canvas_does_nothing(qtbot, qt_theme_applied, deliver):
    from shiboken6 import delete, isValid

    canvas = mm._MaskCanvas()
    done = getattr(canvas, deliver)
    delete(canvas)
    assert not isValid(canvas)
    done(object(), None, None)


# ---------------------------------------------------------------------------
# The readout's object lookup
# ---------------------------------------------------------------------------

@pytest.fixture
def lookup_canvas(qtbot, qt_theme_applied, monkeypatch):
    started, release = threading.Event(), threading.Event()
    work = mm._lookup_for

    def delayed(request):
        started.set()
        assert release.wait(10)
        return work(request)

    monkeypatch.setattr(mm, "_LOOKUP_THREAD_PIXELS", 1)
    monkeypatch.setattr(mm, "_lookup_for", delayed)
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    image = np.ones((20, 20), dtype=np.uint16)
    mask = np.zeros_like(image)
    mask[2:5, 2:5] = 3
    canvas.set_image_and_mask(image, mask)
    canvas.started, canvas.release, canvas.work = started, release, work
    yield canvas
    release.set()
    canvas.close_lookup()


def test_the_same_mask_is_numbered_once_however_often_it_is_asked(qtbot, lookup_canvas):
    canvas = lookup_canvas
    assert canvas._object_lookup() is None
    asked = canvas._lookup_asked
    submitted = []
    real = canvas._lookup_worker.submit
    canvas._lookup_worker.submit = lambda request, **kw: submitted.append(request) or real(
        request, **kw)
    assert canvas._object_lookup() is None
    assert canvas._lookup_asked is asked and submitted == []
    canvas.release.set()
    qtbot.waitUntil(lambda: canvas._lookup is not None, timeout=20000)
    assert canvas._lookup_asked is None


def test_a_closed_lookup_worker_forgets_the_request(lookup_canvas):
    canvas = lookup_canvas
    canvas._object_lookup()
    worker = canvas._lookup_worker
    canvas.release.set()
    assert worker.close(timeout=5)
    canvas._lookup_asked = None
    canvas._ask_for_lookup()
    assert canvas._lookup_asked is None and canvas._lookup is None


def test_a_lookup_that_failed_or_is_for_an_old_mask_is_not_kept(lookup_canvas):
    canvas = lookup_canvas
    canvas._object_lookup()
    asked = canvas._lookup_asked
    canvas._take_lookup((asked, None, RuntimeError("cannot number")))
    assert canvas._lookup is None and canvas._lookup_asked is None
    canvas._lookup_asked = asked
    answer = canvas.work(asked)
    canvas.mask = canvas.mask.copy()
    canvas.mask[10:12, 10:12] = 9
    canvas._take_lookup((asked, answer, None))
    assert canvas._lookup is None and canvas._lookup_asked is None


def test_a_lookup_that_lands_under_a_resting_pointer_updates_the_readout(lookup_canvas,
                                                                         monkeypatch):
    canvas = lookup_canvas
    canvas._object_lookup()
    asked = canvas._lookup_asked
    answer = canvas.work(asked)
    canvas._readout_pos = QPointF(3.0, 3.0)
    read = []
    monkeypatch.setattr(canvas, "update_readout", lambda pos=None, **kw: read.append(pos))
    canvas._take_lookup((asked, answer, None))
    assert canvas._lookup is answer
    assert read == [canvas._readout_pos]


def test_an_empty_field_is_composed_again_instead_of_rescaled(qtbot, qt_theme_applied):
    canvas = mm._MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 400)
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    canvas.set_image_and_mask(image, np.zeros_like(image))
    canvas.refresh()
    assert canvas._composed is not None
    assert canvas._rescale_composed()
    canvas.image = np.zeros((0, 64), np.uint16)
    assert not canvas._rescale_composed()


# ---------------------------------------------------------------------------
# The System card
# ---------------------------------------------------------------------------

def _fake_psutil(monkeypatch, count):
    fake = types.ModuleType("psutil")
    if isinstance(count, Exception):
        def cpu_count(logical=True):
            raise count
    else:
        def cpu_count(logical=True):
            return count
    fake.cpu_count = cpu_count
    monkeypatch.setitem(sys.modules, "psutil", fake)


def _value(bar):
    return bar._bar.value()


def test_per_core_bars_are_made_once_and_fill_from_a_sample(qtbot, monkeypatch):
    card = mm._SystemCard()
    qtbot.addWidget(card)
    _fake_psutil(monkeypatch, 3)
    card.per_core_toggle.setChecked(True)
    assert [bar._label.text() for bar in card._per_core_bars] == ["C00", "C01", "C02"]
    assert not card._per_core_wrap.isHidden()
    card.per_core_toggle.setChecked(False)
    assert card._per_core_wrap.isHidden()
    card.per_core_toggle.setChecked(True)
    assert len(card._per_core_bars) == 3
    card.apply({"ram": 41, "cpu": 12, "per_core": [5, 60, 99]})
    assert _value(card.ram) == 41 and _value(card.cpu) == 12
    assert [_value(bar) for bar in card._per_core_bars] == [5, 60, 99]


def test_a_core_count_that_cannot_be_read_makes_no_bars(qtbot, monkeypatch):
    card = mm._SystemCard()
    qtbot.addWidget(card)
    _fake_psutil(monkeypatch, RuntimeError("no /proc"))
    card._on_toggle_per_core(True)
    assert card._per_core_bars == [] and not card._per_core_wrap.isHidden()


def test_stale_and_empty_samples_leave_the_bars_alone(qtbot):
    card = mm._SystemCard()
    qtbot.addWidget(card)
    card.apply({"ram": 30})
    card.apply({"ram": 90}, generation=card._generation + 1)
    card.apply({})
    assert _value(card.ram) == 30


def test_no_second_sample_is_taken_while_one_is_out(qtbot, monkeypatch):
    card = mm._SystemCard()
    qtbot.addWidget(card)
    asked = []
    monkeypatch.setattr(card._jobs, "is_busy", lambda: True)
    monkeypatch.setattr(card._jobs, "submit", lambda *a, **k: asked.append(a))
    card.refresh()
    assert asked == []


def test_showing_the_card_again_keeps_one_sampling_timer(qtbot, monkeypatch):
    card = mm._SystemCard()
    qtbot.addWidget(card)
    sampled = []
    monkeypatch.setattr(card, "refresh", lambda: sampled.append(1))
    card._timer.start()
    card.show()
    qtbot.waitExposed(card)
    assert sampled == [] and card._timer.isActive()
    card.hide()
    assert not card._timer.isActive()


# ---------------------------------------------------------------------------
# The screen
# ---------------------------------------------------------------------------

def _fields(root: Path, count=3) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        imageio.imwrite(root / f"img_{index:02d}.tif",
                        np.full((32, 32), 10 * (index + 1), dtype=np.uint16))
    return root


@pytest.fixture
def screen(qtbot, qt_theme_applied):
    widget = MakeMasksScreen()
    qtbot.addWidget(widget)
    yield widget
    widget._mask_stage.dirty = False


def test_no_vvvv_export_without_a_folder_to_write_beside(qtbot, screen, tmp_path, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt import prefs

    folder = _fields(tmp_path / "fields")
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    monkeypatch.setattr(preferences, "_vvvv_export_enabled", lambda: True)
    prefs._s().remove("make_masks/vvvv_export_dir")
    written = []
    monkeypatch.setattr(screen, "_on_export_vvvv",
                        lambda root=None, **kw: written.append(root) or root)
    assert screen._canvas.mask is not None
    screen._folder = ""
    assert screen._export_vvvv_on_save() is None
    assert written == []


def test_hide_judged_with_no_folder_is_not_remembered(screen, monkeypatch):
    from spacr.qt import prefs

    stored = []

    class _Settings:
        def setValue(self, key, value):  # noqa: N802
            stored.append((key, value))

        def value(self, *args, **kwargs):
            return False

    monkeypatch.setattr(prefs, "_s", lambda: _Settings())
    assert not screen._folder
    screen._btn_hide_judged.setChecked(True)
    assert screen._hide_judged and stored == []


def test_settings_that_cannot_be_written_or_read_do_not_stop_the_switch(qtbot,
        screen, tmp_path, monkeypatch):
    from spacr.qt import prefs

    folder = _fields(tmp_path / "fields")
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)

    class _Broken:
        def setValue(self, *_a):  # noqa: N802
            raise OSError("settings are read-only")

        def value(self, *_a, **_k):
            raise OSError("settings are unreadable")

    monkeypatch.setattr(prefs, "_s", lambda: _Broken())
    screen._btn_hide_judged.setChecked(True)
    assert screen._hide_judged
    screen._restore_hide_judged()
    assert not screen._btn_hide_judged.isChecked()


def test_hiding_judged_fields_when_every_field_is_judged_says_so(qtbot, screen, tmp_path):
    folder = _fields(tmp_path / "judged", 2)
    for index in range(2):
        path = str(folder / f"img_{index:02d}.tif")
        engine.record_curation(str(folder), path, path, 1, index == 0)
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    screen._btn_hide_judged.setChecked(True)
    assert screen._status_label.text() == "Every field in this queue has Keep or Discard."
    screen._btn_hide_judged.setChecked(False)


def test_a_measure_section_needs_no_object_name_or_tooltip(screen):
    called = []
    action = screen._add_measure_section("Extra section", lambda: called.append(1))
    assert action.objectName() == "" and action.toolTip() == "Extra section"
    assert action in screen._measure_menu.actions()
    action.trigger()
    assert called == [1]


def test_open_file_cancelled_or_unreadable_opens_nothing(screen, tmp_path, monkeypatch):
    asked = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: asked.append(a[2]) or ("", "")))
    assert screen._on_pick_file() is None
    assert asked and asked[0]
    monkeypatch.setattr(screen, "_open_folder", lambda folder, *a, **k: False)
    assert screen._on_pick_file(str(tmp_path / "missing.tif")) is None


def test_open_file_on_the_field_already_shown_stays_there(qtbot, screen, tmp_path, monkeypatch):
    folder = _fields(tmp_path / "fields")
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: not screen._loading and screen._reader_worker is None
                    and screen._load_worker is None)
    current = screen._current_index
    moved = []
    monkeypatch.setattr(screen, "_go_to_index", lambda index: moved.append(index) or True)
    assert screen._on_pick_file(str(folder / screen._image_files[current])) is True
    assert moved == []


def test_a_finished_folder_job_leaves_a_newer_job_in_place(screen):
    reported = []

    class _Job:
        def __init__(self):
            self.deleted = False

        def deleteLater(self):  # noqa: N802
            self.deleted = True

    finished, newer = _Job(), _Job()
    screen._folder_job = newer
    screen._finish_folder_job(finished, reported.append)
    assert reported == [finished]
    assert screen._folder_job is newer and finished.deleted and not newer.deleted
    screen._folder_job = None


def test_measure_3d_needs_a_volume_and_a_free_stage(screen, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append((title, text)))
    assert screen._on_measure_volume() is None
    assert warned[-1] == ("Measure 3D images", "Open a 3D image before measuring.")
    source = tmp_path / "volume.tif"
    tifffile.imwrite(source, np.arange(4 * 10 * 12, dtype=np.uint16).reshape(4, 10, 12),
                     metadata={"axes": "ZYX"}, photometric="minisblack")
    assert screen.open_volume(str(source)) is screen._mask_stage
    screen._volume_measure_running = True
    assert screen._on_measure_volume() is None
    assert warned[-1] == ("Busy", "Wait for the current 3D operation to finish.")
    screen._volume_measure_running = False
    started = []
    monkeypatch.setattr(screen, "_run_organize", lambda dialog: False)
    monkeypatch.setattr(screen, "_start_volume_measurement",
                        lambda *a, **k: started.append(a))
    dialog = screen._on_measure_volume()
    assert dialog is screen._volume_measure_dialog and started == []
    dialog.close()


def test_a_3d_measurement_for_an_image_no_longer_open_is_refused(screen, tmp_path,
                                                                 monkeypatch):
    warned = []
    monkeypatch.setattr(screen, "_warn", lambda title, text: warned.append((title, text)))
    source = tmp_path / "volume.tif"
    tifffile.imwrite(source, np.arange(4 * 10 * 12, dtype=np.uint16).reshape(4, 10, 12),
                     metadata={"axes": "ZYX"}, photometric="minisblack")
    assert screen.open_volume(str(source)) is screen._mask_stage
    other = [{"source_path": str(tmp_path / "other.tif")}]
    assert screen._start_volume_measurement(other) is False
    assert screen._start_volume_measurement([]) is False
    assert warned == [("Measure 3D images",
                       "The current 3D image changed. Open Measure again.")] * 2
    assert not screen._volume_measure_running
