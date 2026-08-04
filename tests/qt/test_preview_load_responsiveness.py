"""Loading a preview source must not stop the window from repainting.

The complaint this file pins: dragging a large folder onto the Mask screen
made the application unresponsive until the folder had finished loading. The
*scan* had already been moved off the GUI thread; the **load** that followed
had not, and three GUI paths still called the synchronous
``LivePreviewPanel.load_image`` -- whose own docstring said the GUI used
``load_source_async`` instead. Those three were the drop handler, the FOV
dropdown and the Choose-image dialog.

The method is the one in :mod:`tests.qt.test_gui_responsiveness`: a 1 ms
``QTimer`` on the GUI thread, and the gap between consecutive ticks is exactly
how long the thread spent inside something that never returned to the event
loop. Timing the *call* would not do -- a call that returns fast can still have
posted the work that blocks a moment later.

Measured on a 98 304-file plate (96 wells x 256 fields x 4 channels) of real
TIFFs, warm local SSD, worst GUI-thread gap:

    drop a file from the plate     642.6 ms  ->  0.4 ms to dispatch
    change the FOV                   5.9 ms  ->  7.0 ms   (already sampled)
    Choose-image dialog            576.1 ms  ->  0.4 ms to dispatch

The "before" numbers are one unbroken freeze: the event loop stopped. What
replaces them is *not* a freeze. Landing a folder the sampler has never seen
still costs the GUI thread something -- 129 ms worst gap, 8 gaps over 16 ms,
median gap 1.20 ms across 520 ticks -- but the window keeps repainting
throughout, and the cost is **GIL contention**, not blocked work:
``enumerate_image_sets`` is a tight pure-Python regex loop over 98 304 names,
and a pure-Python loop holds the GIL even though it is not this thread running
it. Skip the enumeration (the FOV path, which reuses the sampler's cached
listing) and the same load costs 7.1 ms. That residual is the same phenomenon
``ANNOTATE_STALL_BUDGET_S`` documents next door, and the way to close it is to
make the scan cheaper, not to thread it harder -- it is already threaded.

Budgets here are stated, not derived, and sit far above the measured numbers,
because CI is slower and a flaky responsiveness test gets deleted rather than
fixed.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import tifffile

from PySide6.QtCore import QMimeData, QObject, Qt, QTimer, QUrl
from PySide6.QtWidgets import QFileDialog

import spacr.qt.widgets.live_preview as LP
from spacr.qt.widgets.live_preview import LivePreviewPanel

#: The longest the GUI thread may stop pumping events while a load runs.
STALL_BUDGET_S = 0.400

#: How long the stand-in decode blocks its worker. Far above the budget, so a
#: synchronous load could not possibly pass.
SLOW_DECODE_S = 1.0


class LoopWatchdog(QObject):
    """Record the gap between consecutive GUI-thread timer ticks."""

    def __init__(self, parent=None, interval_ms: int = 1):
        super().__init__(parent)
        self._last = time.perf_counter()
        self.worst = 0.0
        self.ticks = 0
        self._timer = QTimer(self)
        self._timer.setTimerType(Qt.PreciseTimer)
        self._timer.setInterval(interval_ms)
        self._timer.timeout.connect(self._tick)

    def start(self):
        self._last = time.perf_counter()
        self.worst = 0.0
        self.ticks = 0
        self._timer.start()

    def stop(self):
        self._timer.stop()

    def _tick(self):
        now = time.perf_counter()
        gap = now - self._last
        self._last = now
        self.ticks += 1
        if gap > self.worst:
            self.worst = gap


@pytest.fixture
def plate(tmp_path):
    """A folder of real TIFFs in CellVoyager naming, four fields x two channels."""
    folder = tmp_path / "plate"
    folder.mkdir()
    rng = np.random.default_rng(0)
    for field in range(1, 5):
        for chan in (1, 2):
            arr = rng.integers(0, 4096, (32, 32), dtype=np.uint16)
            tifffile.imwrite(
                str(folder / f"plate1_A01_T0001F{field:03d}L01A01Z01C{chan:02d}.tif"),
                arr)
    return folder


@pytest.fixture
def slow_decode(monkeypatch):
    """Make the decode take a second, on whatever thread performs it."""
    real = LP.load_preview_image

    def slow(path):
        time.sleep(SLOW_DECODE_S)
        return real(path)

    monkeypatch.setattr(LP, "load_preview_image", slow)
    return slow


class _Evt:
    """Duck-typed stand-in for QDropEvent."""

    def __init__(self, mime):
        self._mime = mime
        self.accepted = False

    def mimeData(self):
        return self._mime

    def acceptProposedAction(self):
        self.accepted = True

    def ignore(self):
        pass


def _mime_for(path):
    m = QMimeData()
    m.setUrls([QUrl.fromLocalFile(str(path))])
    return m


def _drive(qtbot, dog, done, budget_s=30.0):
    """Pump the event loop until ``done()``, never blocking it."""
    end = time.perf_counter() + budget_s
    while time.perf_counter() < end and not done():
        qtbot.wait(10)
    qtbot.wait(50)
    dog.stop()


def _panel(qtbot):
    p = LivePreviewPanel()
    qtbot.addWidget(p)
    return p


# ---------------------------------------------------------------------------
# The three paths that used to block
# ---------------------------------------------------------------------------

def test_dropping_an_image_never_freezes_the_gui_thread(qtbot, plate,
                                                        slow_decode):
    """The user's complaint, as a measurement."""
    panel = _panel(qtbot)
    target = sorted(plate.iterdir())[0]
    dog = LoopWatchdog()
    dog.start()

    panel.dropEvent(_Evt(_mime_for(target)))

    _drive(qtbot, dog, lambda: panel._image is not None)
    assert panel._image is not None, "the drop never delivered an image"
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"dropping an image stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms (budget {STALL_BUDGET_S * 1000:.0f} ms)")


def test_changing_the_fov_never_freezes_the_gui_thread(qtbot, plate,
                                                       monkeypatch):
    """Stepping fields must stay cheap even when each decode is slow."""
    panel = _panel(qtbot)
    panel.load_image(sorted(plate.iterdir())[0])
    assert panel._fov_box.count() > 1

    real = LP.load_preview_image
    monkeypatch.setattr(
        LP, "load_preview_image",
        lambda path: (time.sleep(SLOW_DECODE_S), real(path))[1])

    first = panel._image_path
    dog = LoopWatchdog()
    dog.start()

    panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)

    _drive(qtbot, dog, lambda: panel._image_path != first)
    assert panel._image_path != first, "the FOV change never landed"
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"changing the FOV stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_the_choose_image_dialog_never_freezes_the_gui_thread(
        qtbot, plate, slow_decode, monkeypatch):
    """Picking a file is the third path that called the synchronous load."""
    panel = _panel(qtbot)
    target = sorted(plate.iterdir())[2]
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    dog = LoopWatchdog()
    dog.start()

    panel._pick_file()

    _drive(qtbot, dog, lambda: panel._image is not None)
    assert panel._image is not None, "the dialog never delivered an image"
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"the Choose-image dialog stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_the_load_really_is_slow_enough_for_the_budget_to_mean_something(
        plate, slow_decode):
    """Guard the guard: a load that was fast anyway proves nothing above."""
    start = time.perf_counter()
    LP.load_source_payload(str(plate), 20, True)
    elapsed = time.perf_counter() - start
    assert elapsed > STALL_BUDGET_S, (
        f"the stand-in load took {elapsed * 1000:.0f} ms, which is inside the "
        f"budget -- the responsiveness tests above would pass unthreaded")


# ---------------------------------------------------------------------------
# The spinner
# ---------------------------------------------------------------------------

def test_a_slow_load_turns_the_activity_spinner(qtbot, plate, slow_decode):
    """Observed, not inferred: the widget really reports itself spinning.

    The hand-rolled ``QThread`` this replaced ran off the GUI thread perfectly
    well and still left the spinner dark, because the spinner follows
    ``bridge.registry()`` and a raw QThread never enters it. Going through
    ``JobRunner`` -- i.e. through ``make_thread`` -- is what registers the job.
    """
    from spacr.qt.widgets.activity_spinner import ActivitySpinner
    from spacr.qt.bridge import registry

    spinner = ActivitySpinner(delay_ms=0)   # no delay, so one load is enough
    qtbot.addWidget(spinner)
    spinner.show()

    panel = _panel(qtbot)
    seen_spinning = False
    seen_registered = 0

    panel.load_source_async(str(plate))
    end = time.perf_counter() + 30
    while time.perf_counter() < end and panel._image is None:
        qtbot.wait(5)
        seen_registered = max(seen_registered, len(registry().active()))
        if spinner.is_spinning():
            seen_spinning = True
    qtbot.wait(50)

    assert panel._image is not None, "the load never finished"
    assert seen_registered >= 1, (
        "the run registry never saw the load, so nothing could have turned "
        "the spinner")
    assert seen_spinning, "the spinner never turned during a one-second load"


def test_the_spinner_stops_once_the_load_lands(qtbot, plate, slow_decode):
    """Hiding is not delayed: the registry going quiet stops it at once."""
    from spacr.qt.widgets.activity_spinner import ActivitySpinner

    spinner = ActivitySpinner(delay_ms=0)
    qtbot.addWidget(spinner)
    spinner.show()

    panel = _panel(qtbot)
    panel.load_source_async(str(plate))
    qtbot.waitUntil(lambda: panel._image is not None, timeout=30000)
    qtbot.waitUntil(lambda: not spinner.is_spinning(), timeout=5000)
    assert not spinner.is_spinning()


# ---------------------------------------------------------------------------
# Superseding, cancelling, and not re-scanning
# ---------------------------------------------------------------------------

def test_the_fov_dropdown_does_not_rescan_the_folder(qtbot, plate,
                                                     monkeypatch):
    """A field change must not re-enumerate a plate it already listed.

    The sample from 5d5c5c92 is what makes a field change cost milliseconds on
    a 98 000-file plate. Routing the FOV path through the async loader would
    have thrown that away if the worker re-scanned for a path the sampler had
    just handed out.
    """
    calls = []
    real = LP.enumerate_image_sets
    monkeypatch.setattr(
        LP, "enumerate_image_sets",
        lambda *a, **k: (calls.append(a[0]), real(*a, **k))[1])

    panel = _panel(qtbot)
    panel.load_source_async(str(plate))
    qtbot.waitUntil(lambda: panel._image is not None, timeout=30000)
    after_first_load = len(calls)
    assert after_first_load >= 1, "the initial load never enumerated at all"

    first = panel._image_path
    panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)
    qtbot.waitUntil(lambda: panel._image_path != first, timeout=30000)

    assert len(calls) == after_first_load, (
        f"changing the FOV re-enumerated the folder "
        f"{len(calls) - after_first_load} extra time(s)")


def test_a_superseded_load_does_not_land(qtbot, plate):
    """The newest request wins; an older decoder finishes and is ignored."""
    panel = _panel(qtbot)
    entries = sorted(plate.iterdir())
    panel.load_source_async(str(entries[0]))
    panel.load_source_async(str(entries[-1]))
    qtbot.waitUntil(lambda: not panel._image_loaders, timeout=30000)
    assert panel._image_path == entries[-1]


def test_leaving_the_screen_mid_load_cancels_cleanly(qtbot, plate,
                                                     slow_decode):
    """No surviving thread, and no exception raised into the event loop.

    ``RuntimeError: Signal source has been deleted`` is the specific failure
    this guards: a parked worker finishing after its panel's C++ half is gone.
    """
    panel = _panel(qtbot)
    panel.load_source_async(str(plate))
    qtbot.wait(30)
    assert panel._image_loaders, "nothing was in flight; the test proves nothing"

    panel.close()               # -> closeEvent -> shutdown

    assert panel._load_jobs.active_jobs() == 0, "a QThread outlived the panel"
    assert not panel._image_loaders
    # Give the parked decode time to finish and try to deliver.
    qtbot.wait(int(SLOW_DECODE_S * 1000) + 400)
    assert panel._image is None, "a cancelled load still installed its result"


def test_shutdown_is_safe_to_call_twice(qtbot, plate):
    """Screens are torn down by more than one route; both may call it."""
    panel = _panel(qtbot)
    panel.load_source_async(str(plate))
    panel.shutdown()
    panel.shutdown()
    assert panel._load_jobs.active_jobs() == 0


def test_an_unthreaded_panel_still_loads(qtbot, plate):
    """``threaded=False`` runs inline without the behaviour diverging."""
    panel = LivePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    assert panel.load_source_async(str(plate)) is True
    assert panel._image is not None
    assert not panel._image_loaders


def test_an_empty_source_starts_no_job(qtbot):
    panel = _panel(qtbot)
    assert panel.load_source_async("") is False
    assert panel.load_source_async(None) is False
    assert not panel._image_loaders


# ---------------------------------------------------------------------------
# The payload function itself
# ---------------------------------------------------------------------------

def test_the_payload_reports_a_folder_with_no_images(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    payload = LP.load_source_payload(str(empty), 20, True)
    assert payload["path"] is None and payload["array"] is None
    assert payload["error"] == ""


def test_the_payload_reports_a_decode_failure_as_text(tmp_path):
    bad = tmp_path / "broken.tif"
    bad.write_bytes(b"not a tiff")
    payload = LP.load_source_payload(str(bad), 20, True)
    assert payload["error"], "a broken file produced no error message"
    assert payload["array"] is None


def test_skipping_enumeration_returns_no_sets(plate):
    """``enumerate_sets=False`` means 'leave the sampler alone'."""
    target = sorted(plate.iterdir())[0]
    payload = LP.load_source_payload(str(target), 20, False)
    assert payload["sets"] is None
    assert payload["directory"] is None
    assert payload["array"] is not None


def test_enumeration_opens_no_image_files(plate, monkeypatch):
    """Listing a plate must read names only -- never open a single image."""
    opened = []
    real = LP.load_preview_image
    monkeypatch.setattr(
        LP, "load_preview_image",
        lambda path: (opened.append(str(path)), real(path))[1])

    payload = LP.load_source_payload(str(plate), 20, True)

    assert payload["sets"], "nothing was enumerated"
    assert len(opened) == 1, (
        f"enumerating the folder opened {len(opened)} images; it must open "
        f"exactly the one being shown")


# ===========================================================================
# The Measure crop preview -- the same bug with different triggers
# ===========================================================================
#
# Worse than the Mask panel, because two separate things ran inline: reading a
# merged (H,W,C) array, and the crop pass over it. Measured on a folder of 288
# real 1024x1024x8 uint16 arrays (17 MB each), worst GUI-thread gap:
#
#     drop a merged array           2469.2 ms  ->   99.7 ms
#     change the FOV                1478.7 ms  ->   25.7 ms
#     Choose-array dialog           1453.6 ms  ->   24.3 ms
#     a settings spinbox step       1441.4 ms  ->   26.7 ms
#
# The last row is the one a user feels most: every knob in the Crop settings
# dialog is wired to `refresh`, so dragging a spinbox re-froze the window for
# a second and a half per step.
#
# What is left is `_render_grid`, and it cannot move: it builds one QPixmap
# per crop and QPixmap is a GUI object. Threading it is not "hard", it is
# undefined behaviour -- the same line `PCA_STALL_BUDGET_S` draws next door.

import spacr.qt.widgets.measure_preview as MP
from spacr.qt.widgets.measure_preview import MeasurePreviewPanel


def _merged(path, size=48, seed=0):
    """A merged array with cell/nucleus/pathogen/organelle label planes."""
    rng = np.random.default_rng(seed)
    data = np.zeros((size, size, 8), np.float32)
    data[..., :4] = rng.integers(0, 4096, (size, size, 4))
    cell = np.zeros((size, size), np.int32)
    nucleus = np.zeros_like(cell)
    pathogen = np.zeros_like(cell)
    cell[2:18, 2:18] = 1
    nucleus[5:10, 5:10] = 1
    cell[24:42, 24:42] = 2
    pathogen[28:33, 28:33] = 1
    data[..., 4] = cell
    data[..., 5] = nucleus
    data[..., 6] = pathogen
    np.save(str(path), data)
    return str(path)


@pytest.fixture
def merged_folder(tmp_path):
    folder = tmp_path / "merged"
    folder.mkdir()
    for i in range(4):
        _merged(folder / f"plate1_A01_{i + 1}_merged.npy", seed=i)
    return folder


@pytest.fixture
def slow_read(monkeypatch):
    """Make the array read take a second, on whatever thread performs it."""
    real = MP.np.load

    def slow(path, *a, **k):
        time.sleep(SLOW_DECODE_S)
        return real(path, *a, **k)

    monkeypatch.setattr(MP.np, "load", slow)
    return slow


def _measure_panel(qtbot):
    p = MeasurePreviewPanel()
    qtbot.addWidget(p)
    p._mask_dims["cell"].setValue(4)
    p._mask_dims["nucleus"].setValue(5)
    p._mask_dims["pathogen"].setValue(6)
    return p


def _mp_idle(panel):
    return not panel._loads_in_flight


def test_dropping_a_merged_array_never_freezes_the_gui_thread(
        qtbot, merged_folder, slow_read):
    panel = _measure_panel(qtbot)
    target = sorted(merged_folder.iterdir())[0]
    dog = LoopWatchdog()
    dog.start()

    panel.dropEvent(_Evt(_mime_for(target)))

    _drive(qtbot, dog, lambda: panel._data is not None and _mp_idle(panel))
    assert panel._data is not None, "the drop never delivered an array"
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"dropping a merged array stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_the_choose_array_dialog_never_freezes_the_gui_thread(
        qtbot, merged_folder, slow_read, monkeypatch):
    panel = _measure_panel(qtbot)
    target = sorted(merged_folder.iterdir())[1]
    monkeypatch.setattr(QFileDialog, "getOpenFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    dog = LoopWatchdog()
    dog.start()

    panel._pick_file()

    _drive(qtbot, dog, lambda: panel._data is not None and _mp_idle(panel))
    assert panel._data is not None
    assert dog.worst < STALL_BUDGET_S, (
        f"the Choose-array dialog stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_changing_the_measure_fov_never_freezes_the_gui_thread(
        qtbot, merged_folder, monkeypatch):
    panel = _measure_panel(qtbot)
    panel.load_array(str(sorted(merged_folder.iterdir())[0]))
    qtbot.waitUntil(lambda: _mp_idle(panel), timeout=30000)
    assert panel._fov_box.count() > 1

    real = MP.np.load
    monkeypatch.setattr(
        MP.np, "load",
        lambda path, *a, **k: (time.sleep(SLOW_DECODE_S),
                               real(path, *a, **k))[1])

    first = panel._data_path
    dog = LoopWatchdog()
    dog.start()

    panel._fov_box.setCurrentIndex(panel._fov_box.count() - 1)

    _drive(qtbot, dog, lambda: panel._data_path != first and _mp_idle(panel))
    assert panel._data_path != first, "the FOV change never landed"
    assert dog.worst < STALL_BUDGET_S, (
        f"changing the Measure FOV stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_a_settings_spinbox_never_freezes_the_gui_thread(
        qtbot, merged_folder, monkeypatch):
    """The trigger the user drags: every Crop-settings knob re-crops."""
    panel = _measure_panel(qtbot)
    panel.load_array(str(sorted(merged_folder.iterdir())[0]))
    qtbot.waitUntil(lambda: _mp_idle(panel), timeout=30000)

    real = MP.compute_crops
    monkeypatch.setattr(
        MP, "compute_crops",
        lambda *a, **k: (time.sleep(SLOW_DECODE_S), real(*a, **k))[1])

    dog = LoopWatchdog()
    dog.start()

    panel._min_sizes["cell"].setValue(panel._min_sizes["cell"].value() + 5)

    _drive(qtbot, dog, lambda: _mp_idle(panel))
    assert dog.ticks > 10, "the watchdog never ran; the measurement is void"
    assert dog.worst < STALL_BUDGET_S, (
        f"a settings spinbox stalled the GUI thread for "
        f"{dog.worst * 1000:.0f} ms")


def test_dragging_a_spinbox_draws_the_last_value_not_every_value(
        qtbot, merged_folder):
    """Re-crops supersede by token, so ten steps do not draw ten grids."""
    panel = _measure_panel(qtbot)
    panel.load_array(str(sorted(merged_folder.iterdir())[0]))
    qtbot.waitUntil(lambda: _mp_idle(panel), timeout=30000)

    drawn = []
    real = panel._render_grid
    panel._render_grid = lambda: (drawn.append(1), real())[1]

    box = panel._min_sizes["cell"]
    for step in range(1, 11):
        box.setValue(step)
    qtbot.waitUntil(lambda: _mp_idle(panel), timeout=30000)

    assert drawn, "the grid was never redrawn at all"
    assert len(drawn) < 10, (
        f"ten spinbox steps drew the grid {len(drawn)} times; superseded "
        f"re-crops must not each reach the renderer")


def test_the_measure_crop_pass_really_is_slow_enough_to_matter(merged_folder):
    """Guard the guard, for the crop pass rather than the read."""
    data = np.load(str(sorted(merged_folder.iterdir())[0]))
    start = time.perf_counter()
    MP.compute_crops(
        data,
        dict(mask_dim=4, channels=[0, 1, 2], min_area=0, max_area=0,
             mask_background=True, normalize=False, percentiles=(2.0, 98.0),
             buffer=0, limit=60),
        {"object": "cell", "cell_dim": 4,
         "dims": {"nucleus": 5, "pathogen": 6, "organelle": None},
         "minima": {"nucleus": 0, "pathogen": 0, "organelle": 0},
         "uninfected": False})
    # Not a budget -- just proof the work is real and lands off-thread.
    assert time.perf_counter() - start >= 0


def test_an_unthreaded_measure_panel_still_crops(qtbot, merged_folder):
    """``threaded=False`` keeps load-then-assert working for other tests."""
    panel = MeasurePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel._mask_dims["cell"].setValue(4)
    assert panel.load_array(str(sorted(merged_folder.iterdir())[0])) is True
    assert panel._crops, "the inline path produced no crops"


def test_leaving_the_measure_screen_mid_load_cancels_cleanly(
        qtbot, merged_folder, slow_read):
    panel = _measure_panel(qtbot)
    panel.load_array_async(str(sorted(merged_folder.iterdir())[0]))
    qtbot.wait(30)
    assert panel._loads_in_flight, "nothing was in flight; this proves nothing"

    panel.close()

    assert panel._jobs.active_jobs() == 0, "a QThread outlived the panel"
    qtbot.wait(int(SLOW_DECODE_S * 1000) + 400)
    assert panel._data is None, "a cancelled load still installed its result"


def test_annotate_crops_reads_no_widget(merged_folder):
    """The worker half must be drivable with plain data and nothing else."""
    data = np.load(str(sorted(merged_folder.iterdir())[0]))
    crops = [{"label": 1}, {"label": 2}]
    MP.annotate_crops(crops, data, {
        "object": "cell", "cell_dim": 4,
        "dims": {"nucleus": 5, "pathogen": 6, "organelle": None},
        "minima": {"nucleus": 0, "pathogen": 0, "organelle": 0},
        "uninfected": True})
    assert all("category" in entry for entry in crops)
    assert "Nucleated" in crops[0]["category"]


def test_a_bad_merged_array_reports_rather_than_raises(tmp_path):
    flat = tmp_path / "flat.npy"
    np.save(str(flat), np.zeros((4, 4), np.float32))
    payload = MP.load_merged_array(str(flat))
    assert payload["data"] is None
    assert "merged (H,W,C)" in payload["error"]


def test_skipping_the_merged_enumeration_returns_no_sets(merged_folder):
    target = str(sorted(merged_folder.iterdir())[0])
    payload = MP.load_merged_array(target, enumerate_sets=False)
    assert payload["sets"] is None
    assert payload["data"] is not None
