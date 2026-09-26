"""Item 507: Make Masks has a console, and the Cellpose 3 install button says
where the backend stands.

The maintainer pressed "Install Cellpose 3…", saw a dialog, pressed Install,
and was told nothing; pressing Apply under Enhance then piled "magnifier could
not segment this region" and a stack of progress bars into the bottom-right
corner. These tests pin the four fixes: a console right of the image carrying
those lines with ONE in-place progress line, an install button with three
faces read from the environment, a magnifier failure that is said once, and a
worker's redrawn progress bars read as one line each.
"""
from __future__ import annotations

import types

import pytest
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication

from spacr import _segmentation_backends as SB
from spacr.qt.screens import make_masks as mm
from spacr.qt.widgets import model_zoo_picker as zoo


@pytest.fixture
def screen(qtbot):
    widget = mm.MakeMasksScreen()
    qtbot.addWidget(widget)
    yield widget
    widget._close_levels()
    widget._magnifier.close()
    widget._canvas.close_enhancer()


# ---------------------------------------------------------------------------
# 1. The console: lines in the scrollback, one progress line rewritten
# ---------------------------------------------------------------------------

def test_the_console_keeps_lines_and_rewrites_one_progress_line(qtbot):
    console = mm._MasksConsole()
    qtbot.addWidget(console)
    console.show()
    console.post("Loaded plate1_A02_12.tif")
    for step in range(0, 101, 10):
        console.post(f"Restoring {step}%…", "progress")
        assert console.progress_text() == f"Restoring {step}%…"
        assert console.progress.percent() == step
    text = console.text()
    assert "Loaded plate1_A02_12.tif" in text
    assert "Restoring" not in text, "a progress step went into the scrollback"

    console.post("Loading the model…", "progress")
    assert console.progress.percent() is None, "no number: the bar is busy"
    console.post("Restoration finished in 3.1 s.")
    assert console.progress_text() == ""
    assert not console.progress.isVisibleTo(console)
    console.post("Restoration finished in 3.1 s.")
    assert console.text().count("Restoration finished in 3.1 s.") == 1
    console.post("Cellpose 3 raised ValueError: nope", "error")
    console.post("careful", "warning")
    assert "nope" in console.text() and "careful" in console.text()


def test_the_console_sits_right_of_the_image_under_the_shortcuts(screen, qtbot):
    screen.resize(1500, 900)
    screen._body_stack.setCurrentWidget(screen._body_splitter)
    screen.show()
    qtbot.waitExposed(screen)
    for _ in range(6):
        QApplication.processEvents()
    console = screen._masks_console
    card = screen._shortcut_card
    tabs = screen._view_tabs
    assert screen._shortcut_panel.isAncestorOf(console)
    assert console.isVisible() and card.isVisible()
    left_of_console = console.mapTo(screen, console.rect().topLeft())
    right_of_tabs = tabs.mapTo(screen, tabs.rect().topRight()).x()
    bottom_of_card = card.mapTo(screen, card.rect().bottomLeft()).y()
    assert left_of_console.x() > right_of_tabs, "right of the image"
    assert left_of_console.y() > bottom_of_card, "below the shortcut list"
    column = screen._shortcut_panel
    column.set_collapsed("Console", True, by_user=True)
    qtbot.waitUntil(lambda: column.is_collapsed("Console"))
    column.set_collapsed("Console", False, by_user=True)
    qtbot.waitUntil(lambda: not column.is_collapsed("Console"))


def test_the_corner_shows_one_line_and_the_console_gets_all_of_it(screen):
    label = screen._status_label
    stacked = "Magnifier could not segment this region: boom\n" + "\n".join(
        f"{p}%|####| {p}/100" for p in range(0, 101, 5))
    label.setText(stacked)
    assert label.text() == stacked
    assert "\n" not in QLabelText(label)
    assert len(QLabelText(label)) <= mm._StatusLabel.LIMIT
    assert "Magnifier could not segment this region: boom" in (
        screen._masks_console.text())

    label.setText("Object detection (cyto3) running…")
    assert screen._masks_console.progress_text() == (
        "Object detection (cyto3) running…")
    label.setText("Object detection found 33 objects.")
    assert screen._masks_console.progress_text() == ""
    assert "33 objects" in screen._masks_console.text()


def QLabelText(label):
    """What the corner paints, not the whole text it keeps."""
    return super(mm._StatusLabel, label).text()


def test_restoration_lines_reach_the_console(screen):
    controls = screen._restoration_controls
    controls.said.emit("Loading restoration model…", "progress")
    assert screen._masks_console.progress_text() == "Loading restoration model…"
    controls.said.emit("Restoration unavailable: no weights", "warning")
    assert "no weights" in screen._masks_console.text()
    assert screen._status_label.text() == "Restoration unavailable: no weights"


# ---------------------------------------------------------------------------
# 2. The install button: three faces from the environment, and the reason
# ---------------------------------------------------------------------------

class _Disk:
    """A faked ``_backend_state``: what the environment says, changeable."""

    def __init__(self, state):
        self.state = state
        self.asked = 0

    def __call__(self, name):
        self.asked += 1
        return types.SimpleNamespace(state=self.state, reason="", env="/env")


@pytest.mark.parametrize("state, face, caption, enabled", [
    (SB._INSTALLABLE, zoo._NOT_INSTALLED, "Install Cellpose 3…", True),
    (SB._INSTALLING, zoo._INSTALLING_FACE, "Installing Cellpose 3…", False),
    (SB._INSTALLED, zoo._INSTALLED_FACE, "Cellpose 3 is installed", False),
])
def test_the_button_reads_its_face_from_the_environment(
        qtbot, state, face, caption, enabled):
    disk = _Disk(state)
    button = zoo._BackendInstallButton("cellpose3", probe=disk)
    qtbot.addWidget(button)
    assert button.face() == face
    assert button.text() == caption
    assert button.isEnabled() is enabled
    button.show()
    asked = disk.asked
    disk.state = SB._INSTALLED
    button.hide()
    button.show()
    assert disk.asked > asked, "coming on screen re-reads the environment"
    assert button.face() == zoo._INSTALLED_FACE and not button.isEnabled()


class _Dialog(QObject):
    """The install dialog's four signals, without the dialog."""

    job_started = Signal()
    job_progressed = Signal(str)
    job_failed = Signal(str)
    job_cancelled = Signal()


@pytest.mark.parametrize("outcome", ["installed", "failed", "cancelled"])
def test_the_button_says_installing_then_how_it_ended(qtbot, outcome):
    disk = _Disk(SB._INSTALLABLE)
    said, faces = [], []

    def installer(parent, name, watch=None):
        dialog = _Dialog()
        watch(dialog)
        dialog.job_started.emit()
        faces.append((parent.face(), parent.text(), parent.isEnabled()))
        dialog.job_progressed.emit("Installing torch  (step 2 of 5)")
        if outcome == "installed":
            disk.state = SB._INSTALLED
            return True
        if outcome == "failed":
            dialog.job_failed.emit("pip: No matching distribution for torch")
        else:
            dialog.job_cancelled.emit()
        return False

    button = zoo._BackendInstallButton("cellpose3", probe=disk,
                                      installer=installer)
    qtbot.addWidget(button)
    button.said.connect(lambda text, kind: said.append((kind, text)))
    button.click()
    assert faces == [(zoo._INSTALLING_FACE, "Installing Cellpose 3…", False)]
    assert ("progress", "Installing Cellpose 3…") in said
    assert ("stream", "Installing Cellpose 3…: Installing torch  (step 2 of 5)"
            ) in said, "the install's own lines stream, throttled"
    if outcome == "installed":
        assert said[-1] == ("info", "Cellpose 3 is installed")
        assert button.face() == zoo._INSTALLED_FACE and not button.isEnabled()
    elif outcome == "failed":
        kind, text = said[-1]
        assert kind == "error"
        assert "No matching distribution for torch" in text
        assert button.face() == zoo._NOT_INSTALLED and button.isEnabled()
    else:
        assert said[-1] == ("info", "Cancelled. Nothing was left behind.")
        assert button.face() == zoo._NOT_INSTALLED and button.isEnabled()


# ---------------------------------------------------------------------------
# 3. The magnifier says a failure once, not on every mouse move
# ---------------------------------------------------------------------------

def test_a_magnifier_failure_is_said_once_per_reason(screen):
    magnifier = screen._magnifier
    heard = []
    magnifier.status.connect(heard.append)

    def fail(message):
        request = types.SimpleNamespace(
            key=(magnifier._field, "box"), scope="region", box=(0, 0, 8, 8))
        magnifier._on_delivered((request, None, ValueError(message)))

    for _ in range(25):
        fail("Image enhancement is updating.")
    assert len(heard) == 1 and "enhancement is updating" in heard[0]
    fail("Cellpose 3 raised RuntimeError: out of memory")
    assert len(heard) == 2
    magnifier.forget()
    fail("Cellpose 3 raised RuntimeError: out of memory")
    assert len(heard) == 3, "a new field hears the reason again"


# ---------------------------------------------------------------------------
# 4. A worker's redrawn progress bar is one line, in its last state
# ---------------------------------------------------------------------------

def test_carriage_returns_collapse_to_the_final_state():
    raw = ["loading weights\n", " 10%|#         | 1/10\r",
           " 50%|#####     | 5/10\r", "100%|##########| 10/10\n",
           "done\n", "a\rb\n", "\r\n"]
    assert SB._final_lines(raw) == [
        "loading weights", "100%|##########| 10/10", "done", "b", ""]
    assert SB._final_lines("x 1%\rx 99%\ny\n") == ["x 99%", "y"]


def test_the_worker_tail_holds_each_bar_once():
    worker = SB._WorkerProcess.__new__(SB._WorkerProcess)
    worker.name = "cellpose3"
    worker._stderr = []
    for chunk in ["start\n"] + [f"{p}%|bar\r" for p in range(0, 100, 10)] + [
            "100%|bar\n", "Traceback: boom\n"]:
        worker._said(chunk)
    assert worker._stderr == ["start", "100%|bar", "Traceback: boom"]


# ---------------------------------------------------------------------------
# 5. Where restoration runs, and when a cancel keeps the worker
# ---------------------------------------------------------------------------

def test_restoration_uses_the_gpu_the_environment_was_built_for(monkeypatch):
    from spacr.qt.widgets import restoration_controls as controls

    monkeypatch.delenv(SB._DEVICE_ENV, raising=False)
    assert controls._restoration_device() == "cpu", "nothing installed"

    def state(built):
        return lambda name, root=None: types.SimpleNamespace(
            state=SB._INSTALLED, record={"device": built}, env="/env")

    monkeypatch.setattr(SB, "_backend_state", state("cuda"))
    assert controls._restoration_device() == "auto"
    monkeypatch.setattr(SB, "_backend_state", state("cpu"))
    assert controls._restoration_device() == "cpu"
    monkeypatch.setenv(SB._DEVICE_ENV, "cpu")
    monkeypatch.setattr(SB, "_backend_state", state("cuda"))
    assert controls._restoration_device() == "cpu", "SPACR_DEVICE wins"


def test_a_cancelled_whole_field_on_the_cpu_stops_its_worker():
    import numpy as np

    field = np.zeros((1994, 1994), np.float32)
    box = np.zeros((256, 256), np.float32)
    cpu = types.SimpleNamespace(device="cpu")
    gpu = types.SimpleNamespace(device="cuda:0")
    assert SB._keep_restoring(field, cpu) is False
    assert SB._keep_restoring(box, cpu) is True
    assert SB._keep_restoring(field, gpu) is True


# ---------------------------------------------------------------------------
# 6. Streamed output: a worker's bar and an install's pip lines, live
# ---------------------------------------------------------------------------

_BAR = r'''
import sys, time
sys.stderr.write("loading weights\n")
sys.stderr.flush()
n = 200
for i in range(n + 1):
    sys.stderr.write("\r%3d%%|%-10s| %d/%d" % (
        i * 100 // n, "#" * (i * 10 // n), i, n))
    sys.stderr.flush()
    time.sleep(0.004)
sys.stderr.write("\n")
sys.stderr.flush()
'''


def _record_draws(console):
    """Every text the progress line is drawn with, and when."""
    import time

    drawn = []
    show = console.show_progress

    def recording(text):
        drawn.append((time.monotonic(), text))
        show(text)

    console.show_progress = recording
    return drawn


def _assert_throttled(console, drawn, took):
    """``drawn`` came from streamed output only, and no faster than allowed."""
    gaps = [b[0] - a[0] for a, b in zip(drawn, drawn[1:])]
    assert console.stream_updates == len(drawn) >= 2, "it moved while it ran"
    assert len(drawn) <= took / (console.STREAM_MS / 1000.0) + 3, (
        f"{len(drawn)} draws in {took:.2f} s is more than ten a second")
    assert min(gaps) >= console.STREAM_MS / 1000.0 * 0.8, gaps


def test_a_workers_tqdm_bar_streams_into_the_progress_line(qtbot):
    import subprocess
    import sys
    import threading
    import time

    console = mm._MasksConsole()
    qtbot.addWidget(console)
    console.show()
    drawn = _record_draws(console)
    worker = SB._WorkerProcess.__new__(SB._WorkerProcess)
    worker.name, worker.label, worker._stderr = "cellpose3", "Cellpose 3", []
    proc = subprocess.Popen([sys.executable, "-c", _BAR],
                            stderr=subprocess.PIPE, text=True,
                            encoding="utf-8")
    done = threading.Event()
    start = time.monotonic()
    threading.Thread(target=SB._pump, args=(SB._raw_lines(proc.stderr),
                                            worker._said, done.set),
                     daemon=True).start()
    visible_while_running = []
    while not done.is_set():
        qtbot.wait(20)
        visible_while_running.append(console.progress_text())
        assert time.monotonic() - start < 30
    took = time.monotonic() - start
    proc.wait(timeout=10)
    qtbot.wait(3 * console.STREAM_MS)

    _assert_throttled(console, drawn, took)
    texts = [text for _when, text in drawn]
    assert all(t.startswith("Cellpose 3: ") and "%|" in t for t in texts)
    percents = [int(t.split(": ")[1].split("%")[0]) for t in texts]
    assert percents == sorted(percents), "the line moves forward in place"
    assert any(v.startswith("Cellpose 3: ") for v in visible_while_running)
    assert console.progress_text() == "", "the line ends when the bar does"
    scrollback = console.text()
    final = "Cellpose 3: 100%|##########| 200/200"
    assert scrollback.count(final) == 1, scrollback
    assert "50%" not in scrollback, "a redraw went into the scrollback"
    assert "loading weights" not in scrollback, "chatter stays in the log"
    assert worker._stderr[-1] == "100%|##########| 200/200"


def test_an_install_streams_pip_lines_and_ends_with_one_line(qtbot):
    import time

    console = mm._MasksConsole()
    qtbot.addWidget(console)
    console.show()
    drawn = _record_draws(console)
    disk = _Disk(SB._INSTALLABLE)
    took = []

    def installer(parent, name, watch=None):
        dialog = _Dialog()
        watch(dialog)
        dialog.job_started.emit()
        start = time.monotonic()
        for i in range(150):
            dialog.job_progressed.emit(
                f"Installing packages: Downloading torch-2.4.0.whl "
                f"{i * 5.3:.1f}/797.1 MB  (step 3 of 5)")
            time.sleep(0.004)
            QApplication.processEvents()
        dialog.job_progressed.emit(
            "Installing packages: Successfully installed torch-2.4.0  "
            "(step 3 of 5)")
        took.append(time.monotonic() - start)
        disk.state = SB._INSTALLED
        return True

    button = zoo._BackendInstallButton("cellpose3", probe=disk,
                                      installer=installer)
    qtbot.addWidget(button)
    button.said.connect(console.post)
    button.click()
    qtbot.wait(3 * console.STREAM_MS)

    assert drawn[0][1] == "Installing Cellpose 3…", "the start is said at once"
    streamed = drawn[1:]
    _assert_throttled(console, streamed, took[0])
    assert all(text.startswith("Installing Cellpose 3…: Installing packages: "
                               "Downloading torch") for _when, text in streamed)
    assert console.progress_text() == ""
    scrollback = console.text()
    assert scrollback.count("Cellpose 3 is installed") == 1
    assert "Downloading torch" not in scrollback
    assert "Successfully installed" not in scrollback, (
        "a line still waiting to be drawn does not outlive the end")


def test_a_console_that_went_away_stops_listening(qtbot):
    from PySide6.QtCore import QEvent

    before = len(SB._OUTPUT_LISTENERS)
    console = mm._MasksConsole()
    assert len(SB._OUTPUT_LISTENERS) == before + 1
    console.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert len(SB._OUTPUT_LISTENERS) == before
    SB._tell_listeners("Cellpose 3", " 5%|bar\r")


@pytest.mark.parametrize("model, estimates", [
    ("cellpose3:cyto3", True), ("cellpose3:cyto2", True),
    ("cellpose3:cyto", True), ("cellpose3:nuclei", True),
    ("cellpose3:", True), ("CellPose3: nuclei", True),
    ("cellpose3:/models/cp3_weights.pth", False),
    ("cellpose_dino:/models/cellposedino_vit_b", False),
    ("cpsam", False), ("/models/finetuned_cpsam", False),
    ("dinocell", False), ("samcell", False), ("", False), (None, False),
])
def test_the_diameter_note_follows_the_route_that_estimates(model, estimates):
    """Item 507: Diameter 0 estimates only through a Cellpose 3 size model.

    A Cellpose 3 checkpoint runs at its trained diameter, and Cellpose-DINO
    and Cellpose-SAM run Cellpose 4, which segments at native scale when
    the diameter is 0 -- no estimate, so the note would be untrue there.
    """
    assert mm._diameter_zero_estimates(model) is estimates


def test_the_magnifier_asks_the_route_its_mode_runs():
    assert mm._magnifier_route_model(
        {"mode": "cellpose", "model_name": "cellpose3:cyto2"}) == \
        "cellpose3:cyto2"
    assert mm._magnifier_route_model(
        {"mode": "cellpose3:nuclei", "model_name": "cpsam"}) == \
        "cellpose3:nuclei"
    assert mm._magnifier_route_model(
        {"mode": "dinocell", "model_name": "cellpose3:cyto3"}) == "dinocell"
    assert mm._magnifier_route_model(
        {"mode": "cellpose_dino:/m/vit_b", "model_name": "cpsam"}) == \
        "cellpose_dino:/m/vit_b"


@pytest.fixture
def loaded(qtbot, tmp_path):
    import imageio.v2 as imageio
    import numpy as np

    folder = tmp_path / "fields"
    (folder / "masks").mkdir(parents=True)
    imageio.imwrite(folder / "f_00.tif", np.zeros((40, 40), np.uint16))
    widget = mm.MakeMasksScreen()
    qtbot.addWidget(widget)
    widget._open_folder(str(folder))
    if widget._loading:
        widget._load_worker.wait(30_000)
        widget._on_background_load_finished()
    assert widget._canvas.image is not None
    yield widget
    widget.close_folded()


@pytest.mark.parametrize("model, said", [
    ("cellpose3:cyto3", True),
    ("cellpose3:/models/cp3_weights.pth", False),
    ("cellpose_dino:/models/cellposedino_vit_b", False),
])
def test_object_detection_says_the_note_where_diameter_0_estimates(
        loaded, monkeypatch, model, said):
    reported = []
    submitted = []
    monkeypatch.setattr(loaded, "_report",
                        lambda text, kind="info": reported.append((text, kind)))
    loaded._detection_worker = types.SimpleNamespace(
        submit=submitted.append, close=lambda timeout=0: True)
    if loaded._cp_model.findData(model) < 0:
        loaded._cp_model.addItem(model, model)
    loaded._cp_model.setCurrentIndex(loaded._cp_model.findData(model))
    loaded._cp_diameter.setValue(0)
    loaded._on_detect_cellpose()
    assert len(submitted) == 1
    note = (mm._cellpose3_auto_diameter_note(), "warning")
    assert (note in reported) is said

    reported.clear()
    loaded._detection_request = None
    loaded._cp_diameter.setValue(44)
    loaded._on_detect_cellpose()
    assert note not in reported
    loaded._detection_request = None
