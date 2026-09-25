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
    console.say("Loaded plate1_A02_12.tif")
    for step in range(0, 101, 10):
        console.say(f"Restoring {step}%…", "progress")
        assert console.progress_text() == f"Restoring {step}%…"
        assert console.progress.percent() == step
    text = console.text()
    assert "Loaded plate1_A02_12.tif" in text
    assert "Restoring" not in text, "a progress step went into the scrollback"

    console.say("Loading the model…", "progress")
    assert console.progress.percent() is None, "no number: the bar is busy"
    console.say("Restoration finished in 3.1 s.")
    assert console.progress_text() == ""
    assert not console.progress.isVisibleTo(console)
    console.say("Restoration finished in 3.1 s.")
    assert console.text().count("Restoration finished in 3.1 s.") == 1
    console.say("Cellpose 3 raised ValueError: nope", "error")
    console.say("careful", "warning")
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
    assert ("progress", "Installing Cellpose 3…: Installing torch  (step 2 of 5)"
            ) in said
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
