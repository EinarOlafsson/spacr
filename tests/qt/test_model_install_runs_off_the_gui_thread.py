"""Installing a model from a click, without freezing the window.

Item 419, point 3 (the maintainer, 2026-09-16): "for the new models, these
should be grayed out [...] and clickable to install them in the spacr
environment. [...] these models should also be available in the mask modual
in the same way, that is grayed out untill clicked, which installs them and
makes them accessable to loade."

:class:`spacr.qt.model_install.SegmentationBackendCombo` is the Mask
module's segmentation_backend control; these tests build the Mask
settings form, choose rows the way a click does, and run a real child process
in place of pip so that "the window keeps working" is measured. The Make
Masks side of the same point is tested in
test_make_masks_reads_out_lays_out_and_installs.py, which imports the
helpers below.
"""
from __future__ import annotations

import sys

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from spacr.qt import model_install
from spacr.qt.screens import make_masks as mm


@pytest.fixture
def no_backends(monkeypatch):
    """Neither DINOCell nor SAMCell is here until the stand-in pip runs."""
    present = {"cellpose"}
    monkeypatch.setattr(mm, "find_spec",
                        lambda name: object() if name in present else None)
    monkeypatch.setattr(model_install, "find_spec",
                        lambda name: object() if name in present else None)
    return present


def _fake_pip(monkeypatch, *, code=0, delay=0.6, installs=None, present=None):
    """Replace pip with a real child process that sleeps, prints, and exits."""
    script = (f"import time; print('Collecting stand-in', flush=True); "
              f"time.sleep({delay}); print('Successfully installed stand-in'); "
              f"raise SystemExit({code})")
    started = []

    def command(requirement):
        started.append(requirement)
        if installs and present is not None:
            present.add(installs)
        return [sys.executable, "-c", script]

    monkeypatch.setattr(model_install, "pip_command", command)
    return started


def _choose(box, data):
    """Choose a row the way a click does: current index, then activated."""
    index = box.findData(data)
    assert index >= 0, data
    box.setCurrentIndex(index)
    box.activated.emit(index)


def test_the_mask_module_lists_every_backend_and_installs_on_click(
        qtbot, qt_theme_applied, monkeypatch, no_backends):
    """"these models should also be available in the mask modual"."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens.settings_model import SettingsWidgets

    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    _fake_pip(monkeypatch, installs="dinocell", present=no_backends)
    holder = QWidget()
    qtbot.addWidget(holder)
    form = SettingsWidgets("mask", holder)
    form.build_sections()
    box = form._widgets["segmentation_backend"]
    assert isinstance(box, model_install.SegmentationBackendCombo)
    assert [box.itemData(i) for i in range(box.count())] == [
        "cellpose", "samcell", "dinocell"] or [
        box.itemData(i) for i in range(box.count())] == [
        "cellpose", "dinocell", "samcell"]
    assert form.collect()["segmentation_backend"] == "cellpose"
    for name in ("dinocell", "samcell"):
        row = box.findData(name)
        assert box.itemData(row, Qt.ForegroundRole) is not None
        assert "not installed" in box.itemData(row, Qt.ToolTipRole)

    _choose(box, "dinocell")
    assert box.currentData() == "cellpose", "it rested on a missing backend"
    qtbot.waitUntil(lambda: box.job is not None and not box.job.is_running()
                    and box.isEnabled(), timeout=15_000)
    assert box.currentData() == "dinocell"
    assert form.collect()["segmentation_backend"] == "dinocell"
    assert box.itemData(box.findData("dinocell"), Qt.ForegroundRole) is None
    assert box.itemData(box.findData("samcell"), Qt.ForegroundRole) is not None

    assert form.set_value_for_key("segmentation_backend", "samcell")
    assert form.collect()["segmentation_backend"] == "samcell", (
        "a saved settings file naming a missing backend must still load")


def test_a_refused_install_goes_back_to_the_backend_that_was_chosen(
        qtbot, qt_theme_applied, monkeypatch, no_backends):
    """On SAMCell, a click on a missing DINOCell returns to SAMCell."""
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.Cancel))
    no_backends.add("samcell")
    box = model_install.SegmentationBackendCombo(default="samcell")
    qtbot.addWidget(box)
    assert box.currentData() == "samcell"
    _choose(box, "dinocell")
    assert box.currentData() == "samcell"
    assert box.job is None


def test_an_install_outlives_the_widget_that_started_it(
        qtbot, qt_theme_applied, monkeypatch, no_backends):
    """A settings form is rebuilt; the pip it started must not be killed."""
    monkeypatch.setattr(QMessageBox, "warning",
                        staticmethod(lambda *a, **k: QMessageBox.Yes))
    _fake_pip(monkeypatch, delay=0.8)
    box = model_install.SegmentationBackendCombo()
    _choose(box, "samcell")
    job = box.job
    assert job is not None and job.is_running()
    box.deleteLater()
    del box
    QApplication.processEvents()
    ended = []
    job.finished.connect(lambda worked, _m: ended.append(worked))
    qtbot.waitUntil(lambda: bool(ended), timeout=15_000)
    assert ended == [True], "the install was killed with its widget"
    assert job not in model_install._RUNNING


def test_a_failed_pip_reports_its_exit_code_and_its_last_lines(
        qtbot, qt_theme_applied):
    job = model_install.PackageInstall("x", command=[
        sys.executable, "-c",
        "print('first'); print('ERROR: no matching distribution'); "
        "raise SystemExit(1)"])
    ended = []
    job.finished.connect(lambda worked, message: ended.append((worked, message)))
    assert job.start()
    qtbot.waitUntil(lambda: bool(ended), timeout=15_000)
    worked, message = ended[0]
    assert not worked
    assert "pip exited 1" in message
    assert "ERROR: no matching distribution" in message
    assert job not in model_install._RUNNING


def test_an_installer_that_cannot_start_says_so_at_once(qtbot, qt_theme_applied):
    job = model_install.PackageInstall(
        "x", command=["/nonexistent/spacr-no-such-pip"])
    ended = []
    job.finished.connect(lambda worked, message: ended.append((worked, message)))
    assert job.start() is False
    assert ended and ended[0][0] is False
    assert job not in model_install._RUNNING


def test_a_frozen_build_does_not_offer_pip(qtbot, qt_theme_applied,
                                           monkeypatch):
    told = []
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(QMessageBox, "information",
                        staticmethod(lambda *a, **k: told.append(a)))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(
        lambda *a, **k: pytest.fail("a frozen build asked to run pip")))
    assert not model_install.confirm_backend_install(None, "SAMCell",
                                                     "spacr[samcell]")
    assert told
