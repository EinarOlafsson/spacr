"""Installing a model from a click, without freezing the window.

Item 419, point 3 (the maintainer, 2026-09-16): "for the new models, these
should be grayed out [...] and clickable to install them in the spacr
environment. [...] these models should also be available in the mask modual
in the same way, that is grayed out untill clicked, which installs them and
makes them accessable to loade."

WHERE IT INSTALLS CHANGED ON 2026-09-19, answering item 423: "Isolated env
per backend!". The gesture is the same -- a greyed row that installs itself
when it is chosen -- but the install goes through the Model Zoo's own
dialog, into ``~/.spacr/backends/<name>``, and spaCR's own environment is
never changed. :class:`spacr.qt.model_install.PackageInstall` is still the
off-the-GUI-thread pip runner and is still tested here on its own terms; it
is what a checkpoint download and any future pip install use.

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


@pytest.fixture
def no_backend_environments(monkeypatch):
    """No backend has an environment of its own, and none is importable.

    tests/qt/conftest.py already points SPACR_BACKENDS_DIR at a fresh folder.
    SAMCell may still be importable in the developer's own environment -- an
    older spaCR pip-installed it there, a state item 423 still honours and
    reads as installed -- and these tests are about a backend that is NOT
    here.
    """
    from spacr import _segmentation_backends as backends

    monkeypatch.setattr(backends, "_importable", lambda module: False)


def _fake_backend_install(monkeypatch, answer=True, installs=None,
                          present=None):
    """Stand in for the Model Zoo's install dialog.

    :param answer: what the dialog reports -- installed, or not.
    :param installs: a module name the install makes importable.
    :param present: the set ``installs`` is added to.
    :returns: the list of backends the dialog was opened for.
    """
    from spacr.qt.widgets import model_zoo_picker

    asked = []

    def install(parent, name):
        asked.append(name)
        if answer and installs and present is not None:
            present.add(installs)
        return answer

    monkeypatch.setattr(model_zoo_picker, "install_backend", install)
    return asked


def test_the_mask_module_lists_every_backend_and_installs_on_click(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """"these models should also be available in the mask modual"."""
    from PySide6.QtWidgets import QWidget

    from spacr import _segmentation_backends as backends
    from spacr.qt.screens.settings_model import SettingsWidgets

    installed = set()
    monkeypatch.setattr(backends, "_importable",
                        lambda module: module in installed)
    asked = _fake_backend_install(monkeypatch, installs="dinocell",
                                  present=installed)
    holder = QWidget()
    qtbot.addWidget(holder)
    form = SettingsWidgets("mask", holder)
    form.build_sections()
    box = form._widgets["segmentation_backend"]
    assert isinstance(box, model_install.SegmentationBackendCombo)
    assert box.itemData(0) == "cellpose"
    assert {box.itemData(i) for i in range(box.count())} == {
        "cellpose", "cellpose3", "dinocell", "samcell"}
    assert form.collect()["segmentation_backend"] == "cellpose"
    for name in ("cellpose3", "dinocell", "samcell"):
        row = box.findData(name)
        assert box.itemData(row, Qt.ForegroundRole) is not None
        assert "not installed" in box.itemData(row, Qt.ToolTipRole)

    _choose(box, "dinocell")
    assert asked == ["dinocell"], "the Model Zoo's installer was not used"
    assert box.currentData() == "dinocell"
    assert form.collect()["segmentation_backend"] == "dinocell"
    assert box.itemData(box.findData("dinocell"), Qt.ForegroundRole) is None
    assert box.itemData(box.findData("samcell"), Qt.ForegroundRole) is not None

    assert form.set_value_for_key("segmentation_backend", "samcell")
    assert form.collect()["segmentation_backend"] == "samcell", (
        "a saved settings file naming a missing backend must still load")


def test_nothing_in_the_backend_box_runs_pip_against_spacrs_own_environment(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """The maintainer's 2026-09-19 answer, held down where it was broken.

    Choosing a greyed row used to run `pip install "spacr[<backend>]"`
    against the interpreter spaCR is running in.
    """
    started = _fake_pip(monkeypatch)
    _fake_backend_install(monkeypatch, answer=False)
    box = model_install.SegmentationBackendCombo()
    qtbot.addWidget(box)
    for name in ("samcell", "dinocell", "cellpose3"):
        _choose(box, name)
    assert started == [], f"pip was run against spaCR's own environment: {started}"


def test_a_backend_whose_state_cannot_be_read_falls_back_to_importing_it(
        qtbot, qt_theme_applied, monkeypatch, no_backends):
    """The state is a few file checks, but the folder can be unreadable.

    A backend spaCR cannot ask about is still installed when its package
    imports -- the older arrangement item 423 keeps honouring -- so the row
    is not greyed on a question that could not be asked.
    """
    from spacr import _segmentation_backends as backends

    def _broken(name, root=None):
        raise OSError("the backends folder is unreadable")

    monkeypatch.setattr(backends, "_backend_state", _broken)
    monkeypatch.setattr(model_install, "is_importable",
                        lambda module: module == "samcell")
    box = model_install.SegmentationBackendCombo()
    qtbot.addWidget(box)
    assert "samcell" not in box.missing()
    assert "dinocell" in box.missing()


def test_a_refused_install_goes_back_to_the_backend_that_was_chosen(
        qtbot, qt_theme_applied, monkeypatch, no_backends,
        no_backend_environments):
    """On SAMCell, a click on a missing DINOCell returns to SAMCell."""
    from spacr import _segmentation_backends as backends

    monkeypatch.setattr(backends, "_importable",
                        lambda module: module == "samcell")
    _fake_backend_install(monkeypatch, answer=False)
    box = model_install.SegmentationBackendCombo(default="samcell")
    qtbot.addWidget(box)
    assert box.currentData() == "samcell"
    _choose(box, "dinocell")
    assert box.currentData() == "samcell"
    assert box.job is None


def test_an_install_outlives_the_widget_that_started_it(
        qtbot, qt_theme_applied, monkeypatch):
    """A widget is rebuilt or closed; the pip it started must not be killed.

    The backend box no longer starts one -- it opens the Model Zoo's dialog,
    which owns its own job -- so the property is held down on
    :class:`PackageInstall` itself, which is what any pip started from a
    widget still uses.
    """
    _fake_pip(monkeypatch, delay=0.8)
    ended = []

    def start_and_forget():
        """Start one and drop every reference to it, as a rebuilt form does."""
        job = model_install.PackageInstall("spacr[samcell]")
        job.finished.connect(lambda worked, _m: ended.append(worked))
        assert job.start()
        assert job.is_running() and job in model_install._RUNNING
        return id(job)

    ident = start_and_forget()
    QApplication.processEvents()
    qtbot.waitUntil(lambda: bool(ended), timeout=15_000)
    assert ended == [True], "the install died with the widget that started it"
    assert not [job for job in model_install._RUNNING if id(job) == ident]


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
