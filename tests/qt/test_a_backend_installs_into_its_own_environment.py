"""Item 423, the part a user touches: install, cancel and uninstall a backend.

The model and the protocol are pinned in
``tests/test_backends_live_in_their_own_environment.py``. Here the controls
are pressed: the install dialog's Install, Cancel and Close, the Model Zoo
button's rows and its Install and Uninstall buttons, and the Model Zoo
screen's. Every job is a stand-in that reports progress the way the real
install does, so nothing downloads; the real install was run by hand in a
sandboxed HOME (see the item's ledger note).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

import spacr._segmentation_backends as SB
from spacr import model_zoo
from spacr.qt.widgets import model_zoo_picker as mzp


def _ready(name="cellpose3"):
    return SB._BackendState(name, SB._INSTALLED, "in its own environment.",
                            "/env")


class _Job:
    """A stand-in install: reports two steps, then waits to be released."""

    def __init__(self, outcome=None, hold=False, honour_cancel=True):
        self.outcome = outcome if outcome is not None else _ready()
        self.release = threading.Event()
        self.hold = hold
        self.honour_cancel = honour_cancel
        self.calls = []

    def __call__(self, progress=None, cancel=None):
        self.calls.append(cancel)
        progress(0, 1, "Checking this computer can install it")
        progress(0, 4, "Create the environment")
        progress(1, 4, "Install PyTorch: Downloading torch-2.14.0+cpu.whl")
        while self.hold and not self.release.is_set():
            if self.honour_cancel and cancel is not None and cancel.is_set():
                raise SB._InstallCancelled("the install was cancelled")
            self.release.wait(0.02)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome


@pytest.fixture
def dialog_for(qtbot):
    made = []

    def _make(name="cellpose3", **kwargs):
        dialog = mzp.BackendInstallDialog(name, **kwargs)
        qtbot.addWidget(dialog)
        dialog.show()
        made.append(dialog)
        return dialog

    yield _make
    for dialog in made:
        job = getattr(dialog, "_job", None)
        if isinstance(job, _Job):
            job.release.set()
        if dialog.running:
            dialog._cancel.set()
            qtbot.waitUntil(lambda d=dialog: not d.running, timeout=10_000)


def test_the_dialog_says_where_it_goes_what_it_downloads_and_its_licence(
        dialog_for):
    dialog = dialog_for()
    text = dialog.blurb.text()
    assert "environment of its own" in text
    assert "spaCR's own environment is not changed" in text
    assert "cellpose==3.1.1.3" in text and "torch" in text
    assert "BSD-3-Clause" in text
    assert dialog.windowTitle() == "Install Cellpose 3"
    assert dialog.start_button.text() == "Install"
    assert dialog.start_button.isEnabled()


def test_install_runs_off_the_gui_thread_with_progress(qtbot, dialog_for):
    job = _Job(hold=True)
    dialog = dialog_for(job=job)
    qtbot.mouseClick(dialog.start_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: "Downloading torch" in dialog.status.text(),
                    timeout=10_000)
    assert dialog.running and not dialog.start_button.isEnabled()
    assert dialog.progress.isVisible()
    assert (dialog.progress.maximum(), dialog.progress.value()) == (4, 1)
    assert dialog.progress.format() == "step 2 of 4"
    job.release.set()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    assert dialog.installed and dialog.state.ready
    assert dialog.result() == QDialog.Accepted
    assert dialog.status.text() == "Cellpose 3 is installed and ready."


def test_a_failure_is_shown_verbatim_and_can_be_tried_again(qtbot,
                                                            dialog_for):
    failure = SB._InstallFailed(
        "Install Cellpose 3 failed: `pip install cellpose==3.1.1.3` exited "
        "with code 1.\n\nERROR: No matching distribution found")
    dialog = dialog_for(job=_Job(outcome=failure))
    qtbot.mouseClick(dialog.start_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: dialog.details.isVisible(), timeout=10_000)
    assert dialog.details.toPlainText() == str(failure)
    assert dialog.start_button.text() == "Try again"
    assert dialog.start_button.isEnabled()
    assert dialog.cancel_button.text() == "Close"
    assert "Nothing was left half-built" in dialog.status.text()
    assert not dialog.installed


def test_cancel_stops_the_install_and_closes_once_it_has_stopped(
        qtbot, dialog_for):
    job = _Job(hold=True)
    dialog = dialog_for(job=job)
    qtbot.mouseClick(dialog.start_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: "Downloading" in dialog.status.text(),
                    timeout=10_000)
    qtbot.mouseClick(dialog.cancel_button, Qt.LeftButton)
    assert job.calls[0].is_set(), "Cancel must reach the running install"
    assert not dialog.cancel_button.isEnabled()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    assert dialog.status.text() == "Cancelled. Nothing was left behind."
    assert dialog.result() == QDialog.Rejected and not dialog.isVisible()


def test_a_cancel_that_arrives_after_the_close_request_still_closes(
        qtbot, dialog_for):
    dialog = dialog_for(job=_Job(outcome=SB._InstallCancelled("stop")))
    dialog.start()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    assert dialog.cancel_button.text() == "Close"
    assert dialog.isVisible(), "a cancel nobody asked to close on stays open"


def test_a_failure_after_the_close_request_closes(qtbot, dialog_for):
    job = _Job(outcome=SB._InstallFailed("boom"), hold=True,
               honour_cancel=False)
    dialog = dialog_for(job=job)
    dialog.start()
    dialog.start()
    dialog.reject()
    job.release.set()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    assert not dialog.isVisible() and dialog.details.toPlainText() == "boom"


def test_closing_the_window_mid_install_is_cancel(qtbot, dialog_for):
    job = _Job(hold=True)
    dialog = dialog_for(job=job)
    dialog.start()
    qtbot.waitUntil(lambda: "Downloading" in dialog.status.text(),
                    timeout=10_000)
    dialog.close()
    assert dialog.isVisible(), "the window waits for the worker"
    assert job.calls[0].is_set()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    dialog.close()
    assert not dialog.isVisible()


def test_a_backend_that_cannot_install_here_says_why_and_cannot_start(
        monkeypatch, dialog_for):
    monkeypatch.setattr(SB, "_backend_state", lambda name: SB._BackendState(
        name, SB._UNAVAILABLE, "Cellpose 3 does not run on sunos5.", "/env"))
    dialog = dialog_for()
    assert dialog.reason.text() == (
        "Not installable here: Cellpose 3 does not run on sunos5.")
    assert not dialog.start_button.isEnabled()


def test_no_network_offers_to_try_anyway(monkeypatch, dialog_for):
    monkeypatch.setattr(SB, "_backend_state", lambda name: SB._BackendState(
        name, SB._UNAVAILABLE, "no network: pypi.org could not be reached.",
        "/env"))
    dialog = dialog_for()
    assert dialog.start_button.text() == "Try anyway"
    assert dialog.start_button.isEnabled()


def test_an_install_already_running_cannot_be_started_twice(monkeypatch,
                                                            dialog_for):
    monkeypatch.setattr(SB, "_backend_state", lambda name: SB._BackendState(
        name, SB._INSTALLING, "being installed since noon by process 1.",
        "/env"))
    dialog = dialog_for()
    assert "being installed since noon" in dialog.reason.text()
    assert not dialog.start_button.isEnabled()


def test_uninstall_deletes_the_environment_and_cannot_be_interrupted(
        qtbot, dialog_for):
    removed = SB._BackendState("cellpose3", SB._INSTALLABLE, "", "/env")
    job = _Job(outcome=removed, hold=True)
    dialog = dialog_for(uninstall=True, job=job)
    assert dialog.windowTitle() == "Uninstall Cellpose 3"
    assert "deletes its environment" in dialog.blurb.text()
    assert dialog.start_button.text() == "Uninstall"
    dialog.start()
    assert not dialog.cancel_button.isEnabled()
    dialog.reject()
    assert dialog.running, "half an uninstall is worse than a finished one"
    job.release.set()
    qtbot.waitUntil(lambda: not dialog.running, timeout=10_000)
    assert dialog.removed and not dialog.installed
    assert dialog.status.text() == "Cellpose 3 was uninstalled."


def test_the_real_jobs_are_the_install_and_the_uninstall(qtbot, monkeypatch,
                                                         dialog_for):
    seen = []

    def _install(name, progress=None, cancel=None):
        seen.append(("install", name, cancel is not None))
        return _ready(name)

    def _uninstall(name):
        seen.append(("uninstall", name))
        return SB._BackendState(name, SB._INSTALLABLE)

    monkeypatch.setattr(SB, "_install_backend", _install)
    monkeypatch.setattr(SB, "_uninstall_backend", _uninstall)
    install = dialog_for("samcell")
    assert install._job(progress=print, cancel=threading.Event()).ready
    remove = dialog_for("samcell", uninstall=True)
    assert not remove._job(progress=print, cancel=None).ready
    assert seen == [("install", "samcell", True), ("uninstall", "samcell")]
    install._join()
    assert not install.running


def test_the_worker_object_reports_each_outcome_by_signal():
    """``_BackendJob.run`` on this thread: the same signals the dialog
    receives when it runs on its own."""
    heard = []

    def _listen(job):
        worker = mzp._BackendJob(job, threading.Event())
        worker.progressed.connect(lambda *a: heard.append(("progress", a)))
        worker.succeeded.connect(lambda s: heard.append(("ok", s.name)))
        worker.failed.connect(lambda m: heard.append(("failed", m)))
        worker.cancelled.connect(lambda: heard.append(("cancelled",)))
        worker.run()

    _listen(_Job())

    def _cancelled(progress=None, cancel=None):
        raise SB._BackendCancelled("stopped")

    def _nameless(progress=None, cancel=None):
        raise KeyError()

    _listen(_cancelled)
    _listen(_nameless)
    assert heard[0] == ("progress", (0, 1,
                                     "Checking this computer can install it"))
    assert heard[3:] == [("ok", "cellpose3"), ("cancelled",),
                         ("failed", "KeyError")]


def test_the_one_call_forms_open_the_dialog_and_report(monkeypatch):
    opened = []

    def _exec(self):
        opened.append((self._name, self._uninstall))
        self.installed = not self._uninstall
        self.removed = self._uninstall

    monkeypatch.setattr(mzp.BackendInstallDialog, "exec", _exec)
    assert mzp.install_backend(None, "dinocell") is True
    assert mzp.uninstall_backend(None, "dinocell") is True
    row = SimpleNamespace(uri="backend:samcell", kind="backend")
    assert mzp.install_backend_package(None, row) is True
    model = SimpleNamespace(uri="https://x/y.pth", kind="cellpose3")
    assert mzp.install_backend_package(None, model) is True
    other = SimpleNamespace(uri="https://x/y.pth", kind="cellpose")
    assert mzp.install_backend_package(None, other) is False
    assert opened == [("dinocell", False), ("dinocell", True),
                      ("samcell", False), ("cellpose3", False)]


# ---------------------------------------------------------------------------
# The Model Zoo button
# ---------------------------------------------------------------------------

def _backend_row(name, state, path=""):
    spec = SB._SPECS[name]
    return model_zoo.ModelEntry(
        key=f"{name}_v1", name=spec.label, kind="backend", source=state,
        uri=f"backend:{name}", path=path, trained_on=spec.blurb,
        trained_by=spec.label, licence=spec.licence,
        notes=(f"{state}: the reason.", spec.licence_note))


def _cellpose3_row(model, ready):
    return model_zoo.ModelEntry(
        key=f"cellpose3_{model}", name=model, kind="cellpose3",
        source="stock", path=model if ready else "", uri="backend:cellpose3",
        trained_on="a Cellpose 3 model", licence="BSD-3-Clause")


@pytest.fixture
def picker(qapp, qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(mzp, "DEFAULT_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(mzp, "remembered_model_dir", lambda: str(tmp_path))
    dialog = mzp.ModelZooPicker()
    qtbot.addWidget(dialog)
    yield dialog
    dialog._stop_any_download()


def _select(picker, name):
    for row, (_stem, pairs) in enumerate(picker._groups):
        if pairs[0][1].name == name:
            picker.table.selectRow(row)
            return pairs[0][1]
    raise AssertionError(f"no row {name}")


def test_every_backend_row_says_its_state(picker):
    picker._rebuild([
        _backend_row("cellpose3", "installable"),
        _backend_row("dinocell", "not installable here"),
        _backend_row("samcell", "installing"),
        _cellpose3_row("cyto2", ready=False),
    ])
    status = [picker.table.item(r, 3).text()
              for r in range(picker.table.rowCount())]
    assert status == ["not installed — click to install",
                      "not installable here", "installing…",
                      "needs the Cellpose 3 backend"]
    assert picker.table.item(1, 3).toolTip() == "not installable here: the reason."
    picker._rebuild([_backend_row("cellpose3", "installed", "/env"),
                     _cellpose3_row("nuclei", ready=True)])
    assert [picker.table.item(r, 3).text() for r in range(2)] == [
        "installed", "on this machine"]
    assert mzp._status_text(SimpleNamespace(kind="backend", source="new"),
                            None) == "new"


def test_clicking_an_uninstalled_backend_offers_the_install(picker,
                                                            monkeypatch):
    asked = []
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: asked.append(entry.name) or True)
    for name in ("Cellpose 3", "SAMCell", "cyto3"):
        picker._rebuild([_backend_row("cellpose3", "installable"),
                         _backend_row("samcell", "installed", "/env"),
                         _cellpose3_row("cyto3", ready=False)])
        _select(picker, name)
        picker._row_clicked(None)
    assert asked == ["Cellpose 3", "cyto3"], (
        "an installed backend offers nothing; a Cellpose 3 model whose "
        "backend is missing offers the backend")


def test_a_failed_install_leaves_the_status_line_empty(picker, monkeypatch):
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: False)
    picker._rebuild([_backend_row("dinocell", "installable")])
    picker._install_backend(_backend_row("dinocell", "installable"))
    assert picker.status.text() == ""


def test_the_buttons_follow_the_selected_row(picker, monkeypatch):
    picker._rebuild([_backend_row("cellpose3", "installable"),
                     _backend_row("samcell", "installed", "/env"),
                     _backend_row("dinocell", "installed"),
                     _cellpose3_row("cyto", ready=False)])
    _select(picker, "Cellpose 3")
    assert picker.download_button.text() == "Install"
    assert picker.download_button.isEnabled()
    assert not picker.use_button.isEnabled()
    assert not picker.uninstall_button.isEnabled()
    card = picker.card.toPlainText()
    assert "Segmentation backend — installable: the reason." in card
    assert "BSD-3-Clause" in card

    _select(picker, "SAMCell")
    assert picker.uninstall_button.isEnabled(), "installed in its own env"
    assert not picker.download_button.isEnabled()
    assert picker.download_button.text() == "Download"

    _select(picker, "DINOCell")
    assert not picker.uninstall_button.isEnabled(), (
        "one inside spaCR's own environment is not spaCR's to remove")

    _select(picker, "cyto")
    assert picker.download_button.text() == "Install"
    assert "not installed — press Install" in picker.card.toPlainText()


def test_uninstall_asks_removes_and_redraws(picker, monkeypatch):
    removed = []
    monkeypatch.setattr(mzp, "uninstall_backend",
                        lambda parent, name: removed.append(name) or True)
    refreshed = []
    monkeypatch.setattr(picker, "refresh", lambda: refreshed.append(1))
    picker._rebuild([_backend_row("samcell", "installed", "/env")])
    _select(picker, "SAMCell")
    picker._uninstall_selected()
    assert removed == ["samcell"] and refreshed == [1]
    assert picker.status.text() == "SAMCell was uninstalled."
    picker.table.clearSelection()
    picker._uninstall_selected()
    assert removed == ["samcell"]
    monkeypatch.setattr(mzp, "uninstall_backend",
                        lambda parent, name: removed.append(name) and False)
    picker._rebuild([_backend_row("samcell", "installed", "/env")])
    _select(picker, "SAMCell")
    picker._uninstall_selected()
    assert removed == ["samcell", "samcell"] and refreshed == [1]


def test_download_on_a_backend_row_installs(picker, monkeypatch):
    asked = []
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: asked.append(entry.name) or True)
    picker._rebuild([_cellpose3_row("nuclei", ready=False)])
    _select(picker, "nuclei")
    picker._download_selected()
    assert asked == ["nuclei"]


def test_a_cellpose3_card_says_whether_its_backend_is_here(monkeypatch):
    monkeypatch.setattr(SB, "_backend_state", lambda name: _ready(name))
    card = mzp._cellpose3_card(SimpleNamespace(licence=""))
    assert "which is installed" in card and "Licence" not in card
    plain = mzp._backend_card(SimpleNamespace(notes=(), source="installed",
                                              licence="MIT"))
    assert "installed" in plain and "Licence: MIT" in plain
    bare = mzp._backend_card(SimpleNamespace(notes=(), source="installable"))
    assert "Licence" not in bare


def test_the_zoo_checks_the_network_in_the_background_and_redraws(
        qtbot, picker, monkeypatch):
    refreshed = []
    release = threading.Event()

    def _slow():
        release.wait(10)
        return {"cellpose3": "no network: down"}

    monkeypatch.setattr(SB, "_probe_blockers", _slow)
    monkeypatch.setattr(picker, "refresh", lambda: refreshed.append(1))
    picker._probe_backends()
    qtbot.wait(600)
    assert picker._probe_timer.isActive() and refreshed == [], (
        "the timer waits for the probe; the GUI thread never does")
    release.set()
    qtbot.waitUntil(lambda: refreshed == [1], timeout=10_000)
    assert not picker._probe_timer.isActive()

    monkeypatch.setattr(SB, "_probe_blockers", lambda: {})
    picker._probe_backends()
    qtbot.waitUntil(lambda: not picker._probe_timer.isActive(),
                    timeout=10_000)
    assert refreshed == [1], "a probe that found nothing redraws nothing"


# ---------------------------------------------------------------------------
# The Model Zoo screen
# ---------------------------------------------------------------------------

@pytest.fixture
def screen(qapp, qtbot):
    from spacr.qt.screens.model_zoo import ModelZooScreen

    made = ModelZooScreen(threaded=False)
    qtbot.addWidget(made)
    return made


def test_the_screen_says_each_backends_state(screen):
    from spacr.qt.screens.model_zoo import _status_of

    assert _status_of(_backend_row("dinocell", "not installable here")) == (
        "not installable here")
    assert _status_of(_cellpose3_row("cyto3", ready=True)) == "usable"
    assert _status_of(_cellpose3_row("cyto3", ready=False)) == (
        "needs the Cellpose 3 backend")


def test_the_screen_installs_and_uninstalls_from_its_buttons(screen,
                                                             monkeypatch):
    asked, removed, scanned = [], [], []
    monkeypatch.setattr(mzp, "install_backend_package",
                        lambda parent, entry: asked.append(entry.name) or True)
    monkeypatch.setattr(mzp, "uninstall_backend",
                        lambda parent, name: removed.append(name) or True)
    monkeypatch.setattr(screen, "scan",
                        lambda *a, **k: scanned.append(k) or True)
    screen.set_entries([_backend_row("cellpose3", "installable"),
                        _backend_row("samcell", "installed", "/env"),
                        _cellpose3_row("cyto2", ready=False)])
    screen.select(0)
    assert screen._btn_download.text() == "Install"
    assert screen._btn_download.isEnabled()
    assert not screen._btn_uninstall.isEnabled()
    assert screen.download_selected() is True
    screen._row_clicked(None)
    screen.select(2)
    screen._row_clicked(None)
    assert asked == ["Cellpose 3", "Cellpose 3", "cyto2"]

    screen.select(1)
    assert screen._btn_uninstall.isEnabled()
    assert not screen._btn_download.isEnabled()
    assert screen.uninstall_selected() is True
    assert removed == ["samcell"]
    assert "SAMCell was uninstalled." in screen.status_text()
    screen.select(0)
    assert screen.uninstall_selected() is False
    assert len(scanned) == 4
    screen.select(1)
    screen._row_clicked(None)
    assert asked == ["Cellpose 3", "Cellpose 3", "cyto2"], (
        "an installed backend offers nothing on a click")
    monkeypatch.setattr(mzp, "uninstall_backend", lambda parent, name: False)
    assert screen.uninstall_selected() is False
    assert len(scanned) == 4
