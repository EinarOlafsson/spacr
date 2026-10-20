"""The in-app uninstall coordinator's steps, run on the calling thread.

``tests/qt/test_n703_uninstall_dialog.py`` drives the real worker threads.
Coverage cannot trace code a QThread runs, so here the worker's ``run`` and
the preparation closure the coordinator hands its worker are called directly,
and each refusal, cancellation and late result is checked for what it leaves
behind: the helper disarmed, the window open, the coordinator released.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QDialog, QMessageBox

from spacr import install_cleanup
from spacr.qt import install_consent as consent
from tests.qt.test_n703_uninstall_dialog import Handshake, Window, coordinator, preview  # noqa: F401

pytestmark = pytest.mark.qt


class _Broken(Handshake):
    """A helper handshake whose cancel fails, as a vanished helper's does."""

    def cancel(self):
        self.calls.append("cancel")
        raise OSError("the helper's handshake file is gone")


def test_the_worker_keeps_a_result_or_the_error_text():
    worker = consent._UninstallWorker("preview", lambda: {"installations": []})
    worker.run()
    assert (worker.result, worker.error) == ({"installations": []}, None)

    def refuse():
        raise PermissionError("Administrator access is required")

    failed = consent._UninstallWorker("prepare", refuse)
    failed.run()
    assert (failed.result, failed.error) == (None, "Administrator access is required")


def test_a_late_plan_whose_helper_cannot_be_disarmed_still_releases_the_window(qtbot, monkeypatch):
    window, owner = coordinator(qtbot)
    handshake = _Broken()
    plan = {"platform": "windows"}
    monkeypatch.setattr(install_cleanup, "_WindowsPipHandshake", lambda received: handshake)
    owner.worker = SimpleNamespace(operation="prepare", result=plan, error=None)
    owner.stopped.set()
    owner._finished()
    assert handshake.calls == ["cancel"]
    assert window.isVisible() and window._uninstall_coordinator is None


def test_a_declined_confirmation_prepares_nothing(qtbot, monkeypatch, preview):
    window, owner = coordinator(qtbot)
    started = []
    monkeypatch.setattr(consent._UninstallDialog, "exec", lambda dialog: QDialog.Rejected)
    monkeypatch.setattr(owner, "_work", lambda operation, function: started.append(operation))
    owner.worker = SimpleNamespace(operation="preview", result=preview, error=None)
    owner._finished()
    assert started == [] and owner.stopped.is_set()
    assert window._uninstall_coordinator is None


def _confirmed_prepare(qtbot, monkeypatch, preview):
    """Accept the popup and return the preparation closure handed to the worker."""
    window, owner = coordinator(qtbot)
    started = []
    monkeypatch.setattr(consent._UninstallDialog, "exec", lambda dialog: QDialog.Accepted)
    monkeypatch.setattr(owner, "_work", lambda operation, function: started.append((operation, function)))
    owner.worker = SimpleNamespace(operation="preview", result=preview, error=None)
    owner._finished()
    [(operation, prepare)] = started
    assert operation == "prepare"
    return window, owner, prepare


def test_preparation_runs_the_helper_once_for_the_confirmed_choice(qtbot, monkeypatch, preview):
    window, owner, prepare = _confirmed_prepare(qtbot, monkeypatch, preview)
    calls = []
    plan = {"platform": "linux"}
    monkeypatch.setattr(install_cleanup, "_start_uninstall_helper",
                        lambda shown, choice: calls.append((shown, choice)) or plan)
    assert prepare() is plan
    assert calls[0][0] is preview and calls[0][1]["installations"] == ["current"]


def test_preparation_cancelled_before_it_starts_runs_no_helper(qtbot, monkeypatch, preview):
    window, owner, prepare = _confirmed_prepare(qtbot, monkeypatch, preview)
    monkeypatch.setattr(install_cleanup, "_start_uninstall_helper",
                        lambda *args: pytest.fail("a cancelled preparation started the helper"))
    owner.stopped.set()
    assert prepare() is None


@pytest.mark.parametrize("platform, kind", [("linux", "_FrozenUpdateHandshake"),
                                            ("windows", "_WindowsPipHandshake")])
def test_a_helper_ready_after_cancel_is_disarmed_before_it_is_returned(qtbot, monkeypatch, preview, platform, kind):
    window, owner, prepare = _confirmed_prepare(qtbot, monkeypatch, preview)
    handshake = Handshake()
    plan = {"platform": platform}

    def start(shown, choice):
        owner.stopped.set()
        return plan

    monkeypatch.setattr(install_cleanup, "_start_uninstall_helper", start)
    monkeypatch.setattr(install_cleanup, kind, lambda received: handshake if received is plan else None)
    assert prepare() is plan
    assert handshake.calls == ["cancel"]


def test_a_helper_that_is_not_ready_yet_is_polled_again(qtbot):
    window, owner = coordinator(qtbot)
    owner.handshake = Handshake()
    owner.handshake.status = lambda: None
    owner.timer.start()
    owner._poll()
    assert owner.timer.isActive() and owner.handshake.calls == []
    assert window.isVisible()
    owner.timer.stop()


def test_a_helper_reporting_an_error_stops_uninstall_and_explains_it(qtbot, monkeypatch):
    window, owner = coordinator(qtbot)
    owner.handshake = Handshake()
    owner.handshake.status = lambda: {"state": "error", "error": "the installer is in use"}
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, text: messages.append((title, text)))
    owner._poll()
    assert messages == [("Uninstall stopped", "the installer is in use")]
    assert owner.handshake.calls == ["cancel"]
    assert window.isVisible() and window._uninstall_coordinator is None


def test_a_refusal_while_the_window_closes_shows_no_message(qtbot, monkeypatch):
    window, owner = coordinator(qtbot)
    window._closing = True
    messages = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    owner._fail("the helper could not start")
    assert messages == [] and owner.stopped.is_set()


def test_an_uninstall_close_without_a_helper_is_not_approved(qtbot):
    window, owner = coordinator(qtbot)
    owner.closing_for_uninstall = True
    assert owner.approve_close() is False
    owner.handshake = Handshake()
    owner.stopped.set()
    assert owner.approve_close() is False
    assert owner.handshake.calls == [] and not owner.approved


def test_a_helper_that_cannot_be_cancelled_does_not_stop_the_release(qtbot):
    window, owner = coordinator(qtbot)
    owner.handshake = _Broken()
    owner.cancel()
    assert owner.handshake.calls == ["cancel"]
    assert owner.finished and window._uninstall_coordinator is None
    owner.cancel()
    assert owner.handshake.calls == ["cancel", "cancel"]


def test_a_coordinator_whose_window_moved_on_leaves_the_new_one_in_place(qtbot):
    window, owner = coordinator(qtbot)
    other = object()
    window._uninstall_coordinator = other
    owner.cancel()
    assert owner.finished and window._uninstall_coordinator is other


def test_a_coordinator_whose_dialog_is_already_deleted_finishes_quietly(qtbot):
    import shiboken6

    window, owner = coordinator(qtbot)
    shiboken6.delete(owner.timer)
    owner._finish()
    assert not owner.finished
    assert window._uninstall_coordinator is owner


@pytest.mark.parametrize("state", ["none", "closing", "already"])
def test_uninstall_does_not_start_twice_or_on_a_closing_window(qtbot, monkeypatch, state):
    monkeypatch.setattr(consent, "_UninstallCoordinator", lambda *args: pytest.fail("must not start"))
    if state == "none":
        assert consent._open_uninstall(None) is False
        return
    window = Window()
    qtbot.addWidget(window)
    if state == "closing":
        window._closing = True
    else:
        window._uninstall_coordinator = object()
    assert consent._open_uninstall(window) is False


def test_uninstall_starts_when_the_update_worker_is_already_gone(qtbot, monkeypatch):
    window = Window()
    qtbot.addWidget(window)

    class Retired:
        def isRunning(self):
            raise RuntimeError("Internal C++ object already deleted")

    started = []
    window._update_worker = Retired()
    monkeypatch.setattr(consent._UninstallCoordinator, "start", lambda owner: started.append(owner))
    assert consent._open_uninstall(window) is True
    owner = window._uninstall_coordinator
    assert started == [owner]
    assert window._approve_uninstall_close == owner.approve_close
    owner.cancel()
    assert window._uninstall_coordinator is None
