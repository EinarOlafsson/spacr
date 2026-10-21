"""The actual uninstall popup confirms its frozen scope before an approved close."""
import copy
import threading
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox, QPushButton, QWidget

from spacr import install_cleanup
from spacr.qt import install_consent as consent

pytestmark = pytest.mark.qt


@pytest.fixture
def preview(tmp_path):
    rows = []
    for key, current, selectable, kind in [("current", True, True, "installer"),
                                            ("shared", False, True, "environment"),
                                            ("checkout", False, False, "checkout")]:
        path = str(tmp_path / key)
        rows.append({"key": key, "current": current, "selectable": selectable,
                     "record": {"root": path, "version": "1.5.0.7", "kind": kind},
                     "paths": [path, path + "/spacr"], "reason": ""})
    return {"operation_id": "dialog-operation", "snapshot_id": "frozen-scope",
            "categories": list(install_cleanup._UNINSTALL_CATEGORIES), "installations": rows,
            "cleanup": [{"category": "models", "path": str(tmp_path / "models"), "selectable": True},
                        {"category": "models", "path": str(tmp_path / "shared-project"), "selectable": False}]}


def test_scope_defaults_to_current_and_all_categories_allow_independent_optout(qtbot, preview):
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    assert dialog.choices()["installations"] == ["current"]
    assert dialog.choices()["categories"] == list(install_cleanup._UNINSTALL_CATEGORIES)
    assert dialog._scope.itemText(1) == "All detected spaCR versions"
    dialog._scope.setCurrentIndex(1)
    dialog._category_items["models"].setCheckState(0, Qt.Unchecked)
    assert dialog.choices()["installations"] == ["current", "shared"]
    assert dialog.choices()["categories"] == [key for key in install_cleanup._UNINSTALL_CATEGORIES if key != "models"]
    rows = {dialog._installations.topLevelItem(i).text(0): dialog._installations.topLevelItem(i)
            for i in range(dialog._installations.topLevelItemCount())}
    assert rows[preview["installations"][1]["record"]["root"]].text(2) == "Remove spaCR package only"
    assert rows[preview["installations"][2]["record"]["root"]].text(2) == "Kept"
    models = dialog._category_items["models"]
    assert [models.child(i).text(0) for i in range(models.childCount())] == [row["path"] for row in preview["cleanup"]]
    assert models.child(1).text(1) == "Kept: ownership or path could not be verified"


def test_preview_is_frozen_and_cancel_is_the_default_action(qtbot, preview):
    original = copy.deepcopy(preview)
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    preview["operation_id"] = "changed"
    preview["installations"][0]["key"] = "another-installation"
    assert dialog.choices()["operation_id"] == original["operation_id"]
    assert dialog.choices()["installations"] == ["current"]
    buttons = dialog.findChild(QDialogButtonBox)
    assert next(button for button in buttons.buttons() if buttons.buttonRole(button) == QDialogButtonBox.RejectRole).isDefault()
    assert not dialog._remove.autoDefault()
    dialog.show()
    qtbot.keyClick(dialog, Qt.Key_Escape)
    assert dialog.result() == QDialog.Rejected


def test_no_removable_current_installation_disables_confirmation(qtbot, preview):
    preview["installations"][0]["selectable"] = False
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    assert not dialog._remove.isEnabled()
    dialog._scope.setCurrentIndex(1)
    assert dialog._remove.isEnabled()


class Window(QWidget):
    def __init__(self, veto=False):
        super().__init__()
        self.veto = veto
        self._closing = False

    def closeEvent(self, event):
        if self.veto:
            event.ignore()
            return
        callback = getattr(self, "_approve_uninstall_close", None)
        if callback and not callback():
            event.ignore()
            return
        self._closing = True
        event.accept()


def coordinator(qtbot, veto=False):
    window = Window(veto)
    qtbot.addWidget(window)
    window.show()
    owner = consent._UninstallCoordinator(window)
    window._uninstall_coordinator = owner
    window._approve_uninstall_close = owner.approve_close
    return window, owner


class Handshake:
    def __init__(self):
        self.calls = []

    def status(self):
        return {"state": "ready"}

    def check(self):
        self.calls.append("check")

    def approve(self):
        self.calls.append("approve")

    def cancel(self):
        self.calls.append("cancel")


@pytest.mark.parametrize("veto", [False, True])
def test_helper_approval_follows_the_main_window_close_decision(qtbot, veto):
    window, owner = coordinator(qtbot, veto)
    handshake = owner.handshake = Handshake()
    owner._poll()
    assert handshake.calls == (["check", "cancel"] if veto else ["check", "approve"])
    assert window.isVisible() is veto


def test_ordinary_close_cancels_uninstall_without_approval(qtbot):
    window, owner = coordinator(qtbot)
    handshake = owner.handshake = Handshake()
    assert window.close()
    assert handshake.calls == ["cancel"]


@pytest.mark.parametrize("ordinary_close", [False, True])
def test_discovery_runs_off_gui_thread_and_cancel_keeps_worker_alive(qtbot, monkeypatch, preview, ordinary_close):
    window, owner = coordinator(qtbot)
    started, release = threading.Event(), threading.Event()
    threads = []

    def discover(operation):
        threads.append(threading.get_ident())
        started.set()
        assert release.wait(5)
        return preview

    monkeypatch.setattr(install_cleanup, "_uninstall_preview", discover)
    monkeypatch.setattr(consent, "_UninstallDialog", lambda *args: pytest.fail("cancelled discovery must not open confirmation"))
    owner.start()
    qtbot.waitUntil(started.is_set)
    worker = owner.worker
    if ordinary_close:
        assert not window.close()
    else:
        owner.cancel()
    assert window._uninstall_coordinator is owner and worker in consent._UNINSTALL_WORKERS
    assert threads == [threads[0]] and threads[0] != threading.get_ident()
    release.set()
    qtbot.waitUntil(lambda: window._uninstall_coordinator is None)
    qtbot.waitUntil(lambda: worker not in consent._UNINSTALL_WORKERS)
    qtbot.waitUntil(lambda: window.isVisible() is not ordinary_close)


def test_cancel_after_preparation_return_disarms_the_late_plan(qtbot, monkeypatch):
    window, owner = coordinator(qtbot)
    handshake = Handshake()
    plan = {"platform": "linux"}
    monkeypatch.setattr(install_cleanup, "_FrozenUpdateHandshake", lambda received: handshake if received is plan else None)
    owner.worker = SimpleNamespace(operation="prepare", result=plan, error=None)
    owner.stopped.set()
    owner._finished()
    assert handshake.calls == ["cancel"]
    assert window.isVisible() and window._uninstall_coordinator is None


@pytest.mark.parametrize("denied", [False, True])
def test_confirmed_popup_prepares_in_worker_and_closes_only_after_readiness(qtbot, monkeypatch, preview, denied):
    window, owner = coordinator(qtbot)
    handshake = Handshake()
    calls, messages = [], []
    monkeypatch.setattr(install_cleanup, "_uninstall_preview", lambda operation: preview)

    def accept(dialog):
        dialog._category_items["models"].setCheckState(0, Qt.Unchecked)
        return QDialog.Accepted

    def prepare(shown, selection):
        calls.append((shown, selection, threading.get_ident()))
        if denied:
            raise PermissionError("Administrator access is required; spaCR will stay open")
        return {"platform": "linux"}

    monkeypatch.setattr(consent._UninstallDialog, "exec", accept)
    monkeypatch.setattr(install_cleanup, "_start_uninstall_helper", prepare)
    monkeypatch.setattr(install_cleanup, "_FrozenUpdateHandshake", lambda plan: handshake)
    monkeypatch.setattr(QMessageBox, "warning", lambda *args: messages.append(args))
    owner.start()
    qtbot.waitUntil(lambda: window._uninstall_coordinator is None if denied else not window.isVisible())
    assert len(calls) == 1 and calls[0][0] == preview
    assert calls[0][1]["installations"] == ["current"] and "models" not in calls[0][1]["categories"]
    assert calls[0][2] != threading.get_ident()
    assert handshake.calls == ([] if denied else ["check", "approve"])
    assert bool(messages) is denied
    assert window.isVisible() is denied


def test_update_in_progress_refuses_uninstall_before_starting_discovery(qtbot, monkeypatch):
    window = Window()
    qtbot.addWidget(window)
    window._update_worker = SimpleNamespace(isRunning=lambda: True)
    messages = []
    monkeypatch.setattr(QMessageBox, "information", lambda *args: messages.append(args))
    monkeypatch.setattr(consent, "_UninstallCoordinator", lambda *args: pytest.fail("must not start while updating"))
    assert not consent._open_uninstall(window)
    assert len(messages) == 1


def test_preferences_performance_button_opens_uninstall_for_owning_window(qtbot, monkeypatch):
    from spacr.qt import preferences

    window = Window()
    qtbot.addWidget(window)
    dialog = preferences.PreferencesDialog(window)
    qtbot.addWidget(dialog)
    button = dialog.findChild(QPushButton, "UninstallSpacrButton")
    assert button is not None
    received = []
    monkeypatch.setattr(consent, "_open_uninstall", received.append)
    button.click()
    qtbot.waitUntil(lambda: bool(received))
    assert received == [window]
    assert dialog.result() == QDialog.Rejected


def test_older_versions_only_ticks_everything_but_the_running_installation(qtbot, preview):
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    dialog._scope.setCurrentIndex(dialog._scope.findData("older"))
    assert dialog.choices()["scope"] == "older"
    assert dialog.choices()["installations"] == ["shared"]
    assert dialog._remove.isEnabled()


def test_changing_a_tick_becomes_an_explicit_ticked_selection(qtbot, preview):
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    dialog._scope.setCurrentIndex(dialog._scope.findData("all"))
    row, item = dialog._installation_items["current"]
    item.setCheckState(0, Qt.Unchecked)
    assert dialog._scope.currentData() == "custom"
    assert dialog.choices() == install_cleanup._uninstall_selection(
        dialog._preview, scope="custom", installations=["shared"])
    assert item.text(2) == "Kept"
    _, shared = dialog._installation_items["shared"]
    shared.setCheckState(0, Qt.Unchecked)
    assert not dialog._remove.isEnabled()


def test_an_unverified_installation_has_no_tick_box(qtbot, preview):
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    _, checkout = dialog._installation_items["checkout"]
    assert not checkout.flags() & Qt.ItemIsUserCheckable


def test_keeping_the_running_installation_starts_with_its_data_kept(qtbot, preview):
    dialog = consent._UninstallDialog(preview)
    qtbot.addWidget(dialog)
    assert dialog._kept_note.isHidden()
    dialog._scope.setCurrentIndex(dialog._scope.findData("older"))
    assert dialog.choices()["categories"] == []
    assert not dialog._kept_note.isHidden()
    dialog._category_items["logs"].setCheckState(0, Qt.Checked)
    _, shared = dialog._installation_items["shared"]
    shared.setCheckState(0, Qt.Unchecked)
    shared.setCheckState(0, Qt.Checked)
    assert dialog.choices()["categories"] == ["logs"]
    dialog._scope.setCurrentIndex(dialog._scope.findData("all"))
    assert dialog.choices()["categories"] == list(install_cleanup._UNINSTALL_CATEGORIES)
    assert dialog._kept_note.isHidden()


@pytest.mark.parametrize("phase", ["_maybe_offer_installation_cache_review", "_continue_installer_onboarding"])
def test_startup_onboarding_waits_for_uninstall_confirmation(qtbot, monkeypatch, phase):
    from spacr.qt.app import MainWindow

    window, owner = coordinator(qtbot)
    notices = []
    resumed = []
    window._maybe_offer_installation_cache_review = lambda: resumed.append(True)
    monkeypatch.setattr(consent, "maybe_show_anonymous_report_notice", lambda parent: notices.append(parent))
    getattr(MainWindow, phase)(window)
    assert window._installer_onboarding_pending
    assert notices == []
    owner.cancel()
    qtbot.waitUntil(lambda: resumed == [True])
    assert window._uninstall_coordinator is None
    assert not window._installer_onboarding_pending


@pytest.mark.parametrize("closing", ["approved", "close_after_cancel", "window_closing"])
def test_onboarding_does_not_resume_during_uninstall_shutdown(qtbot, closing):
    window, owner = coordinator(qtbot)
    resumed = []
    window._installer_onboarding_pending = True
    window._maybe_offer_installation_cache_review = lambda: resumed.append(True)
    if closing == "window_closing":
        window._closing = True
    else:
        setattr(owner, closing, True)
    owner._finish()
    from PySide6.QtWidgets import QApplication

    QApplication.processEvents()
    assert resumed == []
