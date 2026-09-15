"""The in-app update finds old copies, shows them, deletes them, then installs.

Features item 416. Help -> Check for updates used to upgrade only the
environment spaCR was running from, so an older desktop install elsewhere --
or the desktop install being run -- was never removed. The handlers in
``MainWindow`` now run the sequence explicitly: find old spaCR files, delete
them, install the new version. The logic lives in ``spacr.install_cleanup``;
these tests pin only the order the window drives it in, and use a stand-in
window, fake records and a fake remover, so nothing on this computer is found
or deleted.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QCheckBox, QDialog, QLabel, QWidget  # noqa: E402

from spacr import install_cleanup, updater  # noqa: E402
from spacr.qt import app as qt_app  # noqa: E402


class _Messages:
    """Every message box the update handlers show, recorded instead."""

    def __init__(self, monkeypatch):
        self.warnings = []
        self.informations = []
        monkeypatch.setattr(
            qt_app.QMessageBox, "warning",
            staticmethod(lambda _p, title, text, *a, **k:
                         self.warnings.append((title, text))))
        monkeypatch.setattr(
            qt_app.QMessageBox, "information",
            staticmethod(lambda _p, title, text, *a, **k:
                         self.informations.append((title, text))))
        monkeypatch.setattr(
            qt_app.QMessageBox, "question",
            staticmethod(lambda *a, **k: qt_app.QMessageBox.Yes))


class _Bar:
    def showMessage(self, *args):
        pass


class _Window:
    """Just enough of ``MainWindow`` for its update handlers."""

    _on_update_check_done = qt_app.MainWindow._on_update_check_done
    _on_old_installs_found = qt_app.MainWindow._on_old_installs_found
    _on_update_sequence_done = qt_app.MainWindow._on_update_sequence_done

    def __init__(self, ticked=()):
        self._closing = False
        self._update_version = "9.9.9"
        self._ticked = ticked
        self.events = []
        self.workers = []
        self.closed = False

    def statusBar(self):
        return _Bar()

    def _confirm_old_installs(self, records):
        self.events.append("shown")
        return self._ticked

    def _start_update_worker(self, operation, fn, on_done):
        self.workers.append(operation)
        on_done(fn())

    def _on_upgrade_done(self, result):
        self.events.append(("upgrade done", result))

    def close(self):
        self.closed = True


def _record(tmp_path, kind, name, running=False):
    return install_cleanup.InstallRecord(
        kind=kind, layout="test", platform="linux", root=str(tmp_path / name),
        version="1.5.0.1", running=running)


def _fake_remover(window, monkeypatch, fail_on=()):
    def remove(record, ticked, keep, system):
        window.events.append(("delete", record.kind, ticked))
        failed = [(record.root, "in use")] if record.kind in fail_on else []
        return install_cleanup.RemovalReport(record, failed=failed)

    monkeypatch.setattr(install_cleanup, "remove_install", remove)
    monkeypatch.setattr(updater, "run_pip_upgrade",
                        lambda: window.events.append("install") or (0, "ok"))


def test_accepting_an_update_starts_by_finding_old_copies(monkeypatch):
    _Messages(monkeypatch)
    window = _Window()
    started = []
    window._start_update_worker = lambda op, fn, done: started.append((op, fn))
    info = updater.UpdateInfo(installed_version="1.5.0.1",
                              latest_release="9.9.9", nightly_sha=None)

    window._on_update_check_done(info)

    assert started == [("find", install_cleanup.find_old_installs)]
    assert window.events == [], "nothing is deleted or installed before the list"


def test_the_list_is_shown_then_old_copies_deleted_then_the_update_installed(
        monkeypatch, tmp_path):
    messages = _Messages(monkeypatch)
    old = _record(tmp_path, "installer", "old")
    env = _record(tmp_path, "environment", "env")
    window = _Window(ticked=(env.root,))
    _fake_remover(window, monkeypatch)

    window._on_old_installs_found([old, env])

    assert window.events == [
        "shown", ("delete", "installer", False), ("delete", "environment", True),
        "install", ("upgrade done", (0, "ok"))]
    assert window.workers == ["upgrade"] and not messages.warnings


def test_nothing_is_installed_when_an_old_copy_could_not_be_removed(
        monkeypatch, tmp_path):
    messages = _Messages(monkeypatch)
    window = _Window()
    _fake_remover(window, monkeypatch, fail_on=("installer",))

    window._on_old_installs_found([_record(tmp_path, "installer", "old")])

    assert "install" not in window.events
    assert "stopped before installing" in messages.warnings[0][1]
    assert "in use" in messages.warnings[0][1]


def test_cancelling_the_list_deletes_and_installs_nothing(monkeypatch, tmp_path):
    _Messages(monkeypatch)
    window = _Window(ticked=None)
    _fake_remover(window, monkeypatch)

    window._on_old_installs_found([_record(tmp_path, "installer", "old")])

    assert window.events == ["shown"] and window.workers == []


def test_with_nothing_old_to_show_the_update_goes_straight_to_installing(
        monkeypatch, tmp_path):
    _Messages(monkeypatch)
    window = _Window()
    _fake_remover(window, monkeypatch)
    here = _record(tmp_path, "environment", "this-env", running=True)

    window._on_old_installs_found([here])

    assert "shown" not in window.events
    assert window.events[-2:] == ["install", ("upgrade done", (0, "ok"))]


def test_a_running_desktop_copy_hands_deleting_and_installing_to_a_helper(
        monkeypatch, tmp_path):
    messages = _Messages(monkeypatch)
    window = _Window()
    _fake_remover(window, monkeypatch)
    handed = []
    monkeypatch.setattr(
        install_cleanup, "start_update_helper",
        lambda records, version, ticked: handed.append((records, version, ticked))
        or {"command": ["python", "install_cleanup.py", "run-plan"], "error": None})
    running = _record(tmp_path, "installer", "desktop", running=True)

    window._on_old_installs_found([running])

    assert handed == [([running], "9.9.9", ())]
    assert window.workers == [] and "install" not in window.events
    assert window.closed
    assert "will close" in messages.informations[0][1]


def test_no_helper_means_the_window_stays_open_and_says_why(monkeypatch, tmp_path):
    messages = _Messages(monkeypatch)
    window = _Window()
    monkeypatch.setattr(
        install_cleanup, "start_update_helper",
        lambda records, version, ticked: {"command": None,
                                          "error": "no Python outside"})

    window._on_old_installs_found([_record(tmp_path, "installer", "desktop",
                                           running=True)])

    assert not window.closed
    assert "no Python outside" in messages.warnings[0][1]


@pytest.mark.parametrize("accept", [True, False])
def test_the_dialog_lists_old_copies_and_ticks_only_your_environments(
        qapp, monkeypatch, tmp_path, accept):
    old = _record(tmp_path, "installer", "old-desktop")
    env = _record(tmp_path, "environment", "your-env")
    here = _record(tmp_path, "environment", "running-env", running=True)
    checkout = _record(tmp_path, "checkout", "source")
    seen = {}

    def fake_exec(dialog, *args, **kwargs):
        seen["labels"] = [w.text() for w in dialog.findChildren(QLabel)]
        boxes = dialog.findChildren(QCheckBox)
        seen["boxes"] = [box.text() for box in boxes]
        assert not any(box.isChecked() for box in boxes), "nothing ticked by default"
        boxes[0].setChecked(True)
        return QDialog.Accepted if accept else QDialog.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    parent = QWidget()
    try:
        ticked = qt_app.MainWindow._confirm_old_installs(
            parent, [old, env, here, checkout])
    finally:
        parent.deleteLater()

    assert any(old.root in text for text in seen["labels"])
    assert len(seen["boxes"]) == 1 and env.root in seen["boxes"][0]
    shown = " ".join(seen["labels"] + seen["boxes"])
    assert here.root not in shown and checkout.root not in shown
    assert ticked == ((env.root,) if accept else None)
