"""Updates prepare a verified replacement before asking to restart the GUI."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QCheckBox, QDialog, QLabel, QPushButton, QWidget,
)

from spacr import install_cleanup, updater  # noqa: E402
from spacr.qt import app as qt_app  # noqa: E402


class _Messages:
    """Every message box the update handlers show, recorded instead."""

    def __init__(self, monkeypatch):
        monkeypatch.setattr(updater, "editable_install_location", lambda: None)
        self.warnings = []
        self.informations = []
        self.restart_confirmation = qt_app.MainWindow._confirm_update_restart
        monkeypatch.setattr(qt_app.MainWindow, "_confirm_update_restart",
                            lambda window, message: qt_app.QMessageBox.information(
                                window, "Updates", message,
                                qt_app.QMessageBox.Ok | qt_app.QMessageBox.Cancel))
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
    _on_update_plan_prepared = qt_app.MainWindow._on_update_plan_prepared
    _on_update_sequence_done = qt_app.MainWindow._on_update_sequence_done
    _removal_reason_text = staticmethod(qt_app.MainWindow._removal_reason_text)

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


@pytest.fixture(params=["macos-frozen-v1", "macos-online-uv-v1", "pip-private-posix-v1", "pip-private-windows-v1", "linux-deb-v1", "windows-frozen-nsis-v1"])
def frozen_window(qtbot, monkeypatch, tmp_path, request):
    from types import SimpleNamespace
    from PySide6.QtWidgets import QMainWindow
    from spacr.qt import bridge

    class _FrozenWindow(qt_app.MainWindow):
        """Use the real closeEvent without building unrelated desktop screens."""

        resizeEvent = QMainWindow.resizeEvent
        changeEvent = QMainWindow.changeEvent

        def __init__(self):
            QMainWindow.__init__(self)
            self._closing = False
            self._screens = {}
            self._update_version = "1.5.1.1"
            self._stack = SimpleNamespace(currentWidget=lambda: SimpleNamespace(
                app_key="mask", _settings_model=SimpleNamespace(collect=lambda: {"src": "/private-images"})))

        def _confirm_old_installs(self, records):
            pytest.fail("Updates must not prompt to remove installations")

        def _release_update_workers(self):
            assert (work / "approved.json").exists() or self._frozen_update_handshake is None
            events.append("teardown")

    work = tmp_path / "helper"
    work.mkdir(mode=0o700)
    pip = request.param in {"pip-private-posix-v1", "pip-private-windows-v1"}
    windows = request.param in {"pip-private-windows-v1", "windows-frozen-nsis-v1"}
    frozen_windows = request.param == "windows-frozen-nsis-v1"
    if windows:
        monkeypatch.setattr(install_cleanup, "_windows_pip_acl", lambda path: {
            "owner": "S-1-5-21-1001", "user": "S-1-5-21-1001", "has_dacl": True,
            "rules": [{"sid": "S-1-5-21-1001", "type": 0, "rights": 0x1f01ff, "propagation": 0}]})
    online = request.param == "macos-online-uv-v1"
    debian = request.param == "linux-deb-v1"
    record = install_cleanup.InstallRecord(
        "environment" if pip else "installer", "venv" if pip else "windows-offline" if frozen_windows else "linux-deb" if debian else "macos-runtime" if online else "macos-app",
        "windows" if windows else "linux" if pip or debian else "macos", str(tmp_path / ("venv" if pip else "spaCR" if frozen_windows else "deb-runtime" if debian else "Application Support" if online else "spaCR.app")), running=True,
        registrations=("deb:spacr",) if debian else ())
    monkeypatch.setattr(install_cleanup, "_macos_online_update_record",
                        lambda records: record if online else None)
    plan = {"adapter": request.param, "command": ["inert-helper"],
            "version": "1.5.1.1", "pid": 123, "workdir": str(work),
            "records": [install_cleanup.asdict(record)], "ticked": [],
            "handshake": {"schema": 1, "token": "a" * 64,
                          "expires": install_cleanup.time.time() + 3600}}
    events = []
    messages = _Messages(monkeypatch)
    monkeypatch.setattr(bridge, "registry", lambda: SimpleNamespace(
        cancel_all=lambda **kwargs: events.append("veto checks") or []))
    monkeypatch.setattr(qt_app, "QApplication", SimpleNamespace(
        instance=lambda: SimpleNamespace(quit=lambda: events.append("quit"))))
    monkeypatch.setattr(install_cleanup, "start_update_helper", lambda *args, **kwargs: plan)
    window = _FrozenWindow()
    qtbot.addWidget(window)
    window.show()
    window._on_old_installs_found([record])
    qtbot.waitUntil(lambda: getattr(window, "_frozen_update_handshake", None) is not None)
    handshake = window._frozen_update_handshake
    yield SimpleNamespace(window=window, handshake=handshake, plan=plan,
                          work=work, events=events, messages=messages)
    window._cancel_frozen_update()


@pytest.mark.parametrize("failure", ["refused", "timeout", "wrong-plan"])
def test_frozen_preparation_failure_keeps_the_window_alive(frozen_window, monkeypatch, failure):
    state = frozen_window
    assert state.window.isVisible()
    assert not state.messages.informations
    if failure == "refused":
        state.handshake.failed("the official DMG has no positive published digest")
    elif failure == "timeout":
        monkeypatch.setattr(install_cleanup.time, "time", lambda: state.handshake.expires + 1)
    else:
        other = dict(state.plan, version="another version")
        type(state.handshake)(other).ready()
    state.window._poll_frozen_update()
    assert state.window.isVisible() and not state.window._closing
    assert state.messages.warnings
    assert state.events == []
    assert not (state.work / "approved.json").exists()
    assert (state.work / "cancelled.json").exists()


def test_unknown_authenticated_readiness_keeps_the_window_alive(frozen_window):
    state = frozen_window
    state.handshake._write("readiness.json", "unexpected")
    state.window._poll_frozen_update()
    assert state.window.isVisible() and not state.window._closing
    assert state.messages.warnings
    assert not state.messages.informations
    assert state.events == []
    assert not (state.work / "approved.json").exists()
    assert (state.work / "cancelled.json").exists()
    assert not getattr(state.window, "_restart_after_update", False)
    with pytest.raises(RuntimeError, match="cancelled"):
        state.handshake.wait_for_shutdown(
            lambda pid: pytest.fail("unknown readiness authorized shutdown"))


def test_background_preparation_can_be_cancelled_without_closing_spacr(frozen_window, qtbot):
    from PySide6.QtCore import Qt

    state = frozen_window
    progress = state.window._update_progress
    assert progress.isVisible()
    button, = progress.findChildren(QPushButton)
    qtbot.mouseClick(button, Qt.LeftButton)
    assert state.window.isVisible() and not state.window._closing
    assert state.window._frozen_update_handshake is None
    assert (state.work / "cancelled.json").exists()
    assert not (state.work / "approved.json").exists()
    assert state.events == []


@pytest.mark.parametrize("frozen_window", ["macos-online-uv-v1"], indirect=True)
@pytest.mark.parametrize("decision", [True, False, "error", "screen-veto"])
def test_uninstall_approval_occurs_only_after_screen_vetoes(frozen_window, qtbot, decision):
    state = frozen_window
    calls = []

    def approve():
        calls.append("approval")
        if decision == "error":
            raise RuntimeError("uninstall helper refused approval")
        return decision is True

    state.window._approve_uninstall_close = approve
    if decision == "screen-veto":
        class Veto(QWidget):
            def closeEvent(self, event):
                event.ignore()
        screen = Veto()
        qtbot.addWidget(screen)
        state.window._screens = {"mask": screen}
    closed = state.window.close()
    assert closed is (decision is True)
    assert calls == ([] if decision == "screen-veto" else ["approval"])
    assert ("quit" in state.events) is (decision is True)
    if decision is not True:
        assert state.window.isVisible() and not state.window._closing
    if decision == "error":
        assert state.messages.warnings[-1][0] == "Uninstall stopped"


def test_an_uninstall_coordinator_blocks_starting_an_update(monkeypatch):
    window = _Window()
    window._uninstall_coordinator = object()
    window._start_update_worker = lambda *args: pytest.fail("Update must wait for uninstall")
    qt_app.MainWindow._check_for_updates(window)
    assert not window.workers


def test_frozen_readiness_cancel_rejects_a_late_ready_message(frozen_window, monkeypatch):
    state = frozen_window
    state.handshake.ready()
    monkeypatch.setattr(qt_app.QMessageBox, "information", lambda *a, **k: qt_app.QMessageBox.Cancel)
    state.window._poll_frozen_update()
    state.handshake._write("readiness.json", "ready")
    state.window._poll_frozen_update()
    assert state.window.isVisible() and not state.window._closing
    assert not (state.work / "approved.json").exists()
    with pytest.raises(RuntimeError, match="cancelled"):
        state.handshake.wait_for_shutdown(lambda pid: pytest.fail("late readiness authorized shutdown"))


def test_ordinary_close_during_preparation_never_authorizes_replacement(frozen_window):
    state = frozen_window
    assert state.window.close()
    assert (state.work / "cancelled.json").exists()
    assert not (state.work / "approved.json").exists()


@pytest.mark.parametrize("failure", ["busy-worker", "expiry-during-close", "approval-write"])
def test_frozen_close_veto_or_approval_failure_preserves_the_gui(
        frozen_window, monkeypatch, failure):
    from types import SimpleNamespace
    from spacr.qt import bridge

    state = frozen_window
    state.handshake.ready()
    deadline = state.handshake.status()["expires"]
    monkeypatch.setattr(qt_app.QMessageBox, "information", lambda *a, **k: qt_app.QMessageBox.Ok)

    def drain(**kwargs):
        if failure == "busy-worker":
            return [SimpleNamespace(app_key="measure")]
        if failure == "expiry-during-close":
            monkeypatch.setattr(install_cleanup.time, "time", lambda: deadline + 1)
        return []

    monkeypatch.setattr(bridge, "registry", lambda: SimpleNamespace(cancel_all=drain))
    write = state.handshake._write

    def publish(name, *args, **kwargs):
        if failure == "approval-write" and name == "approved.json":
            raise OSError("approval disk write failed")
        return write(name, *args, **kwargs)

    monkeypatch.setattr(state.handshake, "_write", publish)
    state.window._poll_frozen_update()
    assert state.window.isVisible() and not state.window._closing
    assert not (state.work / "approved.json").exists()
    assert (state.work / "cancelled.json").exists()
    assert "teardown" not in state.events and "quit" not in state.events
    assert state.messages.warnings


def test_frozen_accepted_close_approves_after_vetoes_before_teardown(frozen_window, monkeypatch):
    state = frozen_window
    state.handshake.ready()
    def accept(parent, title, message, buttons):
        if state.plan["adapter"] == "macos-online-uv-v1":
            assert "1.5.1.1 is ready" in message
            assert "Restart to use the update" in message
            assert "remove" not in message
        elif state.plan["adapter"] == "windows-frozen-nsis-v1":
            assert "updating its Windows application" in message
            assert "administrator authorization window may open" in message
            assert "previous application will be retained for rollback" in message
            assert "data and caches will be kept" in message
            assert "reopen after the update succeeds" in message
            assert "remove the older copies" not in message
        elif state.plan["adapter"] == "linux-deb-v1":
            assert "graphical administrator authorization" in message
            assert "previous package will be retained for rollback" in message
            assert "data and caches will be kept" in message
            assert "remove the older copies" not in message
        return qt_app.QMessageBox.Ok

    monkeypatch.setattr(qt_app.QMessageBox, "information", accept)
    approve = state.handshake.approve

    def approval():
        assert state.window.isVisible() and not state.window._closing
        assert state.events == ["veto checks"]
        approve()
        state.events.append("approved")

    monkeypatch.setattr(state.handshake, "approve", approval)
    state.window._poll_frozen_update()
    assert state.events == ["veto checks", "approved", "teardown", "quit"]
    assert not state.window.isVisible() and state.window._closing
    assert not getattr(state.window, "_restart_after_update", False)
    assert not (state.work / "cancelled.json").exists()
    state.handshake.wait_for_shutdown(lambda pid: True)



def test_update_confirmation_accepts_a_real_button_click(frozen_window, qtbot, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QMessageBox

    state = frozen_window
    state.handshake.ready()

    def accept(box):
        box.open()
        qtbot.waitUntil(box.isVisible)
        assert box.button(QMessageBox.Ok).text() == "Restart"
        qtbot.mouseClick(box.button(QMessageBox.Ok), Qt.LeftButton)
        return box.result()

    monkeypatch.setattr(qt_app.MainWindow, "_confirm_update_restart",
                        state.messages.restart_confirmation)
    monkeypatch.setattr(QMessageBox, "exec", accept)
    state.window._poll_frozen_update()
    assert (state.work / "approved.json").exists()
    assert not state.window.isVisible()
    assert state.events == ["veto checks", "teardown", "quit"]


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
                        lambda **kwargs: window.events.append("install") or (0, "ok"))


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


@pytest.mark.parametrize("old_copies", [False, True])
def test_unidentified_active_environment_never_deletes_or_installs(monkeypatch, tmp_path, old_copies):
    messages = _Messages(monkeypatch)
    window = _Window()
    _fake_remover(window, monkeypatch)
    monkeypatch.setattr(install_cleanup, "start_update_helper", lambda *args, **kwargs:
                        {"command": None, "error": "This environment needs its package manager; spaCR cannot replace it safely. Nothing was changed."})
    records = [_record(tmp_path, "installer", "old")] if old_copies else []
    window._on_old_installs_found(records)
    assert window.events == [] and window.workers == ["prepare"] and not window.closed
    assert "package manager" in messages.warnings[-1][1]


def test_pip_environment_handoff_keeps_every_other_copy(monkeypatch, tmp_path):
    messages = _Messages(monkeypatch)
    window = _Window()
    _fake_remover(window, monkeypatch)
    calls = []
    monkeypatch.setattr(install_cleanup, "start_update_helper", lambda records, version, ticked:
                        calls.append((records, version, ticked)) or {"command": None, "error": "preparation refused"})
    here = _record(tmp_path, "environment", "active", running=True)
    other = _record(tmp_path, "installer", "other")
    window._on_old_installs_found([here, other])
    assert calls == [([here, other], "9.9.9", ())]
    assert window.events == [] and window.workers == ["prepare"]
    assert not window.closed and messages.warnings


def test_an_update_does_not_offer_an_installer_retirement_list(monkeypatch, tmp_path):
    _Messages(monkeypatch)
    window = _Window(ticked=None)
    _fake_remover(window, monkeypatch)
    prepared = []
    monkeypatch.setattr(install_cleanup, "start_update_helper", lambda *args, **kwargs:
                        prepared.append(kwargs) or {"command": None, "error": "not supported"})
    window._on_old_installs_found([_record(tmp_path, "installer", "active", running=True)])
    assert window.events == [] and window.workers == ["prepare"]
    assert prepared == [{"ticked": ()}]
    assert not window.closed


def test_an_unverified_legacy_helper_cannot_close_the_running_desktop(
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
    assert window.workers == ["prepare"] and "install" not in window.events
    assert not window.closed
    assert "cannot stage a verified update" in messages.warnings[0][1]


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


# ---------------------------------------------------------------------------
# What the update shows is translatable
# ---------------------------------------------------------------------------

#: Every caption the update dialog and its messages show, each passed to
#: ``tr`` as a literal in ``spacr/qt/app.py`` so the catalog builder finds it.
DIALOG_CAPTIONS = (
    "Remove older spaCR copies",
    "These older copies of spaCR will be removed before the new version is "
    "installed:",
    "Environments you made. Tick one to uninstall spaCR from it; the "
    "environment itself is kept.",
    "Remove and update",
    "This installation cannot stage a verified update. Nothing was changed.",
    "The update stopped before installing, because an older copy of spaCR "
    "could not be removed:",
    "Upgrade unavailable: {error}",
)

#: The reasons ``spacr.install_cleanup`` gives that an in-app update can show,
#: written there as plain literals. The two administrator reasons are built
#: by ``_needs_admin`` and checked through it. Not here, because no in-app
#: update can show them: the reasons only a sandbox produces ("no registry to
#: change", "no package manager to ask", "refused: outside the computer being
#: cleaned") and the ones only the console and the logs print.
REASONS = (
    "in use or not permitted; close anything using it and try again",
    "not permitted; delete this registry entry by hand",
    "no Python found in it; run pip uninstall spacr in that environment",
    "refused: a shared folder, not a spaCR installation; remove spaCR from "
    "it by hand",
    "refused: not recognisably a spaCR installation; delete it by hand if "
    "it is one",
    "no Python outside the installation could be found to finish the "
    "update; run the new installer instead",
)
ADMIN_REASONS = (
    "needs administrator rights; delete it as an administrator",
    "needs administrator rights; run: {command}",
)


def _marked(text, language=None, **values):
    """A stand-in for ``tr`` that shows which text went through it."""
    return "\u00ab" + str(text).format(**values) + "\u00bb"


def test_every_caption_and_reason_the_update_shows_reaches_the_catalog_source():
    """A string the catalog builder cannot find stays English in every language."""
    from tools import build_i18n_catalogs as builder

    extracted = set(builder.extract_static_ui_sources())

    missing = [text for text in DIALOG_CAPTIONS + REASONS + ADMIN_REASONS
               if text not in extracted]
    assert missing == []


def test_the_translated_reasons_are_the_words_install_cleanup_returns():
    """A reason reworded in the module would silently stop being translated."""
    tree = ast.parse(Path(install_cleanup.__file__).read_text(encoding="utf-8"))
    literals = {node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)}

    assert [reason for reason in REASONS if reason not in literals] == []
    assert install_cleanup._needs_admin() == ADMIN_REASONS[0]
    assert install_cleanup._needs_admin("sudo dpkg -r python3-spacr") == (
        ADMIN_REASONS[1].format(command="sudo dpkg -r python3-spacr"))


def test_the_dialog_passes_its_title_captions_and_button_through_tr(
        qapp, monkeypatch, tmp_path):
    monkeypatch.setattr(qt_app, "tr", _marked)
    seen = {}

    def fake_exec(dialog, *args, **kwargs):
        seen["title"] = dialog.windowTitle()
        seen["labels"] = [w.text() for w in dialog.findChildren(QLabel)]
        seen["buttons"] = [b.text() for b in dialog.findChildren(QPushButton)]
        return QDialog.Rejected

    monkeypatch.setattr(QDialog, "exec", fake_exec)
    parent = QWidget()
    try:
        qt_app.MainWindow._confirm_old_installs(
            parent, [_record(tmp_path, "installer", "old"),
                     _record(tmp_path, "environment", "env")])
    finally:
        parent.deleteLater()

    assert seen["title"] == _marked(DIALOG_CAPTIONS[0])
    assert _marked(DIALOG_CAPTIONS[1]) in seen["labels"]
    assert _marked(DIALOG_CAPTIONS[2]) in seen["labels"]
    assert _marked(DIALOG_CAPTIONS[3]) in seen["buttons"]


def test_the_update_messages_translate_spacr_reasons_and_leave_others_as_they_came(
        monkeypatch, tmp_path):
    monkeypatch.setattr(qt_app, "tr", _marked)
    messages = _Messages(monkeypatch)
    window = _Window()
    old = _record(tmp_path, "installer", "old")
    admin = "needs administrator rights; run: sudo dpkg -r python3-spacr"

    window._on_update_sequence_done(([install_cleanup.RemovalReport(old, failed=[
        (old.root, REASONS[0]),
        (old.root, "Device or resource busy"),
        ("deb:python3-spacr", admin)])], None))

    [(_title, text)] = messages.warnings
    assert text.startswith(_marked(DIALOG_CAPTIONS[5]) + "\n\n")
    assert f"{old.root}: {_marked(REASONS[0])}" in text
    assert f"{old.root}: Device or resource busy" in text, (
        "an operating-system error is not spaCR's wording")
    assert f"deb:python3-spacr: {_marked(admin)}" in text

    plans = iter([{"command": None, "error": REASONS[-1]},
                  {"command": ["python", "install_cleanup.py", "run-plan"],
                   "version": "9.9.9", "error": None}])
    monkeypatch.setattr(install_cleanup, "start_update_helper",
                        lambda records, version, ticked: next(plans))
    running = _record(tmp_path, "installer", "desktop", running=True)

    window._on_old_installs_found([running])
    window._on_old_installs_found([running])

    assert messages.warnings[-2][1] == _marked(
        DIALOG_CAPTIONS[6].format(error=_marked(REASONS[-1])))
    assert messages.warnings[-1][1] == _marked(DIALOG_CAPTIONS[4])
    assert messages.informations == []


@pytest.mark.parametrize("frozen_window", ["pip-private-posix-v1", "pip-private-windows-v1", "linux-deb-v1", "windows-frozen-nsis-v1"], indirect=True)
def test_pip_handoff_preserves_current_settings_before_approved_close(frozen_window, monkeypatch):
    from spacr import restart_state
    state = frozen_window
    state.handshake.ready()
    monkeypatch.setattr(qt_app.QMessageBox, 'information', lambda *args: qt_app.QMessageBox.Ok)
    state.window._poll_frozen_update()
    assert restart_state.peek()['module'] == 'mask'
    assert restart_state.peek()['settings'] == {'src': '/private-images'}
    assert (state.work / 'approved.json').exists()
    restart_state.discard()


@pytest.mark.parametrize("frozen_window", ["pip-private-posix-v1", "pip-private-windows-v1", "linux-deb-v1", "windows-frozen-nsis-v1"], indirect=True)
def test_pip_handoff_refuses_unsaved_state_without_closing_or_approval(frozen_window, monkeypatch):
    from spacr import restart_state
    state = frozen_window
    state.handshake.ready()
    monkeypatch.setattr(qt_app.QMessageBox, 'information', lambda *args: qt_app.QMessageBox.Ok)
    monkeypatch.setattr(restart_state, 'save', lambda **kwargs: None)
    state.window._poll_frozen_update()
    assert state.window.isVisible()
    assert not (state.work / 'approved.json').exists()
    assert (state.work / 'cancelled.json').exists()
    assert 'could not be saved' in state.messages.warnings[-1][1]


@pytest.mark.parametrize("language", ["sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr"])
def test_private_pip_refusal_and_settings_failure_use_reviewed_locale(
        qapp, monkeypatch, language):
    """Real runtime lookup shows the exact reviewed message at both refusal points."""
    import json
    from types import SimpleNamespace
    from spacr import restart_state
    from spacr.qt import i18n

    reviews = json.loads((Path(__file__).resolve().parents[2] / "docs" / "i18n" /
                          "reviewed" / "runtime" / language /
                          "2026-10-14-n703-private-pip-handoff.json").read_text())
    expected = {row["source"]: row["translation"] for row in reviews["records"]}
    monkeypatch.setenv(i18n.ENV_LANGUAGE, language)
    reason = install_cleanup._PRIVATE_PIP_UPDATE_ERROR
    assert qt_app.MainWindow._removal_reason_text(reason) == expected[reason]
    seen = []
    monkeypatch.setattr(qt_app.QMessageBox, "warning", lambda _parent, _title, text: seen.append(text))
    monkeypatch.setattr(restart_state, "save", lambda **_values: None)
    window = SimpleNamespace(_stack=SimpleNamespace(currentWidget=lambda: None))
    assert qt_app.MainWindow._save_pip_update_restart(window) is False
    source = "The current settings could not be saved. The update was cancelled and this window will stay open."
    assert seen == [expected[source]]
    assert all(value != source for source, value in expected.items())


@pytest.mark.parametrize('frozen_window', ['pip-private-windows-v1', 'windows-frozen-nsis-v1'], indirect=True)
@pytest.mark.parametrize('closing', [False, True])
def test_windows_preparation_runs_off_gui_and_disarms_late_results(frozen_window, monkeypatch, qtbot, closing):
    import threading
    from PySide6.QtCore import QThread, QTimer

    state = frozen_window
    state.window._cancel_frozen_update()
    work = state.work.parent / 'second helper'
    work.mkdir(mode=0o700)
    plan = dict(state.plan, workdir=str(work), handshake=dict(state.plan['handshake'], token='b' * 64))
    entered, release = threading.Event(), threading.Event()
    threads = []
    main_thread = threading.get_ident()

    def prepare(*args, **kwargs):
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(3), 'The test did not release background preparation'
        return plan

    monkeypatch.setattr(install_cleanup, 'start_update_helper', prepare)
    record = install_cleanup.InstallRecord(**plan['records'][0])
    state.window._on_old_installs_found([record])
    worker = state.window._update_worker
    try:
        qtbot.waitUntil(entered.is_set)
        assert threads == [threads[0]] and threads[0] != main_thread
        assert state.window._frozen_update_handshake is None
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(threading.get_ident()))
        qtbot.waitUntil(lambda: bool(heartbeat))
        assert heartbeat == [main_thread] and not release.is_set()
        state.window._closing = closing
        release.set()
        if closing:
            qtbot.waitUntil(lambda: (work / 'cancelled.json').exists())
            assert not (work / 'approved.json').exists()
            assert state.window._frozen_update_handshake is None
        else:
            qtbot.waitUntil(lambda: state.window._frozen_update_handshake is not None)
            assert state.window._frozen_update_timer.thread() == QThread.currentThread()
            assert state.window._frozen_update_plan['workdir'] == str(work)
            assert not (work / 'approved.json').exists()
    finally:
        release.set()
        from shiboken6 import isValid
        qtbot.waitUntil(lambda: not isValid(worker) or not worker.isRunning())
        state.window._closing = False


@pytest.mark.parametrize('frozen_window', ['windows-frozen-nsis-v1'], indirect=True)
def test_windows_frozen_plan_without_authenticated_lease_never_closes(frozen_window):
    state = frozen_window
    state.window._cancel_frozen_update()
    plan = dict(state.plan)
    del plan['handshake']
    state.window._on_update_plan_prepared(plan)
    assert state.window.isVisible() and not state.window._closing
    assert state.window._frozen_update_handshake is None
    assert state.messages.warnings
    assert not (state.work / 'approved.json').exists()
    assert not getattr(state.window, '_restart_after_update', False)
