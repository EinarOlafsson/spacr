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
    "spaCR will close, remove the older copies, install {version} and start "
    "again.",
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
                   "error": None}])
    monkeypatch.setattr(install_cleanup, "start_update_helper",
                        lambda records, version, ticked: next(plans))
    running = _record(tmp_path, "installer", "desktop", running=True)

    window._on_old_installs_found([running])
    window._on_old_installs_found([running])

    assert messages.warnings[-1][1] == _marked(
        DIALOG_CAPTIONS[6].format(error=_marked(REASONS[-1])))
    assert messages.informations == [
        (_title, _marked(DIALOG_CAPTIONS[4].format(version="9.9.9")))]
