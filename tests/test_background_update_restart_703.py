"""A private replacement is installed before Restart can close the old app."""
import json
import os
from pathlib import Path
import textwrap
import zipfile

import pytest

from spacr import install_cleanup as ic
from tests.test_macos_runtime_transaction_703 import (
    _installer, _snapshot, _wheel, installation,
)


@pytest.mark.parametrize("decision", ["restart", "cancel", "changed-runtime"])
def test_restart_is_offered_only_after_real_staged_install_and_verification(
        installation, tmp_path, decision):
    record, machine, prefix, _site = installation
    original = _snapshot(prefix)
    uv = _installer(tmp_path, _wheel(tmp_path / "new-wheels", "1.5.1.5"))
    command = [str(uv), "pip", "install", "--upgrade", "--python",
               record.python, "spacr"]
    offered = []
    launched = []

    def offer_restart():
        assert _snapshot(prefix) == original
        transactions = list(prefix.parent.glob(".spacr-runtime-update-*"))
        assert len(transactions) == 1
        transaction = transactions[0]
        journal = json.loads((transaction / "transaction.json").read_text())
        assert journal["state"] == "ready"
        staged_python = str(transaction / "runtime/bin/python")
        code, output = ic._run(ic._runtime_identity_command(staged_python, "1.5.1.5"))
        assert code == 0
        identity = json.loads(output)
        assert identity["verified"] and identity["version"] == "1.5.1.5"
        assert Path(identity["module"]).is_relative_to(transaction)
        assert not launched
        offered.append(identity)
        if decision == "cancel":
            raise RuntimeError("Restart cancelled")
        if decision == "changed-runtime":
            (prefix / "external-change.txt").write_text("Keep this newer change")

    options = dict(machine=machine, before_activate=offer_restart,
                   activate=launched.append)
    if decision == "restart":
        observed, _output = ic._transactional_runtime_update(
            record, "1.5.1.5", command, **options)
        assert observed == "1.5.1.5" and launched == ["1.5.1.5"]
        rollback = list(prefix.parent.glob(".spacr-runtime-update-*/runtime"))
        assert len(rollback) == 1 and _snapshot(rollback[0]) == original
    else:
        with pytest.raises(RuntimeError, match=("Restart cancelled" if decision == "cancel"
                                               else "desktop interpreter")):
            ic._transactional_runtime_update(record, "1.5.1.5", command, **options)
        assert not launched
        preserved = _snapshot(prefix)
        if decision == "changed-runtime":
            assert (prefix / "external-change.txt").read_text() == "Keep this newer change"
            preserved.pop("external-change.txt")
        assert preserved == original
        assert not list(prefix.parent.glob(".spacr-runtime-update-*"))
    assert len(offered) == 1
    assert not (prefix.parent / ".venv.spacr-update-lock").exists()


@pytest.mark.parametrize("outcome", ["ready", "wrong-version", "exit-before-window"])
def test_actual_restarted_process_must_verify_identity_and_reach_its_window(
        installation, tmp_path, monkeypatch, outcome):
    record, machine, prefix, _site = installation
    previous = _snapshot(prefix)
    wheel = _wheel(tmp_path / "gui-wheel", "1.5.1.5")
    events = tmp_path / "windows.json"
    monkeypatch.setenv("SPACR_TEST_LAUNCH_EVENTS", str(events))
    monkeypatch.setenv("PYTHONPATH", "/an/unrelated/old/spacr")
    import PySide6

    program = f'''import json, os, sys
from pathlib import Path
from spacr import install_cleanup as helper
launch = helper._begin_update_launch()
if {outcome!r} == 'exit-before-window':
    sys.exit(9)
sys.path.append({str(Path(PySide6.__file__).parent.parent)!r})
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget
app = QApplication([])
startup, main = QWidget(), QWidget()
startup.show()
seen = ['startup']
def show_main():
    startup.hide()
    main.show()
    seen.append('main')
    Path(os.environ['SPACR_TEST_LAUNCH_EVENTS']).write_text(json.dumps(seen))
    helper._finish_update_launch(launch)
    QTimer.singleShot(500, app.quit)
QTimer.singleShot(0, show_main)
app.exec()
'''
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    files["spacr/qt.py"] = ('if __name__ == "__main__":\n' + textwrap.indent(program, "    ")).encode()
    files["spacr/install_cleanup.py"] = Path(ic.__file__).read_bytes()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    uv = _installer(tmp_path, wheel)
    work = tmp_path / "helper"
    work.mkdir(mode=0o700)
    receipts = []

    def restart(version):
        receipt = ic._launch_updated_desktop(
            {"workdir": str(work)}, [record.python, "-I", "-m", "spacr.qt"],
            "1.5.1.4" if outcome == "wrong-version" else version, str(prefix))
        receipts.append(receipt)

    command = [str(uv), "pip", "install", "--upgrade", "--python", record.python, "spacr"]
    if outcome == "ready":
        observed, _ = ic._transactional_runtime_update(
            record, "1.5.1.5", command, machine=machine, activate=restart)
        assert observed == "1.5.1.5"
        assert json.loads(events.read_text()) == ["startup", "main"]
        assert receipts[0]["version"] == "1.5.1.5"
        assert receipts[0]["executable"] == record.python
        assert Path(receipts[0]["module"]).is_relative_to(prefix)
        assert receipts[0]["pid"] != os.getpid()
    else:
        with pytest.raises(RuntimeError, match="differs|exited"):
            ic._transactional_runtime_update(
                record, "1.5.1.5", command, machine=machine, activate=restart)
        assert _snapshot(prefix) == previous
        assert not receipts and not events.exists()
    message, = work.glob("launch-*/launch-result.json")
    result = json.loads(message.read_text())
    assert result["state"] == {"ready": "ready", "wrong-version": "error",
                               "exit-before-window": "starting"}[outcome]


def test_unconfirmed_live_desktop_retains_both_runtimes_until_its_own_exit(installation, tmp_path):
    record, machine, prefix, _site = installation
    prefix.chmod(0o700)
    prefix.parent.chmod(0o700)
    previous = _snapshot(prefix)
    uv = _installer(tmp_path, _wheel(tmp_path / "new-wheels", "1.5.1.5"))
    command = [str(uv), "pip", "install", "--upgrade", "--python", record.python, "spacr"]

    def still_running(_version):
        raise ic._UpdateLaunchStillRunning(43210)

    with pytest.raises(ic._UpdateLaunchStillRunning):
        ic._transactional_runtime_update(record, "1.5.1.5", command,
                                         machine=machine, activate=still_running)
    transaction, = prefix.parent.glob(".spacr-runtime-update-*")
    state = json.loads((transaction / "transaction.json").read_text())
    assert state["state"] == "launch_unconfirmed" and state["launch_pid"] == 43210
    assert _snapshot(transaction / "runtime") == previous
    assert ic._retiring_runtime_identity(str(prefix))["version"] == "1.5.1.5"
    with pytest.raises(ValueError, match="unconfirmed desktop"):
        ic._recover_runtime_update(str(transaction), 43211, machine=machine, wait=lambda pid: True)
    with pytest.raises(RuntimeError, match="did not close"):
        ic._recover_runtime_update(str(transaction), 43210, machine=machine, wait=lambda pid: False)
    restored = ic._recover_runtime_update(str(transaction), 43210, machine=machine, wait=lambda pid: True)
    assert restored["state"] == "rolled_back"
    assert _snapshot(prefix) == previous


def test_launch_handshake_accepts_private_macos_temp_ancestor_alias(tmp_path):
    parent = tmp_path / "private"
    parent.mkdir()
    work = parent / "helper"
    work.mkdir(mode=0o700)
    alias = tmp_path / "var"
    alias.symlink_to(parent, target_is_directory=True)
    handshake = ic._UpdateLaunchHandshake(str(alias / "helper"), "a" * 64)
    assert handshake.root == str(work)
    direct_alias = parent / "linked-helper"
    direct_alias.symlink_to(work, target_is_directory=True)
    with pytest.raises(ValueError, match="private owner directory"):
        ic._UpdateLaunchHandshake(str(direct_alias), "a" * 64)


def test_ready_receipt_cannot_hide_an_already_exited_desktop(tmp_path):
    from types import SimpleNamespace

    def exited(argv, environment, _cwd):
        root, binding = json.loads(environment["SPACR_UPDATE_LAUNCH"])
        handshake = ic._UpdateLaunchHandshake(root, binding)
        handshake._write("launch-result.json", "ready", details={
            "version": "1.5.1.5", "executable": argv[0], "pid": 43210,
            "module": str(tmp_path / "spacr/__init__.py")})
        return SimpleNamespace(pid=43210, poll=lambda: 9)

    with pytest.raises(RuntimeError, match="exited"):
        ic._launch_updated_desktop({"workdir": str(tmp_path)},
                                    [str(tmp_path / "python")], "1.5.1.5",
                                    str(tmp_path), spawn=exited)
