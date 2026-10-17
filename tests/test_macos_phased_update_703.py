"""Protected macOS preparation and recovery use fixed trusted code and authenticated phases."""
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shlex
import stat
import subprocess
import time
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_frozen_macos_update import _protected_script


@pytest.mark.parametrize("decision", ["confirmed", "cancel", "failed", "retain", "disconnect"])
def test_actual_pipe_bridge_preserves_or_restores_payload_at_fixed_checkpoints(tmp_path, monkeypatch, decision):
    payload = tmp_path / "payload"
    payload.write_text("old")
    journal = tmp_path / "journal"
    checkpoint = re.search(r"(?ms)^phase_checkpoint\(\) \{\n.*?^\}\n", ic._MACOS_PROTECTED_TRANSACTION).group()
    script = "set -eu\nphased=1\nphase=prepared\n"
    script += "payload=" + shlex.quote(str(payload)) + "\njournal=" + shlex.quote(str(journal)) + "\n"
    script += 'write_journal() { printf "%s" "$phase" > "$journal"; }\n'
    script += 'trap \'if [ "$phase" = installed ]; then printf old > "$payload"; fi\' EXIT\n'
    script += checkpoint + '\nphase_checkpoint prepared || exit 79\nprintf new > "$payload"\nphase=installed\n'
    script += 'phase_checkpoint activated || exit 79\nphase=committed\nwrite_journal\nprintf "committed\\t/Applications/.spacr-update.ABCDEF12\\tdetached\\n"\n'
    events = []
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, **kwargs: pid == 4242)

    class Phase:
        def _send(self, phase, **details):
            events.append(phase)
            assert payload.read_text() == ("old" if phase == "prepared" else "new")
            assert details["version"] == "1.5.1.5"

        def _receive(self, expected, **kwargs):
            if "activate" in expected:
                return {"phase": "cancel" if decision == "cancel" else "activate"}
            if decision == "disconnect":
                raise OSError("coordinator disconnected")
            return {"phase": decision}

    if decision == "confirmed":
        output = ic._macos_phased_transaction(script, Phase(), "1.5.1.5", 4242)
        assert output == "committed\t/Applications/.spacr-update.ABCDEF12\tdetached\n"
        assert journal.read_text() == "committed"
    else:
        with pytest.raises(RuntimeError):
            ic._macos_phased_transaction(script, Phase(), "1.5.1.5", 4242)
    assert events == (["prepared"] if decision == "cancel" else ["prepared", "activated"])
    assert payload.read_text() == ("old" if decision in {"cancel", "failed"} else "new")
    if decision in {"retain", "disconnect"}:
        assert journal.read_text() == "desktop-unconfirmed"


@pytest.mark.parametrize("fault", [None, "owner", "writable", "link", "acl", "signature"])
def test_trusted_installed_helper_refuses_user_modification_before_authorization(monkeypatch, fault):
    helper = "/Applications/spaCR.app/Contents/Frameworks/spacr-update-helper"
    target = "/Applications/spaCR.app/Contents"
    native_os = ic.os
    operations = SimpleNamespace(**vars(native_os))
    operations.path = SimpleNamespace(**vars(native_os.path))
    operations.path.realpath = lambda path: path

    def details(path):
        mode = (stat.S_IFREG if path == helper else stat.S_IFDIR) | 0o755
        if path == "/Applications":
            mode |= 0o020
        uid = 1000 if fault == "owner" and path == target else 0
        if fault == "writable" and path == target:
            mode |= 0o020
        if fault == "link" and path == target:
            mode = stat.S_IFLNK | 0o777
        return SimpleNamespace(st_uid=uid, st_gid=80, st_mode=mode)

    operations.lstat = details
    monkeypatch.setattr(ic, "os", operations)
    calls = []

    def native(argv):
        calls.append(argv)
        if argv[0] == "/usr/bin/codesign" and fault == "signature":
            raise RuntimeError("signature refused")
        return " 0: everyone allow write\n" if fault == "acl" and argv[-1] == target else ""

    if fault is None:
        assert ic._macos_frozen_trusted_helper(run=native) == helper
        assert calls[-1] == ["/usr/bin/codesign", "--verify", "--strict", helper]
    else:
        with pytest.raises((ValueError, RuntimeError)):
            ic._macos_frozen_trusted_helper(run=native)
    assert all(argv[0] in {"/bin/ls", "/usr/bin/codesign"} for argv in calls)


@pytest.mark.parametrize("supported", [True, False])
def test_authorization_requires_capability_and_quotes_private_descriptor_as_data(tmp_path, monkeypatch, supported):
    helper = "/Applications/spaCR.app/Contents/Frameworks/spacr-update-helper"
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: helper)
    monkeypatch.setattr(ic, "_macos_native", lambda argv: json.dumps({"schema": 1,
        "commands": ["apply-macos-root"] if supported else []}))
    descriptor = str(tmp_path / "O'Brien;$(touch NEVER_EXECUTED)" / "phase")
    arguments = ["--phase-channel", descriptor, "--desktop-pid", "4242"]
    if not supported:
        with pytest.raises(ValueError, match="does not support"):
            ic._macos_frozen_authorized_command("apply-macos-root", arguments)
        return
    command = ic._macos_frozen_authorized_command("apply-macos-root", arguments)
    script = _protected_script(command)
    line = script.splitlines()[-1]
    assert shlex.split(line)[-5:] == ["apply-macos-root", *arguments]
    assert '/private/var/root/spacr-helper.XXXXXXXX' in script
    assert '/bin/cp -pP "$helper" "$stage/helper"' in script
    assert '/usr/bin/codesign --verify --strict "$stage/helper"' in script
    assert "python" not in script.lower()
    assert not (tmp_path / "NEVER_EXECUTED").exists()
    subprocess.run(["/bin/sh", "-n"], input=script, text=True, check=True, timeout=5)


@pytest.mark.parametrize("code", [0, 6])
def test_success_cleans_only_the_finished_helper_stage_and_failure_retains_it(tmp_path, monkeypatch, code):
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: "/trusted/helper")
    monkeypatch.setattr(ic, "_macos_native", lambda argv: json.dumps({"schema": 1, "commands": ["apply-macos-root"]}))
    script = _protected_script(ic._macos_frozen_authorized_command("apply-macos-root", []))
    trap, = [line for line in script.splitlines() if line.startswith("trap ")]
    stage = tmp_path / "helper-stage"
    stage.mkdir()
    (stage / "helper").write_bytes(b"finished helper fixture")
    recovery = tmp_path / "application-recovery"
    recovery.mkdir()
    (recovery / "previous.app").write_bytes(b"retained previous application")
    result = subprocess.run(["/bin/sh", "-c", "stage=" + shlex.quote(str(stage)) + "\n" + trap + "\nexit " + str(code)],
        timeout=5)
    assert result.returncode == code
    assert stage.exists() is (code != 0)
    assert (recovery / "previous.app").read_bytes() == b"retained previous application"


def test_capability_and_privileged_refusal_do_not_import_application(monkeypatch, capsys):
    import builtins

    native_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "spacr" or name.startswith("spacr."):
            pytest.fail("privileged dispatch must not import application code")
        return native_import(name, *args, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(builtins, "__import__", guarded)
        assert ic._main(["native-phase-protocol"]) == 0
        assert json.loads(capsys.readouterr().out)["commands"] == ["apply-macos-root", "uninstall-macos-root"]
        assert ic._main(["apply-macos-root", "--version", "1.5.1.5", "--digest", "a" * 64,
            "--image", "/untrusted/spaCR-1.5.1.5.dmg", "--phase-channel", "/untrusted/descriptor",
            "--desktop-pid", "4242"]) == 6
        assert json.loads(capsys.readouterr().out)["state"] == "stopped"


@pytest.mark.parametrize("fault", [None, "stage-owner", "stage-writable", "unfrozen", "root-desktop", "other-desktop", "checksum"])
def test_privileged_dispatch_binds_root_copy_real_desktop_and_published_release(monkeypatch, fault):
    root = "/private/var/root/spacr-helper.ABCDEF12"
    helper = root + "/helper"
    desktop = "/Applications/spaCR.app/Contents/MacOS/spacr"
    operations = SimpleNamespace(**vars(ic.os))
    operations.path = SimpleNamespace(**vars(ic.os.path))
    operations.path.realpath = lambda path: path
    operations.geteuid = lambda: 0
    operations.lstat = lambda path: SimpleNamespace(st_uid=1000 if fault == "stage-owner" else 0,
        st_mode=0o755 if fault == "stage-writable" else 0o700, st_dev=1, st_ino=42)
    monkeypatch.setattr(ic, "os", operations)
    monkeypatch.setattr(ic, "sys", SimpleNamespace(executable=helper, frozen=fault != "unfrozen"))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (
        0 if fault == "root-desktop" else 1000, "/other/program" if fault == "other-desktop" else desktop))
    calls = []
    def digest(url):
        assert url == ic._RELEASE_DOWNLOAD + "/v1.5.1.5/spaCR-1.5.1.5.dmg"
        return "b" * 64 if fault == "checksum" else "a" * 64
    monkeypatch.setattr(ic, "_published_digest", digest)
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: calls.append("trusted-helper"))
    class Channel:
        def __init__(self, endpoint, pid, executable):
            assert (endpoint, pid, executable) == ("private-descriptor", 4242, desktop)
            calls.append("channel")
        def __enter__(self):
            return self
        def __exit__(self, *args):
            calls.append("closed")
    monkeypatch.setattr(ic, "_UpdatePhaseClient", Channel)
    def execute(script, channel, version, pid):
        assert script.startswith("phased=1\n")
        assert "target=/Applications/spaCR.app" in script
        assert "original_identity=1:42" in script
        assert (version, pid) == ("1.5.1.5", 4242)
        calls.append("fixed-transaction")
        return "owned fixture receipt"
    monkeypatch.setattr(ic, "_macos_phased_transaction", execute)
    arguments = ("1.5.1.5", "a" * 64, "/owned/spaCR-1.5.1.5.dmg", "private-descriptor", 4242)
    if fault is None:
        assert ic._macos_apply_root_transaction(*arguments) == "owned fixture receipt"
        assert calls == ["trusted-helper", "channel", "fixed-transaction", "closed"]
    else:
        with pytest.raises(ValueError):
            ic._macos_apply_root_transaction(*arguments)
        assert calls == []


@pytest.mark.parametrize("cancel", [False, True])
def test_protected_gui_path_stages_before_restart_and_confirms_one_new_desktop(tmp_path, monkeypatch, cancel):
    target = "/Applications/spaCR.app"
    executable = target + "/Contents/MacOS/spacr"
    state = {"version": "1.5.1.4"}
    events = []
    record = ic.InstallRecord("installer", "macos-app", "macos", target, running=True)
    plan = {"adapter": "macos-frozen-v1", "version": "1.5.1.5", "records": [ic.asdict(record)],
        "workdir": str(tmp_path), "pid": 4242, "ticked": [],
        "handshake": {"schema": 1, "token": "a" * 64, "expires": time.time() + 300}}
    handshake = ic._FrozenUpdateHandshake(plan)
    original_ready = ic._FrozenUpdateHandshake.ready
    def ready(channel):
        assert state["version"] == "1.5.1.4" and events == ["authorized", "staged"]
        original_ready(channel)
        events.append("ready")
        (handshake.cancel if cancel else handshake.approve)()
    monkeypatch.setattr(ic._FrozenUpdateHandshake, "ready", ready)
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (os.getuid(), executable))
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, **kwargs: "restart" in events)
    native_os = ic.os
    operations = SimpleNamespace(**vars(native_os))
    operations.path = SimpleNamespace(**vars(native_os.path))
    operations.path.isdir = lambda path: True if path == target else native_os.path.isdir(path)
    operations.access = lambda path, mode: False if path == "/Applications" else native_os.access(path, mode)
    operations.stat = lambda path: SimpleNamespace(st_mode=0o755) if path == "/Applications" else native_os.stat(path)
    operations.lstat = lambda path: SimpleNamespace(st_dev=1, st_ino=42) if path == target else native_os.lstat(path)
    operations.statvfs = lambda path: SimpleNamespace(f_flag=native_os.ST_RDONLY)
    monkeypatch.setattr(ic, "os", operations)
    monkeypatch.setattr(ic, "_macos_bundle_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(ic, "_macos_tree", lambda path, **kwargs: {"version": state["version"] if path == target else "1.5.1.5"})
    digest = hashlib.sha256(b"owned DMG fixture").hexdigest()
    monkeypatch.setattr(ic, "_published_digest", lambda url: digest)
    monkeypatch.setattr(ic, "_macos_frozen_authorized_command", lambda command, arguments: [command, *arguments])

    def invoke(argv, **kwargs):
        if argv[:2] == ["/usr/bin/hdiutil", "attach"]:
            mount = argv[argv.index("-mountpoint") + 1]
            return plistlib.dumps({"system-entities": [{"mount-point": mount}]})
        if argv[:2] == ["/usr/bin/hdiutil", "detach"]:
            events.append("detached")
            return ""
        assert argv[0] == "apply-macos-root"
        events.extend(["authorized", "staged"])
        endpoint = argv[argv.index("--phase-channel") + 1]
        with ic._UpdatePhaseClient(endpoint, 4242, executable) as channel:
            ic._native_update_ready(channel, "1.5.1.5", 4242)
            state["version"] = "1.5.1.5"
            events.append("activated")
            ic._native_update_activated(channel, "1.5.1.5")
        return "committed\t/Applications/.spacr-update.ABCDEF12\tdetached\n"

    def restart(pid):
        assert pid == 4242
        events.append("restart")
        return True
    def launch(*args, **kwargs):
        assert state["version"] == "1.5.1.5"
        events.append("new-main-window")
        return {"version": "1.5.1.5", "pid": 5678}
    monkeypatch.setattr(ic, "_launch_updated_desktop", launch)
    arguments = dict(run=invoke, download=lambda url, path: Path(path).write_bytes(b"owned DMG fixture"), wait=restart)
    if cancel:
        with pytest.raises(RuntimeError, match="cancelled"):
            ic._run_macos_frozen_update(plan, **arguments)
        assert state["version"] == "1.5.1.4"
        assert events == ["authorized", "staged", "ready", "detached"]
    else:
        result = ic._run_macos_frozen_update(plan, **arguments)
        assert result["relaunch"] == {"status": "ready"}
        assert events == ["authorized", "staged", "ready", "restart", "activated", "new-main-window", "detached"]
