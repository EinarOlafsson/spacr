"""Item 703: native uninstall adapters and their root helpers refuse every unbound case.

The macOS online, macOS frozen and Debian adapters are chosen only for the
fixed package-owned copies they know, and refuse a copy whose shortcuts,
launcher link, helper or marker changed. The administrator helpers refuse to
run as anything but their trusted copy, for any other desktop, or with a
changed bundle identity, and stop when the package manager fails. The
uninstall plan refuses an oversized plan or a native choice that changed,
and the native path deletes only the reviewed remainder.

Everything below ``tmp_path``; administrator rights, package manager, phase
channel and authorization processes are recorded fakes.
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import stat
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)
from tests.test_uninstall_selection_703 import (
    _prepared_helper,
    installation,
    protected_macos,
)

__all__ = ["installation", "protected_macos"]

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor fixtures")


def test_the_macos_online_adapter_is_only_for_macos_and_protected_copies(protected_macos,
                                                                         monkeypatch):
    box, machine, preview, choice, records, helper = protected_macos
    machine.platform = "linux"
    assert ic._macos_native_uninstall_adapter(preview, records, machine) is None
    machine.platform = "macos"
    monkeypatch.setattr(ic.os, "access", lambda path, mode: True)
    assert ic._macos_native_uninstall_adapter(preview, records, machine) is None


def test_the_macos_online_adapter_refuses_an_aliased_helper(protected_macos, monkeypatch):
    box, machine, preview, choice, records, helper = protected_macos
    real = os.path.realpath
    monkeypatch.setattr(ic.os.path, "realpath",
                        lambda path: "/elsewhere/helper" if path == str(helper) else real(path))
    with pytest.raises(PermissionError, match=str(helper)):
        ic._macos_native_uninstall_adapter(preview, records, machine)


def test_the_macos_online_adapter_refuses_extra_shortcuts_or_a_moved_launcher(protected_macos):
    box, machine, preview, choice, records, helper = protected_macos
    with_shortcut = [dataclasses.replace(record, shortcuts=(str(box.home / "Desktop" / "spaCR"),))
                     for record in records]
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_native_uninstall_adapter(preview, with_shortcut, machine)
    launcher = box.system("/usr/local/bin/spacr")
    launcher.unlink()
    launcher.symlink_to(box.home / "another-application")
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_native_uninstall_adapter(preview, records, machine)


def _frozen_app(box):
    app = box.system("/Applications/spaCR.app")
    box.write(app / "Contents/Frameworks/spacr-update-helper")
    box.write(app / "Contents/MacOS/spacr")
    machine = box.machine("macos")
    machine.real = True
    return app, machine


def test_a_user_owned_writable_frozen_app_needs_no_administrator(tmp_path):
    from tests.test_install_cleanup import Box

    box = Box(tmp_path)
    app, machine = _frozen_app(box)
    record = ic.InstallRecord(kind="installer", layout="macos-app", platform="macos",
                              root=str(app), registrations=(str(app),))
    assert ic._macos_frozen_uninstall_adapter({"cleanup": []}, [record], machine) is None


def test_a_protected_frozen_app_with_a_launcher_is_refused(tmp_path, monkeypatch):
    from tests.test_install_cleanup import Box

    box = Box(tmp_path)
    app, machine = _frozen_app(box)
    monkeypatch.setattr(ic.os, "access", lambda path, mode: False)
    record = ic.InstallRecord(kind="installer", layout="macos-app", platform="macos",
                              root=str(app), registrations=(str(app),),
                              launchers=("/usr/local/bin/spacr",))
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper", lambda: pytest.fail("helper"))
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_frozen_uninstall_adapter({"cleanup": []}, [record], machine)


def test_the_native_adapter_dispatches_by_platform(monkeypatch):
    windows = SimpleNamespace(platform="windows", real=True)
    monkeypatch.setattr(ic, "_windows_process_elevated", lambda: False)
    monkeypatch.setattr(ic, "_windows_native_uninstall_adapter",
                        lambda preview, records, machine: {"adapter": "windows"})
    assert ic._native_uninstall_adapter({}, [], windows) == {"adapter": "windows"}
    macos = SimpleNamespace(platform="macos", real=True)
    monkeypatch.setattr(ic, "_macos_frozen_uninstall_adapter",
                        lambda preview, records, machine: {"adapter": "frozen"})
    assert ic._native_uninstall_adapter({}, [], macos) == {"adapter": "frozen"}


def _debian_record(**changes):
    values = dict(kind="installer", layout="linux-deb", platform="linux", root="/opt/spacr",
                  version="1.5.0.7", registrations=("deb:spacr",), needs_admin=True)
    values.update(changes)
    return ic.InstallRecord(**values)


@pytest.fixture
def debian(monkeypatch):
    machine = SimpleNamespace(platform="linux", real=True, executable="/usr/bin/spacr")
    marker = "/opt/spacr/uninstall-protocol.json"
    state = {"mode": stat.S_IFREG | 0o644, "uid": 0, "size": 60,
             "text": json.dumps({"schema": 1, "adapter": "linux-deb-uninstall-v1"}),
             "pkexec": True}
    real_lstat, real_isfile, real_access = os.lstat, os.path.isfile, os.access
    monkeypatch.setattr(ic.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(ic, "_debian_check_package_owner",
                        lambda version, run=None: "/opt/spacr/_internal/spacr-update-helper")
    monkeypatch.setattr(ic.os, "lstat", lambda path, *a, **k: SimpleNamespace(
        st_mode=state["mode"], st_uid=state["uid"], st_size=state["size"])
        if path == marker else real_lstat(path, *a, **k))
    monkeypatch.setattr(ic, "open", lambda path, *a, **k: io.StringIO(state["text"])
                        if path == marker else open(path, *a, **k), raising=False)
    monkeypatch.setattr(ic.os.path, "isfile",
                        lambda path: state["pkexec"] if path == "/usr/bin/pkexec" else real_isfile(path))
    monkeypatch.setattr(ic.os, "access",
                        lambda path, mode: state["pkexec"] if path == "/usr/bin/pkexec"
                        else real_access(path, mode))
    return machine, state


def test_the_debian_adapter_binds_the_package_helper_and_desktop(debian):
    machine, _state = debian
    adapter = ic._native_uninstall_adapter({"cleanup": []}, [_debian_record()], machine)
    assert adapter == {"adapter": "linux-deb-native",
                       "helper": "/opt/spacr/_internal/spacr-update-helper",
                       "roots": ["/opt/spacr"], "desktop_executable": "/usr/bin/spacr",
                       "desktop_uid": os.getuid(), "previous_version": "1.5.0.7"}


@pytest.mark.parametrize("problem, error", [
    ("two-copies", "/opt/spacr"),
    ("cleanup-inside", "/opt/spacr"),
    ("marker-writable", "uninstall-protocol.json"),
    ("marker-content", "uninstall-protocol.json"),
    ("no-pkexec", "graphical administrator authorization is unavailable"),
])
def test_the_debian_adapter_refuses_what_it_cannot_bind(debian, problem, error):
    machine, state = debian
    records = [_debian_record()]
    preview = {"cleanup": []}
    if problem == "two-copies":
        records.append(_debian_record(root="/opt/spacr-old"))
    elif problem == "cleanup-inside":
        preview["cleanup"].append({"path": "/opt/spacr/models"})
    elif problem == "marker-writable":
        state["mode"] = stat.S_IFREG | 0o666
    elif problem == "marker-content":
        state["text"] = json.dumps({"schema": 1, "adapter": "something-else"})
    else:
        state["pkexec"] = False
    with pytest.raises((ValueError, PermissionError), match=error):
        ic._native_uninstall_adapter(preview, records, machine)


@pytest.fixture
def debian_root(monkeypatch):
    events = []
    helper = "/opt/spacr/_internal/spacr-update-helper"
    desktop = "/opt/spacr/spacr"
    answers = {"--no-act": 0, "--remove": 0}
    monkeypatch.setattr(ic, "_debian_authenticated_user", lambda uid: None)
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (1000, desktop))
    monkeypatch.setattr(ic, "_debian_check_package_owner", lambda version: helper)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", helper)
    monkeypatch.setattr(ic, "_debian_frontend_lock", nullcontext)
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: True)

    def runner(command):
        events.append(command)
        code = answers["--no-act"] if "--no-act" in command else answers["--remove"]
        return code, ("x" * 3000 + "dpkg: error: spacr is in use") if code else ""

    monkeypatch.setattr(ic, "_debian_root_runner", runner)

    class Channel:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def _send(self, phase):
            events.append(phase)

        def _receive(self, phases, **kwargs):
            return {"phase": "activate"}

    monkeypatch.setattr(ic, "_UpdatePhaseClient", Channel)

    def call():
        return ic._uninstall_debian_root("endpoint", 12345, desktop, 1000, "1.5.0.7")
    return SimpleNamespace(events=events, answers=answers, call=call)


def test_the_debian_root_helper_must_be_the_packaged_frozen_helper(debian_root, monkeypatch):
    monkeypatch.setattr(ic.sys, "executable", "/usr/bin/python3")
    with pytest.raises(ValueError, match="spacr-update-helper"):
        debian_root.call()
    assert debian_root.events == []


@pytest.mark.parametrize("failing", ["--no-act", "--remove"])
def test_a_failed_dpkg_step_stops_the_debian_uninstall(debian_root, failing):
    debian_root.answers[failing] = 1
    with pytest.raises(RuntimeError, match="Uninstall preparation stopped: x+dpkg: error") as raised:
        debian_root.call()
    assert str(raised.value).endswith("dpkg: error: spacr is in use")
    assert str(raised.value).count("x") == 2000 - len("dpkg: error: spacr is in use")
    assert "complete" not in debian_root.events
    assert ("prepared" in debian_root.events) == (failing == "--remove")


@pytest.fixture
def macos_root(monkeypatch):
    stage = "/private/var/root/spacr-helper.ABC12345/helper"
    target = "/Applications/spaCR.app"
    desktop = "/opt/desktop/spacr"
    state = {"uid": 0, "mode": 0o700, "owner": 1000, "digest": "a" * 64}
    actual_lstat = os.lstat
    monkeypatch.setattr(ic.os, "geteuid", lambda: 0)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", stage)
    monkeypatch.setattr(ic.os, "lstat", lambda path, **k: SimpleNamespace(
        st_uid=state["uid"], st_mode=state["mode"])
        if str(path) in {stage, os.path.dirname(stage)} else actual_lstat(path, **k))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: (state["owner"], desktop))
    monkeypatch.setattr(ic, "_macos_frozen_trusted_helper",
                        lambda: target + "/Contents/Frameworks/spacr-update-helper")
    monkeypatch.setattr(ic, "_macos_digest",
                        lambda path: state["digest"] if path == stage else "a" * 64)
    identities = []
    monkeypatch.setattr(ic, "_macos_bundle_identity", lambda *args: identities.append(args))
    monkeypatch.setattr(ic, "_macos_uninstall_tree", lambda *a, **k: pytest.fail("removal"))

    def call(identity="3:9", keep="[]"):
        return ic._uninstall_macos_root("endpoint", 12345, desktop, "1.5.0.7", identity, keep)
    return SimpleNamespace(state=state, call=call, identities=identities)


@pytest.mark.parametrize("problem, error", [
    ("not-root", "trusted_helper"),
    ("shared-folder", "spacr-helper.ABC12345"),
    ("root-desktop", "could not be identified"),
    ("other-helper", "trusted_helper"),
    ("bad-identity", "installation changed"),
    ("keep-not-list", "installation changed"),
])
def test_the_macos_root_helper_refuses_before_any_removal(macos_root, monkeypatch, problem,
                                                          error):
    arguments = {}
    if problem == "not-root":
        monkeypatch.setattr(ic.os, "geteuid", lambda: 501)
    elif problem == "shared-folder":
        macos_root.state["mode"] = 0o755
    elif problem == "root-desktop":
        macos_root.state["owner"] = 0
    elif problem == "other-helper":
        macos_root.state["digest"] = "b" * 64
    elif problem == "bad-identity":
        arguments["identity"] = "3;9"
    else:
        arguments["keep"] = '{"path": "/Applications/spaCR.app/models"}'
    with pytest.raises(ValueError, match=error):
        macos_root.call(**arguments)
    assert bool(macos_root.identities) == (problem in {"bad-identity", "keep-not-list"})


def test_the_descriptor_walk_refuses_special_files_and_non_directories(tmp_path):
    root = tmp_path / "spaCR.app"
    (root / "Contents").mkdir(parents=True)
    os.mkfifo(root / "Contents" / "pipe")
    for path in (root, root / "Contents"):
        path.chmod(0o755)
    with pytest.raises(ValueError, match="pipe"):
        ic._macos_uninstall_tree(str(root), ic._identity(str(root)), [], owner=os.getuid())
    single = tmp_path / "single-file"
    single.write_text("x")
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_uninstall_tree(str(single), ic._identity(str(single)), [], owner=os.getuid())


def _plain_tree(tmp_path):
    root = tmp_path / "spaCR.app"
    payload = root / "Contents" / "payload"
    payload.parent.mkdir(parents=True)
    payload.write_text("software")
    for path in (root, root / "Contents"):
        path.chmod(0o755)
    return root, payload


def test_a_directory_replaced_while_it_is_opened_is_refused(tmp_path, monkeypatch):
    root, payload = _plain_tree(tmp_path)
    identity = ic._identity(str(root))
    real_fstat = os.fstat

    def moved(descriptor):
        details = real_fstat(descriptor)
        return os.stat_result((details.st_mode, details.st_ino + 1) + tuple(details)[2:])

    monkeypatch.setattr(ic.os, "fstat", moved)
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_uninstall_tree(str(root), identity, [], apply=True, owner=os.getuid())
    assert payload.read_text() == "software"


def test_a_file_replaced_between_check_and_removal_is_refused(tmp_path, monkeypatch):
    root, payload = _plain_tree(tmp_path)
    identity = ic._identity(str(root))
    real_stat = os.stat
    seen = {}

    def replaced(name, *args, **kwargs):
        details = real_stat(name, *args, **kwargs)
        if name == "payload" and kwargs.get("dir_fd") is not None:
            seen[name] = seen.get(name, 0) + 1
            if seen[name] == 2:
                return os.stat_result((details.st_mode, details.st_ino + 1) + tuple(details)[2:])
        return details

    monkeypatch.setattr(ic.os, "stat", replaced)
    with pytest.raises(ValueError, match="installation changed"):
        ic._macos_uninstall_tree(str(root), identity, [], apply=True, owner=os.getuid())
    assert payload.read_text() == "software"


class _Channel:
    endpoint = "phase-endpoint"
    _connection = object()

    def __init__(self, workdir, events, fail_cancel=False, stop_first=False):
        self.events, self.fail_cancel, self.stop_first = events, fail_cancel, stop_first

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def _receive(self, phases, stopped=None, **kwargs):
        self.events.append(sorted(phases))
        if self.stop_first and stopped():
            raise RuntimeError("the authorization process ended")
        return {"phase": next(iter(phases))}

    def _send(self, phase):
        self.events.append(phase)
        if phase == "cancel" and self.fail_cancel:
            raise BrokenPipeError(32, "closed")


class _Process:
    def __init__(self, code=0, done=False):
        self.code, self.done, self.terminated = code, done, False

    def poll(self):
        return self.code if self.done else None

    def wait(self, timeout):
        return self.code

    def terminate(self):
        self.terminated = True


def _native_run(installation, tmp_path, monkeypatch, events, process, **channel):
    _box, _root, _machine, plan, _calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    monkeypatch.setattr(ic, "_UpdatePhaseServer",
                        lambda workdir: _Channel(workdir, events, **channel))
    monkeypatch.setattr(ic.subprocess, "Popen", lambda *a, **k: process)
    monkeypatch.setattr(ic, "_native_uninstall_command", lambda *a: ["native"])
    monkeypatch.setattr(handshake, "ready", lambda: events.append("ready"))
    monkeypatch.setattr(handshake, "wait_for_shutdown", lambda wait=None: events.append("closed"))
    return plan, handshake


def test_the_windows_adapter_runs_through_its_own_native_path(monkeypatch):
    seen = []
    monkeypatch.setattr(ic, "_run_windows_native_uninstall",
                        lambda plan, adapter, handshake, wait=None: seen.append(adapter) or "done")
    adapter = {"adapter": "windows-frozen-native"}
    assert ic._run_native_uninstall({"workdir": "/w"}, adapter, None) == "done"
    assert seen == [adapter]


def test_an_authorization_process_that_ends_early_stops_and_cancels(installation, tmp_path,
                                                                    monkeypatch):
    events = []
    process = _Process(code=1, done=True)
    plan, handshake = _native_run(installation, tmp_path, monkeypatch, events, process,
                                  fail_cancel=True, stop_first=True)
    with pytest.raises(RuntimeError, match="authorization process ended"):
        ic._run_native_uninstall(plan, {"adapter": "macos-online-native"}, handshake)
    assert events == [["prepared"], "cancel"] and not process.terminated


def test_a_native_helper_that_exits_with_failure_is_an_error(installation, tmp_path,
                                                             monkeypatch):
    events = []
    process = _Process(code=4)
    plan, handshake = _native_run(installation, tmp_path, monkeypatch, events, process)
    with pytest.raises(RuntimeError, match="Uninstall preparation stopped"):
        ic._run_native_uninstall(plan, {"adapter": "macos-online-native"}, handshake)
    assert events == [["prepared"], "ready", "closed", "activate", ["complete"]]


def test_a_helper_inside_a_selected_path_or_without_an_interpreter_is_refused(
        installation, tmp_path, monkeypatch):
    from tests.test_uninstall_selection_703 import _preview_choice

    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    with pytest.raises(ValueError, match="outside every selected path"):
        ic._start_uninstall_helper(preview, choice, system=machine,
                                   workdir=str(root / "helper"), spawn=lambda *a: None)
    monkeypatch.setattr(ic, "_helper_command",
                        lambda *a: (None, None, "no Python outside the installation"))
    with pytest.raises(ValueError, match="no Python outside the installation"):
        ic._start_uninstall_helper(preview, choice, system=machine,
                                   workdir=str(tmp_path / "helper"), spawn=lambda *a: None)


def test_an_administrator_debian_copy_is_refused_before_the_helper_starts(
        installation, tmp_path, monkeypatch):
    from tests.test_uninstall_selection_703 import _preview_choice

    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    records, _ = ic._validate_uninstall_selection(preview, choice, machine, after_shutdown=True)
    protected = [dataclasses.replace(record, needs_admin=True, registrations=("deb:spacr",))
                 for record in records]
    monkeypatch.setattr(ic, "_validate_uninstall_selection",
                        lambda *a, **k: (protected, []))
    monkeypatch.setattr(ic, "_native_uninstall_adapter", lambda *a: None)
    monkeypatch.setattr(ic.os, "geteuid", lambda: 1000)
    machine.real = True
    spawned = []
    with pytest.raises(PermissionError, match="Administrator access is required"):
        ic._start_uninstall_helper(preview, choice, system=machine,
                                   workdir=str(tmp_path / "helper"),
                                   spawn=lambda *a: spawned.append(a))
    assert spawned == [] and not (tmp_path / "helper").exists()


def test_a_running_windows_copy_prepares_its_own_uninstall_helper(installation, tmp_path,
                                                                  monkeypatch):
    from tests.test_uninstall_selection_703 import _preview_choice

    box, root, machine = installation
    preview, choice = _preview_choice(machine)
    records, cleanup = ic._validate_uninstall_selection(preview, choice, machine,
                                                        after_shutdown=True)
    machine.platform = "windows"
    machine.executable = str(root / "venv" / "Scripts" / "python.exe")
    installers = [dataclasses.replace(record, kind="installer") for record in records]
    monkeypatch.setattr(ic, "_validate_uninstall_selection", lambda *a, **k: (installers, cleanup))
    monkeypatch.setattr(ic, "_native_uninstall_adapter", lambda *a: None)
    monkeypatch.setattr(ic, "_WindowsPipHandshake", lambda plan: None)
    monkeypatch.delattr(ic.sys, "frozen", raising=False)
    prepared = []

    def prepare(running_root, copied, machine):
        prepared.append((running_root, copied))
        return {"command": ["helper.exe", "-I", "module.py", "run-plan", "x"]}

    monkeypatch.setattr(ic, "_prepare_windows_uninstall", prepare)
    spawned = []
    plan = ic._start_uninstall_helper(preview, choice, system=machine,
                                      workdir=str(tmp_path / "helper"),
                                      spawn=lambda *a: spawned.append(a))
    helper = tmp_path / "helper"
    assert prepared == [(str(root), str(helper / "helper"))]
    assert plan["command"] == ["helper.exe", "-I", "module.py", "run-uninstall-plan",
                               str(helper / "plan.json")]
    assert spawned == [(plan["command"], None, str(helper))]
    assert stat.S_IMODE(os.stat(helper / "helper").st_mode) == 0o700


def test_an_oversized_plan_file_is_refused(tmp_path):
    plan = tmp_path / "plan.json"
    with open(plan, "wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="invalid_uninstall_plan"):
        ic._run_uninstall_plan(str(plan))


def _approved(installation, tmp_path, native=None):
    box, root, machine, plan, calls = _prepared_helper(installation, tmp_path)
    path = tmp_path / "outside-helper" / "plan.json"
    if native is not None:
        plan["native"] = native
        path.write_text(json.dumps(plan))
    handshake = ic._FrozenUpdateHandshake(plan)
    handshake.ready()
    handshake.approve()
    return box, root, machine, plan, path


def test_a_native_choice_that_changed_since_review_stops_the_plan(installation, tmp_path,
                                                                  monkeypatch):
    box, root, machine, plan, path = _approved(installation, tmp_path)
    monkeypatch.setattr(ic, "_native_uninstall_adapter", lambda *a: {"adapter": "new"})
    with pytest.raises(ValueError, match="installation changed"):
        ic._run_uninstall_plan(str(path), system=machine, wait=lambda pid: True)
    assert (root / "venv" / "bin" / "python").exists()
    readiness = json.loads((tmp_path / "outside-helper" / "readiness.json").read_text())
    assert readiness["state"] == "error" and "installation changed" in readiness["error"]


def _native_plan(installation, tmp_path, monkeypatch, remaining_change=None, delete_fails=False):
    leftover = tmp_path / "Desktop" / "spaCR.desktop"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("[Desktop Entry]")
    _box, root, _machine, _plan, _path = _prepared_helper(installation, tmp_path)
    native = {"adapter": "linux-deb-native", "roots": [str(root)],
              "remaining": [{"path": str(leftover),
                             "identity": ic._uninstall_path_state(str(leftover))}]}
    import shutil

    shutil.rmtree(tmp_path / "outside-helper")
    box, root, machine, plan, path = _approved(installation, tmp_path, native=native)
    ran = []
    monkeypatch.setattr(ic, "_native_uninstall_adapter", lambda *a: native)
    monkeypatch.setattr(ic, "_run_native_uninstall",
                        lambda plan, adapter, handshake, wait=None: ran.append(adapter))
    if remaining_change:
        leftover.unlink()
        leftover.mkdir()
    if delete_fails:
        monkeypatch.setattr(ic, "_delete", lambda path, keep, report, reason:
                            report.failed.append((path, reason)))
    return root, machine, path, leftover, ran


def test_after_the_native_removal_only_the_reviewed_remainder_is_deleted(installation, tmp_path,
                                                                        monkeypatch):
    root, machine, path, leftover, ran = _native_plan(installation, tmp_path, monkeypatch)
    assert ic._run_uninstall_plan(str(path), system=machine, wait=lambda pid: True) == 0
    assert ran and not leftover.exists()
    assert (root / "venv" / "bin" / "python").exists()
    assert json.loads((tmp_path / "outside-helper" / "result.json").read_text())["ok"]


@pytest.mark.parametrize("problem, error", [
    ("changed", "installation changed"),
    ("undeletable", "Uninstall preparation stopped"),
])
def test_a_changed_or_undeletable_remainder_stops_after_the_native_removal(
        installation, tmp_path, monkeypatch, problem, error):
    root, machine, path, leftover, ran = _native_plan(
        installation, tmp_path, monkeypatch, remaining_change=problem == "changed",
        delete_fails=problem == "undeletable")
    with pytest.raises((ValueError, PermissionError), match=error):
        ic._run_uninstall_plan(str(path), system=machine, wait=lambda pid: True)
    assert ran and leftover.exists()
