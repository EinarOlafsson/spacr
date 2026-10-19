"""Item 703: the Windows native uninstaller is used only when every binding holds.

The installed uninstaller is chosen only for administrator copies whose
protocol marker, payload manifest and Start Menu shortcuts still match what
was reviewed; its outside-root helper refuses a changed request, a different
desktop, a helper inside the root or a changed helper binary before touching
a file, and removes only manifest files, keeping opted-out folders. The
native folder lookup and the desktop owner lookup read Windows APIs, faked
here with ctypes so the decisions run on any computer.

Every path is below ``tmp_path``; the Windows APIs, ACL reader and phase
channel are recorded fakes.
"""
from __future__ import annotations

import ctypes
import errno
import json
import os
import sys
from types import SimpleNamespace

import pytest

from spacr import install_cleanup as ic
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)
from tests.test_uninstall_selection_703 import (
    _prepared_helper,
    installation,
    protected_windows,
)

__all__ = ["installation", "protected_windows"]


def _record(root, shortcuts=()):
    return ic.InstallRecord(kind="installer", layout="windows-offline", platform="windows",
                            root=str(root), shortcuts=tuple(str(s) for s in shortcuts),
                            needs_admin=True)


def _machine(tmp_path):
    return SimpleNamespace(executable=str(tmp_path / "desktop" / "spacr.exe"))


def test_the_native_adapter_binds_the_reviewed_root_menu_and_remaining_shortcuts(
        protected_windows, tmp_path):
    root, request, _project, kept, shortcut = protected_windows
    desktop_link = tmp_path / "Desktop" / "spaCR.lnk"
    desktop_link.parent.mkdir()
    desktop_link.write_text(str(root))
    preview = {"cleanup": [{"path": str(kept)}, {"path": "registry:Software\\spacr\\qt"},
                           {"path": str(tmp_path / "elsewhere")}]}
    adapter = ic._windows_native_uninstall_adapter(
        preview, [_record(root, [shortcut, desktop_link])], _machine(tmp_path))
    assert adapter["adapter"] == "windows-frozen-native"
    assert adapter["roots"] == [str(root)]
    assert adapter["uninstaller"] == str(root / "Uninstall.exe")
    assert adapter["uninstaller_sha256"] == ic._macos_digest(str(root / "Uninstall.exe"))
    assert adapter["desktop_executable"] == str(tmp_path / "desktop" / "spacr.exe")
    assert adapter["request"] == request
    assert adapter["remaining"] == [{"path": str(desktop_link),
                                     "identity": ic._uninstall_path_state(str(desktop_link))}]


def test_the_native_adapter_is_not_used_without_an_administrator_copy(tmp_path):
    user_copy = ic.InstallRecord(kind="installer", layout="windows-online",
                                 platform="windows", root=str(tmp_path))
    assert ic._windows_native_uninstall_adapter({"cleanup": []}, [user_copy],
                                                _machine(tmp_path)) is None


@pytest.mark.parametrize("marker", ["oversized", "other-adapter"])
def test_the_native_adapter_refuses_a_marker_it_does_not_know(protected_windows, tmp_path,
                                                              marker):
    root, _request, *_ = protected_windows
    path = root / "_internal" / "spacr-uninstall-protocol.json"
    path.write_text(" " * 1100 if marker == "oversized"
                    else json.dumps({"schema": 1, "adapter": "windows-frozen-uninstall-v2"}))
    with pytest.raises(ValueError, match="invalid_windows_uninstall_protocol"):
        ic._windows_native_uninstall_adapter({"cleanup": []}, [_record(root)], _machine(tmp_path))


@pytest.mark.parametrize("change, error", [
    ("shape", "invalid_windows_uninstall_scope"),
    ("identity", "installation changed"),
    ("version", "installation changed"),
    ("shortcut", "invalid_windows_uninstall_shortcut"),
])
def test_the_native_scope_refuses_a_request_that_no_longer_matches(protected_windows,
                                                                   tmp_path, change, error):
    root, request, project, _kept, _shortcut = protected_windows
    target = request["targets"][0]
    if change == "shape":
        request["extra"] = True
    elif change == "identity":
        target["identity"] = [0, 0]
    elif change == "version":
        target["version"] = "1.5.0.6"
    else:
        target["shortcuts"] = [str(tmp_path / "Desktop" / "spaCR.lnk")]
    with pytest.raises(ValueError, match=error):
        ic._windows_uninstall_targets(request)
    assert project.read_text(encoding="utf-8") == "unrelated" and (root / "spacr.exe").exists()


def test_a_special_file_is_never_taken_for_a_system_owned_path(tmp_path, monkeypatch):
    fifo = tmp_path / "pipe"
    os.mkfifo(fifo)
    monkeypatch.setattr(ic, "_windows_pip_acl", lambda path: pytest.fail("ACL read"))
    with pytest.raises(ValueError, match="Uninstall preparation stopped"):
        ic._windows_system_owned(str(fifo))


class _Shell:
    def __init__(self, answer, code=0):
        self.answer, self.code, self.asked = answer, code, []

        def SHGetFolderPathW(owner, identifier, token, flags, buffer):
            self.asked.append(identifier)
            buffer.value = self.answer
            return self.code

        self.SHGetFolderPathW = SHGetFolderPathW


def test_known_folders_come_from_the_shell_api_not_the_environment(tmp_path, monkeypatch):
    folder = tmp_path / "Common Start Menu"
    folder.mkdir()
    shell = _Shell(str(folder))
    monkeypatch.setenv("ProgramData", str(tmp_path / "spoofed"))
    monkeypatch.setattr(ic, "_windows_update_library",
                        lambda name: shell if name == "shell32" else pytest.fail(name))
    checked = []
    monkeypatch.setattr(ic, "_installer_metadata_folder",
                        lambda path, code, windows: checked.append((code, windows)) or path)
    assert ic._windows_known_folder(0x17) == str(folder)
    assert shell.asked == [0x17]
    assert checked == [("windows_uninstall_folder", True)]
    assert shell.SHGetFolderPathW.restype is ctypes.c_int32
    for failing in (_Shell(str(folder), code=1), _Shell("")):
        monkeypatch.setattr(ic, "_windows_update_library", lambda name, s=failing: s)
        with pytest.raises(OSError, match="windows_folder"):
            ic._windows_known_folder(0x26)


def _windows_libraries(*, process=7, image="C:\\Program Files\\spaCR\\spacr.exe",
                       token=True, sid="S-1-5-21-1000"):
    calls = []

    def fn(name, result):
        def call(*args):
            calls.append(name)
            return result(*args)
        return call

    def query(handle, flags, buffer, size):
        if image:
            buffer.value = image
        return bool(image)

    def open_token(handle, access, pointer):
        if token:
            pointer._obj.value = 99
        return token

    def information(handle, kind, buffer, length, needed):
        if buffer is None:
            needed._obj.value = ctypes.sizeof(ctypes.c_void_p)
            return 0
        return 1

    def convert(sid_pointer, text):
        text._obj.value = sid
        return 1

    kernel = SimpleNamespace(OpenProcess=fn("OpenProcess", lambda *a: process),
                             CloseHandle=fn("CloseHandle", lambda *a: 1),
                             QueryFullProcessImageNameW=fn("Query", query),
                             LocalFree=fn("LocalFree", lambda *a: None))
    security = SimpleNamespace(OpenProcessToken=fn("OpenProcessToken", open_token),
                               GetTokenInformation=fn("GetTokenInformation", information),
                               ConvertSidToStringSidW=fn("Convert", convert))
    return {"kernel32": kernel, "advapi32": security}, calls


def test_the_windows_desktop_owner_is_read_from_its_process_token(monkeypatch):
    libraries, calls = _windows_libraries()
    monkeypatch.setattr(ic, "_windows_update_library", libraries.__getitem__)
    with monkeypatch.context() as patched:
        patched.setattr(ic.os, "name", "nt")
        owner, executable = ic._desktop_process_identity(4242)
    assert owner == "S-1-5-21-1000"
    assert executable == os.path.realpath("C:\\Program Files\\spaCR\\spacr.exe")
    assert calls[0] == "OpenProcess" and calls.count("CloseHandle") == 2
    assert "LocalFree" in calls


@pytest.mark.parametrize("failure", ["process", "image", "token"])
def test_a_windows_desktop_that_cannot_be_read_is_not_identified(monkeypatch, failure):
    libraries, calls = _windows_libraries(process=0 if failure == "process" else 7,
                                          image="" if failure == "image" else "C:\\spacr.exe",
                                          token=failure != "token")
    monkeypatch.setattr(ic, "_windows_update_library", libraries.__getitem__)
    with monkeypatch.context() as patched:
        patched.setattr(ic.os, "name", "nt")
        with pytest.raises(OSError, match="could not be identified"):
            ic._desktop_process_identity(4242)
    assert calls.count("CloseHandle") == (0 if failure == "process" else 1)


@pytest.mark.parametrize("pid", [True, "12", 1, 0x1_0000_0000])
def test_only_a_real_process_number_names_a_desktop(pid):
    with pytest.raises(ValueError, match="could not be identified"):
        ic._desktop_process_identity(pid)


class _Channel:
    endpoint = "pipe-endpoint"
    _connection = object()

    def __init__(self, workdir, events, *, fail_cancel=False):
        self.events, self.fail_cancel = events, fail_cancel

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def _send(self, phase):
        self.events.append(phase)
        if phase == "cancel" and self.fail_cancel:
            raise BrokenPipeError(32, "the uninstaller already closed the pipe")

    def _receive(self, phases, **kwargs):
        self.events.append(sorted(phases))
        if kwargs["stopped"]():
            raise RuntimeError("the native uninstaller stopped")
        return {"phase": next(iter(phases))}


def _windows_plan(installation, tmp_path, monkeypatch, *, events, authorize,
                  fail_cancel=False, executable=None):
    from contextlib import nullcontext

    _box, _root, machine, plan, _calls = _prepared_helper(installation, tmp_path)
    handshake = ic._FrozenUpdateHandshake(plan)
    uninstaller = tmp_path / "Uninstall.exe"
    uninstaller.write_bytes(b"native")
    adapter = {"uninstaller": str(uninstaller), "request": {"targets": []},
               "uninstaller_sha256": ic._macos_digest(str(uninstaller)),
               "desktop_executable": executable or machine.executable}
    monkeypatch.setattr(ic, "_windows_pip_private_access", lambda *a, **k: None)
    monkeypatch.setattr(ic, "_windows_frozen_installer_pin", lambda *a: nullcontext())
    monkeypatch.setattr(ic, "_windows_authorized_process", authorize)
    monkeypatch.setattr(ic, "_UpdatePhaseServer",
                        lambda workdir: _Channel(workdir, events, fail_cancel=fail_cancel))
    monkeypatch.setattr(handshake, "ready", lambda: events.append("ready"))
    monkeypatch.setattr(handshake, "wait_for_shutdown", lambda wait=None: events.append("closed"))
    return plan, adapter, handshake


def test_a_native_uninstaller_that_cannot_start_is_reported_and_cancelled(
        installation, tmp_path, monkeypatch):
    events = []

    def authorize(path, arguments, visible):
        raise PermissionError(13, "The operation was canceled by the user")

    plan, adapter, handshake = _windows_plan(installation, tmp_path, monkeypatch,
                                             events=events, authorize=authorize,
                                             fail_cancel=True)
    with pytest.raises(PermissionError, match="canceled by the user"):
        ic._run_windows_native_uninstall(plan, adapter, handshake)
    assert "activate" not in events and events[-1] == "cancel"


def test_a_native_uninstaller_that_reports_failure_after_completion_is_an_error(
        installation, tmp_path, monkeypatch):
    events = []
    plan, adapter, handshake = _windows_plan(
        installation, tmp_path, monkeypatch, events=events,
        authorize=lambda path, arguments, visible: (1, "bootstrap failed"))

    class Late(_Channel):
        def _receive(self, phases, **kwargs):
            self.events.append(sorted(phases))
            return {"phase": next(iter(phases))}

    monkeypatch.setattr(ic, "_UpdatePhaseServer", lambda workdir: Late(workdir, events))
    with pytest.raises(RuntimeError, match="Uninstall preparation stopped"):
        ic._run_windows_native_uninstall(plan, adapter, handshake)
    assert events == [["prepared"], "ready", "closed", "activate", ["complete"]]


@pytest.mark.parametrize("problem, error", [
    ("quote", "invalid_windows_uninstall_argument"),
    ("long", "windows_uninstall_scope_too_large"),
])
def test_native_uninstaller_arguments_cannot_be_injected_or_overflow(
        installation, tmp_path, monkeypatch, problem, error):
    events = []
    executable = ('C:\\spa"CR.exe' if problem == "quote"
                  else "C:\\" + "x" * 900 + "\\spacr.exe")
    plan, adapter, handshake = _windows_plan(
        installation, tmp_path, monkeypatch, events=events,
        authorize=lambda *a, **k: pytest.fail("the uninstaller ran"), executable=executable)
    with pytest.raises(ValueError, match=error):
        ic._run_windows_native_uninstall(plan, adapter, handshake)
    assert events == []


@pytest.fixture
def frozen_windows_helper(protected_windows, tmp_path, monkeypatch):
    root, request, project, kept, shortcut = protected_windows
    staged = tmp_path / "native-private" / "spacr-update-helper.exe"
    staged.parent.mkdir()
    staged.write_bytes((root / "_internal/spacr-update-helper.exe").read_bytes())
    request_path = tmp_path / "private-request.json"
    request_path.write_text(json.dumps(request))
    desktop = str(root / "spacr.exe")
    monkeypatch.setattr(ic, "_windows_process_elevated", lambda: True)
    monkeypatch.setattr(ic.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ic.sys, "executable", str(staged))
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda pid: ("desktop-sid", desktop))
    monkeypatch.setattr(ic, "_windows_pip_private_access", lambda path, **kwargs: None)
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: True)
    events = []

    class Channel:
        def __init__(self, endpoint, pid, executable):
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

    def call(path=None, desktop_executable=desktop):
        return ic._uninstall_frozen_windows(str(path or request_path), str(root),
                                            "endpoint", 12345, desktop_executable)
    return SimpleNamespace(root=root, request=request, request_path=request_path,
                           staged=staged, project=project, kept=kept, events=events,
                           call=call)


@pytest.mark.parametrize("problem, error", [
    ("not-elevated", "invalid_windows_native_uninstall"),
    ("other-desktop", "could not be identified"),
    ("oversized-request", "invalid_windows_uninstall_request"),
    ("helper-inside", "windows_uninstall_helper_must_be_outside"),
    ("helper-changed", "windows_uninstall_helper_changed"),
])
def test_the_outside_root_helper_refuses_before_touching_any_file(
        frozen_windows_helper, monkeypatch, problem, error):
    helper = frozen_windows_helper
    desktop = str(helper.root / "spacr.exe")
    if problem == "not-elevated":
        monkeypatch.setattr(ic, "_windows_process_elevated", lambda: False)
    elif problem == "other-desktop":
        desktop = str(helper.root / "other.exe")
    elif problem == "oversized-request":
        helper.request_path.write_text(" " * 70000)
    elif problem == "helper-inside":
        monkeypatch.setattr(ic.sys, "executable", str(helper.root / "_internal" / "h.exe"))
    else:
        helper.staged.write_bytes(b"a different helper")
    with pytest.raises(ValueError, match=error):
        helper.call(desktop_executable=desktop)
    assert (helper.root / "spacr.exe").exists() and helper.events == []


def test_a_request_replaced_while_it_is_read_is_refused(frozen_windows_helper, monkeypatch):
    helper = frozen_windows_helper
    real_fstat = os.fstat

    def other_file(descriptor):
        details = real_fstat(descriptor)
        return os.stat_result((details.st_mode, details.st_ino + 1) + tuple(details)[2:])

    monkeypatch.setattr(ic.os, "fstat", other_file)
    with pytest.raises(ValueError, match="changed_windows_uninstall_request"):
        helper.call()
    assert (helper.root / "spacr.exe").exists()


def test_the_helper_keeps_opted_out_folders_and_reports_a_folder_it_cannot_remove(
        frozen_windows_helper, monkeypatch):
    helper = frozen_windows_helper
    keep_dir = helper.root / "_internal" / "spacr-1.5.0.7.dist-info"
    helper.request["targets"][0]["keep"].append(str(keep_dir))
    helper.request_path.write_text(json.dumps(helper.request))
    real_rmdir = os.rmdir
    attempted = []

    def rmdir(path, *args, **kwargs):
        attempted.append(os.fspath(path))
        if os.fspath(path) == str(helper.root / "_internal"):
            raise OSError(errno.EACCES, "Access is denied", os.fspath(path))
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(ic.os, "rmdir", rmdir)
    with pytest.raises(PermissionError):
        helper.call()
    assert str(keep_dir) not in attempted
    assert (keep_dir / "METADATA").is_file() and helper.project.read_text(encoding="utf-8") == "unrelated"
    assert "complete" not in helper.events
