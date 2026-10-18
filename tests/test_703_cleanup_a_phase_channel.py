"""Item 703: desktop identity, authenticated phase channel and native update coordination guards."""
import ctypes
import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
import types

import pytest

from spacr import install_cleanup as ic


class _Proxy:
    """Delegate to a real module while overriding a few attributes for this module only."""

    def __init__(self, module, **overrides):
        self._module = module
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self._module, name)


class _Fn:
    """A native function stand-in that accepts ctypes signature assignments."""

    def __init__(self, impl):
        self.impl = impl

    def __call__(self, *args):
        return self.impl(*args)


def _windows_libraries(fault, calls):
    def open_process(*_args):
        return 0 if fault == "open" else 41

    def image(_process, _flags, path, _size):
        if fault == "image":
            return 0
        path.value = "/opt/spaCR/spacr"
        return 1

    def token(_process, _access, ref):
        if fault == "token":
            return 0
        ref._obj.value = 77
        return 1

    def information(_token, _kind, buffer, _size, needed):
        if buffer is None:
            needed._obj.value = 0 if fault == "size" else 16
            return 0
        return 0 if fault == "information" else 1

    def convert(_sid, ref):
        if fault == "sid":
            return 0
        ref._obj.value = "S-1-5-21-7"
        return 1

    kernel = types.SimpleNamespace(
        OpenProcess=_Fn(open_process), QueryFullProcessImageNameW=_Fn(image),
        CloseHandle=_Fn(lambda handle: calls.append(("close", getattr(handle, "value", handle)))),
        LocalFree=_Fn(lambda handle: calls.append("free")))
    security = types.SimpleNamespace(
        OpenProcessToken=_Fn(token), GetTokenInformation=_Fn(information),
        ConvertSidToStringSidW=_Fn(convert))
    return {"kernel32": kernel, "advapi32": security}


@pytest.mark.parametrize("fault", [None, "size", "information", "sid", "token"])
def test_windows_desktop_identity_reads_native_owner_and_releases_every_handle(monkeypatch, fault):
    calls = []
    libraries = _windows_libraries(fault, calls)
    monkeypatch.setattr(ic, "os", _Proxy(os, name="nt"))
    monkeypatch.setattr(ic, "_windows_update_library", lambda name: libraries[name])
    if fault is None:
        owner, executable = ic._desktop_process_identity(4321)
        assert owner == "S-1-5-21-7"
        assert executable == os.path.realpath("/opt/spaCR/spacr")
        assert calls == ["free", ("close", 77), ("close", 41)]
        return
    with pytest.raises((OSError, ValueError), match="could not be identified"):
        ic._desktop_process_identity(4321)
    assert ("close", 41) in calls and "free" not in calls
    assert (("close", 77) in calls) == (fault != "token")


@pytest.mark.parametrize("fault", [None, "path", "owner", "status"])
def test_macos_desktop_identity_uses_libproc_and_numeric_owner(monkeypatch, fault):
    def pidpath(_pid, path, _size):
        if fault == "path":
            return 0
        path.value = b"/Applications/spaCR.app/Contents/MacOS/spaCR"
        return len(path.value)

    library = types.SimpleNamespace(proc_pidpath=_Fn(pidpath))
    opened = []
    monkeypatch.setattr(ctypes, "CDLL", lambda name, use_errno: opened.append(name) or library)
    monkeypatch.setattr(ic, "sys", _Proxy(sys, platform="darwin"))
    monkeypatch.setattr(ic, "os", _Proxy(os, name="posix"))
    commands = []

    def run(argv, timeout):
        commands.append(argv)
        return (1 if fault == "status" else 0), ("staff\n" if fault == "owner" else " 501\n")

    monkeypatch.setattr(ic, "_run", run)
    if fault is None:
        assert ic._desktop_process_identity(812) == (
            501, os.path.realpath("/Applications/spaCR.app/Contents/MacOS/spaCR"))
        assert opened == ["/usr/lib/libproc.dylib"]
        assert commands == [["/bin/ps", "-p", "812", "-o", "uid="]]
        return
    with pytest.raises((OSError, ValueError), match="could not be identified"):
        ic._desktop_process_identity(812)
    assert bool(commands) == (fault != "path")


class _Connection:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.sent = []
        self.closed = False

    def settimeout(self, _value):
        return None

    def recv(self, _size):
        return self.chunks.pop(0) if self.chunks else b""

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True


def _channel(direction, secret, connection=None, listener=None):
    channel = ic._UpdatePhaseChannel()
    channel._connection, channel._listener = connection, listener
    channel._buffer, channel._sent, channel._received = b"", 0, 0
    channel._direction, channel._secret = direction, secret
    return channel


def _signed(secret, *phases):
    peer = _channel("installer", secret, _Connection())
    for phase in phases:
        peer._send(phase, version="2.0.0")
    return b"".join(peer._connection.sent)


def test_phase_message_over_the_size_bound_is_never_sent():
    connection = _Connection()
    channel = _channel("desktop", b"k" * 32, connection)
    with pytest.raises(ValueError, match="invalid operation message"):
        channel._send("prepared", padding="x" * 9000)
    assert connection.sent == []


def test_phase_listener_refuses_a_peer_outside_loopback():
    class Listener:
        def accept(self):
            return _Connection(), ("10.0.0.5", 4000)

    channel = _channel("desktop", b"k" * 32, listener=Listener())
    with pytest.raises(ValueError, match="invalid operation message"):
        channel._receive({"prepared"}, timeout=5)


def test_buffered_second_message_is_verified_without_another_read():
    secret = b"s" * 32
    connection = _Connection([_signed(secret, "prepared", "activated")])
    channel = _channel("desktop", secret, connection)
    assert channel._receive({"prepared"}, timeout=5)["phase"] == "prepared"
    connection.chunks = [b"never read"]
    second = channel._receive({"activated"}, timeout=5)
    assert second["phase"] == "activated" and second["sequence"] == 2
    assert connection.chunks == [b"never read"]


def test_partial_message_then_stop_request_ends_the_wait():
    connection = _Connection([b'{"payload":'])
    channel = _channel("desktop", b"k" * 32, connection)
    with pytest.raises(RuntimeError, match="stopped before confirming"):
        channel._receive({"prepared"}, timeout=5, stopped=lambda: True)
    assert channel._buffer == b'{"payload":'


def test_phase_wait_without_time_left_reports_the_deadline():
    channel = _channel("desktop", b"k" * 32, _Connection())
    with pytest.raises(TimeoutError, match="before the deadline"):
        channel._receive({"prepared"}, timeout=0)


def test_windows_phase_server_restricts_its_descriptor(tmp_path, monkeypatch):
    restricted = []
    monkeypatch.setattr(ic, "os", _Proxy(os, name="nt"))
    monkeypatch.setattr(ic, "_windows_pip_private_access",
                        lambda path, **kwargs: restricted.append((path, kwargs)))
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        endpoint = server.endpoint
        port, secret = open(endpoint, encoding="ascii").read().split(":")
        assert int(port) == server._listener.getsockname()[1] and len(secret) == 64
    assert restricted == [(endpoint, {"private_read": True})]
    assert not os.path.exists(endpoint)


def test_phase_server_removes_its_descriptor_when_restriction_fails(tmp_path, monkeypatch):
    def refuse(path, **_kwargs):
        raise PermissionError(path)

    monkeypatch.setattr(ic, "os", _Proxy(os, name="nt"))
    monkeypatch.setattr(ic, "_windows_pip_private_access", refuse)
    with pytest.raises(PermissionError):
        ic._UpdatePhaseServer(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_phase_server_closes_listener_when_descriptor_cannot_be_created(tmp_path, monkeypatch):
    def unavailable(**_kwargs):
        raise OSError("no space")

    monkeypatch.setattr(ic.tempfile, "mkstemp", unavailable)
    with pytest.raises(OSError, match="no space"):
        ic._UpdatePhaseServer(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def _bind_desktop(monkeypatch, owner=None):
    owner = os.getuid() if owner is None else owner
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: (owner, os.path.realpath(sys.executable)))


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor modes")
def test_phase_client_refuses_a_descriptor_that_is_not_a_regular_file(tmp_path, monkeypatch):
    _bind_desktop(monkeypatch)
    folder = tmp_path / "descriptor"
    folder.mkdir()
    with pytest.raises(ValueError, match="not private"):
        ic._UpdatePhaseClient(str(folder), 1234, sys.executable)


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor modes")
def test_phase_client_refuses_a_descriptor_owned_by_another_user(tmp_path, monkeypatch):
    _bind_desktop(monkeypatch, owner=os.getuid() + 1)
    endpoint = tmp_path / "phase"
    endpoint.write_text("4000:" + "a" * 64, encoding="ascii")
    endpoint.chmod(0o600)
    with pytest.raises(ValueError, match="not private"):
        ic._UpdatePhaseClient(str(endpoint), 1234, sys.executable)


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor modes")
def test_phase_client_refuses_descriptor_replaced_after_inspection(tmp_path, monkeypatch):
    _bind_desktop(monkeypatch)
    endpoint = tmp_path / "phase"
    endpoint.write_text("4000:" + "a" * 64, encoding="ascii")
    endpoint.chmod(0o600)

    def swapped(descriptor):
        actual = os.fstat(descriptor)
        values = list(actual)
        values[1] = actual.st_ino + 1
        return os.stat_result(values)

    monkeypatch.setattr(ic, "os", _Proxy(os, fstat=swapped))
    with pytest.raises(ValueError, match="not private"):
        ic._UpdatePhaseClient(str(endpoint), 1234, sys.executable)


@pytest.mark.skipif(os.name == "nt", reason="POSIX descriptor modes")
@pytest.mark.parametrize("text", ["not-a-descriptor", "70000:" + "a" * 64])
def test_phase_client_refuses_malformed_descriptor_contents(tmp_path, monkeypatch, text):
    _bind_desktop(monkeypatch)
    endpoint = tmp_path / "phase"
    endpoint.write_text(text, encoding="ascii")
    endpoint.chmod(0o600)
    with pytest.raises(ValueError, match="not private"):
        ic._UpdatePhaseClient(str(endpoint), 1234, sys.executable)


def test_windows_phase_client_checks_acl_before_and_after_opening(tmp_path, monkeypatch):
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        checked = []
        monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: ("S-1-5-21-7", os.path.realpath(sys.executable)))
        monkeypatch.setattr(ic, "os", _Proxy(os, name="nt"))
        monkeypatch.setattr(ic, "_windows_pip_private_access", lambda path, **kwargs: checked.append((path, kwargs)))
        with ic._UpdatePhaseClient(server.endpoint, 1234, sys.executable) as client:
            client._send("prepared", version="2.0.0")
            assert server._receive({"prepared"}, timeout=10)["details"] == {"version": "2.0.0"}
        monkeypatch.setattr(ic, "os", os)
    expected = (server.endpoint, {"owner_sid": "S-1-5-21-7", "private_read": True})
    assert checked == [expected, expected]


def test_native_update_ready_stops_when_old_desktop_is_still_running(monkeypatch):
    sent = []
    channel = types.SimpleNamespace(_send=lambda phase, **details: sent.append((phase, details)),
                                    _receive=lambda phases, timeout: {"phase": "activate"})
    monkeypatch.setattr(ic, "_wait_for_exit", lambda pid, timeout: False)
    with pytest.raises(RuntimeError, match="has not closed"):
        ic._native_update_ready(channel, "2.0.0", 999)
    assert sent == [("prepared", {"version": "2.0.0", "desktop_pid": 999})]


class _Handshake:
    def __init__(self):
        self.events = []

    def check(self):
        self.events.append("check")

    def ready(self):
        self.events.append("ready")

    def wait_for_shutdown(self, wait):
        self.events.append("shutdown")


def _coordinate(monkeypatch, tmp_path, replies, operation, *, connected=True, launch=lambda: "desktop", send_error=None):
    sent = []
    released = threading.Event()

    class Listener:
        def close(self):
            released.set()

    class Server:
        def __init__(self, workdir):
            assert workdir == str(tmp_path)
            self.endpoint = "private-endpoint"
            self._connection = _Connection() if connected else None
            self._listener = Listener()

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def _send(self, phase, **_details):
            sent.append(phase)
            if send_error is not None and phase == "cancel":
                raise send_error

        def _receive(self, phases, **_kwargs):
            reply = replies.pop(0)
            assert reply["phase"] in phases
            return reply

    monkeypatch.setattr(ic, "_UpdatePhaseServer", Server)
    plan = {"workdir": str(tmp_path), "version": "2.0.0", "pid": 55}
    handshake = _Handshake()
    call = lambda: ic._coordinate_native_update(
        plan, handshake, lambda endpoint: operation(endpoint, released), check=lambda: None, launch=launch)
    return call, sent, handshake


_PREPARED = {"phase": "prepared", "details": {"version": "2.0.0", "desktop_pid": 55}}
_ACTIVATED = {"phase": "activated", "details": {"version": "2.0.0"}}


def test_coordinated_update_returns_installer_output_and_new_desktop(monkeypatch, tmp_path):
    call, sent, handshake = _coordinate(monkeypatch, tmp_path, [_PREPARED, _ACTIVATED],
                                        lambda endpoint, _released: (0, "installed " + endpoint))
    assert call() == ("installed private-endpoint", "desktop")
    assert sent == ["activate", "confirmed"]
    assert handshake.events == ["check", "ready", "shutdown", "check"]


def test_coordinated_update_cancels_on_mismatched_preparation(monkeypatch, tmp_path):
    wrong = {"phase": "prepared", "details": {"version": "9.9.9", "desktop_pid": 55}}
    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [wrong], lambda _e, _r: (0, ""))
    with pytest.raises(ValueError, match="invalid update message"):
        call()
    assert sent == ["cancel"]


@pytest.mark.parametrize("error", [OSError("peer gone"), ValueError("message too large")])
def test_coordinated_update_keeps_original_error_when_cancel_cannot_be_sent(monkeypatch, tmp_path, error):
    wrong = {"phase": "prepared", "details": {"version": "2.0.0", "desktop_pid": 56}}
    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [wrong], lambda _e, _r: (0, ""), send_error=error)
    with pytest.raises(ValueError, match="invalid update message"):
        call()
    assert sent == ["cancel"]


def test_coordinated_update_retains_both_on_mismatched_activation(monkeypatch, tmp_path):
    wrong = {"phase": "activated", "details": {"version": "1.0.0"}}
    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [_PREPARED, wrong], lambda _e, _r: (0, ""))
    with pytest.raises(ValueError, match="invalid update message"):
        call()
    assert sent == ["activate", "cancel"]


def test_coordinated_update_reraises_installer_worker_error(monkeypatch, tmp_path):
    def failing(_endpoint, _released):
        raise OSError("installer crashed")

    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [_PREPARED, _ACTIVATED], failing)
    with pytest.raises(OSError, match="installer crashed"):
        call()
    assert sent == ["activate", "confirmed", "retain"]


def test_coordinated_update_reports_installer_failure_code(monkeypatch, tmp_path):
    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [_PREPARED, _ACTIVATED],
                                         lambda _e, _r: (2, "disk full"))
    with pytest.raises(RuntimeError, match="disk full"):
        call()
    assert sent[-1] == "retain"


def test_coordinated_update_releases_unconnected_worker_before_raising(monkeypatch, tmp_path):
    wrong = {"phase": "prepared", "details": {}}

    def blocked(_endpoint, released):
        assert released.wait(10)
        return 0, ""

    call, sent, _handshake = _coordinate(monkeypatch, tmp_path, [wrong], blocked, connected=False)
    with pytest.raises(ValueError, match="invalid update message"):
        call()
    assert sent == []


def test_launch_kills_a_child_that_ignores_termination(tmp_path, monkeypatch):
    events = []

    class Process:
        pid = 2468

        def poll(self):
            return None

        def terminate(self):
            events.append("terminate")

        def wait(self, timeout):
            events.append(("wait", timeout))
            if "kill" not in events:
                raise subprocess.TimeoutExpired("spacr", timeout)
            return -9

        def kill(self):
            events.append("kill")

    def unreadable(self, name):
        raise RuntimeError("acknowledgment unreadable")

    monkeypatch.setattr(ic._UpdateLaunchHandshake, "_read", unreadable)
    plan = {"workdir": str(tmp_path)}
    with pytest.raises(RuntimeError, match="acknowledgment unreadable"):
        ic._launch_updated_desktop(plan, [sys.executable], "2.0.0", str(tmp_path),
                                   spawn=lambda argv, environment, cwd: Process())
    assert events == ["terminate", ("wait", 5), "kill", ("wait", 5)]
