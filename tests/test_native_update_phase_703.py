"""Native preparation uses a private descriptor and authenticated bounded phases."""
import hashlib
import hmac
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from spacr import install_cleanup as ic


@pytest.mark.skipif(os.name == "nt", reason="POSIX owner identity; Windows ACL cases are separate")
def test_real_child_exchanges_staging_and_startup_phases_without_secret_in_argv(tmp_path):
    source = str(Path(ic.__file__).parents[1])
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        descriptor = Path(server.endpoint)
        assert descriptor.stat().st_mode & 0o077 == 0
        script = """import sys
sys.path.insert(0, sys.argv[1])
from spacr.install_cleanup import _UpdatePhaseClient
with _UpdatePhaseClient(sys.argv[2], int(sys.argv[3]), sys.argv[4]) as channel:
    channel._send('prepared', version='1.5.1.5', desktop_pid=int(sys.argv[3]))
    assert channel._receive({'activate'})['phase'] == 'activate'
    channel._send('activated', version='1.5.1.5')
    assert channel._receive({'confirmed'})['phase'] == 'confirmed'
"""
        command = [sys.executable, "-I", "-c", script, source, server.endpoint, str(os.getpid()), sys.executable]
        assert descriptor.read_text().split(":")[1] not in " ".join(command)
        child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            assert server._receive({"prepared"}, timeout=15)["details"] == {
                "version": "1.5.1.5", "desktop_pid": os.getpid()}
            server._send("activate")
            assert server._receive({"activated"}, timeout=15)["details"] == {"version": "1.5.1.5"}
            server._send("confirmed")
            output, error = child.communicate(timeout=15)
            assert child.returncode == 0, (output, error)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=5)
    assert not descriptor.exists()


@pytest.mark.parametrize("fault", ["mode", "owner", "symlink", "oversized", "executable"])
def test_untrusted_descriptor_or_desktop_refuses_before_connecting(tmp_path, monkeypatch, fault):
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        endpoint = Path(server.endpoint)
        executable = sys.executable
        owner = os.getuid()
        if fault == "mode":
            endpoint.chmod(0o644)
        elif fault == "owner":
            owner += 1
        elif fault == "symlink":
            kept = endpoint.with_suffix(".kept")
            endpoint.rename(kept)
            endpoint.symlink_to(kept)
        elif fault == "oversized":
            endpoint.write_bytes(b"x" * 129)
        else:
            executable = str(tmp_path / "different-desktop")
        monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: (owner, executable))
        with pytest.raises(ValueError):
            ic._UpdatePhaseClient(server.endpoint, 1234, sys.executable)
        assert server._connection is None


@pytest.mark.parametrize("fault", ["wrong-secret", "replayed-sequence", "reflected-direction", "unknown-phase", "oversized"])
def test_invalid_authenticated_phase_never_allows_activation(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(ic, "_desktop_process_identity", lambda _pid: (os.getuid(), os.path.realpath(sys.executable)))
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        with ic._UpdatePhaseClient(server.endpoint, 1234, sys.executable) as client:
            payload = {"sequence": 1, "direction": "installer", "phase": "prepared", "details": {}}
            if fault == "replayed-sequence":
                payload["sequence"] = 0
            elif fault == "reflected-direction":
                payload["direction"] = "desktop"
            elif fault == "unknown-phase":
                payload["phase"] = "execute-arbitrary-command"
            encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            secret = b"wrong" if fault == "wrong-secret" else client._secret
            message = json.dumps({"payload": payload, "mac": hmac.new(secret, encoded, hashlib.sha256).hexdigest()}).encode() + b"\n"
            if fault == "oversized":
                message = b"x" * 8193
            client._connection.sendall(message)
            with pytest.raises(ValueError):
                server._receive({"prepared"}, timeout=2)


def test_missing_installer_is_not_treated_as_prepared(tmp_path):
    with ic._UpdatePhaseServer(str(tmp_path)) as server:
        with pytest.raises(RuntimeError, match="stopped"):
            server._receive({"prepared"}, timeout=2, stopped=lambda: True)


@pytest.mark.parametrize("foreign_right", [1, 0x80000000, 0x20000])
def test_windows_phase_descriptor_refuses_foreign_read_grants(monkeypatch, foreign_right):
    owner = "S-1-5-21-123"
    evidence = {"owner": owner, "user": "S-1-5-21-456", "has_dacl": True,
        "rules": [{"sid": owner, "type": 0, "rights": 0x1f01ff, "propagation": 0},
                  {"sid": "S-1-1-0", "type": 0, "rights": foreign_right, "propagation": 0}]}
    monkeypatch.setattr(ic, "_windows_pip_acl", lambda _path: evidence)
    with pytest.raises(ValueError):
        ic._windows_pip_private_access("descriptor", owner_sid=owner, private_read=True)
    evidence["rules"].pop()
    ic._windows_pip_private_access("descriptor", owner_sid=owner, private_read=True)
