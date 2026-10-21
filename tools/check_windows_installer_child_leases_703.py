#!/usr/bin/env python3
"""Exercise native installer leases with actual pip children and Windows APIs.

This tests the installer functions verbatim, independently of package downloads
and GUI acceptance. The orphan case deliberately omits the current Job Object
so that a surviving child exercises compatibility with interrupted older work.
"""
from __future__ import annotations

import ctypes
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WHEEL_NAME = "spacr_lease_probe-0.0.1-py3-none-any.whl"


def load_file(name, path):
    """Load a stdlib-only source module without importing the installed spaCR."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def wheel_bytes():
    """Return a real dependency-free wheel used solely inside the temp root."""
    out = io.BytesIO()
    prefix = "spacr_lease_probe-0.0.1.dist-info"
    files = {
        "spacr_lease_probe.py": "NATIVE_LEASE_PROBE = True\n",
        f"{prefix}/METADATA": "Metadata-Version: 2.1\nName: spacr-lease-probe\nVersion: 0.0.1\n",
        f"{prefix}/WHEEL": "Wheel-Version: 1.0\nGenerator: spacr-native-lease\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    files[f"{prefix}/RECORD"] = "".join(f"{name},,\n" for name in [*files, f"{prefix}/RECORD"])
    with zipfile.ZipFile(out, "w") as archive:
        for name, value in files.items():
            archive.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0)), value)
    return out.getvalue()


class WheelGate:
    """Hold the actual pip HTTP request until the parent permits completion."""

    def __init__(self):
        self.requested = threading.Event()
        self.release = threading.Event()
        self.expired = threading.Event()
        gate = self
        payload = wheel_bytes()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/" + WHEEL_NAME:
                    self.send_error(404)
                    return
                gate.requested.set()
                if not gate.release.wait(180):
                    gate.expired.set()
                    self.send_error(504)
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/{WHEEL_NAME}"

    def close(self):
        """Release blocked requests and close the loopback listener."""
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def installer_functions(source):
    """Extract the unchanged production function block, including lease writers."""
    return source[source.index("function Enable-SpacrInstallerJob {"):source.index("\n$MetadataBackup =")]


DRIVER = r'''
param([string]$Mode, [string]$Root, [string]$Python, [string]$Url, [string]$State)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "messages.ps1")
. (Join-Path $PSScriptRoot "functions.ps1")
$Paths = @((Join-Path $Root ".spacr-installer-lock"), (Join-Path $Root ".venv.spacr-update-lock"))
if ($Mode -eq "probe") {
    $Origin = Get-Content -Raw (Join-Path $Root "origin.json") | ConvertFrom-Json
    $Children = Test-SpacrOwnerChildrenLive $Origin.owner $Origin.started
    $Acquired = @()
    foreach ($Path in $Paths) {
        try {
            $Stream = Open-SpacrInstallerLease $Path
            try { $Acquired += $true } finally { $Stream.Dispose() }
        } catch { $Acquired += $false }
    }
    @{ owner = $PID; children = $Children; acquired = $Acquired } | ConvertTo-Json | Set-Content -Encoding UTF8 $State
    exit 0
}
if ($Mode -eq "job" -and -not (Enable-SpacrInstallerJob)) { throw "Native Job Object setup failed" }
$Streams = @()
foreach ($Path in $Paths) { $Streams += Open-SpacrInstallerLease $Path }
$Arguments = @("-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input", "install",
    "--no-deps", "--no-index", "--no-cache-dir", "--target", ('"' + (Join-Path $Root "packages") + '"'), $Url)
$Pip = Start-Process -FilePath $Python -ArgumentList $Arguments -NoNewWindow -PassThru `
    -RedirectStandardOutput (Join-Path $Root "pip.stdout") -RedirectStandardError (Join-Path $Root "pip.stderr")
@{ owner = $PID; started = (Get-SpacrProcessStarted $PID); pip = $Pip.Id; mode = $Mode } |
    ConvertTo-Json | Set-Content -Encoding UTF8 $State
while (-not $Pip.HasExited) { Start-Sleep -Milliseconds 50; $Pip.Refresh() }
throw "pip exited before the harness interrupted its installer: $($Pip.ExitCode)"
'''


def wait_until(predicate, seconds, label):
    """Wait for an authoritative bounded condition, failing with its identity."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.05)
    raise RuntimeError(f"Timed out waiting for {label}")


def read_json(path):
    """Read a complete state written by PowerShell, tolerating a partial write."""
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def native_pip_handle(pid):
    """Retain the real child process handle and its authoritative exit status."""
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000 | 0x100000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle


def run_scenario(mode, folder, cleanup):
    """Kill an owned installer during real pip work and exercise both lease paths."""
    folder.mkdir()
    driver = folder.parent / "driver.ps1"
    origin = folder / "origin.json"
    gate = WheelGate()
    output = (folder / "driver.log").open("w", encoding="utf-8")
    child = subprocess.Popen(
        ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver),
         "-Mode", mode, "-Root", str(folder), "-Python", sys.executable, "-Url", gate.url, "-State", str(origin)],
        stdout=output, stderr=subprocess.STDOUT,
    )
    owner_started = cleanup._process_started(child.pid)
    pip_pid = None
    pip_stamp = None
    kernel = handle = None
    try:
        wait_until(lambda: read_json(origin) is not None or child.poll() is not None, 40, "installer child startup")
        state = read_json(origin)
        if not state or child.poll() is not None:
            raise RuntimeError((folder / "driver.log").read_text())
        pip_pid = state["pip"]
        pip_stamp = cleanup._process_started(pip_pid)
        kernel, handle = native_pip_handle(pip_pid)
        wait_until(gate.requested.is_set, 40, "actual pip wheel request")
        if child.poll() is not None or cleanup._process_gone(pip_pid) or gate.expired.is_set():
            raise RuntimeError("The actual installer and pip child must both be alive at interruption")
        children = cleanup._windows_children(child.pid)
        if children is None or not any(pid == pip_pid for pid, _ in children):
            raise RuntimeError("ToolHelp32 did not identify the real pip child of this installer")
        if state["owner"] != child.pid or not pip_stamp or not pip_stamp.startswith("filetime:"):
            raise RuntimeError("Native child ownership or process identity is unavailable")
        paths = [folder / ".spacr-installer-lock", folder / ".venv.spacr-update-lock"]
        before = [path.read_bytes() for path in paths]
        child.kill()
        child.wait(timeout=15)

        def probe(expected):
            result = folder / "probe.json"
            result.unlink(missing_ok=True)
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(driver),
                 "-Mode", "probe", "-Root", str(folder), "-State", str(result)],
                stdout=output, stderr=subprocess.STDOUT, check=True, timeout=30,
            )
            verdict = read_json(result)
            if verdict is None or verdict["acquired"] != [expected, expected]:
                raise RuntimeError(f"PowerShell restart lease verdict was {verdict}; expected {expected}")
            return verdict

        if mode == "orphan":
            if kernel.WaitForSingleObject(handle, 0) != 0x102 or cleanup._process_gone(pip_pid):
                raise RuntimeError("The legacy orphan scenario requires a verified surviving real pip process")
            if cleanup._owner_children_live(child.pid, state["started"]) is not True:
                raise RuntimeError("ToolHelp32 did not recognize the live orphan")
            for path in paths:
                try:
                    descriptor = cleanup._take_lease_file(str(path))
                except FileExistsError:
                    pass
                else:
                    os.close(descriptor)
                    raise RuntimeError("Python acquired a lease while its orphan child was alive")
            verdict = probe(False)
            if verdict["children"] is not True or kernel.WaitForSingleObject(handle, 0) != 0x102:
                raise RuntimeError("CIM refusal must occur while the real orphan remains alive")
            if [path.read_bytes() for path in paths] != before or gate.expired.is_set():
                raise RuntimeError("Refusal changed a held lease or the local request gate expired")
            gate.release.set()
        if kernel.WaitForSingleObject(handle, 30000) != 0:
            raise RuntimeError("The actual pip process did not exit within the native bound")
        if not cleanup._process_gone(pip_pid):
            raise RuntimeError("Exited child must be authoritatively gone even while its handle is held")
        if mode == "orphan":
            from ctypes import wintypes
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)) or code.value != 0:
                raise RuntimeError("Released orphan pip must complete its real local installation successfully")
            if not (folder / "packages" / "spacr_lease_probe.py").is_file():
                raise RuntimeError("The orphan did not install the local wheel")
        if any(not cleanup._lease_owner_dead(str(path)) for path in paths):
            raise RuntimeError("Python did not recognize abandoned leases after actual child exit")
        probe(True)
        for path in paths:
            descriptor = cleanup._take_lease_file(str(path))
            os.close(descriptor)
            fields = cleanup._lease_fields(path.read_text(encoding="utf-8"))
            if fields["pid"] != str(os.getpid()):
                raise RuntimeError("Python takeover did not write the actual new owner's identity")
        print(json.dumps({"scenario": mode, "installer_pid": child.pid, "pip_pid": pip_pid,
                          "real_pip_request": True, "both_lease_paths": "passed"}), flush=True)
    except BaseException:
        for name in ("driver.log", "pip.stdout", "pip.stderr"):
            path = folder / name
            if path.is_file():
                print(f"{mode}/{name}:\n{path.read_text(encoding='utf-8', errors='replace')[-12000:]}", file=sys.stderr)
        raise
    finally:
        owned = []
        if owner_started and owner_started.startswith("filetime:"):
            for pid, stamp in cleanup._windows_children(child.pid) or []:
                if stamp.startswith("filetime:") and int(stamp.split(":")[1]) >= int(owner_started.split(":")[1]):
                    owned.append((pid, stamp))
        if pip_pid and pip_stamp:
            owned.append((pip_pid, pip_stamp))
        gate.close()
        if child.poll() is None:
            child.kill()
            child.wait(timeout=15)
        for pid, stamp in owned:
            if cleanup._process_started(pid) == stamp and not cleanup._process_gone(pid):
                os.kill(pid, 15)
                wait_until(lambda: cleanup._process_gone(pid), 15, f"owned child {pid} cleanup")
        if handle:
            kernel.CloseHandle(handle)
        output.close()


def main():
    """Run only on a real Windows host; no emulation or process table substitutions."""
    if os.name != "nt":
        raise SystemExit("Native Windows is required for Job Object/CIM/ToolHelp32 acceptance")
    cleanup = load_file("native_installer_cleanup_703", ROOT / "spacr" / "install_cleanup.py")
    renderer = load_file("native_installer_messages_703", ROOT / "packaging" / "i18n" / "render.py")
    with tempfile.TemporaryDirectory(prefix="spacr-native-child-leases-") as temporary:
        folder = Path(temporary)
        (folder / "functions.ps1").write_text(
            installer_functions((ROOT / "packaging" / "online" / "install_spacr_windows.ps1").read_text(encoding="utf-8")),
            encoding="utf-8-sig",
        )
        (folder / "messages.ps1").write_text(renderer.render_powershell(renderer.catalogs()), encoding="utf-8-sig")
        (folder / "driver.ps1").write_text(DRIVER.lstrip(), encoding="utf-8-sig")
        for mode in ("job", "orphan"):
            run_scenario(mode, folder / mode, cleanup)


if __name__ == "__main__":
    main()
