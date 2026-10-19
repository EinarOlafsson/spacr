"""Item 703: the restarted desktop proves who it is, and the helper survives removal.

``_begin_update_launch`` runs inside the restarted application, normally in a
child process that coverage never sees; here it runs in this process against
a private launch folder, so each refusal (expired, consumed, wrong version,
wrong executable, not the frozen bundle, not owned by the update) and the
acknowledgment are checked directly. ``_helper_command`` must find an
interpreter that outlives the copy being removed, and the reinstall and
native-uninstall command builders must produce the exact native argv.

Nothing here touches this computer's installations: every path is below
``tmp_path`` and every native command is a recorded fake.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest

import spacr
from spacr import install_cleanup as ic
from tests.test_install_cleanup import Box
from tests.test_install_cleanup import (
    _nothing_here_reaches_this_computer as _nothing_here_reaches_this_computer,
)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX launch folders")

_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(spacr.__file__)))


def _request(tmp_path, monkeypatch, *, frozen=False, version=None, executable=None,
             root=_ROOT, expires=None):
    folder = tempfile.mkdtemp(prefix="launch-", dir=str(tmp_path))
    handshake = ic._UpdateLaunchHandshake(folder, "e" * 64)
    expected = {"version": version or str(spacr.__version__),
                "executable": executable or os.path.abspath(sys.executable),
                "root": root, "frozen": frozen}
    handshake._write("launch-request.json", "requested", details=expected,
                     expires=expires)
    monkeypatch.setenv("SPACR_UPDATE_LAUNCH", json.dumps([folder, "e" * 64]))
    return handshake


def _result(handshake):
    return handshake._read("launch-result.json")


def test_a_normal_launch_has_no_update_to_verify(monkeypatch):
    monkeypatch.delenv("SPACR_UPDATE_LAUNCH", raising=False)
    assert ic._begin_update_launch() is None
    assert ic._finish_update_launch(None) is None


def test_the_verified_frozen_bundle_starts_then_reports_its_window(tmp_path, monkeypatch):
    handshake = _request(tmp_path, monkeypatch, frozen=True)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    launch = ic._begin_update_launch()
    assert "SPACR_UPDATE_LAUNCH" not in os.environ
    same, actual = launch
    assert same.root == handshake.root
    assert actual["pid"] == os.getpid()
    assert actual["module"] == os.path.realpath(spacr.__file__)
    assert _result(handshake)["state"] == "starting"
    ic._finish_update_launch(launch)
    message = _result(handshake)
    assert message["state"] == "ready" and message["details"] == actual


@pytest.mark.parametrize("change, error", [
    ("expired", "expired or missing"),
    ("consumed", "already consumed"),
    ("version", "differs from the verified update"),
    ("executable", "differs from the verified update"),
    ("outside", "differs from the verified update"),
    ("not-frozen", "not the verified frozen bundle"),
])
def test_a_launch_that_does_not_match_its_request_is_refused_and_recorded(
        tmp_path, monkeypatch, change, error):
    options = {"frozen": change == "not-frozen"}
    if change == "expired":
        options["expires"] = time.time() - 1
    if change == "version":
        options["version"] = "0.0.1"
    if change == "executable":
        options["executable"] = str(tmp_path / "other-python")
    if change == "outside":
        options["root"] = str(tmp_path)
    handshake = _request(tmp_path, monkeypatch, **options)
    monkeypatch.delattr(sys, "frozen", raising=False)
    if change == "consumed":
        handshake._write("launch-result.json", "starting")
    with pytest.raises(ValueError, match=error):
        ic._begin_update_launch()
    message = _result(handshake)
    assert message["state"] == "error" and error in message["error"]


def _distribution(location, version, owned_path):
    return SimpleNamespace(metadata={"Name": "spacr"}, version=version,
                           locate_file=lambda name: location, _path=owned_path)


def test_a_pip_launch_must_be_owned_by_exactly_the_updated_distribution(tmp_path, monkeypatch):
    import importlib.metadata

    module = os.path.realpath(spacr.__file__)
    version = str(spacr.__version__)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "prefix", _ROOT)
    owner = _distribution(module, version, os.path.join(_ROOT, "spacr.dist-info"))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: [owner])
    handshake = _request(tmp_path, monkeypatch)
    launch = ic._begin_update_launch()
    assert launch[1]["prefix"] == os.path.realpath(_ROOT)
    assert _result(handshake)["state"] == "starting"

    for owners in ([], [owner, owner],
                   [_distribution(module, "0.0.1", owner._path)],
                   [_distribution(module, version, str(tmp_path / "elsewhere"))]):
        monkeypatch.setattr(importlib.metadata, "distributions", lambda owners=owners: owners)
        handshake = _request(tmp_path, monkeypatch)
        with pytest.raises(ValueError, match="not owned by the verified update"):
            ic._begin_update_launch()
        assert _result(handshake)["state"] == "error"


def test_a_launch_folder_must_stay_the_folder_that_was_checked(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="invalid update launch binding"):
        ic._UpdateLaunchHandshake(str(tmp_path), "E" * 64)
    folder = tmp_path / "launch"
    folder.mkdir(mode=0o700)
    real_lstat = os.lstat
    calls = []

    def moving_lstat(path, *args, **kwargs):
        details = real_lstat(path, *args, **kwargs)
        if sys._getframe(1).f_code is ic._UpdateLaunchHandshake.__init__.__code__:
            calls.append(path)
        if len(calls) == 2:
            return os.stat_result((details.st_mode, details.st_ino + 1) + tuple(details)[2:])
        return details

    monkeypatch.setattr(ic.os, "lstat", moving_lstat)
    with pytest.raises(ValueError, match="changed during validation"):
        ic._UpdateLaunchHandshake(str(folder), "a" * 64)


def _installer(root, *, running=True, platform="linux"):
    return ic.InstallRecord(kind="installer", layout="linux-online", platform=platform,
                            root=str(root), running=running)


def test_a_helper_inside_the_doomed_copy_runs_the_bootstrap_uv_from_a_safe_copy(tmp_path):
    box = Box(tmp_path)
    root = box.home / ".local" / "share" / "spacr"
    uv = box.write(root / "bootstrap" / "uv", "#!/bin/sh\n")
    machine = box.machine("linux", executable=str(root / "venv" / "bin" / "python"))
    work = tmp_path / "work"
    work.mkdir()
    argv, environment, error = ic._helper_command(
        [_installer(root)], str(work), str(work / "install_cleanup.py"),
        str(work / "plan.json"), machine)
    assert error is None
    copy = work / "uv"
    assert copy.read_text(encoding="utf-8") == uv.read_text(encoding="utf-8")
    assert argv[:9] == [str(copy), "run", "--no-project", "--no-config", "--python",
                        "3.12", "--managed-python", "--", "python"]
    assert argv[9:] == ["-I", str(work / "install_cleanup.py"), "run-plan",
                        str(work / "plan.json")]
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(work / "python")
    assert environment["UV_CACHE_DIR"] == str(work / "cache")
    assert environment["HOME"] == str(box.home)


def test_without_any_interpreter_outside_the_copy_the_helper_says_so(tmp_path):
    box = Box(tmp_path)
    root = box.home / ".local" / "share" / "spacr"
    box.write(root / "venv" / "bin" / "python")
    machine = box.machine("linux", executable=str(root / "venv" / "bin" / "python"))
    argv, environment, error = ic._helper_command(
        [_installer(root, running=False)], str(tmp_path), "m.py", "p.json", machine)
    assert argv is None and environment is None
    assert "no Python outside the installation" in error


def test_on_a_real_computer_a_uv_on_path_outside_the_copy_is_used(tmp_path, monkeypatch):
    root = tmp_path / "spacr"
    outside = tmp_path / "tools" / "uv"
    outside.parent.mkdir()
    outside.write_text("uv")
    inside = root / "bin" / "uv"
    machine = ic._Machine(platform="linux", executable=str(root / "venv" / "bin" / "python"),
                          running_prefix=str(root / "venv"), which=lambda name: str(outside))
    assert machine.real
    work = tmp_path / "work"
    work.mkdir()
    argv, _environment, error = ic._helper_command(
        [_installer(root, running=False)], str(work), "m.py", "p.json", machine)
    assert error is None and argv[0] == str(work / "uv")
    machine.which = lambda name: str(inside)
    argv, _environment, error = ic._helper_command(
        [_installer(root, running=False)], str(work), "m.py", "p.json", machine)
    assert argv is None and error


@pytest.mark.parametrize("platform", ["windows", "macos"])
def test_reinstall_steps_use_the_native_installer_and_launcher(tmp_path, platform):
    box = Box(tmp_path)
    root = box.local / "spaCR" if platform == "windows" else box.system("/Applications/spaCR.app")
    record = ic.InstallRecord(kind="installer", layout="x", platform=platform, root=str(root))
    fetch, install, relaunch = ic._reinstall_steps(
        record, "1.5.1.5", str(tmp_path / "work"), box.machine(platform))
    assert fetch["url"].endswith("/v1.5.1.5/" + os.path.basename(fetch["path"]))
    assert os.path.dirname(fetch["path"]) == str(tmp_path / "work")
    if platform == "windows":
        assert install == [fetch["path"], "/S", f"/D={root}"]
        assert relaunch == [os.path.join(str(root), "venv", "Scripts", "pythonw.exe"), "-I",
                            os.path.join(str(root), "launch_spacr.pyw")]
    else:
        assert install == ["open", "-W", fetch["path"]]
        assert relaunch == ["open", "-a", "/Applications/spaCR.app"]


def test_the_native_uninstall_command_is_chosen_by_its_adapter(monkeypatch):
    seen = []
    monkeypatch.setattr(ic, "_macos_native_uninstall_command",
                        lambda adapter, endpoint, pid: ["macos-online", endpoint, pid])
    monkeypatch.setattr(ic, "_macos_frozen_authorized_command",
                        lambda command, arguments: seen.append((command, arguments)) or ["frozen"])
    assert ic._native_uninstall_command({"adapter": "macos-online-native"}, "ep", 7) == [
        "macos-online", "ep", 7]
    frozen = {"adapter": "macos-frozen-native", "desktop_executable": "/A/spaCR",
              "previous_version": "1.5.1.4", "identity": [3, 4], "keep": ["/k"]}
    assert ic._native_uninstall_command(frozen, "ep", 9) == ["frozen"]
    assert seen == [("uninstall-macos-root", [
        "--phase-channel", "ep", "--desktop-pid", "9", "--desktop-executable", "/A/spaCR",
        "--previous-version", "1.5.1.4", "--app-identity", "3:4", "--keep-json", '["/k"]'])]
    debian = {"adapter": "linux-deb-native", "helper": "/opt/spacr/helper",
              "desktop_executable": "/usr/bin/spacr", "desktop_uid": 1000,
              "previous_version": "1.5.1.4"}
    assert ic._native_uninstall_command(debian, "ep", 11) == [
        "/usr/bin/pkexec", "/opt/spacr/helper", "uninstall-debian-root", "--phase-channel",
        "ep", "--desktop-pid", "11", "--desktop-executable", "/usr/bin/spacr",
        "--desktop-uid", "1000", "--previous-version", "1.5.1.4"]
    with pytest.raises(ValueError, match="unsupported_native_uninstall"):
        ic._native_uninstall_command({"adapter": "other"}, "ep", 1)
