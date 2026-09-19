"""Item 423: every optional backend in an environment of its own.

Maintainer's decision, 2026-09-19: "Isolated env per backend! But with the
addition of adding cellpose 3 and its cyto, nucleus, and cyto2 and cyto3
models."

What is pinned here, without a network, a GPU or a real package:

1. a backend's STATE -- installed, installable, installing, not installable
   here -- and the reason, from file checks alone;
2. the install PLAN never runs pip against spaCR's own interpreter, and the
   install marks an environment finished only after its self-test loaded the
   package; a failed or cancelled install leaves nothing behind;
3. Cancel stops a real running command, and everything it started;
4. the worker's protocol, answered in process with stand-in packages, every
   failure an error reply carrying the backend's message verbatim;
5. the client against a FAKED environment -- a real venv whose own
   site-packages hold stand-in ``torch`` and ``cellpose`` packages -- so the
   pipes, the ``.npy`` hand-off, a crash, a cancel and a restart are real.

The real proof, a Cellpose 3 environment built from PyPI in a sandboxed HOME
segmenting a synthetic field on the CPU, is recorded in
features/future/423_third_party_segmenters_in_the_model_zoo.txt; it downloads
1.6 GB and is not a unit test.
"""
from __future__ import annotations

import io
import json
import os
import runpy
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

import spacr._segmentation_backends as SB

#: The real one, for teardown: a test may stand in for it.
_SHUTDOWN_WORKERS = SB._shutdown_workers


@pytest.fixture(autouse=True)
def _sandboxed_backends(tmp_path, monkeypatch):
    """Every test gets its own backends folder and forgets earlier probes."""
    root = tmp_path / "backends"
    monkeypatch.setenv(SB._ROOT_ENV, str(root))
    monkeypatch.delenv(SB._TORCH_INDEX_ENV, raising=False)
    monkeypatch.delenv(SB._DEVICE_ENV, raising=False)
    monkeypatch.setattr(SB, "_PROBED", {})
    monkeypatch.setattr(SB, "_CANDIDATES", {})
    yield root
    _SHUTDOWN_WORKERS()


def _finished_env(root, name="cellpose3", record=None):
    """An environment that looks finished: its Python and its marker."""
    env = Path(root) / name
    python = Path(SB._env_python(str(env)))
    python.parent.mkdir(parents=True, exist_ok=True)
    python.write_text("")
    SB._write_marker(str(env), record or {"backend": name})
    return env


# ===========================================================================
# 1. States
# ===========================================================================

def test_a_backend_nobody_installed_is_installable_and_says_where_it_goes(
        _sandboxed_backends):
    state = SB._backend_state("cellpose3")
    assert state.state == SB._INSTALLABLE and not state.ready
    assert state.env == str(_sandboxed_backends / "cellpose3")
    assert "leaves spaCR's own environment alone" in state.reason


def test_a_finished_environment_is_installed_with_its_record(
        _sandboxed_backends):
    _finished_env(_sandboxed_backends, record={"packages": {"cellpose": "3"}})
    state = SB._backend_state("Cellpose3")
    assert state.ready and not state.in_process
    assert state.record["packages"] == {"cellpose": "3"}


def test_an_environment_that_lost_its_python_is_installable_again(
        _sandboxed_backends):
    env = _finished_env(_sandboxed_backends)
    Path(SB._env_python(str(env))).unlink()
    state = SB._backend_state("cellpose3")
    assert state.state == SB._INSTALLABLE
    assert "lost its Python" in state.reason


def test_an_unfinished_environment_is_installable_and_says_so(
        _sandboxed_backends, monkeypatch):
    monkeypatch.setattr(SB, "_importable", lambda module: False)
    (_sandboxed_backends / "samcell").mkdir(parents=True)
    state = SB._backend_state("samcell")
    assert state.state == SB._INSTALLABLE
    assert "did not finish" in state.reason


def test_a_live_lock_means_installing_and_a_dead_one_does_not(
        _sandboxed_backends, monkeypatch):
    root = _sandboxed_backends
    root.mkdir(parents=True)
    lock = root / "cellpose3.lock"
    lock.write_text(json.dumps({"pid": os.getpid(),
                                "host": socket.gethostname(),
                                "started": "noon"}))
    state = SB._backend_state("cellpose3")
    assert state.state == SB._INSTALLING and "noon" in state.reason

    monkeypatch.setattr(SB, "_pid_alive", lambda pid: False)
    assert SB._backend_state("cellpose3").state == SB._INSTALLABLE


def test_a_lock_from_another_computer_is_believed(_sandboxed_backends,
                                                  monkeypatch):
    root = _sandboxed_backends
    root.mkdir(parents=True)
    (root / "cellpose3.lock").write_text(json.dumps(
        {"pid": 1, "host": "not-" + socket.gethostname()}))
    monkeypatch.setattr(SB, "_pid_alive", lambda pid: False)
    assert SB._backend_state("cellpose3").state == SB._INSTALLING


@pytest.mark.parametrize("content", ["not json", "[1, 2]",
                                     json.dumps({"pid": "x"})])
def test_a_damaged_lock_is_no_lock(_sandboxed_backends, content):
    root = _sandboxed_backends
    root.mkdir(parents=True)
    (root / "cellpose3.lock").write_text(content)
    assert SB._backend_state("cellpose3").state == SB._INSTALLABLE


def test_a_damaged_marker_is_no_install(_sandboxed_backends):
    env = _finished_env(_sandboxed_backends)
    (env / SB._MARKER).write_text("[]")
    assert SB._read_marker(str(env)) is None


def test_an_unsupported_platform_is_not_installable_here(monkeypatch):
    monkeypatch.setattr(sys, "platform", "sunos5")
    state = SB._backend_state("cellpose3")
    assert state.state == SB._UNAVAILABLE
    assert "does not run on sunos5" in state.reason


def test_no_python_in_range_is_not_installable_here_and_names_the_range(
        monkeypatch):
    monkeypatch.setattr(SB, "_interpreter_candidates", lambda spec: [])
    state = SB._backend_state("cellpose3")
    assert state.state == SB._UNAVAILABLE
    assert "needs Python 3.9 to 3.12" in state.reason


def test_a_folder_spacr_cannot_write_is_not_installable_here(
        tmp_path, monkeypatch):
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0o500)
    try:
        monkeypatch.setenv(SB._ROOT_ENV, str(locked / "backends"))
        if os.access(locked, os.W_OK):
            pytest.skip("this user can write anywhere")
        state = SB._backend_state("cellpose3")
    finally:
        locked.chmod(0o700)
    assert state.state == SB._UNAVAILABLE
    assert "cannot write" in state.reason and SB._ROOT_ENV in state.reason


def test_a_build_with_no_worker_source_is_not_installable_here(monkeypatch):
    monkeypatch.setattr(SB, "_worker_path", lambda: None)
    assert "no Python source" in SB._backend_state("cellpose3").reason


def test_a_recent_probe_failure_is_reported_and_an_old_one_is_not(monkeypatch):
    SB._PROBED["cellpose3"] = ("no network: nope.", time.time())
    state = SB._backend_state("cellpose3")
    assert (state.state, state.reason) == (SB._UNAVAILABLE, "no network: nope.")
    SB._PROBED["cellpose3"] = ("no network: nope.",
                               time.time() - SB._PROBE_SECONDS - 1)
    assert SB._backend_state("cellpose3").state == SB._INSTALLABLE


def test_an_old_install_inside_spacr_still_counts_but_never_for_cellpose3(
        monkeypatch):
    monkeypatch.setattr(SB, "_importable", lambda module: True)
    samcell = SB._backend_state("samcell")
    assert samcell.ready and samcell.in_process
    assert "pip uninstall samcell" in samcell.reason
    assert not SB._backend_state("cellpose3").ready


def test_importable_answers_without_importing(monkeypatch):
    assert SB._importable("json")
    assert not SB._importable("no_such_package_423")
    monkeypatch.setitem(sys.modules, "hidden_423", None)
    assert not SB._importable("hidden_423")
    broken = types.ModuleType("broken_423")
    monkeypatch.setitem(sys.modules, "broken_423", broken)
    assert not SB._importable("broken_423")


def test_the_spec_refuses_spacrs_own_cellpose():
    with pytest.raises(ValueError, match="not an optional backend"):
        SB._spec("cellpose")


def test_the_root_honours_an_explicit_folder_then_the_variable(
        tmp_path, monkeypatch):
    assert SB._backends_root(tmp_path) == str(tmp_path)
    monkeypatch.delenv(SB._ROOT_ENV)
    assert SB._backends_root().endswith(os.path.join(".spacr", "backends"))


def test_windows_environments_keep_python_in_scripts():
    assert SB._env_python("E", windows=True) == os.path.join(
        "E", "Scripts", "python.exe")
    assert SB._env_python("E", windows=False) == os.path.join(
        "E", "bin", "python")


def test_every_backend_records_its_licence_and_its_python_range():
    for spec in SB._SPECS.values():
        assert spec.licence and spec.licence_note and spec.homepage
        assert spec.python[0] <= spec.python[1]
        assert all("==" in r for r in spec.requirements[:1])
    assert SB._SPECS["cellpose3"].models == ("cyto3", "cyto2", "cyto",
                                             "nuclei")


def test_the_pid_check_uses_psutil_or_signal_zero(monkeypatch):
    assert SB._pid_alive(os.getpid())
    real_import = __import__

    def _no_psutil(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("no psutil here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _no_psutil)
    calls = []
    answers = iter([None, ProcessLookupError(), PermissionError()])

    def _kill(pid, sig):
        calls.append(pid)
        answer = next(answers)
        if answer is not None:
            raise answer

    monkeypatch.setattr(SB.os, "kill", _kill)
    assert SB._pid_alive(10**7) is True
    assert SB._pid_alive(10**7) is False
    assert SB._pid_alive(10**7) is True
    assert calls == [10**7] * 3


def test_psutil_answers_for_another_process():
    pytest.importorskip("psutil")
    assert SB._pid_alive(2**22 + 12345) in (True, False)


# ===========================================================================
# 2. Interpreters, the torch index and the plan
# ===========================================================================

def test_spacrs_own_python_comes_first_when_it_is_in_range():
    spec = SB._SPECS["cellpose3"]
    found = SB._interpreter_candidates(
        spec, executable="/own/python", version=(3, 11), frozen=False,
        which=lambda name: f"/usr/bin/{name}" if name == "python3.11" else None,
        windows=False)
    assert found == [("/own/python",), ("/usr/bin/python3.11",)]
    same = SB._interpreter_candidates(
        spec, executable="/usr/bin/python3.11", version=(3, 11),
        frozen=False, which=lambda name: (
            "/usr/bin/python3.11" if name == "python3.11" else None),
        windows=False)
    assert same == [("/usr/bin/python3.11",)], "listed once"


def test_out_of_range_or_frozen_looks_on_path_newest_first():
    spec = SB._SPECS["cellpose3"]
    found = SB._interpreter_candidates(
        spec, executable="/own/python", version=(3, 13), frozen=False,
        which=lambda name: f"/usr/bin/{name}", windows=False)
    assert found[0] == ("/usr/bin/python3.12",)
    assert ("/own/python",) not in found
    frozen = SB._interpreter_candidates(
        spec, executable="/own/python", version=(3, 12), frozen=True,
        which=lambda name: None, windows=False)
    assert frozen == []


def test_windows_asks_the_py_launcher_for_each_version():
    spec = SB._SPECS["cellpose3"]
    found = SB._interpreter_candidates(
        spec, version=(3, 13), frozen=False,
        which=lambda name: "C:/py.exe" if name == "py" else None,
        windows=True)
    assert found[0] == ("C:/py.exe", "-3.12")
    assert found[-1] == ("C:/py.exe", "-3.9")
    assert SB._interpreter_candidates(
        spec, version=(3, 13), frozen=False, which=lambda name: None,
        windows=True) == []


def test_candidates_are_looked_up_once_per_process(monkeypatch):
    calls = []
    monkeypatch.setattr(SB, "_interpreter_candidates",
                        lambda spec: calls.append(spec.name) or [("p",)])
    spec = SB._SPECS["samcell"]
    assert SB._candidates(spec) == SB._candidates(spec) == [("p",)]
    assert calls == ["samcell"]


@pytest.mark.parametrize("version, index", [
    ("2.5.1+cu124", SB._TORCH_WHEELS + "cu124"),
    ("2.5.1+cpu", SB._TORCH_WHEELS + "cpu"),
    ("2.5.1+rocm6.2", SB._TORCH_WHEELS + "rocm6.2"),
    ("2.13.0", None), ("2.13.0+local", None)])
def test_the_backend_gets_the_kind_of_torch_spacr_has(version, index):
    assert SB._torch_index_url(version) == index


def test_the_torch_index_can_be_overridden_or_sent_to_pypi(monkeypatch):
    monkeypatch.setenv(SB._TORCH_INDEX_ENV, "https://mirror/whl")
    assert SB._torch_index_url("2.0+cu118") == "https://mirror/whl"
    monkeypatch.setenv(SB._TORCH_INDEX_ENV, "pypi")
    assert SB._torch_index_url("2.0+cu118") is None


def test_the_torch_index_reads_spacrs_torch_or_none(monkeypatch):
    import importlib.metadata as metadata

    monkeypatch.setattr(metadata, "version", lambda name: "9.9+cpu")
    assert SB._torch_index_url() == SB._TORCH_WHEELS + "cpu"

    def _missing(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", _missing)
    assert SB._torch_index_url() is None


def test_the_plan_runs_pip_only_inside_the_environment():
    spec = SB._SPECS["dinocell"]
    env = "/b/dinocell"
    steps = SB._install_plan(spec, env, ("/own/python",),
                             torch_index="https://t/cpu", worker="/w.py")
    python = SB._env_python(env)
    assert steps[0].argv == ("/own/python", "-m", "venv", env)
    for step in steps[1:]:
        assert step.argv[0] == python, step
    torch = steps[1].argv
    assert torch[-2:] == ("--index-url", "https://t/cpu")
    assert "torch==2.10.0" in torch and "torchvision==0.25.0" in torch
    assert "dinocell==0.74" in steps[2].argv
    assert steps[3].selftest and steps[3].argv[1:] == (
        "-I", "/w.py", "--selftest", "dinocell")
    plain = SB._install_plan(SB._SPECS["cellpose3"], env, ("/p",))
    assert "--index-url" not in plain[1].argv
    assert plain[3].argv[2] == SB._worker_path()
    import dataclasses

    no_torch = dataclasses.replace(SB._SPECS["samcell"], torch=())
    assert [s.label for s in SB._install_plan(no_torch, env, ("/p",))] == [
        "Create the environment", "Install SAMCell", "Check it loads"]


def test_the_environment_variables_cannot_point_pip_elsewhere(monkeypatch):
    for name in SB._STRIPPED_VARIABLES:
        monkeypatch.setenv(name, "elsewhere")
    environ = SB._worker_env("cellpose3", "/b/cellpose3")
    assert not set(SB._STRIPPED_VARIABLES) & set(environ)
    assert environ["PYTHONNOUSERSITE"] == "1"
    assert environ["VIRTUAL_ENV"] == "/b/cellpose3"
    assert environ["PATH"].startswith(
        os.path.dirname(SB._env_python("/b/cellpose3")))
    assert environ["CELLPOSE_LOCAL_MODELS_PATH"] == os.path.join(
        "/b/cellpose3", "models")
    assert "CELLPOSE_LOCAL_MODELS_PATH" not in SB._worker_env("samcell", "/b")


def test_children_get_their_own_process_group_on_every_system(monkeypatch):
    assert SB._detached(windows=False) == {"start_new_session": True}
    monkeypatch.setattr(SB.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200,
                        raising=False)
    monkeypatch.setattr(SB.subprocess, "CREATE_NO_WINDOW", 0x8000000,
                        raising=False)
    assert SB._detached(windows=True) == {"creationflags": 0x8000200}


# ===========================================================================
# 3. Running a step, and cancelling it
# ===========================================================================

def test_a_step_streams_every_line_and_returns_its_exit_code():
    seen = []
    code, tail = SB._run_step(
        [sys.executable, "-c",
         "import sys; print('one', flush=True); "
         "print('two', file=sys.stderr, flush=True); sys.exit(3)"],
        on_line=seen.append)
    assert code == 3
    assert seen == tail == ["one", "two"]


def test_a_step_waits_for_its_process_after_its_output_ends():
    code, tail = SB._run_step(
        [sys.executable, "-c",
         "import os, time; print('last words', flush=True); os.close(1); "
         "os.close(2); time.sleep(0.5)"],
        poll=0.05)
    assert (code, tail) == (0, ["last words"])


def test_cancel_stops_the_command_and_what_it_started(tmp_path):
    marker = tmp_path / "child.pid"
    script = textwrap.dedent(f"""
        import subprocess, sys, time
        child = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(60)"])
        open({str(marker)!r}, "w").write(str(child.pid))
        print("started", flush=True)
        time.sleep(60)
    """)
    cancel = threading.Event()
    started = time.monotonic()

    def _on_line(text):
        if text == "started":
            cancel.set()

    with pytest.raises(SB._InstallCancelled):
        SB._run_step([sys.executable, "-c", script], on_line=_on_line,
                     cancel=cancel, poll=0.05)
    assert time.monotonic() - started < 30
    child = int(marker.read_text())
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        try:
            os.waitpid(child, os.WNOHANG)
        except ChildProcessError:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("the grandchild outlived the cancel")


class _Proc:
    """A process that ignores SIGTERM, for the SIGKILL branch."""

    def __init__(self, exited=False):
        self.pid = 4242
        self.exited = exited
        self.waits = 0

    def poll(self):
        return 0 if self.exited else None

    def wait(self, timeout=None):
        self.waits += 1
        if self.waits == 1:
            raise subprocess.TimeoutExpired("x", timeout)
        return -9


def test_kill_tree_escalates_when_asking_is_not_enough(monkeypatch):
    sent = []
    monkeypatch.setattr(SB.os, "killpg", lambda pid, sig: sent.append(sig))
    proc = _Proc()
    SB._kill_tree(proc, grace=0.01, windows=False)
    assert sent == [signal.SIGTERM, signal.SIGKILL]
    SB._kill_tree(_Proc(exited=True))


def test_kill_tree_survives_a_process_that_is_already_gone(monkeypatch):
    def _gone(pid, sig):
        raise ProcessLookupError()

    monkeypatch.setattr(SB.os, "killpg", _gone)
    SB._kill_tree(_Proc(), grace=0.01, windows=False)


def test_kill_tree_uses_taskkill_on_windows(monkeypatch):
    ran = []
    monkeypatch.setattr(SB.subprocess, "run",
                        lambda argv, **kw: ran.append(argv))
    proc = _Proc()
    proc.kill = lambda: ran.append("kill")
    SB._kill_tree(proc, grace=0.01, windows=True)
    assert ran[0][:4] == ["taskkill", "/F", "/T", "/PID"]
    assert ran[1] == "kill"


def test_a_pump_ends_on_a_closed_stream():
    class _Closed:
        def __iter__(self):
            raise ValueError("I/O operation on closed file")

    ended = []
    SB._pump(_Closed(), ended.append, lambda: ended.append("done"))
    assert ended == ["done"]


def test_a_command_reads_as_typed_and_its_reply_is_found():
    assert SB._quote(["a b", "c"]) == "'a b' c"
    assert SB._last_reply(["noise", '{"ok": true}', "{bad", "[1]"]) == {
        "ok": True}
    assert SB._last_reply(["nothing"]) is None


# ===========================================================================
# 4. Install, uninstall and their guards
# ===========================================================================

def _fake_runner(record, fail_at=None, hello=None, cancel_at=None):
    """A runner that pretends each step ran and succeeded."""
    def _run(argv, env=None, cwd=None, on_line=None, cancel=None):
        record.append((tuple(argv), env, cwd))
        index = len(record) - 1
        if index == cancel_at:
            raise SB._InstallCancelled("cancelled")
        if argv[1:3] == ("-m", "venv"):
            python = Path(SB._env_python(argv[3]))
            python.parent.mkdir(parents=True)
            python.write_text("")
        on_line(f"line from step {index}")
        if index == fail_at:
            return 1, ["pip said:", "ERROR: no matching distribution"]
        if "--selftest" in argv:
            reply = hello if hello is not None else {
                "ok": True, "python": "3.12.0",
                "packages": {"cellpose": "3.1.1.3"}, "device": "cpu"}
            return 0, [json.dumps(reply)]
        return 0, ["ok"]
    return _run


def _no_preflight(spec, root):
    return (sys.executable,)


def test_an_install_builds_the_environment_and_marks_it_finished_last(
        _sandboxed_backends):
    record, progress = [], []
    state = SB._install_backend(
        "cellpose3", runner=_fake_runner(record), preflight=_no_preflight,
        progress=lambda *a: progress.append(a), torch_index="")
    assert state.ready
    assert state.record["packages"] == {"cellpose": "3.1.1.3"}
    assert state.record["licence"] == "BSD-3-Clause"
    assert [r[0][1:3] for r in record][:1] == [("-m", "venv")]
    assert all(r[2] == str(_sandboxed_backends) for r in record)
    assert progress[0] == (0, 1, "Checking this computer can install it")
    assert progress[-1] == (4, 4, "Cellpose 3 is installed")
    assert (0, 4, "Create the environment: line from step 0") in progress
    assert not (_sandboxed_backends / "cellpose3.lock").exists()
    log = (_sandboxed_backends / "cellpose3.log").read_text()
    assert "line from step 3" in log and "--selftest" in log
    again = SB._install_backend("cellpose3", runner=None, preflight=None)
    assert again.ready, "an installed backend is not installed twice"


def test_a_blocked_install_leaves_no_folder_and_releases_its_lock(
        _sandboxed_backends):
    def _blocked(spec, root):
        raise SB._InstallBlocked("no network: down")

    with pytest.raises(SB._InstallBlocked, match="no network"):
        SB._install_backend("cellpose3", preflight=_blocked)
    assert not (_sandboxed_backends / "cellpose3").exists()
    assert not (_sandboxed_backends / "cellpose3.lock").exists()


def test_an_unfinished_folder_is_replaced_not_reused(_sandboxed_backends):
    leftover = _sandboxed_backends / "cellpose3" / "leftover.txt"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("old")
    SB._install_backend("cellpose3", runner=_fake_runner([]),
                        preflight=_no_preflight, torch_index="")
    assert not leftover.exists()


def test_a_failed_step_is_reported_verbatim_and_leaves_nothing(
        _sandboxed_backends):
    with pytest.raises(SB._InstallFailed) as exc:
        SB._install_backend("cellpose3", runner=_fake_runner([], fail_at=2),
                            preflight=_no_preflight, torch_index="")
    message = str(exc.value)
    assert message.startswith("Install Cellpose 3 failed: `")
    assert "exited with code 1" in message
    assert "ERROR: no matching distribution" in message
    assert "cellpose3.log" in message
    assert not (_sandboxed_backends / "cellpose3").exists()
    assert not (_sandboxed_backends / "cellpose3.lock").exists()
    assert SB._backend_state("cellpose3").state == SB._INSTALLABLE


def test_a_package_that_does_not_load_fails_the_install(_sandboxed_backends):
    hello = {"ok": False, "error": {"type": "ModuleNotFoundError",
                                    "message": "No module named 'packaging'",
                                    "traceback": "Traceback ..."}}
    with pytest.raises(SB._InstallFailed,
                       match="does not load in it: ModuleNotFoundError No "
                             "module named 'packaging'"):
        SB._install_backend("cellpose3", runner=_fake_runner([], hello=hello),
                            preflight=_no_preflight, torch_index="")
    assert not (_sandboxed_backends / "cellpose3").exists()


def test_a_cancelled_install_leaves_nothing(_sandboxed_backends):
    with pytest.raises(SB._InstallCancelled):
        SB._install_backend("cellpose3",
                            runner=_fake_runner([], cancel_at=2),
                            preflight=_no_preflight, torch_index="")
    assert not (_sandboxed_backends / "cellpose3").exists()
    assert SB._backend_state("cellpose3").state == SB._INSTALLABLE


def test_a_folder_that_will_not_go_is_logged_not_raised_over_the_failure(
        _sandboxed_backends, monkeypatch):
    def _stuck(path, root):
        raise OSError("busy")

    monkeypatch.setattr(SB, "_remove_tree", _stuck)
    with pytest.raises(SB._InstallFailed):
        SB._install_backend("cellpose3", runner=_fake_runner([], fail_at=1),
                            preflight=_no_preflight, torch_index="")


def test_an_install_uses_the_torch_index_spacr_has_by_default(
        _sandboxed_backends, monkeypatch):
    monkeypatch.setattr(SB, "_torch_index_url", lambda: "https://t/cu999")
    record = []
    SB._install_backend("cellpose3", runner=_fake_runner(record),
                        preflight=_no_preflight)
    assert record[1][0][-2:] == ("--index-url", "https://t/cu999")


def test_a_second_install_is_refused_while_one_runs(_sandboxed_backends):
    SB._acquire_lock(str(_sandboxed_backends), "cellpose3")
    with pytest.raises(SB._InstallFailed, match="already being installed"):
        SB._acquire_lock(str(_sandboxed_backends), "cellpose3")
    SB._release_lock(str(_sandboxed_backends), "cellpose3")
    SB._release_lock(str(_sandboxed_backends), "cellpose3")


def test_a_stale_lock_is_taken_over(_sandboxed_backends, monkeypatch):
    root = _sandboxed_backends
    root.mkdir(parents=True)
    (root / "cellpose3.lock").write_text(json.dumps({"pid": 1}))
    monkeypatch.setattr(SB, "_pid_alive", lambda pid: pid == os.getpid())
    SB._acquire_lock(str(root), "cellpose3")
    assert json.loads((root / "cellpose3.lock").read_text())["pid"] == \
        os.getpid()


def test_a_lock_lost_in_a_race_or_unwritable_is_refused(_sandboxed_backends,
                                                        monkeypatch):
    root = str(_sandboxed_backends)

    def _raced(*args):
        raise FileExistsError("raced")

    monkeypatch.setattr(SB.os, "open", _raced)
    with pytest.raises(SB._InstallFailed, match="already being installed"):
        SB._acquire_lock(root, "cellpose3")

    def _denied(*args):
        raise PermissionError("denied")

    monkeypatch.setattr(SB.os, "open", _denied)
    with pytest.raises(SB._InstallBlocked, match="cannot write"):
        SB._acquire_lock(root, "cellpose3")


def test_a_backends_folder_that_cannot_exist_blocks_the_install(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("in the way")
    with pytest.raises(SB._InstallBlocked, match="cannot create"):
        SB._acquire_lock(str(blocker / "backends"), "cellpose3")


def test_uninstall_removes_the_environment_and_its_workers(
        _sandboxed_backends, monkeypatch):
    _finished_env(_sandboxed_backends)
    closed = []
    monkeypatch.setattr(SB, "_shutdown_workers", closed.append)
    SB._PROBED["cellpose3"] = ("old", time.time())
    state = SB._uninstall_backend("cellpose3")
    assert state.state == SB._INSTALLABLE
    assert not (_sandboxed_backends / "cellpose3").exists()
    assert closed == ["cellpose3"] and "cellpose3" not in SB._PROBED


def test_uninstall_refuses_while_installing_and_inside_spacr(
        _sandboxed_backends, monkeypatch):
    SB._acquire_lock(str(_sandboxed_backends), "cellpose3")
    with pytest.raises(RuntimeError, match="Cancel that install first"):
        SB._uninstall_backend("cellpose3")
    SB._release_lock(str(_sandboxed_backends), "cellpose3")
    monkeypatch.setattr(SB, "_importable", lambda module: True)
    with pytest.raises(RuntimeError, match="pip uninstall samcell"):
        SB._uninstall_backend("samcell")


def test_nothing_outside_the_backends_folder_is_ever_deleted(tmp_path):
    root = tmp_path / "backends"
    with pytest.raises(RuntimeError, match="refusing to delete"):
        SB._remove_tree(str(tmp_path / "elsewhere" / "x"), str(root))
    with pytest.raises(RuntimeError, match="refusing to delete"):
        SB._remove_tree(sys.prefix, os.path.dirname(sys.prefix))
    SB._remove_tree(str(root / "missing"), str(root))


def test_a_read_only_file_is_made_writable_and_removed(tmp_path, monkeypatch):
    root = tmp_path / "backends"
    env = root / "env"
    env.mkdir(parents=True)
    stubborn = env / "stubborn"
    stubborn.write_text("x")
    handlers = []
    real = SB.shutil.rmtree

    def _rmtree(path, **kwargs):
        handler = kwargs.get("onexc") or kwargs.get("onerror")
        handlers.append(handler)
        stubborn.chmod(0o400)
        handler(os.remove, str(stubborn), None)
        real(path)

    monkeypatch.setattr(SB.shutil, "rmtree", _rmtree)
    SB._remove_tree(str(env), str(root))
    assert handlers and not env.exists()
    monkeypatch.setattr(SB.sys, "version_info", (3, 11, 0))
    env.mkdir()
    stubborn.write_text("x")
    SB._remove_tree(str(env), str(root))
    assert not env.exists()


# ===========================================================================
# 5. The preflight and the network probe
# ===========================================================================

class _Done:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def test_the_preflight_picks_the_first_python_that_can_build_one(
        _sandboxed_backends, monkeypatch):
    root = _sandboxed_backends
    root.mkdir(parents=True)
    candidates = [("/a",), ("/b",), ("/c",), ("/d",), ("/e",), ("/f",)]
    monkeypatch.setattr(SB, "_interpreter_candidates", lambda spec: candidates)
    answers = {
        "/a": OSError("not found"),
        "/b": _Done(1, "", "ModuleNotFoundError: No module named 'ensurepip'"),
        "/c": _Done(0, "garbage"),
        "/d": _Done(0, "3.13"),
        "/e": _Done(1, "", ""),
        "/f": _Done(0, "3.12"),
    }

    def _run(argv, **kwargs):
        answer = answers[argv[0]]
        if isinstance(answer, Exception):
            raise answer
        return answer

    chosen = SB._preflight(SB._SPECS["cellpose3"], str(root), run=_run,
                           probe=lambda: "")
    assert chosen == ("/f",)

    del answers["/f"]
    candidates.pop()
    with pytest.raises(SB._InstallBlocked) as exc:
        SB._preflight(SB._SPECS["cellpose3"], str(root), run=_run,
                      probe=lambda: "")
    message = str(exc.value)
    assert "no Python 3.9 to 3.12" in message
    assert "python3-venv package" in message
    assert "/d is Python 3.13" in message and "did not say its version" in message
    assert "exit code 1" in message
    assert SB._backend_state("cellpose3").reason == message


def test_the_preflight_reports_no_candidates_at_all(_sandboxed_backends,
                                                    monkeypatch):
    _sandboxed_backends.mkdir(parents=True)
    monkeypatch.setattr(SB, "_interpreter_candidates", lambda spec: [])
    with pytest.raises(SB._InstallBlocked, match=r"was found\.$"):
        SB._preflight(SB._SPECS["cellpose3"], str(_sandboxed_backends),
                      probe=lambda: "")


def test_the_preflight_stops_at_the_first_thing_wrong(tmp_path, monkeypatch):
    root = tmp_path / "backends"
    with pytest.raises(SB._InstallBlocked, match="cannot write"):
        SB._preflight(SB._SPECS["cellpose3"], str(root))
    root.mkdir()
    monkeypatch.setattr(SB.shutil, "disk_usage",
                        lambda path: types.SimpleNamespace(free=2 ** 29))
    with pytest.raises(SB._InstallBlocked, match="needs about 2 GB free"):
        SB._preflight(SB._SPECS["cellpose3"], str(root))
    monkeypatch.setattr(SB.shutil, "disk_usage",
                        lambda path: types.SimpleNamespace(free=2 ** 40))
    with pytest.raises(SB._InstallBlocked, match="no network: down"):
        SB._preflight(SB._SPECS["cellpose3"], str(root),
                      probe=lambda: "no network: down")
    assert SB._PROBED["cellpose3"][0] == "no network: down"


def test_the_network_probe_counts_any_answer_and_names_the_host(monkeypatch):
    import urllib.error

    class _Ok:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    assert SB._probe_network(opener=lambda request, timeout: _Ok()) == ""

    def _forbidden(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "no", {}, None)

    assert SB._probe_network(opener=_forbidden) == ""

    def _down(request, timeout):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setenv("PIP_INDEX_URL", "https://user:pw@mirror.lab/simple")
    reason = SB._probe_network(opener=_down)
    assert reason.startswith("no network: pip's index at mirror.lab")
    assert "pw" not in reason


def test_a_background_probe_marks_and_clears_rows(monkeypatch):
    blocked = SB._probe_blockers(probe=lambda: "no network: down")
    assert set(blocked) >= {"cellpose3", "dinocell"}
    assert SB._backend_state("cellpose3").reason == "no network: down"
    SB._PROBED["dinocell"] = ("no Python 3.11 to 3.14 ...", time.time())
    assert SB._probe_blockers(names=["cellpose3", "dinocell"],
                              probe=lambda: "") == {}
    assert "cellpose3" not in SB._PROBED
    assert "dinocell" in SB._PROBED, "only network reasons are cleared"


def test_a_probe_with_nothing_to_install_asks_nothing(_sandboxed_backends):
    _finished_env(_sandboxed_backends)

    def _never():
        raise AssertionError("probed although nothing waits")

    assert SB._probe_blockers(names=["cellpose3"], probe=_never) == {}


# ===========================================================================
# 6. The worker, answered in process
# ===========================================================================

_STUB_TORCH = '''
class device:
    def __init__(self, name):
        self.name = str(name)
        self.type = self.name.split(":")[0]

class cuda:
    available = False

    @staticmethod
    def is_available():
        return cuda.available

class backends:
    class mps:
        available = False

        @staticmethod
        def is_available():
            return backends.mps.available
'''

_STUB_MODELS = '''
import os
import sys
import time

import numpy as np

BUILT = []


class _Base:
    def eval(self, image, channels=None, channel_axis=None, diameter=None,
             **kwargs):
        self.calls.append(dict(kwargs, channels=channels,
                               channel_axis=channel_axis, diameter=diameter,
                               shape=np.shape(image)))
        image = np.asarray(image, dtype=float)
        if image.ndim == 3:
            image = np.take(image, 0, axis=channel_axis)
        marker = float(image.flat[0])
        if marker == 7.0:
            sys.stderr.write("the stand-in is going down\\n")
            sys.stderr.flush()
            os._exit(3)
        if marker == 8.0:
            time.sleep(60)
        if marker == 9.0:
            raise RuntimeError("the stand-in refused this image")
        if marker == 6.0:
            print("a library printing to stdout", flush=True)
        labels = (image > 0.5).astype(np.int32) * 5
        rgb = np.zeros(image.shape + (3,), np.uint8)
        return labels, [rgb, np.zeros((2,) + image.shape, np.float32),
                        image.astype(np.float32)], None, 30.0


class Cellpose(_Base):
    def __init__(self, gpu=False, model_type="cyto3", device=None):
        self.calls = []
        self.model_type = model_type
        self.gpu = gpu
        BUILT.append(self)


class CellposeModel(_Base):
    def __init__(self, gpu=False, pretrained_model=None, device=None):
        self.calls = []
        self.pretrained_model = pretrained_model
        BUILT.append(self)
'''


def _module(name, source):
    module = types.ModuleType(name)
    exec(compile(source, f"<stub {name}>", "exec"), module.__dict__)
    return module


@pytest.fixture
def stub_cellpose(monkeypatch):
    """Stand-in torch and cellpose.models in this process."""
    torch = _module("torch", _STUB_TORCH)
    models = _module("cellpose.models", _STUB_MODELS)
    cellpose = types.ModuleType("cellpose")
    cellpose.__path__ = []
    cellpose.models = models
    for name in [m for m in sys.modules if m == "cellpose"
                 or m.startswith("cellpose.") or m == "torch"
                 or m.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "cellpose", cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", models)
    return types.SimpleNamespace(torch=torch, models=models)


def _ask(requests):
    """Serve a list of requests in process; return the replies."""
    stdin = io.StringIO("\n".join(
        r if isinstance(r, str) else json.dumps(r) for r in requests) + "\n")
    stdout = io.StringIO()
    assert SB._serve("cellpose3", stdin, stdout) == 0
    return [json.loads(line) for line in stdout.getvalue().splitlines()]


def _request(op, ident=1, **payload):
    return dict(payload, protocol=SB._PROTOCOL, id=ident, op=op)


def test_the_worker_says_hello_with_its_versions_and_device(stub_cellpose):
    [reply] = _ask([_request("hello")])
    assert reply["ok"] and reply["id"] == 1
    assert reply["protocol"] == SB._PROTOCOL
    assert reply["backend"] == "cellpose3"
    assert reply["device"] == "cpu"
    assert reply["models"] == ["cyto3", "cyto2", "cyto", "nuclei"]
    assert set(reply["packages"]) == {"cellpose", "torch", "numpy"}


def test_a_package_without_metadata_reports_a_blank_version(
        stub_cellpose, monkeypatch):
    import importlib.metadata as metadata

    real = metadata.version

    def _version(name):
        if name == "torch":
            raise metadata.PackageNotFoundError(name)
        return real(name)

    monkeypatch.setattr(metadata, "version", _version)
    assert SB._worker_hello("cellpose3")["packages"]["torch"] == ""


def test_every_failure_is_an_error_reply_and_the_worker_goes_on(
        stub_cellpose):
    replies = _ask([
        "not json", "", "[1, 2]",
        dict(_request("hello", 2), protocol=99),
        _request("dance", 3),
        _request("hello", 4),
    ])
    assert [r["ok"] for r in replies] == [False, False, False, False, True]
    assert replies[0]["error"]["message"] == (
        "a request is one JSON object per line")
    assert "spaCR sent protocol 99" in replies[2]["error"]["message"]
    assert replies[3]["error"] == {**replies[3]["error"],
                                   "type": "ValueError",
                                   "message": "unknown request 'dance'"}
    assert "Traceback" in replies[3]["error"]["traceback"]


def test_shutdown_ends_the_loop(stub_cellpose):
    replies = _ask([_request("shutdown", 5), _request("hello", 6)])
    assert replies == [{"protocol": SB._PROTOCOL, "id": 5, "ok": True}]


def test_a_segment_request_writes_masks_and_flows_beside_the_images(
        stub_cellpose, tmp_path):
    image = np.zeros((6, 8), np.float32)
    image[2:4, 2:5] = 1.0
    paired = np.zeros((6, 8, 2), np.float32)
    paired[1:3, 1:3, 0] = 1.0
    np.save(tmp_path / "a.npy", image)
    np.save(tmp_path / "b.npy", paired)
    request = _request("segment", model="nuclei", device="auto",
                       inputs=[str(tmp_path / "a.npy"),
                               str(tmp_path / "b.npy")],
                       outputs=str(tmp_path),
                       params={"channel_axis": -1, "diameter": 12.0,
                               "min_size": 3})
    first, second = _ask([request, dict(request, id=2)])
    assert first["ok"] and second["ok"] and first["device"] == "cpu"
    assert len(stub_cellpose.models.BUILT) == 1, "the model loads once"
    model = stub_cellpose.models.BUILT[0]
    assert model.model_type == "nuclei"
    calls = model.calls[:2]
    assert calls[0]["channels"] == [0, 0] and calls[0]["channel_axis"] is None
    assert calls[1]["channels"] == [1, 2] and calls[1]["channel_axis"] == -1
    assert calls[0]["diameter"] == 12.0 and calls[0]["min_size"] == 3
    mask = np.load(first["outputs"][0]["mask"])
    assert mask.dtype == np.uint16 and mask.max() == 1
    assert int((mask > 0).sum()) == 6
    flows = first["outputs"][0]["flows"]
    assert np.load(flows[1]).shape == (2, 6, 8) and flows[3] is None


def test_the_worker_saves_only_the_flows_a_backend_has(tmp_path):
    class _Adapter:
        def eval(self, images, **params):
            return ([np.ones((2, 2), np.uint16)] * len(images),
                    [[None, "not an array"]], None)

    np.save(tmp_path / "a.npy", np.zeros((2, 2)))
    np.save(tmp_path / "b.npy", np.zeros((2, 2)))
    adapters = {("", "cpu", "{}"): _Adapter()}
    body = SB._worker_segment(
        "samcell", {"device": "cpu", "inputs": [str(tmp_path / "a.npy"),
                                                str(tmp_path / "b.npy")],
                    "outputs": str(tmp_path)}, adapters)
    assert [o["flows"] for o in body["outputs"]] == [[None] * 4, [None] * 4]


def test_the_device_is_the_one_asked_for_else_the_best_there_is(
        stub_cellpose):
    torch = stub_cellpose.torch
    assert SB._worker_device("cuda:1") == "cuda:1"
    assert SB._worker_device("auto") == "cpu"
    torch.backends.mps.available = True
    assert SB._worker_device(None) == "mps"
    torch.cuda.available = True
    assert SB._worker_device("") == "cuda"


def test_cellpose3_refuses_what_is_not_a_model(stub_cellpose, tmp_path):
    checkpoint = tmp_path / "mine.pth"
    checkpoint.write_bytes(b"weights")
    built = SB._Cellpose3Adapter(str(checkpoint), "cuda")
    assert built._model.pretrained_model == str(checkpoint)
    assert not built._sized
    masks, flows, styles = built.eval([np.zeros((4, 4))], diameter=None)
    assert built._model.calls[-1]["diameter"] is None
    with pytest.raises(FileNotFoundError, match="without a word"):
        SB._Cellpose3Adapter("cpsam", "cpu")
    with pytest.raises(ValueError, match="segments 2-D images"):
        SB._Cellpose3Adapter("cyto3", "cpu").eval([np.zeros((2, 2, 2, 2))])
    single = SB._Cellpose3Adapter("cyto3", "cpu")
    single.eval([np.zeros((4, 4, 1))], channel_axis=None)
    assert single._model.calls[-1]["channels"] == [0, 0]
    assert single._model.calls[-1]["diameter"] == 0.0


def test_other_backends_are_built_from_their_in_process_class(monkeypatch):
    class _Plane:
        def __init__(self, device=None, **options):
            self.device, self.options = device, options

    monkeypatch.setitem(SB._BACKEND_CLASSES, "samcell", _Plane)
    built = SB._worker_adapter("samcell", "", "cpu", {"variant": "cyto"})
    assert (built.device, built.options) == ("cpu", {"variant": "cyto"})


def test_the_worker_command_line(stub_cellpose, capsys):
    assert SB._worker_main(["--serve"]) == 2
    assert "usage" in capsys.readouterr().err
    out = io.StringIO()
    assert SB._worker_main(["--selftest", "cellpose3"], stdout=out) == 0
    assert json.loads(out.getvalue())["ok"]
    assert SB._worker_main(["--selftest", "cellpose3"]) == 0
    assert json.loads(capsys.readouterr().out)["backend"] == "cellpose3"
    served = io.StringIO()
    assert SB._worker_main(
        ["--serve", "cellpose3"],
        stdin=io.StringIO(json.dumps(_request("hello")) + "\n"),
        stdout=served) == 0
    assert json.loads(served.getvalue())["ok"]


def test_a_selftest_that_cannot_import_fails(monkeypatch):
    monkeypatch.setitem(sys.modules, "cellpose.models", None)
    out = io.StringIO()
    assert SB._worker_main(["--selftest", "cellpose3"], stdout=out) == 1
    assert json.loads(out.getvalue())["error"]["type"] == "ImportError" \
        or json.loads(out.getvalue())["error"]["type"] == "ModuleNotFoundError"


def test_the_worker_serves_on_stdin_and_a_private_copy_of_stdout(
        stub_cellpose, monkeypatch):
    duplicated, rewired, opened = [], [], []
    monkeypatch.setattr(SB.os, "dup", lambda fd: duplicated.append(fd) or 99)
    monkeypatch.setattr(SB.os, "dup2",
                        lambda a, b: rewired.append((a, b)))
    channel = io.StringIO()
    monkeypatch.setattr(SB.os, "fdopen",
                        lambda fd, *a, **k: opened.append(fd) or channel)
    monkeypatch.setattr(sys, "stdin",
                        io.StringIO(json.dumps(_request("hello")) + "\n"))
    stdout, stderr = sys.stdout, sys.stderr
    try:
        assert SB._worker_main(["--serve", "cellpose3"]) == 0
        assert sys.stdout is sys.stderr, "a library's print goes to stderr"
    finally:
        sys.stdout = stdout
    assert (duplicated, rewired, opened) == ([1], [(2, 1)], [99])
    assert json.loads(channel.getvalue())["ok"]
    assert sys.stderr is stderr


def test_the_file_runs_as_the_worker_script(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["_segmentation_backends.py"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(SB.__file__, run_name="__main__")
    assert exc.value.code == 2


# ===========================================================================
# 7. The client against a faked environment
# ===========================================================================

@pytest.fixture(scope="module")
def fake_env(tmp_path_factory):
    """A real venv, standing in for ``~/.spacr/backends/cellpose3``.

    Built from this interpreter with its site-packages visible (for numpy),
    and with stand-in ``torch`` and ``cellpose`` packages in the venv's OWN
    site-packages, which come first -- so the worker imports the stand-ins,
    starts in a fraction of a second, and needs no network.
    """
    root = tmp_path_factory.mktemp("fake-backends")
    env = root / "cellpose3"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip",
                    "--system-site-packages", str(env)], check=True,
                   timeout=300, capture_output=True)
    python = SB._env_python(str(env))
    purelib = Path(subprocess.run(
        [python, "-I", "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True, timeout=120
    ).stdout.strip())
    (purelib / "torch").mkdir()
    (purelib / "torch" / "__init__.py").write_text(_STUB_TORCH)
    (purelib / "cellpose").mkdir()
    (purelib / "cellpose" / "__init__.py").write_text("")
    (purelib / "cellpose" / "models.py").write_text(_STUB_MODELS)
    SB._write_marker(str(env), {"backend": "cellpose3"})
    return root


@pytest.fixture
def faked(fake_env, monkeypatch):
    monkeypatch.setenv(SB._ROOT_ENV, str(fake_env))
    yield fake_env
    _SHUTDOWN_WORKERS()


def _field():
    image = np.zeros((12, 12), np.float32)
    image[3:7, 4:9] = 1.0
    return image


def test_a_faked_environment_segments_through_the_real_worker(faked, capsys):
    model = SB._load_backend("cellpose3", model_name="cpsam",
                             object_type="nucleus", device="cpu")
    assert isinstance(model, SB._RemoteBackend)
    assert model.model == "nuclei", "cpsam is not a Cellpose 3 model"
    assert "in its own environment" in capsys.readouterr().out
    masks, flows, styles = model.eval(
        [_field(), _field()[..., None]], channel_axis=-1, normalize=False,
        diameter=np.float64(0.0), flow_threshold=np.float32(0.4),
        cellprob_threshold=0, min_size=3, resample=True, batch_size=2,
        progress=True)
    assert styles is None and len(masks) == 2
    assert masks[0].dtype == np.uint16 and int((masks[0] > 0).sum()) == 20
    assert flows[0][1].shape == (2, 12, 12) and flows[0][3] is None
    worker = SB._WORKERS["cellpose3"]
    assert worker.hello["backend"] == "cellpose3" and worker.alive
    assert not worker.busy
    single, _flows, _ = model.eval(_field())
    assert len(single) == 1, "a bare 2-D image is a batch of one"
    assert SB._worker_for("cellpose3", model.env) is worker, "one worker"


def test_a_library_that_prints_cannot_corrupt_a_reply(faked):
    image = _field()
    image[0, 0] = 6.0
    masks, _flows, _ = SB._RemoteBackend("cellpose3", model="cyto3").eval(
        [image])
    assert masks[0].shape == (12, 12)


def test_the_backends_own_error_reaches_spacr_verbatim(faked):
    image = _field()
    image[0, 0] = 9.0
    with pytest.raises(SB._BackendError) as exc:
        SB._RemoteBackend("cellpose3", model="cyto2").eval([image])
    assert str(exc.value) == (
        "Cellpose 3 raised RuntimeError: the stand-in refused this image")
    assert exc.value.remote_type == "RuntimeError"
    assert "Traceback" in exc.value.remote_traceback
    assert SB._WORKERS["cellpose3"].alive, "an error is not a crash"


def test_a_worker_that_dies_says_so_with_its_last_words_and_restarts(faked):
    image = _field()
    image[0, 0] = 7.0
    model = SB._RemoteBackend("cellpose3", model="cyto")
    with pytest.raises(SB._BackendError) as exc:
        model.eval([image])
    message = str(exc.value)
    assert "The Cellpose 3 backend stopped (exit code 3)" in message
    assert "the stand-in is going down" in message
    dead = SB._WORKERS["cellpose3"]
    assert not dead.alive
    masks, _flows, _ = model.eval([_field()])
    assert SB._WORKERS["cellpose3"] is not dead and masks[0].max() > 0


def test_a_cancelled_request_stops_its_worker(faked):
    image = _field()
    image[0, 0] = 8.0
    model = SB._RemoteBackend("cellpose3", model="cyto3")
    started = time.monotonic()
    asked = []
    with pytest.raises(SB._BackendCancelled):
        model.eval([image], should_cancel=lambda: asked.append(1) or
                   len(asked) > 2)
    assert time.monotonic() - started < 30
    assert not SB._WORKERS["cellpose3"].alive


def test_a_worker_that_cannot_start_is_stopped(faked, monkeypatch):
    real = SB._WorkerProcess.request

    def _refuse(self, op, **kw):
        raise SB._BackendError("hello refused")

    monkeypatch.setattr(SB._WorkerProcess, "request", _refuse)
    killed = []
    monkeypatch.setattr(SB._WorkerProcess, "kill",
                        lambda self: killed.append(self) or SB._kill_tree(
                            self._proc))
    with pytest.raises(SB._BackendError, match="hello refused"):
        SB._WorkerProcess("cellpose3", str(faked / "cellpose3"))
    assert killed
    monkeypatch.setattr(SB._WorkerProcess, "request", real)


def test_the_client_refuses_an_environment_that_is_not_there(tmp_path,
                                                             monkeypatch):
    monkeypatch.setenv(SB._ROOT_ENV, str(tmp_path / "empty"))
    with pytest.raises(ImportError, match="Model Zoo"):
        SB._RemoteBackend("cellpose3")
    with pytest.raises(ImportError) as exc:
        SB._load_backend("cellpose3")
    assert "segmentation_backend='cellpose'" in str(exc.value)


class _Worker:
    """A stand-in worker for the client's bookkeeping."""

    def __init__(self, env="E", alive=True, busy=False, used=0.0,
                 reply=None):
        self.env, self.alive, self.busy = env, alive, busy
        self.last_used = used
        self.closed = 0
        self.reply = reply or {}
        self.asked = []

    def close(self):
        self.closed += 1

    def request(self, op, **payload):
        self.asked.append((op, payload))
        return self.reply


def test_the_pool_restarts_a_dead_or_moved_worker_and_reaps_idle_ones(
        monkeypatch):
    monkeypatch.setattr(SB, "_WORKERS", {})
    monkeypatch.setattr(SB, "_REAPER", [])
    started = []
    monkeypatch.setattr(SB, "_reap_forever", lambda: started.append(1))
    first = SB._worker_for("cellpose3", "E", factory=lambda n, e: _Worker(e))
    assert SB._worker_for("cellpose3", "E", factory=None) is first
    first.alive = False
    second = SB._worker_for("cellpose3", "E", factory=lambda n, e: _Worker(e))
    assert second is not first and first.closed == 1
    third = SB._worker_for("cellpose3", "F", factory=lambda n, e: _Worker(e))
    assert third.env == "F" and second.closed == 1
    time.sleep(0.05)
    assert len(SB._REAPER) == 1 and started == [1]

    SB._WORKERS.update(busy=_Worker(busy=True), fresh=_Worker(used=100.0),
                       idle=_Worker(used=0.0), dead=_Worker(alive=False,
                                                            used=100.0))
    SB._reap_idle(now=100.0 + 1, idle=50.0)
    assert set(SB._WORKERS) == {"busy", "fresh"}
    assert third.closed == 1, "the idle cellpose3 worker was reaped"
    SB._shutdown_workers("fresh")
    assert set(SB._WORKERS) == {"busy"}
    SB._shutdown_workers()
    assert SB._WORKERS == {}


def test_the_reaper_reaps_on_its_own_schedule(monkeypatch):
    reaped = []
    monkeypatch.setattr(SB, "_reap_idle", lambda: reaped.append(1))
    SB._reap_forever(interval=0.0, rounds=2)
    assert reaped == [1, 1]


def test_a_reply_missing_masks_is_an_error(faked, monkeypatch):
    worker = _Worker(reply={"outputs": []})
    model = SB._RemoteBackend("cellpose3",
                              worker_for=lambda name, env: worker)
    with pytest.raises(SB._BackendError, match="returned 0 masks for 1"):
        model.eval([_field()])
    op, payload = worker.asked[0]
    assert op == "segment" and payload["device"] == "auto"
    assert payload["params"] == {"channel_axis": -1, "normalize": True,
                                 "cellprob_threshold": 0.0}


def test_the_device_follows_spacr_device(faked, monkeypatch):
    monkeypatch.setenv(SB._DEVICE_ENV, "cuda:1")
    assert SB._RemoteBackend("cellpose3").device == "cuda:1"


def test_a_reply_for_another_request_is_skipped_and_nonsense_is_an_error():
    class _Pipe:
        def __init__(self):
            self.written = []

        def write(self, text):
            self.written.append(text)

        def flush(self):
            pass

    worker = SB._WorkerProcess.__new__(SB._WorkerProcess)
    worker.label, worker._next_id = "Cellpose 3", 0
    worker._lock = threading.Lock()
    worker._stderr = []
    worker._proc = types.SimpleNamespace(stdin=_Pipe())
    import queue

    worker._replies = queue.Queue()
    worker._replies.put(json.dumps({"id": 99, "ok": True}))
    worker._replies.put(json.dumps({"id": 1, "ok": True,
                                    "protocol": SB._PROTOCOL, "x": 1}))
    assert worker.request("hello")["x"] == 1
    worker._replies.put("garbage")
    with pytest.raises(SB._BackendError, match="not a reply: garbage"):
        worker.request("hello")
    worker._replies.put(json.dumps({"id": 3, "ok": True, "protocol": 2}))
    with pytest.raises(SB._BackendError, match="speaks protocol 2"):
        worker.request("hello")
    worker._replies.put(json.dumps({"id": 4, "ok": False,
                                    "protocol": SB._PROTOCOL}))
    with pytest.raises(SB._BackendError, match="raised an error: $"):
        worker.request("hello")


def test_a_worker_whose_pipe_broke_reports_it_stopped():
    class _Broken:
        def write(self, text):
            raise BrokenPipeError()

        def close(self):
            raise OSError("closed")

    class _Proc:
        stdin = _Broken()

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

        def poll(self):
            return 0

    worker = SB._WorkerProcess.__new__(SB._WorkerProcess)
    worker.label, worker._next_id = "Cellpose 3", 0
    worker._lock = threading.Lock()
    worker._stderr = []
    worker._proc = _Proc()
    with pytest.raises(SB._BackendError, match="exit code None"):
        worker.request("hello")
    worker.close()


def test_closing_a_worker_that_will_not_stop_kills_it():
    class _Stdin:
        def write(self, text):
            raise OSError("gone")

        def flush(self):
            pass

        def close(self):
            pass

    killed = []

    class _Proc:
        stdin = _Stdin()

        def poll(self):
            return None

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("x", timeout)

    worker = SB._WorkerProcess.__new__(SB._WorkerProcess)
    worker._proc = _Proc()
    worker.kill = lambda: killed.append(1)
    worker.close(timeout=0.01)
    assert killed == [1]


# ===========================================================================
# 8. Small pieces
# ===========================================================================

@pytest.mark.parametrize("name, obj, expected", [
    ("cyto2", "cell", "cyto2"), ("nuclei", "cell", "nuclei"),
    ("cpsam", "cell", "cyto3"), (None, "nucleus", "nuclei"),
    ("", "pathogen", "cyto3")])
def test_an_objects_model_setting_picks_its_cellpose3_model(name, obj,
                                                            expected):
    assert SB._cellpose3_model(name, obj) == expected


def test_a_checkpoint_path_is_used_and_a_missing_one_refused(tmp_path):
    checkpoint = tmp_path / "bioimage.pth"
    checkpoint.write_bytes(b"x")
    assert SB._cellpose3_model(str(checkpoint)) == str(checkpoint)
    for missing in (str(tmp_path / "gone.pth"), "models/mine", "mine.pth"):
        with pytest.raises(FileNotFoundError, match="the file is not there"):
            SB._cellpose3_model(missing)


def test_request_parameters_become_plain_json():
    assert SB._plain(np.float32(0.5)) == 0.5
    assert SB._plain(np.int64(3)) == 3 and SB._plain(True) is True
    assert SB._plain(None) is None and SB._plain("x") == "x"
    assert SB._plain(2) == 2

    class _Number:
        def __float__(self):
            return 1.5

    assert SB._plain(_Number()) == 1.5


def test_labels_that_fill_the_field_are_numbered_from_one():
    labels = np.array([[4, 4], [9, 9]])
    np.testing.assert_array_equal(SB._as_label_image(labels),
                                  [[1, 1], [2, 2]])


def test_the_backend_error_keeps_the_workers_type_and_traceback():
    error = SB._BackendError("m", "ValueError", "tb")
    assert (str(error), error.remote_type, error.remote_traceback) == (
        "m", "ValueError", "tb")


def test_the_worker_source_is_this_file(monkeypatch):
    assert SB._worker_path() == os.path.abspath(SB.__file__)
    monkeypatch.setattr(SB, "__file__", "/frozen/archive/module.pyc")
    assert SB._worker_path() is None


def test_nearest_existing_climbs_to_a_folder_that_is_there(tmp_path,
                                                         monkeypatch):
    assert SB._nearest_existing(str(tmp_path / "a" / "b")) == str(tmp_path)
    monkeypatch.setattr(SB.os.path, "exists", lambda path: False)
    assert SB._nearest_existing("/no/drive") == "/"


def test_under_cellpose3_an_objects_model_name_is_read_as_it_was_written(
        capsys):
    """``cyto2`` means Cellpose 3's cyto2 under the Cellpose 3 backend. Under
    spaCR's Cellpose it is a retired spelling of cpsam, mapped with a notice;
    under Cellpose 3 that notice would be false, so it is not given."""
    from spacr.settings import (
        _get_object_settings,
        set_default_settings_preprocess_generate_masks,
    )

    settings = set_default_settings_preprocess_generate_masks(
        {"src": "/nowhere", "segmentation_backend": "Cellpose3",
         "cell_model_name": "cyto2", "nucleus_model_name": None})
    assert _get_object_settings("cell", settings)["model_name"] == "cyto2"
    assert _get_object_settings("nucleus", settings)["model_name"] is None
    assert "predates Cellpose-SAM" not in capsys.readouterr().out
    settings["segmentation_backend"] = "cellpose"
    assert _get_object_settings("cell", settings)["model_name"] == "cpsam"
