"""What a sweep does when the machine underneath it does not cooperate.

Every safeguard in :mod:`spacr.parameter_sweep` reads something the host may
refuse to answer -- free memory, the CPU affinity mask, a systemd user scope,
``/proc/meminfo``, the child's result file. The rule throughout is that a
refusal degrades the safeguard, never the sweep: a machine that cannot be
measured runs fewer workers, a trial whose record cannot be written is still
a trial, and a child killed halfway through its JSON is a "killed" row rather
than a traceback. These drive each of those refusals.
"""
from __future__ import annotations

import builtins
import os
import subprocess
import sys
import types

import pytest

import spacr.parameter_sweep as ps


# ---------------------------------------------------------------------------
# how many workers, on a machine that will not say how much memory it has
# ---------------------------------------------------------------------------

def test_a_budget_that_cannot_read_free_memory_falls_back_to_two_workers(
        monkeypatch):
    """Unmeasurable memory is a reason to be timid, not a reason to stop."""
    monkeypatch.setitem(sys.modules, "psutil", None)

    budget = ps._recommended_worker_budget(requested=8)

    assert budget["available"] is None
    assert 1 <= budget["workers"] <= 2
    assert budget["requested"] == 8
    assert budget["per_trial"] == ps.ASSUMED_TRIAL_GIB


def test_the_reason_says_the_memory_could_not_be_measured(monkeypatch):
    """A reduced worker count the user cannot explain looks like a bug."""
    def refuse():
        raise PermissionError("/proc is hidden in this container")

    monkeypatch.setattr("psutil.virtual_memory", refuse)

    workers, reason = ps.recommended_workers(requested=8)

    assert workers <= 2
    assert "memory could not be measured" in reason
    assert f"{workers} workers" in reason
    assert "GiB free" not in reason


def test_a_platform_without_sched_getaffinity_counts_cores_with_cpu_count(
        monkeypatch):
    """macOS and Windows have no affinity mask; they still get a core count."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 3)
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=400 * 1024 ** 3))

    budget = ps._recommended_worker_budget(requested=8)

    # 200 GiB of budget at 6 GiB a trial affords 33 workers, and 8 were
    # asked for: the answer can only be 1 if the core count came from
    # cpu_count() - 2.
    assert budget["workers"] == 1
    assert budget["available"] == pytest.approx(400.0)


def test_memory_is_never_reported_low_when_it_cannot_be_read(monkeypatch):
    """A check that cannot see the machine must not block every trial."""
    def refuse():
        raise RuntimeError("psutil could not open /proc/meminfo")

    monkeypatch.setattr("psutil.virtual_memory", refuse)

    # A floor no machine could clear: a working check would say True.
    assert ps.memory_is_low(floor_gib=1e9) is False


# ---------------------------------------------------------------------------
# pinning threads, with pieces of the numerical stack missing
# ---------------------------------------------------------------------------

def _sentinel_thread_env(monkeypatch):
    """Record the thread variables so pytest restores them afterwards."""
    for name in ps._THREAD_VARS:
        monkeypatch.setenv(name, "unset-by-test")


def test_pinning_threads_survives_a_missing_threadpoolctl(monkeypatch):
    """Without threadpoolctl the environment pin is all there is; keep it."""
    _sentinel_thread_env(monkeypatch)
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)
    monkeypatch.setattr(ps, "_THREAD_LIMITS", "untouched")

    torch_calls = []
    fake_torch = types.ModuleType("torch")
    fake_torch.set_num_threads = torch_calls.append
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ps._pin_threads(3)

    assert [os.environ[name] for name in ps._THREAD_VARS] == ["3"] * 5
    assert ps._THREAD_LIMITS == "untouched", \
        "no live pool was resized, so nothing should be held open"
    assert torch_calls == [3], "torch is pinned even when threadpoolctl is not"


def test_pinning_threads_survives_a_torch_that_refuses_to_set_threads(
        monkeypatch):
    """torch is optional here, and a build that objects is not fatal."""
    _sentinel_thread_env(monkeypatch)

    limiter = object()
    fake_ctl = types.ModuleType("threadpoolctl")
    fake_ctl.threadpool_limits = lambda limits: limiter
    monkeypatch.setitem(sys.modules, "threadpoolctl", fake_ctl)
    monkeypatch.setattr(ps, "_THREAD_LIMITS", None)

    fake_torch = types.ModuleType("torch")

    def refuse(count):
        raise RuntimeError("this build has no threading support")

    fake_torch.set_num_threads = refuse
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    ps._pin_threads(2)

    assert [os.environ[name] for name in ps._THREAD_VARS] == ["2"] * 5
    assert ps._THREAD_LIMITS is limiter, \
        "the live BLAS pool is still resized and held open"


# ---------------------------------------------------------------------------
# be_polite, one refusal at a time
# ---------------------------------------------------------------------------

def _redirect_oom_file(monkeypatch, replacement):
    """Send the worker's oom_score_adj write to `replacement` instead."""
    real_open = builtins.open
    target = f"/proc/{os.getpid()}/oom_score_adj"

    def routed(path, *args, **kwargs):
        if path == target:
            return real_open(replacement, *args, **kwargs)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", routed)


def test_be_polite_carries_on_when_the_kernel_refuses_a_nice_change(
        monkeypatch, tmp_path):
    """A refused renice must not cost the worker its other concessions."""
    def refuse(increment):
        raise OSError("nice is not permitted here")

    monkeypatch.setattr(os, "nice", refuse, raising=False)
    oom_file = tmp_path / "oom_score_adj"
    _redirect_oom_file(monkeypatch, str(oom_file))
    commands = []
    monkeypatch.setattr(subprocess, "run",
                        lambda command, **kwargs: commands.append(command))

    ps.be_polite()

    assert oom_file.read_text(encoding="utf-8") == "800"
    assert commands and commands[0][:3] == ["ionice", "-c", "3"]


def test_be_polite_carries_on_when_the_oom_score_file_cannot_be_written(
        monkeypatch):
    """Containers and hardened kernels refuse the write; the sweep still runs."""
    nice_calls = []
    monkeypatch.setattr(os, "nice", lambda increment: nice_calls.append(
        increment) or increment, raising=False)

    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if isinstance(path, str) and path.endswith("oom_score_adj"):
            raise PermissionError("read-only /proc")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", deny)
    commands = []
    monkeypatch.setattr(subprocess, "run",
                        lambda command, **kwargs: commands.append(command))

    ps.be_polite()

    assert nice_calls == [19]
    assert commands and commands[0][-1] == str(os.getpid()), \
        "the I/O priority is still lowered after the refused write"


def test_be_polite_carries_on_when_ionice_is_missing(monkeypatch, tmp_path):
    """ionice is a Linux utility a minimal image may simply not ship."""
    nice_calls = []
    monkeypatch.setattr(os, "nice", lambda increment: nice_calls.append(
        increment) or increment, raising=False)
    oom_file = tmp_path / "oom_score_adj"
    _redirect_oom_file(monkeypatch, str(oom_file))

    def missing(command, **kwargs):
        raise FileNotFoundError("ionice")

    monkeypatch.setattr(subprocess, "run", missing)

    assert ps.be_polite() is None
    assert nice_calls == [19]
    assert oom_file.read_text(encoding="utf-8") == "800"


# ---------------------------------------------------------------------------
# containment, and the machine that cannot answer
# ---------------------------------------------------------------------------

def test_containment_is_unavailable_when_the_probe_scope_cannot_start(
        monkeypatch):
    """systemd-run on PATH proves nothing without a live user manager."""
    import shutil

    ps.containment_available.cache_clear()
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/systemd-run")

    def hang(command, **kwargs):
        raise subprocess.TimeoutExpired(cmd=command, timeout=10)

    monkeypatch.setattr(subprocess, "run", hang)
    try:
        assert ps.containment_available() is False
        # And the note the user reads follows the probe, not the PATH lookup.
        assert "Kernel containment is unavailable" in ps.containment_note()
    finally:
        ps.containment_available.cache_clear()


def test_free_memory_is_infinite_where_meminfo_cannot_be_read(monkeypatch):
    """Infinity, not zero: a blind safety check must not refuse every run."""
    real_open = builtins.open

    def deny(path, *args, **kwargs):
        if path == "/proc/meminfo":
            raise PermissionError("no /proc on this host")
        return real_open(path, *args, **kwargs)

    def blind():
        raise RuntimeError("psutil cannot read this host either")

    monkeypatch.setattr(builtins, "open", deny)
    monkeypatch.setattr("psutil.virtual_memory", blind)

    free = ps.free_memory_gb()

    assert free == float("inf")
    assert free > ps.FREE_MEMORY_FLOOR_GB, "an unreadable host still schedules"


@pytest.mark.skipif(os.name != "nt", reason="Windows has no /proc or nice")
def test_windows_reads_free_memory_without_proc(monkeypatch):
    """No /proc on Windows must not switch the memory floor off."""
    monkeypatch.setattr(
        "psutil.virtual_memory",
        lambda: types.SimpleNamespace(available=3 * 10 ** 9))
    assert ps.free_memory_gb() == pytest.approx(3.0)


@pytest.mark.skipif(os.name != "nt", reason="Windows has no /proc or nice")
def test_windows_trial_starts_python_itself_at_idle_priority(
        monkeypatch, tmp_path):
    """There is no `nice` executable on Windows to start the trial through."""
    monkeypatch.setattr(ps, "containment_available", lambda: False)
    seen = {}

    def record(command, **kwargs):
        seen.update(command=command, **kwargs)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", record)
    ps.run_trial_contained({"src": str(tmp_path)}, trial_id=1, timeout=5)

    assert seen["command"][0] == sys.executable
    assert "nice" not in seen["command"]
    assert seen["creationflags"] == subprocess.IDLE_PRIORITY_CLASS


# ---------------------------------------------------------------------------
# a child that died mid-sentence
# ---------------------------------------------------------------------------

def test_a_truncated_result_file_is_a_killed_row_not_a_crash(tmp_path,
                                                             monkeypatch):
    """The kill can land between the open and the last brace."""
    monkeypatch.setattr(ps, "containment_available", lambda: False)

    def killed_mid_write(command, **kwargs):
        with open(command[-1], "w") as handle:
            handle.write('{"trial_id": 4, "status": "ok", "n_hi')
        return subprocess.CompletedProcess(command, 137, "", "Killed\n")

    monkeypatch.setattr(subprocess, "run", killed_mid_write)

    row = ps.run_trial_contained({"src": str(tmp_path)}, trial_id=4,
                                 memory_max="8G")

    assert row["status"] == "killed"
    assert row["trial_id"] == 4
    assert row["error_type"] == "MemoryMax"
    assert "MemoryMax=8G" in row["error"]
    assert "Killed" in row["error"]


# ---------------------------------------------------------------------------
# a trial whose paperwork cannot be filed
# ---------------------------------------------------------------------------

def test_a_trial_is_built_even_when_its_settings_cannot_be_saved(tmp_path,
                                                                monkeypatch):
    """The saved copy is a convenience; losing it must not lose the trial."""
    from spacr import utils

    def refuse(settings, **kwargs):
        raise OSError("the settings folder is read-only")

    monkeypatch.setattr(utils, "save_settings", refuse)

    settings, folder = ps._trial_settings(
        {"score_data": ["a.csv"], "regression_qc": True},
        {"trial_id": 12, "regression_type": "ols"},
        str(tmp_path))

    assert folder == os.path.join(str(tmp_path), "trial_0012")
    assert os.path.isdir(folder)
    assert settings["src"] == folder
    assert settings["regression_type"] == "ols"
    assert settings["regression_qc"] is False


@pytest.mark.parametrize("permission_denied", [False, True])
def test_windows_priority_and_memory_fallback_are_scoped_to_worker(monkeypatch, permission_denied):
    import psutil

    class WindowsOS:
        name = "nt"
        def __getattr__(self, name):
            return getattr(os, name)

    native_os_name = os.name
    calls = []
    def priority(value):
        calls.append(value)
        if permission_denied:
            raise PermissionError("priority adjustment denied")
    monkeypatch.setattr(ps, "os", WindowsOS())
    monkeypatch.setattr(psutil, "IDLE_PRIORITY_CLASS", 64, raising=False)
    monkeypatch.setattr(psutil, "Process", lambda: types.SimpleNamespace(nice=priority))
    ps._idle_priority_on_windows()
    assert calls == [64]
    original_open = builtins.open
    def inaccessible(path, *args, **kwargs):
        if path == "/proc/meminfo":
            raise OSError("no proc filesystem")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", inaccessible)
    def memory():
        if permission_denied:
            raise PermissionError("memory query denied")
        return types.SimpleNamespace(available=3500000000)
    monkeypatch.setattr(psutil, "virtual_memory", memory)
    assert ps.free_memory_gb() == (float("inf") if permission_denied else 3.5)
    assert os.name == native_os_name


def test_windows_trial_launch_preserves_child_payload_and_thread_budget(tmp_path, monkeypatch):
    import json

    class WindowsOS:
        name = "nt"
        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(ps, "os", WindowsOS())
    monkeypatch.setattr(ps, "containment_available", lambda: False)
    monkeypatch.setattr(subprocess, "IDLE_PRIORITY_CLASS", 64, raising=False)
    captured = []
    def run(command, **kwargs):
        captured.append((command, kwargs))
        payload = json.loads(Path(command[-2]).read_text())
        assert payload == {"settings": {"src": str(tmp_path)}, "trial_id": 9, "controls": {"group": "control"}}
        Path(command[-1]).write_text(json.dumps({"trial_id": 9, "status": "ok", "n_hi": 3}))
        return subprocess.CompletedProcess(command, 0, "", "")
    from pathlib import Path
    monkeypatch.setattr(subprocess, "run", run)
    result = ps.run_trial_contained({"src": str(tmp_path)}, trial_id=9, controls={"group": "control"}, timeout=7)
    assert result["trial_id"] == 9 and result["status"] == "ok" and result["n_hi"] == 3
    command, options = captured[0]
    assert command[:3] == [sys.executable, "-m", "spacr.sweep_child"]
    assert "nice" not in command and options["creationflags"] == 64
    assert options["timeout"] == 7
    assert all(options["env"][key] == "1" for key in ps._THREAD_VARS)
