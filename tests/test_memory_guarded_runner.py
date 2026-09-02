"""The local long-run wrapper yields before the desktop loses its RAM."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "run_memory_guarded.py"
SPEC = importlib.util.spec_from_file_location("run_memory_guarded", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def test_the_default_stops_at_115_gib():
    assert guard.DEFAULT_LIMIT_GIB == 115.0
    assert guard.MEMORY_ABORT_EXIT == 137


def test_a_busy_machine_refuses_to_start_a_child(monkeypatch, capsys):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("the child started after the RAM limit")

    monkeypatch.setattr(guard.subprocess, "Popen", forbidden)
    result = guard.run_guarded(
        [sys.executable, "-c", "raise SystemExit(0)"],
        memory_reader=lambda: (125.0, 115.0),
    )
    assert result == 137
    assert "refusing to start" in capsys.readouterr().err


def test_a_running_child_is_terminated_when_usage_crosses_the_limit(
    monkeypatch, capsys,
):
    readings = iter(((125.0, 10.0), (125.0, 116.0)))
    started = []
    real_popen = subprocess.Popen

    def recording(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started.append(process)
        return process

    monkeypatch.setattr(guard.subprocess, "Popen", recording)
    result = guard.run_guarded(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        poll_seconds=0.01,
        grace_seconds=0.2,
        memory_reader=lambda: next(readings),
    )
    assert result == 137
    assert len(started) == 1
    assert started[0].poll() is not None
    assert "stopping pid" in capsys.readouterr().err


def test_a_safe_child_returns_its_own_exit_code():
    result = guard.run_guarded(
        [sys.executable, "-c", "raise SystemExit(7)"],
        poll_seconds=0.01,
        memory_reader=lambda: (125.0, 10.0),
    )
    assert result == 7
