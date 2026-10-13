"""The memory watchdog: what it picks, and what it refuses to touch.

Asked for on 2026-09-20 after a python holding 113 GiB was OOM-killed and
VS Code went down with it -- the other session's process, confirmed by the
maintainer.

The whole point of the selection rules is that they are the difference
between saving a session and ending one, so they are tested on made-up
process lists rather than on whatever happens to be running.
"""

from __future__ import annotations

import importlib.util
import os
import signal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "memory_watchdog", ROOT / "tools" / "memory_watchdog.py")
watchdog = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(watchdog)

P = watchdog.Process


def test_used_memory_is_total_minus_available_not_minus_free():
    """The page cache is not a leak. Counting it as used would fire this
    on a machine that is merely busy."""
    meminfo = ("MemTotal:       131836276 kB\n"
               "MemFree:          2000000 kB\n"
               "MemAvailable:    31836276 kB\n")
    assert watchdog.memory_used_gb(meminfo) == pytest.approx(95.4, abs=0.2)


def test_a_meminfo_it_cannot_read_is_zero_not_a_crash():
    assert watchdog.memory_used_gb("nonsense\n") == 0.0


def test_the_largest_uncapped_process_is_chosen():
    processes = [P(10, "python", 40.0, False), P(11, "python", 90.0, False)]
    assert watchdog.choose(processes, 16.0, me=999).pid == 11


def test_the_editor_and_the_desktop_are_never_chosen():
    """Killing the compositor to save memory is not saving the session."""
    processes = [P(10, "code", 100.0, False),
                 P(11, "Xorg", 90.0, False),
                 P(12, "gnome-shell", 80.0, False),
                 P(13, "python", 20.0, False)]
    assert watchdog.choose(processes, 16.0, me=999).pid == 13


def test_a_process_already_under_a_cap_is_left_alone():
    """It will die in its own scope without the machine noticing, which is
    what run_capped.sh is for. Freezing it would only confuse whoever is
    waiting on it."""
    processes = [P(10, "python", 90.0, True), P(11, "python", 20.0, False)]
    assert watchdog.choose(processes, 16.0, me=999).pid == 11


def test_nothing_below_the_floor_is_touched():
    processes = [P(10, "python", 4.0, False), P(11, "python", 15.9, False)]
    assert watchdog.choose(processes, 16.0, me=999) is None


def test_it_never_chooses_itself_or_pid_one():
    processes = [P(7, "python", 90.0, False), P(1, "systemd", 80.0, False)]
    assert watchdog.choose(processes, 16.0, me=7) is None


def test_below_the_threshold_it_does_nothing_at_all(monkeypatch):
    acted = []
    monkeypatch.setattr(watchdog, "memory_used_gb", lambda *a: 40.0)
    monkeypatch.setattr(watchdog, "scan", lambda: (_ for _ in ()).throw(
        AssertionError("it scanned when it had no reason to")))
    monkeypatch.setattr(watchdog, "freeze", lambda pid: acted.append(pid))
    watchdog.watch(100.0, 112.0, 16.0, 0.0, once=True, log=lambda *_a: None)
    assert acted == []


def test_at_the_threshold_it_freezes_rather_than_kills(monkeypatch):
    """SIGSTOP stops it allocating in the time a signal takes, and leaves
    the memory there to look at. Nothing is killed at this point."""
    stopped, killed, scored = [], [], []
    monkeypatch.setattr(watchdog, "memory_used_gb", lambda *a: 101.0)
    monkeypatch.setattr(watchdog, "scan",
                        lambda: [P(42, "python", 90.0, False)])
    monkeypatch.setattr(watchdog, "freeze", lambda pid: stopped.append(pid) or True)
    monkeypatch.setattr(watchdog, "make_the_kernel_prefer",
                        lambda pid: scored.append(pid) or True)
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))
    said = []
    watchdog.watch(100.0, 112.0, 16.0, 0.0, once=True, log=said.append)
    assert stopped == [42] and scored == [42]
    assert killed == [], "nothing may be killed at the freeze threshold"
    assert "FROZEN" in said[0] and "kill -CONT 42" in said[0]


def test_the_message_says_how_to_undo_it(monkeypatch):
    """A frozen process looks like a hung one. The line has to say what
    happened and how to put it back."""
    monkeypatch.setattr(watchdog, "memory_used_gb", lambda *a: 101.0)
    monkeypatch.setattr(watchdog, "scan",
                        lambda: [P(42, "pytest", 90.0, False)])
    monkeypatch.setattr(watchdog, "freeze", lambda pid: True)
    monkeypatch.setattr(watchdog, "make_the_kernel_prefer", lambda pid: True)
    said = []
    watchdog.watch(100.0, 112.0, 16.0, 0.0, once=True, log=said.append)
    assert "Nothing was killed" in said[0]
    assert "resumes it" in said[0]


def test_nothing_to_act_on_is_said_rather_than_passed_over(monkeypatch):
    monkeypatch.setattr(watchdog, "memory_used_gb", lambda *a: 101.0)
    monkeypatch.setattr(watchdog, "scan", lambda: [P(9, "code", 99.0, False)])
    said = []
    watchdog.watch(100.0, 112.0, 16.0, 0.0, once=True, log=said.append)
    assert "nothing to do" in said[0]


def test_a_frozen_process_is_killed_only_if_memory_keeps_climbing(monkeypatch):
    """The second threshold, driven through two turns of the real loop.

    Freezing one process does not help if something else is still growing,
    and at that point the choice is between one process and the machine.
    """
    killed = []
    readings = iter([101.0, 113.0])
    monkeypatch.setattr(watchdog, "memory_used_gb", lambda *a: next(readings))
    monkeypatch.setattr(watchdog, "scan",
                        lambda: [P(42, "python", 90.0, False)])
    monkeypatch.setattr(watchdog, "freeze", lambda pid: True)
    monkeypatch.setattr(watchdog, "make_the_kernel_prefer", lambda pid: True)
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append((pid, sig)))

    turns = {"n": 0}

    def _sleep(_seconds):
        turns["n"] += 1
        if turns["n"] >= 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(watchdog.time, "sleep", _sleep)
    said = []
    with pytest.raises(KeyboardInterrupt):
        watchdog.watch(100.0, 112.0, 16.0, 0.0, once=False, log=said.append)

    assert killed == [(42, signal.SIGKILL)], (
        "the frozen process must be killed once memory passes the second "
        "threshold, and nothing else must be")
    assert any("killing frozen" in line for line in said)


def test_a_cgroup_with_no_limit_is_not_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(watchdog, "_read", lambda path: "max\n")
    assert watchdog.is_capped("0::/user.slice/thing.scope") is False


def test_a_cgroup_with_a_limit_is_capped(monkeypatch):
    monkeypatch.setattr(watchdog, "_read", lambda path: "8589934592\n")
    assert watchdog.is_capped("0::/user.slice/capped.scope") is True


def test_the_protected_list_names_the_editor_and_the_compositor():
    for name in ("code", "Xorg", "gnome-shell", "mutter-x11-fram"):
        assert name in watchdog.PROTECTED, name
