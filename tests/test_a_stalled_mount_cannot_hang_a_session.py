"""The NAS guard returns on a deadline, whatever the command is doing.

A session had to be restarted by hand on 2026-09-20 because it was waiting
on `/nas_mnt`, which is NFSv3 mounted `hard`. These tests hold the one
property that failure needs: the guard comes back, and it comes back with
the truth about what happened.

Nothing here touches a network mount. The whole point is that the guard is
testable without one.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "tools" / "nas_guard.sh"


def _guard(*args, timeout=60):
    return subprocess.run([str(GUARD), *args], capture_output=True,
                          text=True, timeout=timeout)


def test_the_guard_is_there_and_can_be_run():
    assert GUARD.is_file()
    assert GUARD.stat().st_mode & 0o111, "the guard must be executable"


def test_a_command_that_would_hang_gives_the_session_back():
    """The property the restarted session needed and did not have."""
    started = time.monotonic()
    done = _guard("run", "2", "sleep", "300")
    elapsed = time.monotonic() - started
    assert done.returncode == 124
    assert elapsed < 30, f"the guard took {elapsed:.1f}s to give up on a 2s deadline"
    assert "gave up after 2s" in done.stderr


def test_giving_up_says_the_command_was_abandoned_rather_than_killed():
    """Because SIGKILL does not reach a process in an uninterruptible wait,
    and a message that claimed otherwise would send the next reader looking
    for a process that is never going to die."""
    done = _guard("run", "1", "sleep", "300")
    assert "abandoned, not killed" in done.stderr


def test_a_command_that_finishes_keeps_its_output_and_its_exit_code():
    done = _guard("run", "30", "bash", "-c", "echo out; echo err >&2; exit 7")
    assert done.returncode == 7
    assert "out" in done.stdout
    assert "err" in done.stderr


def test_a_path_that_answers_passes_and_one_that_is_not_there_does_not(tmp_path):
    assert _guard("check", str(tmp_path), "10").returncode == 0
    assert _guard("check", str(tmp_path / "nowhere"), "10").returncode == 3


def test_the_deadline_is_not_enforced_with_timeout(tmp_path):
    """`timeout` waits for the child it signalled, and a hard NFS wait
    ignores the signal, so the guard would hang exactly where the session
    hung. If someone simplifies this script into a `timeout` call, this
    test is the note explaining why they must not."""
    source = GUARD.read_text()
    body = "\n".join(line for line in source.splitlines()
                     if not line.lstrip().startswith("#"))
    assert "timeout " not in body


def test_it_never_waits_on_the_process_that_touches_the_mount():
    source = GUARD.read_text()
    assert "setsid" in source, "the probe has to run in a session of its own"
    commands = [line.strip() for line in source.splitlines()
                if not line.lstrip().startswith("#")]
    assert not [line for line in commands
                if line == "wait" or line.startswith("wait ")]


def test_listing_network_mounts_does_not_ask_any_server_anything():
    """`paths` has to work when the server is gone, so it may only read what
    the kernel already knows."""
    done = _guard("paths", timeout=30)
    assert done.returncode in (0, 1)
    source = GUARD.read_text()
    assert "findmnt" in source
