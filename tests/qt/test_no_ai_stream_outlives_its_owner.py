"""An AI provider subprocess must not outlive whoever asked for it.

The stream is read by a worker thread that BLOCKS on the child's stdout, so
it cannot be stopped by setting a flag -- only ending the child lets that
read return. When the owner goes away and nothing ends the child, the thread
stays blocked, and the process dies the moment Qt collects that thread's
wrapper: `Fatal Python error: Aborted`, which takes every remaining test in
the run rather than one.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import providers                                # noqa: E402


def _a_child_that_never_finishes():
    """A process that will sit there until something ends it."""
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)


@pytest.fixture
def registry_is_its_own(monkeypatch):
    live = []
    monkeypatch.setattr(providers, "_LIVE_STREAMS", live)
    yield live
    for proc in list(live):
        try:
            proc.kill()
        except Exception:                                        # noqa: BLE001
            pass


def test_a_live_stream_is_ended(registry_is_its_own):
    proc = _a_child_that_never_finishes()
    registry_is_its_own.append(proc)
    assert proc.poll() is None, "the child should still be running"

    assert providers.terminate_all_streams() == 1
    assert proc.wait(timeout=5) is not None
    assert registry_is_its_own == []


def test_every_live_stream_is_ended(registry_is_its_own):
    """One that refuses to die must not stop the others being reached."""
    procs = [_a_child_that_never_finishes() for _ in range(3)]
    registry_is_its_own.extend(procs)

    assert providers.terminate_all_streams() == 3
    for proc in procs:
        assert proc.wait(timeout=5) is not None
    assert registry_is_its_own == []


def test_an_already_finished_stream_is_stepped_over(registry_is_its_own):
    """A crash can skip the removal, leaving a dead entry behind."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait(timeout=10)
    registry_is_its_own.append(proc)

    assert providers.terminate_all_streams() == 0
    assert registry_is_its_own == [], "the dead entry was not cleared"


def test_it_is_safe_with_nothing_running(registry_is_its_own):
    assert providers.terminate_all_streams() == 0


def test_a_stream_registers_and_deregisters_itself(monkeypatch):
    """The registry is filled by the streaming path, not by hand."""
    live = []
    monkeypatch.setattr(providers, "_LIVE_STREAMS", live)
    seen = []

    lines = list(providers._stream_process(
        [sys.executable, "-c", "print('hello'); print('there')"]))
    seen.extend(lines)

    assert [line.strip() for line in seen] == ["hello", "there"]
    assert live == [], "a finished stream left itself in the registry"
