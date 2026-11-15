"""Shutting down AI provider subprocesses: who is signalled, killed, reaped and forgotten.

``spacr.qt.ai.providers`` reads each vendor CLI on a worker thread that blocks
on the child's stdout, so :func:`terminate_all_streams` is what keeps a stuck
child from outliving the window. Each case scripts the children's teardown
and asserts the exact poll/terminate/wait/kill sequence, the count returned
and what is left in the registry. ``subprocess.Popen`` is the boundary;
nothing spawns a real CLI.
"""
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import providers  # noqa: E402

pytestmark = pytest.mark.qt


class _FakeStdout:
    """A finite line source that records its close."""

    def __init__(self, lines):
        self._lines = iter(list(lines))
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._lines)

    def close(self):
        self.closed = True


class _FakeChild:
    """Stand-in for ``subprocess.Popen`` whose teardown is scripted."""

    def __init__(self, *, lines=(), wait_raises=0, terminate_raises=None,
                 on_terminate=None, running=True):
        self.stdout = _FakeStdout(lines)
        self.stdin = None
        self.calls = []
        self.wait_timeouts = []
        self._wait_raises = wait_raises
        self._terminate_raises = terminate_raises
        self._on_terminate = on_terminate
        self.running = running

    def poll(self):
        self.calls.append("poll")
        return None if self.running else 0

    def terminate(self):
        self.calls.append("terminate")
        if self._terminate_raises is not None:
            raise self._terminate_raises
        if self._on_terminate is not None:
            self._on_terminate(self)
        self.running = False

    def wait(self, timeout=None):
        self.calls.append("wait")
        self.wait_timeouts.append(timeout)
        if self._wait_raises > 0:
            self._wait_raises -= 1
            raise subprocess.TimeoutExpired(cmd="fake-cli", timeout=timeout or 0)
        self.running = False
        return 0

    def kill(self):
        self.calls.append("kill")
        self.running = False


SIGNALLED = ["poll", "terminate", "wait"]


def test_terminate_all_streams_ends_reaps_and_forgets_each_child(
        subtests, monkeypatch):
    """Count = children signalled; kill only the stubborn; keep only the unreachable live one."""
    with subtests.test(msg="a stubborn child is killed, a polite one is not"):
        stubborn, polite = _FakeChild(wait_raises=1), _FakeChild()
        registry = [stubborn, polite]
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 2
        assert stubborn.calls == ["poll", "terminate", "wait", "kill", "wait"]
        assert stubborn.wait_timeouts == [1, 1]
        assert polite.calls == SIGNALLED
        assert polite.wait_timeouts == [1]
        assert (stubborn.running, polite.running) == (False, False)
        assert (stubborn._spacr_stopped_it, polite._spacr_stopped_it) == (
            True, True)
        assert registry == []

    with subtests.test(msg="a live child that cannot be signalled stays for retry"):
        unreachable = _FakeChild(terminate_raises=OSError("no such process"))
        healthy = _FakeChild()
        registry = [unreachable, healthy]
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 1
        assert unreachable.calls == ["poll", "terminate", "poll"]
        assert healthy.calls == SIGNALLED
        assert healthy.running is False
        assert registry == [unreachable]
        unreachable._terminate_raises = None
        assert providers.terminate_all_streams() == 1
        assert unreachable.calls[3:] == SIGNALLED
        assert unreachable.running is False
        assert registry == []

    with subtests.test(msg="a child that dies while refusing the signal is dropped"):
        def _dies(child):
            child.running = False
            raise OSError("no such process")

        gone = _FakeChild()
        gone.terminate = lambda: (gone.calls.append("terminate"), _dies(gone))
        registry = [gone]
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 0
        assert gone.calls == ["poll", "terminate", "poll"]
        assert registry == []

    with subtests.test(msg="an entry the reader already removed is not removed twice"):
        def _reader_wins_the_race(child):
            registry.remove(child)

        racer = _FakeChild(on_terminate=_reader_wins_the_race)
        behind_it = _FakeChild()
        registry = [racer, behind_it]
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 2
        assert racer.calls == SIGNALLED
        assert behind_it.calls == SIGNALLED
        assert registry == []

    with subtests.test(msg="a dead entry is dropped without being signalled"):
        corpse, still_running = _FakeChild(running=False), _FakeChild()
        registry = [corpse, still_running]
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 1
        assert corpse.calls == ["poll"]
        assert not hasattr(corpse, "_spacr_stopped_it")
        assert still_running.calls == SIGNALLED
        assert registry == []

    with subtests.test(msg="an empty registry ends nothing"):
        registry = []
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        assert providers.terminate_all_streams() == 0
        assert registry == []

    with subtests.test(msg="a shutdown mid-stream leaves the reader nothing to remove"):
        registry = []
        monkeypatch.setattr(providers, "_LIVE_STREAMS", registry)
        child = _FakeChild(lines=["first chunk\n", "second chunk\n"])
        seen = {}

        def _popen(argv, **kwargs):
            seen["argv"] = list(argv)
            return child

        monkeypatch.setattr(providers.subprocess, "Popen", _popen)
        provider = providers.ClaudeCliProvider()
        stream = providers._stream_process(["claude", "-p", "hi"],
                                           provider=provider)
        assert next(stream) == "first chunk\n"
        assert seen["argv"] == ["claude", "-p", "hi"]
        assert registry == [child]
        assert provider._current_proc is child
        assert providers.terminate_all_streams() == 1
        assert registry == []
        assert list(stream) == ["second chunk\n"]
        assert child.stdout.closed is True
        assert provider._current_proc is None
        assert registry == []
