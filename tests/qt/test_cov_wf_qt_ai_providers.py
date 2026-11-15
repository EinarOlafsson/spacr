"""What happens to an AI provider subprocess when shutting it down goes wrong.

``spacr.qt.ai.providers`` reads each vendor CLI on a worker thread that BLOCKS
on the child's stdout. Nothing unblocks that read except the child ending, and
a thread still blocked when Qt collects its ``QThread`` wrapper aborts the
whole process -- so :func:`terminate_all_streams` is what stands between a
stuck ``claude``/``codex``/``gemini`` child and a hard crash on exit.

The happy paths are covered elsewhere (``test_ai_providers_offline.py``,
``test_no_ai_stream_outlives_its_owner.py``). What is exercised here is the
part that only shows up when a child misbehaves or when two cleanup paths
race: a child that ignores SIGTERM, a child that can no longer be signalled at
all, and the registry entry being removed twice because the reader's own
teardown and the global shutdown both reached for it. A child whose exit is
confirmed must be reaped and removed; one that is still live after signalling
failed must remain registered, because dropping its only handle guarantees a
blocked reader thread survives the next shutdown attempt.

The transport boundary is ``subprocess.Popen``; nothing here spawns a real
vendor CLI or touches the network.
"""
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import providers  # noqa: E402

# ---------------------------------------------------------------------------
# A scriptable stand-in for the child process
# ---------------------------------------------------------------------------

class _FakeStdout:
    """The child's stdout: a finite line source that records its close."""

    def __init__(self, lines):
        self._lines = list(lines)
        self._i = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._i >= len(self._lines):
            raise StopIteration
        line = self._lines[self._i]
        self._i += 1
        return line

    def close(self):
        self.closed = True


class _FakeChild:
    """Stand-in for ``subprocess.Popen`` whose teardown can be scripted.

    ``calls`` records the exact sequence of poll/terminate/wait/kill so a test
    can assert that the escalation happened in the order the shutdown code
    claims, not merely that the process ended up gone.
    """

    def __init__(self, *, lines=(), wait_raises=0, terminate_raises=None,
                 on_terminate=None):
        self.stdout = _FakeStdout(lines)
        self.stdin = None
        self.calls = []
        self.wait_timeouts = []
        self._wait_raises = wait_raises          # how many wait()s time out
        self._terminate_raises = terminate_raises
        self._on_terminate = on_terminate
        self._running = True

    # -- Popen surface used by providers ------------------------------------
    def poll(self):
        self.calls.append("poll")
        return None if self._running else 0

    def terminate(self):
        self.calls.append("terminate")
        if self._terminate_raises is not None:
            raise self._terminate_raises
        if self._on_terminate is not None:
            self._on_terminate(self)
        self._running = False

    def wait(self, timeout=None):
        self.calls.append("wait")
        self.wait_timeouts.append(timeout)
        if self._wait_raises > 0:
            self._wait_raises -= 1
            raise subprocess.TimeoutExpired(cmd="fake-cli", timeout=timeout or 0)
        self._running = False
        return 0

    def kill(self):
        self.calls.append("kill")
        self._running = False

    # -- test-side helper ---------------------------------------------------
    @property
    def running(self):
        return self._running


@pytest.fixture()
def registry(monkeypatch):
    """Give the test its own ``_LIVE_STREAMS`` so a leak cannot escape it."""
    live = []
    monkeypatch.setattr(providers, "_LIVE_STREAMS", live)
    return live


def _install_fake_popen(monkeypatch, child):
    """Route ``providers.subprocess.Popen`` to ``child``, recording argv."""
    seen = {}

    def _popen(argv, **kwargs):
        seen["argv"] = list(argv)
        seen.update(kwargs)
        return child

    monkeypatch.setattr(providers.subprocess, "Popen", _popen)
    return seen


# ---------------------------------------------------------------------------
# terminate_all_streams — the shutdown path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _stream_process — the reader side of the same race
# ---------------------------------------------------------------------------


def test_a_stream_that_ends_on_its_own_removes_its_own_entry(monkeypatch,
                                                             registry):
    """The registry is filled and emptied by the streaming path itself.

    This is the other half of the race above and the reason the registry can
    be trusted at shutdown: a stream that simply finishes must take its own
    entry out. If it did not, every completed answer would leave a dead Popen
    behind, shutdown would walk a list of corpses, and a genuinely stuck child
    would be indistinguishable from the pile of finished ones.
    """
    child = _FakeChild(lines=["alpha\n", "omega\n"])
    _install_fake_popen(monkeypatch, child)
    provider = providers.CodexCliProvider()

    stream = providers._stream_process(["codex", "exec", "q"],
                                       provider=provider)
    assert next(stream) == "alpha\n"
    assert registry == [child], "the live stream was never registered"

    assert list(stream) == ["omega\n"]
    assert registry == []                       # removed by its own finally
    assert child.calls == ["wait"]              # exited normally, no signals
    assert provider._current_proc is None


def test_a_stream_escalation_reaps_the_child_after_killing_it(monkeypatch,
                                                               registry):
    """The reader-side teardown must reap after its SIGKILL escalation.

    This is distinct from global shutdown: a generator can be closed by its
    own consumer while the vendor CLI ignores both the initial wait and
    SIGTERM. Killing without the final wait leaves a zombie even though the
    registry no longer calls it live.
    """
    stubborn = _FakeChild(lines=["done\n"], wait_raises=2)
    _install_fake_popen(monkeypatch, stubborn)
    provider = providers.GeminiCliProvider()

    assert list(providers._stream_process(["gemini", "hello"],
                                          provider=provider)) == ["done\n"]

    assert stubborn.calls == [
        "wait", "terminate", "wait", "kill", "wait"]
    assert stubborn.wait_timeouts == [1, 1, 1]
    assert stubborn.running is False
    assert registry == []
    assert provider._current_proc is None
