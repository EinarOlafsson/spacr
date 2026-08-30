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
teardown and the global shutdown both reached for it. Every one of those must
end with the child dead and the registry empty, because the alternative is a
blocked reader thread that takes the application down with it.

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

def test_a_child_that_ignores_terminate_is_killed_not_merely_asked(registry):
    """Asking politely is not enough to free a blocked reader thread.

    ``terminate()`` only requests an exit; a vendor CLI mid-tool-call can sit
    on SIGTERM. If shutdown stopped there, the reader thread would still be
    blocked on that child's stdout when Qt collected its QThread wrapper and
    the application would abort on close instead of quitting. So a child that
    has not been reaped within the one-second wait must be killed outright --
    while a child that does exit on SIGTERM must NOT be killed, because a
    gratuitous SIGKILL costs the CLI its own cleanup (session files, logs).
    """
    stubborn = _FakeChild(wait_raises=1)        # the reaping wait times out
    polite = _FakeChild()                       # dies on SIGTERM
    registry.extend([stubborn, polite])

    assert providers.terminate_all_streams() == 2

    assert stubborn.calls == ["poll", "terminate", "wait", "kill"]
    assert polite.calls == ["poll", "terminate", "wait"]   # no gratuitous kill
    assert stubborn.running is False
    assert polite.running is False
    assert registry == [], "a terminated stream stayed in the registry"


def test_a_child_that_cannot_be_signalled_does_not_strand_the_others(registry):
    """One unreachable process must not abandon every process behind it.

    The children are terminated in a plain loop, so an OSError from one
    ``terminate()`` -- the child was already reaped by the OS, or its pid is
    gone -- would, unhandled, abandon every stream later in the list. Those
    readers stay blocked and the abort-on-exit is back, caused by a process
    that was ALREADY dead. The failure is swallowed per-child instead, and the
    dead one is still dropped from the registry so shutdown does not retry it
    forever; only the child actually asked to stop is counted.
    """
    unreachable = _FakeChild(terminate_raises=OSError("no such process"))
    healthy = _FakeChild()
    registry.extend([unreachable, healthy])

    # The one that raised is not counted; the one behind it still is.
    assert providers.terminate_all_streams() == 1

    assert unreachable.calls == ["poll", "terminate"]      # blew up mid-step
    assert healthy.calls == ["poll", "terminate", "wait"]  # reached anyway
    assert healthy.running is False
    assert registry == [], "the unsignallable entry was left to be retried"


def test_an_already_deregistered_stream_is_not_removed_twice(registry):
    """The reader's own teardown and the global shutdown race for one entry.

    Both remove the child from ``_LIVE_STREAMS``: the reader thread does it in
    its ``finally`` when the stream ends, and shutdown does it after
    terminating. Terminating is exactly what makes the reader's stream end, so
    the reader can win that race and take the entry out first. A bare
    ``list.remove`` would then raise ValueError out of shutdown, leaving every
    later child alive with its thread still blocked -- the crash-on-exit this
    whole mechanism exists to prevent, triggered by cleanup working too well.
    """
    def _reader_wins_the_race(child):
        registry.remove(child)          # the reader's finally got there first

    racer = _FakeChild(on_terminate=_reader_wins_the_race)
    behind_it = _FakeChild()
    registry.extend([racer, behind_it])

    assert providers.terminate_all_streams() == 2

    assert racer.calls == ["poll", "terminate", "wait"]
    # The one behind the race was still reached and removed by shutdown.
    assert behind_it.calls == ["poll", "terminate", "wait"]
    assert behind_it.running is False
    assert registry == []


def test_a_dead_entry_is_dropped_without_being_signalled(registry):
    """A crashed reader leaves a corpse in the registry; shutdown must clear it.

    Entries are normally removed by the reader that owns them, so a reader
    that died between the child exiting and the removal leaves a finished
    Popen behind. Shutdown must not report it as a stream it ended (the count
    is what the caller shows and waits on) and must not signal a pid that the
    OS may already have recycled -- but it must still drop the entry, or the
    registry grows a corpse per crash for the life of the session.
    """
    corpse = _FakeChild()
    corpse.terminate()                  # already exited, before shutdown ran
    corpse.calls.clear()
    still_running = _FakeChild()
    registry.extend([corpse, still_running])

    assert providers.terminate_all_streams() == 1   # only the live one counts

    assert corpse.calls == ["poll"], "a finished child was signalled again"
    assert still_running.calls == ["poll", "terminate", "wait"]
    assert registry == []


# ---------------------------------------------------------------------------
# _stream_process — the reader side of the same race
# ---------------------------------------------------------------------------

def test_a_shutdown_mid_stream_leaves_the_reader_nothing_to_remove(monkeypatch,
                                                                   registry):
    """Closing the app mid-answer must not raise inside the reader thread.

    The user hits quit while a provider is still streaming: shutdown
    terminates the child and takes its entry out of the registry, then the
    reader wakes from its blocked read and runs its own ``finally``, which
    reaches for the same entry. If that second removal raised, the exception
    would surface from the streaming generator -- the console would report a
    ValueError as if the model had failed, on every single quit-while-typing.
    The reader must instead finish quietly: stdout closed, the provider's
    process reference cleared so a later cancel does not poke a dead pid.
    """
    child = _FakeChild(lines=["first chunk\n", "second chunk\n"])
    seen = _install_fake_popen(monkeypatch, child)
    provider = providers.ClaudeCliProvider()

    stream = providers._stream_process(["claude", "-p", "hi"],
                                       provider=provider)
    assert next(stream) == "first chunk\n"
    assert seen["argv"] == ["claude", "-p", "hi"]
    # Registered while the read is in flight — this is what shutdown finds.
    assert registry == [child]
    assert provider._current_proc is child

    # The user quits: the global shutdown ends the child and deregisters it.
    assert providers.terminate_all_streams() == 1
    assert registry == []

    # The reader drains what already arrived and tears down over the gap.
    assert list(stream) == ["second chunk\n"]
    assert child.stdout.closed is True
    assert provider._current_proc is None
    assert registry == []


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
