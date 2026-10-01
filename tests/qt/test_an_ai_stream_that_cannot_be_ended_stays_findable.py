"""An AI CLI child that cannot be confirmed gone stays findable for shutdown.

Pinned behaviour of :func:`spacr.qt.ai.providers._stream_process`:

* a child whose exit can neither be waited for nor forced -- ``wait`` times
  out, ``terminate`` is refused, and the handle does not even accept the
  "spaCR stopped it" mark -- still has its output delivered, and stays in
  the live-stream registry and on the provider so a later shutdown can try
  again;
* a CLI that fails after printing blank lines is quoted without them.

The transport boundary is ``subprocess.Popen``; no vendor CLI is spawned.
"""
from __future__ import annotations

import subprocess

import pytest

pytest.importorskip("PySide6")

from spacr.qt.ai import providers  # noqa: E402

pytestmark = pytest.mark.qt


class _Stdout:
    def __init__(self, lines):
        self._lines = iter(lines)
        self.closed = False

    def __iter__(self):
        return self._lines

    def close(self):
        self.closed = True


class _UnendableChild:
    """A Popen stand-in that never confirms an exit and takes no new attributes."""

    __slots__ = ("stdout", "stdin", "calls")

    def __init__(self, lines):
        self.stdout = _Stdout(lines)
        self.stdin = None
        self.calls = []

    def poll(self):
        self.calls.append("poll")
        return None

    def wait(self, timeout=None):
        self.calls.append("wait")
        raise subprocess.TimeoutExpired(cmd="fake-cli", timeout=timeout or 0)

    def terminate(self):
        self.calls.append("terminate")
        raise PermissionError("operation not permitted")


class _FailingChild:
    def __init__(self, lines, status):
        self.stdout = _Stdout(lines)
        self.stdin = None
        self._status = status

    def wait(self, timeout=None):
        return self._status


@pytest.fixture()
def registry(monkeypatch):
    live = []
    monkeypatch.setattr(providers, "_LIVE_STREAMS", live)
    return live


def _popen_returns(monkeypatch, child):
    monkeypatch.setattr(providers.subprocess, "Popen",
                        lambda argv, **kwargs: child)


def test_a_child_that_cannot_be_ended_stays_registered(monkeypatch, registry):
    child = _UnendableChild(["partial answer\n"])
    _popen_returns(monkeypatch, child)
    provider = providers.ClaudeCliProvider()

    lines = list(providers._stream_process(["claude", "-p", "hi"],
                                           provider=provider))

    assert lines == ["partial answer\n"]
    assert child.stdout.closed is True
    assert child.calls == ["wait", "terminate", "poll"]
    assert registry == [child]
    assert provider._current_proc is child


def test_a_failure_is_quoted_without_the_blank_lines(monkeypatch, registry):
    child = _FailingChild(["\n", "Error: not signed in\n", "   \n"], status=2)
    _popen_returns(monkeypatch, child)

    stream = providers._stream_process(["codex", "exec", "q"])
    with pytest.raises(providers.ProviderFailed) as failure:
        list(stream)

    assert failure.value.exit_status == 2
    assert failure.value.output_tail == "Error: not signed in"
    assert registry == []
