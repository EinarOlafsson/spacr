"""A queued job whose child process ignores the polite stop is killed.

Stop has to end the job, not ask it to end. Cellpose wedged on a CUDA driver
does not act on SIGTERM, and a queue that sent one and then waited would hold
the remaining jobs -- and the GUI's Stop button -- open for the rest of the
night.

The order matters as much as the outcome: terminate first, wait out a grace
period, and only then kill. A child that would have exited cleanly on SIGTERM
gets the chance to flush its log before anything harsher arrives.

This is deliberately timed against a child that has SAID it is ready. The
grace period only elapses if the ignore handler is already installed when the
signal lands, and a child that is asked to stop while the interpreter is still
starting up dies on the first SIGTERM -- which looks like a pass and proves
nothing about the kill.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time

import pytest

from spacr import batch
from spacr.batch import Job
from spacr.cancellation import (
    CancellationToken, PipelineCancelled, installed_token,
)


def _deaf_child(ready_path):
    """A child that ignores SIGTERM and says so before it starts sleeping."""
    return [sys.executable, "-c", textwrap.dedent(f"""
        import pathlib, signal, sys, time
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        pathlib.Path({str(ready_path)!r}).write_text("deaf")
        sys.stdout.flush()
        time.sleep(120)
    """)]


def test_a_child_that_ignores_sigterm_is_killed_after_the_grace_period(
        tmp_path, monkeypatch):
    """SIGTERM, five seconds, then SIGKILL -- and the queue is free again."""
    log = tmp_path / "01_mask-1.log"
    ready = tmp_path / "child-is-deaf"
    monkeypatch.setattr(batch, "job_command",
                        lambda job, settings, python=None: _deaf_child(ready))

    started = {}
    real_popen = subprocess.Popen

    def recording(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        started["process"] = process
        return process

    monkeypatch.setattr(batch.subprocess, "Popen", recording)

    token = CancellationToken()

    def stop_once_the_child_is_deaf():
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline and not ready.exists():
            time.sleep(0.02)
        token.cancel("the user pressed Stop")

    watcher = threading.Thread(target=stop_once_the_child_is_deaf, daemon=True)
    watcher.start()

    began = time.monotonic()
    with installed_token(token):
        with pytest.raises(PipelineCancelled):
            batch.subprocess_runner(Job(module="mask", id="mask-1"), "",
                                    str(log))
    elapsed = time.monotonic() - began
    watcher.join(timeout=5.0)

    assert ready.exists(), "the child never installed its SIGTERM handler"
    process = started["process"]
    assert process.poll() is not None, "the wedged child outlived the queue"
    assert process.returncode == -9, (
        f"the child ended with {process.returncode}, not by SIGKILL")
    assert elapsed >= 5.0, (
        "the child was killed without being given the chance to exit cleanly")
    assert "cancelled safely" in log.read_text()
