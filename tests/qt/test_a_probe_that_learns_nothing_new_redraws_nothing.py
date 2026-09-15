"""A background answer that changes nothing redraws nothing -- on every run.

``path_probe`` answers from its cache and checks an unknown path on a worker
thread. When the check finishes, ``probes.answered`` emits only if the answer
differs from what the cache already holds, so a widget redraws for news and
not for a repeat.

The repeat arm is taken when the cache is filled WHILE the worker's stat is
still running: ``prime()`` from a file dialog that already knows the path is
there, or a ``wait=True`` caller that answered first. That is a race, and so
was its coverage. CI runs with no source change counted ``path_probe.py`` at
one and then two uncovered branches, and a coverage ratchet that moves with
thread scheduling fails a release on a coin toss.

These tests force the ordering instead of hoping for it. The worker's stat for
one path is held on an Event until the test has done what the race would have
done, so each arm runs every time. Nothing sleeps.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

from spacr.qt import path_probe

pytestmark = pytest.mark.qt

#: Upper bound on every wait. Only reached when something is broken; the
#: events it waits for arrive in milliseconds.
WAIT_S = 30.0


@pytest.fixture
def held(monkeypatch, tmp_path):
    """A path whose background stat does not return until the test says so.

    The emitted answers are recorded in the worker thread itself, through a
    stand-in for ``probes``, so no event loop has to run for them to be seen.
    Any other path still gets the real stat.
    """
    path = str(tmp_path / "a_path_the_probe_has_never_seen")
    started, release = threading.Event(), threading.Event()
    real_stat = path_probe._stat_with_timeout
    emitted = []

    def stat(text, want_dir=False):
        if text != path:
            return real_stat(text, want_dir)
        started.set()
        release.wait(WAIT_S)        # never raise here: it would end a worker
        return True

    monkeypatch.setattr(path_probe, "_stat_with_timeout", stat)
    monkeypatch.setattr(path_probe, "probes", SimpleNamespace(
        answered=SimpleNamespace(
            emit=lambda text, answer: emitted.append((text, answer)))))
    path_probe.forget(path)
    try:
        yield SimpleNamespace(path=path, started=started, release=release,
                              emitted=emitted)
    finally:
        release.set()
        _drained()
        path_probe.forget(path)


def _drained() -> bool:
    """Wait, bounded, until every queued probe has finished."""
    done = threading.Event()

    def join():
        path_probe._queue.join()
        done.set()

    threading.Thread(target=join, daemon=True).start()
    return done.wait(WAIT_S)


def _ask_and_hold(held) -> None:
    """Ask about the path, and wait until a worker is inside its stat."""
    assert path_probe.exists(held.path) is True, "the optimistic default"
    assert held.started.wait(WAIT_S), "no worker ever probed the path"


def _answers_for(held):
    return [answer for text, answer in held.emitted if text == held.path]


def test_a_worker_answer_the_cache_already_had_emits_nothing(held):
    _ask_and_hold(held)
    path_probe.prime(held.path, True)       # the file dialog got there first
    held.release.set()
    assert _drained()

    assert _answers_for(held) == [], (
        "the probe told widgets to redraw for an answer they already had")
    assert path_probe.known(held.path) is True


def test_a_first_answer_is_news_and_is_emitted_once(held):
    """The same held stat with nothing primed: the control for the test above."""
    _ask_and_hold(held)
    held.release.set()
    assert _drained()

    assert _answers_for(held) == [True]
    assert path_probe.known(held.path) is True


def test_a_worker_answer_that_corrects_a_primed_one_is_emitted(held):
    """A primed answer is a claim; the filesystem's answer wins, out loud."""
    _ask_and_hold(held)
    path_probe.prime(held.path, False)
    held.release.set()
    assert _drained()

    assert _answers_for(held) == [True]
    assert path_probe.known(held.path) is True
