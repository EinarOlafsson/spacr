"""The print-render process's parent and child sides, with every wait bounded.

The parent side is driven with a fake process (a real pipe for the sentinel)
and real pipes for the connection, so each way the process can fail to start,
answer or stop is reached without spawning anything; the child side is driven
in this process with a scripted connection.
"""
from __future__ import annotations

import multiprocessing
import os
import pickle
import threading
import time

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import figure_queue as fq

pytestmark = pytest.mark.qt


class _Process:
    """A process stand-in whose sentinel becomes readable when it ends."""

    def __init__(self, alive=True, stubborn=False):
        self._alive = alive
        self.stubborn = stubborn
        self.sentinel, self._write = os.pipe()
        self.killed = self.terminated = self.closed = 0

    def is_alive(self):
        return self._alive

    def _end(self):
        self._alive = False
        try:
            os.close(self._write)
        except OSError:
            pass

    def kill(self):
        self.killed += 1
        self._end()

    def terminate(self):
        self.terminated += 1
        if not self.stubborn:
            self._end()

    def join(self, timeout=None):
        return None

    def close(self):
        self.closed += 1
        if self._alive:
            raise ValueError("still running")


@pytest.fixture
def parts():
    """A fresh, unstarted renderer with a fake process on a real pipe."""
    renderer = fq._PrintRenderProcess()
    here, there = multiprocessing.Pipe()
    process = _Process()
    renderer._process, renderer._connection = process, here
    renderer._started = time.monotonic()
    yield renderer, process, there
    renderer._cancel_idle()
    for end in (here, there):
        try:
            end.close()
        except OSError:
            pass
    for descriptor in (process.sentinel,):
        try:
            os.close(descriptor)
        except OSError:
            pass


def test_a_process_that_cannot_be_started_leaves_the_caller_to_draw(
        monkeypatch):
    renderer = fq._PrintRenderProcess()

    def refuse(_method):
        raise OSError("no processes")

    monkeypatch.setattr(multiprocessing, "get_context", refuse)
    assert renderer.warm() is False
    assert not renderer.running()


def test_an_unpicklable_figure_is_left_to_the_caller():
    class Unpicklable:
        def __reduce__(self):
            raise TypeError("no")

    assert fq._PrintRenderProcess().render(Unpicklable(), "x.png", False) is None


def test_a_render_with_no_process_to_take_it_is_left_to_the_caller(
        monkeypatch):
    renderer = fq._PrintRenderProcess()
    monkeypatch.setattr(renderer, "warm", lambda: False)
    assert renderer.render(b"blob", "x.png", False) is None


def test_a_process_still_starting_is_kept_for_the_next_save(parts, monkeypatch):
    renderer, process, _there = parts
    rearmed = []
    monkeypatch.setattr(renderer, "_rearm", lambda: rearmed.append(1))
    monkeypatch.setattr(fq, "_PRINT_PROCESS_READY_S", 0.05)
    assert renderer.render(b"blob", "x.png", False) is None
    assert renderer._process is process and process.is_alive()
    assert len(rearmed) == 2


@pytest.mark.parametrize("answer, expected", [
    (fq._PRINT_PROCESS_WROTE, True), (fq._PRINT_PROCESS_REFUSED, False),
    (99, None)])
def test_each_answer_of_a_ready_process_is_reported(
        parts, monkeypatch, answer, expected):
    renderer, _process, _there = parts
    monkeypatch.setattr(renderer, "_wait_until_ready", lambda _t: True)
    monkeypatch.setattr(renderer, "_exchange", lambda _r, _t: answer)
    assert renderer.render(b"blob", "x.png", False) is expected
    renderer._cancel_idle()


def test_a_process_that_does_not_answer_is_ended(parts, monkeypatch):
    renderer, process, _there = parts
    monkeypatch.setattr(renderer, "_wait_until_ready", lambda _t: True)
    monkeypatch.setattr(renderer, "_exchange", lambda _r, _t: None)
    assert renderer.render(b"blob", "x.png", False) is None
    assert process.killed == 1 and renderer._process is None


def test_a_ready_process_is_not_waited_for_again(parts):
    renderer, _process, _there = parts
    renderer._ready = True
    assert renderer._wait_until_ready(0.01) is True


def test_a_process_saying_it_is_ready_is_ready(parts):
    renderer, _process, there = parts
    there.send(fq._PRINT_PROCESS_READY)
    assert renderer._wait_until_ready(2.0) is True
    assert renderer._ready is True


def test_a_process_saying_anything_else_is_ended(parts):
    renderer, process, there = parts
    there.send("not ready")
    assert renderer._wait_until_ready(2.0) is False
    assert process.killed == 1 and renderer._process is None


def test_a_process_whose_pipe_closed_while_starting_is_ended(parts):
    renderer, process, there = parts
    there.close()
    assert renderer._wait_until_ready(2.0) is False
    assert process.killed == 1 and renderer._process is None


def test_a_process_that_died_quietly_while_starting_is_ended(parts):
    renderer, process, _there = parts
    process._alive = False
    assert renderer._wait_until_ready(0.05) is False
    assert renderer._process is None


def test_a_young_silent_process_is_left_to_start(parts):
    renderer, process, _there = parts
    assert renderer._wait_until_ready(0.05) is False
    assert renderer._process is process and process.killed == 0


def test_a_process_starting_past_the_limit_is_ended(parts, monkeypatch):
    renderer, process, _there = parts
    monkeypatch.setattr(fq, "_PRINT_PROCESS_START_LIMIT_S", 0.0)
    assert renderer._wait_until_ready(0.05) is False
    assert process.killed == 1 and renderer._process is None


def test_an_answer_already_waiting_is_returned(parts):
    renderer, process, there = parts
    there.send(fq._PRINT_PROCESS_WROTE)
    assert renderer._exchange(("blob", "x.png", False), 5.0) == \
        fq._PRINT_PROCESS_WROTE
    assert process.killed == 0


def test_no_answer_in_time_is_none_and_the_watchdog_ends_the_process(parts):
    renderer, process, _there = parts
    assert renderer._exchange(("blob", "x.png", False), 0.05) is None
    deadline = time.monotonic() + 5
    while not process.killed and time.monotonic() < deadline:
        threading.Event().wait(0.01)
    assert process.killed >= 1


def test_a_send_into_a_broken_pipe_is_none(parts):
    renderer, _process, _there = parts

    class Broken:
        def send(self, _request):
            raise OSError("broken pipe")

    renderer._connection = Broken()
    assert renderer._exchange(("blob", "x.png", False), 1.0) is None


def test_killing_with_no_process_only_forgets(parts):
    renderer, _process, _there = parts
    renderer._process = None
    renderer._kill()
    assert renderer._connection is None


def test_stopping_asks_then_terminates_then_kills(parts):
    renderer, process, _there = parts
    process.stubborn = True

    class Closed:
        def send(self, _request):
            raise OSError("closed")

        def close(self):
            raise OSError("closed")

    renderer._connection = Closed()
    renderer.stop(timeout_s=0.01)
    assert process.terminated == 1 and process.killed == 1
    assert renderer._process is None and renderer._connection is None


def test_stopping_a_process_that_ends_when_asked_does_not_terminate_it(parts):
    renderer, process, there = parts

    class Polite:
        def send(self, _request):
            process._end()

        def close(self):
            return None

    renderer._connection = Polite()
    renderer.stop(timeout_s=0.01)
    assert process.terminated == 0 and process.killed == 0


def test_closing_forgets_a_process_that_will_not_close(parts):
    renderer, process, _there = parts
    renderer._ready = True
    renderer._close()
    assert renderer._ready is False and renderer._process is None
    assert process.closed == 0
    other = fq._PrintRenderProcess()
    ended = _Process(alive=False)
    other._process = ended
    other._close()
    assert ended.closed == 1


def test_closing_survives_a_dead_process_that_refuses_to_close():
    renderer = fq._PrintRenderProcess()
    refusing = _Process(alive=False)
    refusing.close = lambda: (_ for _ in ()).throw(ValueError("closed"))
    renderer._process = refusing
    renderer._close()
    assert renderer._process is None


def test_the_idle_clock_stops_the_process_when_it_runs_out(monkeypatch):
    renderer = fq._PrintRenderProcess()
    stopped = threading.Event()
    monkeypatch.setattr(renderer, "stop", stopped.set)
    monkeypatch.setattr(fq, "_PRINT_PROCESS_IDLE_S", 0.01)
    renderer._rearm()
    assert stopped.wait(5.0)
    renderer._cancel_idle()


def test_ending_a_process_that_cannot_be_asked_is_harmless():
    class Odd:
        def is_alive(self):
            raise OSError("gone")

    fq._end_a_process(Odd())


def test_a_live_process_is_killed_and_a_dead_one_left():
    live, dead = _Process(), _Process(alive=False)
    fq._end_a_process(live)
    fq._end_a_process(dead)
    assert live.killed == 1 and dead.killed == 0


def test_the_exit_hook_stops_the_shared_process(monkeypatch):
    stopped = []
    monkeypatch.setattr(fq._PRINT_RENDER_PROCESS, "stop",
                        lambda *a: stopped.append(1))
    fq._stop_the_print_render_process()
    assert stopped == [1]


def test_the_module_function_hands_the_render_to_the_shared_process(
        monkeypatch):
    seen = []
    monkeypatch.setattr(fq._PRINT_RENDER_PROCESS, "render",
                        lambda *a: seen.append(a) or True)
    assert fq._render_print_in_a_process(b"blob", "x.png", True) is True
    assert seen == [(b"blob", "x.png", True)]


class _Scripted:
    """The child's end of the pipe, answering from a script."""

    def __init__(self, requests, fail_send_at=None):
        self.requests = list(requests)
        self.sent = []
        self.fail_send_at = fail_send_at

    def send(self, message):
        if self.fail_send_at is not None and \
                len(self.sent) == self.fail_send_at:
            raise OSError("closed")
        self.sent.append(message)

    def recv(self):
        if not self.requests:
            raise EOFError
        return self.requests.pop(0)


def _figure_blob():
    from matplotlib.figure import Figure

    figure = Figure(figsize=(2, 2))
    figure.add_subplot(111).plot([0, 1], [0, 1])
    return pickle.dumps(figure)


def test_the_child_says_ready_then_answers_each_request(
        tmp_path, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    good, bad = tmp_path / "good.png", tmp_path / "bad.png"
    connection = _Scripted([(_figure_blob(), str(good), False),
                            (b"not a pickle", str(bad), False), None])
    fq._serve_print_renders(connection, True)
    assert preferences._SAFE_MODE is True
    assert connection.sent == [fq._PRINT_PROCESS_READY,
                               fq._PRINT_PROCESS_WROTE,
                               fq._PRINT_PROCESS_REFUSED]
    assert good.stat().st_size > 0 and not bad.exists()


def test_a_child_that_cannot_say_ready_ends(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    connection = _Scripted([None], fail_send_at=0)
    fq._serve_print_renders(connection, False)
    assert connection.sent == [] and preferences._SAFE_MODE is False


def test_a_child_whose_parent_left_ends_after_its_render(
        tmp_path, monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    connection = _Scripted([(_figure_blob(), str(tmp_path / "a.png"), False),
                            None], fail_send_at=1)
    fq._serve_print_renders(connection, False)
    assert connection.sent == [fq._PRINT_PROCESS_READY]
    assert connection.requests == [None]


@pytest.mark.parametrize("error", [EOFError, OSError])
def test_a_closed_pipe_asks_the_child_to_stop(error):
    class Closed:
        def recv(self):
            raise error

    assert fq._next_print_request(Closed()) is None
