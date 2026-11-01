"""Save graph writes its PNG even when the print-render process hangs or dies.

A real capture once got no PNG within 30 s: the save waited on the spawned
print process without a limit. Every wait is now bounded; a process that does
not answer is ended and the graph is drawn on the worker. The hang is real
here: the process is stopped with SIGSTOP, or killed, at the moment that
matters.
"""
from __future__ import annotations

import os
import signal
import sys
import time

import pytest
from PySide6.QtWidgets import QFileDialog

from tests.qt.test_gate_editor_work_stays_off_the_gui_thread import (  # noqa: F401
    STALL_BUDGET_S, _gap_while, _pump, screen)

pytestmark = pytest.mark.skipif(sys.platform == "win32",
                                reason="SIGSTOP simulates the hung process")

ANSWER_S = 1.5


def _figure():
    import numpy as np
    from matplotlib.figure import Figure

    rng = np.random.default_rng(5)
    figure = Figure(figsize=(4, 3))
    figure.add_subplot(111).scatter(rng.random(5000), rng.random(5000), s=1)
    return figure


@pytest.fixture
def renderer(monkeypatch):
    from spacr.qt.widgets import figure_queue

    monkeypatch.setattr(figure_queue, "_PRINT_PROCESS_ANSWER_S", ANSWER_S)
    process = figure_queue._PRINT_RENDER_PROCESS
    process.stop()
    yield process
    process.stop()
    assert not process.running()


def _ready(renderer):
    assert renderer.warm()
    with renderer._lock:
        assert renderer._wait_until_ready(120.0)
    return renderer._process.pid


def _gone(pid, within=5.0):
    end = time.time() + within
    while time.time() < end:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def _pid(renderer):
    process = renderer._process
    return None if process is None else process.pid


def _save(tmp_path, name):
    from spacr.qt.screens.gate_editor import _render_the_graph

    png = tmp_path / name
    started = time.perf_counter()
    assert _render_the_graph(_figure(), str(png), False) is True
    assert png.stat().st_size > 1000
    return time.perf_counter() - started


def test_a_stopped_process_is_ended_and_the_save_is_drawn_here(
        renderer, tmp_path):
    pid = _ready(renderer)
    os.kill(pid, signal.SIGSTOP)
    spent = _save(tmp_path, "stopped.png")
    assert spent < ANSWER_S + 10.0, spent
    assert _gone(pid)
    assert not renderer.running()
    _save(tmp_path, "after.png")
    assert renderer.running()
    assert _pid(renderer) != pid


def test_a_dead_process_is_noticed_without_waiting_out_the_limit(
        renderer, tmp_path):
    pid = _ready(renderer)
    os.kill(pid, signal.SIGKILL)
    renderer._process.join(5.0)
    spent = _save(tmp_path, "dead.png")
    assert spent < ANSWER_S + 10.0, spent
    assert _pid(renderer) != pid


def test_a_process_that_never_gets_ready_does_not_hold_the_save(
        renderer, monkeypatch, tmp_path):
    from spacr.qt.widgets import figure_queue

    monkeypatch.setattr(figure_queue, "_PRINT_PROCESS_READY_S", 0.3)
    assert renderer.warm()
    process = renderer._process
    pid = process.pid
    os.kill(pid, signal.SIGSTOP)
    _save(tmp_path, "starting.png")
    assert renderer._process is process and process.is_alive()
    monkeypatch.setattr(figure_queue, "_PRINT_PROCESS_START_LIMIT_S", 0.0)
    _save(tmp_path, "gave_up.png")
    assert _gone(pid)
    assert renderer._process is None


def test_a_hung_process_leaves_the_window_turning_and_the_file_written(
        screen, renderer, monkeypatch, tmp_path):
    pid = _ready(renderer)
    os.kill(pid, signal.SIGSTOP)
    target = tmp_path / "graph.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    lines = []
    monkeypatch.setattr(screen.console, "write", lines.append)
    gap = _gap_while(lambda: screen.save_graph_in_background(),
                     lambda: any(line.startswith("Saved the graph")
                                 for line in lines),
                     limit=ANSWER_S + 20.0)
    assert target.stat().st_size > 1000
    assert lines == [f"Saved the graph to {target}"]
    assert gap < STALL_BUDGET_S
    assert _gone(pid)
