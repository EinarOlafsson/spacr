"""A background graph save leaves the backdrop field its frame rate.

Workstation, Gate Editor "Save graph": while the worker drew the print
raster (matplotlib Agg, then the PNG), the field paced at 30 delivered 22-27
frames a second and the GUI thread sat waiting in ``_on_tick``. Two causes:
Agg draws a scatter in one call that keeps the interpreter lock for tens of
milliseconds, so no switch interval lets the GUI thread in, and a paced
field whose frame came in just after its tick showed it a whole period late.
The print raster is drawn in its own process now, a paced field asks again
on the same beat and starts the next frame before it repaints. Each repaint
here costs what the window over the field costs on the workstation (18 ms,
over the limit that paces a run), and the field is counted against the rate
it is paced to, within 10 %.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox, QWidget  # noqa: E402

from spacr.qt import gil_priority  # noqa: E402
from spacr.qt.screens.gate_editor import GateEditorScreen  # noqa: E402
from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt

FIELD_FPS = 60
REPAINT_S = 0.018
TOLERANCE = 0.10
#: Saves made back to back and counted as one span: a save served by the
#: ready process took 0.21 s, too short alone to count a rate over.
SAVES = 6


@pytest.fixture(autouse=True)
def _a_field_that_runs(monkeypatch):
    from spacr.qt import preferences

    monkeypatch.setattr(preferences, "get_reduce_motion", lambda: False)
    monkeypatch.setattr(ambient, "_system_asks_for_stillness", lambda: False)
    yield
    ambient._retire_shading_threads(timeout_s=3.0)


def _pump(seconds, until=None):
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.0005)
    return until() if until is not None else True


def _a_ready_print_process():
    """Start the shared print-render process and wait for its ready signal.

    A save waits at most ``_PRINT_PROCESS_READY_S`` (3 s) for a process
    that is still importing and then draws on the worker. A cold start in a
    test (every test begins without one) took 3.8-8.3 s on a loaded
    workstation, so a test of the process itself warms it first, as the save
    dialog does while it is open.
    """
    from spacr.qt.widgets import figure_queue

    renderer = figure_queue._PRINT_RENDER_PROCESS
    assert renderer.warm()
    with renderer._lock:
        assert renderer._wait_until_ready(120.0), "the process never got ready"
    return renderer


def _table(rows=200000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"area": rng.normal(0, 1, rows),
                         "intensity": rng.normal(0, 1, rows)})


def test_the_field_keeps_its_rate_while_the_gate_graph_is_saved(
        qapp, qtbot, monkeypatch, tmp_path):
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    screen = GateEditorScreen()
    qtbot.addWidget(screen)
    screen.resize(1200, 800)
    screen.show()
    screen.set_frame(_table(), label="test")
    _pump(1.0)
    screen._y.setCurrentText("intensity")
    _pump(0.6)

    stamps = []
    original = ambient.AmbientWidget.paintEvent

    def counting(self, event):
        stamps.append(time.perf_counter())
        time.sleep(REPAINT_S)
        return original(self, event)

    monkeypatch.setattr(ambient.AmbientWidget, "paintEvent", counting)
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(1280, 720)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=FIELD_FPS)
    field.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    _pump(1.0)

    targets = [tmp_path / f"graph{index}.png" for index in range(SAVES)]
    chosen = iter(targets)
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(next(chosen)), "")))
    _a_ready_print_process()
    stamps.clear()
    started = time.perf_counter()
    held = []
    for target in targets:
        screen.save_graph_in_background()
        assert _pump(120, lambda: (held.append(gil_priority.active())
                                   or (target.exists() and target.stat().st_size > 1000)))
    finished = time.perf_counter()
    span = finished - started
    assert any(held), "the save ran without the responsive-GUI claim"
    assert span > 0.5, f"the save took {span:.2f} s, too short to count a rate"
    target_rate = field._rate()
    assert target_rate == ambient._RUN_FPS, "the expensive field was not paced"
    rate = sum(1 for t in stamps if started <= t <= finished) / span
    assert rate >= target_rate * (1.0 - TOLERANCE), (
        f"the field delivered {rate:.1f} of {target_rate} frames a second "
        f"over the {span:.1f} s save")


def test_a_paced_field_whose_frame_lands_after_its_tick_keeps_its_rate(
        qtbot, monkeypatch):
    """No worker: a 25 ms shade behind an 18 ms repaint delivered 15 of 30."""
    stamps = []
    original = ambient.AmbientWidget.paintEvent

    def counting(self, event):
        stamps.append(time.perf_counter())
        time.sleep(REPAINT_S)
        return original(self, event)

    shade = ambient._DataArtEngine.shade

    def slow(self, *args, **kwargs):
        time.sleep(0.025)
        return shade(self, *args, **kwargs)

    monkeypatch.setattr(ambient.AmbientWidget, "paintEvent", counting)
    monkeypatch.setattr(ambient._DataArtEngine, "shade", slow)
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(640, 360)
    field = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=FIELD_FPS)
    field.setGeometry(host.rect())
    qtbot.addWidget(host)
    gil_priority.claim()
    try:
        host.show()
        _pump(1.0)
        stamps.clear()
        _pump(3.0)
        target_rate = field._rate()
    finally:
        gil_priority.release()
    assert target_rate == ambient._RUN_FPS
    rate = len(stamps) / 3.0
    assert rate >= target_rate * (1.0 - TOLERANCE), (target_rate, rate)
    assert rate <= target_rate * (1.0 + TOLERANCE), (target_rate, rate)


def _scatter_figure(points=200000):
    from matplotlib.figure import Figure

    rng = np.random.default_rng(3)
    figure = Figure(figsize=(6, 4))
    figure.add_subplot(111).scatter(rng.random(points), rng.random(points), s=1)
    return figure


def test_saves_share_one_warm_process_no_slower_than_drawing_here(tmp_path):
    """One process serves every save, a warm save costs no more than a draw
    on the worker did, and stopping it ends the process."""
    from spacr.qt.widgets import figure_queue

    renderer = figure_queue._PRINT_RENDER_PROCESS
    figure_queue.render_figure_to_png(_scatter_figure(), str(tmp_path / "w.png"),
                                      for_print=True, write_pdf=False)
    started = time.perf_counter()
    assert figure_queue.render_figure_to_png(
        _scatter_figure(), str(tmp_path / "here.png"), for_print=True,
        write_pdf=False)
    here = time.perf_counter() - started
    try:
        assert _a_ready_print_process() is renderer
        assert figure_queue._render_print_in_a_process(
            _scatter_figure(), str(tmp_path / "first.png"), False) is True
        pid = renderer._process.pid
        spent = []
        for index in range(3):
            started = time.perf_counter()
            assert figure_queue._render_print_in_a_process(
                _scatter_figure(), str(tmp_path / f"{index}.png"), False) is True
            spent.append(time.perf_counter() - started)
            assert renderer._process.pid == pid
        assert (tmp_path / "2.png").stat().st_size > 1000
        assert min(spent) <= here * 1.25 + 0.05, (spent, here)
    finally:
        renderer.stop()
    assert not renderer.running()


def test_a_pickled_snapshot_writes_the_files_a_copied_figure_wrote(
        tmp_path, monkeypatch):
    """Save graph pickles the figure on the GUI thread and hands the bytes
    on; it unpickled a copy there first. The process, and the worker when no
    process takes the save, write the same PNG either way."""
    import pickle

    from spacr.qt.screens import gate_editor
    from spacr.qt.widgets import figure_queue
    from spacr.qt.widgets.save_figure_dialog import copy_figure

    figure = _scatter_figure()
    renderer = _a_ready_print_process()
    try:
        assert figure_queue._render_print_in_a_process(
            copy_figure(figure), str(tmp_path / "copied.png"), False) is True
        assert gate_editor._render_the_graph(
            pickle.dumps(figure), str(tmp_path / "pickled.png"), False) is True
    finally:
        renderer.stop()
    assert (tmp_path / "pickled.png").read_bytes() == \
        (tmp_path / "copied.png").read_bytes()
    monkeypatch.setattr(figure_queue, "_render_print_in_a_process",
                        lambda *a, **k: None)
    assert figure_queue.render_figure_to_png(
        copy_figure(figure), str(tmp_path / "here_copied.png"),
        for_print=True, write_pdf=False)
    assert gate_editor._render_the_graph(
        pickle.dumps(figure), str(tmp_path / "here_pickled.png"), False)
    assert (tmp_path / "here_pickled.png").read_bytes() == \
        (tmp_path / "here_copied.png").read_bytes()


def test_save_graph_unpickles_nothing_on_the_gui_thread(
        qapp, monkeypatch, tmp_path):
    import pickle
    import threading

    from spacr.qt.widgets import save_figure_dialog

    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: None))
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: None))
    screen = GateEditorScreen()
    screen.resize(1200, 800)
    screen.show()
    screen.set_frame(_table(20000), label="test")
    _pump(1.0)
    screen._y.setCurrentText("intensity")
    _pump(0.6)
    on_gui = []
    loads = pickle.loads

    def watched(*args, **kwargs):
        if threading.current_thread() is threading.main_thread():
            on_gui.append("loads")
        return loads(*args, **kwargs)

    monkeypatch.setattr(pickle, "loads", watched)
    monkeypatch.setattr(save_figure_dialog, "copy_figure",
                        lambda *_a, **_k: on_gui.append("copy"))
    target = tmp_path / "graph.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    _a_ready_print_process()
    screen.save_graph_in_background()
    assert _pump(60, lambda: target.exists() and target.stat().st_size > 1000)
    assert on_gui == []
    screen.close()
