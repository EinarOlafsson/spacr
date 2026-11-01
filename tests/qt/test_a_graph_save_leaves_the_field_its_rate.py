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

    target = tmp_path / "graph.png"
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(target), "")))
    stamps.clear()
    started = time.perf_counter()
    held = []
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
