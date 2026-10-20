"""A field frame over budget during a run waits before the next one.

During lesson 13's run a 3840x2105 field frame took a median 39 ms of GUI
thread, and every return to Python then waited behind the pipeline's hold on
the interpreter lock, so backdrop frames filled the GUI thread. While a run
paces the field, a frame over its budget now holds the next tick back until
the frame's own cost has been matched by free time. Idle frames, and frames
within budget, keep the normal rate, and no pixel changes. Counted and
compared, not timed.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QColor, QImage  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture()
def field(qtbot):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(320, 200)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30)
    backdrop.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    backdrop._kept_host = host
    if not backdrop._timer.isActive():
        backdrop._timer.start()
    return backdrop


def _pace(backdrop, paced: bool, spent: float) -> int:
    backdrop._run_paced = paced
    backdrop._apply_rate()
    backdrop._frame_cost_ms = spent
    backdrop._schedule_next_tick()
    return backdrop._tick_interval


def test_an_over_budget_frame_in_a_run_holds_the_next_tick(field):
    period = 1000.0 / min(field._fps, ambient._RUN_FPS)
    spent = 120.0
    interval = _pace(field, True, spent)
    held = spent * (1.0 / ambient._OVER_BUDGET_SHARE - 1.0)
    assert interval >= held - 2
    assert interval > period * 2


def test_a_frame_within_budget_keeps_the_rate(field):
    period = 1000.0 / min(field._fps, ambient._RUN_FPS)
    interval = _pace(field, True, ambient._FRAME_BUDGET_MS / 2)
    assert interval <= period + 2


def test_idle_frames_are_never_held_back(field):
    period = 1000.0 / field._fps
    interval = _pace(field, False, 400.0)
    assert interval <= period + 2


@pytest.mark.parametrize("background", ["#191919", "#f4f4f4"])
def test_pacing_changes_no_pixel(qtbot, background):
    app = QApplication.instance()
    sheet = app.styleSheet()
    app.setStyleSheet("")
    try:
        host = QWidget()
        host.setProperty("spacrIndependentBackdrop", True)
        host.resize(320, 200)
        backdrop = ambient.AmbientWidget(host, theme="blobs", fps=30,
                                         background=QColor(background))
        backdrop.setGeometry(host.rect())
        qtbot.addWidget(host)
        host.show()
        qtbot.waitExposed(host)
        backdrop._timer.stop()
        producer = backdrop._producer_box[0]
        if producer is not None:
            producer.stop()
        before = backdrop.grab().toImage().convertToFormat(QImage.Format_RGB32)
        backdrop._run_paced = True
        backdrop._frame_cost_ms = 500.0
        backdrop._schedule_next_tick()
        after = backdrop.grab().toImage().convertToFormat(QImage.Format_RGB32)
        assert before == after
    finally:
        app.setStyleSheet(sheet)
