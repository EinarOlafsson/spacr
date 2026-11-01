"""A field shown by a fold paints at rate at once, and a field in a run keeps its rate.

Lesson 19 (a folded screen's own field painted once in its first second:
its first frame was shaded on the GUI thread at show) and lesson 21 (a run
loading image pairs made field repaints 40-45 ms, and a frame over the 20 ms
budget halved the paced rate again: 30 fell to 11 a second). The first frame
of a hidden field is now shaded off the GUI thread while its screen is built,
and the hold-back after a long frame is kept for frames four budgets long.
Counted with a paint counter, not timed by the clock alone.
"""
from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from spacr.qt.widgets import ambient  # noqa: E402

pytestmark = pytest.mark.qt


def _pump(seconds):
    app = QApplication.instance()
    end = time.time() + seconds
    while time.time() < end:
        app.processEvents()
        time.sleep(0.003)


@pytest.fixture()
def field(qtbot):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(320, 200)
    backdrop = ambient.AmbientWidget(host, theme="blobs", fps=60)
    backdrop.setGeometry(host.rect())
    qtbot.addWidget(host)
    host.show()
    qtbot.waitExposed(host)
    backdrop._kept_host = host
    if not backdrop._timer.isActive():
        backdrop._timer.start()
    return backdrop


def _interval(backdrop, spent):
    backdrop._run_paced = True
    backdrop._apply_rate()
    backdrop._frame_cost_ms = spent
    backdrop._schedule_next_tick()
    return backdrop._tick_interval


def test_a_frame_between_the_budget_and_four_budgets_keeps_the_paced_rate(field):
    period = 1000.0 / min(field._fps, ambient._RUN_FPS)
    for spent in (ambient._FRAME_BUDGET_MS * 1.2, 45.0, 75.0):
        assert _interval(field, spent) <= period + 2, spent


def test_a_frame_over_four_budgets_still_holds_the_next_tick(field):
    spent = ambient._HOLD_BACK_ABOVE_MS * 1.5
    held = spent * (1.0 / ambient._OVER_BUDGET_SHARE - 1.0)
    assert _interval(field, spent) >= held - 2


def test_the_first_frame_of_a_hidden_field_is_kept_and_used_at_show(
        qtbot, monkeypatch):
    shaded_on = []
    original = ambient._DataArtEngine.shade

    def counting(self, *args, **kwargs):
        import threading

        shaded_on.append(threading.current_thread().name)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(ambient._DataArtEngine, "shade", counting)
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(320, 200)
    backdrop = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=30)
    backdrop.setGeometry(host.rect())
    qtbot.addWidget(host)
    backdrop._warm_first_frame((640, 400))
    assert backdrop._warmed_first_frame[0] == (640, 400)
    before = len(shaded_on)
    host.show()
    qtbot.waitExposed(host)
    _pump(0.3)
    main_thread = [name for name in shaded_on[before:]
                   if name == "MainThread"]
    assert backdrop._producer_box[0] is not None
    assert "_warmed_first_frame" not in backdrop.__dict__
    assert main_thread == []


def test_a_kept_frame_smaller_than_the_field_is_not_used(qtbot):
    host = QWidget()
    host.setProperty("spacrIndependentBackdrop", True)
    host.resize(320, 200)
    backdrop = ambient.AmbientWidget(host, theme="data_art_impulse_lens", fps=30)
    backdrop.setGeometry(host.rect())
    qtbot.addWidget(host)
    backdrop._warm_first_frame((64, 40))
    host.show()
    qtbot.waitExposed(host)
    _pump(0.2)
    assert "_warmed_first_frame" not in backdrop.__dict__
    assert backdrop._producer_box[0] is not None
