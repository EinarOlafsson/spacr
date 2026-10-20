"""A slow shader must consume GUI clock and mouse input between frames."""

import threading
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QTimer, Qt
from PySide6.QtGui import QCursor
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

from spacr.qt.widgets import ambient


def _widget(qtbot, family="impulse_lens"):
    widget = ambient.AmbientWidget(theme=f"data_art_{family}", palette="spacr",
                                   background="#101418", fps=60, seed=42,
                                   blur=0, speed=1, size=1, resolution=1,
                                   density=1, direction=ambient.DEFAULT_DRIFT_DIRECTION,
                                   gravity_radius=0.5)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    return widget


def test_overrunning_real_shader_keeps_clock_pointer_and_clicks_moving(qtbot, monkeypatch):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    backdrop = ambient.AmbientWidget(host, theme="data_art_impulse_lens",
                                     palette="spacr", background="#101418", fps=60,
                                     seed=42, blur=0, speed=1, size=1, resolution=1,
                                     density=1, direction=ambient.DEFAULT_DRIFT_DIRECTION,
                                     gravity_radius=0.5)
    backdrop.setGeometry(host.rect())
    button = QPushButton("Run", host)
    button.setGeometry(120, 80, 80, 40)
    clicked = []
    button.clicked.connect(lambda: clicked.append(True))
    original = backdrop.engine.shade
    snapshots = []
    entered = threading.Event()

    def slow_shade(width, height):
        entered.set()
        time.sleep(0.07)
        snapshots.append((backdrop.engine.time, backdrop.engine.pointer,
                          tuple(backdrop.engine._gravity_impulses)))
        return original(width, height)

    monkeypatch.setattr(backdrop.engine, "shade", slow_shade)
    monkeypatch.setattr(ambient.QApplication, "activeWindow", lambda: host)
    host.show()
    qtbot.waitExposed(host)
    qtbot.waitUntil(entered.is_set)
    target = button.mapToGlobal(QPoint(20, 15))
    QCursor.setPos(target)
    expected = ((140.5 / 360), (95.5 / 240))
    assert backdrop._data_art_pointer_for_tick() == pytest.approx(expected)
    heartbeat = []
    timer = QTimer()
    timer.setInterval(5)
    timer.timeout.connect(lambda: heartbeat.append(time.perf_counter()))
    timer.start()
    try:
        qtbot.mouseClick(button, Qt.LeftButton, pos=QPoint(20, 15))
        qtbot.mouseClick(host, Qt.LeftButton, pos=QPoint(40, 200))
        qtbot.waitUntil(lambda: len(snapshots) >= 7, timeout=4000)
        assert clicked == [True]
        clocks = [clock for clock, _, _ in snapshots[-5:]]
        assert all(later > earlier for earlier, later in zip(clocks, clocks[1:]))
        assert clocks[-1] - clocks[0] >= 0.20
        assert any(pointer == pytest.approx(expected) for _, pointer, _ in snapshots)
        assert any(strength == 1.0 for _, _, impulses in snapshots
                   for _, _, strength in impulses)
        assert len(heartbeat) >= 30
        assert backdrop._art_input._snapshot[1] >= backdrop._art_input._applied_elapsed
    finally:
        timer.stop()
        host.hide()
    assert not backdrop.shading_thread_alive()


def test_cumulative_input_is_applied_once_and_only_latest_pointer_survives():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", seed=42)
    engine.set_gravity_radius(0.5)
    queued = ambient._QueuedArtInput()
    queued._consume(engine)
    assert engine.time == 0
    queued._offer(0.1, (0.2, 0.3), ((0.2, 0.3),))
    queued._offer(0.2, (0.8, 0.6), ((0.8, 0.6),))
    queued._consume(engine)
    assert engine.time == pytest.approx(0.3)
    assert engine.pointer == (0.8, 0.6)
    clicks = [event for event in engine._gravity_impulses if event[2] == 1.0]
    assert [event[1] for event in clicks] == [(0.2, 0.3), (0.8, 0.6)]
    before = tuple(engine._gravity_impulses)
    queued._consume(engine)
    queued._offer(0, None, ())
    queued._consume(engine)
    assert engine.time == pytest.approx(0.3)
    assert tuple(engine._gravity_impulses) == before
    queued._offer(0.1, None, ())
    queued._consume(engine)
    assert engine.time == pytest.approx(0.4)
    assert tuple(engine._gravity_impulses) == before


def test_click_snapshot_is_bounded_and_can_be_discarded_on_pause():
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101418", seed=42)
    engine.set_gravity_radius(0.5)
    queued = ambient._QueuedArtInput()
    for index in range(40):
        queued._offer(0.01, None, ((index / 40, 0.5),))
    assert len(queued._snapshot[3]) == 16
    queued._consume(engine, discard_clicks=True)
    assert engine.time == pytest.approx(0.4)
    assert engine._gravity_impulses == []
    queued._offer(0.1, None, ((0.9, 0.9),))
    queued._consume(engine)
    assert len(engine._gravity_impulses) == 1
    stamp, point, strength = engine._gravity_impulses[0]
    assert stamp == pytest.approx(0.5)
    assert (point, strength) == ((0.9, 0.9), 1.0)


def test_set_time_and_explicit_step_do_not_replay_ticks_from_before_the_change(qtbot):
    widget = _widget(qtbot)
    queued = widget._art_input
    queued._offer(0.4, (0.3, 0.2), ())
    widget.set_time(50.0)
    queued._consume(widget.engine)
    assert widget.time() == 50.0
    queued._offer(0.1, None, ())
    widget.advance_frame(0.2)
    assert widget.time() == pytest.approx(50.3)
    queued._consume(widget.engine)
    assert widget.time() == pytest.approx(50.3)
    queued._offer(0.1, None, ())
    widget.set_speed(2.0)
    assert widget.time() == pytest.approx(50.4)
    queued._offer(0.1, None, ())
    queued._consume(widget.engine)
    assert widget.time() == pytest.approx(50.6)


def test_pause_and_theme_change_reset_queue_without_resuming_hidden_elapsed(qtbot):
    widget = _widget(qtbot)
    queued = widget._art_input
    queued._offer(0.2, (0.2, 0.3), ((0.2, 0.3),))
    widget.stop()
    assert widget.time() == pytest.approx(0.2)
    assert widget._art_input is not queued
    assert widget._art_input._snapshot == (0, 0.0, None, ())
    assert widget.engine._gravity_impulses == []
    qtbot.wait(40)
    assert widget.time() == pytest.approx(0.2)
    widget._art_input._offer(0.1, None, ())
    widget.set_theme("data_art_fungal_growth")
    assert widget.time() == pytest.approx(0.3)
    assert widget._art_input._snapshot == (0, 0.0, None, ())
    widget._art_input._offer(0.1, None, ())
    widget.set_theme("blobs")
    assert widget.time() == pytest.approx(0.4)
    assert widget._art_input is None
    widget.set_theme("data_art_point_atlas")
    assert widget.time() == pytest.approx(0.4)
    assert widget._art_input._snapshot == (0, 0.0, None, ())


def test_fast_gui_tick_retains_immediate_time_and_max_dt_contract(qtbot, monkeypatch):
    widget = _widget(qtbot, "tissue_facets")
    monkeypatch.setattr(widget, "_follow_the_run", lambda: None)
    widget.set_time(2.0)
    widget._clock.start()
    qtbot.wait(20)
    widget._on_tick()
    assert 2.0 < widget.time() <= 2.0 + ambient.MAX_DT
    before = widget.time()
    widget._art_input._consume(widget.engine)
    assert widget.time() == before


def test_pause_does_not_wait_for_an_unavailable_engine_lock(qtbot):
    widget = _widget(qtbot)
    acquired = threading.Event()
    release = threading.Event()

    def hold_engine():
        with widget._engine_lock:
            acquired.set()
            release.wait(3)

    thread = threading.Thread(target=hold_engine)
    thread.start()
    assert acquired.wait(1)
    try:
        widget._art_input._offer(0.2, (0.3, 0.4), ((0.3, 0.4),))
        started = time.perf_counter()
        widget.stop()
        assert time.perf_counter() - started < 0.1
        assert widget._art_input._snapshot == (0, 0.0, None, ())
    finally:
        release.set()
        thread.join(1)
    widget._art_input._offer(0.1, None, ())
    widget._art_input._consume(widget.engine)
    assert widget.time() == pytest.approx(0.1)
    assert widget.engine._gravity_impulses == []


@pytest.mark.parametrize("family", ("point_atlas", "tissue_facets", "fungal_growth"))
def test_queued_clicks_cannot_create_lens_bursts_in_other_materials(family, monkeypatch):
    engine = ambient.make_engine(f"data_art_{family}", "spacr", "#101418", seed=42)
    queued = ambient._QueuedArtInput()
    calls = []
    monkeypatch.setattr(engine, "_add_impulse", lambda *args, **kwargs: calls.append(args),
                        raising=False)
    queued._offer(0.2, None, ((0.3, 0.4),))
    queued._consume(engine)
    assert engine.time == pytest.approx(0.2)
    assert calls == []


def test_legacy_busy_tick_still_carries_time_to_the_next_gui_tick(qtbot, monkeypatch):
    widget = _widget(qtbot)
    widget.set_theme("blobs")
    monkeypatch.setattr(widget, "_follow_the_run", lambda: None)
    acquired = threading.Event()
    release = threading.Event()

    def hold_engine():
        with widget._engine_lock:
            acquired.set()
            release.wait(3)

    thread = threading.Thread(target=hold_engine)
    thread.start()
    assert acquired.wait(1)
    try:
        widget._clock.start()
        qtbot.wait(20)
        widget._on_tick()
        carried = widget._pending_dt
        assert carried > 0
        assert widget.engine.time == 0
    finally:
        release.set()
        thread.join(1)
    widget._on_tick()
    assert carried <= widget.time() <= carried + ambient.MAX_DT
    assert widget._pending_dt == 0
