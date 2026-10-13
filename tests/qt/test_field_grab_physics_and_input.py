"""A field handle changes only the backdrop and returns through a bounded spring."""

import math
import threading
import time

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QImage, QMouseEvent
from PySide6.QtWidgets import QApplication, QDialog, QLineEdit, QPushButton, QScrollArea, QWidget

from spacr.qt.widgets import ambient


def _engine(**kwargs):
    return ambient.make_engine("data_art_impulse_lens", "spacr", "#101418",
                               seed=42, blur=0, **kwargs)


def _frame(engine):
    return bytes(engine.shade(640, 360).constBits())


def _step(engine, elapsed):
    engine.advance(elapsed)
    engine._step_field_grab()


def test_constant_target_and_release_are_independent_of_step_partition():
    single, split = _engine(), _engine()
    for engine in (single, split):
        engine._set_field_grab(((.5, .5), (.12, .04)))
    _step(single, .3)
    for _ in range(3):
        _step(split, .1)
    assert single._field_grab_offset == pytest.approx(split._field_grab_offset, abs=1e-12)
    assert single._field_grab_velocity == pytest.approx(split._field_grab_velocity, abs=1e-12)
    before = single._field_grab_offset, single._field_grab_velocity
    for engine in (single, split):
        engine._set_field_grab(None)
    assert before == (single._field_grab_offset, single._field_grab_velocity)
    _step(single, .6)
    for _ in range(6):
        _step(split, .1)
    assert single._field_grab_offset == pytest.approx(split._field_grab_offset, abs=1e-12)
    assert single._field_grab_velocity == pytest.approx(split._field_grab_velocity, abs=1e-12)
    assert math.hypot(*single._field_grab_offset) < math.hypot(*before[0])


def test_release_and_regrab_are_continuous_and_restore_the_same_seeded_field():
    engine = _engine()
    engine._set_field_grab(((.5, .5), (.15, .05)))
    _step(engine, .3)
    held = _frame(engine)
    lattice = next(iter(engine._material_cache.values()))
    engine._set_field_grab(None)
    assert _frame(engine) == held
    state = engine._field_grab_center, engine._field_grab_offset, engine._field_grab_velocity
    engine._set_field_grab(((.8, .2), (.01, 0)))
    assert state == (engine._field_grab_center, engine._field_grab_offset,
                     engine._field_grab_velocity)
    assert _frame(engine) == held
    engine._set_field_grab(None)
    _step(engine, 4)
    assert engine._field_grab_center is None
    assert engine._field_grab_offset == engine._field_grab_velocity == (0, 0)
    reference = _engine()
    reference.set_time(engine.time)
    assert _frame(engine) == _frame(reference)
    assert next(iter(engine._material_cache.values())) is lattice


def test_handle_is_bounded_under_large_targets_direction_changes_and_clock_jumps():
    engine = _engine()
    for target in ((10, 10), (-10, 10), (-10, -10), (10, -10)) * 3:
        engine._set_field_grab(((.5, .5), target))
        assert math.hypot(*engine._field_grab_target) <= .180000001
        _step(engine, .1)
        assert math.hypot(*engine._field_grab_offset) <= .180000001
    before = engine._field_grab_offset
    engine.set_time(-1)
    engine._step_field_grab()
    assert engine._field_grab_offset == before
    engine.set_time(1000)
    engine._step_field_grab()
    assert all(math.isfinite(value) for value in engine._field_grab_offset)
    assert math.hypot(*engine._field_grab_offset) <= .180000001
    engine._set_field_grab(None, reset=True)
    assert engine._field_grab_origin is None
    assert not engine._field_grab_held


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_handle_does_not_enable_field_displacement(invalid):
    engine = _engine()
    engine._set_field_grab(((invalid, .5), (.1, 0)))
    assert not engine._field_grab_held
    assert engine._field_grab_target == (0, 0)
    engine._set_field_grab(((.5, .5), (0, invalid)))
    assert not engine._field_grab_held


def test_grab_has_finite_material_reach_at_zero_hover_gravity(monkeypatch):
    engine = _engine()
    captured = []

    def material(width, height, x, y, light, spread=False):
        captured.append((x.copy(), y.copy(), light.copy()))
        image = QImage(width, height, QImage.Format_RGB32)
        image.fill(engine.identity)
        return image

    monkeypatch.setattr(engine, "_point_material", material)
    engine.set_time(.3)
    engine.shade(640, 360)
    idle = captured[-1]
    engine._set_field_grab(((.5, .5), (.1, 0)))
    _step(engine, .3)
    engine.shade(640, 360)
    moved = captured[-1]
    reference = _engine()
    monkeypatch.setattr(reference, "_point_material", material)
    reference.set_time(engine.time)
    reference.shade(640, 360)
    baseline = captured[-1]
    assert engine.gravity_radius == 0
    lattice = next(iter(engine._material_cache.values()))
    distance = np.hypot((lattice[0] - .5) * 640 / 360, lattice[1] - .5)
    outside = distance >= .34
    assert np.array_equal(moved[0][outside], baseline[0][outside])
    assert np.array_equal(moved[1], baseline[1])
    assert np.any(moved[0] != baseline[0])
    assert np.max(np.abs(moved[0] - baseline[0])) <= .18 * 360
    assert np.array_equal(moved[2], baseline[2])
    assert len(idle[0]) == len(moved[0])


def test_worker_input_is_latest_immutable_bounded_and_never_replays_a_handle():
    engine = _engine()
    queued = ambient._QueuedArtInput()
    for index in range(100):
        queued._offer(.01, None, (), grab=((.4, .5), (index / 1000, .02)))
    assert queued._snapshot[3] == ()
    assert queued._grab_snapshot == (100, ((.4, .5), (.099, .02)))
    queued._consume(engine)
    assert engine.time == pytest.approx(1)
    assert engine._field_grab_target == (.099, .02)
    engine.shade(640, 360)
    state = engine._field_grab_offset, engine._field_grab_velocity
    queued._consume(engine)
    assert state == (engine._field_grab_offset, engine._field_grab_velocity)
    queued._offer(0, None, (), grab=None)
    queued._consume(engine)
    assert not engine._field_grab_held
    assert state == (engine._field_grab_offset, engine._field_grab_velocity)
    queued._consume(engine, discard_clicks=True)
    assert engine._field_grab_center is None
    assert engine._field_grab_offset == (0, 0)


@pytest.fixture
def field(qtbot, monkeypatch):
    class Screen(QWidget):
        pass

    host = Screen()
    qtbot.addWidget(host)
    host.resize(360, 240)
    host.setMouseTracking(True)
    monkeypatch.setattr(ambient, "_ready_packed_scatter", lambda: None)
    backdrop = ambient.install_ambient(host, theme="data_art_impulse_lens",
                                       palette="spacr", seed=42, gravity_radius=0)
    host.show()
    qtbot.waitExposed(host)
    backdrop._timer.setInterval(100_000)
    yield host, backdrop
    host.close()
    assert not backdrop.shading_thread_alive()


def _move_held(widget, position, *, buttons=Qt.LeftButton):
    point = QPointF(position)
    event = QMouseEvent(QEvent.MouseMove, point,
                        QPointF(widget.mapToGlobal(position)), Qt.NoButton,
                        buttons, Qt.NoModifier)
    QApplication.sendEvent(widget, event)


def test_actual_background_press_drag_release_drives_field_without_hover_gravity(
        field, qtbot):
    host, backdrop = field
    assert backdrop._interaction_app is not None
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    assert backdrop._field_grab is not None
    _move_held(host, QPoint(210, 120))
    assert math.hypot(*backdrop._field_grab[1]) <= .180000001
    backdrop.advance_frame(.3)
    assert backdrop.engine._field_grab_held
    assert math.hypot(*backdrop.engine._field_grab_offset) > .01
    before = backdrop.engine._field_grab_offset
    qtbot.mouseRelease(host, Qt.LeftButton, pos=QPoint(210, 120))
    assert backdrop._field_grab is None
    assert not backdrop.engine._field_grab_held
    assert backdrop.engine._field_grab_offset == before
    backdrop.advance_frame(4)
    assert backdrop.engine._field_grab_center is None


def test_buttons_inputs_and_scientific_canvas_still_receive_their_original_events(
        field, qtbot):
    host, backdrop = field
    button = QPushButton("Run", host)
    button.setGeometry(10, 10, 70, 30)
    clicked = []
    button.clicked.connect(lambda: clicked.append(True))
    button.show()
    qtbot.mouseClick(button, Qt.LeftButton)
    assert clicked == [True]
    assert backdrop._field_grab is None
    editor = QLineEdit(host)
    editor.setGeometry(90, 10, 100, 30)
    editor.show()
    qtbot.mouseClick(editor, Qt.LeftButton)
    qtbot.keyClicks(editor, "scientific input")
    assert editor.text() == "scientific input"
    assert backdrop._field_grab is None

    class ScientificCanvas(QWidget):
        def mousePressEvent(self, event):
            self.received.append("press")
            event.accept()

        def mouseMoveEvent(self, event):
            self.received.append("move")
            event.accept()

        def mouseReleaseEvent(self, event):
            self.received.append("release")
            event.accept()

    canvas = ScientificCanvas(host)
    canvas.received = []
    canvas.setGeometry(10, 60, 200, 150)
    canvas.show()
    qtbot.mousePress(canvas, Qt.LeftButton, pos=QPoint(10, 10))
    _move_held(canvas, QPoint(40, 40))
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=QPoint(40, 40))
    assert canvas.received == ["press", "move", "release"]
    assert backdrop._field_grab is None
    assert not backdrop.engine._field_grab_held
    assert not backdrop.engine._gravity_impulses


def test_scroll_viewports_dialogs_and_right_clicks_do_not_begin_grabbing(field, qtbot):
    host, backdrop = field
    area = QScrollArea(host)
    area.setGeometry(10, 10, 180, 100)
    area.show()
    qtbot.mouseClick(area.viewport(), Qt.LeftButton)
    assert backdrop._field_grab is None
    popup = QDialog(host)
    qtbot.addWidget(popup)
    popup.show()
    qtbot.waitExposed(popup)
    qtbot.mouseClick(popup, Qt.LeftButton)
    assert backdrop._field_grab is None
    popup.close()
    qtbot.mouseClick(host, Qt.RightButton, pos=QPoint(220, 170))
    assert backdrop._field_grab is None


def test_grab_offers_do_not_block_a_busy_worker_and_hide_discards_stale_handle(
        field, qtbot):
    host, backdrop = field
    acquired, release = threading.Event(), threading.Event()

    def hold():
        with backdrop._engine_lock:
            acquired.set()
            release.wait(3)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(1)
    try:
        started = time.perf_counter()
        qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
        _move_held(host, QPoint(200, 110))
        assert time.perf_counter() - started < .15
        snapshot = backdrop._art_input._grab_snapshot
        assert snapshot[1] == backdrop._field_grab
        qtbot.mouseRelease(host, Qt.LeftButton, pos=QPoint(200, 110))
        assert backdrop._art_input._grab_snapshot[1] is None
    finally:
        release.set()
        thread.join(1)
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    _move_held(host, QPoint(200, 110))
    backdrop.advance_frame(.3)
    host.hide()
    assert backdrop._field_grab is None
    assert backdrop._interaction_app is None
    assert backdrop.engine._field_grab_offset == (0, 0)
    host.show()
    qtbot.waitExposed(host)
    assert not backdrop.engine._field_grab_held
    backdrop.set_theme("data_art_point_atlas")
    assert backdrop._field_grab is None
    assert backdrop._interaction_app is None


def test_real_mask_canvas_ignored_and_box_tool_input_never_grabs_background(field, qtbot):
    from spacr.qt.screens.make_masks import MODE_BOX, _MaskCanvas

    host, backdrop = field
    canvas = _MaskCanvas(host)
    canvas.setGeometry(10, 10, 200, 180)
    canvas.show()
    qtbot.mouseClick(canvas, Qt.LeftButton, pos=QPoint(30, 30))
    assert backdrop._field_grab is None
    canvas.set_image_and_mask(np.arange(10000, dtype=np.uint16).reshape(100, 100),
                              np.zeros((100, 100), dtype=np.uint16))
    canvas.mode = MODE_BOX
    start = canvas._image_to_canvas(25, 25)
    end = canvas._image_to_canvas(75, 75)
    assert start is not None and end is not None
    qtbot.mousePress(canvas, Qt.LeftButton, pos=start)
    _move_held(canvas, end)
    qtbot.mouseRelease(canvas, Qt.LeftButton, pos=end)
    assert len(canvas.boxes) == 1
    assert canvas.boxes[0][1] < canvas.boxes[0][3]
    assert backdrop._field_grab is None
    assert not backdrop.engine._field_grab_held


def test_custom_screen_plain_container_dpr_and_resize_cancel_are_consistent(
        field, qtbot, monkeypatch):
    host, backdrop = field
    container = QWidget(host)
    container.setGeometry(100, 50, 220, 170)
    container.show()
    targets = []
    for ratio in (1, 2):
        monkeypatch.setattr(backdrop, "devicePixelRatioF", lambda ratio=ratio: ratio)
        assert backdrop._art_render_size(360, 240) == (360 * ratio, 240 * ratio)
        qtbot.mousePress(container, Qt.LeftButton, pos=QPoint(40, 40))
        _move_held(container, QPoint(65, 50))
        targets.append(backdrop._field_grab[1])
        qtbot.mouseRelease(container, Qt.LeftButton, pos=QPoint(65, 50))
    assert targets[0] == targets[1]
    qtbot.mousePress(container, Qt.LeftButton, pos=QPoint(40, 40))
    _move_held(container, QPoint(65, 50))
    backdrop.advance_frame(.3)
    assert backdrop.engine._field_grab_offset != (0, 0)
    host.resize(400, 250)
    assert backdrop.size() == host.size()
    assert backdrop._field_grab is None
    qtbot.waitUntil(lambda: not backdrop.engine._field_grab_held, timeout=1000)
    backdrop.advance_frame(4)
    assert backdrop.engine._field_grab_center is None


def test_leave_retains_handle_until_release_and_deactivation_cancels(field, qtbot):
    host, backdrop = field
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    _move_held(host, QPoint(210, 120))
    QApplication.sendEvent(host, QEvent(QEvent.Leave))
    assert backdrop._field_grab is not None
    qtbot.mouseRelease(host, Qt.LeftButton, pos=QPoint(390, 270))
    assert backdrop._field_grab is None
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    _move_held(host, QPoint(210, 120))
    _move_held(host, QPoint(220, 130), buttons=Qt.NoButton)
    assert backdrop._field_grab is None
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    _move_held(host, QPoint(210, 120))
    QApplication.sendEvent(host, QEvent(QEvent.WindowDeactivate))
    assert backdrop._field_grab is None
    assert not backdrop.engine._field_grab_held


def test_other_theme_and_detached_background_cannot_keep_a_field_handle(field, qtbot):
    host, backdrop = field
    other = ambient.make_engine("data_art_point_atlas", "spacr", "#101418", seed=42)
    other._set_field_grab(((.5, .5), (.1, .1)))
    assert not other._field_grab_held
    assert other._field_grab_offset == (0, 0)
    detached = QWidget()
    qtbot.addWidget(detached)
    assert not backdrop._field_grab_background(detached)
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    assert backdrop._field_grab is not None
    backdrop.set_theme("drift")
    assert backdrop._art_input is None
    assert backdrop._field_grab is None
    backdrop._offer_field_grab()
    assert backdrop._field_grab is None


def test_hover_gravity_changes_preserve_an_explicit_held_material_patch(field, qtbot):
    host, backdrop = field
    backdrop.set_gravity_radius(.5)
    qtbot.mousePress(host, Qt.LeftButton, pos=QPoint(160, 100))
    _move_held(host, QPoint(210, 120))
    backdrop.advance_frame(.3)
    with backdrop._engine_lock:
        state = backdrop.engine._field_grab_offset, backdrop.engine._field_grab_velocity
    backdrop.set_gravity_radius(0)
    with backdrop._engine_lock:
        assert backdrop.engine._field_grab_held
        assert state == (backdrop.engine._field_grab_offset,
                         backdrop.engine._field_grab_velocity)
        assert not backdrop.engine._gravity_impulses
    assert backdrop._field_grab is not None
    qtbot.mouseRelease(host, Qt.LeftButton, pos=QPoint(210, 120))
    assert backdrop._field_grab is None
