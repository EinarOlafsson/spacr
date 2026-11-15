"""The ambient backdrop widget, its frame producer, animation policy and field input at their edges.

The widget composes held and fresh frames, the producer shades on its own
thread, the resource policy pauses and resumes every live backdrop, and the
field receives node visits, growing regions and ripples from widgets that may
be hidden, retired or deleted.  Each case asserts the exact published frame,
painted pixels, normalised visit point, wave counter or call order -- a
missing answer costs a frame, never the screen.

One general test per unit, each walking a case table with pytest's
``subtests`` fixture; every case builds its own host and backdrop.
"""
from __future__ import annotations

import threading
import time
import weakref
from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6")

import shiboken6  # noqa: E402
from PySide6.QtCore import QRect, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter, QPaintEvent, QRegion  # noqa: E402
from PySide6.QtWidgets import QFrame, QWidget  # noqa: E402

from spacr.qt import preferences as prefs  # noqa: E402
from spacr.qt.widgets import ambient  # noqa: E402
from spacr.qt.widgets.ambient import AmbientWidget, _FrameProducer  # noqa: E402

pytestmark = pytest.mark.qt

DARK = "#101418"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Private preferences and no shared animation policy."""
    settings = QSettings(str(tmp_path / "ambient-widget.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: settings)
    monkeypatch.setattr(ambient, "_RESOURCE_POLICY", None)
    return settings


def test_the_backdrop_widget_paints_held_and_fresh_regions_from_real_frames(qtbot, monkeypatch, subtests):
    """AmbientWidget._paint_held composes held/fresh frames, and its held-rows probe covers only real paint."""
    def hosted(size, **kwargs):
        host = QWidget()
        qtbot.addWidget(host)
        host.resize(*size)
        backdrop = AmbientWidget(host, theme="blobs", seed=7, **kwargs)
        backdrop.resize(*size)
        return host, backdrop

    with subtests.test(msg="wallpaper_held_region_and_fresh_region"):
        wallpaper = QImage(120, 100, QImage.Format_RGB32)
        wallpaper.fill(QColor("#102030"))
        _host, backdrop = hosted((120, 100), backdrop=wallpaper)
        old, fresh = QImage(120, 100, QImage.Format_RGB32), QImage(120, 100, QImage.Format_RGB32)
        old.fill(QColor("#d02030"))
        fresh.fill(QColor("#1040d0"))
        held = backdrop._held_rows
        held.frame = old
        held.region = QRegion(QRect(24, 24, 48, 48))
        backdrop._last_frame = fresh
        backdrop._timer.start()
        references = []
        for frame in (old, fresh):
            reference = QImage(120, 100, QImage.Format_ARGB32_Premultiplied)
            reference.fill(0)
            renderer = QPainter(reference)
            backdrop._paint_base(renderer, backdrop.rect())
            backdrop._engine.blit(renderer, frame, 120, 100)
            renderer.end()
            references.append(reference)
        expected = references[1].copy()
        renderer = QPainter(expected)
        renderer.setClipRegion(held.region)
        renderer.drawImage(0, 0, references[0])
        renderer.end()
        canvas = QImage(120, 100, QImage.Format_ARGB32_Premultiplied)
        canvas.fill(0)
        painter = QPainter(canvas)
        try:
            assert backdrop._paint_held(painter, QPaintEvent(backdrop.rect()), 120, 100)
        finally:
            painter.end()
            backdrop.stop()
        assert canvas == expected
        assert canvas.pixelColor(40, 40) == references[0].pixelColor(40, 40)
        assert canvas.pixelColor(100, 80) == references[1].pixelColor(100, 80)

    with subtests.test(msg="no_fresh_frame_cannot_claim_a_complete_held_paint"):
        _host, backdrop = hosted((80, 60))
        held = backdrop._held_rows
        held.frame = QImage(80, 60, QImage.Format_RGB32)
        held.frame.fill(QColor("#123456"))
        held.region = QRegion(backdrop.rect())
        assert backdrop._last_frame is None
        backdrop._timer.start()
        canvas = QImage(80, 60, QImage.Format_ARGB32_Premultiplied)
        canvas.fill(0)
        painter = QPainter(canvas)
        try:
            assert not backdrop._paint_held(painter, QPaintEvent(backdrop.rect()), 80, 60)
        finally:
            painter.end()
            backdrop.stop()
        assert canvas.pixelColor(40, 30).alpha() == 0

    with subtests.test(msg="parentless_field_and_zero_sized_background_cover_nothing"):
        backdrop = AmbientWidget(theme="blobs", seed=7)
        qtbot.addWidget(backdrop)
        assert backdrop._held_rows._covered().isEmpty()
        widget = QWidget()
        qtbot.addWidget(widget)
        widget.setAttribute(Qt.WA_StyledBackground, True)
        widget.resize(0, 0)
        assert backdrop._held_rows._paints_a_background(widget) is False
        backdrop.stop()

    with subtests.test(msg="line_separator_with_a_child_is_not_a_hollow_ring"):
        host, backdrop = hosted((80, 60))
        line = QFrame(host)
        line.setFrameShape(QFrame.HLine)
        line.setLineWidth(2)
        line.setAttribute(Qt.WA_NoSystemBackground, True)
        QWidget(line)
        assert line.frameWidth() > 0
        assert backdrop._held_rows._own_paint(line) is None
        backdrop.stop()

    with subtests.test(msg="popup_backdrop_is_not_warmed_with_the_fields"):
        from spacr.qt.screens import make_masks as mm

        screen = QWidget()
        qtbot.addWidget(screen)
        backdrop = AmbientWidget(screen)
        backdrop.setProperty("spacrPopupBackdrop", True)
        with monkeypatch.context() as mp:
            mp.setattr(AmbientWidget, "_art_render_size",
                       lambda self, w, h: pytest.fail("rendered a popup backdrop"))
            mm.MakeMasksScreen._warm_the_fields_of(SimpleNamespace(), screen)
        assert backdrop.property("spacrPopupBackdrop") is True
        assert not backdrop.isVisible()
        backdrop.stop()


def test_the_animation_policy_applies_to_live_backdrops_only(qapp, monkeypatch, subtests):
    """_AnimationResourcePolicy._apply emits once per decision and skips strangers and dead widgets."""
    with subtests.test(msg="non_backdrop_widget_is_left_alone"):
        stranger = QWidget()
        with monkeypatch.context() as mp:
            mp.setattr(ambient, "_LIVE_AMBIENT_WIDGETS", weakref.WeakSet([stranger]))
            policy = ambient._AnimationResourcePolicy()
            seen = []
            policy.changed.connect(lambda: seen.append(True))
            policy._apply()
        assert seen == [True]
        assert not hasattr(stranger, "_sync_run_state")

    with subtests.test(msg="a_widget_whose_object_is_gone_is_skipped"):
        class Dead:
            pass

        dead = Dead()

        def sync():
            raise RuntimeError("Internal C++ object already deleted")

        dead._sync_run_state = sync
        with monkeypatch.context() as mp:
            mp.setattr(ambient, "AmbientWidget", Dead)
            mp.setattr(ambient, "_live_ambient_widgets", lambda: [dead])
            mp.setattr(ambient, "_live_spaceout_fractals", lambda: [])
            policy = ambient._AnimationResourcePolicy()
            emitted = []
            policy.changed.connect(lambda: emitted.append(1))
            policy.cpu_busy = True
            policy._apply()
        assert emitted == [1]

    with subtests.test(msg="applies_without_a_running_application"):
        policy = ambient._AnimationResourcePolicy()
        held, emitted = [], []
        policy.changed.connect(lambda: emitted.append(1))
        with monkeypatch.context() as mp:
            mp.setattr(policy, "_hold_fractals", lambda: held.append("hold"))
            mp.setattr(policy, "_release_fractals", lambda: held.append("release"))
            mp.setattr(ambient.QApplication, "instance", staticmethod(lambda: None))
            policy.gpu_busy = True
            policy._apply()
            policy.gpu_busy = False
            policy._apply()
        assert held == ["hold", "release"] and emitted == [1, 1]


def _lens_field(qtbot, size, node_geometry=None):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(*size)
    backdrop = AmbientWidget(host, theme="data_art_impulse_lens", seed=7)
    backdrop.setGeometry(host.rect())
    node = None
    if node_geometry is not None:
        node = QWidget(host)
        node.setGeometry(*node_geometry)
    host.show()
    qtbot.waitUntil(lambda: backdrop._should_run())
    return host, backdrop, node


def test_field_node_visits_and_growing_regions_reach_the_real_field(qtbot, monkeypatch, subtests):
    """_field_visit_node publishes a normalised visit only when visible, enabled and moving; regions are clipped."""
    with subtests.test(msg="visits_follow_visibility_and_motion_preferences"):
        prefs.set_ambient_enabled(True)
        prefs.set_reduce_motion(False)
        host, backdrop, node = _lens_field(qtbot, (320, 240), (80, 60, 100, 80))
        try:
            serial = backdrop._art_input._node_visit[0]
            ambient._field_visit_node(node)
            assert backdrop._art_input._node_visit == (serial + 1, (129 / 320, 99 / 240))
            qtbot.waitUntil(lambda: backdrop._engine._node_visit is not None)
            ambient._field_visit_node(node, active=False)
            assert backdrop._art_input._node_visit[1] is None
            qtbot.waitUntil(lambda: backdrop._engine._node_visit is None)
            prefs.set_reduce_motion(True)
            ambient._field_visit_node(node)
            assert backdrop._art_input._node_visit[1] is None
            prefs.set_reduce_motion(False)
            node.hide()
            ambient._field_visit_node(node)
            assert backdrop._art_input._node_visit[1] is None
            node.show()
            prefs.set_ambient_enabled(False)
            ambient._field_visit_node(node)
            assert backdrop._art_input._node_visit[1] is None
        finally:
            backdrop.stop()
            host.close()

    with subtests.test(msg="visit_expires_with_engine_time"):
        prefs.set_ambient_enabled(True)
        prefs.set_reduce_motion(False)
        host, backdrop, node = _lens_field(qtbot, (240, 180), (20, 20, 80, 60))
        try:
            ambient._field_visit_node(node)
            with backdrop._engine_lock:
                backdrop._art_input._consume(backdrop._engine)
                engine = backdrop._engine
                assert engine._node_visit is not None
                engine.advance(7.0)
                assert engine._field_center()[2] == 0.0 and engine._node_visit is None
        finally:
            backdrop.stop()
            host.close()

    with subtests.test(msg="visit_waits_while_the_shader_holds_the_engine"):
        with monkeypatch.context() as mp:
            mp.setattr(prefs, "get_ambient_enabled", lambda: True)
            mp.setattr(prefs, "get_reduce_motion", lambda: False)
            host, backdrop, node = _lens_field(qtbot, (320, 240), (80, 60, 100, 80))
            holding, release = threading.Event(), threading.Event()

            def hold():
                with backdrop._engine_lock:
                    holding.set()
                    release.wait(5)

            holder = threading.Thread(target=hold, daemon=True)
            consumed = []
            try:
                holder.start()
                assert holding.wait(5)
                mp.setattr(backdrop._art_input, "_consume",
                           lambda engine, **kw: consumed.append(engine))
                serial = backdrop._art_input._node_visit[0]
                ambient._field_visit_node(node)
                assert backdrop._art_input._node_visit[0] == serial + 1
                assert consumed == []
            finally:
                release.set()
                holder.join(5)
                backdrop.stop()
            ambient._field_visit_node(node)
            assert consumed == [backdrop._engine]
            host.close()

    with subtests.test(msg="growing_regions_clip_with_holding_disabled"):
        prefs.set_ambient_enabled(True)
        prefs.set_reduce_motion(False)
        host, backdrop, _node = _lens_field(qtbot, (240, 180))
        try:
            backdrop._held_rows.enabled = False
            inside = QRect(8, 8, 20, 30)
            outside = QRect(-100, -100, 10, 10)
            ambient._field_growing_regions_for_widget(host, (outside, inside), invalidate=True)
            width, height = backdrop.width(), backdrop.height()
            assert backdrop._art_input._growing_regions == (
                (8 / width, 8 / height, 28 / width, 38 / height),)
            assert backdrop._held_rows.frame is None
        finally:
            backdrop.stop()
            host.close()

    with subtests.test(msg="a_classic_backdrop_takes_no_field_input"):
        host = QWidget()
        qtbot.addWidget(host)
        host.resize(240, 180)
        classic = AmbientWidget(host, theme="blobs", seed=7)
        node = QWidget(host)
        ambient._field_visit_node(node)
        ambient._field_growing_regions_for_widget(node, (QRect(-10, -10, 4, 4),))
        assert classic._art_input is None
        classic.stop()


def test_field_ripples_skip_deleted_or_retiring_widgets(qtbot, monkeypatch, subtests):
    """field_ripple_for_widget/_field_ripple_for_popup drop feedback for deleted or retiring owners."""
    with subtests.test(msg="deleted_widget_counts_a_container_ripple_and_returns_none"):
        gone = QWidget()
        shiboken6.delete(gone)
        assert not shiboken6.isValid(gone)
        before = ambient._CONTAINER_RIPPLES[0]
        assert ambient.field_ripple_for_widget(gone, edge="left") is None
        assert ambient._CONTAINER_RIPPLES[0] == before + 1

    with subtests.test(msg="deleted_popup_beside_a_classic_backdrop"):
        host = QWidget()
        qtbot.addWidget(host)
        classic = AmbientWidget(host, theme="blobs", seed=7)
        popup = QWidget(host)
        shiboken6.delete(popup)
        assert ambient._field_ripple_for_popup(popup) is None
        assert ambient.field_ripple_for_widget(popup) is None
        classic.stop()

    for route in ("popup", "widget"):
        with subtests.test(msg=f"owner_retiring_during_{route}_lookup"):
            owner = QWidget()
            qtbot.addWidget(owner)
            widget = QWidget(owner)
            qtbot.addWidget(widget)
            attribute = "parentWidget" if route == "popup" else "window"
            lookup = getattr(widget, attribute)

            def retired():
                previous = lookup()
                widget.setParent(None)
                shiboken6.delete(owner)
                assert not shiboken6.isValid(previous)
                return previous

            with monkeypatch.context() as mp:
                mp.setattr(widget, attribute, retired)
                if route == "popup":
                    ambient._field_ripple_for_popup(widget)
                else:
                    ambient.field_ripple_for_widget(widget)
            assert shiboken6.isValid(widget) and not shiboken6.isValid(owner)


class _Blank:
    """An engine with no frame to give."""

    def __init__(self):
        self.calls = []

    def shade(self, width, height):
        self.calls.append((width, height))
        return None


def _run_producer(engine, size, until):
    producer = _FrameProducer(engine, threading.Lock(), 240, size)
    producer.start()
    try:
        deadline = time.monotonic() + 5
        while not until() and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        producer.stop()
    assert not producer.is_alive()
    return producer


def test_the_frame_producer_publishes_only_real_frames(qtbot, subtests):
    """_FrameProducer never shades an empty canvas and never publishes a refused frame."""
    with subtests.test(msg="no_canvas_never_shades"):
        engine = _Blank()
        started = time.monotonic()
        producer = _run_producer(engine, (0, 120), lambda: time.monotonic() - started > 0.05)
        assert engine.calls == []
        assert producer.frames_shaded == 0 and producer.latest() is None

    with subtests.test(msg="refused_frame_is_not_published"):
        engine = _Blank()
        producer = _run_producer(engine, (40, 30), lambda: len(engine.calls) >= 3)
        assert len(engine.calls) >= 3 and set(engine.calls) == {(40, 30)}
        assert producer.frames_shaded == 0 and producer.latest() is None

    with subtests.test(msg="republish_with_no_canvas_publishes_nothing"):
        widget = AmbientWidget(theme="blobs", palette="spacr", background=DARK, seed=3)
        qtbot.addWidget(widget)
        widget.resize(240, 160)
        producer = _FrameProducer(widget.engine, widget._engine_lock, 30, (0, 0))
        widget._producer_box[0] = producer
        widget.set_time(3.0)
        assert producer.latest() is None
        producer.size = (240, 160)
        widget.set_time(4.0)
        frame = producer.latest()
        assert (frame.width(), frame.height()) == widget.engine.buffer_size(240, 160)
        widget.stop()


def test_setting_the_graphics_backend_restarts_only_on_a_real_change(monkeypatch, subtests):
    """AmbientWidget._set_graphics_backend keeps ready frames when unchanged and restarts the producer otherwise."""
    def probe(mp, requested, engine_backend):
        engine = object.__new__(ambient.DriftEngine)
        engine._graphics_backend = engine_backend
        ready = object()
        changes = []
        widget = SimpleNamespace(
            _graphics_backend=requested, _engine=engine, _engine_lock=threading.Lock(),
            _timer=SimpleNamespace(isActive=lambda: True),
            _producer_box=[ready], _partial_frame=ready,
            _last_frame=ready, _last_composed=ready,
            _start_producer=lambda: changes.append("start"),
            update=lambda: changes.append("update"))

        def retire(box):
            changes.append("retire")
            box[0] = None

        mp.setattr(ambient, "_retire_producer", retire)
        return widget, ready, changes

    for backend, choice in [("gpu", "GPU"), ("cpu", "unknown")]:
        with subtests.test(msg=f"unchanged_{backend}_via_{choice}"):
            with monkeypatch.context() as mp:
                widget, ready, changes = probe(mp, backend, backend)
                AmbientWidget._set_graphics_backend(widget, choice)
            assert changes == []
            assert widget._producer_box[0] is ready
            assert (widget._partial_frame, widget._last_frame, widget._last_composed) == (ready,) * 3
            assert widget._graphics_backend == widget._engine._graphics_backend == backend

    for case_id, requested, engine_backend, choice, expected in [
            ("changed_cpu_to_gpu", "cpu", "cpu", "GPU", "gpu"),
            ("mismatched_engine_repaired", "cpu", "gpu", "CPU", "cpu")]:
        with subtests.test(msg=case_id):
            with monkeypatch.context() as mp:
                widget, _ready, changes = probe(mp, requested, engine_backend)
                AmbientWidget._set_graphics_backend(widget, choice)
            assert widget._graphics_backend == widget._engine._graphics_backend == expected
            assert changes == ["retire", "start", "update"]
            assert widget._producer_box[0] is None
            assert (widget._partial_frame, widget._last_frame, widget._last_composed) == (None,) * 3


def _partial_field(qtbot, mp):
    """A blobs field holding one published frame, with no shading thread running."""
    import numpy as np

    widget = AmbientWidget(theme="blobs", background="#152839", seed=2)
    qtbot.addWidget(widget)
    widget.resize(97, 73)
    widget._timer.stop()
    widget._card_tints = None
    data = np.random.default_rng(4).integers(0, 256, (31, 43, 4), dtype=np.uint8)
    frame = QImage(data.data, 43, 31, data.strides[0], QImage.Format_RGBA8888)
    widget._last_frame = frame.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    producer = _FrameProducer(widget._engine, widget._engine_lock, 30, (97, 73))
    producer.publish(widget._last_frame)
    producer.box = widget._producer_box
    widget._producer_box[0] = producer
    mp.setattr(widget, "_sync_run_state", lambda: None)
    return widget


def _partial_paint(widget, region=QRect(3, 5, 53, 41)):
    """Ask for one partial paint onto a fresh image clipped to ``region``."""
    target = QImage(97, 73, QImage.Format_ARGB32_Premultiplied)
    target.fill(QColor("#7a3e59"))
    painter = QPainter(target)
    painter.setClipRegion(QRegion(region))
    try:
        drawn = widget._paint_partial_frame(painter, QPaintEvent(QRegion(region)), 97, 73)
    finally:
        painter.end()
    return drawn, target


def test_partial_frame_paints_reuse_one_exact_raster(qtbot, monkeypatch, subtests):
    """_paint_partial_frame keys the first paint, draws the held raster after, and caches False on failure."""
    with subtests.test(msg="first_paint_keys_second_draws_the_held_raster"):
        with monkeypatch.context() as mp:
            widget = _partial_field(qtbot, mp)
            drawn, _ = _partial_paint(widget)
            assert drawn is False and widget._partial_frame[1] is None
            drawn, first = _partial_paint(widget)
            assert drawn is True and isinstance(widget._partial_frame[1], QImage)
            held = widget._partial_frame[1]
            drawn, second = _partial_paint(widget)
            assert drawn is True and widget._partial_frame[1] is held
            assert first == second
            assert first.pixelColor(20, 20) != QColor("#7a3e59")
            widget.stop()

    original_image, original_painter = ambient.QImage, ambient.QPainter

    class _Factory(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original_image)

    class _NullImage(metaclass=_Factory):
        Format_RGB32 = original_image.Format_RGB32
        Format_ARGB32 = original_image.Format_ARGB32
        Format_ARGB32_Premultiplied = original_image.Format_ARGB32_Premultiplied

        def __new__(cls, *args):
            return original_image()

    def inactive_painter(*args):
        return original_painter()

    def raising_blit(*args, **kwargs):
        raise RuntimeError("frame released mid-blit")

    for case_id, attribute, replacement in [("null_raster", "QImage", _NullImage),
                                            ("inactive_surface", "QPainter", inactive_painter),
                                            ("raising_blit", "blit", raising_blit)]:
        with subtests.test(msg=f"{case_id}_caches_false"):
            with monkeypatch.context() as mp:
                widget = _partial_field(qtbot, mp)
                assert _partial_paint(widget)[0] is False
                with monkeypatch.context() as failing:
                    if attribute == "blit":
                        failing.setattr(widget._engine, "blit", replacement)
                    else:
                        failing.setattr(ambient, attribute, replacement)
                    assert _partial_paint(widget)[0] is False
                assert widget._partial_frame[1] is False
                drawn, target = _partial_paint(widget)
                assert drawn is False and widget._partial_frame[1] is False
                assert target.pixelColor(20, 20) == QColor("#7a3e59")
                widget.stop()

    with subtests.test(msg="paint_event_returns_after_a_reused_partial_frame"):
        with monkeypatch.context() as mp:
            widget = _partial_field(qtbot, mp)
            clip = QRegion(QRect(3, 5, 53, 41))
            target = QImage(97, 73, QImage.Format_ARGB32_Premultiplied)
            calls = []

            class PaintToImage(original_painter):
                def __init__(self, device=None):
                    if device is None:
                        super().__init__()
                        return
                    super().__init__(target if device is widget else device)
                    if device is widget:
                        self.setClipRegion(clip)

            mp.setattr(ambient, "QPainter", PaintToImage)
            real_base = widget._paint_base
            mp.setattr(widget, "_paint_base", lambda *a: calls.append("base") or real_base(*a))
            widget._paint_ambient(QPaintEvent(clip))
            assert calls == ["base"] and widget._partial_frame[1] is None
            calls.clear()
            widget._paint_ambient(QPaintEvent(clip))
            assert calls == ["base"] and isinstance(widget._partial_frame[1], QImage)
            calls.clear()
            widget._paint_ambient(QPaintEvent(clip))
            assert calls == []
            widget.stop()
