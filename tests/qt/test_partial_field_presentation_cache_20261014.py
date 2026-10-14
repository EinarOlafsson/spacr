"""Repeated partial paints reuse the held frame without changing its pixels.

The original preview performance contract stays in its owning timers file.
These tests exercise the actual field painter on Qt raster devices, including
clipping, device ratios, new frames, ineligible surfaces and cache retirement.
No shading thread or graphics context is started by this fixture.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import types
import weakref

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QEvent, QPoint, QRect
from PySide6.QtGui import QColor, QHideEvent, QImage, QPaintEvent, QPainter, QPixmap, QRegion

from spacr.qt.widgets import ambient

pytestmark = pytest.mark.qt


def _frame(seed=4):
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (31, 43, 4), dtype=np.uint8)
    image = QImage(data.data, 43, 31, data.strides[0], QImage.Format_RGBA8888)
    return image.convertToFormat(QImage.Format_ARGB32_Premultiplied)


@pytest.fixture
def field(qtbot, monkeypatch):
    widget = ambient.AmbientWidget(theme="blobs", background="#152839", seed=2)
    qtbot.addWidget(widget)
    widget.resize(97, 73)
    widget._timer.stop()
    widget._card_tints = None
    widget._last_frame = _frame()
    producer = ambient._FrameProducer(widget._engine, widget._engine_lock, 30, (97, 73))
    producer.publish(widget._last_frame)
    producer.box = widget._producer_box
    widget._producer_box[0] = producer
    monkeypatch.setattr(widget, "_sync_run_state", lambda: None)
    original_painter = ambient.QPainter

    def render(*, ratio=1., origin=(0., 0.), region=None,
               format=QImage.Format_ARGB32_Premultiplied, reference=False,
               cache_active=True):
        clip = region if region is not None else QRegion(QRect(3, 5, 53, 41))
        size = (math.ceil((widget.width() + 30) * ratio),
                math.ceil((widget.height() + 30) * ratio))
        target = QImage(*size, format)
        target.setDevicePixelRatio(ratio)
        target.fill(QColor("#7a3e59"))

        class PaintToImage(original_painter):
            def __init__(self, device):
                if device is not widget and not cache_active:
                    super().__init__()
                else:
                    super().__init__(target if device is widget else device)
                if device is widget:
                    self.translate(*origin)
                    self.setClipRegion(clip)

        with monkeypatch.context() as patch:
            patch.setattr(ambient, "QPainter", PaintToImage)
            if reference:
                patch.setattr(widget, "_paint_partial_frame", lambda *args: False)
            widget._paint_ambient(QPaintEvent(clip))
        return target

    widget._test_render = render
    yield widget
    widget.stop()


@pytest.mark.parametrize("ratio", [1., 1.25, 1.5, 2.])
@pytest.mark.parametrize("origin", [(0., 0.), (7., 9.)])
@pytest.mark.parametrize("mode,background", [
    (QPainter.CompositionMode_Screen, "#152839"),
    (QPainter.CompositionMode_Multiply, "#e9f1f7"),
])
def test_partial_raster_pixels_match_the_original_painter(field, ratio, origin, mode, background):
    field._engine.mode = mode
    field._background = QColor(background)
    region = QRegion(QRect(3, 5, 17, 19)).united(QRect(43, 27, 31, 23))
    want = field._test_render(ratio=ratio, origin=origin, region=region, reference=True)
    first = field._test_render(ratio=ratio, origin=origin, region=region)
    second = field._test_render(ratio=ratio, origin=origin, region=region)
    repeated = field._test_render(ratio=ratio, origin=origin, region=region)
    assert first == want
    assert second == want
    assert repeated == want


@pytest.mark.parametrize("ratio", [1., 1.25, 1.5, 2.])
@pytest.mark.parametrize("origin", [(7, 9), (13, 17)])
def test_actual_widget_render_redirection_preserves_clipped_pixels(field, monkeypatch, ratio, origin):
    """Exercise Qt's QWidget paint event and real redirected paint engine.

    These are offscreen widget captures, not physical-monitor/HDR acceptance.
    The direct-image controls above separately exercise the algorithm path.
    """
    region = QRegion(QRect(3, 5, 17, 19)).united(QRect(43, 27, 31, 23))

    def capture(reference=False):
        target = QImage(math.ceil(150 * ratio), math.ceil(130 * ratio),
                        QImage.Format_ARGB32_Premultiplied)
        target.setDevicePixelRatio(ratio)
        target.fill(QColor("#7a3e59"))
        painter = QPainter(target)
        try:
            with monkeypatch.context() as patch:
                if reference:
                    patch.setattr(field, "_paint_partial_frame", lambda *args: False)
                field.render(painter, QPoint(*origin), region)
        finally:
            painter.end()
        return target

    want = capture(reference=True)
    assert capture() == want
    assert capture() == want
    assert capture() == want


def test_first_partial_is_original_second_builds_once_and_third_reuses(field, monkeypatch):
    calls = []
    blit = field._engine.blit
    monkeypatch.setattr(field._engine, "blit", lambda *args: (calls.append(1), blit(*args))[-1])
    key = field._last_frame.cacheKey()
    field._test_render()
    assert len(calls) == 1
    assert field._partial_frame[1] is None
    field._test_render()
    image = field._partial_frame[1]
    assert isinstance(image, QImage)
    assert len(calls) == 2
    field._test_render()
    assert len(calls) == 2
    assert field._partial_frame[1] is image
    assert field._partial_frame[2] is field._last_frame
    assert field._last_frame.cacheKey() == key
    assert image.sizeInBytes() <= 48 << 20


def test_regions_with_the_same_bounds_do_not_share_a_presentation(field):
    first = QRegion(QRect(3, 5, 53, 41))
    changed = QRegion(QRect(3, 5, 20, 41)).united(QRect(40, 5, 16, 41))
    assert first.boundingRect() == changed.boundingRect()
    assert first != changed
    field._test_render(region=first)
    field._test_render(region=first)
    old = field._partial_frame[1]
    want = field._test_render(region=changed, reference=True)
    assert field._test_render(region=changed) == want
    assert field._partial_frame[1] is None
    assert field._test_render(region=changed) == want
    assert field._partial_frame[1] is not old


@pytest.mark.parametrize("ratio", [1.25, 1.5])
def test_fractional_ratios_keep_the_original_path_without_an_allocation(field, ratio):
    want = field._test_render(ratio=ratio, reference=True)
    for _ in range(3):
        assert field._test_render(ratio=ratio) == want
        assert field._partial_frame is None


def test_the_real_mask_backdrop_uses_the_compatible_native_cache(qtbot, qapp, monkeypatch):
    """Observe the actual partial QWidget paint in the original Mask layout."""
    from tests.qt.test_module_screen_timers_do_not_stall_20261012 import (
        _hide_tours, _stop_the_field, _window_on_mask,
    )
    from spacr.qt.preferences import apply_preferences_to_app
    from spacr.qt.widgets.live_preview import LivePreviewPanel, numpy_to_qpixmap
    from spacr.qt.widgets.field_fade import invalidate_field_fade, uninstall_field_fade

    sheet, palette, font = qapp.styleSheet(), qapp.palette(), qapp.font()
    apply_preferences_to_app(qapp)
    try:
        window, screen = _window_on_mask(qtbot, (2560, 1440))
        screen._on_preview_switch(True)
        qtbot.wait(3000)
        _hide_tours(window)
        backdrop = _stop_the_field(window)
        panel = next(p for p in screen.findChildren(LivePreviewPanel) if p.isVisibleTo(window))
        views = (panel._src_view, panel._mask_view)
        data = (np.random.default_rng(0).random((2048, 2048, 3)) * 255).astype(np.uint8)
        for view in views:
            view.set_pixmap(numpy_to_qpixmap(data))
        qtbot.wait(50)
        region = QRegion()
        for view in views:
            corner = backdrop.mapFrom(window, view.viewport().mapTo(window, QPoint(0, 0)))
            region = region.united(QRect(corner, view.viewport().size()))
        observations = []
        original = backdrop._paint_partial_frame
        blit = backdrop._engine.blit
        blits = []
        parity = []
        compare_pixels = [False]
        monkeypatch.setattr(backdrop._engine, "blit", lambda *args: (blits.append(1), blit(*args))[-1])

        def describe(region):
            return dict(bounds=list(region.boundingRect().getRect()), rectangles=region.rectCount())

        def observe(painter, event, width, height):
            engine = painter.paintEngine()
            target = None if engine is None else engine.paintDevice()
            transform = painter.deviceTransform()
            system = QRegion() if engine is None else engine.systemClip()
            row = dict(event_region=describe(event.region()),
                       system_clip=describe(system),
                       explicit_clip=describe(painter.clipRegion()),
                       has_explicit_clip=painter.hasClipping(),
                       device_type=type(painter.device()).__name__,target_type=type(target).__name__,
                       target_format=str(target.format()) if isinstance(target, QImage) else None,
                       ratio=target.devicePixelRatioF() if target is not None else None,
                       device_transform=[transform.m11(),transform.m12(),transform.m21(),transform.m22(),transform.dx(),transform.dy()],
                       world_transform_identity=painter.worldTransform().isIdentity(),
                       system_equals_mapped_region=system == transform.map(event.region()))
            result = original(painter, event, width, height)
            row["used_presentation"] = result
            if result and compare_pixels[0] and isinstance(target, QImage):
                bounds = transform.map(event.region()).boundingRect()
                cached = target.copy(bounds)
                backdrop._paint_base(painter, backdrop.rect())
                blit(painter, backdrop._last_frame, width, height)
                parity.append(cached == target.copy(bounds))
            observations.append(row)
            return result

        monkeypatch.setattr(backdrop, "_paint_partial_frame", observe)
        for _ in range(4):
            backdrop.repaint(region)
        warm = len(blits)
        for _ in range(12):
            backdrop.repaint(region)
        repeated = len(blits)
        compare_pixels[0] = True
        backdrop.repaint(region)
        report = dict(observations=observations,cache_present=backdrop._partial_frame is not None,
                      warmup_blits=warm,repeated_blits=repeated-warm,native_field_pixel_parity=parity,
                      window_size=[window.width(),window.height()],viewports=[[v.viewport().width(),v.viewport().height()] for v in views])
        destination = os.environ.get("PREVIEW_CACHE_NATIVE_OBSERVATION")
        if destination:
            Path(destination).write_text(json.dumps(report,indent=2)+"\n")
        assert any(row["used_presentation"] for row in observations), report
        assert isinstance(backdrop._partial_frame[1], QImage)
        assert warm > 0
        assert repeated == warm, report
        assert parity and all(parity), report
    finally:
        uninstall_field_fade(qapp)
        invalidate_field_fade()
        qapp.setFont(font)
        qapp.setPalette(palette)
        qapp.setStyleSheet(sheet)


def test_full_paint_selects_fresh_frame_and_drops_the_old_presentation(field):
    field._test_render()
    field._test_render()
    old = weakref.ref(field._partial_frame[1])
    fresh = _frame(9)
    field._producer_box[0].publish(fresh)
    full = QRegion(field.rect())
    want = field._test_render(region=full, reference=True)
    got = field._test_render(region=full)
    assert got == want
    assert field._last_frame is fresh
    assert field._partial_frame is None
    assert old() is None
    field._test_render()
    assert field._partial_frame[1] is None


@pytest.mark.parametrize("change", ["frame", "mutated_frame", "mode", "base", "size", "ratio", "engine"])
def test_presentation_key_changes_use_original_first_partial(field, change):
    field._test_render()
    field._test_render()
    old = field._partial_frame[1]
    args = {}
    if change == "frame":
        field._last_frame = _frame(8)
    elif change == "mutated_frame":
        field._last_frame.fill(QColor("#334477"))
    elif change == "mode":
        field._engine.mode = QPainter.CompositionMode_SourceOver
    elif change == "base":
        field._background = QColor("#0c2233")
    elif change == "size":
        field.resize(101, 79)
    elif change == "ratio":
        args["ratio"] = 2.
    else:
        other = ambient.AmbientWidget(theme="blobs", background="#152839", seed=3)
        field._engine = other._engine
        other.deleteLater()
    want = field._test_render(reference=True, **args)
    assert field._test_render(**args) == want
    assert field._partial_frame[1] is None
    assert field._test_render(**args) == want
    assert field._partial_frame[1] is not old


@pytest.mark.parametrize("kind", ["wallpaper", "corners", "alpha", "cards", "nonbuffered", "no_producer", "null_frame"])
def test_ineligible_partial_paints_keep_the_original_path(field, monkeypatch, kind):
    field._test_render()
    field._test_render()
    if kind == "wallpaper":
        pixmap = QPixmap(17, 19)
        pixmap.fill(QColor("#69754c"))
        field.set_backdrop(pixmap)
    elif kind == "corners":
        field._corner_radius = 8
    elif kind == "alpha":
        field._background = QColor(20, 40, 60, 127)
    elif kind == "cards":
        field._card_tints = types.SimpleNamespace(active=True)
        monkeypatch.setattr(ambient, "_draw_cards", lambda *args: None)
        field._card_tints.plan = None
    elif kind == "nonbuffered":
        monkeypatch.setattr(ambient, "_BufferedEngine", type("NotThisEngine", (), {}))
    elif kind == "no_producer":
        field._producer_box[0] = None
    else:
        field._last_frame = QImage()
    want = field._test_render(reference=True)
    assert field._test_render() == want
    assert field._partial_frame is None
    field._card_tints = None


@pytest.mark.parametrize("format", [QImage.Format_RGBA64_Premultiplied, QImage.Format_RGB30])
def test_high_depth_surfaces_are_never_quantized_through_the_rgb32_cache(field, format):
    want = field._test_render(format=format, reference=True)
    for _ in range(3):
        assert field._test_render(format=format) == want
        assert field._partial_frame is None


def test_oversize_presentation_is_rejected_before_allocation(field, monkeypatch):
    original = ambient.QImage
    allocations = []

    class Factory(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original)

    class Forbidden(metaclass=Factory):
        Format_RGB32 = original.Format_RGB32
        Format_ARGB32 = original.Format_ARGB32
        Format_ARGB32_Premultiplied = original.Format_ARGB32_Premultiplied

        def __new__(cls, *args):
            allocations.append(args)
            raise AssertionError("oversize cache allocation")

    image = QImage(4, 4, QImage.Format_RGB32)
    painter = QPainter(image)
    try:
        monkeypatch.setattr(ambient, "QImage", Forbidden)
        assert not field._paint_partial_frame(painter, QPaintEvent(QRect(0, 0, 1, 1)), 20000, 20000)
    finally:
        painter.end()
    assert allocations == []
    assert field._partial_frame is None


@pytest.mark.parametrize("failure", ["raise", "null"])
def test_failed_cache_allocation_falls_back_without_publishing_a_partial_image(field, monkeypatch, failure):
    want = field._test_render(reference=True)
    field._test_render()
    original = ambient.QImage

    class Factory(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, original)

    class Failed(metaclass=Factory):
        Format_RGB32 = original.Format_RGB32
        Format_ARGB32 = original.Format_ARGB32
        Format_ARGB32_Premultiplied = original.Format_ARGB32_Premultiplied

        def __new__(cls, *args):
            if failure == "raise":
                raise MemoryError("controlled allocation failure")
            return original()

    monkeypatch.setattr(ambient, "QImage", Failed)
    assert field._test_render() == want
    assert field._partial_frame[1] is False
    assert field._test_render() == want


@pytest.mark.parametrize("action", ["stop", "hide", "republish", "rebuild", "restart", "backdrop", "cards"])
def test_lifecycle_changes_release_the_presentation(field, monkeypatch, action):
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    if action == "stop":
        field.stop()
    elif action == "hide":
        field.hideEvent(QHideEvent())
    elif action == "republish":
        field._republish()
    elif action == "rebuild":
        field._rebuild_engine()
    elif action == "restart":
        monkeypatch.setattr(field, "_should_run", lambda: False)
        field._start_producer()
    elif action == "backdrop":
        field.set_backdrop(None)
    else:
        field._drop_card_plan()
    assert field._partial_frame is None
    assert held() is None


def test_inactive_paint_device_releases_the_presentation(field):
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    painter = QPainter()
    assert not painter.isActive()
    assert not field._paint_partial_frame(painter, QPaintEvent(QRect(3, 5, 53, 41)), 97, 73)
    assert field._partial_frame is None
    assert held() is None


@pytest.mark.parametrize("screen", [None, types.SimpleNamespace(depth=lambda: 30)])
def test_widget_without_a_supported_screen_keeps_the_original_path(field, monkeypatch, screen):
    field._test_render()
    field._test_render()
    monkeypatch.setattr(field, "screen", lambda: screen)
    painter = types.SimpleNamespace(device=lambda: field)
    assert not field._paint_partial_frame(painter, QPaintEvent(QRect(3, 5, 53, 41)), 97, 73)
    assert field._partial_frame is None


@pytest.mark.parametrize("failure", ["screen_metadata", "device_transform", "system_clip"])
def test_unavailable_native_metadata_releases_the_cache(field, monkeypatch, failure):
    """A disappearing native surface cannot leave stale pixels eligible."""
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    image = QImage(97, 73, QImage.Format_RGB32)
    painter = QPainter(image)
    painter.setClipRegion(QRegion(QRect(3, 5, 53, 41)))

    def unavailable():
        raise RuntimeError("native surface metadata unavailable")

    class Surface:
        def __getattr__(self, name):
            return getattr(painter, name)

    surface = Surface()
    if failure == "screen_metadata":
        surface.device = lambda: field
        monkeypatch.setattr(field, "screen", unavailable)
    elif failure == "device_transform":
        surface.deviceTransform = unavailable
    else:
        surface.paintEngine = lambda: types.SimpleNamespace(systemClip=unavailable)
    try:
        assert not field._paint_partial_frame(surface, QPaintEvent(QRect(3, 5, 53, 41)), 97, 73)
    finally:
        painter.end()
    assert field._partial_frame is None
    assert held() is None


@pytest.mark.parametrize("size", [(0, 73), (97, 0)])
def test_zero_size_presentation_retires_the_old_raster(field, size):
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    image = QImage(97, 73, QImage.Format_RGB32)
    painter = QPainter(image)
    region = QRegion(QRect(3, 5, 53, 41))
    painter.setClipRegion(region)
    try:
        assert not field._paint_partial_frame(painter, QPaintEvent(region), *size)
    finally:
        painter.end()
    assert field._partial_frame is None
    assert held() is None


def test_failed_raster_painter_keeps_original_pixels_and_remembers_the_failure(field):
    want = field._test_render(reference=True)
    field._test_render()
    assert field._test_render(cache_active=False) == want
    assert field._partial_frame[1] is False
    assert field._test_render(cache_active=False) == want


def test_renderer_retiring_its_completed_surface_preserves_pixels_without_double_end(field, monkeypatch, qtlog):
    """A renderer may retire its raster painter after completing the image."""
    want = field._test_render(reference=True)
    field._test_render()
    blit = field._engine.blit
    ended = []

    def complete_and_retire(painter, *args):
        blit(painter, *args)
        native_end = painter.end

        def retire_once():
            ended.append(native_end())
            return ended[-1]

        monkeypatch.setattr(painter, "end", retire_once)
        assert painter.end()

    monkeypatch.setattr(field._engine, "blit", complete_and_retire)
    assert field._test_render() == want
    assert isinstance(field._partial_frame[1], QImage)
    assert field._test_render() == want
    assert ended == [True]
    assert not any("QPainter::end" in record.message for record in qtlog.records)


@pytest.mark.parametrize("event", [QEvent.Move, QEvent.ScreenChangeInternal])
def test_host_move_or_screen_change_releases_the_old_presentation(field, monkeypatch, event):
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    field._watched = weakref.ref(field)
    followed = []
    monkeypatch.setattr(field, "_follow_screen", lambda: followed.append(True))
    assert not field.eventFilter(field, QEvent(event))
    assert followed == [True]
    assert field._partial_frame is None
    assert held() is None


def test_running_drift_backend_change_drops_the_old_presentation(field, monkeypatch):
    field._test_render()
    field._test_render()
    held = weakref.ref(field._partial_frame[1])
    engine = ambient.DriftEngine(colors=[QColor("#4488aa")], background=field._background, seed=3)
    field._engine = engine
    field._timer.start(10000)
    restarted = []
    monkeypatch.setattr(field, "_start_producer", lambda: restarted.append(True))
    field._set_graphics_backend("cpu")
    field._timer.stop()
    assert restarted == [True]
    assert field._partial_frame is None
    assert held() is None
    assert field._last_frame is None
    assert field._last_composed is None
