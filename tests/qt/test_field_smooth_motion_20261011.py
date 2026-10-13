"""The spaCR field moves as smoothly as the website's port of it.

Measured offscreen at 1400x900 on CPU before this change: dots were stamped
at whole pixels (each glided 0.19 px a frame, so it hopped one pixel every
five frames), the shading thread kept its own beat, so the time shown by a
paint wandered 4.7-5.8 ms (std) from the time between paints, 1-5 % of paints
repeated a frame, and a 60 fps cap ran at 62.5 Hz. These tests hold the fixes:
fractional-centre grains, wall-time motion, one fresh frame per tick, and a
fractional tick schedule.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QWidget

from spacr.qt.widgets import ambient


def _widget(qtbot, parent=None, fps=60):
    widget = ambient.AmbientWidget(parent, theme="data_art_impulse_lens",
                                   palette="spacr", background="#101418",
                                   fps=fps, seed=42, blur=0, speed=1, size=1,
                                   resolution=1, density=0.25,
                                   direction=ambient.DEFAULT_DRIFT_DIRECTION,
                                   gravity_radius=0.0)
    if parent is None:
        qtbot.addWidget(widget)
    return widget


class _Clock:
    def __init__(self, steps_ms):
        self._steps = list(steps_ms)

    def restart(self):
        return self._steps.pop(0)

    def isValid(self):
        return True


@pytest.mark.parametrize("steps", [[10] * 12, [40, 40, 40], [5, 55, 7, 53]])
def test_motion_follows_elapsed_wall_time_not_tick_count(qtbot, steps):
    widget = _widget(qtbot)
    widget.resize(160, 100)
    widget._clock = _Clock(steps)
    for _ in steps:
        widget._on_tick()
    assert widget.engine.time == pytest.approx(sum(steps) / 1000.0, abs=1e-9)

    reference = _widget(qtbot)
    reference.resize(160, 100)
    reference.set_time(sum(steps) / 1000.0)
    with widget._engine_lock:
        moved = widget.engine.shade(160, 100)
    with reference._engine_lock:
        expected = reference.engine.shade(160, 100)
    assert moved == expected


def _grain(x, kernel):
    lookup = (np.arange(256, dtype=np.uint32) * np.uint32(0x010101)) | np.uint32(0xff000000)
    flat = np.zeros(16 * 16, dtype=np.uint32)
    xs = np.array([x], np.float32)
    ys = np.array([8.5], np.float32)
    values = np.array([1.0], np.float32)
    if kernel is None:
        ambient._subpixel_grains_numpy(np, flat, xs, ys, values, lookup, 16, 16, True)
    else:
        kernel(flat, xs, ys, values, lookup, 16, 16, True, np.zeros(16))
    return (flat & 255).reshape(16, 16).astype(float)


def test_grains_glide_between_pixels_with_constant_energy():
    centres, energies = [], []
    for step in range(21):
        x = 6.0 + step / 20.0
        image = _grain(x, None)
        centres.append((image.sum(0) * (np.arange(16) + 0.5)).sum() / image.sum() - x)
        energies.append(image.sum() / 255.0)
    assert max(abs(c) for c in centres) < 0.05
    assert max(energies) - min(energies) < 0.05
    aligned = _grain(6.5, None)[7:10, 5:8]
    assert aligned.tolist() == [[61, 173, 61], [173, 255, 173], [61, 173, 61]]


def test_compiled_and_numpy_grains_agree():
    ambient._warm_packed_scatter()
    kernel = ambient._SUBPIXEL_SCATTER
    if kernel is None:
        pytest.skip("numba is not available")
    for x in (3.0, 3.13, 3.5, 3.77, -0.6, 15.2):
        assert np.array_equal(_grain(x, kernel), _grain(x, None))


def test_each_tick_paints_one_fresh_frame(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 160)
    widget = _widget(qtbot, host)
    widget.setGeometry(host.rect())
    host.show()
    qtbot.waitExposed(host)
    widget._timer.setInterval(100_000)
    producer = widget._producer_box[0]
    assert producer is not None and producer.tick_driven
    qtbot.waitUntil(lambda: producer.frames_shaded >= 1, timeout=3000)
    qtbot.wait(50)
    widget._clock = _Clock([1000.0 / 60] * 9)
    shaded = producer.frames_shaded
    widget._on_tick()
    qtbot.waitUntil(lambda: producer.frames_shaded > shaded, timeout=5000)
    repeated = widget.repeated_frames
    shown, clocks = [], []
    for _ in range(8):
        shaded = producer.frames_shaded
        widget._on_tick()
        shown.append(widget._last_frame.cacheKey())
        clocks.append(widget.engine.time)
        qtbot.waitUntil(lambda: producer.frames_shaded > shaded, timeout=5000)
    assert widget.repeated_frames == repeated
    assert len(set(shown)) == len(shown)
    assert all(b > a for a, b in zip(clocks, clocks[1:]))


def test_the_tick_schedule_keeps_the_fractional_rate(qtbot, monkeypatch):
    widget = _widget(qtbot, fps=60)
    widget._exact_fps = 60.0
    widget._apply_rate()
    now = [1000.0]
    monkeypatch.setattr(ambient, "time", types.SimpleNamespace(
        perf_counter=lambda: now[0] / 1000.0, monotonic=lambda: now[0] / 1000.0))
    monkeypatch.setattr(widget._timer, "isActive", lambda: True)
    intervals = []
    for _ in range(120):
        widget._schedule_next_tick()
        intervals.append(widget._timer.interval())
        now[0] += widget._timer.interval()
    assert set(intervals[1:]) <= {16, 17}
    assert sum(intervals[1:61]) == pytest.approx(1000.0, abs=2)


@pytest.mark.parametrize("refresh,expected", [(60.0, 60.0), (59.94, 59.94),
                                              (75.0, 37.5), (120.0, 60.0),
                                              (144.0, 48.0), (0.0, 60.0)])
def test_the_field_rate_divides_the_screen_refresh(monkeypatch, refresh, expected):
    from spacr.qt import hidpi

    screen = types.SimpleNamespace(refreshRate=lambda: refresh)
    monkeypatch.setattr(hidpi, "screen_for_widget", lambda widget: screen)
    assert ambient._display_hz(None, ambient.FIELD_FPS) == pytest.approx(expected)


def test_installed_field_runs_at_the_display_rate_and_paces_runs(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    field = ambient.install_ambient(host, theme="data_art_impulse_lens", seed=3)
    assert field.fps() == round(ambient._display_hz(field, ambient.FIELD_FPS))
    field._run_paced = True
    field._apply_rate()
    assert field._rate() == ambient._RUN_FPS
    assert field._timer.interval() == 1000 // ambient._RUN_FPS
    field.set_theme("data_art_point_atlas")
    assert field.fps() == ambient.DEFAULT_FPS
