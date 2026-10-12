"""N670 maintainer feedback of 2026-10-12 on field ripples.

1. Closing a window left a shape in the middle of where it had been: the
   wave from its outline also ran inward, and the waves from opposite edges
   met on the outline's centre lines as a shrinking box. An outline's wave
   now runs outward only; the backdrop's own window (an outline that is the
   whole canvas) keeps its inward wave, faded out before the centre.
2. The Field ripples slider made no difference between 100 and 200 %:
   section-fold rings clamped their gathering and saturated their glow at
   100 %, and the other waves moved dots by 1.5 px at 100 %. Every wave now
   scales with the slider, 200 % about twice 100 %, and 0 % is off.

Set ``SPACR_N670_FRAMES`` to a directory to save the frames these tests
measure.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient

pytest.importorskip("scipy")

SIZE = (480, 300)
BOX = ((0.3, 0.3, 0.7, 0.3), (0.3, 0.7, 0.7, 0.7),
       (0.3, 0.3, 0.3, 0.7), (0.7, 0.3, 0.7, 0.7))
CANVAS = ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 1.0, 1.0),
          (0.0, 0.0, 0.0, 1.0), (1.0, 0.0, 1.0, 1.0))


def _engine(dark=True, intensity=1.0):
    made = ambient.make_engine(
        "data_art_impulse_lens", "spacr", "#191919" if dark else "#f4f4f4",
        seed=670, density=1, blink_percent=0, popup_wave_frequency=0)
    made.set_gravity_radius(0)
    made._ripple_intensity = intensity
    made.set_time(10.0)
    return made


def _grey(image):
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format_RGB888)
    raw = np.frombuffer(image.constBits(), np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 3]
    return raw.reshape(image.height(), image.width(), 3).astype(np.float64).mean(axis=2)


def _save(image, name):
    folder = os.environ.get("SPACR_N670_FRAMES")
    if folder:
        os.makedirs(folder, exist_ok=True)
        image.save(os.path.join(folder, name))


def _with_and_without(engine, size=SIZE):
    """Shade ``engine`` with its waves and again, at the same time, without."""
    waved = engine.shade(*size).copy()
    kept = (list(engine._popup_waves), list(engine._column_waves),
            list(engine._fold_waves), dict(engine._gravity_ripple_impulses))
    engine._popup_waves.clear()
    engine._column_waves.clear()
    engine._fold_waves.clear()
    engine._gravity_ripple_impulses.clear()
    plain = engine.shade(*size).copy()
    (engine._popup_waves[:], engine._column_waves[:], engine._fold_waves[:]) = kept[:3]
    engine._gravity_ripple_impulses.update(kept[3])
    return waved, plain


def _points(engine, size=SIZE):
    seen = {}
    original = engine._point_material

    def capture(width, height, x, y, light, spread=False):
        seen["x"], seen["y"] = np.array(x, copy=True), np.array(y, copy=True)
        seen["light"] = np.array(light, dtype=np.float64, copy=True)
        return original(width, height, x, y, light, spread)

    engine._point_material = capture
    try:
        engine.shade(*size)
    finally:
        del engine._point_material
    return seen


class TestAClosedWindowLeavesNothingAtItsCentre:

    def test_the_outline_weight_is_outward_only(self):
        np_ = ambient._numpy()
        x = np.array([0.5, 0.1, 0.31, 0.69], dtype=np.float32)
        y = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        weight = ambient._outside_outline(np_, BOX, x, y, np.zeros(4), (1.6, 1.0))
        assert weight.tolist() == [0.0, 1.0, 0.0, 0.0]
        edge = ambient._outside_outline(np_, ((0.0, 0.0, 1.0, 0.0),), x, y,
                                        np.zeros(4), (1.6, 1.0))
        assert edge.tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_the_whole_canvas_keeps_its_inward_wave_and_fades_before_the_centre(self):
        np_ = ambient._numpy()
        x = np.full(3, 0.5, dtype=np.float32)
        y = np.full(3, 0.5, dtype=np.float32)
        distance = np.array([0.01, 0.2, 0.45])
        weight = ambient._outside_outline(np_, CANVAS, x, y, distance, (1.6, 1.0))
        assert weight[0] == pytest.approx(1.0)
        assert 0.0 < weight[1] < 1.0
        assert weight[2] == 0.0

    @pytest.mark.parametrize("dark", [True, False])
    def test_the_centre_stays_at_background_level_while_the_close_wave_runs(self, dark):
        engine = _engine(dark)
        engine._add_ripple(BOX)
        width, height = SIZE
        rows = slice(int(0.36 * height), int(0.64 * height))
        cols = slice(int(0.36 * width), int(0.64 * width))
        moved_outside, previous = 0.0, 0.0
        for step, age in enumerate((0.3, 0.6, 0.9, 1.2, 1.8, 2.4)):
            engine.advance(age - previous)
            previous = age
            waved, plain = _with_and_without(engine)
            _save(waved, f"close_{'dark' if dark else 'light'}_{step}.png")
            difference = np.abs(_grey(waved) - _grey(plain))
            assert difference[rows, cols].max() == 0.0
            outside = np.concatenate((difference[:int(0.25 * height)].ravel(),
                                      difference[int(0.75 * height):].ravel()))
            moved_outside = max(moved_outside, outside.mean())
        assert moved_outside > 0.0

    def test_moving_or_resizing_the_window_itself_still_ripples_inward(self):
        engine = _engine()
        engine._add_ripple(CANVAS)
        engine.advance(0.6)
        waved, plain = _with_and_without(engine)
        difference = np.abs(_grey(waved) - _grey(plain))
        height, width = difference.shape
        assert difference[: height // 4].mean() > 0.0
        assert difference[height // 2 - 20:height // 2 + 20,
                          width // 2 - 20:width // 2 + 20].max() == 0.0

    def test_a_real_dialog_closing_over_the_field(self, qtbot, monkeypatch):
        monkeypatch.setattr(ambient.AmbientWidget, "_start_producer", lambda self: None)
        owner = QWidget()
        qtbot.addWidget(owner)
        owner.resize(640, 400)
        field = ambient.AmbientWidget(
            owner, theme="data_art_impulse_lens", palette="spacr",
            popup_wave_frequency=0, gravity_radius=0, blink_percent=0,
            ripples_enabled=True, seed=670)
        field.setGeometry(owner.rect())
        owner.show()
        field.show()
        field._timer.stop()
        popup = QDialog(owner)
        qtbot.addWidget(popup)
        popup.resize(240, 160)
        popup.move(owner.mapToGlobal(owner.rect().center()) - popup.rect().center())
        popup.open()
        qtbot.waitUntil(lambda: QApplication.activeModalWidget() is popup)
        field._on_tick()
        field._timer.stop()
        engine = field.engine
        engine._popup_waves.clear()
        popup.reject()
        qtbot.waitUntil(lambda: not popup.isVisible())
        field._on_tick()
        field._timer.stop()
        assert len(engine._popup_waves) == 1
        size = (field.width(), field.height())
        centre = (slice(size[1] // 2 - 40, size[1] // 2 + 40),
                  slice(size[0] // 2 - 70, size[0] // 2 + 70))
        changed = 0.0
        for step in range(8):
            engine.advance(0.3)
            waved, plain = _with_and_without(engine, size)
            _save(waved, f"dialog_close_{step}.png")
            difference = np.abs(_grey(waved) - _grey(plain))
            assert difference[centre].max() == 0.0
            changed = max(changed, difference.mean())
        assert changed > 0.0
        field.stop()


def _strength(kind, intensity, dark=True):
    """Largest dot displacement (px) and dot brightness change for one wave."""
    waved, quiet = _engine(dark, intensity), _engine(dark, intensity)
    age = 1.0
    if kind == "click":
        waved._add_ripple((0.5, 0.5))
    elif kind == "outline":
        waved._add_ripple(BOX)
    elif kind == "column":
        waved._add_column_wave((0.2, 0.6, 0.4))
    else:
        waved._add_ripple(("rings", 0.5, 0.5))
        age = 0.6
    for engine in (waved, quiet):
        engine.advance(age)
    moved, still = _points(waved), _points(quiet)
    shift = np.hypot(moved["x"] - still["x"], moved["y"] - still["y"]).max()
    glow = np.abs(moved["light"] - still["light"]).max() / np.median(still["light"])
    return shift, glow


class TestIntensityScalesEveryWave:

    @pytest.mark.parametrize("kind", ["click", "outline", "column", "fold"])
    def test_strength_is_monotonic_and_200_is_about_twice_100(self, kind):
        shifts, glows = zip(*(_strength(kind, value) for value in (0.0, 0.5, 1.0, 1.5, 2.0)))
        assert shifts[0] == 0.0 and glows[0] == 0.0
        assert all(b > a for a, b in zip(shifts, shifts[1:]))
        assert all(b > a for a, b in zip(glows, glows[1:]))
        assert shifts[4] >= 1.7 * shifts[2]
        assert glows[4] >= 1.7 * glows[2]
        assert shifts[2] >= 0.0045 * min(SIZE)

    @pytest.mark.parametrize("dark", [True, False])
    def test_frames_at_50_100_and_200_percent(self, dark):
        from scipy.ndimage import gaussian_filter

        spread = []
        for value in (0.5, 1.0, 2.0):
            waved, quiet = _engine(dark, value), _engine(dark, value)
            waved._add_ripple((0.5, 0.5))
            for engine in (waved, quiet):
                engine.advance(1.0)
            frame = waved.shade(*SIZE)
            _save(frame, f"click_{'dark' if dark else 'light'}_{int(value * 100)}.png")
            difference = np.abs(gaussian_filter(_grey(frame), 8)
                                - gaussian_filter(_grey(quiet.shade(*SIZE)), 8))
            spread.append(difference.max())
        assert spread[0] < spread[1] < spread[2]
        assert spread[2] >= 1.7 * spread[1]

    def test_dots_never_cross_even_at_200_percent(self):
        np_ = ambient._numpy()
        engine = _engine(intensity=2.0)
        distance = np.linspace(0.0, 1.0, 4001)
        steepest = 0.0
        for age in np.linspace(0.05, ambient.RIPPLE_LIFETIME_S - 0.05, 40):
            shift, _glow = engine._ripple_wave(distance, float(age))
            steepest = max(steepest, float(np_.abs(np_.diff(shift) / np_.diff(distance)).max()))
        assert steepest < 0.8


class TestTheSliderReachesTheShadingThread:

    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        made = QSettings(str(tmp_path / "ripples.ini"), QSettings.IniFormat)
        monkeypatch.setattr(prefs, "_settings", lambda: made)
        return made

    def test_applied_intensity_is_the_producers_engine_value(self, qtbot, store):
        host = QWidget()
        qtbot.addWidget(host)
        host.resize(320, 200)
        field = ambient.AmbientWidget(
            host, theme="data_art_impulse_lens", gravity_radius=0,
            popup_wave_frequency=0, blink_percent=0, seed=670)
        field.setGeometry(host.rect())
        host.show()
        field.show()
        qtbot.waitUntil(lambda: field._producer_box[0] is not None)
        producer = field._producer_box[0]
        for value in (2.0, 0.5, 1.0):
            prefs._set_field_ripple_intensity(value)
            prefs.apply_ambient_preferences(QApplication.instance())
            assert field._ripple_intensity == value
            assert producer._engine is field.engine
            with field._engine_lock:
                assert producer._engine._ripple_intensity == value
        field.stop()
