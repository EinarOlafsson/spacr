"""One user-facing control per question, and none of them can be jerky.

Asked for 2026-08-28: "there need to be fewer options for speed so one
option for speed that is user facing, one option for steering and so on...
mak the settings easy to navigate aross themes." And: "i tried to set
steering to a minimum and get jerkey moovements."
"""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QDoubleSpinBox, QLabel

from spacr.qt import preferences as P
from spacr.qt.widgets.fractal_mandelbrot import steering_from_one_number


@pytest.fixture
def store(monkeypatch):
    values = {}

    class _Mem:
        def value(self, key, default=None, type=None):
            return values.get(key, default)

        def setValue(self, key, value):
            values[key] = value

        def sync(self):
            pass

    monkeypatch.setattr(P, "_settings", lambda: _Mem())
    monkeypatch.setattr(P, "_SAFE_MODE", False)
    return values


@pytest.mark.parametrize("amount", [i / 20.0 for i in range(21)])
def test_no_steering_setting_is_jerky(amount):
    """A move must never take more than half the gap before the next one.

    Set by hand a short interval and a long duration re-target the camera
    before it has finished moving, so every frame is mid-course-correction.
    Derived together they cannot disagree.
    """
    derived = steering_from_one_number(amount, 24.0)
    gap = derived["steering_interval_decades"] * 24.0
    assert derived["steering_duration"] <= 0.5 * gap, (amount, derived)


def test_calmer_means_gentler_and_rarer_not_the_same_moves_crammed():
    """The way down to zero has to be gentler moves further apart."""
    calm = steering_from_one_number(0.0, 24.0)
    busy = steering_from_one_number(1.0, 24.0)
    assert calm["steering_strength"] < busy["steering_strength"]
    assert calm["steering_interval_decades"] > busy["steering_interval_decades"]


def test_a_move_always_takes_real_time():
    """A move that takes no time is a jump."""
    for amount in (0.0, 0.5, 1.0):
        assert steering_from_one_number(amount, 24.0)["steering_duration"] >= 0.5
    # Even when the descent is very fast, so the interval in seconds is tiny.
    assert steering_from_one_number(1.0, 0.5, )["steering_duration"] >= 0.5


def test_the_control_is_clamped_to_its_own_range():
    for silly in (-3.0, 7.0):
        derived = steering_from_one_number(silly, 24.0)
        assert 0.0 <= derived["steering_strength"] <= 1.0


def test_setting_steering_writes_the_three_numbers(store):
    """One control, three numbers, so nothing can be set into disagreement."""
    P.set_fractal_settings(steering=1.0)
    values = P.get_fractal_settings()
    expected = steering_from_one_number(1.0, values["seconds_per_decade"])
    for name, value in expected.items():
        assert values[name] == pytest.approx(value), name


def test_steering_is_offered_as_one_field(qtbot):
    from spacr.qt.theme import enable_spaceout

    enable_spaceout()
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    assert dlg.findChild(QDoubleSpinBox, "FractalSteering") is not None


def test_the_detail_sits_under_an_advanced_heading(qtbot):
    """A number somebody needs and cannot reach is worse than a long panel,
    but it does not belong beside the control it explains."""
    from spacr.qt.theme import enable_spaceout

    enable_spaceout()
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    headings = [l.text() for l in dlg.findChildren(QLabel)
                if l.objectName() == "FractalGroupHeading"]
    assert any("Advanced" in h for h in headings), headings


def test_the_same_controls_serve_every_pattern():
    """"easy to navigate aross themes" -- the questions do not change with
    the pattern, so neither do the controls."""
    from spacr.qt.widgets.fractal_travel import PATTERNS

    values = P.get_fractal_settings()
    for name in ("speed", "steering", "quality", "scale", "supersampling",
                 "pointer_gravity"):
        assert name in values, name
    assert len(PATTERNS) == 4
