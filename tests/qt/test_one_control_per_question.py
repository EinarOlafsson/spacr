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
def spaceout_only(monkeypatch):
    """Turn spaceout on for one test and OFF again afterwards.

    It is a module-level flag, so a test that enables it and walks away
    leaves the Fractal tab on the Preferences dialog for every test that
    runs after it -- which is what broke
    `test_the_dialog_has_the_expected_subject_tabs_in_order`.
    """
    from spacr.qt import theme

    monkeypatch.setattr(theme, "_SPACEOUT", True)
    yield



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


def test_steering_is_offered_as_one_field(qtbot, spaceout_only):
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    assert dlg.findChild(QDoubleSpinBox, "FractalSteering") is not None


def test_the_derived_numbers_are_settings_but_not_panel_rows(qtbot, spaceout_only):
    """They are still read by the renderer and still carried by a settings
    file -- they are simply not offered beside the control that sets them.

    Putting them behind an "Advanced" heading was the first attempt and was
    not what was asked for: it left the same panel with one more row on it,
    and a hand-set combination could still contradict the control above.
    """
    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)
    headings = [l.text() for l in dlg.findChildren(QLabel)
                if l.objectName() == "FractalGroupHeading"]
    assert not any("Advanced" in h for h in headings), headings

    # Still settings, though: the renderer reads them.
    values = P.get_fractal_settings()
    for name in ("steering_strength", "steering_interval_decades",
                 "steering_duration", "speed_min", "speed_max"):
        assert name in values, name


def test_the_same_controls_serve_every_pattern():
    """"easy to navigate aross themes" -- the questions do not change with
    the pattern, so neither do the controls."""
    from spacr.qt.widgets.fractal_travel import PATTERNS

    values = P.get_fractal_settings()
    for name in ("speed", "steering", "quality", "scale", "supersampling",
                 "pointer_gravity"):
        assert name in values, name
    assert len(PATTERNS) == 4


def test_the_camera_follows_continuously_rather_than_moving_in_steps():
    """Reported twice: "jerkey moovements", then "still jerking arround,
    the jerks are abit smoother".

    The first fix clamped each move to a fraction of the gap, which made
    every slide smoother and left the STARTING and STOPPING exactly where it
    was -- and that alternation is the jerk. A continuous follow never
    starts and never stops; only its speed changes.
    """
    import math

    def follow(target_of, frames=200, duration=4.0, fps=30.0):
        here = 0.0
        path = []
        step = 1.0 / fps
        for index in range(frames):
            wanted = target_of(index * step)
            rate = 1.0 - math.exp(-step / max(0.5, duration))
            here += rate * (wanted - here)
            path.append(here)
        return path

    def stepping(seconds):
        return [0.0, 0.30, -0.20, 0.45][int(seconds // 2.0) % 4]

    path = follow(stepping)
    velocity = [b - a for a, b in zip(path, path[1:])]
    acceleration = [abs(b - a) for a, b in zip(velocity, velocity[1:])]

    # The old shape -- ease over part of the window, then hold -- for
    # comparison: its acceleration spikes when the move ends.
    held = []
    for index in range(200):
        phase = (index % 60) / 60.0
        if phase < 0.6:
            eased = phase / 0.6
            held.append(eased * eased * (3 - 2 * eased) * 0.3)
        else:
            held.append(0.3)
    old_velocity = [b - a for a, b in zip(held, held[1:])]
    old_acceleration = [abs(b - a)
                        for a, b in zip(old_velocity, old_velocity[1:])]

    assert max(acceleration) < max(old_acceleration) / 10.0, (
        f"following peaks at {max(acceleration):.4f} against "
        f"{max(old_acceleration):.4f} for discrete moves")


def test_the_renderer_reconciles_the_numbers_wherever_they_came_from():
    """Deriving them in the panel does not stop a settings file, or an
    older install, carrying a combination that contradicts itself."""
    import inspect

    from spacr.qt.widgets import fractal_travel as ft

    source = inspect.getsource(ft._make_gpu_widget)
    # Zero reach means do not steer, not "steer in an arbitrary direction".
    assert "if strength <= 0.0:" in source
    # And the move is bounded by the gap at the point of USE.
    assert "0.45 * interval * seconds_per_decade" in source


def test_the_panel_offers_one_control_per_question(qtbot, spaceout_only):
    """"there need to be fewer options"."""
    from PySide6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox

    dlg = P.PreferencesDialog(None)
    qtbot.addWidget(dlg)

    named = []
    for kind in (QComboBox, QDoubleSpinBox, QSpinBox):
        named += [c.objectName() for c in dlg.findChildren(kind)
                  if c.objectName().startswith("Fractal")]

    # It was twenty-one.
    assert len(named) <= 10, sorted(named)
    # And none of the derived numbers is offered beside the control that
    # already sets it.
    for gone in ("FractalSpeedMin", "FractalSpeedMax", "FractalSpeedPeriod",
                 "FractalPointerSize", "FractalPointerStrength",
                 "Fractal_steering_strength", "Fractal_steering_duration",
                 "Fractal_steering_interval_decades"):
        assert gone not in named, gone
