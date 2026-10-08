"""Requested defaults, relative blue rims and configurable dot flicker."""

import numpy as np
import pytest
from PySide6.QtCore import QEvent, QRectF, QSettings
from PySide6.QtGui import QColor, QImage
from PySide6.QtWidgets import QComboBox, QPushButton, QSlider, QWidget

from spacr.qt import preferences as prefs
from spacr.qt.widgets import ambient
from spacr.qt.widgets.setup_card import SetupCard


@pytest.fixture
def empty_store(monkeypatch, tmp_path):
    monkeypatch.setattr(prefs, "_settings", lambda: QSettings(
        str(tmp_path / "preferences.ini"), QSettings.IniFormat))


def test_fresh_requested_defaults(empty_store, qapp):
    assert prefs.get_theme_choice() == "dark"
    assert prefs.get_ambient_animation() == "data_art_impulse_lens"
    assert prefs.get_ambient_palette() == "spacr"
    assert prefs.get_ambient_density() == 0.1
    assert prefs._ambient_blink_percent() == 0.0
    assert prefs._ambient_gravity_radius() == 0.15
    assert prefs._field_popup_wave_frequency() == 5.0
    assert prefs._rim_length_fraction() == 0.17
    assert prefs.get_ambient_resolution() == 1.0
    assert prefs.get_ambient_speed() == 1.0
    assert prefs.get_ambient_size() == 1.0
    assert prefs.get_popup_backdrop() == "off"
    assert prefs.get_pane_opacity() == 0.6
    assert prefs.get_field_fade_enabled()
    assert prefs.get_rim_lag() == 0.5
    assert prefs.get_rim_alignment() == "centre"
    assert prefs.get_rim_mode() == "beat"
    assert prefs.get_rim_period() == 1.5


def test_reset_restores_requested_controls_without_writing(empty_store, qtbot):
    prefs.set_ambient_density(2.0)
    prefs._set_ambient_gravity_radius(0.7)
    prefs.set_popup_backdrop("drift")
    prefs.set_rim_lag(0.9)
    prefs.set_rim_mode("rainbow")
    prefs.set_rim_period(4.0)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    assert dialog.findChild(QWidget, "SettingAnimationsEnabled") is None
    dialog.findChild(QPushButton, "PreferencesReset").click()
    for name, value in (("AmbientDensity", 10), ("AmbientGravityRadius", 15),
                        ("RimLag", 50), ("RimPeriod", 15)):
        assert dialog.findChild(QSlider, name).value() == value
    assert dialog.findChild(QComboBox, "PopupBackdrop").currentData() == "off"
    assert prefs.get_ambient_density() == 2.0
    assert prefs.get_rim_mode() == "rainbow"


def test_relative_rim_fraction_is_identical_across_sizes(empty_store, qapp):
    prefs._set_rim_length_fraction(0.3)
    card = SetupCard()
    for width, height in ((120, 90), (980, 700), (3840, 2160)):
        assert card.accent_span(QRectF(0, 0, width, height)) == 0.3
    prefs._set_rim_length_fraction(0.12)
    card.reread_the_preferences()
    assert card.accent_span(QRectF(0, 0, 3840, 2160)) == 0.12


@pytest.mark.parametrize("raw", ["broken", "nan", "inf", "-inf"])
def test_invalid_saved_rim_length_uses_safe_default(empty_store, qapp, raw):
    prefs._settings().setValue(prefs._KEY_RIM_LENGTH_FRACTION, raw)
    assert prefs._rim_length_fraction() == 0.17
    card = SetupCard()
    assert card.accent_span(QRectF(0, 0, 640, 480)) == 0.17


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_appearance_controls_persist_safe_values(empty_store, raw):
    prefs._set_rim_length_fraction(raw)
    prefs._set_field_popup_wave_frequency(raw)
    assert prefs._rim_length_fraction() == 0.17
    assert prefs._field_popup_wave_frequency() == 0


def test_safe_mode_refuses_idle_prewarm_even_when_requested(empty_store, monkeypatch):
    monkeypatch.setenv("SPACR_PREWARM", "1")
    monkeypatch.setattr(prefs, "_SAFE_MODE", True)
    assert prefs._screen_prewarm_allowed() == (False, "safe mode")


@pytest.mark.parametrize("palette", ["spacr", "ocean", "random"])
def test_dot_palettes_flicker_the_selected_percentage(qapp, palette):
    engine = ambient.make_engine("data_art_impulse_lens", palette, "#101010",
                                 seed=42, density=1.0, blink_percent=1.0)
    frame = engine.shade(640, 480)
    assert frame is not None
    key, first = engine._field_flicker
    _tick, visible, count = key
    assert abs(count - visible / 100) <= 1
    assert len(first) == len(np.unique(first)) == count
    assert np.any(np.frombuffer(frame.constBits(), dtype=np.uint32)
                  == 0xffffffff)
    original = bytes(frame.constBits())
    engine.time += 0.3
    later = engine.shade(640, 480)
    assert not np.array_equal(first, engine._field_flicker[1])
    assert bytes(frame.constBits()) == original
    assert bytes(later.constBits()) != original


def test_home_tile_description_is_instant(empty_store, qtbot):
    from spacr.qt.widgets.home import HomePage
    prefs._set_tooltip_delay(4)
    screen = HomePage([("mask", "Mask", "Generate masks", "stable")],
                      lambda _key: None)
    qtbot.addWidget(screen)
    tile = next(iter(screen._tile_hints))
    screen.eventFilter(tile, QEvent(QEvent.Enter))
    assert "Generate masks" in screen._hint_bar.text()


def test_flicker_lights_exactly_one_percent_of_visible_centres(qapp):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101010",
                                 seed=19, blink_percent=1.0)
    image = QImage(320, 320, QImage.Format_RGB32)
    image.fill(QColor("black"))
    x, y = np.meshgrid(np.arange(8, 308, 10), np.arange(8, 308, 10))
    engine._flicker_field_dots(image, x.ravel(), y.ravel())
    white = np.count_nonzero(np.frombuffer(image.constBits(), dtype=np.uint32)
                            == 0xffffffff)
    assert white == 900 // 100


def test_flicker_keeps_selected_identities_when_visible_centres_move(qapp):
    engine = ambient.make_engine("data_art_impulse_lens", "spacr", "#101010",
                                 seed=19, blink_percent=10.0)
    x, y = np.meshgrid(np.arange(10, 110, 10), np.arange(10, 110, 10))
    first = QImage(320, 160, QImage.Format_RGB32)
    first.fill(QColor("black"))
    engine._flicker_field_dots(first, x.ravel(), y.ravel())
    cached = engine._field_flicker
    selected = cached[1]
    assert len(selected) == 10

    moved_x = x.ravel() + 150
    second = QImage(320, 160, QImage.Format_RGB32)
    second.fill(QColor("black"))
    engine._flicker_field_dots(second, moved_x, y.ravel())

    assert engine._field_flicker is cached
    white = np.flatnonzero(np.frombuffer(second.constBits(), dtype=np.uint32)
                           == 0xffffffff)
    expected = y.ravel()[selected] * second.width() + moved_x[selected]
    assert np.array_equal(np.sort(white), np.sort(expected))


@pytest.mark.parametrize("theme", ["dark", "light", "data_art_impulse_lens",
                                    "data_art_fungal_growth", "high_contrast"])
def test_blue_rim_remains_visible_in_each_theme(empty_store, qapp, monkeypatch, theme):
    from spacr.qt import theme as styling
    monkeypatch.setattr(styling, "active_palette", lambda: styling.palette_for(theme))
    card = SetupCard(mode="beat")
    card.resize(400, 280)
    card._timer.stop()
    card._phase = 1.5 / 4
    image = QImage(card.size(), QImage.Format_RGB32)
    image.fill(QColor("black"))
    card.render(image)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint32).reshape(280, 400)
    red = (pixels >> 16) & 255
    green = (pixels >> 8) & 255
    blue = pixels & 255
    edge = np.zeros((280, 400), dtype=bool)
    edge[:3, :] = edge[-3:, :] = True
    edge[:, :3] = edge[:, -3:] = True
    assert np.any(edge & (blue > green + 12) & (green > red + 12))
