"""Native display sampling and persisted custom colours for procedural art."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QComboBox, QColorDialog, QDialogButtonBox, QPushButton, QWidget

from spacr.qt import preferences
from spacr.qt.widgets import ambient


@pytest.fixture
def color_store(monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "colors.ini"), QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)
    return settings


def test_native_4k_detail_respects_real_screen_budget_and_lower_detail():
    engine = ambient.make_engine("data_art_point_atlas", "mono", "#101418", seed=7)
    engine.set_max_pixels(3840 * 2160)
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    assert engine.buffer_scale(3840, 2160) == 1.0
    engine.set_resolution(2.0)
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    image = engine.shade(3840, 2160)
    assert image.size().width() == 3840
    assert image.size().height() == 2160
    engine.set_resolution(0.5)
    assert engine.buffer_size(3840, 2160) == (1920, 1080)
    assert engine.buffer_scale(3840, 2160) == 2.0
    engine.set_resolution(1.0)
    engine.set_max_pixels(1920 * 1080)
    assert engine.buffer_size(3840, 2160) == (1920, 1080)
    width, height = engine.buffer_size(5000, 3000)
    assert width * height <= engine.max_pixels
    assert ambient.BUFFER_EDGE_CEILING == 2048


def test_high_dpi_worker_uses_physical_pixels_and_keeps_budget_on_theme_switch(
    qtbot, monkeypatch
):
    monkeypatch.setattr(ambient, "screen_pixels", lambda _widget: 3840 * 2160)
    widget = ambient.AmbientWidget(theme="data_art_point_atlas", blur=0, seed=7)
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "devicePixelRatioF", lambda: 2.0)
    widget.resize(320, 200)
    widget.show()
    qtbot.waitExposed(widget)
    qtbot.waitUntil(lambda: widget.frames_shaded() > 0)
    assert widget._producer_box[0].size == (640, 400)
    assert widget._producer_box[0].latest().width() == 640
    widget.set_theme("data_art_chromatin_ribbon")
    assert widget.engine.max_pixels == 3840 * 2160
    assert widget._producer_box[0].size == (640, 400)
    widget.close()
    assert not widget.shading_thread_alive()


def test_installed_art_uses_smooth_idle_rate_but_preserves_explicit_caps(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(240, 160)
    host.show()
    widget = ambient.install_ambient(host, theme="blobs", seed=7)
    assert widget.fps() == 12
    widget.set_theme("data_art_point_atlas")
    assert widget.fps() == 24
    widget._run_paced = True
    assert widget._rate() == ambient._RUN_FPS
    widget.set_fps(8)
    widget.set_theme("data_art_chromatin_ribbon")
    assert widget.fps() == 8


def test_user_color_pair_round_trips_and_live_custom_palette_reloads(color_store, qtbot):
    preferences._set_ambient_custom_colors(("#12abef", "#ff8844"))
    assert preferences._ambient_custom_colors() == ("#12abef", "#ff8844")
    assert ambient.palette_colors("data_art_point_atlas", "custom") == ("#12abef", "#ff8844")
    widget = ambient.AmbientWidget(theme="data_art_point_atlas", palette="custom", seed=7)
    qtbot.addWidget(widget)
    assert [color.name() for color in widget.engine._colors] == ["#12abef", "#ff8844"]
    preferences._set_ambient_custom_colors(("#445566", "#aabbcc"))
    widget.set_palette("custom")
    assert [color.name() for color in widget.engine._colors] == ["#445566", "#aabbcc"]


@pytest.mark.parametrize("colors", [("red",), ("invalid", "blue")])
def test_invalid_custom_color_pair_does_not_partially_write(color_store, colors):
    before = preferences._ambient_custom_colors()
    with pytest.raises(ValueError):
        preferences._set_ambient_custom_colors(colors)
    assert preferences._ambient_custom_colors() == before


def test_invalid_saved_custom_color_falls_back_without_writing(color_store):
    color_store.setValue("prefs/ambient_primary", "not-a-color")
    assert preferences._ambient_custom_colors() == ("#3b82f6", "#ff00ff")
    assert color_store.value("prefs/ambient_primary") == "not-a-color"


def test_point_grain_is_a_round_disc_with_diagonal_antialiasing():
    engine = ambient.make_engine("data_art_point_atlas", "mono", "#101418", seed=7)
    image = engine._point_material(9, 9, [4], [4], [1.0], spread=True)
    levels = [image.pixelColor(x, y).red() for x, y in ((4, 4), (4, 3), (3, 3), (4, 2))]
    assert levels[0] > levels[1] > levels[2] > levels[3] == 0
    assert image.pixelColor(3, 3) == image.pixelColor(5, 5)


@pytest.mark.parametrize("background", ("#101418", "#f6f7f9"))
def test_overlapping_round_grains_keep_the_brightest_sample(background):
    engine = ambient.make_engine("data_art_point_atlas", "mono", background, seed=7)
    first = engine._point_material(9, 9, [4, 4], [4, 4], [1.0, 0.2], spread=True)
    reference = engine._point_material(9, 9, [4], [4], [1.0], spread=True)
    assert first == reference


@pytest.mark.parametrize("background", ("#101418", "#f6f7f9"))
@pytest.mark.parametrize("colors", (("#ff0011", "#00eeff"), ("#000077", "#00ff00")))
def test_circular_grain_overlap_matches_independent_per_channel_composition(background, colors):
    engine = ambient.make_engine("data_art_point_atlas", "mono", background, seed=7)
    engine.set_colors(colors)
    points = ((2, 2, 0.2), (2, 2, 0.9), (3, 2, 0.6),
              (0, 0, 0.7), (6, 4, 0.8), (-2, 3, 1.0), (7, 3, 1.0))
    actual = engine._point_material(7, 5, [p[0] for p in points],
                                    [p[1] for p in points], [p[2] for p in points],
                                    spread=True)
    individual = [engine._point_material(7, 5, [x], [y], [level], spread=True)
                  for x, y, level in points]
    choose = max if engine.dark else min
    for y in range(5):
        for x in range(7):
            expected = tuple(choose(getattr(image.pixelColor(x, y), channel)()
                                    for image in individual)
                             for channel in ("red", "green", "blue"))
            assert actual.pixelColor(x, y).getRgb()[:3] == expected


def test_gravity_click_passes_through_to_real_controls_and_stops_when_hidden(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(360, 240)
    backdrop = ambient.install_ambient(host, theme="data_art_impulse_lens", seed=7)
    button = QPushButton("Run", host)
    button.setGeometry(140, 90, 80, 40)
    clicks = []
    button.clicked.connect(lambda: clicks.append(True))
    host.show()
    qtbot.waitExposed(host)
    assert backdrop._interaction_app is not None
    before = len(backdrop.engine._gravity_impulses)
    qtbot.mouseClick(button, Qt.LeftButton)
    backdrop._on_tick()
    assert clicks == [True]
    qtbot.waitUntil(lambda: any(
        strength == 1.0 for _stamp, _origin, strength
        in backdrop.engine._gravity_impulses[before:]))
    host.hide()
    assert backdrop._interaction_app is None
    assert not backdrop._pending_art_impulses
    assert not backdrop.shading_thread_alive()


def test_color_dialog_cancel_preserves_store_and_save_applies_complete_pair(
    color_store, qtbot, monkeypatch
):
    preferences.set_theme_choice("data_art_point_atlas")
    original = preferences._ambient_custom_colors()
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    primary = dialog.findChild(QPushButton, "AmbientPrimaryColor")
    accent = dialog.findChild(QPushButton, "AmbientAccentColor")
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    assert primary.isEnabled() and accent.isEnabled()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor())
    primary.click()
    assert preferences._ambient_custom_colors() == original
    assert palette.currentData() != "custom"
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor("#1199aa"))
    primary.click()
    assert "#1199aa" in primary.text()
    assert palette.currentData() == "custom"
    assert preferences._ambient_custom_colors() == original
    dialog.reject()
    assert preferences._ambient_custom_colors() == original

    saved = preferences.PreferencesDialog()
    qtbot.addWidget(saved)
    saved.findChild(QPushButton, "AmbientPrimaryColor").click()
    boxes = saved.findChildren(QDialogButtonBox)
    buttons = [box.button(QDialogButtonBox.Save) for box in boxes]
    next(button for button in buttons if button is not None).click()
    assert preferences._ambient_custom_colors() == ("#1199aa", original[1])
    assert preferences.get_ambient_palette() == "custom"


def test_reset_custom_art_colors_is_cancelable_and_saves_valid_defaults(
    color_store, qtbot
):
    preferences.set_theme_choice("data_art_point_atlas")
    preferences.set_ambient_palette("custom")
    original = ("#194f77", "#da82a1")
    preferences._set_ambient_custom_colors(original)

    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    assert palette.currentData() == "custom"
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert palette.currentIndex() >= 0
    assert palette.currentData() in ambient.palettes_for("blobs")
    assert "#3b82f6" in dialog.findChild(QPushButton, "AmbientPrimaryColor").text()
    assert "#ff00ff" in dialog.findChild(QPushButton, "AmbientAccentColor").text()
    assert preferences._ambient_custom_colors() == original
    dialog.reject()
    assert preferences._ambient_custom_colors() == original
    assert preferences.get_ambient_palette() == "custom"

    saved = preferences.PreferencesDialog()
    qtbot.addWidget(saved)
    saved.findChild(QPushButton, "PreferencesReset").click()
    boxes = saved.findChildren(QDialogButtonBox)
    next(box.button(QDialogButtonBox.Save) for box in boxes
         if box.button(QDialogButtonBox.Save) is not None).click()
    assert preferences._ambient_custom_colors() == ("#3b82f6", "#ff00ff")
    assert preferences.get_ambient_palette() in ambient.palettes_for(
        preferences.get_ambient_animation())


def test_invalid_saved_art_palette_never_selects_an_invalid_dialog_row(
    color_store, qtbot
):
    preferences.set_theme_choice("data_art_point_atlas")
    color_store.setValue(preferences._KEY_AMBIENT_PALETTE, "retired_palette")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    assert palette.currentIndex() >= 0
    assert palette.currentData() in ambient.palettes_for("data_art_point_atlas")
    assert dialog.findChild(QPushButton, "AmbientPrimaryColor").isEnabled()
    assert dialog.findChild(QPushButton, "AmbientAccentColor").isEnabled()


def test_no_animation_saves_without_a_palette_row(color_store, qtbot):
    preferences.set_theme_choice("data_art_point_atlas")
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    animation = dialog.findChild(QComboBox, "AmbientTheme")
    palette = dialog.findChild(QComboBox, "AmbientPalette")
    animation.setCurrentIndex(animation.findData("none"))
    assert palette.count() == 0
    assert palette.currentData() is None
    boxes = dialog.findChildren(QDialogButtonBox)
    next(box.button(QDialogButtonBox.Save) for box in boxes
         if box.button(QDialogButtonBox.Save) is not None).click()
    assert preferences.get_ambient_animation() == "none"
