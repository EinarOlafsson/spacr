"""Native display sampling and persisted custom colours for procedural art."""

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QWidget

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
    engine.set_resolution(2.0)
    assert engine.buffer_size(3840, 2160) == (3840, 2160)
    image = engine.shade(3840, 2160)
    assert image.size().width() == 3840
    assert image.size().height() == 2160
    engine.set_resolution(0.5)
    assert engine.buffer_size(3840, 2160) == (1920, 1080)
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
