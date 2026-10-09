"""The launchers and settings windows keep independent animation choices."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog, QDialogButtonBox, QSlider, QWidget

from spacr.qt import preferences as prefs, theme
from spacr.qt.widgets import ambient
from spacr.qt.widgets.glass import _install_the_backdrop


@pytest.fixture
def separate_modes(tmp_path, monkeypatch):
    store = QSettings(str(tmp_path / "preferences.ini"), QSettings.IniFormat)
    monkeypatch.setattr(prefs, "_settings", lambda: store)
    was_spaceout = theme.spaceout_enabled()
    theme.disable_spaceout()
    yield store
    theme.enable_spaceout() if was_spaceout else theme.disable_spaceout()


def test_the_two_launchers_have_distinct_theme_palette_and_off_choices(
        separate_modes):
    store = separate_modes
    store.setValue("prefs/brand_palette_revision", True)
    assert prefs._animation_choices() == ambient.AMBIENT_THEMES + ("none",)
    assert prefs.get_popup_backdrop() == "drift"
    prefs.set_ambient_theme("blobs")
    prefs.set_ambient_palette("ocean")
    prefs.set_ambient_animation("none")
    assert not prefs.get_ambient_enabled()

    theme.enable_spaceout()
    assert prefs.get_ambient_enabled()
    assert prefs.get_ambient_animation() == ambient.DEFAULT_SPACEOUT_THEME
    assert set(prefs._animation_choices()) == (
        set(ambient.AMBIENT_THEMES) | set(ambient.SPACEOUT_ONLY_THEMES)
        | {ambient.SPACEOUT_THEME, "none"})
    assert prefs._popup_backdrop_choices() == ("off",) + tuple(sorted(
        key for key in prefs._animation_choices() if key != "none"))
    prefs.set_ambient_theme("aurora")
    prefs.set_ambient_palette("pastel")
    prefs.set_ambient_animation("none")
    assert not prefs.get_ambient_enabled()

    theme.disable_spaceout()
    assert prefs.get_ambient_animation() == "none"
    assert prefs.get_ambient_theme() == ambient.DEFAULT_THEME
    assert prefs.get_ambient_palette() == "ocean"
    assert not prefs.get_ambient_enabled()
    prefs.set_ambient_theme("blobs")
    assert prefs.get_ambient_palette() == "ocean"
    prefs.set_ambient_enabled(True)
    theme.enable_spaceout()
    assert not prefs.get_ambient_enabled()
    assert prefs.get_ambient_theme() == ambient.DEFAULT_SPACEOUT_THEME
    prefs.set_ambient_enabled(True)
    assert prefs.get_ambient_theme() == ambient.DEFAULT_SPACEOUT_THEME
    prefs.set_ambient_theme("aurora")
    assert prefs.get_ambient_theme() == "aurora"
    assert prefs.get_ambient_palette() == "pastel"


def test_a_spaceout_settings_popup_keeps_its_own_theme_and_motion(
        qtbot, separate_modes):
    store = separate_modes
    store.setValue("prefs/brand_palette_revision", True)
    theme.enable_spaceout()
    prefs.set_popup_backdrop("blobs")
    prefs._set_popup_backdrop_motion("speed", 1.7)
    prefs._set_popup_backdrop_motion("density", 0.4)
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(360, 260)
    dialog.show()
    backdrop = _install_the_backdrop(dialog)
    assert isinstance(backdrop, ambient.AmbientWidget)
    assert backdrop.theme() == "blobs"
    assert backdrop._separate_theme
    assert backdrop._speed == pytest.approx(1.7)
    assert backdrop._density == pytest.approx(0.4)
    assert backdrop.property("spacrPopupBackdrop") is True
    assert prefs.get_ambient_theme() == ambient.DEFAULT_SPACEOUT_THEME

    prefs.set_popup_backdrop("aurora")
    replacement = _install_the_backdrop(dialog)
    assert replacement is not backdrop
    assert replacement.theme() == "aurora"
    assert dialog._spacr_popup_backdrop is replacement
    prefs.set_popup_backdrop("off")
    assert _install_the_backdrop(dialog) is None
    assert replacement.isHidden()
    prefs.set_popup_backdrop("blobs")
    assert _install_the_backdrop(dialog).theme() == "blobs"


def test_popup_motion_remains_independent_when_module_animation_is_off(
        qtbot, separate_modes):
    separate_modes.setValue("prefs/brand_palette_revision", True)
    prefs.set_ambient_animation("none")
    prefs.set_popup_backdrop("drift")
    module = ambient.AmbientWidget(theme=ambient.DEFAULT_THEME)
    qtbot.addWidget(module)
    module.resize(320, 200)
    module.show()
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 200)
    dialog.show()
    backdrop = _install_the_backdrop(dialog)
    prefs.apply_ambient_preferences(QApplication.instance())
    assert not module.is_running() and module.isHidden()
    assert backdrop is dialog._spacr_popup_backdrop
    assert backdrop.theme() == "drift"
    assert backdrop.is_running() and not backdrop.isHidden()


def test_spaceout_popup_can_select_fractal_without_replacing_main_theme(
        qtbot, monkeypatch, separate_modes):
    separate_modes.setValue("prefs/brand_palette_revision", True)
    theme.enable_spaceout()
    prefs.set_ambient_theme("aurora")
    prefs.set_popup_backdrop(ambient.SPACEOUT_THEME)

    class Fractal(QWidget):
        def __init__(self):
            super().__init__()
            self._spaceout_built_from = ("popup",)
            self.retired = False

        def resume(self):
            pass

        def shutdown(self):
            self.retired = True

    monkeypatch.setattr(ambient, "_build_the_spaceout_fractal", lambda settings: Fractal())
    dialog = QDialog()
    qtbot.addWidget(dialog)
    dialog.resize(320, 200)
    dialog.show()
    fractal = _install_the_backdrop(dialog)
    assert isinstance(fractal, Fractal)
    assert fractal.property("spacrPopupBackdrop") is True
    assert prefs.get_ambient_theme() == "aurora"
    prefs.set_popup_backdrop("blobs")
    regular = _install_the_backdrop(dialog)
    assert isinstance(regular, ambient.AmbientWidget)
    assert regular.theme() == "blobs"
    assert fractal.retired
    assert prefs.get_ambient_theme() == "aurora"


def test_settings_motion_controls_save_without_changing_module_motion(
        qtbot, separate_modes):
    store = separate_modes
    store.setValue("prefs/brand_palette_revision", True)
    prefs.set_ambient_speed(2.0)
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    for _ in range(5):
        qtbot.wait(10)
    popup_speed = dialog.findChild(QSlider, "PopupBackdropSpeed")
    assert popup_speed is not None and popup_speed.isEnabled()
    popup_speed.setValue(175)
    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()
    assert prefs._popup_backdrop_motion()["speed"] == pytest.approx(1.75)
    assert prefs.get_ambient_speed() == pytest.approx(2.0)


def test_brand_palette_migrates_once_then_preserves_new_custom_colours(
        separate_modes):
    store = separate_modes
    store.setValue("prefs/ambient_palette", "mono")
    store.setValue("prefs/ambient_primary", "#111111")
    prefs._restore_brand_palette_once()
    assert store.value("prefs/ambient_palette") == "spacr"
    assert not store.contains("prefs/ambient_primary")
    prefs._set_ambient_custom_colors(("#123456", "#abcdef"))
    prefs._set_ambient_background_choice("#223344")
    prefs._restore_brand_palette_once()
    assert prefs._ambient_custom_colors() == ("#123456", "#abcdef")
    assert prefs._ambient_background_choice() == "#223344"


def test_live_fractal_and_regular_replacement_keeps_owner_and_pause(
        qtbot, monkeypatch, separate_modes):
    store = separate_modes
    store.setValue("prefs/brand_palette_revision", True)
    theme.enable_spaceout()

    class Fractal(QWidget):
        def __init__(self, parent):
            super().__init__(parent)
            self._spaceout_built_from = ("test",)
            self._paused = False
            self.retired = False

        def pause(self):
            self._paused = True

        def resume(self):
            self._paused = False

        def is_paused(self):
            return self._paused

        def shutdown(self):
            self.retired = True

    actual_install = ambient.install_ambient

    def install(host, *, theme, palette):
        if theme == ambient.SPACEOUT_THEME:
            widget = Fractal(host)
            widget.show()
            return widget
        return actual_install(host, theme=theme, palette=palette)

    monkeypatch.setattr(ambient, "install_ambient", install)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 200)
    host.show()
    first = Fractal(host)
    first.show()
    first.pause()
    host._ambient = first

    prefs.set_ambient_theme("blobs")
    ambient._apply_spaceout_animation_choice(QApplication.instance())
    regular = host._ambient
    assert isinstance(regular, ambient.AmbientWidget)
    assert regular.theme() == "blobs"
    assert not regular.is_animating()
    assert first.retired

    prefs.set_ambient_theme(ambient.SPACEOUT_THEME)
    ambient._apply_spaceout_animation_choice(QApplication.instance())
    last = host._ambient
    assert isinstance(last, Fractal)
    assert last is not first
    assert last.is_paused()
    assert regular.parentWidget() is None


def test_field_boundary_feedback_is_bounded_and_independent_of_gravity(
        qtbot, separate_modes):
    widget = ambient.AmbientWidget(theme="data_art_impulse_lens",
                                   gravity_radius=0.0)
    qtbot.addWidget(widget)
    widget.resize(320, 200)
    widget.show()
    qtbot.wait(10)
    widget._ripple_from_edge("right")
    assert len(widget._art_input._boundary_waves) == 1
    assert widget._art_input._applied_boundary_serial == 1
    right = ((1.0, 0.0, 1.0, 1.0),)
    assert [origin for _when, origin in widget._engine._popup_waves] == [right]
    assert [origin for _serial, origin in widget._art_input._boundary_waves] == [right]
    for _ in range(20):
        widget._ripple_from_edge("top")
    assert len(widget._art_input._boundary_waves) == 16
    assert len(widget._engine._popup_waves) == 6
    assert widget._engine._popup_waves[-1][1] == ((0.0, 0.0, 1.0, 0.0),)
    widget.hide()
    assert widget._art_input._boundary_serial == 0
    widget._ripple_from_edge("left")
    assert widget._art_input._boundary_serial == 0
