"""Preferences > Appearance > Accessibility (item 655, 2026-10-12).

One folded category gathers what was spread over the dialog or missing:

* High contrast, a switch kept in step with the Theme control; clearing it
  goes back to the theme it replaced, which the store remembers;
* Text size, the Font scale preference shown a second time, not a copy;
* Reduce motion, which holds every animated backdrop still and sets
  ripples, click ripples, popup waves and mouse gravity to zero, and gives
  the user's own values back exactly when it is cleared;
* Stronger focus outlines, a thick focus ring from ``spacr.qt.theme``;
* a Keyboard shortcuts button that opens the shortcut editor (item 656).

Apply, Cancel and Save mean what they mean everywhere else in the dialog,
and the kept-dialog comparison sees the new keys.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, Qt, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication, QCheckBox, QComboBox, QDialogButtonBox, QLabel,
    QMainWindow, QPushButton, QSlider, QWidget,
)

from spacr.qt import preferences  # noqa: E402

pytestmark = pytest.mark.qt


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preference store on the default theme."""
    path = tmp_path / "preferences.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)
    monkeypatch.setenv("SPACR_NETWORK_CONFIG", str(tmp_path / "network.json"))
    preferences.set_theme_choice("dark")
    preferences.set_ambient_animation("none")
    yield path
    from spacr import updater

    updater._undo_network_exports()


def _dialog(qtbot, parent=None):
    dialog = preferences.PreferencesDialog(parent)
    qtbot.addWidget(dialog)
    return dialog


def _theme_combo(dialog):
    return next(combo for combo in dialog.findChildren(QComboBox)
                if combo.findData("glass") >= 0)


def _check(dialog, name):
    control = dialog.findChild(QCheckBox, name)
    assert control is not None, f"no {name}"
    return control


def _close(dialog, which):
    dialog.findChild(QDialogButtonBox).button(which).click()


def _motion(**values):
    return {**preferences._MOTION_DEFAULTS, **values}


class _Fractal:
    """A stand-in spaceout fractal: pause, resume and a property."""

    def __init__(self, paused=False):
        self.paused = paused
        self.props = {}

    def is_paused(self):
        return self.paused

    def pause(self):
        self.paused = True
        return True

    def resume(self):
        self.paused = False
        return True

    def setProperty(self, name, value):  # noqa: N802 - Qt naming
        self.props[name] = value

    def property(self, name):
        return self.props.get(name)


class _GoneFractal(_Fractal):
    """A fractal whose C++ half has been deleted."""

    def is_paused(self):
        raise RuntimeError("Internal C++ object already deleted.")

    def property(self, name):
        raise RuntimeError("Internal C++ object already deleted.")


class _GoneBackdrop(QWidget):
    """A backdrop that is already being torn down."""

    def _sync_run_state(self):
        raise RuntimeError("Internal C++ object already deleted.")


# ---------------------------------------------------------------------------
# the category and its controls
# ---------------------------------------------------------------------------

def test_the_category_sits_on_appearance_with_every_control(store, qtbot):
    dialog = _dialog(qtbot)
    holder = dialog.findChild(QWidget, "PreferencesTabAccessibility")
    appearance = dialog.findChild(QWidget, "PreferencesTabAppearance")
    assert holder is not None and appearance.isAncestorOf(holder)
    for name in ("AccessibilityHighContrast", "AccessibilityTextSize",
                 "AccessibilityReduceMotion", "AccessibilityStrongFocus",
                 "AccessibilityShortcuts", "AccessibilityHelp"):
        control = dialog.findChild(QWidget, name)
        assert control is not None and holder.isAncestorOf(control), name
    for name in ("AccessibilityHighContrast", "AccessibilityTextSize",
                 "AccessibilityReduceMotion", "AccessibilityStrongFocus",
                 "AccessibilityShortcuts"):
        assert dialog.findChild(QWidget, name).accessibleDescription(), (
            f"{name} has no help text")
    assert dialog.findChild(QLabel, "AccessibilityHelp").text()


# ---------------------------------------------------------------------------
# High contrast
# ---------------------------------------------------------------------------

def test_high_contrast_and_the_theme_control_move_together(store, qtbot):
    preferences.set_theme_choice("light")
    dialog = _dialog(qtbot)
    contrast = _check(dialog, "AccessibilityHighContrast")
    combo = _theme_combo(dialog)
    assert not contrast.isChecked()

    contrast.setChecked(True)
    assert combo.currentData() == "high_contrast"
    contrast.setChecked(False)
    assert combo.currentData() == "light", "clearing it did not go back"

    combo.setCurrentIndex(combo.findData("high_contrast"))
    assert contrast.isChecked()
    combo.setCurrentIndex(combo.findData("glass"))
    assert not contrast.isChecked()
    combo.setCurrentIndex(combo.findData("high_contrast"))
    contrast.setChecked(False)
    assert combo.currentData() == "glass"


def test_saving_high_contrast_remembers_the_theme_it_replaced(store, qtbot):
    preferences.set_theme_choice("glass")
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityHighContrast").setChecked(True)
    _close(dialog, QDialogButtonBox.Save)
    assert preferences.get_theme_choice() == "high_contrast"
    assert preferences.get_high_contrast()
    assert preferences._high_contrast_previous_theme() == "glass"

    again = _dialog(qtbot)
    contrast = _check(again, "AccessibilityHighContrast")
    assert contrast.isChecked()
    contrast.setChecked(False)
    assert _theme_combo(again).currentData() == "glass"
    _close(again, QDialogButtonBox.Save)
    assert preferences.get_theme_choice() == "glass"
    assert preferences._high_contrast_previous_theme() == ""


def test_a_lost_previous_theme_goes_back_to_dark(store, qtbot):
    preferences.set_theme_choice("high_contrast")
    preferences._settings().setValue(
        preferences._KEY_HIGH_CONTRAST_PREVIOUS, "no-such-theme")
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityHighContrast").setChecked(False)
    assert _theme_combo(dialog).currentData() == preferences.DEFAULT_THEME


def test_set_high_contrast_round_trips_through_the_store(store):
    preferences.set_theme_choice("light")
    assert preferences.set_high_contrast(True) == "high_contrast"
    assert preferences._high_contrast_previous_theme() == "light"
    assert preferences.set_high_contrast(True) == "high_contrast"
    assert preferences._high_contrast_previous_theme() == "light"
    assert preferences.set_high_contrast(False) == "light"
    assert preferences._high_contrast_previous_theme() == ""
    assert preferences.set_high_contrast(False) == "light"
    assert not preferences.get_high_contrast()


def test_high_contrast_with_nothing_to_go_back_to(store):
    preferences.set_theme_choice("high_contrast")
    preferences._remember_the_theme_before_contrast("high_contrast", None)
    assert preferences._high_contrast_previous_theme() == ""
    assert preferences.set_high_contrast(False) == preferences.DEFAULT_THEME
    assert preferences._theme_to_go_back_to("high_contrast") == "dark"
    assert preferences._theme_to_go_back_to(None) == "dark"


# ---------------------------------------------------------------------------
# Text size
# ---------------------------------------------------------------------------

def test_text_size_is_the_font_scale_shown_twice(store, qtbot):
    dialog = _dialog(qtbot)
    text = dialog.findChild(QSlider, "AccessibilityTextSize")
    font = dialog.findChild(QSlider, "FontScale")
    assert (text.minimum(), text.maximum(), text.value()) == (
        font.minimum(), font.maximum(), font.value())
    text.setValue(130)
    assert font.value() == 130
    assert dialog.findChild(QLabel, "AccessibilityTextSizeValue").text() == "130%"
    font.setValue(110)
    assert text.value() == 110
    text.setValue(125)
    dialog.findChild(QTimer, "ScaleSettle").stop()
    _close(dialog, QDialogButtonBox.Save)
    assert preferences.get_font_scale() == pytest.approx(1.25)


def test_a_held_text_size_waits_for_its_release(store, qtbot, monkeypatch):
    from spacr.qt import gui_scale

    asked = []
    monkeypatch.setattr(
        gui_scale, "change_scales",
        lambda owner, *, gui, font, on_done: asked.append((font, on_done)))
    dialog = _dialog(qtbot)
    text = dialog.findChild(QSlider, "AccessibilityTextSize")
    settle = dialog.findChild(QTimer, "ScaleSettle")
    start = text.value()
    text.setSliderDown(True)
    text.setValue(start + 25)
    assert not settle.isActive(), "a held Text size applied mid-drag"
    text.setSliderDown(False)
    text.sliderReleased.emit()
    assert settle.isActive()
    settle.stop()
    settle.timeout.emit()
    (font, on_done), = asked
    assert font == pytest.approx((start + 25) / 100.0)
    on_done(False)
    assert text.value() == start, "a reverted scale left Text size behind"


# ---------------------------------------------------------------------------
# Reduce motion: the store
# ---------------------------------------------------------------------------

def test_reduce_motion_stores_and_gives_back_exact_values(store):
    preferences._write_motion_values(_motion(
        field_ripple_intensity=1.37, field_click_ripples=False,
        field_popup_wave_frequency=12.3, ambient_gravity_radius=0.123))
    before = preferences._motion_values()

    preferences.set_reduce_motion(True)
    assert preferences.get_reduce_motion()
    assert preferences._motion_values() == preferences._STILL_MOTION
    assert preferences._reduce_motion_previous() == before
    preferences.set_reduce_motion(True)
    assert preferences._reduce_motion_previous() == before

    preferences.set_reduce_motion(False)
    assert not preferences.get_reduce_motion()
    assert preferences._motion_values() == before
    store_keys = preferences._settings().allKeys()
    assert not [key for key in store_keys
                if key.startswith(preferences._KEY_REDUCE_MOTION_BEFORE)]


def test_unreadable_previous_motion_reads_as_the_defaults(store):
    preferences._settings().setValue(
        f"{preferences._KEY_REDUCE_MOTION_BEFORE}/ambient_gravity_radius",
        "not a number")
    preferences._settings().setValue(
        f"{preferences._KEY_REDUCE_MOTION_BEFORE}/field_click_ripples",
        "false")
    kept = preferences._reduce_motion_previous()
    assert kept["ambient_gravity_radius"] == pytest.approx(
        preferences.DEFAULT_AMBIENT_GRAVITY_RADIUS)
    assert kept["field_click_ripples"] is False
    assert kept["field_ripple_intensity"] == 1.0


# ---------------------------------------------------------------------------
# Reduce motion: the dialog
# ---------------------------------------------------------------------------

def _motion_controls(dialog):
    from PySide6.QtWidgets import QDoubleSpinBox

    return (dialog.findChild(QDoubleSpinBox, "FieldRippleIntensity"),
            _check(dialog, "FieldClickRipples"),
            dialog.findChild(QDoubleSpinBox, "FieldPopupWaveFrequencyValue"),
            dialog.findChild(QSlider, "AmbientGravityRadius"))


def test_the_switch_holds_the_controls_still_and_gives_them_back(store, qtbot):
    preferences._write_motion_values(_motion(
        field_ripple_intensity=1.5, field_popup_wave_frequency=8.0,
        ambient_gravity_radius=0.3))
    dialog = _dialog(qtbot)
    ripple, click, waves, gravity = _motion_controls(dialog)
    switch = _check(dialog, "AccessibilityReduceMotion")
    assert not switch.isChecked()

    switch.setChecked(True)
    assert (ripple.value(), click.isChecked(), waves.value(),
            gravity.value()) == (0.0, False, 0.0, 0)
    switch.setChecked(False)
    assert (ripple.value(), click.isChecked(), waves.value(),
            gravity.value()) == (150.0, True, 8.0, 30)


def test_save_with_reduce_motion_and_save_again_without(store, qtbot):
    preferences._write_motion_values(_motion(
        field_ripple_intensity=1.37, field_popup_wave_frequency=12.3,
        ambient_gravity_radius=0.123))
    before = preferences._motion_values()
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityReduceMotion").setChecked(True)
    _close(dialog, QDialogButtonBox.Save)
    assert preferences.get_reduce_motion()
    assert preferences._motion_values() == preferences._STILL_MOTION
    assert preferences._reduce_motion_previous()["field_popup_wave_frequency"] \
        == pytest.approx(12.3)

    again = _dialog(qtbot)
    switch = _check(again, "AccessibilityReduceMotion")
    assert switch.isChecked()
    switch.setChecked(False)
    _close(again, QDialogButtonBox.Save)
    assert not preferences.get_reduce_motion()
    assert preferences._motion_values() == before, (
        "the previous values did not come back exactly")


def test_a_control_moved_before_the_switch_is_what_comes_back(store, qtbot):
    preferences._write_motion_values(_motion(ambient_gravity_radius=0.123))
    dialog = _dialog(qtbot)
    switch = _check(dialog, "AccessibilityReduceMotion")
    _ripple, _click, _waves, gravity = _motion_controls(dialog)
    gravity.setValue(20)
    switch.setChecked(True)
    switch.setChecked(False)
    assert gravity.value() == 20
    switch.setChecked(True)
    _close(dialog, QDialogButtonBox.Save)
    assert preferences._reduce_motion_previous()["ambient_gravity_radius"] \
        == pytest.approx(0.20)


def test_a_control_moved_after_clearing_the_switch_is_what_is_saved(
        store, qtbot):
    preferences.set_reduce_motion(True)
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityReduceMotion").setChecked(False)
    _ripple, _click, _waves, gravity = _motion_controls(dialog)
    gravity.setValue(40)
    _close(dialog, QDialogButtonBox.Save)
    assert preferences._ambient_gravity_radius() == pytest.approx(0.40)


def test_cancel_leaves_reduce_motion_as_it_was(store, qtbot):
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityReduceMotion").setChecked(True)
    _check(dialog, "AccessibilityStrongFocus").setChecked(True)
    _check(dialog, "AccessibilityHighContrast").setChecked(True)
    _close(dialog, QDialogButtonBox.Cancel)
    assert not preferences.get_reduce_motion()
    assert not preferences.get_strong_focus()
    assert preferences.get_theme_choice() == "dark"


def test_reset_to_defaults_clears_every_accessibility_control(store, qtbot):
    preferences.set_reduce_motion(True)
    preferences.set_strong_focus(True)
    preferences.set_high_contrast(True)
    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "PreferencesReset").click()
    assert not _check(dialog, "AccessibilityReduceMotion").isChecked()
    assert not _check(dialog, "AccessibilityStrongFocus").isChecked()
    assert not _check(dialog, "AccessibilityHighContrast").isChecked()
    _ripple, click, _waves, gravity = _motion_controls(dialog)
    assert click.isChecked()
    assert gravity.value() == round(
        preferences.DEFAULT_AMBIENT_GRAVITY_RADIUS * 100)


def test_apply_then_revert_puts_reduce_motion_back(store, qtbot):
    owner = QMainWindow()
    qtbot.addWidget(owner)
    dialog = _dialog(qtbot, owner)
    dialog.show()
    _check(dialog, "AccessibilityReduceMotion").setChecked(True)
    _close(dialog, QDialogButtonBox.Apply)
    qtbot.waitUntil(lambda: dialog._apply_confirmation is not None)
    assert preferences.get_reduce_motion()
    question = dialog._apply_confirmation
    next(b for b in question.buttons() if b.text() == "Revert").click()
    qtbot.waitUntil(lambda: dialog._apply_confirmation is None)
    assert not preferences.get_reduce_motion()
    assert preferences._motion_values()["field_click_ripples"] is True
    _close(dialog, QDialogButtonBox.Cancel)


# ---------------------------------------------------------------------------
# Reduce motion: the backdrops
# ---------------------------------------------------------------------------

def test_reduce_motion_stops_the_field_and_starts_it_again(store, qtbot):
    from spacr.qt.widgets.ambient import AmbientWidget

    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = AmbientWidget(host, theme="data_art_impulse_lens", seed=1)
    backdrop.follow_parent()
    host.show()
    qtbot.waitExposed(host)
    backdrop._sync_run_state()
    assert backdrop.is_running()

    preferences.set_reduce_motion(True)
    preferences._hold_the_backdrops_still(QApplication.instance())
    assert not backdrop.is_running(), "the field kept moving"
    assert not backdrop._should_run()

    preferences.set_reduce_motion(False)
    preferences._hold_the_backdrops_still(QApplication.instance())
    assert backdrop.is_running(), "the field did not start again"
    backdrop.stop()


def test_reduce_motion_holds_the_dna_rain_too(store, qtbot):
    from spacr.qt.widgets.dna_rain import DnaRainWidget

    rain = DnaRainWidget(seed=1)
    qtbot.addWidget(rain)
    rain.resize(320, 240)
    rain.show()
    qtbot.waitExposed(rain)
    assert rain._should_run()
    preferences.set_reduce_motion(True)
    assert not rain._should_run()
    preferences.set_reduce_motion(False)
    rain.stop()


def test_fractals_are_paused_and_only_those_resumed(store, qtbot, monkeypatch):
    from spacr.qt.widgets import ambient

    moving, held_by_a_run, gone = _Fractal(), _Fractal(paused=True), _GoneFractal()
    monkeypatch.setattr(ambient, "_live_spaceout_fractals",
                        lambda: [moving, held_by_a_run, gone])
    tearing_down = _GoneBackdrop()
    qtbot.addWidget(tearing_down)

    preferences.set_reduce_motion(True)
    asked = preferences._hold_the_backdrops_still(QApplication.instance())
    assert asked >= 2
    assert moving.paused and moving.property(
        preferences._MOTION_HELD_PROPERTY)
    assert held_by_a_run.property(preferences._MOTION_HELD_PROPERTY) is None

    preferences.set_reduce_motion(False)
    preferences._hold_the_backdrops_still(QApplication.instance())
    assert not moving.paused
    assert held_by_a_run.paused, "a fractal a run paused was resumed"


def test_an_ambient_module_without_fractals_is_not_a_failure(
        store, qtbot, monkeypatch):
    from spacr.qt.widgets import ambient

    monkeypatch.delattr(ambient, "_live_spaceout_fractals")
    preferences.set_reduce_motion(True)
    assert preferences._hold_the_backdrops_still(
        QApplication.instance()) >= 0


def test_a_fractal_installed_under_reduce_motion_starts_still(
        store, qtbot, monkeypatch):
    from spacr.qt.widgets import ambient

    fractal = _Fractal()
    monkeypatch.setattr(ambient, "_the_spaceout_fractal",
                        lambda host, theme: fractal)
    host = QWidget()
    qtbot.addWidget(host)
    assert ambient.install_ambient(host) is fractal
    assert not fractal.paused
    preferences.set_reduce_motion(True)
    assert ambient.install_ambient(host) is fractal
    assert fractal.paused
    assert not preferences._hold_still_when_motion_is_reduced(fractal)


def test_apply_ambient_preferences_brings_the_backdrops_along(
        store, qtbot, monkeypatch):
    calls = []
    monkeypatch.setattr(preferences, "_hold_the_backdrops_still",
                        lambda app: calls.append(app))
    from spacr.qt.widgets import ambient  # noqa: F401

    preferences.apply_ambient_preferences(QApplication.instance())
    assert calls == [QApplication.instance()]


# ---------------------------------------------------------------------------
# Stronger focus outlines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theme_name", ["dark", "light", "high_contrast"])
def test_the_focus_ring_is_thick_and_legible(theme_name):
    from spacr.qt import theme

    ring = theme.focus_ring_colour(theme_name)
    base = theme.palette_for(theme_name)
    assert theme.contrast_ratio(ring, base["bg"]) >= 3.0
    assert theme.contrast_ratio(ring, base["surface_alt"]) >= 3.0
    rules = theme.focus_ring_qss(theme_name)
    assert ":focus" in rules
    assert f"{theme.FOCUS_RING_PX}px solid {ring}" in rules
    assert theme.FOCUS_RING_PX >= 3


def test_a_faint_accent_falls_back_to_the_ink(monkeypatch):
    from spacr.qt import theme

    faint = dict(theme.palette_for("dark"), accent="#050505")
    monkeypatch.setattr(theme, "palette_for", lambda _name: faint)
    assert theme.focus_ring_colour("dark") == faint["fg"]


def test_the_stylesheet_carries_the_ring_only_when_asked(store, qtbot):
    from spacr.qt import theme

    ring = theme.focus_ring_qss("dark")
    assert ring not in theme.stylesheet("dark", load_widget_registrars=False)
    preferences.set_strong_focus(True)
    assert preferences.get_strong_focus()
    assert ring in theme.stylesheet("dark", load_widget_registrars=False)


def test_saving_stronger_focus_outlines_stores_it(store, qtbot):
    dialog = _dialog(qtbot)
    _check(dialog, "AccessibilityStrongFocus").setChecked(True)
    _close(dialog, QDialogButtonBox.Save)
    assert preferences.get_strong_focus()
    assert _check(_dialog(qtbot), "AccessibilityStrongFocus").isChecked()


# ---------------------------------------------------------------------------
# Keyboard shortcuts
# ---------------------------------------------------------------------------

def test_the_shortcuts_button_opens_the_editor(store, qtbot):
    from spacr.qt.shortcuts import _KeymapDialog

    owner = QMainWindow()
    qtbot.addWidget(owner)
    dialog = _dialog(qtbot, owner)
    dialog.findChild(QPushButton, "AccessibilityShortcuts").click()
    editor = dialog._shortcuts_editor
    assert isinstance(editor, _KeymapDialog)
    assert editor.isVisible()
    assert editor.parent() is owner
    assert editor.windowModality() == Qt.ApplicationModal
    editor.reject()


def test_the_shortcuts_editor_opens_without_a_window(store, qtbot):
    dialog = _dialog(qtbot)
    dialog.findChild(QPushButton, "AccessibilityShortcuts").click()
    editor = dialog._shortcuts_editor
    assert editor.parent() is dialog
    editor.reject()


# ---------------------------------------------------------------------------
# the kept dialog sees the new keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    (preferences._KEY_REDUCE_MOTION, True),
    (preferences._KEY_STRONG_FOCUS, True),
    (preferences._KEY_HIGH_CONTRAST_PREVIOUS, "light"),
])
def test_a_kept_dialog_notices_an_accessibility_key_change(
        store, qtbot, key, value):
    dialog = _dialog(qtbot)
    dialog._reusable = True
    preferences._note_what_the_dialog_shows(dialog)
    assert preferences._still_shows_the_store(dialog)
    preferences._settings().setValue(key, value)
    assert not preferences._still_shows_the_store(dialog)


def test_a_kept_dialog_notices_a_cancelled_accessibility_edit(store, qtbot):
    dialog = _dialog(qtbot)
    dialog._reusable = True
    preferences._note_what_the_dialog_shows(dialog)
    _check(dialog, "AccessibilityReduceMotion").setChecked(True)
    assert not preferences._still_shows_the_store(dialog, saved_by_it=True)


def test_macos_low_power_or_reduce_motion_holds_the_field_and_releases_it(store, qtbot, monkeypatch):
    """The system's own stillness request stops the backdrop, not the preference."""
    from spacr.qt.widgets import ambient
    from spacr.qt.widgets.ambient import AmbientWidget

    asked = {"still": False}
    monkeypatch.setattr(ambient, "_macos_asks_for_stillness", lambda: asked["still"])
    monkeypatch.setattr(ambient, "_MACOS_STILLNESS_RECHECK_MS", 50)
    host = QWidget()
    qtbot.addWidget(host)
    host.resize(320, 240)
    backdrop = AmbientWidget(host, theme="data_art_impulse_lens", seed=1)
    backdrop.follow_parent()
    host.show()
    qtbot.waitExposed(host)
    backdrop._sync_run_state()
    assert backdrop.is_running()

    asked["still"] = True
    backdrop._on_tick()
    assert not backdrop.is_running() and not backdrop._should_run()
    assert preferences.get_reduce_motion() is False, "the spaCR preference is left alone"
    pending = backdrop._stillness_recheck
    assert pending is not None and pending.isActive()
    backdrop._sync_run_state()
    assert backdrop._stillness_recheck is pending, "one pending check at a time"

    asked["still"] = False
    qtbot.waitUntil(backdrop.is_running, timeout=2000)
    backdrop.stop()


def test_macos_stillness_is_false_elsewhere_and_cached(monkeypatch):
    from spacr.qt.widgets import ambient

    monkeypatch.setattr(ambient.sys, "platform", "linux")
    monkeypatch.setattr(ambient, "_MACOS_STILLNESS", [float("-inf"), True])
    assert ambient._macos_asks_for_stillness() is False
    monkeypatch.setattr(ambient.sys, "platform", "darwin")
    monkeypatch.setattr(ambient, "_MACOS_STILLNESS", [ambient.time.monotonic(), True])
    assert ambient._macos_asks_for_stillness() is True, "a fresh answer is reused, not re-read"
