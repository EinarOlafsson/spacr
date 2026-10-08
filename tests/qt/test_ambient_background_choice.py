"""Saved animation fills follow the active page while remaining optional."""

import pytest
from types import SimpleNamespace

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QSettings
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QColorDialog, QDialog, QDialogButtonBox, QLabel, QPushButton, QVBoxLayout,
)

from spacr.qt import preferences
from spacr.qt.widgets import ambient
from spacr.qt.widgets import glass
from spacr.qt.widgets.home import HomePage
from spacr.qt.screens.app_screen import AppScreen
from tests.qt.test_preferences_apply import (
    _answer, _apply, _dialog, private_preferences,
)


@pytest.fixture
def background_store(monkeypatch, tmp_path):
    settings = QSettings(str(tmp_path / "ambient-background.ini"),
                         QSettings.IniFormat)
    monkeypatch.setattr(preferences, "_settings", lambda: settings)
    return settings


def _save(dialog):
    boxes = dialog.findChildren(QDialogButtonBox)
    next(box.button(QDialogButtonBox.Save) for box in boxes
         if box.button(QDialogButtonBox.Save) is not None).click()


def test_background_choice_is_optional_matte_and_constrained_by_page(
    background_store, monkeypatch
):
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    from spacr.qt.theme import active_page_colour

    assert preferences._ambient_background_choice() is None
    assert preferences._effective_ambient_background() == QColor(active_page_colour())
    preferences._set_ambient_background_choice("#dddddd")
    dark = preferences._effective_ambient_background()
    assert dark.alpha() == 255
    assert dark.lightness() <= 72
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "light")
    light = preferences._effective_ambient_background()
    assert light.alpha() == 255
    assert light.lightness() >= 184
    preferences._set_ambient_background_choice("#333333")
    assert preferences._effective_ambient_background().lightness() >= 184
    monkeypatch.setattr(preferences, "resolve_effective_theme",
                        lambda: "high_contrast")
    assert preferences._effective_ambient_background().lightness() <= 72
    background_store.setValue(preferences._KEY_AMBIENT_BACKGROUND,
                              "not-a-colour")
    assert preferences._ambient_background_choice() is None
    preferences._set_ambient_background_choice("#333333")
    with pytest.raises(ValueError, match="invalid"):
        preferences._set_ambient_background_choice("not-a-colour")
    assert preferences._ambient_background_choice() == "#333333"
    preferences._set_ambient_background_choice(None)
    assert preferences._ambient_background_choice() is None


def test_background_picker_is_independent_of_palette_and_cancelable(
    background_store, monkeypatch, qtbot
):
    preferences.set_theme_choice("data_art_point_atlas")
    original_pair = preferences._ambient_custom_colors()
    original_palette = preferences.get_ambient_palette()
    monkeypatch.setattr(QColorDialog, "getColor",
                        lambda *_args: QColor("#3a3a3a"))
    dialog = preferences.PreferencesDialog()
    qtbot.addWidget(dialog)
    choose = dialog.findChild(QPushButton, "AmbientBackgroundColor")
    reset = dialog.findChild(QPushButton, "AmbientBackgroundReset")
    assert choose.isEnabled()
    assert not reset.isEnabled()
    monkeypatch.setattr(QColorDialog, "getColor", lambda *_args: QColor())
    choose.click()
    assert not reset.isEnabled()
    monkeypatch.setattr(QColorDialog, "getColor",
                        lambda *_args: QColor("#3a3a3a"))
    choose.click()
    assert "#3a3a3a" in choose.text()
    assert reset.isEnabled()
    assert preferences._ambient_background_choice() is None
    dialog.reject()
    assert preferences._ambient_background_choice() is None

    saved = preferences.PreferencesDialog()
    qtbot.addWidget(saved)
    saved.findChild(QPushButton, "AmbientBackgroundColor").click()
    _save(saved)
    assert preferences._ambient_background_choice() == "#3a3a3a"
    assert preferences._ambient_custom_colors() == original_pair
    assert preferences.get_ambient_palette() == original_palette

    reverted = preferences.PreferencesDialog()
    qtbot.addWidget(reverted)
    reverted.findChild(QPushButton, "AmbientBackgroundReset").click()
    reverted.reject()
    assert preferences._ambient_background_choice() == "#3a3a3a"
    restored = preferences.PreferencesDialog()
    qtbot.addWidget(restored)
    restored.findChild(QPushButton, "AmbientBackgroundReset").click()
    _save(restored)
    assert preferences._ambient_background_choice() is None


def test_saved_background_reaches_new_and_live_ambient_widgets(
    background_store, monkeypatch, qtbot, qapp
):
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    preferences.set_ambient_animation("blobs")
    preferences.set_popup_backdrop("blobs")
    preferences._set_ambient_background_choice("#444444")
    expected = preferences._effective_ambient_background()
    home = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=3)
    popup = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=3)
    manual = ambient.AmbientWidget(theme="blobs", palette="spacr",
                                  background="#123456", seed=3)
    qtbot.addWidget(home)
    qtbot.addWidget(popup)
    qtbot.addWidget(manual)
    popup.setProperty("spacrPopupBackdrop", True)
    assert home.background_color() == expected
    assert popup.background_color() == expected
    assert home.engine.background == expected
    preferences._set_ambient_background_choice("#242424")
    preferences.apply_ambient_preferences(qapp)
    changed = preferences._effective_ambient_background()
    assert home.background_color() == changed
    assert popup.background_color() == changed
    assert home.engine.background == changed
    assert popup.engine.background == changed
    assert manual.background_color() == QColor("#123456")
    home.close()
    popup.close()
    manual.close()


def test_apply_revert_restores_background_and_unchanged_apply_does_not_reshade(
    private_preferences, monkeypatch, qtbot, qapp
):
    preferences.set_ambient_animation("blobs")
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=5)
    qtbot.addWidget(widget)
    dialog, _owner, _accepted = _dialog(qtbot)
    monkeypatch.setattr(QColorDialog, "getColor",
                        lambda *_args: QColor("#454545"))
    dialog.findChild(QPushButton, "AmbientBackgroundColor").click()
    initial = widget.background_color()
    question = _apply(dialog, qtbot)
    assert preferences._ambient_background_choice() == "#454545"
    assert widget.background_color() != initial
    _answer(question, "Revert", qtbot)
    assert preferences._ambient_background_choice() is None
    assert widget.background_color() == QColor(
        preferences._effective_ambient_background())

    dialog.findChild(QPushButton, "AmbientBackgroundReset").click()
    question = _apply(dialog, qtbot)
    _answer(question, "Keep", qtbot)
    calls = []
    original = widget._apply_background

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(widget, "_apply_background", record)
    preferences.apply_ambient_preferences(qapp)
    assert calls == []
    dialog.reject()
    widget.close()


def test_background_falls_back_to_theme_when_settings_cannot_be_read(
    background_store, monkeypatch, qtbot
):
    baseline = ambient._theme_background()
    monkeypatch.setattr(preferences, "_ambient_background_choice",
                        lambda: (_ for _ in ()).throw(RuntimeError("store failed")))
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=5)
    qtbot.addWidget(widget)
    assert widget.background_color() == baseline
    widget.close()


def test_chosen_background_updates_on_application_palette_change(
    background_store, monkeypatch, qtbot
):
    preferences._set_ambient_background_choice("#202020")
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    widget = ambient.AmbientWidget(theme="blobs", palette="spacr", seed=5)
    qtbot.addWidget(widget)
    dark = widget.background_color()
    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "light")
    widget.changeEvent(QEvent(QEvent.ApplicationPaletteChange))
    assert widget.background_color().lightness() >= 184
    assert widget.background_color() != dark
    widget.close()


def test_real_home_module_and_glassed_popup_use_saved_background(
    background_store, qtbot, qapp
):
    preferences.set_ambient_animation("blobs")
    preferences.set_popup_backdrop("blobs")
    preferences._set_ambient_background_choice("#3d3d3d")
    home = HomePage([("mask", "Mask", "Segment cells", "Core")],
                    lambda _key: None)
    qtbot.addWidget(home)
    module = AppScreen("regression")
    qtbot.addWidget(module)
    module._install_ambient()
    popup = QDialog(home)
    qtbot.addWidget(popup)
    QVBoxLayout(popup).addWidget(QLabel("Readable settings", popup))
    assert glass.glass(popup)
    backdrop = popup._spacr_popup_backdrop
    widgets = (home._ambient, module._ambient, backdrop)
    assert all(isinstance(widget, ambient.AmbientWidget)
               for widget in widgets)
    expected = preferences._effective_ambient_background()
    assert all(widget.background_color() == expected for widget in widgets)
    preferences._set_ambient_background_choice("#222222")
    preferences.apply_ambient_preferences(qapp)
    changed = preferences._effective_ambient_background()
    assert all(widget.background_color() == changed for widget in widgets)
    assert all(widget.engine.background == changed for widget in widgets)
    preferences.set_theme_choice("light")
    preferences.apply_preferences_to_app()
    module._retheme_backdrops()
    light = preferences._effective_ambient_background()
    assert light.lightness() >= 184
    assert all(widget.background_color() == light for widget in widgets)
    fills = []
    module._dna_rain = SimpleNamespace(
        set_background_color=fills.append)
    module._backdrop_applied = None
    module._retheme_backdrops()
    from spacr.qt.theme import page_colour

    assert fills == [page_colour("light")]
    assert backdrop.parentWidget() is popup
    popup.close()
    module.close()
    home.close()
