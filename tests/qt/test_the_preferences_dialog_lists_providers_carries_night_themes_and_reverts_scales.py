"""The Preferences dialog's AI provider list, night-theme hand-off, scale
revert and Save, each at an edge the ordinary tests do not reach.

* The Provider combo lists the vendors that are configured; after the
  Providers dialog is accepted it is re-listed and keeps the user's pick,
  and a listing that fails leaves just "Automatic".
* A night theme moves the Animation control and the Sound set control --
  unless the Animation control says none, when only the sound moves.
* A live scale change that the user then reverts puts the sliders back at
  the values in force; one that is kept leaves them where they were
  dragged; a slider held down does not start the settle timer.
* Save still stores and closes when the live GUI scale and the spaceout
  backdrop rebuild both fail.

Every test runs against a throwaway INI store.
"""
from __future__ import annotations

import os
import tempfile
import types

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings, QTimer  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPushButton,
    QSlider,
)

from spacr.qt import preferences as prefs  # noqa: E402

pytestmark = pytest.mark.qt


def _boom(*_args, **_kwargs):
    raise RuntimeError("this part will not answer")


@pytest.fixture
def private_store(monkeypatch):
    path = os.path.join(tempfile.mkdtemp(prefix="spacr-prefs-"), "user.ini")
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    return path


def _dialog(qtbot):
    dialog = prefs.PreferencesDialog()
    qtbot.addWidget(dialog)
    dialog.resize(900, 700)
    return dialog


def _provider(label, name):
    return types.SimpleNamespace(label=label, name=name)


def _items(combo):
    return [combo.itemData(i) for i in range(combo.count())]


# ---------------------------------------------------------------------------
# AI providers
# ---------------------------------------------------------------------------

class _ProvidersDialog:
    answer = QDialog.Rejected

    def __init__(self, _parent):
        pass

    def exec(self):
        return _ProvidersDialog.answer


def test_the_provider_list_is_refreshed_after_the_providers_dialog(
        private_store, qtbot, monkeypatch):
    import spacr.qt.ai as ai
    from spacr.qt.widgets import ai_chat_panel

    monkeypatch.setattr(ai, "configured_providers",
                        lambda: [_provider("Claude", "claude")])
    monkeypatch.setattr(ai_chat_panel, "_ProvidersDialog", _ProvidersDialog)
    dialog = _dialog(qtbot)
    combo = dialog.findChild(QComboBox, "AiProvider")
    button = dialog.findChild(QPushButton, "AiProvidersButton")
    assert _items(combo) == ["", "claude"]
    combo.setCurrentIndex(1)

    monkeypatch.setattr(ai, "configured_providers",
                        lambda: [_provider("Codex", "codex"),
                                 _provider("Claude", "claude")])
    _ProvidersDialog.answer = QDialog.Rejected
    button.click()
    assert _items(combo) == ["", "claude"], "a cancelled dialog changes nothing"

    _ProvidersDialog.answer = QDialog.Accepted
    button.click()
    assert _items(combo) == ["", "codex", "claude"]
    assert combo.currentData() == "claude"

    monkeypatch.setattr(ai, "configured_providers", _boom)
    button.click()
    assert _items(combo) == [""]
    assert combo.currentIndex() == 0


def test_a_provider_list_that_cannot_be_read_offers_automatic_only(
        private_store, qtbot, monkeypatch):
    import spacr.qt.ai as ai

    monkeypatch.setattr(ai, "configured_providers", _boom)
    dialog = _dialog(qtbot)

    assert _items(dialog.findChild(QComboBox, "AiProvider")) == [""]


# ---------------------------------------------------------------------------
# Night themes
# ---------------------------------------------------------------------------

def _theme_combo(dialog):
    from spacr.qt.preferences_navigation import row_field

    combo = row_field(dialog, "Theme")
    assert isinstance(combo, QComboBox)
    return combo


def _choose(combo, key):
    index = combo.findData(key)
    assert index >= 0, key
    combo.setCurrentIndex(index)


def _no_animation():
    try:
        from spacr.qt.widgets.ambient import NO_ANIMATION
    except ImportError:
        NO_ANIMATION = prefs._no_animation_key()
    return NO_ANIMATION


def _two_night_themes():
    from spacr.qt.night_themes import NIGHT_THEME_KEYS, theme_for

    first = NIGHT_THEME_KEYS[0]
    second = next(key for key in NIGHT_THEME_KEYS[1:]
                  if theme_for(key).ambient != theme_for(first).ambient)
    return theme_for, first, second


def test_a_night_theme_moves_the_animation_but_not_past_none(
        private_store, qtbot):
    theme_for, first, second = _two_night_themes()
    dialog = _dialog(qtbot)
    theme = _theme_combo(dialog)
    ambient = dialog.findChild(QComboBox, "AmbientTheme")

    _choose(ambient, _no_animation())
    _choose(theme, first)
    assert ambient.currentData() == _no_animation()

    other = next(ambient.itemData(i) for i in range(ambient.count())
                 if ambient.itemData(i) not in (_no_animation(),
                                                theme_for(second).ambient))
    _choose(ambient, other)
    _choose(theme, second)
    assert ambient.currentData() == theme_for(second).ambient


def test_a_night_theme_shows_its_sound_set_even_with_no_animation(
        private_store, qtbot, monkeypatch):
    theme_for, first, second = _two_night_themes()
    monkeypatch.setattr(prefs, "sound_is_offered", lambda: True)
    dialog = _dialog(qtbot)
    theme = _theme_combo(dialog)
    ambient = dialog.findChild(QComboBox, "AmbientTheme")
    sound = dialog.findChild(QComboBox, "SoundTheme")

    _choose(ambient, _no_animation())
    _choose(theme, first)
    assert sound.currentData() == theme_for(first).sound
    assert ambient.currentData() == _no_animation()

    other = next(ambient.itemData(i) for i in range(ambient.count())
                 if ambient.itemData(i) != _no_animation())
    _choose(ambient, other)
    _choose(theme, second)
    assert sound.currentData() == theme_for(second).sound


# ---------------------------------------------------------------------------
# Live scales, kept and reverted
# ---------------------------------------------------------------------------

def test_a_reverted_scale_puts_the_sliders_back(private_store, qtbot,
                                                monkeypatch):
    from spacr.qt import gui_scale

    asked = []
    monkeypatch.setattr(
        gui_scale, "change_scales",
        lambda owner, *, gui, font, on_done: asked.append(
            (gui, font, on_done)))
    dialog = _dialog(qtbot)
    font_slider = dialog.findChild(QSlider, "FontScale")
    settle = dialog.findChild(QTimer, "ScaleSettle")
    start = font_slider.value()

    font_slider.setSliderDown(True)
    font_slider.setValue(start + 25)
    assert not settle.isActive(), "a held slider waits for its release"
    font_slider.setSliderDown(False)
    settle.timeout.emit()

    (_gui, font, on_done), = asked
    assert font == pytest.approx((start + 25) / 100.0)
    on_done(True)
    assert font_slider.value() == start + 25

    on_done(False)
    assert font_slider.value() == start
    shown = [label.text() for label in dialog.findChildren(QLabel)]
    assert f"{start}%" in shown


def test_an_unchanged_scale_is_not_applied_again(private_store, qtbot,
                                                 monkeypatch):
    from spacr.qt import gui_scale

    asked = []
    monkeypatch.setattr(gui_scale, "change_scales",
                        lambda *a, **k: asked.append(k))
    dialog = _dialog(qtbot)

    dialog.findChild(QTimer, "ScaleSettle").timeout.emit()

    assert asked == []


# ---------------------------------------------------------------------------
# Save when the live parts fail
# ---------------------------------------------------------------------------

def test_save_stores_and_closes_when_the_live_parts_fail(
        private_store, qtbot, monkeypatch):
    from spacr.qt import gui_scale, theme
    from spacr.qt.widgets import ambient, fractal_travel

    monkeypatch.setattr(theme, "spaceout_enabled", lambda: True)
    monkeypatch.setattr(fractal_travel, "_LIVE_CONTROLS", [], raising=False)
    monkeypatch.setattr(gui_scale, "set_gui_scale_live", _boom)
    monkeypatch.setattr(ambient, "rebuild_the_spaceout_backdrops", _boom)
    monkeypatch.setattr(prefs, "apply_preferences_to_app", lambda *a: None)
    dialog = _dialog(qtbot)
    gui_slider = dialog.findChild(QSlider, "GuiScale")
    gui_slider.setValue(125)

    dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Save).click()

    assert dialog.result() == QDialog.Accepted
    assert prefs.get_gui_scale() == pytest.approx(1.25)
