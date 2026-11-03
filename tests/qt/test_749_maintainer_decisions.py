"""The four UI-audit decisions the maintainer made on 2026-10-31 (item 749).

1. The Navigator's empty guide is grey per theme and reaches 4.5:1.
2. Semantic red is kept for destructive actions only.
3. Window-wide shortcuts follow platform conventions.
4. Field and panel borders reach 3:1 against every page surface.
"""
from __future__ import annotations

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QPushButton


@pytest.mark.parametrize("theme", ["light", "dark"])
@pytest.mark.parametrize("opacity", [0.4, 0.6, 0.8, 1.0])
def test_the_empty_guide_grey_reaches_aa_on_its_panel(theme, opacity):
    from spacr.qt.preferences import resolve_effective_theme  # noqa: F401
    from spacr.qt.theme import (composite, contrast_ratio, palette_for,
                                panel_alpha)
    from spacr.qt.widgets.home import _empty_guide_colour

    colour = _empty_guide_colour(theme, opacity)
    panel = composite("#191919", panel_alpha(theme, "surface_alt", opacity),
                      palette_for(theme)["bg"])
    if contrast_ratio("#ffffff", panel) >= 4.5:
        assert contrast_ratio(colour, panel) >= 4.5
        darker = "#%02x%02x%02x" % ((int(colour[1:3], 16) - 1,) * 3)
        start = 0xe0 if theme == "light" else 0x8a
        assert int(colour[1:3], 16) == start or contrast_ratio(darker, panel) < 4.5, (
            "the grey is lightened only as far as the panel needs")
    else:
        assert colour == "#ffffff", "a panel too thin for any grey gets the lightest ink"
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    assert red == green == blue, "the guide stays a neutral grey"


def test_the_default_pane_opacity_reaches_aa_in_both_themes():
    from spacr.qt.preferences import DEFAULT_PANE_OPACITY_PCT
    from spacr.qt.theme import composite, contrast_ratio, palette_for, panel_alpha
    from spacr.qt.widgets.home import _empty_guide_colour

    opacity = DEFAULT_PANE_OPACITY_PCT / 100
    for theme in ("light", "dark"):
        panel = composite("#191919", panel_alpha(theme, "surface_alt", opacity),
                          palette_for(theme)["bg"])
        assert contrast_ratio(_empty_guide_colour(theme, opacity), panel) >= 4.5, theme


def test_the_empty_guide_starts_from_the_maintainers_greys():
    from spacr.qt.widgets.home import _empty_guide_colour

    assert _empty_guide_colour("light", 1.0) == "#e0e0e0"
    assert _empty_guide_colour("dark", 1.0) == "#8a8a8a"
    assert _empty_guide_colour("dark", 0.6) == "#8a8a8a"
    assert _empty_guide_colour("light", 0.6) != "#808080"


def test_the_guide_shown_on_home_uses_the_theme_grey(qtbot, monkeypatch):
    from spacr.qt import preferences
    from spacr.qt.widgets import home

    monkeypatch.setattr(preferences, "resolve_effective_theme", lambda: "dark")
    page = home._EasyHomePage()
    qtbot.addWidget(page)
    page._show_empty_guide()
    assert page._guide_visible
    found = page.field.details.document().find("Explore:")
    assert found.charFormat().foreground().color().name() == home._empty_guide_colour("dark")
    assert found.charFormat().foreground().color().name() != "#808080"


@pytest.mark.parametrize("label", ["Delete", "Delete profile", "Remove selected",
                                   "Discard", "Terminate", "Stop", "Abort"])
def test_destructive_labels_stay_red(label):
    from spacr.qt.button_roles import action_role

    assert action_role(label) == "negative"


@pytest.mark.parametrize("label", ["Cancel", "Close", "Close window", "Quit",
                                   "Reset view", "Reset zoom", "Reset levels",
                                   "Reset to defaults", "Clear", "Clear all",
                                   "Clear console", "Reject the rest of the page"])
def test_non_destructive_labels_are_neutral(label):
    from spacr.qt.button_roles import action_role

    assert action_role(label) is None


def test_a_named_danger_button_keeps_red_when_its_label_is_neutral(qapp):
    from spacr.qt import button_roles

    button = QPushButton("Clear all objects")
    button.setObjectName("DangerButton")
    button_roles._SemanticButtonFilter().classify(button)
    assert button.property("buttonActionRole") == "negative"
    button.deleteLater()


def test_close_buttons_are_no_longer_named_danger(qtbot):
    from spacr.qt.widgets.feature_dictionary import FeatureDictionaryDialog

    dialog = FeatureDictionaryDialog()
    qtbot.addWidget(dialog)
    close = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Close")
    assert close.objectName() != "DangerButton"
    assert close.property("buttonActionRole") is None


OLD_TO_NEW = {
    "Go to home": ("Ctrl+0", "Alt+Home"),
    "Open preferences": ("Ctrl+P", "Ctrl+,"),
    "Pause or resume the background": ("Ctrl+T", "Ctrl+Shift+T"),
    "Blank the background": ("Ctrl+B", "Ctrl+Shift+B"),
}


def test_the_shortcut_map_carries_the_new_keys():
    from spacr.qt.shortcuts import SHORTCUTS

    by_label = {}
    for spec in SHORTCUTS:
        by_label.setdefault(spec.label, []).append(spec.keys)
    for label, (old, new) in OLD_TO_NEW.items():
        assert new in by_label[label], label
        assert old not in by_label[label], label
    assert by_label["Zoom in"] == ["Ctrl+=", "Ctrl++"]
    assert by_label["Zoom out"] == ["Ctrl+-"]
    assert by_label["Reset zoom"] == ["Ctrl+0"]
    assert "Restart the background" not in by_label
    assert not [spec for spec in SHORTCUTS if spec.keys == "Ctrl+R"]


def test_window_wide_keys_do_not_repeat():
    from spacr.qt.shortcuts import SHORTCUTS

    keys = [spec.keys for spec in SHORTCUTS if spec.keys != "F11"]
    assert len(keys) == len(set(keys))


def test_preferences_answers_the_platform_key_first(qapp):
    from spacr.qt.app import _preferences_shortcuts

    keys = _preferences_shortcuts()
    assert keys[0] == QKeySequence("Ctrl+,")
    for standard in QKeySequence.keyBindings(QKeySequence.StandardKey.Preferences):
        assert standard in keys
    assert len(keys) == len(set(key.toString() for key in keys))


def test_keyboard_zoom_steps_and_resets(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import gui_scale, preferences, shortcuts

    applied = []
    asked = []
    state = {"font": 0.9}
    monkeypatch.setattr(preferences, "get_font_scale", lambda: state["font"])

    def change(parent=None, *, font=None, ask=True, previous=None, **_kw):
        if previous is None:
            state["font"] = font
            applied.append(font)
        else:
            asked.append((font, previous))
        return None

    monkeypatch.setattr(gui_scale, "change_scales", change)
    window = QMainWindow()
    qtbot.addWidget(window)

    assert shortcuts._step_zoom(window, 1) == pytest.approx(0.95)
    assert shortcuts._step_zoom(window, 1) == pytest.approx(1.0)
    assert shortcuts._step_zoom(window, -1) == pytest.approx(0.95)
    assert applied == [pytest.approx(0.95), pytest.approx(1.0), pytest.approx(0.95)]
    assert asked == []
    window._spacr_keyboard_zoom_timer.stop()
    shortcuts._settle_keyboard_zoom(window)
    assert asked and asked[0][0] == pytest.approx(0.95)
    assert asked[0][1][1] == pytest.approx(0.9), "a burst asks once, from where it began"

    shortcuts._step_zoom(window, 1)
    shortcuts._reset_zoom(window)
    assert state["font"] == pytest.approx(preferences.DEFAULT_FONT_SCALE)
    assert not window._spacr_keyboard_zoom_timer.isActive()


def test_keyboard_zoom_stops_at_the_bounds(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import gui_scale, preferences, shortcuts

    monkeypatch.setattr(preferences, "get_font_scale", lambda: preferences.FONT_SCALE_MAX)
    monkeypatch.setattr(gui_scale, "change_scales",
                        lambda *a, **k: pytest.fail("Zoom moved past its maximum"))
    window = QMainWindow()
    qtbot.addWidget(window)
    assert shortcuts._step_zoom(window, 1) == pytest.approx(preferences.FONT_SCALE_MAX)


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_borders_reach_three_to_one_on_every_page_surface(theme):
    from spacr.qt.theme import (PAGE_SURFACES, contrast_ratio,
                                effective_surface, palette_for)

    border = palette_for(theme)["border"]
    for surface in PAGE_SURFACES:
        assert contrast_ratio(border, effective_surface(theme, surface)) >= 3.0, surface


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_inputs_draw_their_edge_in_the_border_role(theme):
    from spacr.qt.theme import palette_for, stylesheet

    qss = stylesheet(theme)
    start = qss.index("QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {")
    rule = qss[start:qss.index("}", start)]
    assert f"border: 1px solid {palette_for(theme)['border']}" in rule


class _Store:
    """A preferences store holding only what a test puts in it."""

    def __init__(self, data):
        self.data = dict(data)

    def value(self, key, default=""):
        return self.data.get(key, default)

    def setValue(self, key, value):  # noqa: N802 (Qt naming)
        self.data[key] = value


def _keymap_saved_before_the_change():
    """A keymap exactly as spaCR wrote it before 2026-10-31: no version key."""
    import json

    from spacr.qt import shortcuts as S

    return {S._KEYMAP_KEY: json.dumps({
        "Ctrl+0": "Ctrl+Alt+H",
        "Ctrl+P": "F2",
        "Ctrl+B": "",
        "Ctrl+R": "Ctrl+Alt+R",
        "Ctrl+F": "Ctrl+Shift+T",
        "Ctrl+K": "Ctrl+Shift+K",
    })}


def test_a_keymap_saved_before_the_change_keeps_every_custom_key_on_its_action(monkeypatch):
    """Home, Preferences and Blank keep the user's keys under their new defaults."""
    import json

    from spacr.qt import shortcuts as S

    store = _Store(_keymap_saved_before_the_change())
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    keymap = S._load_keymap()
    assert keymap == {
        "Alt+Home": "Ctrl+Alt+H",
        "Ctrl+,": "F2",
        "Ctrl+Shift+T": "",
        "Ctrl+Shift+B": "",
        "Ctrl+F": "Ctrl+Shift+T",
        "Ctrl+K": "Ctrl+Shift+K",
    }
    assert S._effective("Ctrl+0", keymap) == "Ctrl+0", "Reset zoom did not inherit Home's key"
    assert "Ctrl+R" not in keymap
    assert not S._conflicts(keymap, {})
    assert store.data[S._KEYMAP_VERSION_KEY] == S._KEYMAP_VERSION
    assert "Ctrl+0" not in json.loads(store.data[S._KEYMAP_KEY])
    assert S._load_keymap() == keymap, "the migration runs once"


def test_a_user_key_that_became_a_new_default_stays_the_users(monkeypatch):
    """Search was on Ctrl+Shift+T before the backdrop toggle took that default.

    The backdrop toggle was never rebound, so it is the action left without a
    key rather than the user's Search.
    """
    from spacr.qt import shortcuts as S

    store = _Store(_keymap_saved_before_the_change())
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    keymap = S._load_keymap()
    assert S._effective("Ctrl+F", keymap) == "Ctrl+Shift+T"
    assert S._effective("Ctrl+Shift+T", keymap) == ""


def test_a_keymap_saved_after_the_change_is_read_as_it_is(monkeypatch):
    import json

    from spacr.qt import shortcuts as S

    store = _Store({S._KEYMAP_KEY: json.dumps({"Ctrl+0": "Ctrl+Alt+Z"}),
                    S._KEYMAP_VERSION_KEY: S._KEYMAP_VERSION})
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    assert S._load_keymap() == {"Ctrl+0": "Ctrl+Alt+Z"}


def test_saving_marks_the_keymap_as_current(monkeypatch):
    from spacr.qt import shortcuts as S

    store = _Store({})
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    S._save_keymap({"Ctrl+0": "Ctrl+Alt+Z"}, {})
    assert store.data[S._KEYMAP_VERSION_KEY] == S._KEYMAP_VERSION
    assert S._load_keymap() == {"Ctrl+0": "Ctrl+Alt+Z"}


def test_the_migrated_keymap_reaches_the_right_holders(qtbot, monkeypatch):
    from PySide6.QtGui import QAction, QShortcut
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import shortcuts as S

    store = _Store(_keymap_saved_before_the_change())
    monkeypatch.setattr("spacr.qt.preferences._settings", lambda: store)
    window = QMainWindow()
    qtbot.addWidget(window)
    home = QAction("Home", window)
    home.setShortcut(QKeySequence("Alt+Home"))
    window.addAction(home)
    zoom = QShortcut(QKeySequence("Ctrl+0"), window)
    S._apply_keymap(window)
    assert home.shortcut() == QKeySequence("Ctrl+Alt+H")
    assert zoom.key() == QKeySequence("Ctrl+0")


class _FakeStore:
    """A settings store whose reads and writes can be made to fail."""

    def __init__(self, value=None, read_error=None, write_error=None):
        self._value, self._read_error, self._write_error = value, read_error, write_error
        self.written = {}

    def value(self, key, default=None):
        if self._read_error is not None:
            raise self._read_error
        return self._value

    def setValue(self, key, value):
        if self._write_error is not None:
            raise self._write_error
        self.written[key] = value


def test_the_stored_keymap_version_falls_back_without_raising(monkeypatch):
    from spacr.qt import preferences, shortcuts

    def version_of(store):
        monkeypatch.setattr(preferences, "_settings", lambda: store)
        return shortcuts._stored_keymap_version()

    # a value that is not a number, or nothing recorded: the pre-version keymap
    assert version_of(_FakeStore(value=object())) == 1
    assert version_of(_FakeStore(value=None)) == 1
    # an unreadable store: assume the keymap is current rather than migrate it
    assert version_of(_FakeStore(read_error=RuntimeError("unreadable"))) == (
        shortcuts._KEYMAP_VERSION)


def test_a_migrated_keymap_that_cannot_be_saved_is_dropped_quietly(monkeypatch, caplog):
    import json
    import logging

    from spacr.qt import preferences, shortcuts

    failing = _FakeStore(write_error=OSError("read-only store"))
    monkeypatch.setattr(preferences, "_settings", lambda: failing)
    with caplog.at_level(logging.DEBUG, logger=shortcuts.LOG.name):
        shortcuts._store_migrated_keymap({"Ctrl+,": "Ctrl+Shift+P"})
    assert failing.written == {}
    assert any("could not save the migrated keymap" in r.getMessage()
               for r in caplog.records)

    working = _FakeStore()
    monkeypatch.setattr(preferences, "_settings", lambda: working)
    shortcuts._store_migrated_keymap({"Ctrl+,": "Ctrl+Shift+P"})
    assert json.loads(working.written[shortcuts._KEYMAP_KEY]) == {
        "Ctrl+,": "Ctrl+Shift+P"}
    assert working.written[shortcuts._KEYMAP_VERSION_KEY] == shortcuts._KEYMAP_VERSION


def test_a_keyboard_zoom_that_cannot_read_the_scale_changes_nothing(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import gui_scale, preferences, shortcuts

    def unreadable():
        raise RuntimeError("no scale")

    monkeypatch.setattr(preferences, "get_font_scale", unreadable)
    monkeypatch.setattr(gui_scale, "change_scales",
                        lambda *a, **k: pytest.fail("Zoom changed"))
    window = QMainWindow()
    qtbot.addWidget(window)
    assert shortcuts._step_zoom(window, 1) is None
    assert getattr(window, "_spacr_keyboard_zoom_from", None) is None


def test_settling_a_zoom_burst_asks_only_when_the_scale_moved(qtbot, monkeypatch):
    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import gui_scale, preferences, shortcuts

    asked = []
    monkeypatch.setattr(preferences, "get_font_scale", lambda: 1.0)
    monkeypatch.setattr(gui_scale, "current_scale", lambda: 1.0)
    monkeypatch.setattr(gui_scale, "change_scales",
                        lambda window, **kw: asked.append(kw))
    window = QMainWindow()
    qtbot.addWidget(window)

    # no burst under way: nothing to ask
    window._spacr_keyboard_zoom_from = None
    shortcuts._settle_keyboard_zoom(window)
    assert asked == []

    # the burst ended where it began: nothing to keep
    window._spacr_keyboard_zoom_from = 1.0
    shortcuts._settle_keyboard_zoom(window)
    assert asked == []
    assert window._spacr_keyboard_zoom_from is None

    # the scale moved: ask once, with where it began
    window._spacr_keyboard_zoom_from = 0.9
    shortcuts._settle_keyboard_zoom(window)
    assert asked == [{"font": 1.0, "previous": (1.0, 0.9)}]


def test_a_zoom_question_or_reset_that_fails_is_swallowed(qtbot, monkeypatch, caplog):
    import logging

    from PySide6.QtWidgets import QMainWindow

    from spacr.qt import gui_scale, preferences, shortcuts

    def broken(*args, **kwargs):
        raise RuntimeError("no dialog")

    monkeypatch.setattr(preferences, "get_font_scale", lambda: 1.0)
    monkeypatch.setattr(gui_scale, "current_scale", lambda: 1.0)
    monkeypatch.setattr(gui_scale, "change_scales", broken)
    window = QMainWindow()
    qtbot.addWidget(window)
    with caplog.at_level(logging.DEBUG, logger=shortcuts.LOG.name):
        window._spacr_keyboard_zoom_from = 0.9
        shortcuts._settle_keyboard_zoom(window)
        assert window._spacr_keyboard_zoom_from is None
        # a reset with no pending burst has no timer to stop
        assert getattr(window, "_spacr_keyboard_zoom_timer", None) is None
        shortcuts._reset_zoom(window)
    messages = [r.getMessage() for r in caplog.records]
    assert "could not ask to keep the keyboard Zoom" in messages
    assert "could not reset Zoom" in messages


def test_preferences_keys_add_each_platform_binding_once_and_skip_empty_ones(
        monkeypatch):
    from spacr.qt.app import _preferences_shortcuts

    bindings = [QKeySequence(), QKeySequence("Ctrl+Shift+,"),
                QKeySequence("Ctrl+,"), QKeySequence("Ctrl+Shift+,")]
    monkeypatch.setattr(QKeySequence, "keyBindings",
                        staticmethod(lambda standard: bindings))
    assert _preferences_shortcuts() == [QKeySequence("Ctrl+,"),
                                        QKeySequence("Ctrl+Shift+,")]
