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
