"""A pinned figure colour that fights the theme is handed back to it.

Reported 2026-08-19: "all of the graphs need to be adapted to dark mode. as
it is now all of them have black axees and lines the grid is black. in dark
mode these elements need to be all white."

THE CAUSE WAS A PREFERENCE, NOT A RENDERER. `_style_figure_colors` has been
walking every axis, spine, tick and legend and reassigning colours for a long
time. It was being handed ('#ffffff', '#000000') -- an EXPLICIT pair -- on a
dark theme, so it painted black ink faithfully. A one-shot migration exists
to undo exactly that freeze, but it writes a marker and never looks again, so
a store frozen after it ran stays frozen for good. Measured on the
maintainer's machine, which is where the report came from.
"""
import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    path = tmp_path / "spacr.ini"
    monkeypatch.setattr(
        preferences, "_settings",
        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _pin(store, bg, fg):
    settings = store._settings()
    settings.setValue(store._KEY_FIG_BG, bg)
    settings.setValue(store._KEY_FIG_FG, fg)
    # MARK THE STORE AS ALREADY MIGRATED, so the one-shot pass is not what
    # does the work here. These tests are about the hole it leaves: a store
    # frozen AFTER it ran. `FIGURE_COLOR_SCALE` is the marker it checks.
    settings.setValue(store._KEY_FIG_COLOR_SCALE, store.FIGURE_COLOR_SCALE)


def test_a_pair_that_fights_the_theme_goes_back_to_auto(store, monkeypatch):
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    _pin(store, "#ffffff", "#000000")          # black ink on a dark theme

    bg, fg = store.get_figure_color_tokens()

    assert store.figure_color_is_auto(bg)
    assert store.figure_color_is_auto(fg)
    assert store.get_figure_colors() == store.auto_figure_colors()


def test_a_pair_that_already_matches_the_theme_is_left_alone(store,
                                                             monkeypatch):
    """Silent for everybody it does not help."""
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    auto_bg, auto_fg = store.auto_figure_colors()
    _pin(store, auto_bg, auto_fg)

    bg, fg = store.get_figure_color_tokens()

    assert (bg, fg) == (auto_bg, auto_fg)


def test_a_colour_auto_never_produces_is_kept(store, monkeypatch):
    """A deliberate choice survives. 'auto' has only ever produced black,
    white or transparent, so teal was somebody's decision."""
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    _pin(store, "#123456", "#00ffcc")

    assert store.get_figure_color_tokens() == ("#123456", "#00ffcc")


def test_a_current_explicit_black_and_white_choice_is_kept(store,
                                                            monkeypatch):
    """The repair must not undo a choice made through the current setter."""
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    store.set_figure_colors("#ffffff", "#000000")

    assert store.get_figure_color_tokens() == ("#ffffff", "#000000")


def test_a_half_that_is_already_auto_is_not_disturbed(store, monkeypatch):
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    _pin(store, store.AUTO_FIGURE_COLOR, "#000000")

    bg, _fg = store.get_figure_color_tokens()

    assert store.figure_color_is_auto(bg)


def test_the_dark_theme_gets_light_ink(store, monkeypatch):
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "dark")
    _bg, fg = store.auto_figure_colors()
    assert fg == "#ffffff"


def test_the_light_theme_gets_dark_ink(store, monkeypatch):
    monkeypatch.setattr(store, "resolve_effective_theme", lambda: "light")
    _bg, fg = store.auto_figure_colors()
    assert fg == "#000000"


def test_the_background_is_transparent_on_both(store, monkeypatch):
    """A figure is not a window (INVARIANTS 2). An opaque slab behind every
    plot is where "the black rectangle" came from."""
    for theme in ("dark", "light"):
        monkeypatch.setattr(store, "resolve_effective_theme", lambda: theme)
        bg, _fg = store.auto_figure_colors()
        assert bg == store.TRANSPARENT_FIGURE_BG


def test_an_unreadable_store_cannot_survive_a_theme_switch(store, monkeypatch):
    """The whole point: the same store, read on either theme, gives ink that
    can be seen against it."""
    _pin(store, "#ffffff", "#000000")
    for theme, expected in (("dark", "#ffffff"), ("light", "#000000")):
        monkeypatch.setattr(store, "resolve_effective_theme", lambda: theme)
        _bg, fg = store.get_figure_colors()
        assert fg == expected, theme
