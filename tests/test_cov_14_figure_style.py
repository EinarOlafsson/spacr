"""House-style colours survive a settings store that is not there.

Figures are rendered from headless workers and from bare unit runs, neither of
which has a Qt settings store. Every read of user preferences in this module
therefore has to have an answer ready:

* the ink falls back to the screen colour rather than to ``None``, because a
  figure drawn with no text colour is a figure with invisible axes;
* a per-graph override that cannot be resolved contributes no rcParams rather
  than raising through the caller's plotting code;
* an explicit ink argument outranks both, so a caller that already knows the
  colour never has to reach preferences at all.
"""
from __future__ import annotations

import pytest

from spacr.figures import style


def test_an_explicit_ink_wins_over_the_target():
    """A caller that names the colour gets exactly that colour."""
    assert style.resolve_ink("print", ink="#ff00ff") == "#ff00ff"
    assert style.resolve_ink("screen", ink="#ff00ff") == "#ff00ff"


def test_without_an_override_the_target_decides():
    """Print and screen still differ when no ink is supplied."""
    assert style.resolve_ink("print") == style.INK_PRINT
    assert style.resolve_ink("screen") == style.INK_SCREEN


def test_an_unresolvable_override_contributes_no_rcparams(monkeypatch):
    """A stored style the resolver chokes on yields ``{}``, not an exception.

    The overrides are read on the way into every figure; letting the failure
    out would take down the plot instead of the customisation.
    """
    import spacr.figure_style as figure_style
    import spacr.qt.preferences as preferences

    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: {"font.size": 11}, raising=False)
    monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                        lambda: {}, raising=False)

    def _explode(*args, **kwargs):
        raise RuntimeError("unreadable style")

    monkeypatch.setattr(figure_style, "resolve", _explode, raising=False)

    assert style.user_overrides("bar") == {}


def test_a_resolvable_override_does_reach_the_rcparams(monkeypatch):
    """The same path returns the differing keys when the resolver works."""
    import spacr.figure_style as figure_style
    import spacr.qt.preferences as preferences

    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: {"font.size": 11}, raising=False)
    monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                        lambda: {}, raising=False)
    monkeypatch.setattr(figure_style, "resolve",
                        lambda kind, *rest: {"chosen": bool(rest)},
                        raising=False)
    monkeypatch.setattr(
        figure_style, "rc_params",
        lambda chosen: {"font.size": 11 if chosen.get("chosen") else 9},
        raising=False)

    assert style.user_overrides("bar") == {"font.size": 11}


def test_no_settings_store_means_the_screen_target(monkeypatch):
    """``theme_target`` answers 'screen' when preferences cannot be read.

    spaCR's themes are dark, so guessing 'print' would put dark ink on a dark
    ground and produce a figure with nothing legible in it.
    """
    import spacr.qt.preferences as preferences

    def _explode():
        raise RuntimeError("no settings store")

    monkeypatch.setattr(preferences, "get_figure_colors", _explode,
                        raising=False)

    assert style.theme_target() == "screen"


def test_a_white_figure_ground_means_the_print_target(monkeypatch):
    """A readable store with a white ground does select 'print'."""
    import spacr.qt.preferences as preferences

    monkeypatch.setattr(preferences, "get_figure_colors",
                        lambda: ("#FFFFFF", "#000000"), raising=False)

    assert style.theme_target() == "print"
