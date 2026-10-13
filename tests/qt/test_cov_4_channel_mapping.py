"""The RGB mapping field still works when the theme cannot be reached.

The widget asks the theme to stop its container painting the window colour
over what is behind it. That is decoration: if the import or the call fails,
the three spin boxes must still be built, still hold the mapping, and still
announce changes. A field that refused to appear because a colour helper
broke would take the whole settings panel with it.
"""
from __future__ import annotations

import pytest

from spacr.qt import theme
from spacr.qt.widgets.channel_mapping import ChannelMappingWidget


def _explode(*_args, **_kwargs):
    raise RuntimeError("theme unavailable")


def test_a_broken_theme_helper_still_yields_a_working_field(qapp, monkeypatch):
    """Decoration failing must not cost the user the editor itself."""
    monkeypatch.setattr(theme, "make_transparent", _explode)
    widget = ChannelMappingWidget({"r": 2, "g": 1, "b": 0})
    assert widget.get_value() == {"r": 2, "g": 1, "b": 0}


def test_a_broken_theme_helper_still_announces_edits(qapp, monkeypatch):
    """The signal the panel listens on is not part of the decoration."""
    monkeypatch.setattr(theme, "make_transparent", _explode)
    widget = ChannelMappingWidget({"r": 2, "g": 1, "b": 0})
    seen = []
    widget.valueChanged.connect(seen.append)
    widget.set_value({"r": 0, "g": 1, "b": 2})
    assert seen == [{"r": 0, "g": 1, "b": 2}], seen


def test_the_theme_helper_is_asked_for_when_it_works(qapp, monkeypatch):
    """The transparent container is the intended look, not an accident."""
    asked = []
    monkeypatch.setattr(theme, "make_transparent", asked.append)
    widget = ChannelMappingWidget(None)
    assert asked == [widget]
