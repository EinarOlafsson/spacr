"""Preferences does not repaint the whole application for unrelated saves."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import preferences, theme  # noqa: E402


def test_an_identical_visual_save_does_not_recompose_the_global_sheet(
    qapp, monkeypatch
):
    """A logging/cache/export change must not repolish every live widget."""
    previous = getattr(qapp, "_spacr_preferences_style_signature", None)
    if hasattr(qapp, "_spacr_preferences_style_signature"):
        delattr(qapp, "_spacr_preferences_style_signature")

    composed: list[dict] = []
    real_stylesheet = theme.stylesheet

    def counted_stylesheet(*args, **kwargs):
        composed.append(dict(kwargs))
        return real_stylesheet(*args, **kwargs)

    monkeypatch.setattr(theme, "stylesheet", counted_stylesheet)
    try:
        preferences.apply_preferences_to_app(qapp)
        first_sheet = qapp.styleSheet()
        preferences.apply_preferences_to_app(qapp)

        assert len(composed) == 1
        assert qapp.styleSheet() == first_sheet
    finally:
        if previous is None:
            if hasattr(qapp, "_spacr_preferences_style_signature"):
                delattr(qapp, "_spacr_preferences_style_signature")
        else:
            setattr(qapp, "_spacr_preferences_style_signature", previous)


def test_a_visual_change_still_rebuilds_the_global_sheet(qapp, monkeypatch):
    """The shortcut may never strand a changed theme or opacity."""
    preferences.apply_preferences_to_app(qapp)
    original = getattr(qapp, "_spacr_preferences_style_signature")
    changed = list(original)
    changed[1] = float(changed[1]) + 0.01
    setattr(qapp, "_spacr_preferences_style_signature", tuple(changed))

    composed = []
    real_stylesheet = theme.stylesheet

    def counted_stylesheet(*args, **kwargs):
        composed.append(True)
        return real_stylesheet(*args, **kwargs)

    monkeypatch.setattr(theme, "stylesheet", counted_stylesheet)
    try:
        preferences.apply_preferences_to_app(qapp)
        assert composed == [True]
    finally:
        setattr(qapp, "_spacr_preferences_style_signature", original)
