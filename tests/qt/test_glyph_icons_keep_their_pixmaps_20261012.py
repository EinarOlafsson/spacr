"""A glyph icon renders each pixmap once and keeps it.

QtAwesome renders its glyph in Python every time Qt asks an icon for a
pixmap, and a button asks on every paint: over the window's moving field
that was a Python glyph render per frame for the preferences cog every
module screen carries (measured 2026-10-12, about a third of a millisecond
of a 2560x1440 Mask Generation frame). The semantic icons now keep each
pixmap per size, mode and state; the pixels are QtAwesome's own.
"""
from __future__ import annotations

import pytest
from PySide6.QtGui import QIcon

from spacr.qt import iconset


@pytest.fixture
def provider(qapp, monkeypatch):
    qta = pytest.importorskip("qtawesome")
    monkeypatch.setattr(iconset, "_try_qta", lambda: qta)
    monkeypatch.setattr(qapp, "_spacr_glyph_providers", {}, raising=False)
    monkeypatch.setitem(qta._resource, "iconic", None)
    return qta


def _counting(font, monkeypatch):
    painted = []
    real = font.painter.paint

    def paint(*args):
        painted.append(args[3])
        return real(*args)

    monkeypatch.setattr(font.painter, "paint", paint)
    return painted


def test_a_pixmap_is_rendered_once_per_size_mode_and_state(provider,
                                                           monkeypatch):
    font = iconset._glyph_provider(provider, "fa5s.cog")
    assert isinstance(font, iconset._kept_pixmap_font(provider))
    assert iconset._kept_pixmap_font(provider) is type(font)
    painted = _counting(font, monkeypatch)
    icon = iconset.icon("settings")
    first = icon.pixmap(18, 18).toImage()
    for _again in range(3):
        assert icon.pixmap(18, 18).toImage() == first
    assert len(painted) == 1
    icon.pixmap(24, 24)
    icon.pixmap(18, 18, QIcon.Disabled)
    assert len(painted) == 3
    stock = provider.icon("fa5s.cog", color=iconset._theme_palette(None)[
        "fg_muted"])
    assert stock.pixmap(18, 18).toImage() == first


def test_an_animated_icon_renders_every_time(provider, monkeypatch):
    font = iconset._glyph_provider(provider, "fa5s.cog")
    painted = _counting(font, monkeypatch)

    class _Still:
        def setup(self, *_args):
            return None

    icon = font.icon("fa5s.cog", color="white", animation=_Still())
    assert not icon.isNull()
    icon.pixmap(16, 16)
    icon.pixmap(16, 16)
    assert len(painted) == 2


def test_a_font_that_is_not_a_class_is_used_as_it_is(provider):
    class _Fake:
        IconicFont = staticmethod(lambda *args: None)

    assert iconset._kept_pixmap_font(_Fake) is _Fake.IconicFont
