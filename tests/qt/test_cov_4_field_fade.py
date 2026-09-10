"""Field chrome falls back to the shipped look when preferences cannot be read.

The fade is decoration, and decoration is never load-bearing: an unreadable
settings store, or a theme that cannot be resolved, must still leave every
form field painted. Falling back to "off" would silently ship a different
look to whoever has a broken settings file, which is the one user least able
to tell that is what happened.
"""
from __future__ import annotations

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QLineEdit

from spacr.qt import preferences
from spacr.qt.widgets import field_fade


@pytest.fixture(autouse=True)
def _restore_cache():
    """The enabled flag is process-wide; put back whatever was cached."""
    saved = field_fade._enabled
    yield
    field_fade._enabled = saved


def _explode(*_args, **_kwargs):
    raise RuntimeError("settings store unreadable")


def test_an_unreadable_preference_leaves_the_fade_on(monkeypatch):
    """Off would be a different look nobody chose and nobody can see why."""
    monkeypatch.setattr(preferences, "get_field_fade_enabled", _explode)
    field_fade._enabled = None
    assert field_fade.field_fade_enabled() is True


def test_a_readable_preference_is_honoured(monkeypatch):
    """The fallback must not shadow the stored answer."""
    monkeypatch.setattr(preferences, "get_field_fade_enabled", lambda: False)
    field_fade._enabled = None
    assert field_fade.field_fade_enabled() is False


def test_an_unresolvable_theme_still_paints_the_field(qapp, monkeypatch):
    """A field that refused to paint would be an invisible control."""
    monkeypatch.setattr(preferences, "resolve_effective_theme", _explode)
    field = QLineEdit()
    field.setFixedSize(120, 24)
    image = QImage(field.size(), QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        field_fade.paint_field_fade(field, painter)
    finally:
        painter.end()
    painted = sum(1 for x in range(image.width())
                  if image.pixelColor(x, image.height() // 2).alpha() > 0)
    assert painted > 0


def test_a_field_with_no_room_inside_its_border_is_not_painted(qapp):
    """A one-pixel field has nothing left once the outline is inset."""
    field = QLineEdit()
    field.setFixedSize(1, 1)
    image = QImage(8, 8, QImage.Format_ARGB32)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        field_fade.paint_field_fade(field, painter, theme="dark")
    finally:
        painter.end()
    assert all(image.pixelColor(x, y).alpha() == 0
               for x in range(8) for y in range(8))
