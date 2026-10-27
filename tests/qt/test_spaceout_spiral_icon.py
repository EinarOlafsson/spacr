"""The Spaceout command keeps its white infinity spiral without icon fonts."""
import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QIcon, QImage, QPainter

from spacr.qt import iconset


def _pixels(icon, side):
    image = QImage(side, side, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        icon.paint(painter, QRect(0, 0, side, side))
    finally:
        painter.end()
    return np.frombuffer(image.constBits(), np.uint8).reshape(side, side, 4).copy()


@pytest.mark.parametrize('side', [16, 96])
@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_spaceout_spiral_has_theme_ink_transparency_and_distinct_lobes(
        qapp, monkeypatch, side, theme):
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    pixels = _pixels(iconset.app_icon('__easy_command__:Spaceout', theme=theme), side)
    fallback = _pixels(iconset.app_icon('__unregistered_icon__', theme=theme), side)
    assert not np.array_equal(pixels, fallback)
    alpha = pixels[:, :, 3]
    assert 0.1 < (alpha > 12).mean() < 0.6
    assert alpha[:side // 8].max() == 0 and alpha[-side // 8:].max() == 0
    assert alpha[side // 2 - 1:side // 2 + 2, side // 2 - 1:side // 2 + 2].max() > 150
    assert (alpha[:, :side // 2] > 12).sum() > side
    assert (alpha[:, side // 2:] > 12).sum() > side
    ink = pixels[alpha > 150, :3]
    if theme == 'dark':
        assert ink.min() == 255
    else:
        assert ink.mean() < 100
    if side == 96:
        assert alpha[43:57, 22:36].max() > 200
        assert alpha[43:57, 64:78].max() > 200


def test_native_spiral_does_not_ask_optional_font_provider(qapp, monkeypatch):
    monkeypatch.setattr(iconset, '_try_qta', lambda: pytest.fail('Native spiral must not request a font'))
    icon = iconset.app_icon('__easy_command__:Spaceout', theme='dark')
    assert not icon.isNull()
    assert icon.pixmap(96, 96).toImage() == icon.pixmap(96, 96).toImage()
    copied = QIcon(icon)
    assert np.array_equal(_pixels(icon, 96), _pixels(copied, 96))
