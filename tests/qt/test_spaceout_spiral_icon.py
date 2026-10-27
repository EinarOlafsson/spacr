"""The Spaceout command keeps its white top-down spiral without icon fonts."""
import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QIcon, QImage, QPainter

from spacr.qt import iconset
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


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
def test_spaceout_spiral_has_theme_ink_transparency_and_top_down_turns(
        qapp, monkeypatch, side, theme):
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    pixels = _pixels(iconset.app_icon('__easy_command__:Spaceout', theme=theme), side)
    fallback = _pixels(iconset.app_icon('__unregistered_icon__', theme=theme), side)
    assert not np.array_equal(pixels, fallback)
    alpha = pixels[:, :, 3]
    assert 0.1 < (alpha > 12).mean() < 0.6
    assert alpha[:max(1, side // 16), :max(1, side // 16)].max() == 0
    assert alpha[-max(1, side // 16):, -max(1, side // 16):].max() == 0
    assert alpha[side // 2 - 1:side // 2 + 2, side // 2 - 1:side // 2 + 2].max() > 100
    assert (alpha[:, :side // 2] > 12).sum() > side
    assert (alpha[:, side // 2:] > 12).sum() > side
    ink = pixels[alpha > 150, :3]
    if theme == 'dark':
        assert ink.min() == 255
    else:
        assert ink.mean() < 100
    if side == 96:
        rows, columns = np.where(alpha > 100)
        assert rows.max() - rows.min() > 64
        assert columns.max() - columns.min() > 64
        assert (np.diff((alpha[48] > 100).astype(int)) == 1).sum() >= 4


def test_native_spiral_does_not_ask_optional_font_provider(qapp, monkeypatch):
    monkeypatch.setattr(iconset, '_try_qta', lambda: pytest.fail('Native spiral must not request a font'))
    icon = iconset.app_icon('__easy_command__:Spaceout', theme='dark')
    assert not icon.isNull()
    assert icon.pixmap(96, 96).toImage() == icon.pixmap(96, 96).toImage()
    copied = QIcon(icon)
    assert np.array_equal(_pixels(icon, 96), _pixels(copied, 96))


def test_spaceout_card_rotates_cached_icon_only_while_visible(qtbot, private_ai_profile):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.home import _EasySuggestionCard

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(False)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(300, 300)
    card = _EasySuggestionCard(('__easy_command__:Spaceout', 'Spaceout', '', ''), parent)
    card.resize(224, 224)
    parent.show()
    qtbot.wait(10)
    assert card._spaceout_timer.isActive()
    before = card.icon.pixmap().toImage()
    qtbot.wait(120)
    assert card.icon.pixmap().toImage() != before
    assert card.icon.size() == card.icon.sizeHint()
    cached = card._spaceout_picture.cacheKey()
    card.hide()
    assert not card._spaceout_timer.isActive()
    preferences.set_reduce_motion(True)
    card.show()
    assert not card._spaceout_timer.isActive()
    assert card.icon.pixmap().cacheKey() == cached
    preferences.set_reduce_motion(False)
    card._sync_run_state()
    assert card._spaceout_timer.isActive()
    target = QWidget(parent)
    target.resize(280, 280)
    target.show()
    card.setParent(target)
    card.show()
    assert card._spaceout_timer.isActive()
    assert card._spaceout_picture.cacheKey() == cached
    other = _EasySuggestionCard(('measure', 'Measure', '', ''), parent)
    assert other._spaceout_timer is None
    parent.close()
