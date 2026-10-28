"""The provided Spaceout GIF remains unchanged and only plays on visible cards."""
import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QIcon, QImage, QImageReader, QMovie, QPainter

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
def test_static_icon_uses_actual_black_white_reference_frame(qapp, monkeypatch, side, theme):
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    pixels = _pixels(iconset.app_icon('__easy_command__:Spaceout', theme=theme), side)
    fallback = _pixels(iconset.app_icon('__unregistered_icon__', theme=theme), side)
    assert not np.array_equal(pixels, fallback)
    reader = QImageReader(str(iconset._spaceout_gif_path()))
    original = reader.read()
    assert original.pixelColor(0, 0).getRgb()[:3] == (32, 32, 32)
    source = original.scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    reference = source.convertToFormat(QImage.Format_RGBA8888)
    expected = np.frombuffer(reference.constBits(), np.uint8).reshape(side, side, 4)
    assert np.array_equal(pixels, expected)
    assert pixels[:, :, 3].min() == 255
    assert pixels[:, :, :3].max() > 100
    assert np.ptp(pixels[:, :, :3].astype(int), axis=2).max() < 10


@pytest.mark.parametrize('ratio', [1, 2])
def test_static_still_uses_the_real_paint_device_density_and_logical_center(qapp, ratio):
    canvas = QImage(90 * ratio, 65 * ratio, QImage.Format_RGBA8888)
    canvas.setDevicePixelRatio(ratio)
    canvas.fill(Qt.transparent)
    engine = iconset._SpaceoutIconEngine()
    painter = QPainter(canvas)
    try:
        engine.paint(painter, QRect(7, 5, 72, 48), QIcon.Normal, QIcon.Off)
    finally:
        painter.end()
    original = QImageReader(str(iconset._spaceout_gif_path())).read()
    assert original.width() == original.height() and not original.isNull()
    reference = original.scaled(48 * ratio, 48 * ratio,
                                Qt.KeepAspectRatio, Qt.SmoothTransformation)
    reference = reference.convertToFormat(QImage.Format_RGBA8888)
    expected = np.zeros((65 * ratio, 90 * ratio, 4), np.uint8)
    expected[5 * ratio:53 * ratio, 19 * ratio:67 * ratio] = np.frombuffer(
        reference.constBits(), np.uint8).reshape(48 * ratio, 48 * ratio, 4)
    actual = np.frombuffer(canvas.constBits(), np.uint8).reshape(65 * ratio, 90 * ratio, 4)
    np.testing.assert_array_equal(actual, expected)


def test_native_still_does_not_ask_optional_font_provider(qapp, monkeypatch):
    monkeypatch.setattr(iconset, '_try_qta', lambda: pytest.fail('GIF must not request a font'))
    icon = iconset.app_icon('__easy_command__:Spaceout', theme='dark')
    assert not icon.isNull()
    assert icon.pixmap(96, 96).toImage() == icon.pixmap(96, 96).toImage()
    assert np.array_equal(_pixels(icon, 96), _pixels(QIcon(icon), 96))


def test_spaceout_card_plays_uncached_movie_only_while_visible(qtbot, private_ai_profile, monkeypatch):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets import home

    preferences, _, _ = private_ai_profile
    preferences.set_reduce_motion(False)
    monkeypatch.setattr(home, 'device_ratio', lambda widget: 2.0)
    parent = QWidget()
    qtbot.addWidget(parent)
    parent.resize(300, 300)
    card = home._EasySuggestionCard(('__easy_command__:Spaceout', 'Spaceout', '', ''), parent)
    card.resize(224, 224)
    movie = card._spaceout_movie
    assert movie.cacheMode() == QMovie.CacheNone and movie.state() == QMovie.NotRunning
    assert Path(movie.fileName()) == iconset._spaceout_gif_path()
    parent.show()
    qtbot.waitUntil(lambda: movie.state() == QMovie.Running)
    assert movie.scaledSize().width() == 448
    assert card.content.isHidden() and not card.title.isVisible()
    before = card._spaceout_display.toImage()
    qtbot.waitUntil(lambda: movie.currentFrameNumber() >= 2)
    assert card._spaceout_display.toImage() != before
    assert movie.currentPixmap().width() == 448
    assert card._spaceout_display.deviceIndependentSize().width() == 224
    card.hide()
    assert movie.state() == QMovie.NotRunning
    frame = movie.currentFrameNumber()
    qtbot.wait(120)
    assert movie.currentFrameNumber() == frame
    preferences.set_reduce_motion(True)
    card.show()
    assert movie.state() == QMovie.NotRunning
    still = card._spaceout_display.toImage()
    qtbot.wait(120)
    assert card._spaceout_display.toImage() == still
    preferences.set_reduce_motion(False)
    card._sync_run_state()
    assert movie.state() == QMovie.Running
    target = QWidget(parent)
    target.resize(280, 280)
    target.show()
    card.setParent(target)
    card.show()
    assert card._spaceout_movie is movie and movie.state() == QMovie.Running
    other = home._EasySuggestionCard(('measure', 'Measure', '', ''), parent)
    assert other._spaceout_movie is None and not other.findChildren(QMovie)
    parent.close()
    assert movie.state() == QMovie.NotRunning


def test_reference_asset_is_unchanged_and_explicitly_packaged():
    root = Path(__file__).resolve().parents[2]
    source = iconset._spaceout_gif_path()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        '5dfae77c1ecee2736861b5ad129d12912ebb202fb2d72177858eb9b3f8d42b94')
    tree = ast.parse((root / 'setup.py').read_text())
    declarations = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                    for keyword in node.keywords if keyword.arg == 'package_data']
    assert len(declarations) == 1
    patterns = ast.literal_eval(declarations[0])['spacr']
    assert 'resources/icons/spaceout.gif' in patterns
    assert 'resources/icons/*.gif' not in patterns
    manifest = (root / 'MANIFEST.in').read_text()
    assert 'recursive-include spacr/resources/icons *' in manifest
    assert 'exclude spacr/resources/icons/spaceout.gif' not in manifest


def test_native_spaceout_card_fill_matches_gif_on_hover_and_press(qtbot, private_ai_profile):
    from PySide6.QtCore import QPoint

    from spacr.qt.theme import STAGE_HOVER
    from spacr.qt.widgets.home import _EasyHomePage

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(True)
    preferences.set_reduce_motion(True)
    page = _EasyHomePage()
    qtbot.addWidget(page)
    page.resize(1000, 900)
    page.show()
    page.field.set_cards([('__easy_command__:Spaceout', 'Spaceout', '', ''),
                          ('measure', 'Measure', '', '')])
    page.field._finish_growth()
    card, ordinary = page.field.cards
    qtbot.mouseMove(page, QPoint(0, 0))
    qtbot.wait(10)
    assert card.content.isHidden() and not card.title.isVisible() and not card.icon.isVisible()
    assert card.accessibleName() == 'Spaceout'
    def pixels(widget):
        image = widget.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
        return np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.width(), 4).copy()
    normal = pixels(card)
    assert tuple(pixels(ordinary)[20, 20, :3]) == (25, 25, 25)
    assert (normal[20:-20, 12:40, :3] > 60).any()
    assert normal[0, 0, 3] == 0
    qtbot.mouseMove(card, QPoint(20, 20))
    qtbot.wait(30)
    hovered = pixels(card)
    assert np.array_equal(normal[20:-20, 20:-20], hovered[20:-20, 20:-20])
    from PySide6.QtGui import QColor

    expected = QColor(STAGE_HOVER[card._stage]).getRgb()[:3]
    border = card.grab().toImage().pixelColor(0, card.height() // 2).getRgb()[:3]
    assert border == expected
    qtbot.mousePress(card, Qt.LeftButton, pos=QPoint(20, 20))
    assert np.array_equal(hovered[20:-20, 20:-20], pixels(card)[20:-20, 20:-20])
    qtbot.mouseRelease(card, Qt.LeftButton, pos=QPoint(20, 20))
    activated = []
    card.activated.connect(lambda key, path: activated.append((key, path)))
    card.setFocus()
    qtbot.keyClick(card, Qt.Key_Return)
    assert activated == [('__easy_command__:Spaceout', '')]
    page.close()
