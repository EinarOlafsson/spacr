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
    source = reader.read().scaled(side, side, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    reference = source.convertToFormat(QImage.Format_RGBA8888)
    expected = np.frombuffer(reference.constBits(), np.uint8).reshape(side, side, 4)
    assert np.array_equal(pixels, expected)
    assert pixels[:, :, 3].min() == 255
    assert pixels[0, 0, :3].max() < 10
    assert pixels[:, :, :3].max() > 100
    assert np.ptp(pixels[:, :, :3].astype(int), axis=2).max() < 10


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
    assert movie.scaledSize().width() == 192
    assert Path(movie.fileName()) == iconset._spaceout_gif_path()
    parent.show()
    qtbot.waitUntil(lambda: movie.state() == QMovie.Running)
    before = card.icon.pixmap().toImage()
    qtbot.waitUntil(lambda: movie.currentFrameNumber() >= 2)
    assert card.icon.pixmap().toImage() != before
    assert movie.currentPixmap().width() == 192
    assert card.icon.pixmap().deviceIndependentSize().width() == 96
    card.hide()
    assert movie.state() == QMovie.NotRunning
    frame = movie.currentFrameNumber()
    qtbot.wait(120)
    assert movie.currentFrameNumber() == frame
    preferences.set_reduce_motion(True)
    card.show()
    assert movie.state() == QMovie.NotRunning
    still = card.icon.pixmap().toImage()
    qtbot.wait(120)
    assert card.icon.pixmap().toImage() == still
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
        '1eaf358e2859ee4527633267652d9c9b65770d65bb0b3d68b02c358dff2fe9a4')
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
