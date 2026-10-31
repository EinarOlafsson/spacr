"""ParaScape cards use the separate application's bundled canonical artwork."""
import ast
import hashlib
from pathlib import Path

import pytest
from PySide6.QtGui import QIcon

from spacr.qt import app, iconset
from spacr.qt.organisms import ORGANISMS
from spacr.qt.widgets.home import _EasySuggestionCard
from tests.qt.test_733_semantic_card_icons import _raster
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile as private_ai_profile


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_parascape_native_card_uses_canonical_artwork_without_optional_fonts(
        qtbot, private_ai_profile, monkeypatch, theme):
    preferences, _, _ = private_ai_profile
    preferences.set_show_alpha(True)
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    path = Path(iconset.bundled_icon_path('parascape'))
    assert path.name == 'parascape.svg'
    expected = _raster(QIcon(iconset.themed_pixmap(str(path), theme)))
    actual = _raster(app._icon_for_app('parascape', theme=theme))
    assert actual == expected
    assert actual != _raster(iconset.app_icon('default', theme=theme))
    assert actual != _raster(iconset.app_icon('replication', theme=theme))
    row = next(row for row in app._easy_module_search('parascape') if row[0] == 'parascape')
    card = _EasySuggestionCard(row)
    qtbot.addWidget(card)
    card.show()
    assert _raster(card._module_icon) == _raster(app._icon_for_app('parascape', theme='dark'))
    assert card.icon.pixmap() is not None and not card.icon.pixmap().isNull()
    assert next(row[-1] for row in ORGANISMS['toxoplasma']['modules']
                if row[0] == 'parascape') == 'parascape'


def test_parascape_canonical_artwork_and_its_license_ship_in_packages():
    root = Path(__file__).resolve().parents[2]
    asset = root / 'spacr/resources/icons/parascape.svg'
    assert hashlib.sha256(asset.read_bytes()).hexdigest() == '08d41740a6013ef887bfcd993a045015290f7c4f6b41d3a99a326fffdf24b1f8'
    tree = ast.parse((root / 'setup.py').read_text())
    declarations = [keyword.value for node in ast.walk(tree) if isinstance(node, ast.Call)
                    for keyword in node.keywords if keyword.arg == 'package_data']
    patterns = ast.literal_eval(declarations[0])['spacr']
    assert 'resources/icons/parascape.svg' in patterns
    assert 'resources/icons/parascape-dark.svg' in patterns
    assert 'resources/icons/parascape-LICENSE.txt' in patterns
    license_text = (asset.parent / 'parascape-LICENSE.txt').read_text()
    assert 'MIT License' in license_text and 'Copyright (c) 2026 Einar Olafsson' in license_text
    assert 'recursive-include spacr/resources/icons *' in (root / 'MANIFEST.in').read_text()


def test_parascape_dark_pixels_are_white_with_transparent_surroundings(qapp):
    """The shared renderer exposes white logo geometry instead of a paper tile."""
    import xml.etree.ElementTree as ET

    import numpy as np
    from PySide6.QtGui import QImage

    canonical = Path(iconset.bundled_icon_path('parascape'))
    dark_asset = canonical.with_name('parascape-dark.svg')
    original_tree = ET.fromstring(canonical.read_bytes())
    dark_tree = ET.fromstring(dark_asset.read_bytes())
    assert not any(node.tag.endswith('}rect') for node in dark_tree.iter())

    def geometry(tree):
        """Compare all outline, star and transform attributes independently of ink."""
        return [(node.tag, {key: value for key, value in node.attrib.items()
                            if key not in {'fill', 'stroke'}})
                for node in tree.iter() if not node.tag.endswith('}rect')]

    assert geometry(original_tree) == geometry(dark_tree)
    assert original_tree[0].text == dark_tree[0].text
    image = iconset.themed_qimage(str(canonical), 'dark').convertToFormat(QImage.Format_RGBA8888)
    pixels = np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        image.height(), image.bytesPerLine())[:, :image.width() * 4].reshape(
            image.height(), image.width(), 4)
    alpha = pixels[:, :, 3]
    assert not alpha[0, 0] and not alpha[0, -1] and not alpha[-1, 0] and not alpha[-1, -1]
    assert 100 < np.count_nonzero(alpha) < alpha.size // 4
    opaque = alpha == 255
    assert opaque.any() and np.all(pixels[opaque, :3] == 255)
    light = iconset.themed_qimage(str(canonical), 'light')
    assert light.pixelColor(light.width() // 10, light.height() // 2).alpha() == 255
    assert image.pixelColor(image.width() // 10, image.height() // 2).alpha() == 0
