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
    assert 'resources/icons/parascape-LICENSE.txt' in patterns
    license_text = (asset.parent / 'parascape-LICENSE.txt').read_text()
    assert 'MIT License' in license_text and 'Copyright (c) 2026 Einar Olafsson' in license_text
    assert 'recursive-include spacr/resources/icons *' in (root / 'MANIFEST.in').read_text()
