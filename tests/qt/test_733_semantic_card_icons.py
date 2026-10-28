"""Actual card catalogs retain semantic artwork without optional icon fonts."""
import ast
import importlib
import inspect
import textwrap
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QImage, QPainter

from spacr.qt import app, iconset
from spacr.qt.organisms import ORGANISMS
from spacr.qt.screens.map_barcodes import FOLD_FALLBACK, FOLD_HOST_MODULES
from spacr.qt.widgets.home import _EasySuggestionCard, _EasyWorkflowNode
from tests.qt.test_733_help_native_routes import _help_actions
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


def _raster(icon):
    assert icon is not None and not icon.isNull()
    image = QImage(96, 96, QImage.Format_RGBA8888)
    image.fill(Qt.transparent)
    painter = QPainter(image)
    try:
        icon.paint(painter, QRect(0, 0, 96, 96))
    finally:
        painter.end()
    pixels = bytes(image.constBits())
    assert any(pixels[3::4])
    return pixels


def _actual_keys():
    keys = {row[0] for row in app.APPS}
    keys.update(key for children in app.folded_children().values() for key in children)
    pages = {key for module in FOLD_HOST_MODULES.values()
             for key in getattr(importlib.import_module('spacr.qt.screens.' + module), 'PAGE_FOLDS', ())}
    assert {'spatial_masks', 'spatial_map'} <= pages
    keys.update(pages)
    keys.update(FOLD_FALLBACK)
    keys.update(row[0] for row in app._HELP_MODULES)
    keys.update(('overview', '3d', 'feature_dict'))
    return keys


def _navigation_labels():
    tree = ast.parse(textwrap.dedent(inspect.getsource(app.MainWindow._easy_navigation_suggestions)))
    return next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'labels' for target in node.targets))


@pytest.mark.parametrize('theme', ['dark', 'light'])
def test_all_actual_card_keys_render_semantic_pixels_without_qtawesome(
        qtbot, private_ai_profile, monkeypatch, theme):
    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    preferences.set_reduce_motion(True)
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    window = app.MainWindow(initial_app='__home__')
    qtbot.addWidget(window)
    window._tour_timer.stop()
    window._consent_timer.stop()
    try:
        keys = _actual_keys()
        actions = [action for action in _help_actions(window)
                   if action.isEnabled() and action.isVisible() and not action.isSeparator()]
        labels = [*_navigation_labels(), *[action.text().split('\t', 1)[0].replace('&', '')
                                          for action in actions]]
        for label in _navigation_labels():
            assert window._easy_navigation_suggestions(label), label
        synthetic = set()
        for label in labels:
            rows = window._easy_navigation_suggestions(label)
            keys.update(row[0] for row in rows)
            synthetic.update(row[0] for row in rows if row[0].startswith('__easy_command__:'))
        assert {'__easy_command__:Preferences', '__easy_command__:About'} <= synthetic
        defaults = {_raster(iconset.app_icon('__unknown_catalog_entry__', theme=theme)),
                    _raster(iconset.app_icon('default', theme=theme))}
        missing = []
        for key in sorted(keys):
            pixels = _raster(app._icon_for_app(key, theme=theme))
            if pixels in defaults:
                missing.append(key)
        if missing:
            pytest.fail('Generic icons in the actual card catalog:\n' + '\n'.join(missing))
        for organism in ORGANISMS.values():
            for key, _title, _description, artwork in organism['modules']:
                icon = (app._icon_for_app(artwork, theme=theme) if key else
                        iconset.app_icon(artwork, override=f'organism_{artwork}.svg', theme=theme))
                assert _raster(icon) not in defaults, artwork
    finally:
        window.close()


@pytest.mark.parametrize('key', ['overview', '3d', '__easy_command__:Preferences'])
def test_real_cards_and_network_nodes_keep_their_own_semantic_icons(qtbot, monkeypatch, key):
    monkeypatch.setattr(iconset, '_try_qta', lambda: None)
    expected = _raster(app._icon_for_app(key, theme='dark'))
    card = _EasySuggestionCard((key, 'Tool', 'Tool details', ''))
    qtbot.addWidget(card)
    assert _raster(card._module_icon) == expected
    node = _EasyWorkflowNode(SimpleNamespace(_node_side=192), key)
    assert _raster(node._glyph) == expected
    if key in ('overview', '3d'):
        assert expected != _raster(app._icon_for_app('mask', theme='dark'))
    alpha_key = next(key for key, stage in app.APP_STAGE.items() if stage == 'alpha')
    alpha_card = _EasySuggestionCard((alpha_key, 'Alpha tool', 'Tool details', ''))
    qtbot.addWidget(alpha_card)
    assert alpha_card._stage == 'alpha'
    assert not alpha_card.alpha.isHidden()
