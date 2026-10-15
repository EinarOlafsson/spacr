"""Diffusion controls use the real alpha-gated Activation Maps settings form."""
import json

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QDoubleSpinBox, QFormLayout, QWidget

KEYS = ('counterfactual_strength', 'counterfactual_structure_weight')
NAMES = ('ActivationCounterfactualStrength', 'ActivationCounterfactualStructureWeight')


@pytest.fixture
def private_preferences(tmp_path, monkeypatch):
    from spacr.qt import preferences

    path = tmp_path / 'preferences.ini'
    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_diffusion_controls_are_in_the_actual_counterfactuals_heading(
        qtbot, private_preferences):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus

    screen = AppScreen('activation')
    qtbot.addWidget(screen)
    try:
        screen._open_the_heading_of('counterfactuals')
        screen._refresh_alpha_visibility()
        model = screen._settings_model
        heading = next(section for section in screen.rendered_settings_sections()
                       if section.property('settingsCategorySource') == 'Counterfactuals')
        for key in KEYS:
            editor = model._widgets[key]
            fields = [heading._form.itemAt(index, QFormLayout.FieldRole).widget()
                      for index in range(heading._form.rowCount())]
            assert any(field is editor or getattr(field, '_spacr_field', None) is editor
                       for field in fields)
            assert not screen.setting_row_is_visible(key)
        private_preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        for key, name in zip(KEYS, NAMES):
            editor = model._widgets[key]
            assert screen.setting_row_is_visible(key)
            assert screen.findChild(QDoubleSpinBox, name) is editor
            assert editor.property('settingKey') == key
            assert editor.property('apiTooltipDescriptionSource')
        assert model.set_value_for_key(KEYS[0], 0.0)
        assert model.set_value_for_key(KEYS[1], 0.35)
        private_preferences._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in KEYS)
        saved = model.collect()
        assert saved[KEYS[0]] == 0.0
        assert saved[KEYS[1]] == 0.35
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()


@pytest.mark.parametrize('lazy', [False, True], ids=['eager', 'lazy'])
def test_float_controls_keep_defaults_and_json_roundtrip(qtbot, lazy):
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets('activation', parent=owner)
    if lazy:
        model.categories_may_wait = lambda _title, _keys: True
    model.build_sections()
    if lazy:
        assert all(not model._widgets.is_built(key) for key in KEYS)
    for key, name, value in zip(KEYS, NAMES, (0.95, 0.0)):
        editor = model._widgets[key]
        assert isinstance(editor, QDoubleSpinBox)
        assert editor.objectName() == name
        assert editor.value() == value
        assert isinstance(model.collect()[key], float)
    assert model.set_value_for_key(KEYS[0], 0.0)
    assert model.set_value_for_key(KEYS[1], 0.125)
    saved = json.loads(json.dumps(model.collect()))
    restored = SettingsWidgets('activation', parent=owner, current=saved)
    restored.build_sections()
    for key, value in zip(KEYS, (0.0, 0.125)):
        assert restored._widgets[key].value() == value
        assert restored.collect()[key] == value
