"""The real Activation Maps form preserves explicit local checkpoint identity."""
import json

import pytest
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QFileDialog, QFormLayout, QLineEdit, QWidget


KEYS = ('counterfactual_checkpoint', 'counterfactual_checkpoint_sha256')
NAMES = ('ActivationCounterfactualCheckpoint', 'ActivationCounterfactualCheckpointSHA256')
PIN = 'aB09' * 16


@pytest.fixture
def private_preferences(tmp_path, monkeypatch):
    from spacr.qt import preferences

    path = tmp_path / 'preferences.ini'
    monkeypatch.setattr(preferences, '_settings',
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def test_checkpoint_controls_are_in_the_actual_alpha_heading(qtbot, private_preferences):
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
        fields = [heading._form.itemAt(index, QFormLayout.FieldRole).widget()
                  for index in range(heading._form.rowCount())]
        for key in KEYS:
            editor = model._widgets[key]
            assert any(field is editor or getattr(field, '_spacr_field', None) is editor
                       for field in fields)
            assert not screen.setting_row_is_visible(key)
        private_preferences._set_show_alpha_features(True)
        screen._refresh_alpha_visibility()
        for key, name in zip(KEYS, NAMES):
            assert screen.setting_row_is_visible(key)
            assert screen.findChild(QLineEdit, name) is model._widgets[key]
            assert model._widgets[key].property('settingKey') == key
            assert model._widgets[key].property('apiTooltipDescriptionSource')
        model.set_value_for_key(KEYS[0], '/explicit/checkpoint.pt')
        model.set_value_for_key(KEYS[1], PIN)
        private_preferences._set_show_alpha_features(False)
        screen._refresh_alpha_visibility()
        assert not any(screen.setting_row_is_visible(key) for key in KEYS)
        assert [model.collect()[key] for key in KEYS] == ['/explicit/checkpoint.pt', PIN]
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()


@pytest.mark.parametrize('lazy', [False, True], ids=['eager', 'lazy'])
def test_blank_defaults_and_explicit_identity_survive_json_roundtrip(qtbot, lazy, tmp_path):
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets('activation', parent=owner)
    if lazy:
        model.categories_may_wait = lambda _title, _keys: True
    model.build_sections()
    if lazy:
        assert all(not model._widgets.is_built(key) for key in KEYS)
    defaults = model.collect()
    assert [defaults[key] for key in KEYS] == ['', '']
    default_generator = defaults['counterfactual_generator']
    if lazy:
        assert all(not model._widgets.is_built(key) for key in KEYS)
    path = str(tmp_path / 'not opened checkpoint.pt')
    for key, name, value in zip(KEYS, NAMES, (path, PIN)):
        editor = model._widgets[key]
        assert isinstance(editor, QLineEdit)
        assert editor.objectName() == name
        assert model.set_value_for_key(key, value)
        assert editor.text() == value
    saved = json.loads(json.dumps(model.collect()))
    assert saved['counterfactual_generator'] == default_generator
    restored = SettingsWidgets('activation', parent=owner, current=saved)
    if lazy:
        restored.categories_may_wait = lambda _title, _keys: True
    restored.build_sections()
    assert [restored.collect()[key] for key in KEYS] == [path, PIN]
    if lazy:
        assert all(not restored._widgets.is_built(key) for key in KEYS)
    for key in KEYS:
        assert restored._widgets[key].text() == saved[key]
        assert restored.set_value_for_key(key, '')
    assert [restored.collect()[key] for key in KEYS] == ['', '']
    assert not (tmp_path / 'not opened checkpoint.pt').exists()


@pytest.mark.parametrize('chosen', ['', '/explicit/new checkpoint.pt'], ids=['cancel', 'choose'])
def test_local_chooser_commits_only_the_selected_path(qtbot, monkeypatch, chosen):
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets('activation', parent=owner, current={
        KEYS[0]: '/old/checkpoint.pt', KEYS[1]: PIN,
        'counterfactual_generator': 'autoencoder',
    })
    model.build_sections()
    editor = model._widgets[KEYS[0]]
    calls = []
    commits = []
    model._enable_commit_observation(commits.append)

    def choose_file(parent, title, initial):
        calls.append((parent, title, initial))
        return chosen, ''

    monkeypatch.setattr(QFileDialog, 'getOpenFileName', choose_file)
    actions = editor.actions()
    assert len(actions) == 1
    actions[0].trigger()
    assert len(calls) == 1
    assert calls[0][0] is editor
    assert calls[0][2] == '/old/checkpoint.pt'
    assert editor.text() == (chosen or '/old/checkpoint.pt')
    assert commits == ([KEYS[0]] if chosen else [])
    saved = model.collect()
    assert saved[KEYS[1]] == PIN
    assert saved['counterfactual_generator'] == 'autoencoder'


def test_imported_invalid_identity_is_not_silently_dropped_or_replaced(qtbot):
    from spacr.qt.screens.settings_model import SettingsWidgets

    owner = QWidget()
    qtbot.addWidget(owner)
    current = {KEYS[0]: '/missing/file.pt', KEYS[1]: 'not-a-sha',
               'counterfactual_generator': 'autoencoder'}
    model = SettingsWidgets('activation', parent=owner, current=current)
    model.build_sections()
    assert {key: model.collect()[key] for key in current} == current

