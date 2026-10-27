"""Integer counts reveal configured organelles progressively in the live form."""
import pytest
from PySide6.QtWidgets import QComboBox, QLineEdit, QWidget

from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import SettingsWidgets, object_of_setting
from spacr.organelle_types import organelle_role, organelle_role_of
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile


@pytest.fixture
def mask(qtbot, private_ai_profile):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    return screen


def _commit(screen, key, value):
    widget = screen._settings_model._widgets[key]
    if isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, QComboBox):
        widget.setCurrentIndex(widget.findData(value))
    else:
        widget.set_value(value)
    widget.editingFinished.emit()


def _slot_keys(model):
    return {key for key in model._widgets if organelle_role_of(key)}


@pytest.mark.parametrize('count_value', [10, 702])
def test_count_ten_builds_only_first_channel_then_reveals_numbered_slot(mask, count_value):
    model = mask._settings_model
    count = model._widgets['number_of_organelles']
    assert isinstance(count, QLineEdit)
    assert not isinstance(count, QComboBox)
    _commit(mask, 'number_of_organelles', count_value)
    assert _slot_keys(model) == {'organelle_channel'}
    channel = model._widgets['organelle_channel']
    _commit(mask, 'organelle_channel', 0)
    keys = _slot_keys(model)
    assert 'organelleb_channel' in keys
    assert 'organelle_type' in keys
    assert 'organellec_channel' not in keys
    assert not any(key.startswith('organelleb_') and key != 'organelleb_channel' for key in keys)
    assert model.collect()['organelle_channel'] == 0
    assert model._widgets['organelle_channel'] is channel
    assert mask._settings_model is model
    _commit(mask, 'organelleb_channel', 2)
    assert 'organellec_channel' in model._widgets
    assert 'organelleb_type' in model._widgets
    assert 'organelled_channel' not in model._widgets
    from spacr.qt.widgets.section import Section, _sections_below
    sections = [section for section in _sections_below(mask) if isinstance(section, Section)]
    for title in ('Organelle Segmentation', 'Organelle Segmentation (advanced)', 'Additional Settings'):
        category = next(section for section in sections
                        if section.property('settingsCategorySource') == title)
        names = {section.property('settingsCategorySource') for section in _sections_below(category)
                 if isinstance(section, Section)}
        assert {'Organelle 1', 'Organelle 2'} <= names


def test_clear_lower_restore_preserves_existing_widgets_and_values(mask):
    model = mask._settings_model
    _commit(mask, 'number_of_organelles', 10)
    _commit(mask, 'organelle_channel', 0)
    _commit(mask, 'organelleb_channel', 2)
    model.set_value_for_key('organelleb_diameter', 9)
    second = model._widgets['organelleb_diameter']
    _commit(mask, 'organelle_channel', '')
    assert 'organelle_type' in model._hidden_by_their_object
    assert 'organelleb_type' not in model._hidden_by_their_object
    _commit(mask, 'number_of_organelles', 1)
    assert 'organelleb_type' in model._hidden_by_their_object
    _commit(mask, 'number_of_organelles', 10)
    assert 'organelleb_type' not in model._hidden_by_their_object
    assert model._widgets['organelleb_diameter'] is second
    assert model.collect()['organelleb_diameter'] == 9


@pytest.mark.parametrize('value', ['-1', '1.5', 'not a number', '703'])
def test_invalid_count_preserves_text_and_does_not_spawn_or_mutate(mask, value):
    model = mask._settings_model
    _commit(mask, 'number_of_organelles', 10)
    before = set(model._widgets)
    _commit(mask, 'number_of_organelles', value)
    assert model._widgets['number_of_organelles'].text() == value
    assert set(model._widgets) == before
    assert model._setting_value('number_of_organelles') == 10
    with pytest.raises(ValueError, match='integer from 0 to 702'):
        model.collect()


def test_imported_populated_slots_build_only_relevant_numbered_groups(qtbot, private_ai_profile):
    current = {'number_of_organelles': 10, 'organelle_channel': 0,
               'organelleb_channel': 2, 'organelleb_diameter': 9}
    owner = QWidget()
    qtbot.addWidget(owner)
    model = SettingsWidgets('mask', parent=owner, current=current)
    sections = model.build_sections()
    assert 'organellec_channel' in model._widgets
    assert 'organellec_type' not in model._widgets
    assert 'organelled_channel' not in model._widgets
    flat = {node.title: node for section in sections for node in section.walk()}
    for title in ('Organelle Segmentation', 'Organelle Segmentation (advanced)', 'Additional Settings'):
        category = flat[title]
        numbered = [child for child in category.children if child.title.startswith('Organelle ')]
        assert numbered
        for child in numbered:
            roles = {object_of_setting(key) for key, widget in model._widgets.items()
                     if any(widget is field for _label, field in child.rows)}
            assert len(roles) == 1
            assert roles <= {organelle_role(1), organelle_role(2)}


def test_saved_mapping_applies_progressively_without_reloading(mask, monkeypatch):
    model = mask._settings_model
    monkeypatch.setattr(mask, '_rebuild_the_form', lambda *args: pytest.fail('Reloaded the form'))
    mask.apply_settings_dict({'number_of_organelles': 10,
                              'organelle_channel': 0, 'organelleb_channel': 2,
                              'organelleb_diameter': 9})
    assert mask._settings_model is model
    assert model.collect()['organelleb_diameter'] == 9
    assert 'organellec_channel' in model._widgets
    assert 'organellec_type' not in model._widgets
    assert 'organelled_channel' not in model._widgets


def test_sparse_import_exposes_its_configured_slot_without_building_unused_slots(qtbot, private_ai_profile):
    owner = QWidget()
    qtbot.addWidget(owner)
    tenth = organelle_role(10)
    model = SettingsWidgets('mask', parent=owner, current={
        'number_of_organelles': 10, f'{tenth}_channel': 0, f'{tenth}_diameter': 12})
    model.build_sections()
    assert f'{tenth}_type' in model._widgets
    assert 'organelle_channel' in model._widgets
    assert 'organelle_type' not in model._widgets
    assert 'organelleb_channel' not in model._widgets
    assert f'{organelle_role(9)}_channel' not in model._widgets
    assert model.collect()[f'{tenth}_diameter'] == 12


def test_measure_progresses_through_mask_planes_without_raw_channels(qtbot, private_ai_profile):
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    model = screen._settings_model
    _commit(screen, 'number_of_organelles', 10)
    assert _slot_keys(model) == {'organelle_mask_dim'}
    _commit(screen, 'organelle_mask_dim', 0)
    assert 'organelleb_mask_dim' in model._widgets
    assert 'organelle_type' in model._widgets
    assert 'organelle_channel' not in model._widgets
    assert 'organelleb_type' not in model._widgets
    _commit(screen, 'organelleb_mask_dim', 6)
    assert 'organellec_mask_dim' in model._widgets
    assert 'organellec_type' not in model._widgets
    assert 'organelled_mask_dim' not in model._widgets
    _commit(screen, 'number_of_organelles', 1)
    _commit(screen, 'number_of_organelles', 10)
    assert model.collect()['organelleb_mask_dim'] == 6
    assert screen._settings_model is model


def test_imported_measure_mask_plane_keeps_its_settings_reachable(qtbot, private_ai_profile):
    screen = AppScreen('measure')
    qtbot.addWidget(screen)
    model = screen._settings_model
    screen.apply_settings_dict({'number_of_organelles': 10,
                                'organelle_mask_dim': 0, 'organelleb_mask_dim': 6,
                                'organelleb_min_area': 12})
    assert screen._settings_model is model
    assert 'organelleb_type' in model._widgets
    assert 'organellec_mask_dim' in model._widgets
    assert 'organelled_mask_dim' not in model._widgets
    assert 'organelle_channel' not in model._widgets
    assert model.collect()['organelleb_min_area'] == 12
    assert 'organelleb_type' not in model._hidden_by_their_object
