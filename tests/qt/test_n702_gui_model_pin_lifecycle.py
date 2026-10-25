"""Real form and snapshot boundaries preserve exact model evidence."""
import copy

import pytest
from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget

from spacr import model_zoo as zoo
from spacr.cli import load_settings_file
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.settings_model import SettingsWidgets


def payload():
    return zoo._capture_model_metadata('mask', {'src': '/laptop/plate',
        'pathogen_model_name': 'toxoplasma_pv_v4', 'segmentation_backend': 'cellpose'})


def model(qtbot, values):
    parent = QWidget()
    qtbot.addWidget(parent)
    return SettingsWidgets('mask', parent, current=values)


def test_form_rebuild_keeps_hidden_deep_copied_model_evidence(qtbot):
    original = payload()
    form = model(qtbot, original)
    assert form.collect()['model_metadata'] == original['model_metadata']
    assert 'model_metadata' not in form._widgets
    original['model_metadata']['settings'].clear()
    assert form.collect()['model_metadata']['settings']
    returned = form.collect()
    returned['model_metadata']['settings'].clear()
    assert form.collect()['model_metadata']['settings']


@pytest.mark.parametrize('value', ['cpsam', 'toxoplasma_pv_v4'])
def test_explicit_model_edit_invalidates_even_when_text_returns_to_original(qtbot, value):
    form = model(qtbot, payload())
    assert form.set_hidden_value('pathogen_model_name', value)
    assert form.collect()['model_metadata']['settings'] == {}


def test_unrelated_edit_retains_the_pin(qtbot):
    values = payload()
    form = model(qtbot, values)
    form.set_hidden_value('src', '/another/plate')
    assert form.collect()['model_metadata'] == values['model_metadata']


def test_collect_refuses_a_binding_changed_without_a_commit_signal(qtbot):
    form = model(qtbot, payload())
    form._defaults['pathogen_model_name'] = '/unannounced/edit'
    with pytest.raises(zoo.ModelZooError):
        form.collect()


def test_real_bulk_import_retains_pins_and_refuses_stale_input_before_changes(qtbot):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    values = payload()
    screen.apply_settings_dict(values)
    assert screen._settings_model.collect()['model_metadata'] == values['model_metadata']
    before = copy.deepcopy(screen._settings_model.collect())
    values['pathogen_model_name'] = 'cpsam'
    with pytest.raises(zoo.ModelZooError):
        screen.apply_settings_dict(values)
    assert screen._settings_model.collect() == before


def test_actual_export_captures_the_selected_catalogue_key(qtbot, monkeypatch, tmp_path):
    screen = AppScreen('analyze_plaques')
    qtbot.addWidget(screen)
    destination = tmp_path / 'plaque.csv'
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *args, **kwargs: (str(destination), ''))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: pytest.fail(str(args[2])))
    screen._on_export_settings()
    values = load_settings_file(str(destination))
    record = values['model_metadata']['settings']['plaque_model']
    assert record['key'] == values['plaque_model'] == 'toxoplasma_plaque_v6'
    assert record['version_status'] == 'unspecified'


def test_cluster_snapshot_is_independent_of_later_form_edits(qtbot):
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen

    screen = DistributedJobsScreen()
    qtbot.addWidget(screen)
    values = payload()
    expected = copy.deepcopy(values)
    screen.configure_submission('mask', values)
    values['model_metadata']['settings'].clear()
    assert screen._settings_snapshot == expected


def test_bulk_edit_invalidates_only_the_explicit_model_selection(qtbot):
    screen = AppScreen('mask')
    qtbot.addWidget(screen)
    values = payload()
    screen.apply_settings_dict(values)
    screen.apply_settings_dict({'src': '/changed/plate'})
    assert screen._settings_model.collect()['model_metadata'] == values['model_metadata']
    screen.apply_settings_dict({'pathogen_model_name': 'cpsam'})
    assert screen._settings_model.collect()['model_metadata']['settings'] == {}


def test_cluster_rejects_wrong_consumer_before_replacing_its_snapshot(qtbot):
    from spacr.qt.screens.distributed_jobs import DistributedJobsScreen

    screen = DistributedJobsScreen()
    qtbot.addWidget(screen)
    values = payload()
    screen.configure_submission('mask', values)
    before = copy.deepcopy(screen._settings_snapshot)
    with pytest.raises(zoo.ModelZooError):
        screen.configure_submission('classify', values)
    assert screen._settings_snapshot == before
