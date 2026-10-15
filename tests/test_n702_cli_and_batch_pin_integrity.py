"""CLI and queue edits preserve exact pins and reject invalid imported evidence."""
from copy import deepcopy
import json

import pytest

from spacr import batch, cli


@pytest.fixture
def inputs(monkeypatch):
    module = cli.Module('mask', 'test module', 'spacr.fake:run', None, 'mask')
    defaults = {'cell_model_name': '/models/cell7', 'nucleus_model_name': '/models/nucleus7',
                'src': '/data', 'segmentation_backend': 'cellpose'}
    monkeypatch.setattr(cli, 'module_defaults', lambda _module: deepcopy(defaults))
    monkeypatch.setattr(batch, 'module_defaults', lambda _module: deepcopy(defaults))
    monkeypatch.setattr(batch, 'resolve_module', lambda _name: module)
    settings = deepcopy(defaults)
    settings['model_metadata'] = {'schema_version': 1, 'settings': {
        name: {'requested_value': settings[name], 'key': name + '_r7',
               'version': '01', 'version_status': 'declared', 'sha256': '7' * 64,
               'kind': 'cellpose', 'filename': name + '_r7.CP_model'}
        for name in ('cell_model_name', 'nucleus_model_name')}}
    return module, settings


@pytest.mark.parametrize('route', ['cli', 'batch_file', 'batch_inline'])
def test_explicit_repeated_model_override_invalidates_only_its_pin(inputs, tmp_path, route):
    module, supplied = inputs
    original = deepcopy(supplied)
    overrides = ['cell_model_name=/models/other', 'cell_model_name=/models/cell7']
    if route == 'batch_inline':
        resolved = batch.resolve_job_settings(batch.Job(module='mask', settings=supplied, overrides=overrides))
    else:
        path = tmp_path / 'settings.json'
        path.write_text(json.dumps(supplied))
        resolved = (cli.resolve_settings(module, str(path), overrides) if route == 'cli'
                    else batch.resolve_job_settings(batch.Job(module='mask', settings=str(path), overrides=overrides)))
    assert resolved['cell_model_name'] == supplied['cell_model_name']
    assert 'cell_model_name' not in resolved['model_metadata']['settings']
    assert resolved['model_metadata']['settings']['nucleus_model_name'] == supplied['model_metadata']['settings']['nucleus_model_name']
    assert supplied == original


@pytest.mark.parametrize('route', ['cli', 'batch_file', 'batch_inline'])
def test_a_stale_import_is_rejected_before_an_override_can_make_it_match(inputs, tmp_path, route):
    module, supplied = inputs
    supplied['model_metadata']['settings']['cell_model_name']['requested_value'] = '/models/stale'
    original = deepcopy(supplied)
    overrides = ['cell_model_name=/models/stale']
    with pytest.raises(cli.SettingsError):
        if route == 'batch_inline':
            batch.resolve_job_settings(batch.Job(module='mask', settings=supplied, overrides=overrides))
        else:
            path = tmp_path / 'settings.json'
            path.write_text(json.dumps(supplied))
            if route == 'cli':
                cli.resolve_settings(module, str(path), overrides)
            else:
                batch.resolve_job_settings(batch.Job(module='mask', settings=str(path), overrides=overrides))
    assert supplied == original


def test_nonmodel_override_preserves_an_independent_pin_copy(inputs):
    _module, supplied = inputs
    resolved = batch.resolve_job_settings(batch.Job(module='mask', settings=supplied, overrides=['src=/new-data']))
    assert resolved['model_metadata'] == supplied['model_metadata']
    resolved['model_metadata']['settings']['cell_model_name']['key'] = 'mutated-result'
    assert supplied['model_metadata']['settings']['cell_model_name']['key'] != 'mutated-result'


@pytest.mark.parametrize('override', ['model_metadata={}', 'model_metadata.settings={}'])
def test_metadata_cannot_be_replaced_through_generic_overrides(inputs, override):
    module, supplied = inputs
    before = deepcopy(supplied)
    with pytest.raises(cli.SettingsError) as error:
        cli.apply_overrides(supplied, [override], module)
    assert 'model_metadata' in str(error.value)
    assert '--settings' in str(error.value)
    assert supplied == before


def test_invalid_pins_refuse_export_before_either_existing_output_is_touched(inputs, tmp_path):
    _module, supplied = inputs
    supplied['model_metadata']['schema_version'] = True
    csv_path, json_path = tmp_path / 'settings.csv', tmp_path / 'settings.json'
    csv_path.write_bytes(b'previous complete settings CSV')
    json_path.write_bytes(b'previous complete settings JSON')
    with pytest.raises(cli.SettingsError):
        cli.write_settings_file(supplied, csv_path)
    assert csv_path.read_bytes() == b'previous complete settings CSV'
    assert json_path.read_bytes() == b'previous complete settings JSON'


def test_csv_and_json_preserve_opaque_version_and_hash_strings(inputs, tmp_path):
    _module, supplied = inputs
    paths = cli.write_settings_file(supplied, tmp_path / 'settings.csv')
    for path in paths:
        assert cli.load_settings_file(path)['model_metadata'] == supplied['model_metadata']


def test_one_to_one_model_setting_rename_moves_its_pin_with_the_value(inputs, monkeypatch):
    from spacr import settings as registry
    _module, supplied = inputs
    supplied['old_cell_model'] = supplied.pop('cell_model_name')
    supplied['model_metadata']['settings']['old_cell_model'] = supplied['model_metadata']['settings'].pop('cell_model_name')
    original = deepcopy(supplied)
    monkeypatch.setitem(registry.RENAMED_SETTINGS, 'old_cell_model', 'cell_model_name')
    result = batch.resolve_job_settings(batch.Job(module='mask', settings=supplied))
    assert result['cell_model_name'] == supplied['old_cell_model']
    assert 'old_cell_model' not in result
    assert 'old_cell_model' not in result['model_metadata']['settings']
    assert result['model_metadata']['settings']['cell_model_name'] == supplied['model_metadata']['settings']['old_cell_model']
    assert supplied == original


def test_conflicting_old_and_new_pins_are_not_resolved_by_first_wins(inputs, monkeypatch):
    from spacr import settings as registry
    _module, supplied = inputs
    supplied['old_cell_model'] = supplied['cell_model_name']
    record = deepcopy(supplied['model_metadata']['settings']['cell_model_name'])
    record['sha256'] = '8' * 64
    supplied['model_metadata']['settings']['old_cell_model'] = record
    original = deepcopy(supplied)
    monkeypatch.setitem(registry.RENAMED_SETTINGS, 'old_cell_model', 'cell_model_name')
    with pytest.raises(cli.SettingsError):
        batch.resolve_job_settings(batch.Job(module='mask', settings=supplied))
    assert supplied == original


def test_one_to_many_setting_migration_cannot_duplicate_a_model_pin(inputs, monkeypatch):
    from spacr import settings as registry
    _module, supplied = inputs
    supplied['old_cell_model'] = supplied.pop('cell_model_name')
    supplied['model_metadata']['settings']['old_cell_model'] = supplied['model_metadata']['settings'].pop('cell_model_name')
    original = deepcopy(supplied)
    monkeypatch.setitem(registry.RENAMED_SETTINGS, 'old_cell_model', ('cell_model_name', 'nucleus_model_name'))
    with pytest.raises(cli.SettingsError):
        batch.resolve_job_settings(batch.Job(module='mask', settings=supplied))
    assert supplied == original


@pytest.mark.parametrize('flag', ['--force', '--no-preflight'])
def test_run_flags_cannot_bypass_invalid_imported_pins(inputs, tmp_path, monkeypatch, flag):
    import logging
    _module, supplied = inputs
    supplied['model_metadata']['settings']['cell_model_name']['requested_value'] = '/models/stale'
    path = tmp_path / 'settings.json'
    path.write_text(json.dumps(supplied))
    monkeypatch.setattr(cli, 'setup_logging', lambda *args: logging.getLogger('n702-cli-test'))
    monkeypatch.setattr(cli, 'import_entry', lambda *args: pytest.fail('invalid pin reached pipeline import'))
    assert cli.main(['mask', '--settings', str(path), flag,
                     '--set', 'cell_model_name=/models/stale']) == cli.EXIT_USAGE


def test_failed_later_override_cannot_leave_a_partly_edited_pinned_payload(inputs):
    module, supplied = inputs
    original = deepcopy(supplied)
    with pytest.raises(cli.SettingsError):
        cli.apply_overrides(supplied, ['cell_model_name=/models/other', 'definitely_not_a_setting=1'], module)
    assert supplied == original
