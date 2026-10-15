"""Exercise Work's model pin fixtures through real settings serializers."""
import copy
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr.cli import SettingsError, load_settings_file, write_settings_file


FIXTURES = Path(__file__).resolve().parents[1] / 'features/data/702_work_model_reference_schema_2026-10-14'
CASES = [row for row in json.loads((FIXTURES / 'fixture-index-final.json').read_text())
         if row['file'].endswith('.json')]


@pytest.mark.parametrize('case', CASES, ids=lambda case: Path(case['file']).stem)
def test_independently_prepared_identity_fixtures(case):
    settings = json.loads((FIXTURES / case['file']).read_text())
    original = copy.deepcopy(settings)
    if case['expected'] == 'reject':
        with pytest.raises(SettingsError):
            load_settings_file(str(FIXTURES / case['file']))
        with pytest.raises(zoo.ModelZooError):
            zoo._validate_model_metadata(settings, model_kinds={'pathogen_model_name': ('cellpose',)})
    else:
        result = zoo._validate_model_metadata(settings, model_kinds={'pathogen_model_name': ('cellpose',)})
        assert result == settings.get('model_metadata')
        if result:
            result['settings'].clear()
            assert settings == original
    assert settings == original


@pytest.mark.parametrize('extension', ['csv', 'json'])
@pytest.mark.parametrize('fixture', ['valid_synthetic_declared_001', 'valid_synthetic_declared_opaque_version',
                                     'valid_windows_request', 'valid_missing_builtin_unspecified_version'])
def test_real_settings_roundtrip_keeps_opaque_identity_and_path(tmp_path, extension, fixture):
    settings = load_settings_file(str(FIXTURES / ('fixtures/' + fixture + '.json')))
    original = copy.deepcopy(settings)
    outputs = write_settings_file(settings, tmp_path / 'settings.csv')
    destination = next(path for path in outputs if path.endswith('.' + extension))
    restored = load_settings_file(destination)
    assert restored['model_metadata'] == settings['model_metadata']
    assert zoo._validate_model_metadata(restored) == settings['model_metadata']
    assert settings == original


def test_edit_invalidation_is_explicit_and_cannot_repair_a_stale_import():
    settings = load_settings_file(str(FIXTURES / 'fixtures/valid_missing_builtin_unspecified_version.json'))
    original = copy.deepcopy(settings)
    edited = zoo._invalidate_model_metadata(settings, ['pathogen_model_name'])
    assert edited['model_metadata']['settings'] == {}
    assert settings == original
    settings['pathogen_model_name'] += '_edited'
    with pytest.raises(zoo.ModelZooError):
        zoo._invalidate_model_metadata(settings, ['pathogen_model_name'])


@pytest.mark.parametrize('version', [None, '', ' ', '001', '01', '2026.10-beta'])
def test_raw_catalogue_versions_do_not_inherit_modelentry_cache_versions(version):
    raw = {'key': 'synthetic', 'name': 'checkpoint.pt', 'kind': 'classifier',
           'sha256': '1' * 64}
    if version is not None:
        raw['version'] = version
    result = zoo._model_reference_from_record('/laptop/checkpoint.pt', raw)
    assert result['version'] == (version if version and version.strip() else None)
    assert result['version_status'] == ('declared' if version and version.strip() else 'unspecified')
    assert zoo._match_model_reference(result, [raw, dict(raw)]) == result


def test_current_catalogue_cannot_silently_relabel_recorded_bytes():
    raw = {'key': 'synthetic', 'name': 'checkpoint.pt', 'kind': 'classifier',
           'sha256': '1' * 64, 'version': '001'}
    record = zoo._model_reference_from_record('/laptop/checkpoint.pt', raw)
    conflicting = {**raw, 'sha256': '2' * 64}
    with pytest.raises(zoo.ModelZooError):
        zoo._match_model_reference(record, [raw, conflicting])
    with pytest.raises(zoo.ModelZooError):
        zoo._match_model_reference(record, [{**raw, 'version': '1'}])


def test_export_pins_explicit_catalogue_selection_without_disk_or_network(monkeypatch, tmp_path):
    raw = next(row for row in zoo.BUNDLED_REMOTE_MODELS if row['key'] == 'toxoplasma_plaque_v4')
    settings = {'src': '/laptop/plate', 'plaque_model': raw['key'], 'well_detection': False}
    original = copy.deepcopy(settings)

    def forbidden(*args, **kwargs):
        pytest.fail('Export attempted filesystem, plugin or catalogue traffic')

    with monkeypatch.context() as protected:
        protected.setattr(zoo, 'catalogue', forbidden)
        protected.setattr(zoo, 'shared_catalogue', forbidden)
        protected.setattr(zoo, 'sha256_file', forbidden)
        protected.setattr(zoo.os.path, 'isfile', forbidden)
        protected.setattr(Path, 'is_file', forbidden)
        exported = zoo._capture_model_metadata('analyze_plaques', settings)
    assert settings == original
    record = exported['model_metadata']['settings']['plaque_model']
    assert record['key'] == raw['key'] and record['sha256'] == raw['sha256']
    assert record['version'] is None and record['version_status'] == 'unspecified'
    outputs = write_settings_file(exported, tmp_path / 'settings.csv')
    assert load_settings_file(outputs[0])['model_metadata'] == exported['model_metadata']


def test_same_basename_local_file_is_not_claimed_verified_on_export(tmp_path):
    raw = next(row for row in zoo.BUNDLED_REMOTE_MODELS if row['key'] == 'toxoplasma_plaque_v4')
    path = tmp_path / raw['name']
    path.write_bytes(b'custom bytes, not the published checkpoint')
    for request in (str(path), raw['name']):
        settings = {'plaque_model': request}
        assert zoo._capture_model_metadata('analyze_plaques', settings) == settings


@pytest.mark.parametrize('module,key,kind', [('analyze_plaques', 'well_detection', 'detector'),
    ('cellpose_masks', 'custom_model', 'cellpose'), ('train_cellpose', 'base_model', 'cellpose'),
    ('train_only', 'custom_model_path', 'classifier'), ('activation', 'model_path', 'classifier'),
    ('ops', 'cellpose_model', 'cellpose'), ('spatial_masks', 'spatial_cellpose_model', 'cellpose')])
def test_consumer_registry_distinguishes_actual_module_inputs(module, key, kind):
    assert zoo._model_input_kinds(module, {key: 'requested'}) == {key: (kind,)}


def test_model_prefix_precedes_global_backend_and_classifier_family_is_respected():
    settings = {'pathogen_model_name': 'cellpose_dino:/model.pt',
                'pathogen_model': 'cellpose3:/other.pt', 'segmentation_backend': 'cellpose'}
    assert zoo._model_input_kinds('mask', settings) == {
        'pathogen_model_name': ('cellpose_dino',), 'pathogen_model': ('cellpose3',)}
    assert zoo._model_input_kinds('classify_merged', {'model_path': '/model.pt', 'classifier_family': 'ml'}) == {}
    assert zoo._model_input_kinds('classify_merged', {'model_path': '/model.pt', 'classifier_family': 'cv'}) == {
        'model_path': ('classifier',)}
