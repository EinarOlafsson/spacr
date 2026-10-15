"""Exercise Work's model pin fixtures through real settings serializers."""
import copy
import json
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr.cli import load_settings_file, write_settings_file


FIXTURES = Path(__file__).resolve().parents[1] / 'features/data/702_work_model_reference_schema_2026-10-14'
CASES = [row for row in json.loads((FIXTURES / 'fixture-index-final.json').read_text())
         if row['file'].endswith('.json')]


@pytest.mark.parametrize('case', CASES, ids=lambda case: Path(case['file']).stem)
def test_independently_prepared_identity_fixtures(case):
    settings = load_settings_file(str(FIXTURES / case['file']))
    original = copy.deepcopy(settings)
    if case['expected'] == 'reject':
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
