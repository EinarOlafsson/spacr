"""Planning honors exact recorded identities without discovering or fetching models."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import validate


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'pinned planning bytes'
    raw = {'key': 'planned-model', 'name': 'current-published-name.pt', 'kind': 'classifier',
           'sha256': hashlib.sha256(payload).hexdigest(), 'version': '01'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pinned planning discovered providers'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('planning acquired model weights'))
    def settings(name='model_path', kind='classifier'):
        raw['kind'] = kind
        request = str(tmp_path / ('obsolete-' + name + '.pt'))
        return {name: request, 'model_metadata': {'schema_version': 1, 'settings': {
            name: zoo._model_reference_from_record(request, raw)}}}
    return settings, raw, payload


def test_required_classifier_path_accepts_exact_pin_with_obsolete_basename(pinned):
    make_settings, _raw, _payload = pinned
    settings = make_settings()
    settings.update(train=False, test=True)
    original = deepcopy(settings)
    assert not [p for p in validate._check_required_paths(settings, 'classify') if p.setting == 'model_path']
    assert settings == original


def test_required_custom_cellpose_path_accepts_exact_pin_without_fetch(pinned):
    make_settings, _raw, _payload = pinned
    settings = make_settings('custom_model', 'cellpose')
    assert not [p for p in validate._check_required_paths(settings, 'cellpose_masks') if p.setting == 'custom_model']


@pytest.mark.parametrize('kind', ['cellpose', 'cellpose3', 'cellpose_dino'])
def test_pathogen_pin_does_not_report_missing_or_stock_fallback(pinned, kind):
    make_settings, _raw, _payload = pinned
    settings = make_settings('pathogen_model', kind)
    settings.update(pathogen_channel=0)
    assert not [p for p in validate._check_app_specific(settings, 'mask') if p.setting == 'pathogen_model']


def test_actual_validate_refuses_corrupt_local_pinned_bytes_without_fetch(pinned):
    make_settings, _raw, _payload = pinned
    settings = make_settings()
    Path(settings['model_path']).write_bytes(b'corrupt supplied checkpoint')
    problems = validate.validate_settings(settings, 'classify')
    assert len(problems) == 1
    assert problems[0].setting == 'model_metadata'
    assert problems[0].is_error
    assert 'sha256=' in problems[0].message


def test_actual_validate_rejects_invalid_kind_before_generic_or_plugin_validation(pinned):
    make_settings, _raw, _payload = pinned
    settings = make_settings('cell_model_name', 'classifier')
    problems = validate.validate_settings(settings, 'mask')
    assert len(problems) == 1
    assert problems[0].setting == 'model_metadata'
    assert problems[0].is_error


def test_actual_validate_accepts_recorded_missing_classifier_as_planned_not_missing(pinned, tmp_path):
    make_settings, _raw, _payload = pinned
    settings = make_settings()
    settings.update(src=str(tmp_path), train=False, test=True)
    original = deepcopy(settings)
    problems = validate.validate_settings(settings, 'classify')
    assert not [p for p in problems if p.setting in ('model_metadata', 'model_path') and p.is_error]
    assert settings == original
