"""Real-object file classifiers honor exact exported pins without changing folders."""
from copy import deepcopy
import hashlib

import pytest

from spacr import model_zoo as zoo
from spacr import object_classifier as classifier


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    payload = b'controlled single-file real-object classifier'
    raw = {'key': 'real-object-v001', 'name': 'published.joblib',
           'kind': 'classifier', 'version': '001',
           'sha256': hashlib.sha256(payload).hexdigest()}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    requested = str(tmp_path / 'old-laptop' / 'unrelated-local-name.joblib')
    record = zoo._model_reference_from_record(requested, raw)
    settings = {'real_object_classifier': requested,
                'model_metadata': {'schema_version': 1,
                                   'settings': {'real_object_classifier': record}}}
    effective = tmp_path / 'published_v2.joblib'
    effective.write_bytes(payload)
    acquired = []

    def acquire(entry, *, download=True):
        acquired.append((entry.key, entry.version, download))
        return effective

    monkeypatch.setattr(zoo, '_ensure_catalogue_model_file', acquire)
    return settings, raw, effective, acquired


@pytest.mark.parametrize('module', ['mask', 'make_masks', 'timelapse'])
def test_single_file_classifier_is_an_explicit_exported_model_input(checkpoint, module):
    settings, raw, _, _ = checkpoint
    settings = {'real_object_classifier': raw['key']}
    assert zoo._model_input_kinds(module, settings)['real_object_classifier'] == ('classifier',)
    captured = zoo._capture_model_metadata(module, settings)
    record = captured['model_metadata']['settings']['real_object_classifier']
    assert record['key'] == raw['key'] and record['version'] == '001'
    assert record['sha256'] == raw['sha256']


def test_actual_filter_loads_exact_acquired_file_without_rewriting_original_settings(
        checkpoint, tmp_path, monkeypatch):
    settings, _, effective, acquired = checkpoint
    before = deepcopy(settings)
    loaded = []
    monkeypatch.setattr(classifier, '_real_classifier_bundles',
                        lambda path: loaded.append(str(path)) or {})
    assert classifier._drop_unreal_objects(str(tmp_path), settings) == {}
    assert loaded == [str(effective)]
    assert acquired == [('real-object-v001', '001', True)]
    assert settings == before


def test_changed_local_pinned_bytes_refuse_before_classifier_deserialization(
        checkpoint, tmp_path, monkeypatch):
    settings, _, _, acquired = checkpoint
    local = tmp_path / 'changed.joblib'
    local.write_bytes(b'changed classifier bytes')
    settings['real_object_classifier'] = str(local)
    settings['model_metadata']['settings']['real_object_classifier']['requested_value'] = str(local)
    monkeypatch.setattr(classifier, '_real_classifier_bundles',
                        lambda path: pytest.fail('deserialized unverified classifier'))
    with pytest.raises(zoo.ModelZooError):
        classifier._drop_unreal_objects(str(tmp_path), settings)
    assert acquired == []


@pytest.mark.parametrize('is_directory', [False, True])
def test_unpinned_existing_file_and_directory_keep_original_loader_contract(
        checkpoint, tmp_path, monkeypatch, is_directory):
    _, _, _, acquired = checkpoint
    path = tmp_path / ('bundle' if is_directory else 'private.joblib')
    if is_directory:
        path.mkdir()
    else:
        path.write_bytes(b'private existing file')
    settings = {'real_object_classifier': str(path)}
    loaded = []
    monkeypatch.setattr(classifier, '_real_classifier_bundles',
                        lambda value: loaded.append(value) or {})
    assert classifier._drop_unreal_objects(str(tmp_path), settings) == {}
    assert loaded == [str(path)] and acquired == []
    captured = zoo._capture_model_metadata('mask', settings)
    assert captured == settings
