"""Classifier consumers receive verified pin paths while retaining recorded settings."""
from copy import deepcopy
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch

from spacr import deep_spacr as deep
from spacr import model_zoo as zoo
from tests.test_cov_deep_spacr_apply_model import _tar_settings
from tests.test_cov_deep_spacr_entry_fusion import ds_stubs


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'recorded classifier checkpoint'
    raw = {'key': 'recorded-classifier', 'name': 'current-classifier.pth',
           'kind': 'classifier', 'sha256': hashlib.sha256(payload).hexdigest(),
           'version': '01', 'uri': 'https://trusted.invalid/weights'}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin reached provider discovery'))
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'cache')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    def fetch(entry, destination):
        actual = Path(destination) / entry.name
        actual.write_bytes(payload)
        return actual
    monkeypatch.setattr(zoo, 'fetch', fetch)
    monkeypatch.setattr(deep, 'pick_device', lambda **kwargs: (torch.device('cpu'), None))
    def settings(name='model_path'):
        requested = str(tmp_path / 'old-laptop-classifier.pth')
        return {name: requested, 'model_metadata': {'schema_version': 1, 'settings': {
            name: zoo._model_reference_from_record(requested, raw)}}}
    return settings, raw, payload


@pytest.mark.parametrize('wrong_local', [False, True])
def test_actual_tar_loader_verifies_recorded_bytes_before_loading(pinned, monkeypatch, wrong_local):
    make_settings, _raw, payload = pinned
    supplied = make_settings()
    settings = _tar_settings('unused.tar', supplied['model_path'], normalize=False)
    settings.update(supplied)
    original = deepcopy(settings)
    class Stop(Exception):
        pass
    def load(actual, _device):
        assert Path(actual).read_bytes() == payload
        raise Stop
    monkeypatch.setattr(deep, '_load_inference_model', load)
    if wrong_local:
        Path(settings['model_path']).write_bytes(b'incorrect supplied weights')
    with pytest.raises(zoo.ModelZooError if wrong_local else Stop):
        deep.apply_model_to_tar(settings)
    assert settings == original


@pytest.mark.parametrize('name', ['custom_model_path', 'resume_checkpoint'])
def test_training_policy_and_actual_train_call_receive_same_verified_path(pinned, tmp_path, monkeypatch, name):
    from spacr import io, utils
    make_settings, _raw, payload = pinned
    settings = make_settings(name)
    settings.update(src=str(tmp_path), classes=['negative', 'positive'], train=True, test=False,
                    cross_validation_folds=0, leakage_audit_train_test=False, epochs=1)
    original = deepcopy(settings['model_metadata'])
    seen = []
    def policy(actual):
        assert Path(actual).read_bytes() == payload
        seen.append(actual)
        return 'canonical'
    class Stop(Exception):
        pass
    def train(**kwargs):
        assert kwargs[name] == seen[0]
        assert Path(kwargs[name]).read_bytes() == payload
        raise Stop
    monkeypatch.setattr(deep, 'initialization_policy', policy)
    monkeypatch.setattr(io, 'generate_loaders', lambda *a, **k: (None, None, None))
    monkeypatch.setattr(utils, 'save_settings', lambda *a, **k: None)
    monkeypatch.setattr(deep, 'train_model', train)
    with pytest.raises(Stop):
        deep.train_test_model(settings)
    assert len(seen) == 1
    assert settings[name] == original['settings'][name]['requested_value']
    assert settings['model_metadata'] == original


def test_actual_orchestrator_does_not_skip_a_missing_but_exact_pinned_classifier(pinned, tmp_path, ds_stubs):
    make_settings, _raw, _payload = pinned
    settings = make_settings()
    tar = tmp_path / 'dataset.tar'
    tar.write_bytes(b'private fixture')
    settings.update(src=str(tmp_path), train=False, test=False, generate_training_dataset=False,
                    generate_full_dataset=False, apply_model_to_dataset=True, tar_path=str(tar))
    original = deepcopy(settings['model_metadata'])
    ds_stubs.apply_ret = pd.DataFrame({'path': ['a.png'], 'pred': [0.1]})
    deep.deep_spacr(settings)
    assert len(ds_stubs.apply_tar) == 1
    assert settings['model_path'] == original['settings']['model_path']['requested_value']
    assert settings['model_metadata'] == original


def test_new_training_output_invalidates_only_the_replaced_model_path_pin(pinned, tmp_path, ds_stubs):
    make_settings, _raw, _payload = pinned
    settings = make_settings()
    original = deepcopy(settings['model_metadata'])
    settings['custom_model_path'] = settings['model_path']
    settings['model_metadata']['settings']['custom_model_path'] = deepcopy(original['settings']['model_path'])
    model = tmp_path / 'new-training-output.pth'
    model.write_bytes(b'new output weights')
    ds_stubs.train_ret = str(model)
    settings.update(src=str(tmp_path), train=True, test=False, generate_training_dataset=False,
                    generate_full_dataset=False, apply_model_to_dataset=False)
    deep.deep_spacr(settings)
    assert settings['model_path'] == str(model)
    assert 'model_path' not in settings['model_metadata']['settings']
    assert settings['model_metadata']['settings']['custom_model_path'] == original['settings']['model_path']
    assert zoo._validate_model_metadata(settings) == settings['model_metadata']
