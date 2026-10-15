"""Real consumer boundaries receive verified pinned paths without settings rewrites."""
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from spacr import model_zoo as zoo
from spacr import submodules as sm


@pytest.fixture
def pinned(tmp_path, monkeypatch):
    payload = b'verified consumer checkpoint'
    raw = {'key': 'recorded', 'name': 'current-published.pt', 'version': '01',
           'kind': 'cellpose', 'sha256': hashlib.sha256(payload).hexdigest(),
           'uri': 'https://trusted.invalid/weights'}
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'home')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, 'catalogue', lambda *a, **k: pytest.fail('pin reached legacy discovery'))
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    def fetch(entry, destination):
        target = Path(destination) / entry.name
        target.write_bytes(payload)
        return target
    monkeypatch.setattr(zoo, 'fetch', fetch)
    def settings(name, kind='cellpose'):
        raw['kind'] = kind
        requested = str(tmp_path / ('old-' + name + '.pt'))
        return {name: requested, 'model_metadata': {'schema_version': 1, 'settings': {
            name: zoo._model_reference_from_record(requested, raw)}}}
    return settings, raw, payload


@pytest.mark.parametrize('name,kind,resolver', [
    ('plaque_model', 'cellpose', sm._resolve_plaque_model),
    ('well_detection', 'detector', sm._resolve_well_detector),
])
def test_native_consumers_use_exact_pinned_identity_before_name_fallback(pinned, name, kind, resolver):
    make_settings, raw, payload = pinned
    settings = make_settings(name, kind)
    original = deepcopy(settings)
    resolved = resolver(settings)
    assert Path(resolved).name == raw['name']
    assert Path(resolved).read_bytes() == payload
    assert settings == original


@pytest.mark.parametrize('name,kind,resolver', [
    ('plaque_model', 'cellpose', sm._resolve_plaque_model),
    ('well_detection', 'detector', sm._resolve_well_detector),
])
def test_existing_wrong_pinned_file_is_not_passed_to_native_consumer(pinned, name, kind, resolver):
    make_settings, _raw, _payload = pinned
    settings = make_settings(name, kind)
    local = Path(settings[name])
    local.write_bytes(b'incorrect supplied model')
    with pytest.raises(zoo.ModelZooError):
        resolver(settings)
    assert local.read_bytes() == b'incorrect supplied model'


def test_plaque_preview_never_acquires_missing_pinned_weights(pinned, monkeypatch):
    make_settings, _raw, _payload = pinned
    settings = make_settings('plaque_model')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('preview fetched weights'))
    with pytest.raises(sm.ModelZooMissing):
        sm._resolve_plaque_model(settings, fetch=False)


def test_actual_training_construction_gets_verified_base_path(pinned, tmp_path, monkeypatch):
    make_settings, raw, payload = pinned
    settings = make_settings('base_model')
    settings['src'] = str(tmp_path)
    original = deepcopy(settings)
    (tmp_path / 'train/images').mkdir(parents=True)
    (tmp_path / 'train/masks').mkdir(parents=True)
    tifffile.imwrite(tmp_path / 'train/images/one.tif', np.zeros((12, 16), np.uint16))
    tifffile.imwrite(tmp_path / 'train/masks/one.tif', np.zeros((12, 16), np.uint16))
    seen = {}
    class Stop(Exception):
        pass
    def construct(**kwargs):
        seen.update(kwargs)
        raise Stop
    monkeypatch.setattr(sm.cp_models, 'CellposeModel', construct)
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    with pytest.raises(Stop):
        sm.train_cellpose(settings)
    assert Path(seen['pretrained_model']).name == raw['name']
    assert Path(seen['pretrained_model']).read_bytes() == payload
    assert settings == original


def test_actual_figure_consumer_gets_verified_detector_path(pinned, monkeypatch, tmp_path):
    from spacr import plaque_papers
    make_settings, raw, payload = pinned
    settings = make_settings('figure_detector', 'detector')
    settings['src'] = str(tmp_path)
    original = deepcopy(settings)
    seen = {}
    class Stop(Exception):
        pass
    def measure(*args, **kwargs):
        seen.update(kwargs)
        raise Stop
    monkeypatch.setattr(plaque_papers, 'figure_folders', lambda _source: ['private-test-folder'])
    monkeypatch.setattr(plaque_papers, 'measure_figure_folder', measure)
    with pytest.raises(Stop):
        sm._analyze_plaque_figures(settings, 'supplied-segmenter')
    assert Path(seen['detector']).name == raw['name']
    assert Path(seen['detector']).read_bytes() == payload
    assert settings == original
