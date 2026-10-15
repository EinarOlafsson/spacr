"""Direct Cellpose apply/test APIs load verified derived paths and retain requests."""

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from spacr import model_zoo as zoo
from spacr import run_journal as journal
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS


class ReachedEvaluation(RuntimeError):
    pass


@pytest.fixture
def direct(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg', force=True)
    from spacr import accelerator, submodules, workspace

    source = tmp_path / 'data'
    source.mkdir()
    images = source / 'test/images'
    masks = source / 'test/masks'
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    image = np.arange(256, dtype=np.uint16).reshape(16, 16)
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[3:8, 3:8] = 1
    tifffile.imwrite(source / 'input.tif', image)
    tifffile.imwrite(images / 'input.tif', image)
    tifffile.imwrite(masks / 'input.tif', mask)
    payload = b'exact direct API checkpoint fixture'
    mirror = tmp_path / 'mirror.CP_model'
    mirror.write_bytes(payload)
    raw = {'key': 'direct-api-v001', 'name': 'published.CP_model', 'kind': 'cellpose',
           'version': '001', 'sha256': hashlib.sha256(payload).hexdigest(), 'uri': mirror.as_uri()}
    cache = tmp_path / 'private-cache'
    monkeypatch.setenv('SPACR_MODEL_CACHE', str(cache))
    monkeypatch.delenv('SPACR_MODEL_DOWNLOAD', raising=False)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [raw])
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [zoo._entry_from_mapping(raw)])
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    calls = []

    class Model:
        def __init__(self, *, pretrained_model, **kwargs):
            calls.append(pretrained_model)

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None, flow3D_smooth=0,
                 stitch_threshold=0.0, min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            assert len(x) == 1 and np.asarray(x[0]).shape == (16, 16)
            raise ReachedEvaluation('intercepted before real inference')

    monkeypatch.setattr(submodules, 'cp_models', SimpleNamespace(CellposeModel=Model))
    home = tmp_path / 'home'
    runs = home / 'runs'
    runs.mkdir(parents=True)
    monkeypatch.setattr(journal, '_spacr_home', lambda: home)
    monkeypatch.setattr(journal, 'runs_root', lambda: runs)
    monkeypatch.setattr(journal, '_env_snapshot', lambda: {'packages': {}})
    monkeypatch.setattr(journal, '_write_environment_lock', lambda *a: {})
    monkeypatch.setattr(journal, '_check_lock_for_run', lambda *a: None)
    monkeypatch.setattr(journal, 'begin_recording', lambda *a, **k: None)
    monkeypatch.setattr(journal, 'finish_recording', lambda *a, **k: None)
    monkeypatch.setattr(journal, '_notify_run_finished', lambda *a: None)
    for name in ('_capture_initial_provenance', '_capture_final_provenance', '_snapshot_log_tail'):
        monkeypatch.setattr(journal.Run, name, lambda self: None)
    monkeypatch.setattr(workspace, 'save_for_run', lambda *a, **k: None)
    settings = {'src': str(source), 'model_path': str(tmp_path / 'old-laptop.CP_model'),
                'batch_size': 1, 'target_size': 16, 'normalize': False, 'save': False}
    return submodules, settings, raw, payload, cache, calls


@pytest.mark.parametrize('api', ['apply_cellpose_model', 'test_cellpose_model'])
@pytest.mark.parametrize('mode', ['pinned_cache', 'pinned_fetch', 'pinned_local', 'legacy_fetch', 'legacy_local', 'legacy_stock'])
def test_direct_api_constructor_and_journal_use_effective_bytes_while_snapshots_keep_original(direct, api, mode, monkeypatch):
    module, settings, raw, payload, cache, calls = direct
    if mode.startswith('pinned'):
        record = zoo._model_reference_from_record(settings['model_path'], raw)
        settings['model_metadata'] = {'schema_version': 1, 'settings': {'model_path': record}}
    elif mode == 'legacy_local':
        Path(settings['model_path']).write_bytes(payload)
    elif mode == 'legacy_fetch':
        settings['model_path'] = raw['key']
    else:
        settings['model_path'] = 'cyto_fake'
    if mode == 'pinned_local':
        Path(settings['model_path']).write_bytes(payload)
        monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('local pin fetched'))
    if mode == 'pinned_cache':
        cache.mkdir()
        (cache / raw['name']).write_bytes(payload)
        monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('cached model fetched'))
    original = deepcopy(settings)
    with journal.open_run('direct-api-fixture', settings) as run:
        with pytest.raises(ReachedEvaluation):
            getattr(module, api)(settings)
    expected = ('cyto_fake' if mode == 'legacy_stock' else original['model_path']
                if mode.endswith('_local') else str(cache / raw['name']))
    assert calls == [expected]
    assert settings['model_path'] == original['model_path']
    assert settings.get('model_metadata') == original.get('model_metadata')
    saved = json.loads((Path(settings['src']) / 'settings' / (api + '.json')).read_text())
    assert saved['model_path'] == original['model_path']
    assert saved.get('model_metadata') == original.get('model_metadata')
    manifest = json.loads((run.dir / 'manifest.json').read_text())
    if mode == 'legacy_stock':
        assert manifest['model_files'] == {}
    else:
        assert Path(expected).read_bytes() == payload
        recorded = manifest['model_files']['model_path']
        assert recorded['path'] == expected and recorded['sha256'] == raw['sha256']


@pytest.mark.parametrize('api', ['apply_cellpose_model', 'test_cellpose_model'])
@pytest.mark.parametrize('failure', ['local_hash', 'kind', 'offline'])
def test_direct_api_bad_pin_refuses_before_constructor(direct, api, failure, monkeypatch):
    module, settings, raw, payload, cache, calls = direct
    record = zoo._model_reference_from_record(settings['model_path'], raw)
    settings['model_metadata'] = {'schema_version': 1, 'settings': {'model_path': record}}
    if failure == 'local_hash':
        Path(settings['model_path']).write_bytes(b'wrong original bytes')
    elif failure == 'kind':
        record['kind'] = 'classifier'
        monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', [])
    else:
        monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', '0')
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('invalid/offline model fetched'))
    original = deepcopy(settings)
    with pytest.raises(zoo.ModelZooError):
        getattr(module, api)(settings)
    assert calls == []
    assert settings['model_path'] == original['model_path']
    assert settings['model_metadata'] == original['model_metadata']
    if failure == 'local_hash':
        assert Path(settings['model_path']).read_bytes() == b'wrong original bytes'
