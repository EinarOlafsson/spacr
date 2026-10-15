"""Private backend requests retain exact identity and refuse unknown replacements."""
from pathlib import Path

import pytest

from spacr import _segmentation_backends as backends
from spacr import model_zoo as zoo


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino', 'stardist'])
def test_unknown_backend_request_refuses_with_actual_context_before_construction(
        monkeypatch, backend):
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda *a, **k: None)
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Unknown must not download'))
    value = 'typo-checkpoint' if backend == 'cellpose3' else f'{backend}:/missing/checkpoint.pt'
    with pytest.raises(zoo.ModelZooError) as error:
        if backend == 'cellpose3':
            backends._cellpose3_model(value, object_type='cell')
        elif backend == 'cellpose_dino':
            backends._cellpose_dino_model(value)
        else:
            backends._prefixed_model(backend, value)
    assert 'model_name' in str(error.value)
    assert repr(value) in str(error.value)


def _resolve(backend, value, *, setting='pathogen_model'):
    if backend == 'cellpose3':
        return backends._cellpose3_model(value, object_type='pathogen', setting=setting)
    if backend == 'cellpose_dino':
        return backends._cellpose_dino_model(value, setting=setting)
    return backends._prefixed_model(backend, value, setting=setting)


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino', 'stardist'])
def test_explicit_unknown_context_retains_setting_and_original_prefixed_value(monkeypatch, backend):
    calls = []
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda request, **kwargs:
                        calls.append((request, kwargs)) or None)
    original = f'{backend}:unknown-checkpoint' + ('#cells' if backend == 'stardist' else '')
    with pytest.raises(zoo.ModelZooError) as error:
        _resolve(backend, original)
    assert 'pathogen_model' in str(error.value)
    assert repr(original) in str(error.value)
    request, kwargs = calls[0]
    assert request == 'unknown-checkpoint'
    assert kwargs['setting'] == 'pathogen_model'
    assert kwargs['original_request'] == original
    assert kwargs['kinds'] == (backend,)


@pytest.mark.parametrize('value,role,expected', [
    (None, 'cell', 'cyto3'), ('', 'nucleus', 'nuclei'),
    ('cpsam', 'cell', 'cyto3'), ('cpsam', 'nucleus', 'nuclei'),
    ('cellpose3:', 'cell', 'cyto3'), ('cyto2', 'cell', 'cyto2'),
    ('cellpose3:nuclei', 'cell', 'nuclei')])
def test_known_cellpose3_defaults_and_stock_are_unchanged(monkeypatch, value, role, expected):
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda *a, **k: pytest.fail('Stock must not acquire'))
    assert backends._cellpose3_model(value, object_type=role) == expected


@pytest.mark.parametrize('backend', ['stardist', 'instanseg', 'omnipose'])
def test_prefixed_blank_and_vendor_stock_keep_existing_semantics(monkeypatch, backend):
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda *a, **k: pytest.fail('Stock must not acquire'))
    spec = backends._SPECS[backend]
    assert backends._prefixed_model(backend, backend + ':') == spec.default_model
    assert backends._prefixed_model(backend, spec.models[0]) == spec.models[0]


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino', 'stardist'])
def test_existing_file_or_supported_folder_wins_without_acquisition(monkeypatch, tmp_path, backend):
    local = tmp_path / 'checkpoint'
    if backend == 'stardist':
        local.mkdir()
        (local / 'weights').write_bytes(b'unchanged folder bytes')
    else:
        local.write_bytes(b'unchanged checkpoint bytes')
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda *a, **k: pytest.fail('Local must not acquire'))
    value = f'{backend}:{local}' + ('#cells' if backend == 'stardist' else '')
    assert _resolve(backend, value) == str(local.resolve())
    assert (local / 'weights' if local.is_dir() else local).read_bytes().startswith(b'unchanged')


@pytest.mark.parametrize('backend', ['cellpose3', 'cellpose_dino', 'stardist'])
def test_actual_parallel_backend_worker_refuses_before_environment_or_constructor(monkeypatch, backend):
    from copy import deepcopy
    from spacr import _mask_workers as workers

    value = f'{backend}:unknown-checkpoint'
    settings = {'pathogen_model': value, 'pathogen_model_name': 'cpsam', 'pathogen_channel': 0}
    original = deepcopy(settings)
    monkeypatch.setattr(zoo, '_ensure_model_file', lambda *a, **k: None)
    monkeypatch.setattr(backends, '_read_marker', lambda *a, **k: pytest.fail('Unknown reached environment'))
    monkeypatch.setattr(backends, '_load_backend', lambda *a, **k: pytest.fail('Unknown reached constructor'))
    with pytest.raises(zoo.ModelZooError) as error:
        workers._resolved_backend_model(settings, 'pathogen')
    assert 'pathogen_model' in str(error.value)
    assert repr(value) in str(error.value)
    assert settings == original


@pytest.fixture
def declared_backend(tmp_path, monkeypatch, request):
    import hashlib
    import socket

    backend = request.param
    source = tmp_path / 'publisher_weights.pt'
    payload = b'controlled backend checkpoint bytes'
    source.write_bytes(payload)
    raw = {'key': 'controlled_backend', 'name': source.name, 'kind': backend,
           'version': '001', 'sha256': hashlib.sha256(payload).hexdigest(), 'uri': source.as_uri()}
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [zoo._entry_from_mapping(raw)])
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'node')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'on')
    monkeypatch.setattr(socket.socket, 'connect', lambda *a, **k: pytest.fail('No network is allowed'))
    return backend, raw, source, payload


@pytest.mark.parametrize('declared_backend', ['cellpose3', 'cellpose_dino', 'stardist'], indirect=True)
def test_real_shared_acquisition_logs_original_prefix_and_target(declared_backend, caplog):
    import logging

    backend, raw, source, payload = declared_backend
    value = f'{backend}:/laptop/{source.name}' + ('#cells' if backend == 'stardist' else '')
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        result = _resolve(backend, value)
    messages = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(messages) == 1
    assert all(part in messages[0] for part in ('pathogen_model', value, raw['key'], '001', raw['sha256'], str(result)))
    assert Path(result).read_bytes() == payload
    assert source.read_bytes() == payload


@pytest.mark.parametrize('declared_backend', ['cellpose3', 'cellpose_dino', 'stardist'], indirect=True)
def test_offline_backend_refusal_never_runs_provider_or_transport(declared_backend, monkeypatch):
    backend, raw, source, payload = declared_backend
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('Offline provider execution'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Offline transport'))
    with pytest.raises(zoo.ModelUnreadable) as error:
        _resolve(backend, f'{backend}:/laptop/{source.name}')
    assert raw['sha256'] in str(error.value)
    assert source.read_bytes() == payload


@pytest.mark.parametrize("declared_backend", ["stardist"], indirect=True)
def test_prefixed_download_false_keeps_pure_planning(declared_backend, monkeypatch):
    backend, raw, source, payload = declared_backend
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: pytest.fail('Planning provider execution'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Planning transport'))
    result = backends._prefixed_model(backend, f'{backend}:/laptop/{source.name}',
                                     download=False, setting='pathogen_model')
    assert not Path(result).exists()
    assert source.read_bytes() == payload
