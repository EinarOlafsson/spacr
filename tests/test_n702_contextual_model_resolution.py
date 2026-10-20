"""Model acquisition reports exact context and unknown requests refuse precisely."""
from copy import deepcopy
import hashlib
import logging
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import utils

_REAL_CATALOGUE = zoo.catalogue


@pytest.fixture
def declared_model(tmp_path, monkeypatch):
    source = tmp_path / 'publisher-model.CP_model'
    payload = b'controlled exact publisher checkpoint bytes'
    source.write_bytes(payload)
    raw = {'key': 'declared_cells', 'name': source.name, 'kind': 'cellpose',
           'version': '001', 'sha256': hashlib.sha256(payload).hexdigest(),
           'uri': source.as_uri()}
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'on')
    monkeypatch.setattr(zoo, '_spacr_home', lambda: tmp_path / 'node')
    monkeypatch.setattr(zoo, '_on_the_qt_gui_thread', lambda: False)
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [zoo._entry_from_mapping(raw)])
    return raw, source, payload


def test_successful_pinned_download_logs_original_setting_and_exact_identity(
        declared_model, tmp_path, caplog):
    raw, source, payload = declared_model
    requested = str(tmp_path / 'laptop' / 'old-name.CP_model')
    settings = {'cell_model_name': requested, 'model_metadata': {
        'schema_version': 1, 'settings': {'cell_model_name':
            zoo._model_reference_from_record(requested, raw)}}}
    before = deepcopy(settings)
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        effective = zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('cellpose',))
    lines = [record.getMessage() for record in caplog.records
             if 'downloaded zoo model' in record.getMessage()]
    assert len(lines) == 1
    for value in ('cell_model_name', requested, raw['key'], '001', raw['sha256'], str(effective)):
        assert value in lines[0]
    assert effective.read_bytes() == payload and source.read_bytes() == payload
    assert settings == before
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        assert zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('cellpose',)) == effective
    assert not any('downloaded zoo model' in record.getMessage() for record in caplog.records)


def test_unpinned_download_has_explicit_actual_setting_context(declared_model, caplog):
    raw, source, payload = declared_model
    requested = '/old-laptop/' + source.name
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        result = zoo._ensure_model_file(requested, kinds=('cellpose',), setting='plaque_model')
    messages = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(messages) == 1
    assert all(value in messages[0] for value in ('plaque_model', requested, raw['key'], '001', raw['sha256'], str(result)))
    assert result.read_bytes() == payload


def test_failed_checksum_never_logs_success(declared_model, caplog):
    raw, source, payload = declared_model
    source.write_bytes(b'wrong mirror bytes')
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        with pytest.raises(zoo.ChecksumMismatch):
            zoo._ensure_model_file('/laptop/' + source.name, kinds=('cellpose',))
    assert not any('downloaded zoo model' in record.getMessage() for record in caplog.records)
    assert source.read_bytes() == b'wrong mirror bytes'


@pytest.mark.parametrize('requested', ['unknown_checkpoint', 'unknown.pt', '/laptop/unknown.CP_model', r'C:\laptop\unknown.pt'])
def test_unknown_cellpose_request_names_setting_and_declared_candidates_before_construction(
        declared_model, monkeypatch, requested):
    raw, source, payload = declared_model
    monkeypatch.setattr(utils, '_installed_cellpose_models', lambda: ('cpsam',))
    monkeypatch.setattr(utils.cp_models, 'CellposeModel', lambda **kwargs: pytest.fail('Unknown reached constructor'))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Unknown reached fetch'))
    with pytest.raises(zoo.ModelZooError) as error:
        utils._choose_model(requested, device='cpu', object_type='cell')
    message = str(error.value)
    assert repr(requested) in message
    assert 'cell_model_name' in message
    assert raw['key'] in message and raw['sha256'] in message and '001' in message
    assert source.read_bytes() == payload


def test_unmatched_sam_and_legacy_still_select_their_original_stock(declared_model, monkeypatch):
    monkeypatch.setattr(utils, '_installed_cellpose_models', lambda: ('cpsam',))
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Stock reached catalogue fetch'))
    for requested in ('sam', *utils.LEGACY_CELLPOSE_MODELS, None, ''):
        assert utils._resolve_cellpose_pretrained(requested, object_type='cell') == utils.CPSAM_MODEL


@pytest.mark.parametrize('cached', [False, True])
def test_offline_resolution_never_runs_plugin_metadata_or_network(
        declared_model, monkeypatch, tmp_path, cached):
    import socket
    from types import SimpleNamespace
    from spacr import plugins

    raw, source, payload = declared_model
    monkeypatch.setenv('SPACR_MODEL_DOWNLOAD', 'off')
    for name in ('stock_cellpose_entries', 'installable_backend_entries',
                 '_cellpose3_model_entries', '_prefixed_model_entries', 'bioimageio_entries'):
        monkeypatch.setattr(zoo, name, lambda: [])
    monkeypatch.setattr(zoo, 'catalogue', _REAL_CATALOGUE)
    monkeypatch.setattr(plugins, 'model_providers', lambda: [
        ('controlled', SimpleNamespace(provider='controlled:offline', key='offline'))])
    calls = []
    def forbidden_transport(*args, **kwargs):
        calls.append('network')
        raise RuntimeError('Transport tripwire; no network was attempted')
    def provider():
        calls.append('provider')
        socket.create_connection(('forbidden.invalid', 443))
        return []
    monkeypatch.setattr(socket, 'create_connection', forbidden_transport)
    monkeypatch.setattr(plugins, 'load_object', lambda _: provider)
    monkeypatch.setattr(plugins, 'record_diagnostic', lambda *args: None)
    monkeypatch.setattr(zoo, 'fetch', lambda *a, **k: pytest.fail('Offline fetch'))
    target = tmp_path / 'node' / 'models' / source.name
    if cached:
        target.parent.mkdir(parents=True)
        target.write_bytes(payload)
        assert zoo._ensure_model_file('/laptop/' + source.name, kinds=('cellpose',)) == target
    else:
        with pytest.raises(zoo.ModelUnreadable) as error:
            zoo._ensure_model_file('/laptop/' + source.name, kinds=('cellpose',))
        assert raw['sha256'] in str(error.value) and str(target) in str(error.value)
    assert calls == []
    assert source.read_bytes() == payload


def test_undeclared_publisher_version_is_not_logged_as_default_one(declared_model, monkeypatch, caplog):
    raw, source, payload = declared_model
    raw.pop('version')
    monkeypatch.setattr(zoo, 'BUNDLED_REMOTE_MODELS', (raw,))
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        result = zoo._ensure_model_file('/laptop/' + source.name,
                                        kinds=('cellpose',), setting='plaque_model')
    messages = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(messages) == 1 and raw['key'] + ' None ' in messages[0]
    assert result.read_bytes() == payload


@pytest.mark.parametrize('consumer', ['training', 'detector', 'plaque'])
def test_actual_legacy_consumer_download_log_retains_its_setting(
        declared_model, tmp_path, caplog, consumer):
    from spacr import submodules

    raw, source, payload = declared_model
    requested = '/laptop/' + source.name
    if consumer == 'detector':
        raw['kind'] = 'detector'
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        if consumer == 'training':
            result = submodules._resolve_training_base(requested)
            setting = 'base_model'
        elif consumer == 'detector':
            result = submodules._resolve_detector_weights(requested, 'figure_detector')
            setting = 'figure_detector'
        else:
            result = submodules._resolve_plaque_model({'plaque_model': requested})
            setting = 'plaque_model'
    messages = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(messages) == 1 and setting in messages[0] and requested in messages[0]
    assert raw['sha256'] in messages[0] and '001' in messages[0]
    assert source.read_bytes() == payload
    assert Path(result).read_bytes() == payload


@pytest.mark.parametrize('name', ['custom_model_path', 'resume_checkpoint'])
@pytest.mark.parametrize('known', [False, True])
def test_actual_classifier_initialization_preserves_context_before_loading_or_training(
        declared_model, monkeypatch, tmp_path, caplog, name, known):
    import torch
    from spacr import deep_spacr as deep

    raw, source, payload = declared_model
    raw['kind'] = 'classifier'
    requested = '/laptop/' + (source.name if known else 'unknown-classifier.pth')
    monkeypatch.setattr(utils, 'choose_model', lambda *a, **k: torch.nn.Linear(1, 2))
    monkeypatch.setattr(deep, 'pick_device', lambda **k: (torch.device('cpu'), None))
    class Loaded(Exception):
        pass
    def load(path, **kwargs):
        assert known
        assert Path(path).read_bytes() == payload
        raise Loaded
    monkeypatch.setattr(deep, 'load_model_artifact', load)
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        with pytest.raises(Loaded if known else zoo.ModelZooError) as error:
            deep.train_model(str(tmp_path), str(tmp_path / 'out'), 'tiny', [],
                             **{name: requested}, tensorboard=False)
    if known:
        messages = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
        assert len(messages) == 1 and name in messages[0] and requested in messages[0]
        assert raw['sha256'] in messages[0]
    else:
        assert name in str(error.value) and repr(requested) in str(error.value)
        assert raw['key'] in str(error.value)
    assert source.read_bytes() == payload


@pytest.mark.parametrize('route', ['prepare', 'robustness', 'generator'])
@pytest.mark.parametrize('override', [False, True])
def test_actual_pathogen_selection_names_the_input_field_that_won(
        declared_model, monkeypatch, tmp_path, route, override):
    from spacr import _mask_workers, object as objects
    from spacr.settings import set_default_settings_preprocess_generate_masks

    requested = 'unknown_pathogen_override' if override else 'unknown_pathogen_role'
    settings = set_default_settings_preprocess_generate_masks({
        'cell_channel': 0, 'nucleus_channel': 1, 'pathogen_channel': 2,
        'channels': [0, 1, 2], 'pathogen_model_name': 'unknown_pathogen_role',
        'pathogen_model': requested if override else None, 'verbose': False})
    expected_setting = 'pathogen_model' if override else 'pathogen_model_name'
    monkeypatch.setattr(utils, '_installed_cellpose_models', lambda: ('cpsam',))
    monkeypatch.setattr(utils.cp_models, 'CellposeModel', lambda **k: pytest.fail('Unknown reached constructor'))
    with pytest.raises(zoo.ModelZooError) as error:
        if route == 'prepare':
            _mask_workers._prepare_mask_model(settings, 'pathogen')
        elif route == 'robustness':
            objects._robustness_segmenter(settings, 'pathogen')
        else:
            folder = tmp_path / 'batches'
            folder.mkdir()
            objects.generate_cellpose_masks_sam(str(folder), settings, 'pathogen')
    assert expected_setting + ':' in str(error.value)
    assert repr(requested) in str(error.value)


def test_foreign_uri_cannot_borrow_a_bundled_publisher_version(declared_model, monkeypatch, tmp_path, caplog):
    from dataclasses import replace

    raw, source, payload = declared_model
    foreign = tmp_path / 'another-publisher' / source.name
    foreign.parent.mkdir()
    foreign.write_bytes(payload)
    entry = replace(zoo._entry_from_mapping(raw), uri=foreign.as_uri(), version='002')
    monkeypatch.setattr(zoo, 'catalogue', lambda **k: [entry])
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        actual = zoo._ensure_model_file('/laptop/' + source.name,
                                        kinds=('cellpose',), setting='plaque_model')
    lines = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(lines) == 1 and "ModelEntry.version='002'" in lines[0]
    assert actual.read_bytes() == payload and source.read_bytes() == payload
    assert foreign.read_bytes() == payload


@pytest.mark.parametrize('name', ['model_path', 'custom_model_path', 'resume_checkpoint'])
@pytest.mark.parametrize('known', [False, True])
def test_named_classifier_input_resolves_before_any_checkpoint_consumer(
        declared_model, name, known, caplog):
    from spacr import deep_spacr as deep

    raw, source, payload = declared_model
    raw['kind'] = 'classifier'
    requested = '/laptop/' + (source.name if known else 'unknown-classifier.pth')
    settings = {name: requested}
    before = deepcopy(settings)
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        if known:
            result = deep._classifier_model_input(settings, name)
            assert Path(result).read_bytes() == payload
            lines = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
            assert len(lines) == 1 and name in lines[0] and requested in lines[0]
        else:
            with pytest.raises(zoo.ModelZooError) as error:
                deep._classifier_model_input(settings, name)
            assert name in str(error.value) and repr(requested) in str(error.value)
            assert raw['key'] in str(error.value)
    assert settings == before and source.read_bytes() == payload


@pytest.mark.parametrize('value', [None, '', '  '])
def test_optional_classifier_input_retains_blank_new_training_choices(declared_model, monkeypatch, value):
    from spacr import deep_spacr as deep

    monkeypatch.setattr(zoo, 'catalogue', lambda **k: pytest.fail('Empty input discovered models'))
    assert deep._classifier_model_input({'custom_model_path': value}, 'custom_model_path') == value


@pytest.mark.parametrize('directory', [False, True])
def test_existing_classifier_input_paths_retain_original_semantics(declared_model, monkeypatch, tmp_path, directory):
    from spacr import deep_spacr as deep

    path = tmp_path / 'existing-classifier'
    if directory:
        path.mkdir()
    else:
        path.write_bytes(b'private file, no model decode')
    monkeypatch.setattr(zoo, 'catalogue', lambda **k: pytest.fail('Existing input discovered models'))
    assert deep._classifier_model_input({'model_path': str(path)}, 'model_path') == str(path)


def test_initialization_policy_receives_resolved_checkpoint_before_reading(declared_model, monkeypatch):
    import torch
    from spacr import deep_spacr as deep
    from spacr.classification_pixels import initialization_policy

    raw, source, payload = declared_model
    raw['kind'] = 'classifier'
    requested = '/laptop/' + source.name
    class ReadBoundary(Exception):
        pass
    def read(path, **kwargs):
        assert str(path) != requested and Path(path).read_bytes() == payload
        raise ReadBoundary
    monkeypatch.setattr(torch, 'load', read)
    with pytest.raises(ReadBoundary):
        initialization_policy(deep._classifier_model_input({'custom_model_path': requested}, 'custom_model_path'))


def test_lookup_and_original_backend_request_remain_distinct_in_success_log(declared_model, caplog):
    raw, source, payload = declared_model
    original = 'backend-prefix:/laptop/' + source.name + '#target'
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        result = zoo._ensure_model_file(raw['key'], kinds=('cellpose',),
                                        setting='pathogen_model', original_request=original)
    lines = [record.getMessage() for record in caplog.records if 'downloaded zoo model' in record.getMessage()]
    assert len(lines) == 1 and original in lines[0] and 'pathogen_model' in lines[0]
    assert raw['key'] in lines[0] and raw['sha256'] in lines[0]
    assert result.read_bytes() == payload and source.read_bytes() == payload


@pytest.mark.parametrize('mode', ['pinned', 'prefixed', 'foreign_uri', 'corrupt_cache'])
def test_corrupt_acquisition_preserves_checksum_type_and_original_context(
        declared_model, tmp_path, monkeypatch, caplog, mode):
    raw, source, payload = declared_model
    requested = str(tmp_path / 'laptop' / source.name)
    setting = 'cell_model_name'
    if mode == 'prefixed':
        requested = 'cellpose3:' + requested + '#nucleus'
    if mode == 'foreign_uri':
        alternate = tmp_path / 'foreign-source.CP_model'
        alternate.write_bytes(b'foreign corrupt mirror')
        foreign = dict(raw, uri=alternate.as_uri())
        monkeypatch.setattr(zoo, 'catalogue', lambda **kwargs: [zoo._entry_from_mapping(foreign)])
        corrupted_source = alternate
    else:
        source.write_bytes(b'corrupt controlled mirror')
        corrupted_source = source
    bad_bytes = corrupted_source.read_bytes()
    cached = tmp_path / 'node' / 'models' / source.name
    if mode == 'corrupt_cache':
        cached.parent.mkdir(parents=True)
        cached.write_bytes(b'existing corrupt cache retained')
    before_cache = cached.read_bytes() if cached.exists() else None
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        with pytest.raises(zoo.ChecksumMismatch) as error:
            if mode == 'pinned':
                settings = {setting: requested, 'model_metadata': {
                    'schema_version': 1, 'settings': {setting:
                        zoo._model_reference_from_record(requested, raw)}}}
                before = deepcopy(settings)
                zoo._resolve_pinned_model_file(settings, setting, kinds=('cellpose',))
            else:
                zoo._ensure_model_file('/laptop/' + source.name,
                                       kinds=('cellpose',), setting=setting,
                                       original_request=requested)
    message = str(error.value)
    assert all(value in message for value in (setting, repr(requested), raw['key'],
                                              raw['sha256'], 'expected sha256',
                                              hashlib.sha256(bad_bytes).hexdigest()))
    assert 'no existing checkpoint or exact model-zoo identity matches' not in message
    assert isinstance(error.value.__cause__, zoo.ChecksumMismatch)
    assert str(error.value.__cause__) in message
    assert not any('downloaded zoo model' in record.getMessage() for record in caplog.records)
    assert corrupted_source.read_bytes() == bad_bytes
    assert (cached.read_bytes() if cached.exists() else None) == before_cache
    assert not list(cached.parent.glob('*_v*'))
    if mode == 'pinned':
        assert settings == before


@pytest.mark.parametrize('existing', [True, False])
def test_corrupt_pinned_local_or_returned_path_names_actual_setting_and_identity(
        declared_model, tmp_path, monkeypatch, caplog, existing):
    raw, source, payload = declared_model
    requested = str(tmp_path / 'laptop' / 'exact-pinned.CP_model')
    bad_path = Path(requested) if existing else tmp_path / 'returned.CP_model'
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt = b'controlled corrupt pinned bytes'
    bad_path.write_bytes(corrupt)
    settings = {'cell_model_name': requested, 'model_metadata': {
        'schema_version': 1, 'settings': {'cell_model_name':
            zoo._model_reference_from_record(requested, raw)}}}
    before = deepcopy(settings)
    acquisitions = []
    def acquired(entry, **context):
        acquisitions.append((entry.key, context))
        return bad_path
    monkeypatch.setattr(zoo, '_ensure_catalogue_model_file', acquired)
    with caplog.at_level(logging.INFO, logger='spacr.model_zoo'):
        with pytest.raises(zoo.ModelZooError) as error:
            zoo._resolve_pinned_model_file(settings, 'cell_model_name', kinds=('cellpose',))
    message = str(error.value)
    assert all(value in message for value in ('cell_model_name', repr(requested),
                                              raw['key'], raw['sha256'], str(bad_path),
                                              'actual=' + hashlib.sha256(corrupt).hexdigest()))
    assert 'no existing checkpoint or exact model-zoo identity matches' not in message
    assert not isinstance(error.value, zoo.ChecksumMismatch)
    assert len(acquisitions) == (0 if existing else 1)
    assert bad_path.read_bytes() == corrupt and source.read_bytes() == payload
    assert settings == before
    assert not any('downloaded zoo model' in record.getMessage() for record in caplog.records)
