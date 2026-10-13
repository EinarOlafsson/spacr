"""Parallel mask resume identifies the actual weights, without loading a model."""

import copy
import hashlib

import pytest

from spacr._mask_workers import _prepare_mask_model


@pytest.fixture
def weights(tmp_path, monkeypatch):
    from cellpose import models

    path = tmp_path / 'stock.pt'
    path.write_bytes(b'stock-model-content')
    monkeypatch.setattr(models, 'MODEL_NAMES', ['cpsam', 'cpsam_v2'])
    calls = []

    def resolve(name):
        calls.append(name)
        return str(path)

    def forbidden(**kwargs):
        pytest.fail('model constructed during signature preparation')

    monkeypatch.setattr(models, 'cache_model_path', resolve, raising=False)
    monkeypatch.setattr(models, 'CellposeModel', forbidden)
    return path, calls


def test_stock_weights_resolve_once_and_parent_settings_remain_unchanged(weights):
    path, calls = weights
    settings = {'cell_model_name': 'cpsam', 'cell_channel': 0,
                'mask_gpu_indices': [0, 1]}
    before = copy.deepcopy(settings)
    prepared, signature, identity = _prepare_mask_model(settings, 'cell')
    assert calls == ['cpsam']
    assert settings == before
    assert prepared['cell_model_name'] == str(path)
    assert identity == {'path': str(path), 'bytes': path.stat().st_size,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    assert len(signature) == 64
    prepared['mask_gpu_indices'].append(4)
    assert settings == before


@pytest.mark.parametrize('role', ['cell', 'nucleus', 'pathogen'])
def test_custom_checkpoint_never_falls_back_to_stock(weights, tmp_path, role):
    _, calls = weights
    custom = tmp_path / 'custom.pt'
    custom.write_bytes(b'custom-checkpoint')
    prepared, _, identity = _prepare_mask_model({f'{role}_model_name': str(custom)}, role)
    assert calls == []
    assert prepared[f'{role}_model_name'] == str(custom)
    assert identity['sha256'] == hashlib.sha256(custom.read_bytes()).hexdigest()


def test_pathogen_override_controls_both_signature_and_worker_settings(weights, tmp_path):
    _, calls = weights
    custom = tmp_path / 'pathogen.pt'
    custom.write_bytes(b'pathogen-checkpoint')
    prepared, _, identity = _prepare_mask_model(
        {'pathogen_model_name': 'cpsam', 'pathogen_model': str(custom)}, 'pathogen')
    assert prepared['pathogen_model'] == prepared['pathogen_model_name'] == str(custom)
    assert identity['path'] == str(custom)
    assert calls == []


def test_same_filename_with_new_weights_cannot_resume_old_masks(weights):
    path, _ = weights
    _, before, _ = _prepare_mask_model({'cell_model_name': str(path)}, 'cell')
    path.write_bytes(b'changed-model-content')
    _, after, _ = _prepare_mask_model({'cell_model_name': str(path)}, 'cell')
    assert before != after


def test_gpu_selection_and_cosmetics_do_not_change_material_identity(weights):
    settings = {'cell_model_name': 'cpsam', 'mask_parallel': True,
                'mask_gpu_indices': [0, 1], 'plot': False, 'verbose': False}
    _, before, _ = _prepare_mask_model(settings, 'cell')
    settings.update(mask_parallel=False, mask_gpu_indices=[2, 4], plot=True, verbose=True)
    _, after, _ = _prepare_mask_model(settings, 'cell')
    assert before == after
    settings['cell_flow_threshold'] = 0.2
    assert _prepare_mask_model(settings, 'cell')[1] != before


def test_a_missing_custom_checkpoint_is_not_downloaded_as_stock(weights, tmp_path):
    _, calls = weights
    with pytest.raises(FileNotFoundError):
        _prepare_mask_model({'cell_model_name': str(tmp_path / 'missing.pt')}, 'cell')
    assert calls == []


def test_empty_or_changing_weights_are_refused(weights, monkeypatch):
    from spacr import model_zoo

    path, _ = weights
    path.write_bytes(b'')
    with pytest.raises(ValueError, match='nonempty'):
        _prepare_mask_model({'cell_model_name': str(path)}, 'cell')
    path.write_bytes(b'valid')
    original = model_zoo.sha256_file

    def changed(path):
        result = original(path)
        path.write_bytes(b'replaced-with-new-weights')
        return result

    monkeypatch.setattr(model_zoo, 'sha256_file', changed)
    with pytest.raises(ValueError, match='Model changed'):
        _prepare_mask_model({'cell_model_name': str(path)}, 'cell')


@pytest.mark.parametrize('settings,role', [({'segmentation_backend': 'samcell'}, 'cell'),
                                          ({}, '../outside')])
def test_unimplemented_model_contracts_fail_before_loading(weights, settings, role):
    with pytest.raises(ValueError):
        _prepare_mask_model(settings, role)
    assert weights[1] == []


def test_minimum_cellpose_uses_its_zero_argument_cpsam_cache(weights, monkeypatch):
    from cellpose import models

    path, calls = weights
    monkeypatch.setattr(models, 'cache_model_path', None)
    monkeypatch.setattr(models, 'cache_CPSAM_model_path',
                        lambda: str(path), raising=False)
    prepared, _, identity = _prepare_mask_model({'cell_model_name': 'cpsam'}, 'cell')
    assert prepared['cell_model_name'] == identity['path'] == str(path)
    assert calls == []
    with pytest.raises(ValueError, match='cannot resolve weights'):
        _prepare_mask_model({'cell_model_name': 'cpsam_v2'}, 'cell')
