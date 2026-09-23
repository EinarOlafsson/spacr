"""Inference routes agree on CPU precision without changing supported GPU defaults."""
import json

import numpy as np
import pytest
import torch

from spacr import accelerator as acc
from tests.test_cellpose_api_contract import contract_cellpose


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_mask_comparison_streaming_and_editor_share_precision(
        tmp_path, monkeypatch, contract_cellpose, device):
    from spacr import model_compare, pipeline_v2, utils
    from spacr.qt.screens.make_masks import load_cellpose_model

    gpu = device != 'cpu'
    monkeypatch.setattr(acc, '_CACHED', acc.Accelerator(
        kind='cuda' if gpu else 'cpu', device=device, label=device))

    def check():
        arguments = contract_cellpose.last.init_kwargs
        assert str(arguments['device']) == device
        assert arguments['gpu'] is gpu
        assert arguments['use_bfloat16'] is gpu
        assert arguments['pretrained_model'] == 'cpsam'

    utils._choose_model('cpsam', torch.device(device))
    check()
    model_compare.segment_with_cellpose([np.zeros((16, 16))], model_compare.ModelConfig(model='cpsam'))
    check()
    load_cellpose_model('cpsam')
    check()

    path = tmp_path / 'stack.npy'
    image = np.ones((16, 16, 1), np.uint16)
    np.save(path, image)
    (tmp_path / 'channel_order.json').write_text(json.dumps({'image_channels': ['DAPI']}))
    stack = pipeline_v2.StackFile('field', path, image.shape, ['DAPI'])
    pipeline_v2.stream_masks_from_stack([stack], model_name='cpsam')
    check()
    saved = np.load(path)
    np.testing.assert_array_equal(saved[..., :1], image)
    assert saved.shape == (16, 16, 2) and saved[1:4, 1:4, -1].all()


def test_explicit_cpu_device_overrides_resolved_gpu_precision(monkeypatch, contract_cellpose):
    from spacr.utils import _choose_model

    monkeypatch.setattr(acc, '_CACHED', acc.Accelerator(kind='cuda', device='cuda:0', label='GPU'))
    model = _choose_model('cpsam', torch.device('cpu'))
    assert str(model.init_kwargs['device']) == 'cpu'
    assert model.init_kwargs['gpu'] is False
    assert model.init_kwargs['use_bfloat16'] is False


def test_explicit_gpu_does_not_inherit_cpu_only_precision(monkeypatch, contract_cellpose):
    from spacr.utils import _choose_model

    monkeypatch.setattr(acc, '_CACHED', acc._CPU)
    model = _choose_model('cpsam', torch.device('cuda:0'))
    assert str(model.init_kwargs['device']) == 'cuda:0'
    assert model.init_kwargs['gpu'] is True
    assert model.init_kwargs['use_bfloat16'] is True


@pytest.mark.parametrize('gpu', [False, True])
def test_ops_cpu_precision_and_default_weights_are_explicit(monkeypatch, contract_cellpose, gpu):
    from spacr.ops_engine import _cellpose_model

    monkeypatch.setattr(acc, '_CACHED', acc.Accelerator(kind='cuda', device='cuda:0', label='GPU'))
    model = _cellpose_model({'cellpose_model': 'cpsam'}, gpu=gpu)
    assert model.init_kwargs['gpu'] is gpu
    assert model.init_kwargs['use_bfloat16'] is gpu


@pytest.mark.parametrize('entry', ['test_cellpose_model', 'apply_cellpose_model'])
def test_test_and_apply_model_use_cpu_precision(tmp_path, monkeypatch, contract_cellpose, entry):
    from spacr import submodules, utils
    import tifffile

    monkeypatch.setattr(acc, '_CACHED', acc._CPU)
    monkeypatch.setattr(utils, 'save_settings', lambda *args, **kwargs: None)
    if entry == 'test_cellpose_model':
        for folder in ('test/images', 'test/masks'):
            (tmp_path / folder).mkdir(parents=True)
            tifffile.imwrite(tmp_path / folder / 'field.tif', np.ones((16, 16), np.uint16))
    else:
        tifffile.imwrite(tmp_path / 'field.tif', np.ones((16, 16), np.uint16))

    class ConstructionReached(Exception):
        pass

    def construct(*args, **kwargs):
        model = contract_cellpose(*args, **kwargs)
        assert model.init_kwargs['gpu'] is False
        assert model.init_kwargs['use_bfloat16'] is False
        assert str(model.init_kwargs['device']) == 'cpu'
        raise ConstructionReached

    monkeypatch.setattr(submodules.cp_models, 'CellposeModel', construct)
    with pytest.raises(ConstructionReached):
        getattr(submodules, entry)({'src': str(tmp_path), 'model_path': 'cpsam'})
