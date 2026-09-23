"""Cellpose 4 receives native fields, explicit channels and unambiguous training pairs."""
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
import tifffile
from spacr import submodules as sub
from spacr.settings import get_train_cellpose_default_settings


def paired(folder, *, masks=None, name='field', suffix='', channels=0):
    folder.mkdir(parents=True, exist_ok=True)
    masks = masks or folder/'masks'
    masks.mkdir(parents=True, exist_ok=True)
    image = np.arange(20*30, dtype=np.uint16).reshape(20, 30)
    if channels:
        image = np.stack([image + 1000*c for c in range(channels)], axis=-1)
    label = np.zeros((20, 30), np.uint32)
    label[3:9, 4:12] = 70000
    tifffile.imwrite(folder/(name+'.tif'), image, photometric='minisblack')
    tifffile.imwrite(masks/(name+suffix+'.tiff'), label)
    return image, label


@pytest.fixture
def training(monkeypatch):
    calls = []
    signature = inspect.signature(sub.train_cp.train_seg)
    def train(net, **kwargs):
        signature.bind(net, **kwargs)
        calls.append(kwargs)
        return 'actual/checkpoint', [1.0], [2.0]
    monkeypatch.setattr(sub.cp_models, 'CellposeModel', lambda **kw: SimpleNamespace(net=object()))
    monkeypatch.setattr(sub.train_cp, 'train_seg', train)
    monkeypatch.setattr(sub, '_cellpose_use_gpu', lambda: False)
    monkeypatch.setattr(sub, 'plot_cellpose_batch', lambda *a: None)
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    return calls


def test_defaults_match_cellpose4_finetuning_and_drop_retired_controls():
    defaults = get_train_cellpose_default_settings({})
    assert [defaults[k] for k in ('learning_rate','weight_decay','n_epochs','batch_size')] == [1e-5, .1, 100, 1]
    assert defaults['percentiles'] == [1, 99]
    assert not {'target_size','augment','diameter','width_height','model_type','from_scratch','background'} & defaults.keys()


@pytest.mark.parametrize('separate', [True, False])
def test_sources_match_suffix_and_extensions_without_resizing(tmp_path, training, separate):
    images = tmp_path/'images'
    masks = tmp_path/'labels' if separate else None
    image, labels = paired(images, masks=masks, suffix='_masks', channels=3)
    result = sub.train_cellpose(dict(src=str(images), mask_src=str(masks) if masks else '',
                                     channels=[2, 0], min_train_masks=1))
    call = training[0]
    np.testing.assert_array_equal(call['train_data'][0], np.moveaxis(image[..., [2, 0]], -1, 0))
    np.testing.assert_array_equal(call['train_labels'][0], labels)
    assert call['train_labels'][0].max() == 70000
    assert call['channel_axis'] == 0 and call['rescale'] is False
    assert call['normalize'] == dict(normalize=True, percentile=[1, 99])
    assert result[0] == 'actual/checkpoint'


def test_validation_and_checkpoint_controls_reach_cellpose(tmp_path, training):
    for name in ('training','validation'):
        paired(tmp_path/name)
    sub.train_cellpose(dict(src=str(tmp_path/'training'), test_src=str(tmp_path/'validation'),
                            save_path=str(tmp_path/'checkpoints'), save_each=True, save_every=10,
                            nimg_per_epoch=12, nimg_test_per_epoch=2, scale_range=.3))
    call = training[0]
    assert len(call['test_data']) == len(call['test_labels']) == 1
    assert call['save_path'] == str(tmp_path/'checkpoints')
    assert call['save_each'] and call['save_every'] == 10
    assert call['nimg_per_epoch'] == 12 and call['nimg_test_per_epoch'] == 2
    assert call['scale_range'] == .3


@pytest.mark.parametrize('problem, match', [('missing','Missing masks'), ('ambiguous','Ambiguous masks'),
                                            ('overlap','must be separate'), ('channels','at most three')])
def test_invalid_pairs_stop_before_model_construction(tmp_path, monkeypatch, problem, match):
    paired(tmp_path/'images', channels=4 if problem == 'channels' else 0)
    settings = dict(src=str(tmp_path/'images'))
    if problem == 'missing':
        (tmp_path/'images/masks/field.tiff').unlink()
    if problem == 'ambiguous':
        (tmp_path/'images/masks/field_masks.tiff').write_bytes((tmp_path/'images/masks/field.tiff').read_bytes())
    if problem == 'overlap':
        settings['test_src'] = settings['src']
    monkeypatch.setattr(sub.cp_models,'CellposeModel',lambda **kw: pytest.fail('model constructed before input validation'))
    with pytest.raises(ValueError, match=match):
        sub.train_cellpose(settings)


def test_cellpose_normalizes_channels_without_mixing_or_changing_geometry(tmp_path):
    paired(tmp_path/'images', channels=3)
    pairs = sub._cellpose_training_pairs(tmp_path/'images')
    images, labels = sub._cellpose_training_arrays(pairs, dict(channels=[2, 0]))
    actual = sub.train_cp._reshape_norm(images, channel_axis=0,
                                        normalize_params=dict(normalize=True, percentile=[1, 99]))
    assert actual[0].shape == (3, 20, 30)
    assert labels[0].shape == (20, 30)
    np.testing.assert_allclose(actual[0][0], actual[0][1], atol=1e-6)
    assert not actual[0][2].any()
