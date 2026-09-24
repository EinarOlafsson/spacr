"""The current Cellpose normalization API must retain the requested inversion."""
import numpy as np
import pytest

from spacr.model_compare import ModelConfig, segment_with_cellpose
from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call


@pytest.fixture
def current_cellpose(monkeypatch):
    from cellpose import models

    seen = {}

    class CurrentModel:
        def __init__(self, **kwargs):
            seen['constructor'] = kwargs

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True, rescale=None,
                 diameter=None, flow_threshold=0.4, cellprob_threshold=0.0,
                 do_3D=False, anisotropy=None, flow3D_smooth=0,
                 stitch_threshold=0.0, min_size=15, max_size_fraction=0.4,
                 niter=None, augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis, require_channel_axis=False)
            seen['normalization'] = normalize
            seen['images'] = x
            return [np.zeros(image.shape, dtype=np.int32) for image in x], None, None

    monkeypatch.setattr(models, 'CellposeModel', CurrentModel)
    monkeypatch.setattr('spacr.accelerator.cellpose_kwargs', lambda: {'gpu': False})
    return seen


@pytest.mark.parametrize('normalize,invert', [(True, False), (True, True), (False, False)])
def test_current_signature_receives_normalization_and_inversion(current_cellpose, normalize, invert):
    image = np.arange(16, dtype=np.uint16).reshape(4, 4)
    before = image.copy()
    masks = segment_with_cellpose([image], ModelConfig(normalize=normalize, invert=invert))
    assert current_cellpose['normalization'] == {'normalize': normalize, 'invert': invert}
    assert np.array_equal(image, before)
    assert np.array_equal(current_cellpose['images'][0], image)
    assert masks[0].dtype == np.int32


def test_current_signature_preserves_additional_normalization_options(current_cellpose):
    options = {'normalize': True, 'percentile': [2, 98], 'tile_norm_blocksize': 64}
    config = ModelConfig(invert=True, extra={'normalize': options})
    segment_with_cellpose([np.zeros((4, 4))], config)
    assert current_cellpose['normalization'] == {**options, 'invert': True}
    assert 'invert' not in options


def test_current_signature_refuses_silently_ignored_inversion(current_cellpose):
    with pytest.raises(ValueError, match='Enable normalization'):
        segment_with_cellpose([np.zeros((4, 4))], ModelConfig(normalize=False, invert=True))
    assert 'images' not in current_cellpose
