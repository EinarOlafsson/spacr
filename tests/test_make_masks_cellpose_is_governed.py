"""Make Masks runs every in-process Cellpose call under the GPU memory governor.

Follow-up to items 801 and 803. A fake card stands in for CUDA (the fixture
of tests/test_gpu_memory_governor.py) and a threshold model stands in for
Cellpose-SAM, so nothing touches a GPU. With memory to spare the calls and
the masks are exactly those of an ungoverned run; an out-of-memory error is
retried instead of stopping the run.
"""

import ast
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from spacr import _gpu_governor as G  # noqa: E402
from tests.cellpose_api_contract import (  # noqa: E402
    MISSING_CHANNEL_AXIS, eval_arguments)
from tests.test_gpu_memory_governor import _oom, card  # noqa: E402,F401
from tests.test_segmentation_illumination_with_separate_preprocessing import (  # noqa: E402
    FIELD, MISSING_CHANNEL_AXIS, _prepared_inputs, check_cellpose_eval_call,
)

ROOT = Path(__file__).resolve().parent.parent


class _Calls:
    """What the stand-in models were asked to do, shared across instances."""

    sizes = []
    fail_first = 0


def _models(monkeypatch, *, on_gpu):
    """Install a threshold Cellpose-SAM stand-in, with or without a device."""
    from scipy import ndimage

    import spacr.object as O

    class ThresholdModel:
        def __init__(self, **kwargs):
            self.pretrained_model = kwargs.get('pretrained_model')
            if on_gpu:
                self.net = torch.nn.Linear(1, 1)
                self.device = torch.device('cuda')
                self.gpu = True

        # The whole Cellpose eval signature, as in test_io_v1_illumination.
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=None,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            _Calls.sizes.append(batch_size)
            if _Calls.fail_first:
                _Calls.fail_first -= 1
                raise _oom()
            masks, flows = [], []
            for image in x:
                plane = np.asarray(image)[..., 0]
                masks.append(ndimage.label(plane >= 0.6)[0].astype(np.uint16))
                flows.append((np.zeros((*plane.shape, 3), dtype=np.float32),
                              np.zeros((2, *plane.shape), dtype=np.float32),
                              np.zeros(plane.shape, dtype=np.float32), None))
            return masks, flows, None

    monkeypatch.setattr(O, 'cp_models', type('Models', (), {'CellposeModel': ThresholdModel}))
    monkeypatch.setattr(O.accelerator, 'cellpose_kwargs', lambda: {'gpu': on_gpu})
    _Calls.sizes = []
    _Calls.fail_first = 0


def _segment(root):
    import spacr.object as O

    settings, src = _prepared_inputs(root, illumination=False)
    O.generate_cellpose_masks_sam(str(Path(src) / 'masks'), settings, 'cell')
    return np.load(Path(src) / 'masks' / 'cell_mask_stack' / FIELD)


def test_with_memory_to_spare_the_masks_and_calls_are_unchanged(tmp_path, monkeypatch, card):
    _models(monkeypatch, on_gpu=False)
    plain = _segment(tmp_path / 'plain')
    plain_sizes = list(_Calls.sizes)

    governed = []
    real = G._governed_eval
    monkeypatch.setattr(G, '_governed_eval', lambda *args, **kwargs: (
        governed.append(kwargs['site']), real(*args, **kwargs))[1])
    _models(monkeypatch, on_gpu=True)
    masks = _segment(tmp_path / 'governed')

    np.testing.assert_array_equal(masks, plain)
    assert _Calls.sizes == plain_sizes
    assert governed == ['mask_sam']
    assert G._counters()['oom_retries'] == 0


def test_an_out_of_memory_error_is_retried_not_fatal(tmp_path, monkeypatch, card):
    _models(monkeypatch, on_gpu=False)
    plain = _segment(tmp_path / 'plain')
    _models(monkeypatch, on_gpu=True)
    _Calls.fail_first = 1

    masks = _segment(tmp_path / 'retried')

    np.testing.assert_array_equal(masks, plain)
    assert len(_Calls.sizes) == 2
    assert G._counters()['oom_retries'] == 1


def test_a_model_without_a_device_is_called_directly(monkeypatch):
    from spacr.object import _mask_eval

    monkeypatch.setattr(G, '_governed_eval', lambda *a, **k: pytest.fail('governed'))

    class Remote:
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            called = eval_arguments(locals())
            assert channel_axis is MISSING_CHANNEL_AXIS
            return x, called

    x, called = _mask_eval(Remote(), [1], site='mask_sam', batch_size=3,
                           normalize=False)
    assert x == [1]
    assert (called['batch_size'], called['normalize']) == (3, False)


def test_a_model_with_a_device_reaches_the_governor_with_its_keywords(monkeypatch):
    from spacr.object import _mask_eval

    seen = []
    monkeypatch.setattr(G, '_governed_eval', lambda model, x, **kwargs: seen.append(
        (x, kwargs)) or 'out')

    class Local:
        device = torch.device('cuda')

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            pytest.fail(f'called directly with channel_axis={channel_axis!r}')

    assert _mask_eval(Local(), [1], site='mask_organelle', settings={'a': 1},
                      batch_size=4, diameter=None) == 'out'
    assert seen == [([1], {'site': 'mask_organelle', 'settings': {'a': 1},
                           'batch_size': 4, 'diameter': None})]
    seen.clear()
    _mask_eval(Local(), [1], site='mask_z')
    assert seen[0][1]['batch_size'] == 8


@pytest.mark.parametrize('module', ['spacr/object.py', 'spacr/pipeline_v2.py'])
def test_make_masks_has_no_ungoverned_cellpose_call(module):
    """Every ``.eval(...)`` call with arguments goes through ``_mask_eval``.

    ``model.eval()`` with no arguments is PyTorch's evaluation-mode switch,
    not a Cellpose call, and the governed helper's own direct call is the
    path for models that run outside this process.
    """
    tree = ast.parse((ROOT / module).read_text(encoding='utf-8'))

    def eval_calls(root):
        return {node.lineno for node in ast.walk(root)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'eval' and (node.args or node.keywords)}

    helper = {line for function in ast.walk(tree)
              if isinstance(function, ast.FunctionDef) and function.name == '_mask_eval'
              for line in eval_calls(function)}
    assert not eval_calls(tree) - helper
