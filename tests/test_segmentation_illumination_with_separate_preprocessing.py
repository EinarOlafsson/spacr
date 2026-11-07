"""Item 802: a preprocess-only run prepares the segmentation flat field.

With ``illumination_for_later_masks``, ``preprocess=True, masks=False``
applies and records the segmentation illumination correction, and a later
``preprocess=False, masks=True`` run reads that record back, as Make Masks
does, and segments the corrected inputs. The masks equal those of one
combined run. A threshold stand-in replaces Cellpose; no GPU is used.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

FIELD = 'plate1_A01_F001.npy'


def _model(path):
    """A known vignette: bright centre, a dim corner where an object hides."""
    from spacr.illumination import IlluminationField, IlluminationModel

    rows, cols = np.indices((16, 16), dtype=np.float32)
    radius = np.sqrt((rows - 7.5) ** 2 + (cols - 7.5) ** 2)
    flat = np.clip(1.0 - 0.07 * radius, 0.30, 1.0).astype(np.float32)
    model = IlluminationModel(
        fields={'plate1': IlluminationField(
            plate='plate1', channels=(0,), flatfield=flat[None, ...],
            dark=np.zeros(1, dtype=np.float32), n_fields=20,
            estimator='polynomial', degree=2, bin_size=1)},
        meta={'application_contract_version': 1,
              'channel_index_space': 'persisted-intensity-axis',
              'estimated_from_intensity_state': 'raw'})
    return model.save(path), flat


def _threshold_model(monkeypatch):
    """Replace Cellpose with a threshold on the normalised input."""
    import torch
    from scipy import ndimage

    import spacr.object as O

    class ThresholdModel:
        def __init__(self, **kwargs):
            self.pretrained_model = kwargs.get('pretrained_model')

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
            masks, flows = [], []
            for image in x:
                plane = np.asarray(image)[..., 0]
                masks.append(ndimage.label(plane >= 0.6)[0].astype(np.uint16))
                flows.append((np.zeros((*plane.shape, 3), dtype=np.float32),
                              np.zeros((2, *plane.shape), dtype=np.float32),
                              np.zeros(plane.shape, dtype=np.float32), None))
            return masks, flows, None

    monkeypatch.setattr(O, 'cp_models', type('Models', (), {'CellposeModel': ThresholdModel}))
    monkeypatch.setattr(O.accelerator, 'cellpose_kwargs',
                        lambda: {'gpu': False, 'device': torch.device('cpu')})


def _prepared_inputs(root, *, illumination, masks=True, later=False):
    """Run Make Masks preprocessing on one vignetted field.

    :returns: the settings and source folder ``preprocess_img_data`` returns.
    """
    from spacr.io import preprocess_img_data
    from spacr.settings import set_default_settings_preprocess_generate_masks

    model_path, flat = _model(Path(root).parent / f'{Path(root).name}_vignette.npz')
    raw = np.rint(np.full(flat.shape, 10, dtype=np.float32) * flat)
    truth = np.zeros(flat.shape, dtype=bool)
    truth[1:4, 1:4] = True
    truth[7:10, 7:10] = True
    raw[truth] = np.rint(1000 * flat[truth])
    stack = Path(root) / 'stack'
    stack.mkdir(parents=True)
    np.save(stack / FIELD, raw.astype(np.uint16)[..., None])
    settings = set_default_settings_preprocess_generate_masks({
        'src': str(stack.parent), 'channels': [0], 'cell_channel': 0,
        'nucleus_channel': None, 'pathogen_channel': None, 'randomize': False,
        'timelapse': False, 'batch_size': 1, 'plot': False, 'lower_percentile': 2,
        'verbose': False, 'masks': masks, 'illumination_for_later_masks': later,
        'illumination_correction': illumination,
        'illumination_model': str(model_path) if illumination else '',
        'illumination_qc': False})
    return preprocess_img_data(settings)


def _masks_only(settings, src):
    """The ``preprocess=False, masks=True`` run, as Make Masks performs it."""
    import spacr.object as O
    from spacr.illumination import load_segmentation_illumination_resume
    from spacr.io import _normalized_npz_field_ids

    settings = dict(settings, preprocess=False, masks=True)
    mask_src = Path(src) / 'masks'
    if settings.get('illumination_correction', False):
        load_segmentation_illumination_resume(
            settings,
            provenance_path=str(Path(src) / 'illumination' / 'segmentation_application.json'),
            pipeline_style='v1', expected_fields=_normalized_npz_field_ids(str(mask_src)),
            verbose=False)
    O.generate_cellpose_masks_sam(str(mask_src), settings, 'cell')
    return np.load(mask_src / 'cell_mask_stack' / FIELD)


def _archive(src):
    """The normalised segmentation input the masks were made from."""
    archive = next((Path(src) / 'masks').glob('*.npz'))
    with np.load(archive) as data:
        return np.array(data['data'])


def test_the_setting_is_off_by_default_and_alpha():
    from spacr.settings import ALPHA_FEATURES, set_default_settings_preprocess_generate_masks

    assert set_default_settings_preprocess_generate_masks({})[
        'illumination_for_later_masks'] is False
    assert ALPHA_FEATURES[802]['settings'] == ('illumination_for_later_masks',)


def test_split_runs_give_the_masks_of_one_combined_run(tmp_path, monkeypatch):
    import spacr.object as O

    _threshold_model(monkeypatch)
    combined_settings, combined_src = _prepared_inputs(tmp_path / 'combined',
                                                       illumination=True)
    O.generate_cellpose_masks_sam(str(Path(combined_src) / 'masks'), combined_settings, 'cell')
    combined = np.load(Path(combined_src) / 'masks' / 'cell_mask_stack' / FIELD)

    split_settings, split_src = _prepared_inputs(tmp_path / 'split', illumination=True,
                                                 masks=False, later=True)
    assert not (Path(split_src) / 'masks' / 'cell_mask_stack').exists()
    np.testing.assert_array_equal(_archive(split_src), _archive(combined_src))
    split = _masks_only(split_settings, split_src)

    np.testing.assert_array_equal(split, combined)
    assert int(combined.max()) == 2, 'the dim corner object was not recovered'


def test_without_the_setting_a_preprocess_only_run_stays_uncorrected(tmp_path, monkeypatch):
    from spacr.illumination import IlluminationError

    _threshold_model(monkeypatch)
    settings, src = _prepared_inputs(tmp_path / 'old', illumination=True, masks=False)
    assert not (Path(src) / 'illumination' / 'segmentation_application.json').exists()
    _plain_settings, plain_src = _prepared_inputs(tmp_path / 'plain', illumination=False)
    np.testing.assert_array_equal(_archive(src), _archive(plain_src))
    with pytest.raises(IlluminationError, match='segmentation_application.json'):
        _masks_only(settings, src)
    assert int(_masks_only(dict(settings, illumination_correction=False), src).max()) == 1
