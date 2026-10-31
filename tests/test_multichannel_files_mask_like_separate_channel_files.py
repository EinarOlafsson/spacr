"""One multichannel file and one file per channel give byte-identical masks.

The Generate Masks batch route (and Make Masks' Mask folder, which runs the
same preprocessing) reads separate channel files by their CellVoyager names.
A multichannel file is first split by :func:`spacr.io.convert_to_yokogawa`
into the same per-channel planes. Each layout here carries the SAME raw
planes; the masks every role writes, the stacks they were cut from and the
inputs each role's model received must be identical, byte for byte, and raw
channel identity must hold (channel 2 of the file is the C03 plane).
"""
import copy
import types

import numpy as np
import pytest
import tifffile

import spacr.object as objects
from spacr.io import convert_to_yokogawa, preprocess_img_data
from spacr.settings import set_default_settings_preprocess_generate_masks
from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS
from tests.conftest import check_cellpose_eval_call

ROLES = ('cell', 'nucleus', 'pathogen')


def _planes(n, dtype=np.uint16):
    rows, cols = np.indices((24, 26))
    planes = []
    for index in range(n):
        plane = 50 + ((rows * (index + 3) + cols * (7 - index)) * 13) % 200
        plane[4 + index:9 + index, 5:12] += 400
        planes.append(plane.astype(dtype))
    return planes


def _separate(root, planes):
    for index, plane in enumerate(planes, 1):
        tifffile.imwrite(root / f'plate1_A01_T0001F001L01A01Z01C{index:02d}.tif', plane)


def _imagej_cyx(root, planes):
    tifffile.imwrite(root / 'field.tif', np.stack(planes), imagej=True,
                     metadata={'axes': 'CYX'})


def _channel_last(root, planes):
    tifffile.imwrite(root / 'field.tif', np.stack(planes, axis=-1),
                     metadata={'axes': 'YXC'})


def _ome_tiff(root, planes):
    tifffile.imwrite(root / 'field.ome.tif', np.stack(planes), ome=True,
                     metadata={'axes': 'CYX'})


def _ome_tiff_z(root, planes):
    stack = np.stack([np.stack(planes), np.stack(planes) // 2])
    tifffile.imwrite(root / 'field.ome.tif', stack, ome=True,
                     metadata={'axes': 'ZCYX'})


def _rgb_png(root, planes):
    import imageio.v2 as imageio
    imageio.imwrite(root / 'field.png', np.stack(planes, axis=-1))


def _ome_zarr(root, planes):
    from spacr.ome_zarr import write_ome_zarr
    write_ome_zarr(root / 'field.zarr', np.stack(planes), axes=('c', 'y', 'x'), levels=1)


def _masks_for(root, settings, monkeypatch, received):
    def eval_masks(image):
        signal = np.asarray(image)[..., 0]
        labels, _ = __import__('scipy.ndimage', fromlist=['label']).label(
            signal > signal.mean())
        return labels.astype(np.uint16)

    class Model:
        def __init__(self, pretrained_model=None, **kwargs):
            self.role = pretrained_model

        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            received.setdefault(str(root), []).extend(
                np.asarray(image).copy() for image in x)
            return ([eval_masks(image) for image in x],
                    [np.zeros(np.asarray(image).shape[:2]) for image in x], None)

    monkeypatch.setattr(objects, 'cp_models', types.SimpleNamespace(CellposeModel=Model))
    monkeypatch.setattr(objects.accelerator, 'is_gpu', lambda: False)
    monkeypatch.setattr(objects.accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    settings, _source = preprocess_img_data(settings)
    out = {}
    for role in ROLES:
        objects.generate_cellpose_masks_sam(str(root / 'masks'), copy.deepcopy(settings),
                                           role, run_qc=False)
        for path in sorted((root / 'masks' / (role + '_mask_stack')).glob('*.npy')):
            out[(role, path.name)] = path.read_bytes()
    stacks = {path.name: np.load(path, allow_pickle=False)
              for path in sorted((root / 'stack').glob('*.npy'))}
    return out, stacks


def _settings(root, metadata_type, n, tmp_path):
    checkpoints = {}
    for role in ROLES:
        path = tmp_path / (role + '.pth')
        path.write_bytes(role.encode())
        checkpoints[role] = str(path)
    return set_default_settings_preprocess_generate_masks(dict(
        src=str(root), metadata_type=metadata_type, custom_regex=None,
        cell_channel=n - 1, nucleus_channel=0, pathogen_channel=1,
        organelle_channel=None, channels=list(range(n)), number_of_organelles=0,
        randomize=False, timelapse=False, plot=False, save=True, verbose=False,
        n_jobs=1, batch_size=1, resume=False, save_original_images=True,
        adjust_cells=False, merge_pathogens=False, seg_qc=False,
        **{role + '_model_name': checkpoints[role] for role in ROLES}))


@pytest.mark.parametrize('n, writer, dtype', [
    (4, _imagej_cyx, np.uint16),
    (4, _channel_last, np.uint16),
    (6, _ome_tiff, np.uint16),
    (5, _ome_tiff_z, np.uint16),
    (3, _rgb_png, np.uint8),
    (4, _ome_zarr, np.uint16),
], ids=['imagej-cyx', 'channel-last', 'ome-tiff-6ch', 'ome-tiff-zcyx',
        'rgb-png', 'ome-zarr'])
def test_a_multichannel_file_masks_like_its_separate_channel_files(
        tmp_path, monkeypatch, n, writer, dtype):
    planes = _planes(n, dtype)
    separate = tmp_path / 'separate'
    multi = tmp_path / 'multi'
    separate.mkdir()
    multi.mkdir()
    _separate(separate, planes)
    writer(multi, planes)
    convert_to_yokogawa(str(multi))
    written = sorted(p.name for p in multi.glob('plate1_A01_*C*.tif'))
    assert written == [f'plate1_A01_T0001F001L01C{c:02d}.tif' for c in range(1, n + 1)]
    for c, plane in enumerate(planes, 1):
        np.testing.assert_array_equal(
            tifffile.imread(multi / f'plate1_A01_T0001F001L01C{c:02d}.tif'), plane)

    received = {}
    masks_a, stacks_a = _masks_for(separate, _settings(separate, 'cellvoyager', n, tmp_path),
                                   monkeypatch, received)
    masks_b, stacks_b = _masks_for(multi, _settings(multi, 'auto', n, tmp_path),
                                   monkeypatch, received)
    assert list(stacks_a) == list(stacks_b) and len(stacks_a) == 1
    field = next(iter(stacks_a.values()))
    np.testing.assert_array_equal(field, next(iter(stacks_b.values())))
    for c, plane in enumerate(planes):
        np.testing.assert_array_equal(field[..., c], plane)
    assert sorted(masks_a) == sorted(masks_b) and len(masks_a) == len(ROLES)
    for key in masks_a:
        assert masks_a[key] == masks_b[key], key
    inputs_a, inputs_b = received[str(separate)], received[str(multi)]
    assert len(inputs_a) == len(inputs_b) == len(ROLES)
    for a, b in zip(inputs_a, inputs_b):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
    assert any(np.load(separate / 'masks' / 'cell_mask_stack' / name).any()
               for _role, name in masks_a)
