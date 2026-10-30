"""Real normalized archives feed each role's checkpoint its own raw channel."""
import copy
import json

import numpy as np
import pytest

import spacr.object as objects
from spacr.io import _normalize_img_batch, concatenate_and_normalize
from spacr.settings import set_default_settings_preprocess_generate_masks
from spacr.utils import MASK_CHANNEL_ROLE_ORDER
from tests.test_cov_object_masks_sam import fake_model as fake_model


@pytest.mark.parametrize('channels', [(3, 3, 2), (3, 1, 2)])
@pytest.mark.parametrize('stale_positions', [False, True])
def test_three_sequential_roles_keep_checkpoint_and_normalized_raw_pixels(
        tmp_path, monkeypatch, fake_model, channels, stale_positions):
    root = tmp_path / 'project'
    stack = root / 'stack'
    stack.mkdir(parents=True)
    rows, cols = np.indices((24, 26))
    raw = np.stack([100 + rows * 7, 200 + cols * 11,
                    300 + (rows * 13 + cols * 3) % 97,
                    400 + (cols * 17 + rows * 2) % 113], axis=-1).astype(np.uint16)
    np.save(stack / 'plate1_A01_1.npy', raw)
    roles = ('cell', 'nucleus', 'pathogen')
    checkpoints = {}
    for role in roles:
        checkpoint = tmp_path / (role + '.pth')
        checkpoint.write_bytes(('fixture-' + role).encode())
        checkpoints[role] = str(checkpoint)
    settings = set_default_settings_preprocess_generate_masks(dict(
        src=str(root), cell_channel=channels[0], nucleus_channel=channels[1],
        pathogen_channel=channels[2], organelle_channel=None, batch_size=1,
        randomize=False, timelapse=False, plot=False, save=True, verbose=False,
        n_jobs=1, resume=False, lower_percentile=2, seg_qc=False,
        **{role + '_model_name': checkpoints[role] for role in roles},
        **{role + '_cellprob_threshold': value for role, value in zip(roles, (.1, .2, .3))}))
    selected = list(dict.fromkeys(settings[key] for key in MASK_CHANNEL_ROLE_ORDER
                                  if settings.get(key) is not None))
    expected = _normalize_img_batch(raw[None].copy(), selected, np.float32, settings)[..., selected]
    folder = concatenate_and_normalize(str(stack), selected, settings=settings)
    archives = sorted((root / 'masks').glob('*.npz'))
    assert len(archives) == 1
    with np.load(archives[0], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['data'], expected)
    if stale_positions:
        for role in roles:
            settings['cellpose_' + role + '_channel'] = settings[role + '_channel']
    settings = json.loads(json.dumps(settings))
    monkeypatch.setattr(objects.accelerator, 'is_gpu', lambda: False)
    monkeypatch.setattr(objects.accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    before = (stack / 'plate1_A01_1.npy').read_bytes()
    from spacr._mask_workers import _mask_model_contract

    seen, actual_pixels, expected_pixels, actual_thresholds = [], [], [], []
    for role in roles:
        prepared, signature, identity = _mask_model_contract(copy.deepcopy(settings), role)
        assert identity['path'] == checkpoints[role] and len(signature) == 64
        worker_settings = json.loads(json.dumps(prepared))
        objects.generate_cellpose_masks_sam(folder, worker_settings, role,
                                           batch_paths=[str(archives[0])], run_qc=False)
        model = fake_model['model']
        assert str(model.loaded_model) == checkpoints[role]
        inputs = model.eval_inputs[0]
        indices = [selected.index(settings[role + '_channel'])]
        if role == 'cell':
            indices.append(selected.index(settings['nucleus_channel']))
        own = expected[0][..., indices].astype(np.float32)
        if own.max() > 1:
            own /= own.max()
        actual_pixels.append(inputs[0])
        expected_pixels.append(own)
        actual_thresholds.append(model.eval_kwargs[0]['cellprob_threshold'])
        seen.append(model.loaded_model)
        saved = root / 'masks' / (role + '_mask_stack') / 'plate1_A01_1.npy'
        assert saved.is_file()
    assert seen == [checkpoints[role] for role in roles]
    assert actual_thresholds == [.1, .2, .3]
    for actual, expected_input in zip(actual_pixels, expected_pixels):
        np.testing.assert_array_equal(actual, expected_input)
    assert (stack / 'plate1_A01_1.npy').read_bytes() == before


def test_exact_cluster_raw_cellvoyager_shared_channel_outputs_keep_all_role_associations(
        tmp_path, monkeypatch):
    import types

    import tifffile

    from spacr.crops import read_merged_plane_layout
    from spacr.io import _load_and_concatenate_arrays, preprocess_img_data

    root = tmp_path / 'plate1'
    root.mkdir()
    rows, cols = np.indices((24, 26))
    raw_planes = [(100 + rows * (index + 3) + cols * (7 - index)).astype(np.uint16)
                  for index in range(4)]
    original_bytes = {}
    for index, plane in enumerate(raw_planes, 1):
        path = root / f'plate1_A01_T0001F001L01A01Z01C0{index}.tif'
        tifffile.imwrite(path, plane)
        original_bytes[path.name] = path.read_bytes()
    roles = ('cell', 'nucleus', 'pathogen')
    checkpoints = {}
    for role in roles:
        path = tmp_path / (role + '.pth')
        path.write_bytes(('role-specific-' + role).encode())
        checkpoints[role] = str(path)
    settings = set_default_settings_preprocess_generate_masks(dict(
        src=str(root), metadata_type='cellvoyager', cell_channel=0,
        nucleus_channel=0, pathogen_channel=1, organelle_channel=None,
        channels=[0, 1, 2, 3], number_of_organelles=0,
        randomize=False, timelapse=False, plot=False, save=True, verbose=False,
        n_jobs=1, batch_size=1, resume=False, save_original_images=True,
        adjust_cells=True, merge_pathogens=False, seg_qc=False,
        **{role + '_model_name': checkpoints[role] for role in roles}))
    settings, source = preprocess_img_data(settings)
    fields = sorted((root / 'stack').glob('*.npy'))
    assert len(fields) == 1
    field = np.load(fields[0], allow_pickle=False)
    assert field.shape == (24, 26, 4)
    for index, original in enumerate(raw_planes):
        np.testing.assert_array_equal(field[..., index], original)
    assert [settings['cellpose_' + role + '_channel'] for role in roles] == [0, 0, 1]
    expected_inputs = _normalize_img_batch(field[None].copy(), [0, 1], np.float32, settings)[0][..., [0, 1]]
    masks = {role: np.zeros(field.shape[:2], np.uint16) for role in roles}
    masks['cell'][2:10, 2:10] = 11
    masks['nucleus'][4:7, 4:7] = 22
    masks['pathogen'][15:19, 18:23] = 33
    received, constructed = {}, []

    class Model:
        def __init__(self, pretrained_model, **kwargs):
            self.role = next(role for role in roles if checkpoints[role] == pretrained_model)
            constructed.append((self.role, pretrained_model))
        def eval(self, x, **kwargs):
            received[self.role] = [np.asarray(image).copy() for image in x]
            return ([masks[self.role].copy() for _image in x],
                    [np.zeros(field.shape[:2]) for _image in x], None)

    monkeypatch.setattr(objects, 'cp_models', types.SimpleNamespace(CellposeModel=Model))
    monkeypatch.setattr(objects.accelerator, 'is_gpu', lambda: False)
    monkeypatch.setattr(objects.accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    for role in roles:
        objects.generate_cellpose_masks_sam(str(root / 'masks'), copy.deepcopy(settings), role, run_qc=False)
        saved = np.load(root / 'masks' / (role + '_mask_stack') / fields[0].name, allow_pickle=False)
        np.testing.assert_array_equal(saved, masks[role])
    assert constructed == [(role, checkpoints[role]) for role in roles]
    for role, indices in (('cell', [0, 0]), ('nucleus', [0]), ('pathogen', [1])):
        expected = expected_inputs[..., indices]
        if expected.max() > 1:
            expected = expected / expected.max()
        np.testing.assert_array_equal(received[role][0], expected)
    from spacr.utils import adjust_cell_masks
    mask_paths = {role: root / 'masks' / (role + '_mask_stack') / fields[0].name for role in roles}
    unchanged = {role: mask_paths[role].read_bytes() for role in ('nucleus', 'pathogen')}
    adjust_cell_masks(str(mask_paths['pathogen'].parent), str(mask_paths['cell'].parent),
                      str(mask_paths['nucleus'].parent), n_jobs=1)
    for role, before in unchanged.items():
        assert mask_paths[role].read_bytes() == before
    masks['cell'] = np.load(mask_paths['cell'], allow_pickle=False)
    _load_and_concatenate_arrays(source, settings['channels'],
                                 settings['cell_channel'], settings['nucleus_channel'],
                                 settings['pathogen_channel'], None)
    layout = read_merged_plane_layout(str(root / 'merged'))
    assert layout['intensity_channels'] == [0, 1, 2, 3]
    assert layout['mask_dims'] == {'cell': 4, 'nucleus': 5, 'pathogen': 6}
    merged = np.load(root / 'merged' / fields[0].name, allow_pickle=False)
    np.testing.assert_array_equal(merged[..., :4], field)
    for role in roles:
        np.testing.assert_array_equal(merged[..., layout['mask_dims'][role]], masks[role])
    for name, original in original_bytes.items():
        matches = [path for path in root.rglob(name) if path.is_file()]
        assert matches and any(path.read_bytes() == original for path in matches)


@pytest.mark.parametrize('generator', [objects.generate_cellpose_masks_sam,
                                       objects.generate_cellpose_masks])
def test_swapped_archive_channels_fail_before_loading_any_role_model(
        tmp_path, monkeypatch, generator):
    root = tmp_path / 'masks'
    root.mkdir()
    np.savez(root / 'stack_0_norm.npz',
             data=np.zeros((1, 20, 20, 2), dtype=np.float32),
             filenames=np.array(['plate1_A01_1.npy']),
             source_channels=np.array([2, 3], dtype=np.int64))
    settings = set_default_settings_preprocess_generate_masks(dict(
        cell_channel=3, nucleus_channel=3, pathogen_channel=2,
        organelle_channel=None, number_of_organelles=0, verbose=False,
        cell_model_name='missing-checkpoint-must-not-load'))
    monkeypatch.setattr(objects, '_pinned_object_model',
                        lambda *args, **kwargs: pytest.fail('model resolved before channel validation'))
    with pytest.raises(ValueError, match='Normalized image channels do not match'):
        generator(str(root), settings, 'cell')
