"""Real normalized archives feed each role's checkpoint its own raw channel."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tests.cellpose_api_contract import MISSING_CHANNEL_AXIS
from tests.conftest import check_cellpose_eval_call

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
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
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


@pytest.mark.parametrize('raw_channels', [(cell, nucleus, pathogen)
    for cell in range(4) for nucleus in range(4) for pathogen in range(4)])
def test_exhaustive_four_channel_role_routing_across_uneven_archives(
        tmp_path, monkeypatch, raw_channels, archive_counts=(2, 3, 1), batch_size=2, save=False):
    import types

    from spacr import io

    roles = ('cell', 'nucleus', 'pathogen')
    folder = tmp_path / 'masks'
    folder.mkdir()
    checkpoints = {}
    for role in roles:
        path = tmp_path / (role + '.pth')
        path.write_bytes(('controlled-' + role).encode())
        checkpoints[role] = str(path)
    settings = set_default_settings_preprocess_generate_masks(dict(
        src=str(tmp_path), cell_channel=raw_channels[0], nucleus_channel=raw_channels[1],
        pathogen_channel=raw_channels[2], organelle_channel=None, number_of_organelles=0,
        batch_size=batch_size, randomize=False, timelapse=False, plot=False, save=save,
        verbose=False, n_jobs=1, resume=False, seg_qc=False,
        **{role + '_model_name': checkpoints[role] for role in roles}))
    selected = list(dict.fromkeys(settings[key] for key in MASK_CHANNEL_ROLE_ORDER
                                  if settings.get(key) is not None))
    rows, cols = np.indices((8, 9))
    raw_fields = {f'plate1_A01_{field + 1}.npy': np.stack([
        .0005 * (field + 1) + .1 * channel + .001 * (rows * (channel + 1) + cols)
        for channel in range(4)], axis=-1).astype(np.float32) for field in range(sum(archive_counts))}
    filenames = list(raw_fields)
    archives = []
    start = 0
    for archive, count in enumerate(archive_counts):
        names = filenames[start:start + count]
        path = folder / f'stack_{archive}_norm.npz'
        np.savez(path, data=np.stack([raw_fields[name][..., selected] for name in names]),
                 filenames=np.asarray(names), source_channels=np.asarray(selected, dtype=np.int64))
        archives.append(str(path))
        start += count
    original = {path: Path(path).read_bytes() for path in archives}
    active, seen = {}, {}

    class Model:
        def __init__(self, pretrained_model, **kwargs):
            self.role = next(role for role in roles if checkpoints[role] == pretrained_model)
            assert self.role == active['role']
            active['constructed'].append(pretrained_model)
        def eval(self, x, batch_size=8, resample=True, channels=None,
                 channel_axis=MISSING_CHANNEL_AXIS, z_axis=None,
                 normalize=True, invert=False, rescale=None, diameter=None,
                 flow_threshold=0.4, cellprob_threshold=0.0, do_3D=False,
                 anisotropy=None, flow3D_smooth=0, stitch_threshold=0.0,
                 min_size=15, max_size_fraction=0.4, niter=None,
                 augment=False, tile_overlap=0.1, bsize=256,
                 compute_masks=True, progress=None):
            check_cellpose_eval_call(x, channel_axis)
            offset = active['evaluated']
            expected_names = filenames[offset:offset + len(x)]
            for image, name in zip(x, expected_names):
                raw = raw_fields[name]
                channels = [settings[self.role + '_channel']]
                if self.role == 'cell' and len(selected) > 1:
                    channels.append(settings['nucleus_channel'])
                np.testing.assert_array_equal(image, raw[..., channels])
                seen.setdefault((active['stale'], self.role), []).append(name)
            active['eval_lengths'].append(len(x))
            active['evaluated'] += len(x)
            masks = [np.zeros(image.shape[:2], np.uint16) for image in x]
            for mask, name in zip(masks, expected_names):
                mask[2:5, 2:5] = 1000 * (roles.index(self.role) + 1) + filenames.index(name) + 1
            return masks, [np.zeros(image.shape[:2]) for image in x], None

    def check_before_filters(masks, intensity_images, settings, object_type, batch_filenames):
        assert object_type == active['role']
        for image, name, mask in zip(intensity_images, batch_filenames, masks):
            channels = [settings[object_type + '_channel']]
            if object_type == 'cell' and len(selected) > 1:
                channels.append(settings['nucleus_channel'])
            np.testing.assert_array_equal(image, raw_fields[name][..., channels])
            assert mask[2, 2] == 1000 * (roles.index(object_type) + 1) + filenames.index(name) + 1
            active['filtered'].append(name)
        return masks

    monkeypatch.setattr(objects, 'cp_models', types.SimpleNamespace(CellposeModel=Model))
    monkeypatch.setattr(objects, 'merge_split_filter_masks', check_before_filters)
    monkeypatch.setattr(objects.accelerator, 'is_gpu', lambda: False)
    monkeypatch.setattr(objects.accelerator, 'cellpose_kwargs', lambda: {'gpu': False})
    monkeypatch.setattr(objects.gc, 'collect', lambda: 0)
    monkeypatch.setattr(io, '_create_database', lambda *args, **kwargs: None)
    monkeypatch.setattr(io, '_save_object_counts_to_database', lambda *args, **kwargs: None)
    for stale in (False, True):
        if save and stale:
            for role in roles:
                for output in (folder / (role + '_mask_stack')).glob('*.npy'):
                    output.unlink()
        for role in roles:
            worker = copy.deepcopy(settings)
            if stale:
                for changed in roles:
                    worker['cellpose_' + changed + '_channel'] = (worker[changed + '_channel'] + 1) % 4
            active.update(role=role, stale=stale, evaluated=0, filtered=[], constructed=[], eval_lengths=[])
            objects.generate_cellpose_masks_sam(str(folder), json.loads(json.dumps(worker)), role,
                                               batch_paths=archives, run_qc=False)
            assert active['constructed'] == [checkpoints[role]]
            assert active['evaluated'] == len(filenames)
            assert active['filtered'] == filenames
            assert seen[(stale, role)] == filenames
            assert active['eval_lengths'] == [min(batch_size, count - offset)
                for count in archive_counts for offset in range(0, count, batch_size)]
            if save:
                for index, name in enumerate(filenames):
                    mask = np.load(folder / (role + '_mask_stack') / name, allow_pickle=False)
                    label = 1000 * (roles.index(role) + 1) + index + 1
                    assert mask[2, 2] == label
                    np.testing.assert_array_equal(np.unique(mask[mask > 0]), [label])
    assert all(Path(path).read_bytes() == contents for path, contents in original.items())


def test_single_archive_103_fields_keeps_mask_names_across_50_image_offsets(tmp_path, monkeypatch):
    test_exhaustive_four_channel_role_routing_across_uneven_archives(
        tmp_path, monkeypatch, (0, 0, 1), archive_counts=(103,), batch_size=50, save=True)
