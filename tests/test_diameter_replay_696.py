"""Exact per-field diameters replay without invoking a size estimator again."""
import json

import numpy as np
import pytest

from spacr import object as objects
from spacr.cli import load_settings_file, write_settings_file
from spacr.settings import RESOLVED_PARAMETERS_KEY


class _Model:
    """Expose exact input-dependent size estimates and all probability flows."""

    def __init__(self, checkpoint=None):
        """Keep call evidence and an optional actual checkpoint identity."""
        self.pretrained_model = str(checkpoint) if checkpoint is not None else None
        self.calls = []
        self.estimated_diameters = []
        self.allow_estimate = True

    def eval(self, images, *, diameter=None, **kwargs):
        """Return deterministic masks and probabilities at the requested scale."""
        self.calls.append((diameter, [np.asarray(image).copy() for image in images]))
        self.estimated_diameters = []
        masks, flows = [], []
        for image in images:
            marker = float(np.asarray(image).flat[0])
            if diameter == 0:
                assert self.allow_estimate, 'the size model ran during replay'
                used = marker * 13.125000000000002
                self.estimated_diameters.append(used)
            else:
                used = diameter
            value = int(round(used * 1000)) if used is not None else int(marker)
            shape = np.asarray(image).shape[:2]
            masks.append(np.full(shape, value, np.uint32))
            flows.append([np.full((*shape, 3), value, np.uint32),
                          np.full((2, *shape), value, np.float64),
                          np.full(shape, value / 100.0, np.float64), None])
        return masks, flows, None


def _images():
    """Return distinguishable spatial inputs with identical field geometry."""
    return [np.full((5, 6, 1), marker, np.float32) for marker in (1, 2, 3)]


def _plain(settings):
    """Emulate the plain JSON payload that switches a saved context to replay."""
    return json.loads(json.dumps(settings))


def _forbid_estimate(*args, **kwargs):
    """Refuse any first-pass measurement during saved-run replay."""
    pytest.fail('the SAM diameter estimator ran during replay')


@pytest.mark.parametrize('extension', ['.csv', '.json'])
def test_sam_saved_diameters_replay_alone_reordered_and_with_native_none(
        monkeypatch, tmp_path, extension):
    """Saved per-image sizes, including fallback None, bypass the first pass."""
    from cellpose import utils as cellpose_utils

    estimates = {1: 13.125000000000002, 2: 27.500000000000004, 3: 0.0}

    def measure(mask):
        """Measure a different exact diameter for each first-pass image."""
        return estimates[int(mask.flat[0])], None

    monkeypatch.setattr(cellpose_utils, 'diameters', measure)
    settings = {'nucleus_diameter': 'auto', 'estimate_diameters_per_image': True}
    ids = ['plate1_A01_1_T0001.npy', 'plate1_A01_1_T0002.npy', 'plate1_A01_2.npy']
    images = _images()
    options = {'normalize': False, 'channel_axis': -1, 'flow_threshold': 0.4}
    model = _Model()
    masks, flows, used = objects._per_image_diameter_masks(
        model, images, options, None, settings=settings,
        field_ids=ids, object_type='nucleus')
    assert used == [estimates[1], estimates[2], None]
    assert len(model.calls) == 6
    fields = settings[RESOLVED_PARAMETERS_KEY]['fields']
    assert set(fields) == {name[:-4] for name in ids}
    assert fields[ids[2][:-4]]['diameter:nucleus']['value'] is None
    destination = tmp_path / 'mask.csv'
    write_settings_file(settings, destination)
    monkeypatch.setattr(cellpose_utils, 'diameters', _forbid_estimate)
    replay_model = _Model()
    for indices in ([1], [2, 0, 1]):
        loaded = load_settings_file(str(destination.with_suffix(extension)))
        loaded['batch_size'] = 97
        result = objects._per_image_diameter_masks(
            replay_model, [images[index] for index in indices], options, None,
            settings=loaded, field_ids=[ids[index] for index in indices],
            object_type='nucleus')
        assert result[2] == [used[index] for index in indices]
        for position, index in enumerate(indices):
            np.testing.assert_array_equal(result[0][position], masks[index])
            np.testing.assert_array_equal(result[1][position], flows[index])
    assert [call[0] for call in replay_model.calls] == [used[1], used[2], used[0], used[1]]


def test_cellpose3_records_each_returned_size_and_replays_probability_order():
    """Two distinct size-model results survive reordered single-field inference."""
    settings = {'cellpose3_size_model': True, 'nucleus_diameter': 'auto'}
    images = _images()[:2]
    ids = ['plate1_A01_1.npy', 'plate1_A01_2.npy']
    model = _Model()
    original = objects._cellpose3_masks(
        model, images, settings, 'nucleus', min_size=1, default_diameter=60,
        field_ids=ids, probabilities=True)
    expected = [13.125000000000002, 26.250000000000004]
    assert [settings[RESOLVED_PARAMETERS_KEY]['fields'][name[:-4]]
            ['diameter:nucleus']['value'] for name in ids] == expected
    assert model.calls[0][0] == 0.0
    replay = _Model()
    replay.allow_estimate = False
    result = objects._cellpose3_masks(
        replay, images[::-1], _plain(settings), 'nucleus', min_size=1,
        default_diameter=60, batch_size=1, field_ids=ids[::-1], probabilities=True)
    assert [call[0] for call in replay.calls] == expected[::-1]
    for part in range(3):
        for actual, wanted in zip(result[part], original[part][::-1]):
            np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize('diameter', [None, 26.245000000000005])
def test_fixed_and_native_diameters_have_exact_field_provenance(diameter):
    """Ordinary spatial evaluation records native None without estimating it."""
    settings = {}
    images, ids = _images()[:2], ['field_1', 'field_2']
    original = objects._diameter_eval(
        _Model(), images, settings=settings, object_type='cell', field_ids=ids,
        site='mask_sam', diameter=diameter, batch_size=2, normalize=False)
    replay_model = _Model()
    replay = objects._diameter_eval(
        replay_model, images[::-1], settings=_plain(settings), object_type='cell',
        field_ids=ids[::-1], site='mask_sam', diameter=diameter,
        batch_size=17, normalize=False)
    assert [call[0] for call in replay_model.calls] == [diameter]
    for actual, wanted in zip(replay[0], original[0][::-1]):
        np.testing.assert_array_equal(actual, wanted)
    assert [record['diameter:cell']['value']
            for record in settings[RESOLVED_PARAMETERS_KEY]['fields'].values()] == [diameter] * 2


@pytest.mark.parametrize('changed', ['missing', 'pixels', 'configuration', 'checkpoint'])
def test_loaded_diameter_refuses_changed_identity_before_inference(tmp_path, changed):
    """A loaded field cannot silently estimate against a different input or model."""
    checkpoint = tmp_path / 'model'
    checkpoint.write_bytes(b'original checkpoint')
    settings, images = {}, _images()[:1]
    objects._diameter_eval(
        _Model(checkpoint), images, settings=settings, object_type='cell',
        field_ids=['field'], site='mask_sam', diameter=23.125, normalize=False)
    loaded = _plain(settings)
    ids, normalize = ['field'], False
    if changed == 'missing':
        ids = ['new_field']
    elif changed == 'pixels':
        images = [images[0] + 1]
    elif changed == 'configuration':
        normalize = True
    else:
        checkpoint.write_bytes(b'changed checkpoint')
    model = _Model(checkpoint)
    with pytest.raises(ValueError, match='saved run'):
        objects._diameter_eval(
            model, images, settings=loaded, object_type='cell', field_ids=ids,
            site='mask_sam', diameter=23.125, normalize=normalize)
    assert model.calls == []


def test_preview_without_field_ids_keeps_existing_probability_api():
    """Preview callers can still evaluate without inventing saved field identities."""
    settings = {'nucleus_diameter': 12.5}
    result = objects._cellpose3_masks(
        _Model(), _images()[:1], settings, 'nucleus', min_size=1,
        default_diameter=60, probabilities=True)
    assert len(result) == 3 and len(result[2]) == 1
    assert RESOLVED_PARAMETERS_KEY not in settings


def test_same_checkpoint_replays_between_prepared_path_and_stock_setting(tmp_path):
    """Worker path preparation does not change the identity of loaded weights."""
    checkpoint = tmp_path / 'cpsam'
    checkpoint.write_bytes(b'exact weights')
    settings = {'cell_model_name': str(checkpoint)}
    objects._diameter_eval(
        _Model(checkpoint), _images()[:1], settings=settings, object_type='cell',
        field_ids=['field'], site='mask_sam', diameter=None)
    replay = _plain(settings)
    replay['cell_model_name'] = 'cpsam'
    model = _Model(checkpoint)
    objects._diameter_eval(
        model, _images()[:1], settings=replay, object_type='cell',
        field_ids=['field'], site='mask_sam', diameter=None)
    assert [call[0] for call in model.calls] == [None]


def test_saved_replay_reprocesses_complete_masks_while_fresh_resume_skips(tmp_path):
    """Existing valid outputs cannot bypass replay identity checks or inference."""
    settings = {'nucleus_channel': 0, 'nucleus_flow_threshold': 0.4,
                'nucleus_cellprob_threshold': 0.0, 'plot': False}
    images = np.stack(_images()[:2])
    ids = ['field1.npy', 'field2.npy']
    initial = objects._segment_cellpose_sam(
        images, ids, _Model(), settings, 'nucleus', str(tmp_path))
    for name, mask in zip(ids, initial):
        np.save(tmp_path / name, mask)
    replay_model = _Model()
    repeated = objects._segment_cellpose_sam(
        images, ids, replay_model, _plain(settings), 'nucleus', str(tmp_path))
    assert len(replay_model.calls) == 1
    for actual, expected in zip(repeated, initial):
        np.testing.assert_array_equal(actual, expected)
    fresh = {key: value for key, value in settings.items()
             if key != RESOLVED_PARAMETERS_KEY}
    fresh_model = _Model()
    assert objects._segment_cellpose_sam(
        images, ids, fresh_model, fresh, 'nucleus', str(tmp_path)) is None
    assert fresh_model.calls == []


@pytest.mark.parametrize('mode', ['fixed', 'per-image'])
def test_diameter_publication_preserves_parent_role_settings(tmp_path, monkeypatch, mode):
    """Role-adapted inference publishes records without overwriting parent settings."""
    from cellpose import utils as cellpose_utils
    from spacr import run_journal
    from spacr.object_roles import organelle_settings_view

    requested = {
        'src': str(tmp_path / 'project'),
        'organelle_channel': 0, 'organelleb_channel': 1,
        'organelle_diameter': 13.125, 'organelleb_diameter': 27.5,
        'cellpose_organelle_channel': 0, 'cellpose_organelleb_channel': 1,
        'object_filters': {'organelle': [{'feature': 'area', 'min': 2}],
                           'organelleb': [{'feature': 'area', 'min': 5}]},
    }
    parent = _plain(requested)
    context = run_journal._resolved_parameters(parent)
    context.source_root = requested['src']
    local = organelle_settings_view(parent, 'organelleb')
    local['src'] = str(tmp_path / 'project' / 'masks')
    local['object_filters'] = {'organelle': requested['object_filters']['organelleb']}
    run_directory = tmp_path / 'run'
    run_directory.mkdir()
    run = run_journal.Run(app_key='mask', settings=parent, dir=run_directory)

    def active_run():
        """Expose an actual parent journal without unrelated environment capture."""
        return run

    def estimate(mask):
        """Return the exact diameter selected for this organelle field."""
        return 27.5, None

    monkeypatch.setattr(run_journal, 'current_run', active_run)
    monkeypatch.setattr(cellpose_utils, 'diameters', estimate)
    if mode == 'fixed':
        objects._diameter_eval(
            _Model(), _images()[:1], settings=local, object_type='organelleb',
            field_ids=['field'], site='mask_organelle', diameter=27.5)
    else:
        objects._per_image_diameter_masks(
            _Model(), _images()[:1], {}, None, settings=local,
            field_ids=['field'], object_type='organelleb')
    saved = json.loads((run_directory / 'settings.json').read_text('utf-8'))
    for name, value in requested.items():
        assert run.settings[name] == value
        assert saved[name] == value
    assert saved[RESOLVED_PARAMETERS_KEY]['fields']['field']['diameter:organelleb']['value'] == 27.5
