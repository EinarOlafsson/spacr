"""Exact field normalization survives saved settings and changing batch membership."""

import json
import pickle

import numpy as np
import pytest

from spacr import _mask_batches as batches
from spacr import io


def _settings(**overrides):
    """Return a small ordinary two-channel Mask Generation configuration."""
    settings = {
        'background': 20, 'Signal_to_noise': 3, 'remove_background': False,
        'lower_percentile': 2, 'nucleus_channel': None, 'cell_channel': None,
        'pathogen_channel': None, 'organelle_channel': None,
        'timelapse': False, 'randomize': False, 'batch_size': 2,
        'plot': False, 'figuresize': 4, 'cmap': 'inferno', 'nr': 1,
    }
    settings.update(overrides)
    return settings


def _fields():
    """Create distinct field distributions whose cohort and isolated bounds differ."""
    first = np.arange(128, dtype=np.uint16).reshape(8, 8, 2) + 3
    second = first * 9 + 200
    return np.stack([first, second])


def _saved(settings):
    """Simulate saved JSON settings, losing runtime fresh-recording state."""
    return json.loads(json.dumps(settings))


def _no_estimation(*args, **kwargs):
    """Fail if a saved replay attempts percentile estimation or donor loading."""
    raise AssertionError('saved normalization must not estimate')


def test_saved_cohort_bounds_replay_each_field_without_percentile_estimation(monkeypatch):
    """Saved fields retain exact float bytes alone and in a different order."""
    raw = _fields()
    settings = _settings()
    expected = io._normalize_img_batch(raw.copy(), [0, 1], np.float32, _settings())
    actual = io._normalize_img_batch(
        raw.copy(), [0, 1], np.float32, settings,
        field_ids=['plate1_A01_1', 'plate1_A01_2'])
    assert actual.tobytes() == expected.tobytes()
    isolated_fresh = io._normalize_img_batch(
        raw[:1].copy(), [0, 1], np.float32, _settings(), field_ids=['plate1_A01_1'])
    assert isolated_fresh.tobytes() != actual[:1].tobytes()
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    replay = _saved(settings)
    for index in (1, 0):
        restored = io._normalize_img_batch(
            raw[index:index + 1].copy(), [0, 1], np.float32, replay,
            field_ids=[f'plate1_A01_{index + 1}'])
        assert restored.tobytes() == actual[index:index + 1].tobytes()
    reordered = io._normalize_img_batch(
        raw[::-1].copy(), [0, 1], np.float32, _saved(settings),
        field_ids=['plate1_A01_2', 'plate1_A01_1'])
    assert reordered.tobytes() == actual[::-1].tobytes()


@pytest.mark.parametrize('change', ['pixels', 'field', 'floor', 'channels', 'missing'])
def test_saved_normalization_refuses_unrelated_source_configuration_or_field(monkeypatch, change):
    """Replay rejects every mismatch before a new percentile can be estimated."""
    raw = _fields()[:1]
    settings = _settings()
    io._normalize_img_batch(raw.copy(), [0, 1], np.float32, settings, field_ids=['field1'])
    replay = _saved(settings)
    identity = 'field1'
    channels = [0, 1]
    if change == 'pixels':
        raw[0, 0, 0, 0] += 1
    elif change == 'field':
        identity = 'field2'
    elif change == 'floor':
        replay['background'] += 1
    elif change == 'channels':
        channels = [1, 0]
    else:
        replay['spacr_resolved_parameters']['fields'].clear()
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    with pytest.raises(ValueError, match='saved run'):
        io._normalize_img_batch(raw.copy(), channels, np.float32, replay, field_ids=[identity])


def test_saved_normalization_preserves_empty_channels_and_background_removal(monkeypatch):
    """Explicit empty-channel bounds replay zero pixels without fitting."""
    raw = _fields()
    raw[..., 1] = 0
    settings = _settings(remove_background=True, background=80)
    expected = io._normalize_img_batch(
        raw.copy(), [0, 1], np.float32, settings, field_ids=['one', 'two'])
    bounds = settings['spacr_resolved_parameters']['fields']['one']['normalization']['value']['bounds']
    assert bounds['1'] == [0.0, 0.0]
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    actual = io._normalize_img_batch(
        raw[:1].copy(), [0, 1], np.float32, _saved(settings), field_ids=['one'])
    assert actual.tobytes() == expected[:1].tobytes()


def test_saved_padding_geometry_is_retained_and_changed_extent_is_refused(monkeypatch):
    """Unpadded source identity remains stable while model-input geometry is enforced."""
    raw = _fields()
    small = raw[0, :4, :5].copy()
    raw[0] = np.pad(small, ((0, 4), (0, 3), (0, 0)))
    settings = _settings()
    expected = io._normalize_img_batch(
        raw.copy(), [0, 1], np.float32, settings, field_ids=['small', 'large'],
        field_shapes=[(4, 5), (8, 8)])
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    restored = io._normalize_img_batch(
        raw[:1].copy(), [0, 1], np.float32, _saved(settings),
        field_ids=['small'], field_shapes=[(4, 5)])
    assert restored.tobytes() == expected[:1].tobytes()
    with pytest.raises(ValueError, match='requires padded shape'):
        io._normalize_img_batch(
            small[None].copy(), [0, 1], np.float32, _saved(settings), field_ids=['small'])


def test_separate_v2_roles_retain_original_channel_mapping(monkeypatch):
    """Role-specific records coexist and reject a changed original channel selection."""
    raw = _fields()
    settings = _settings()
    expected = {}
    for role, channel in [('cell', 0), ('nucleus', 1)]:
        expected[role] = io._normalize_img_batch(
            raw[..., channel:channel + 1].copy(), [0], np.float32, settings,
            field_ids=['one', 'two'], source_channels=[channel], normalization_scope=role)
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    for role, channel in [('cell', 0), ('nucleus', 1)]:
        actual = io._normalize_img_batch(
            raw[:1, ..., channel:channel + 1].copy(), [0], np.float32, _saved(settings),
            field_ids=['one'], source_channels=[channel], normalization_scope=role)
        assert actual.tobytes() == expected[role][:1].tobytes()
    with pytest.raises(ValueError, match='configuration'):
        io._normalize_img_batch(
            raw[:1, ..., :1].copy(), [0], np.float32, _saved(settings),
            field_ids=['one'], source_channels=[3], normalization_scope='cell')


def _plan(tmp_path, settings, *, borrowed=False):
    """Write an in-memory plan with two donor fields and an optional tiny batch."""
    stack = tmp_path / 'stack'
    masks = tmp_path / 'masks'
    stack.mkdir()
    masks.mkdir()
    names = ['plate1_A01_1.npy', 'plate1_A01_2.npy']
    for name, field in zip(names, _fields()):
        np.save(stack / name, field)
    groups = [names]
    if borrowed:
        extra_name = 'plate1_A01_3.npy'
        np.save(stack / extra_name, _fields()[0] * 3)
        groups.append([extra_name])
    batches._write_plan(
        masks, stack, groups, [0, 1], settings, np.float32,
        extra={'borrowed': [1] if borrowed else [], 'normalisation': {}})
    return masks, batches._read_plan(masks)


def test_in_memory_plan_retains_fresh_worker_context_after_json_reload(tmp_path):
    """A pickled fresh context still records fields after a worker rereads its plan."""
    settings = _settings()
    masks, _ = _plan(tmp_path, settings)
    worker_settings = pickle.loads(pickle.dumps(settings))
    batches._CACHE.clear()
    with batches._open_batch(masks / 'stack_0_norm.npz', settings=worker_settings) as loaded:
        assert set(loaded['spacr_resolved_parameters']['fields']) == {'plate1_A01_1', 'plate1_A01_2'}
        assert loaded['data'].shape == (2, 8, 8, 2)
    assert worker_settings['spacr_resolved_parameters'].replay is False


def test_plan_input_signature_stays_stable_while_field_records_accumulate(tmp_path):
    """Recording effective bounds cannot invalidate unchanged worker input identities."""
    masks, plan = _plan(tmp_path, _settings())
    before = batches._plan_records(plan)
    batches._load_batch(plan, 0)
    after = batches._plan_records(batches._read_plan(masks))
    assert after == before
    changed = dict(batches._read_plan(masks))
    changed['settings'] = dict(changed['settings'], lower_percentile=5)
    assert batches._plan_records(changed) != before


@pytest.mark.parametrize('signature_kind', ['plan', 'model'])
def test_saved_replay_verifies_matching_receipts_and_refuses_other_cohorts(
        tmp_path, signature_kind):
    """Fresh completed masks replay directly while another valid cohort is refused."""
    from spacr._mask_workers import _MaskBatchLedger, _prepare_mask_model
    from spacr.run_journal import _record_resolved_parameter, _resolved_source

    weights = tmp_path / 'checkpoint.pth'
    weights.write_bytes(b'fixture-model-identity')
    settings = _settings(cell_model_name=str(weights))
    masks, plan = _plan(tmp_path, settings)

    def signature(current):
        """Obtain the real plan or model identity using the supplied replay context."""
        if signature_kind == 'model':
            return _prepare_mask_model(current, 'cell')[1]
        current_plan = dict(plan, settings=current, _runtime_settings=current)
        return batches._plan_records(current_plan)[0]['sha256']

    before = signature(settings)
    normalized = batches._load_batch(plan, 0)['data']
    for index, image in enumerate(normalized):
        _record_resolved_parameter(settings, f'plate1_A01_{index + 1}', 'diameter:cell',
                                   _resolved_source(image), {'mode': 'fixture'},
                                   None if index == 0 else 23.125)
    assert signature(settings) == before
    records = batches._plan_records(plan)
    ledger = _MaskBatchLedger(str(masks), records, 'cell', before)
    ledger.output_root.mkdir()
    for name in records[0]['fields']:
        np.save(ledger.output_root / name, np.ones((8, 8), dtype=np.uint16))
    ledger.mark(records[0]['path'], settings=settings)
    receipt = ledger.store.completed[records[0]['path']]
    assert set(receipt) == set(records[0]['fields']) | {'spacr_resolved_parameters'}
    recorded = receipt['spacr_resolved_parameters']['fields']
    assert set(recorded) == {'plate1_A01_1', 'plate1_A01_2'}
    assert all(set(values) == {'normalization', 'diameter:cell'} for values in recorded.values())
    _record_resolved_parameter(settings, 'another_field', 'diameter:cell', {}, {}, 99.0)
    _record_resolved_parameter(settings, 'plate1_A01_1', 'diameter:nucleus', {}, {}, 12.0)
    cohort = _saved(settings)
    assert signature(cohort) == before
    resumed = _MaskBatchLedger(str(masks), records, 'cell', signature(cohort))
    assert resumed.verified(settings=cohort) == {records[0]['path']}
    hydrated = _settings(cell_model_name=str(weights))
    assert resumed.verified(settings=hydrated) == {records[0]['path']}
    assert hydrated['spacr_resolved_parameters']['fields'] == recorded
    isolated = _settings(cell_model_name=str(weights))
    for index in range(2):
        normalized = io._normalize_img_batch(
            _fields()[index:index + 1].copy(), [0, 1], np.float32, isolated,
            field_ids=[f'plate1_A01_{index + 1}'])
        _record_resolved_parameter(isolated, f'plate1_A01_{index + 1}', 'diameter:cell',
                                   _resolved_source(normalized[0]), {'mode': 'fixture'},
                                   None if index == 0 else 23.125)
    isolated = _saved(isolated)
    first = cohort['spacr_resolved_parameters']['fields']['plate1_A01_1']['normalization']
    second = isolated['spacr_resolved_parameters']['fields']['plate1_A01_1']['normalization']
    assert first['source'] == second['source']
    assert first['configuration'] == second['configuration']
    assert first['value']['bounds'] != second['value']['bounds']
    assert signature(isolated) == before
    mismatched = _MaskBatchLedger(str(masks), records, 'cell', signature(isolated))
    before_refusal = mismatched.store.path.read_bytes()
    with pytest.raises(ValueError, match='Completed mask parameters'):
        mismatched.verified(settings=isolated)
    assert mismatched.store.path.read_bytes() == before_refusal


@pytest.mark.parametrize('missing', ['receipt', 'saved_diameter', 'saved_normalization'])
def test_saved_replay_cannot_reuse_receipts_with_missing_parameters(tmp_path, missing):
    """Complete mask files alone cannot establish the missing effective values."""
    from spacr._mask_workers import _MaskBatchLedger
    from spacr.run_journal import _record_resolved_parameter

    settings = _settings()
    masks, plan = _plan(tmp_path, settings)
    batches._load_batch(plan, 0)
    for identity in ('plate1_A01_1', 'plate1_A01_2'):
        _record_resolved_parameter(settings, identity, 'diameter:cell', {}, {}, None)
    records = batches._plan_records(plan)
    ledger = _MaskBatchLedger(str(masks), records, 'cell', 'a' * 64)
    ledger.output_root.mkdir()
    for name in records[0]['fields']:
        np.save(ledger.output_root / name, np.ones((8, 8), dtype=np.uint16))
    ledger.mark(records[0]['path'], settings=None if missing == 'receipt' else settings)
    replay = _saved(settings)
    if missing != 'receipt':
        kind = 'diameter:cell' if missing == 'saved_diameter' else 'normalization'
        replay['spacr_resolved_parameters']['fields']['plate1_A01_1'].pop(kind)
    with pytest.raises(ValueError, match='Completed mask parameters|no resolved diameter:cell'):
        ledger.verified(settings=replay)


def test_tiny_batch_saved_bounds_replay_without_loading_donors(tmp_path, monkeypatch):
    """Borrowed median bounds become exact field records independent of donor availability."""
    settings = _settings()
    masks, plan = _plan(tmp_path, settings, borrowed=True)
    batches._load_batch(plan, 0)
    expected = batches._load_batch(plan, 1)['data']
    replay = _saved(settings)
    monkeypatch.setattr(batches, '_borrowed_values', _no_estimation)
    monkeypatch.setattr(io.np, 'percentile', _no_estimation)
    for name in plan['batches'][0]['fields']:
        (tmp_path / 'stack' / name).unlink()
    batches._CACHE.clear()
    with batches._open_batch(masks / 'stack_1_norm.npz', settings=replay) as loaded:
        assert loaded['data'].tobytes() == expected.tobytes()


def test_parent_batch_loop_merges_normalization_before_running_roles(tmp_path, monkeypatch):
    """Background loader records reach parent settings before a model consumes the batch."""
    from spacr.run_journal import _resolved_parameters

    settings = _settings()
    masks, plan = _plan(tmp_path, settings)
    parent = _settings()
    _resolved_parameters(parent)
    published = []
    monkeypatch.setattr('spacr.run_journal._publish_resolved_settings',
                        lambda current: published.append(_saved(current)))
    worker_plan = dict(plan, _runtime_settings=pickle.loads(pickle.dumps(settings)))
    feeder = batches._BatchFeeder(
        worker_plan, [0], store=lambda array: None, wait_for_ram=lambda estimate: None)
    calls = []

    def consume(role, path):
        """Confirm the parent owns field records before role evaluation starts."""
        calls.append((role, set(parent['spacr_resolved_parameters']['fields'])))

    batches._batch_first(
        masks, [masks / 'stack_0_norm.npz'], ['cell'], consume,
        feeder=feeder, settings=parent)
    assert calls == [('cell', {'plate1_A01_1', 'plate1_A01_2'})]
    assert published[-1]['spacr_resolved_parameters'] == _saved(parent)['spacr_resolved_parameters']
