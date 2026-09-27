"""Native preference changes must never erase or replace captured results."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from capture_external_masks import check_retained_console_state


def state():
    return {
        'settings': {'cytoplasm': True, 'channels': [0], 'dst': '/private/project'},
        'figure_count': 6,
        'figures': [bytes([index]) for index in range(6)],
        'console_blocks': [
            {'kind': 'stdout', 'text': 'Preview: 2 fields, nothing written'},
            {'kind': 'stdout', 'text': 'Prepared 2 fields. tables: cell, cytoplasm'},
            {'kind': 'stdout', 'text': 'Finished'},
        ],
    }


def test_exact_results_survive_a_native_refresh_without_mutation():
    before = state()
    after = deepcopy(before)
    snapshot = deepcopy((before, after))
    check_retained_console_state(before, after)
    assert (before, after) == snapshot


@pytest.mark.parametrize('which', ['before', 'after'])
@pytest.mark.parametrize('count', [0, 5, 7])
def test_both_states_require_exactly_six_figures(which, count):
    states = {'before': state(), 'after': state()}
    states[which]['figure_count'] = count
    with pytest.raises(RuntimeError, match='all six figures'):
        check_retained_console_state(**states)


@pytest.mark.parametrize('which', ['before', 'after'])
def test_reported_count_cannot_hide_a_missing_image(which):
    states = {'before': state(), 'after': state()}
    states[which]['figures'].pop()
    with pytest.raises(RuntimeError, match='all six figures'):
        check_retained_console_state(**states)


@pytest.mark.parametrize('change', ['content', 'order'])
def test_same_count_cannot_hide_changed_figure_content_or_order(change):
    before, after = state(), state()
    if change == 'content':
        after['figures'][3] = b'not the original pixels'
    else:
        after['figures'].reverse()
    with pytest.raises(RuntimeError, match='ordered figure images'):
        check_retained_console_state(before, after)


@pytest.mark.parametrize('change', ['content', 'order', 'kind', 'missing', 'added'])
def test_exact_ordered_console_history_is_required(change):
    before, after = state(), state()
    if change == 'content':
        after['console_blocks'][1]['text'] = 'Finished'
    elif change == 'order':
        after['console_blocks'].reverse()
    elif change == 'kind':
        after['console_blocks'][1]['kind'] = 'traceback'
    elif change == 'missing':
        after['console_blocks'].pop(0)
    else:
        after['console_blocks'].append({'kind': 'stdout', 'text': 'replacement'})
    with pytest.raises(RuntimeError, match='console history'):
        check_retained_console_state(before, after)


@pytest.mark.parametrize('blocks', [[], [{'kind': 'stdout', 'text': ''}]])
def test_empty_history_is_not_a_valid_baseline(blocks):
    before = state()
    before['console_blocks'] = blocks
    with pytest.raises(RuntimeError, match='actual console text'):
        check_retained_console_state(before, deepcopy(before))


def test_preferences_may_not_change_a_nested_measurement_setting():
    before, after = state(), state()
    after['settings']['channels'].append(1)
    with pytest.raises(RuntimeError, match='measurement settings'):
        check_retained_console_state(before, after)


@pytest.fixture
def retained_inputs(tmp_path):
    import hashlib
    from capture_external_masks import private_input_records

    stage = tmp_path / 'stage'
    root = stage / 'retained'
    old_root = Path('/original/tutorial')
    record = {'neutral_stem': 'fov01', 'objects': 44}
    for key, relative in (
        ('source', 'example_data/plate1/merged/field.npy'),
        ('image', 'foreign_runs/example/images/fov01_C1.tif'),
        ('mask', 'foreign_runs/example/cell_masks/fov01_cell_mask.tif'),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key.encode())
        record[key] = str(old_root / relative)
        record[key + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'run': str(old_root / 'foreign_runs/example'), 'records': [record]}
    return private_input_records, manifest, stage, root


def test_private_input_paths_preserve_manifest_and_every_original_hash(retained_inputs):
    resolve, manifest, stage, root = retained_inputs
    before = deepcopy(manifest)
    records = resolve(manifest, stage, root)
    assert manifest == before
    assert records[0]['objects'] == 44
    for key in ('source', 'image', 'mask'):
        assert Path(records[0][key]).is_relative_to(root)
        assert records[0][key + '_sha256'] == before['records'][0][key + '_sha256']
    assert resolve(manifest, stage, None) == before['records']


@pytest.mark.parametrize('key', ['source', 'image', 'mask'])
def test_private_input_changed_bytes_are_rejected(retained_inputs, key):
    resolve, manifest, stage, root = retained_inputs
    records = resolve(manifest, stage, root)
    Path(records[0][key]).write_bytes(b'different pixels')
    with pytest.raises(ValueError, match='accepted source hash'):
        resolve(manifest, stage, root)


@pytest.mark.parametrize('escape', ['outside-root', 'original-path', 'symlink', 'traversal'])
def test_private_input_paths_cannot_escape_recording(retained_inputs, tmp_path, escape):
    resolve, manifest, stage, root = retained_inputs
    records = resolve(manifest, stage, root)
    if escape == 'outside-root':
        root = tmp_path
    elif escape == 'original-path':
        manifest['records'][0]['image'] = '/other/fov01_C1.tif'
    elif escape == 'traversal':
        manifest['records'][0]['image'] = '/original/tutorial/../../../other.tif'
    else:
        image = Path(records[0]['image'])
        outside = tmp_path / 'outside.tif'
        outside.write_bytes(image.read_bytes())
        image.unlink()
        image.symlink_to(outside)
    with pytest.raises(ValueError, match='inside the private stage|original workspace|private input root'):
        resolve(manifest, stage, root)
