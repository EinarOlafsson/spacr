"""Positive and broken counterparts for the reachable Measure preview tour."""
from copy import deepcopy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from measure_controls_evidence import check_controls
from measure_evidence import _check_preview_sources


def example():
    crop = {'label': 4, 'area': 12000, 'category': 'cell', 'shape': [120, 100, 3],
            'sha256': 'colour', 'max': 255, 'equal_rgb': False}
    original = {'source': 'field1.npy', 'params': {'display_channel': None}, 'objects': [crop]}
    single = deepcopy(original)
    single['params']['display_channel'] = 0
    single['objects'][0].update(sha256='channel-zero', equal_rgb=True)
    second = deepcopy(original)
    second['source'] = 'field2.npy'
    return {'source_unchanged': True, 'batch_settings_restored': True,
            'batch_started': False, 'crop_dialog_opened': False,
            'application_layout_fixed': False, 'original': original,
            'single_channel': single, 'restored': deepcopy(original), 'second_field': second}


def test_real_shape_positive_counterpart():
    check_controls(example())


def test_preview_and_completed_measurements_share_the_exact_source_arrays():
    proof = example()
    proof['source_hashes'] = {'field1.npy': 'first', 'field2.npy': 'second'}
    result = _check_preview_sources(proof, {'field1': 'first', 'field2': 'second'})
    assert result['preview_source_arrays_preserved_after_batch'] is True
    assert result['live_grid_counts'] == [1, 1, 1]


@pytest.mark.parametrize('fault', ['missing', 'changed', 'extra', 'changed_controls'])
def test_preview_evidence_cannot_substitute_other_arrays_or_controls(fault):
    proof = example()
    proof['source_hashes'] = {'field1.npy': 'first', 'field2.npy': 'second'}
    if fault == 'missing':
        del proof['source_hashes']['field2.npy']
    elif fault == 'changed':
        proof['source_hashes']['field2.npy'] = 'different'
    elif fault == 'extra':
        proof['source_hashes']['field3.npy'] = 'third'
    else:
        proof['single_channel']['objects'][0]['area'] += 1
    with pytest.raises(ValueError):
        _check_preview_sources(proof, {'field1': 'first', 'field2': 'second'})


@pytest.mark.parametrize('flag', ['source_unchanged', 'batch_settings_restored',
                                 'batch_started', 'crop_dialog_opened', 'application_layout_fixed'])
def test_scope_cannot_be_widened(flag):
    proof = example()
    proof[flag] = not proof[flag]
    with pytest.raises(ValueError):
        check_controls(proof)


@pytest.mark.parametrize('kind', ['empty', 'not_restored', 'same_field', 'other_channel',
                                 'identity_changed', 'black', 'not_grey', 'unchanged_pixels'])
def test_each_named_claim_has_a_broken_counterpart(kind):
    proof = example()
    if kind == 'empty':
        proof['second_field']['objects'] = []
    elif kind == 'not_restored':
        proof['restored']['objects'][0]['sha256'] = 'other'
    elif kind == 'same_field':
        proof['second_field']['source'] = proof['original']['source']
    elif kind == 'other_channel':
        proof['single_channel']['params']['display_channel'] = 2
    elif kind == 'identity_changed':
        proof['single_channel']['objects'][0]['label'] = 3
    elif kind == 'black':
        proof['single_channel']['objects'][0]['max'] = 0
    elif kind == 'not_grey':
        proof['single_channel']['objects'][0]['equal_rgb'] = False
    else:
        proof['single_channel']['objects'][0]['sha256'] = 'colour'
    with pytest.raises(ValueError):
        check_controls(proof)
