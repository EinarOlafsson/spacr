"""Protocol decisions retain real destinations and reproducible stage entry points."""
from copy import deepcopy

import pytest

from tools import build_module_workflows as workflows


def test_protocol_catalog_covers_nine_existing_pathways_and_complete_navigation():
    data = workflows.load()
    workflows._validate_protocols(data)
    assert len(data['pathways']) == 9
    assert set(data['protocol_choices']['data']) == {
        'images', 'images_masks', 'fastq', 'barcode_counts', 'features', 'crops',
        'segmentation_model', 'classifier', 'regression', 'run_records', 'planning'}
    assert {p['pathway'] for p in data['protocols'].values()} == set(data['pathways'])
    for protocol in data['protocols']:
        assert any(route['protocol'] == protocol for route in data['protocol_routes'])
    vendors = {'yokogawa', 'zeiss', 'nikon', 'leica', 'other_microscope', 'organized'}
    vendor_routes = [r for r in data['protocol_routes']
                     if r['protocol'] == 'organize_images' and r['details'] in vendors]
    assert {r['details'] for r in vendor_routes} == vendors
    assert all(r['start_module'] == 'import_images' for r in vendor_routes)


def test_prepared_routes_start_at_real_appropriate_steps_and_parallel_tools_are_not_handoffs():
    data = workflows.load()
    for route in data['protocol_routes']:
        steps = {step['module'] for step in data['protocols'][route['protocol']]['steps']}
        assert route['start_module'] in steps
        if route['protocol'] == 'tracking':
            assert route['start_module'] == ('external_masks' if route['data'] == 'images_masks' else 'timelapse')
        if route['details'] == 'reviewed_labels' and 'classify_merged' in steps:
            assert route['start_module'] == 'classify_merged'
        if route['details'] == 'merged_project':
            assert route['start_module'] == 'measure'
    for key in ('explain_classifier', 'condition_compare'):
        assert all(not step['after'] for step in data['protocols'][key]['steps'])
    execution = data['protocols']['multi_plate']['steps']
    assert not execution[0]['after'] and not execution[1]['after']
    assert set(execution[2]['after']) == {'queue', 'batch'}
    assert 'OR' in execution[2]['action']


@pytest.mark.parametrize('mutation', [
    lambda d: d['protocols']['generate_masks'].update(title=' '),
    lambda d: d['protocols']['generate_masks'].update(summary=''),
    lambda d: d['protocols']['generate_masks'].update(pathway='not-a-pathway'),
    lambda d: d['protocols']['generate_masks'].update(inputs=['not-an-artifact']),
    lambda d: d['protocols']['generate_masks']['steps'][0].update(module='not-a-module'),
    lambda d: d['protocols']['generate_masks']['steps'][0].update(after=['mask']),
    lambda d: d['protocols']['high_content']['steps'][0].update(after=['graph_builder']),
    lambda d: d['protocol_routes'][0].update(goal='not-a-choice'),
    lambda d: d['protocol_routes'][0].update(protocol='not-a-protocol'),
    lambda d: d['protocol_routes'][0].update(start_module='run_history'),
    lambda d: d['protocol_routes'].append(deepcopy(d['protocol_routes'][0])),
    lambda d: d['protocol_choices']['details'].update(standard=' '),
    lambda d: d['protocol_routes'].__setitem__(slice(None), [r for r in d['protocol_routes'] if r['protocol'] != 'generate_masks']),
])
def test_invalid_protocol_catalog_is_rejected_before_browser_use(mutation):
    data = deepcopy(workflows.load())
    mutation(data)
    with pytest.raises(ValueError):
        workflows._validate_protocols(data)


def test_legacy_maps_without_optional_protocol_catalog_remain_valid():
    data = workflows.load()
    for field in ('protocols', 'protocol_routes', 'protocol_choices'):
        del data[field]
    workflows._validate_protocols(data)
    data['protocols'] = {}
    with pytest.raises(ValueError):
        workflows._validate_protocols(data)
