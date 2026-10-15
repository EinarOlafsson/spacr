"""Physical stain identity is preserved before imported measurements are read."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spacr import measure
from spacr.tabular import read_table, write_database
from tests.test_cell_cycle_phase_three_ways import _nucleus_table
from tests.test_live_dead_viability import _signal_table

SOURCE = 'Measurement channels must be nonnegative integer indices.'


@pytest.fixture(autouse=True)
def english(monkeypatch):
    monkeypatch.setenv('SPACR_LANGUAGE', 'en')


@pytest.mark.parametrize('bad', [.5, True, np.bool_(False), np.array(True), np.array(.5)])
@pytest.mark.parametrize('where', ['requested', 'measured'])
def test_resolver_refuses_typed_identity_loss(bad, where):
    config, requested = ({'channels': [0, 1, 2]}, bad) if where == 'requested' else ({'channels': [bad, 1, 2]}, 1)
    before = copy.deepcopy(config)
    with pytest.raises(ValueError) as error:
        measure._measured_channel_column(config, requested)
    assert str(error.value) == SOURCE
    np.testing.assert_equal(config, before)


@pytest.mark.parametrize('value', [1, '1', np.int64(1), 1., np.array(1), np.array(1.), np.array('1')])
def test_legal_scalar_forms_preserve_physical_identity_and_measured_order(value):
    config = {'channels': [2, 0, 1], 'cell_cycle_channel': value}
    before = copy.deepcopy(config)
    assert measure._measured_channel_column(config, value) == 2
    assert measure._cell_cycle_channel(config) == 1
    np.testing.assert_equal(config, before)


@pytest.mark.parametrize('config,expected', [({}, 0), ({'channels': [2, 0, 1]}, 2),
    ({'cell_cycle_channel': None, 'nucleus_channel': '1', 'channels': [2, 0, 1]}, 1),
    ({'cell_cycle_channel': '', 'nucleus_channel': '', 'channels': [2, 0, 1]}, 2)])
def test_existing_automatic_channel_fallbacks_remain(config, expected):
    assert measure._cell_cycle_channel(config) == expected


def test_large_integer_identity_empty_and_duplicate_mapping_remain_lossless():
    large = 2**53 + 1
    assert measure._measured_channel_column({'channels': [large + 2, str(large)]}, str(large)) == 1
    assert measure._cell_cycle_channel({'cell_cycle_channel': str(large)}) == large
    assert measure._measured_channel_column({'channels': []}, '2') == 2
    assert measure._measured_channel_column({'channels': [0, 1, 1]}, 1) == 1


def write_case(tmp_path, route):
    path = tmp_path / 'measurements.db'
    if route == 'cycle':
        objects, _ = _nucleus_table(n=150)
        objects['nucleus_channel_1_mean_intensity'] = np.where(np.arange(len(objects)) % 2, 50., 900.)
        objects['nucleus_channel_1_integrated_intensity'] = objects['nucleus_channel_1_mean_intensity']
    else:
        objects = _signal_table(n_live=30, n_dead=10)
    write_database(objects, path, 'nucleus', if_exists='replace', canonicalise=False)
    target = 'cell_cycle' if route == 'cycle' else 'viability'
    write_database(pd.DataFrame({'prior_accepted': [42]}), path, target, if_exists='replace', canonicalise=False)
    return path, target


def base_config(route):
    if route == 'cycle':
        return dict(channels=[0, 1], cell_cycle_channel=0, cell_cycle_method='measurements',
                    cell_cycle_mitotic_ratio='', plot=False)
    return dict(channels=[0, 1, 2], viability_dead_channel=1, viability_live_channel=None,
                viability_thresholds=[100, None], plot=False)


def invoke(path, config, route):
    return (measure._classify_cell_cycle if route == 'cycle' else measure._classify_viability)(str(path), config, plot=False)


def assert_refusal(path, target, config, route, expected):
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    original = copy.deepcopy(config)
    with pytest.raises(ValueError) as error:
        invoke(path, config, route)
    assert str(error.value) == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    prior = read_table(path, table=target, report=None, migrate=False, read_only=True)
    assert prior.prior_accepted.tolist() == [42]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    np.testing.assert_equal(config, original)


@pytest.mark.parametrize('role', ['dead', 'live', 'morphology'])
@pytest.mark.parametrize('bad', [.5, np.array(True)])
def test_actual_viability_refuses_before_reading_or_migrating_database(tmp_path, role, bad):
    path, target = write_case(tmp_path, 'viability')
    config = base_config('viability')
    if role == 'dead':
        config['viability_dead_channel'] = bad
    elif role == 'live':
        config.update(viability_dead_channel=None, viability_live_channel=bad)
    else:
        config.update(viability_dead_channel=None, nucleus_channel=bad)
    assert_refusal(path, target, config, 'viability', SOURCE)


@pytest.mark.parametrize('case', ['dna', 'fallback', 'measured', 'fucci'])
def test_actual_cell_cycle_refuses_before_reading_or_migrating_database(tmp_path, case):
    path, target = write_case(tmp_path, 'cycle')
    config = base_config('cycle')
    if case == 'dna':
        config['cell_cycle_channel'] = .5
    elif case == 'fallback':
        config.update(cell_cycle_channel=None, nucleus_channel=np.array(True))
    elif case == 'measured':
        config['channels'] = [.5, 1]
    else:
        config['cell_cycle_fucci_channels'] = [0, 1.5]
    assert_refusal(path, target, config, 'cycle', SOURCE)


def test_valid_reordered_viability_reads_the_same_physical_stain(tmp_path):
    objects = _signal_table(n_live=30, n_dead=10)
    rename = {}
    for column in objects.columns:
        for old, new in [(0, 1), (1, 2), (2, 0)]:
            prefix = f'nucleus_channel_{old}_'
            if column.startswith(prefix):
                rename[column] = column.replace(prefix, f'nucleus_channel_{new}_', 1)
    objects = objects.rename(columns=rename)
    path = tmp_path / 'measurements.db'
    write_database(objects, path, 'nucleus', if_exists='replace', canonicalise=False)
    original = read_table(path, table='nucleus', report=None)
    config = base_config('viability')
    config.update(channels=['2', '0', '1'], viability_dead_channel=np.array('1'))
    table, _ = invoke(path, config, 'viability')
    assert table.viability_state.value_counts().to_dict() == {'live': 30, 'dead': 10}
    pd.testing.assert_frame_equal(read_table(path, table='nucleus', report=None), original)


def test_valid_cell_cycle_integral_and_fucci_values_still_execute(tmp_path):
    path, _ = write_case(tmp_path, 'cycle')
    original = read_table(path, table='nucleus', report=None)
    config = base_config('cycle')
    config.update(channels=['0', '1'], cell_cycle_channel=np.array(0.), cell_cycle_fucci_channels=[np.int64(0), '1'])
    table, report = invoke(path, config, 'cycle')
    assert len(table) == len(original) and report['methods'] == ['measurements']
    assert table.fucci_state.notna().all()
    pd.testing.assert_frame_equal(read_table(path, table='nucleus', report=None), original)


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'zh_CN', 'fr', 'pt', 'hi', 'ko', 'is'])
@pytest.mark.parametrize('route', ['cycle', 'viability'])
def test_installed_reviewed_language_at_actual_classifier_boundary(tmp_path, language, route, monkeypatch):
    review = Path(__file__).parents[1] / 'docs/i18n/reviewed/runtime' / language / '2026-10-15-measured-channel-identity.json'
    record = json.loads(review.read_text())['records'][0]
    path, target = write_case(tmp_path, route)
    config = base_config(route)
    config['cell_cycle_channel' if route == 'cycle' else 'viability_dead_channel'] = np.array(True)
    monkeypatch.setenv('SPACR_LANGUAGE', language)
    assert record['translation'] != SOURCE
    assert_refusal(path, target, config, route, record['translation'])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, np.complex64(1 + 1j)])
@pytest.mark.parametrize('where', ['requested', 'measured'])
def test_nonfinite_negative_and_complex_identity_refuses_before_mapping(bad, where):
    config, requested = ({'channels': [0, 1, 2]}, bad) if where == 'requested' else ({'channels': [bad, 1, 2]}, 1)
    before = copy.deepcopy(config)
    with pytest.raises(ValueError) as error:
        measure._measured_channel_column(config, requested)
    assert str(error.value) == SOURCE
    np.testing.assert_equal(config, before)
