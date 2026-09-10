"""Hand-calculated positives; every corruption test checks its positive first."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tabulate_evidence import expected_grid, verify_pivot, verify_csv


@pytest.fixture
def example():
    records = [dict(row='r2', col='c1', x=1), dict(row='r2', col='c1', x=3),
               dict(row='r2', col='c2', x=None), dict(row='r10', col='c1', x=5)]
    spec = dict(rows=('row',), cols=('col',), values=('x',),
                aggs=('n', 'mean', 'median', 'sd', 'sem', 'min', 'max', 'quantile'), quantile=.75)
    arrays = {
        'n': [[2, 0], [1, np.nan]], 'mean': [[2, np.nan], [5, np.nan]],
        'median': [[2, np.nan], [5, np.nan]], 'sd': [[2**.5, np.nan], [np.nan, np.nan]],
        'sem': [[1, np.nan], [np.nan, np.nan]], 'min': [[1, np.nan], [5, np.nan]],
        'max': [[3, np.nan], [5, np.nan]], 'quantile': [[2.5, np.nan], [5, np.nan]],
    }
    result = SimpleNamespace(spec=SimpleNamespace(**spec), n_source_rows=4, hidden_rows=0,
        row_keys=('row',), col_keys=('col',), row_levels=(('r2',), ('r10',)),
        col_levels=(('c1',), ('c2',)), shape=(2, 2), sizes=np.array([[2, 1], [1, 0]]),
        present=np.array([[True, True], [True, False]]),
        layers={('x', k): np.array(v) for k, v in arrays.items()})
    return records, spec, result


def test_hand_calculated_counts_sample_sd_and_interpolated_quantile(example):
    records, spec, result = example
    proof = verify_pivot(result, records, **spec)
    assert proof['present'] == 3
    assert proof['empty'] == [dict(row=['r10'], col=['c2'])]
    assert proof['max_numeric_error'] < 1e-12
    assert proof['csv_rows'][0][1:9] == [2, 2, 2, 2**.5, 1, 1, 3, 2.5]


@pytest.mark.parametrize('field,value', [('n_source_rows', 3), ('hidden_rows', 1), ('row_keys', ('wrong',))])
def test_population_corruption(example, field, value):
    records, spec, result = example
    assert verify_pivot(result, records, **spec)['source_rows'] == 4
    setattr(result, field, value)
    with pytest.raises(ValueError, match='population'):
        verify_pivot(result, records, **spec)


@pytest.mark.parametrize('field,value', [('rows', ('col',)), ('values', ('other',)), ('quantile', .5)])
def test_requested_specification_corruption(example, field, value):
    records, spec, result = example
    assert verify_pivot(result, records, **spec)['source_rows'] == 4
    setattr(result.spec, field, value)
    with pytest.raises(ValueError, match='population'):
        verify_pivot(result, records, **spec)


@pytest.mark.parametrize('kind', ['order', 'layer', 'shape'])
def test_structure_corruption(example, kind):
    records, spec, result = example
    assert verify_pivot(result, records, **spec)['present'] == 3
    if kind == 'order': result.row_levels = tuple(reversed(result.row_levels))
    elif kind == 'layer': result.layers.pop(('x', 'n'))
    else: result.sizes = result.sizes[:1]
    with pytest.raises(ValueError, match='axis levels'):
        verify_pivot(result, records, **spec)


@pytest.mark.parametrize('kind', ['size', 'presence'])
def test_absent_group_cannot_be_called_measured(example, kind):
    records, spec, result = example
    assert verify_pivot(result, records, **spec)['present'] == 3
    if kind == 'size': result.sizes[1, 1] = 1
    else: result.present[1, 1] = True
    with pytest.raises(ValueError, match='membership'):
        verify_pivot(result, records, **spec)


@pytest.mark.parametrize('key,at,value', [('mean', (0, 0), 3), ('n', (0, 1), 1),
    ('n', (1, 1), 0), ('sd', (1, 0), 0), ('sem', (0, 0), 2),
    ('quantile', (0, 0), 2), ('mean', (0, 0), float('nan'))])
def test_statistic_and_blank_corruption(example, key, at, value):
    records, spec, result = example
    assert verify_pivot(result, records, **spec)['present'] == 3
    result.layers[('x', key)][at] = value
    with pytest.raises(ValueError, match='statistic differs'):
        verify_pivot(result, records, **spec)


@pytest.mark.parametrize('kind', ['header', 'row', 'value', 'blank'])
def test_export_corruption_after_real_positive(example, tmp_path, kind):
    import csv
    records, spec, result = example
    proof = verify_pivot(result, records, **spec)
    path = tmp_path/'summary.csv'
    data = [proof['csv_headers'], *proof['csv_rows']]
    def save():
        with path.open('w', newline='') as stream: csv.writer(stream).writerows(data)
    save()
    assert verify_csv(path, proof)['rows'] == 2
    if kind == 'header': data[0] = ['wrong'] + data[0][1:]
    elif kind == 'row': data.pop()
    elif kind == 'value': data[1] = data[1][:1] + [999] + data[1][2:]
    else: data[2] = [0 if v is None else v for v in data[2]]
    save()
    with pytest.raises(ValueError, match='Export'):
        verify_csv(path, proof)


def test_count_only_keeps_missing_value_rows(example):
    records, _, _ = example
    wanted = expected_grid(records, rows=('row',), cols=('col',))
    assert wanted['cells'][1]['data'][('', 'n')] == 1
    assert wanted['cells'][3]['data'][('', 'n')] is None
