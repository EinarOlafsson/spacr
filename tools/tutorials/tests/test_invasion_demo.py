"""Small positive/corrupt-input tests for the tutorial-only independent reference."""
import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from invasion_demo import expected, quantile, check_preserved, verify_figure, verify_histograms
from replication_demo import digest


def rows():
    result = []
    for i, (column, value) in enumerate([('c1', 10)]*10 + [('c2', 10), ('c2', 11)], 1):
        result.append(dict(object_label=i, plateID='p', rowID='r1', columnID=column,
            fieldID='f1', prcf='p_r1_'+column+'_f1', cell_id=i,
            pathogen_area=300, pathogen_channel_0_mean_intensity=200,
            pathogen_channel_1_mean_intensity=value))
    return result


def test_interpolated_quantile_uses_both_neighbours():
    assert quantile([40, 0, 20, 10], .75) == 25
    assert quantile([7], 0) == quantile([7], 1) == 7


@pytest.mark.parametrize('values,q', [([], .5), ([1, float('nan')], .5), ([1], 1.1)])
def test_invalid_quantile_inputs(values, q):
    assert quantile([0, 10], .5) == 5
    with pytest.raises(ValueError, match='finite observations'):
        quantile(values, q)


def test_baseline_is_excluded_and_equality_is_invaded():
    result = expected(rows())
    assert result['threshold'] == 10 and result['control_objects'] == 10
    calls = result['tables']['parasite_calls.csv'][1]
    assert set(calls) == {'11', '12'}
    assert calls['11']['invasion_class'] == 'invaded'
    assert calls['12']['invasion_class'] == 'attached'
    well = result['tables']['well_invasion.csv'][1]['p_r1_c2']
    assert well['n_total'] == 2 and well['invasion_efficiency'] == .5
    assert well['invasion_efficiency_low_threshold'] == 0
    assert well['invasion_efficiency_high_threshold'] == 1
    assert well['qc_flag_threshold_inflates'] and well['qc_flag_low_total']


def test_condition_mean_does_not_pool_unequal_wells():
    valid = rows()
    second = valid[-1]
    second.update(rowID='r2', prcf='p_r2_c2_f1')
    valid.extend([dict(second, object_label=i) for i in (13, 14)])
    summary = expected(valid)['tables']['condition_summary.csv'][1]['HeLa_vehicle']
    assert summary['n_total'] == 4 and summary['n_wells'] == 2
    assert summary['invasion_efficiency'] == .5
    assert summary['invasion_efficiency_pooled'] == .25
    assert summary['invasion_efficiency_sd'] == pytest.approx(2**-.5)
    assert summary['invasion_efficiency_sem'] == pytest.approx(.5)


@pytest.mark.parametrize('field,value', [('pathogen_area', 900), ('cell_id', 0),
    ('pathogen_channel_0_mean_intensity', 99), ('pathogen_channel_1_mean_intensity', float('nan'))])
def test_changed_filter_premises(field, value):
    valid = rows()
    assert expected(valid)['threshold'] == 10
    invalid = copy.deepcopy(valid)
    invalid[-1][field] = value
    with pytest.raises(ValueError, match='premise'):
        expected(invalid)


def test_duplicate_identity_is_not_counted_twice():
    valid = rows()
    assert expected(valid)['control_objects'] == 10
    with pytest.raises(ValueError, match='unique'):
        expected(valid + [copy.deepcopy(valid[-1])])


def test_insufficient_baseline_does_not_fall_through_to_a_teaching_claim():
    valid = rows()
    assert expected(valid)['threshold'] == 10
    with pytest.raises(ValueError, match='at least ten'):
        expected(valid[1:])


def test_nonpositive_baseline_does_not_use_the_wrong_sensitivity_scale():
    valid = rows()
    assert expected(valid)['threshold'] == 10
    for row in valid[:10]:
        row['pathogen_channel_1_mean_intensity'] = 0
    with pytest.raises(ValueError, match='positive threshold'):
        expected(valid)


@pytest.mark.parametrize('which', ['source', 'database'])
def test_both_original_and_copy_are_preserved(tmp_path, which):
    manifest = {}
    for name in ('source', 'database'):
        file = tmp_path/name
        file.write_bytes(b'preserved')
        manifest.update({name: str(file), name+'_sha256': digest(file)})
    check_preserved(manifest)
    Path(manifest[which]).write_bytes(b'changed')
    with pytest.raises(ValueError, match='database changed'):
        check_preserved(manifest)


def test_figure_reads_counts_not_efficiency_labels():
    table = {'a': dict(n_attached=3, n_invaded=1, n_total=4)}
    bars = [dict(bucket='attached', rectangles=[dict(height=.75, bottom=0)]),
            dict(bucket='invaded', rectangles=[dict(height=.25, bottom=.75)])]
    assert verify_figure(['a'], bars, table) == 4
    bars[1]['rectangles'][0]['height'] = .75
    with pytest.raises(ValueError, match='height'):
        verify_figure(['a'], bars, table)


@pytest.mark.parametrize('defect', ['identity', 'range', 'threshold', 'denominator',
    'gap', 'height', 'base'])
def test_actual_histogram_geometry_and_cuts(defect):
    calls = {str(i): dict(prc='a', outside_intensity=value) for i, value in enumerate([0, 1, 2])}
    panels = [dict(well='a', denominator=3, thresholds=[[1, 1]], bars=[
        dict(left=0, width=1, height=1, bottom=0),
        dict(left=1, width=1, height=2, bottom=0)])]
    assert verify_histograms(panels, calls, 1) == dict(panels=1, bins_checked=2,
        threshold_lines_checked=1, observations=3)
    altered = copy.deepcopy(panels)
    panel = altered[0]
    if defect == 'identity':
        altered.append(copy.deepcopy(panel))
    elif defect == 'range':
        panel['bars'][0]['left'] = -.2
    elif defect == 'threshold':
        panel['thresholds'][0][0] = 1.1
    elif defect == 'denominator':
        panel['denominator'] = 4
    elif defect == 'gap':
        # Preserve both endpoint/range guards, so this isolates continuity.
        panel['bars'][0]['width'] = .5
    elif defect == 'height':
        panel['bars'][1]['height'] = 1
    else:
        panel['bars'][1]['bottom'] = 1
    guard = dict(identity='identities', range='range', threshold='threshold',
        denominator='denominator', gap='geometry', height='height', base='base')[defect]
    with pytest.raises(ValueError, match=guard):
        verify_histograms(altered, calls, 1)


def test_first_histogram_edge_allows_only_bounded_artist_roundtrip():
    calls = {str(i): dict(prc='a', outside_intensity=v) for i, v in enumerate([1., 2., 3.])}
    left = math.nextafter(1., math.inf)
    panels = [dict(well='a', denominator=3, thresholds=[[2., 2.]], bars=[
        dict(left=left, width=2-left, height=1, bottom=0),
        dict(left=2, width=1, height=2, bottom=0)])]
    assert verify_histograms(panels, calls, 2)['observations'] == 3
    panels[0]['bars'][0].update(left=1.+8*math.ulp(1.), width=1.-8*math.ulp(1.))
    with pytest.raises(ValueError, match='height'):
        verify_histograms(panels, calls, 2)
