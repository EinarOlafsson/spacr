"""Plaque figure mode: absolute and relative plaque size against reported sizes (item 911, alpha).

Built on item 900: the same condition matcher and the gold rows of
compare_reported for the log2-to-reference scale.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest

from spacr import plaque_papers as pp


def test_well_size_readouts():
    r = pp.plaque_size_readouts([10, 30, 20], (0, 0, 20, 10))
    assert r['plaque_count'] == 3 and r['plaque_mean_area_px2'] == 20
    assert r['plaque_median_area_px2'] == 20 and r['plaque_max_area_px2'] == 30
    assert r['plaque_max_over_mean'] == 1.5 and r['area_rel_well_max'] == [1 / 3, 1, 2 / 3]
    assert r['well_extent_px2'] == 200 and r['well_extent_px'] == 15
    assert r['plaque_area_rel_well_extent'] == 0.1
    empty = pp.plaque_size_readouts([], (0, 0, 0, 5))
    assert empty['plaque_mean_area_px2'] == 0 and np.isnan(empty['plaque_max_over_mean'])
    assert np.isnan(empty['plaque_area_rel_well_extent']) and empty['area_rel_well_max'] == []
    assert 'well_extent_px2' not in pp.plaque_size_readouts([5])


def _gold(scale=1000.0, sizes=(40.0, 20.0, 10.0, 5.0)):
    """Two panels of four strains: area in a lab unit and counts, one well per strain."""
    names = ['WT', 'dA', 'dB', 'dC']
    boxes = {str(i): {'strain': n, 'treatment': None} for i, n in enumerate(names)}
    area = [{'strain': n, 'treatment': None, 'value': v * scale, 'is_reference': i == 0}
            for i, (n, v) in enumerate(zip(names, sizes))]
    count = [{'strain': n, 'treatment': None, 'value': 10 + i, 'is_reference': i == 0}
             for i, n in enumerate(names)]
    return {'PMC9__f1.jpg': {'boxes': boxes, 'quantification': [
        {'panel': 'B', 'metric': 'plaque_area', 'unit': 'a.u.', 'values': area},
        {'panel': 'C', 'metric': 'plaque_number', 'unit': 'count', 'values': count},
        {'panel': 'D', 'metric': 'plaque_efficiency', 'unit': '%', 'values': count}]}}


def _segmentation(sizes=(40.0, 20.0, 10.0, 5.0)):
    """One well per box with ``10 + box`` plaques of the planted size."""
    def seg(figure, box):
        i = int(box)
        return {'areas': [sizes[i]] * (10 + i), 'bbox': [0, 0, 100, 100]}
    return seg


def test_condition_sizes_scales_and_agreement_recover_planted_sizes():
    conditions = pp.figure_condition_sizes(_gold(), _segmentation())
    assert len(conditions) == 8 and (conditions['n_wells'] == 1).all()
    area = conditions[conditions['metric'] == 'plaque_area'].set_index('strain')
    assert area.at['dA', 'spacr_mean_area_px2'] == 20 and area.at['dA', 'well_box_px2'] == 10000
    assert area.at['WT', 'is_ref'] and area.at['dC', 'spacr_n'] == 13
    scales = pp.figure_size_scales(conditions)
    assert set(scales['scale']) == {'absolute', 'rel_panel_max', 'rel_well_extent'}
    rel = scales[(scales['scale'] == 'rel_panel_max') & (scales['metric'] == 'plaque_area')]
    assert np.allclose(rel['reported_value'], rel['spacr_value'])
    absolute = scales[scales['scale'] == 'absolute']
    assert list(absolute['reported_value']) == [10, 11, 12, 13] == list(absolute['spacr_value'])
    agreement = pp.figure_size_agreement(scales, {'figure_scale_permutations': 50})
    row = agreement.set_index(['scale', 'readout']).loc[('rel_panel_max', 'all')]
    assert row['r2'] == pytest.approx(1.0) and row['lin_ccc'] == pytest.approx(1.0)
    assert row['null_r2_p95'] < 1 and row['n'] == 8 and row['figures'] == 1
    assert row['n_without_panel_max'] == 6
    again = pp.figure_size_agreement(scales, {'figure_scale_permutations': 50})
    pd.testing.assert_frame_equal(agreement, again)
    table = pp.figure_size_table(conditions).set_index('figure')
    area_rows = table.loc['PMC9 B']
    assert list(area_rows['reported_rel']) == [1.0, 0.5, 0.25, 0.125]
    assert list(area_rows['spacr_rel']) == [1.0, 0.5, 0.25, 0.125]
    assert (area_rows['largest_over_mean'] == 1).all()
    assert table.loc['PMC9 C', 'largest_over_mean'].isna().all()
    unmatched = pp.figure_condition_sizes(_gold(), lambda f, b: None)
    assert unmatched.empty and list(unmatched.columns)[-1] == 'well_box_px'
    medians = pp.figure_condition_sizes(_gold(), _segmentation(),
                                        {'figure_condition_aggregate': 'median',
                                         'figure_size_stat': 'median'})
    assert medians.loc[0, 'spacr_mean_area_px2'] == 40


def test_shuffled_spacr_values_fall_to_the_null_and_absolute_units_are_filtered():
    rng = np.random.default_rng(0)
    gold = {}
    for f in range(6):
        sizes = list(rng.uniform(5, 50, 4))
        g = _gold(sizes=sizes)
        gold[f'PMC{f}__f.jpg'] = g['PMC9__f1.jpg']
    planted = {f: _segmentation([v['value'] / 1000 for v in gold[f]['quantification'][0]['values']])
               for f in gold}
    conditions = pp.figure_condition_sizes(gold, lambda f, b: planted[f](f, b))
    scales = pp.figure_size_scales(conditions, settings={'figure_scales': 'rel_panel_max',
                                                         'figure_absolute_units': 'plaques'})
    good = pp.figure_size_agreement(scales, {'figure_scale_permutations': 100}).iloc[0]
    assert good['r2'] > good['null_r2_p95']
    shuffled = scales.copy()
    shuffled['spacr_value'] = shuffled.groupby(['pmcid', 'panel'])['spacr_value'].transform(
        lambda v: v.sample(frac=1, random_state=1).to_numpy())
    bad = pp.figure_size_agreement(shuffled, {'figure_scale_permutations': 100}).iloc[0]
    assert bad['r2'] < good['r2']
    none = pp.figure_size_scales(conditions, settings={'figure_scales': ['absolute'],
                                                      'figure_absolute_units': ['plaques']})
    assert none.empty
    small = pp.figure_size_agreement(scales.head(2), {})
    assert np.isnan(small.loc[0, 'r2'])


def test_reference_scale_reuses_the_compare_reported_rows():
    rows = pd.DataFrame({'figure': ['P1__f'] * 3, 'panel': ['A'] * 3,
                         'metric': ['plaque_area', 'plaque_area', 'plaque_number'],
                         'labels': ['gold', 'vlm', 'gold'], 'strain': ['dA', 'dA', 'dB'],
                         'treatment': [None] * 3, 'reported_ratio': [0.5, 0.5, 0.001],
                         'measured_ratio': [0.25, 1.0, 2.0]})
    scales = pp.figure_size_scales(pp.figure_condition_sizes({}, lambda f, b: None), rows,
                                   {'figure_scales': 'rel_reference'})
    assert list(scales['reported_value']) == [-1.0, np.log2(0.02)]
    assert list(scales['spacr_value']) == [-2.0, 1.0] and list(scales['readout']) == ['area', 'number']


def test_a_figure_sizes_settings_run_writes_its_tables_and_figure(tmp_path, monkeypatch):
    from tests.test_plaque_measured_vs_reported_900 import _figure_folder

    settings = _figure_folder(tmp_path)
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    from spacr.submodules import analyze_plaques

    result = analyze_plaques(dict(settings, plaque_mode='figure_sizes', compare_permutations=0,
                                  figure_scale_permutations=10))
    for name in ('conditions_measured_vs_reported.csv', 'agreement_by_scale.csv',
                 'size_scales.csv'):
        assert os.path.isfile(os.path.join(result['folder'], name))
    conditions = result['conditions'].set_index('strain')
    assert conditions.at['WT', 'spacr_mean_area_px2'] == 200
    # the stored segmentations carry no well box, so there is no well-extent scale
    assert set(result['scales']['scale']) == {'rel_panel_max', 'rel_reference'}
    figure = pp.figure_size_figure(result['scales'], result['agreement'])
    assert len(figure.axes) == 4
    empty = pp.figure_size_figure(result['scales'].iloc[0:0], result['agreement'].iloc[0:0])
    assert len(empty.axes) == 4
    no_det = pp.compare_figure_sizes(dict(settings, figure_detections='',
                                          figure_score_folder=str(tmp_path / 'nd')))
    assert 'rel_reference' not in set(no_det['scales']['scale'])


def test_size_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES, expected_types, get_analyze_plaque_settings, tooltips

    defaults = get_analyze_plaque_settings({})
    names = ('figure_condition_aggregate', 'figure_size_stat', 'figure_scales',
             'figure_absolute_units', 'figure_ratio_floor', 'figure_scale_permutations',
             'figure_scale_seed')
    assert set(names) <= set(ALPHA_FEATURES[501]['settings'])
    for key in names:
        assert key in expected_types and tooltips[key].endswith(f'Default {defaults[key]}.')
