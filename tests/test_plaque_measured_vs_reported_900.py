"""Plaque figure mode: measured-vs-reported scoring and the well merge (item 900, alpha).

The reference is the BMC reanalysis plaque_meta code (score_lib.cond_match,
score_figure_mode.measured_by_condition and the agreement printout, and
merge_wells.py). On the reanalysis data the spaCR functions reproduce the
83-row results_measured_vs_reported tables and every merge table exactly;
these tests pin the rules on small hand-made figures.
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

from spacr import plaque_papers as pp


@pytest.mark.parametrize('gold, pred, expected', [
    (('RH', '1 nM halofuginone'), ('RH 1 nM halofuginone', '1 nM halofuginone RH'), True),
    (('RH', '1 nM halofuginone'), ('RH 2 nM halofuginone', '2 nM halofuginone RH'), False),
    (('RHΔku80', None), ('', ''), True),
    (('ΔGRA12', '+IAA'), ('dGRA12 +IAA', '+IAA dGRA12'), True),
    (('ΔGRA12', '-IAA'), ('dGRA12 untreated', 'untreated dGRA12'), True),
    (('ΔGRA12', '+IAA'), ('dGRA12 -IAA', '-IAA dGRA12'), False),
    (('TgCDPK3-AID', None), ('CDPK3 AID', ''), True),
    (('ΔMIC2', None), ('RH', ''), False),
    ((None, '0 nM'), ('anything control', 'control anything'), True),
])
def test_condition_matching_follows_the_reference_rules(gold, pred, expected):
    assert pp.condition_matches(gold[0], gold[1], pred[0], pred[1]) is expected


def test_treatment_tokens_and_wild_type_names():
    assert pp.treatment_tokens('+IFN-γ 500 um, 3 days') == {'+ifn', '500um', '3d'}
    assert pp.treatment_tokens('+IFN-γ 500 µM') == {'+ifn'}
    assert pp.treatment_tokens('ID') == {'depleted'}
    assert pp.is_wild_type('RHΔku80') and pp.is_wild_type('Parental (TIR1)')
    assert not pp.is_wild_type('ΔGRA12')
    assert pp.strain_matches('WT', 'tati line') and not pp.strain_matches('WT', 'dGRA12')
    assert pp.treatment_matches('', 'x') and pp.treatment_matches('(none)', 'x')
    assert pp.strain_matches('dX clone 2', 'dX clone2') and not pp.strain_matches('dX clone 2', 'dX')


def _regions(values):
    """Segmented regions keyed by box with mean areas ``values``."""
    return {str(i): {'seg': True, 'n': 2 * v, 'mean_area': float(v), 'frac_area': v / 100}
            for i, v in enumerate(values)}


def test_measured_by_condition_reads_the_chosen_metric_and_skips_unlabelled_wells():
    regions = _regions([10, 20, 30])
    regions['3'] = {'seg': False, 'n': 0, 'mean_area': 0.0, 'frac_area': 0}
    labeler = {'0': ('WT', ''), '1': ('dX', ''), '3': ('dX', '')}.get
    assert pp.measured_by_condition(regions, 'plaque_area', labeler) == [('WT', '', 10.0),
                                                                          ('dX', '', 20.0)]
    assert [v for *_, v in pp.measured_by_condition(regions, 'number', labeler)] == [20, 40]
    assert [v for *_, v in pp.measured_by_condition(regions, 'cleared_fraction',
                                                    labeler)] == [0.1, 0.2]
    with pytest.raises(ValueError):
        pp.measured_by_condition(regions, 'perimeter', labeler)


def _figure(values, reported, metric='plaque_area'):
    """One figure: WT plus mutants m1.. with measured ``values`` and ``reported`` values."""
    names = ['WT'] + [f'dM{i}' for i in range(1, len(values))]
    labels = {str(i): (n, '') for i, n in enumerate(names)}
    quant = [{'panel': 'A', 'metric': metric,
              'values': [{'strain': n, 'treatment': None, 'value': r, 'is_reference': i == 0}
                         for i, (n, r) in enumerate(zip(names, reported))]}]
    return {'quantification': quant, 'regions': _regions(values),
            'labelers': {'gold': labels.get}}


def test_compare_reported_gives_ratios_rho_and_a_seeded_permutation_control():
    figures = {'f1': _figure([10, 5, 20, 40, 15], [100, 40, 210, 380, 120]),
               'f2': _figure([4, 3, 9], [10, 6, 22], metric='plaque_number')}
    settings = {'compare_permutations': 200, 'compare_seed': 3}
    result = pp.compare_reported(figures, settings)
    rows = result['rows']
    assert len(rows) == 6
    first = rows.iloc[0]
    assert first['measured_ratio'] == 0.5 and first['reported_ratio'] == 0.4
    number = rows[rows['metric'] == 'plaque_number']
    assert list(number['measured_ratio']) == [0.75, 2.25]
    summary = {s['readout']: s for s in result['summary']}
    assert summary['all']['n'] == 6 and summary['all']['spearman'] == pytest.approx(1.0)
    assert summary['plaque_number']['n'] == 2 and 'spearman' not in summary['plaque_number']
    assert summary['plaque_area']['direction_agree'] == summary['plaque_area']['direction_n'] == 4
    assert summary['all']['permutation_p'] < 0.05
    again = pp.compare_reported(figures, settings)['summary']
    assert again == result['summary']
    plain = pp.compare_reported(figures, {})['summary']
    assert 'permutation_p' not in plain[0]
    fraction = pp.compare_reported(figures, {'figure_metric': 'cleared_fraction'})['rows']
    assert fraction.iloc[0]['measured_ratio'] == pytest.approx(0.5)
    assert pp.compare_reported({'f': {**figures['f1'], 'regions': {}}})['rows'].empty
    other = _figure([10, 5], [1, 2], metric='plaque_efficiency')
    assert pp.compare_reported({'f': other})['rows'].empty


def _figure_folder(tmp_path):
    """A Figure-mode run of one figure with two segmented wells, as files."""
    root = tmp_path / 'fm'
    os.makedirs(root / 'PMC1')
    with sqlite3.connect(root / 'PMC1' / 'plaque_figures.db') as con:
        pd.DataFrame([dict(x0=0, y0=0, x1=10, y1=10, condition='WT', legend_text=None,
                           label_text='WT', strength='strong', conflict=None),
                      dict(x0=20, y0=0, x1=30, y1=10, condition='dGRA1', legend_text='',
                           label_text='dGRA1', strength='weak', conflict=None),
                      dict(x0=50, y0=50, x1=60, y1=60, condition='lost', legend_text='',
                           label_text='', strength='weak', conflict=None)]).to_sql(
            'regions', con, index=False)
    seg = tmp_path / 'seg'
    os.makedirs(seg)
    for box, areas in ((0, [100, 300]), (1, [50, 50])):
        (seg / f'PMC1__f1.jpg__b{box}.json').write_text(json.dumps(
            {'areas': areas, 'cleared_frac': sum(areas) / 1000}))
    (tmp_path / 'det.csv').write_text('figure,boxes\nPMC1__f1.jpg,"[[0,0,10,10,0.9],[20,0,30,10,0.8]]"\n')
    gold = {'PMC1__f1.jpg': {'boxes': {'0': {'strain': 'WT', 'treatment': None},
                                       '1': {'strain': 'ΔGRA1', 'treatment': None}},
                             'quantification': [{'panel': 'B', 'metric': 'plaque_area', 'values': [
                                 {'strain': 'WT', 'treatment': None, 'value': 2.0, 'is_reference': True},
                                 {'strain': 'ΔGRA1', 'treatment': None, 'value': 1.0,
                                  'is_reference': False}]}]},
            'PMC2__f1.jpg': {'boxes': {}, 'quantification': []}}
    (tmp_path / 'gold.json').write_text(json.dumps(gold))
    vlm = [{'figure': 'PMC1__f1.jpg', 'boxes': [
        {'i': 0, 'is_plaque_image': True, 'strain': 'parental', 'treatment': None},
        {'i': 1, 'is_plaque_image': True, 'strain': 'dGRA1', 'treatment': ''}]}]
    (tmp_path / 'vlm.json').write_text(json.dumps(vlm))
    return {'figure_condition_labels': str(tmp_path / 'gold.json'),
            'figure_model_labels': str(tmp_path / 'vlm.json'),
            'figure_detections': str(tmp_path / 'det.csv'), 'figure_mode_folder': str(root),
            'figure_segmentation_folder': str(seg), 'src': str(tmp_path),
            'plaque_mode': 'figure_score'}


def test_a_figure_score_settings_run_writes_the_table_and_summary(tmp_path, monkeypatch):
    settings = _figure_folder(tmp_path)
    regions = pp.figure_mode_regions(
        os.path.join(settings['figure_mode_folder'], 'PMC1', 'plaque_figures.db'),
        [[0, 0, 10, 10], [20, 0, 30, 10]], lambda j: None)
    assert set(regions) == {'0', '1'} and not regions['0']['seg']
    assert pp.figure_mode_regions(
        os.path.join(settings['figure_mode_folder'], 'PMC1', 'plaque_figures.db'), [],
        lambda j: None) == {}
    monkeypatch.setattr('spacr.utils.save_settings', lambda *a, **k: None)
    from spacr.submodules import analyze_plaques

    result = analyze_plaques(dict(settings, compare_permutations=0))
    rows = pd.read_csv(os.path.join(result['folder'], 'results_measured_vs_reported.csv'))
    assert list(rows['labels']) == ['gold', 'vlm']
    assert (rows['reported_ratio'] == 0.5).all() and (rows['measured_ratio'] == 0.25).all()
    with open(os.path.join(result['folder'], 'summary.json')) as handle:
        assert json.load(handle)[0]['n'] == 1
    with pytest.raises(ValueError):
        pp.run_figure_scoring({'plaque_mode': 'plaque'})


def test_merge_figure_wells_flags_every_record_and_tabulates_effects(tmp_path):
    root = tmp_path
    os.makedirs(root / 'heldout')
    os.makedirs(root / 'vlm_index_all' / 'check_snapped')
    (root / 'heldout' / 'detections_v3.csv').write_text(
        'figure,boxes\nF1,"[[0,0,5,5,0.9],[5,0,9,5,0.8]]"\nF2,"[[1,1,4,4,0.9]]"\n')
    (root / 'heldout' / 'snapped_box_index_map.csv').write_text('figure,box_index,flags\nF1,2,soft;edge\n')
    (root / 'heldout' / 'detections_missed_snapped_kept.csv').write_text(
        'figure,new_boxes,n_existing\nF1,"[[10,10,20,20]]",2\n')
    (root / 'heldout' / 'heldout_manifest.csv').write_text(
        'figure,pmcid,detector_trained\nF1,PMC1,1\nF2,PMC2,0\n')
    box = dict(panel='A', replicate=None, notes='', is_parental_control=False, confidence=0.9)
    full = [{'figure': 'F1', 'organism': 'Toxoplasma', 'boxes': [
                {**box, 'i': 0, 'is_plaque_image': True, 'strain': 'WT', 'treatment': None,
                 'is_parental_control': True},
                {**box, 'i': 1, 'is_plaque_image': True, 'strain': 'dX', 'treatment': 'zoom inset'},
                {**box, 'i': 7, 'is_plaque_image': True, 'strain': 'dY', 'treatment': None}],
             'quantification': [{'panel': 'A', 'metric': 'plaque_area', 'unit': 'mm2',
                                 'statistic': 'mean', 'confidence': 0.5,
                                 'significance_vs_control': '*', 'values': [
                                     {'condition': 'WT', 'strain': 'WT', 'treatment': None,
                                      'mean': 4.0, 'is_parental_control': True},
                                     {'condition': 'dX', 'strain': 'dX', 'treatment': None,
                                      'mean': 0.0, 'is_parental_control': False},
                                     {'condition': 'dZ', 'strain': 'dZ', 'treatment': None,
                                      'mean': None, 'is_parental_control': False}]}]},
            {'figure': 'F2', 'organism': 'Neospora', 'boxes': [
                {**box, 'i': 0, 'is_plaque_image': False, 'strain': None, 'treatment': None,
                 'confidence': 0.5}],
             'quantification': []}]
    (root / 'vlm_index_all' / 'annotations.json').write_text(json.dumps(full))
    chk = [{'figure': 'F1', 'boxes': [{**box, 'i': 2, 'is_plaque_image': True, 'strain': 'dX',
                                       'treatment': None}]}]
    (root / 'vlm_index_all' / 'check_snapped' / 'annotations_new.json').write_text(json.dumps(chk))
    (root / 'vlm_index_all' / 'check_snapped' / 'index_new.json').write_text(
        json.dumps({'new0001': {'figure': 'F1'}}))
    result = pp.merge_figure_wells(str(root), {'figure_second_look': 'new0001'})
    wells = {(w['figure'], w['box_index']): w for w in result['wells_all']}
    assert len(wells) == 4 and ('F1', 7) not in wells
    assert wells[('F1', 1)]['flags'] == 'zoom_duplicate' and wells[('F1', 1)]['measurable'] == 0
    assert wells[('F1', 2)]['flags'] == 'snap_soft;snap_edge;reader_second_look'
    assert wells[('F1', 2)]['x0'] == 10 and wells[('F1', 2)]['task'] == 'new0001'
    assert wells[('F2', 0)]['flags'] == 'rejected_not_plaque;low_confidence;organism_Neospora'
    effects = result['reported_effects_toxo']
    assert len(effects) == 1 and effects[0]['ratio'] == 0.0
    assert effects[0]['log2_ratio'] == pytest.approx(np.log2(1e-3))
    assert effects[0]['flags'] == 'low_panel_confidence'
    counts = result['counts']
    assert counts['measurable_wells_toxo'] == 2 and counts['toxo_reference_wells'] == 1
    assert counts['needs_second_look_wells'] == 1 and counts['reported_values_all'] == 3
    assert counts['toxo_figures_without_reference_well'] == 0
    out = pp.run_figure_scoring({'plaque_mode': 'figure_merge', 'src': str(root),
                                 'figure_second_look': ''})
    assert os.path.isfile(os.path.join(out['folder'], 'merge_counts.json'))
    assert len(pd.read_csv(os.path.join(out['folder'], 'wells_toxo.csv'))) == 2


def test_segment_wells_counts_areas_and_resolves_models(tmp_path, monkeypatch):
    def fake(crop):
        """Two plaques in a 10x10 crop."""
        labels = np.zeros((10, 10), int)
        labels[:2, :2] = 1
        labels[5:, 5:] = 2
        return labels

    out = pp.segment_wells([np.zeros((10, 10))], model=fake)
    assert out[0]['n'] == 2 and out[0]['areas'] == [4, 25] and out[0]['cleared_frac'] == 0.29
    checkpoint = tmp_path / 'my_model'
    checkpoint.write_bytes(b'x')
    assert pp.resolve_plaque_model(str(checkpoint))[0] == str(checkpoint)
    seen = {}

    def segmenter(path, well_resize=None):
        """Record what segment_wells asked for."""
        seen['path'], seen['plan'] = path, well_resize
        return fake

    monkeypatch.setattr(pp, '_cellpose_segmenter', segmenter)
    monkeypatch.setattr(pp, '_model_well_resize', lambda name: {'plan': name})
    pp.segment_wells([np.zeros((4, 4))], settings={'plaque_model': str(checkpoint)})
    assert seen == {'path': str(checkpoint), 'plan': {'plan': str(checkpoint)}}
    pp.segment_wells([np.zeros((4, 4))], settings={'plaque_model': str(checkpoint),
                                                   'plaque_well_resize': 'off'})
    assert seen['plan'] is None


def test_figure_scoring_settings_are_registered_alpha():
    from spacr.settings import ALPHA_FEATURES, expected_types, get_analyze_plaque_settings, tooltips

    defaults = get_analyze_plaque_settings({})
    names = ('figure_condition_labels', 'figure_model_labels', 'figure_detections',
             'figure_mode_folder', 'figure_segmentation_folder', 'figure_score_folder',
             'figure_metric', 'compare_permutations', 'compare_seed', 'figure_low_confidence',
             'figure_second_look')
    assert set(names) <= set(ALPHA_FEATURES[900]['settings'])
    for key in names:
        assert key in expected_types and tooltips[key].endswith(f'Default {defaults[key]}.')
    assert defaults['compare_permutations'] == 1000 and defaults['figure_metric'] == 'area'
