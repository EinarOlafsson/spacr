from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sweep_evidence import (same_value,check_display,count_result_rows,check_result_family,
                            check_trial_set,check_fitted_support)


def fixture():
    saved=[{'trial_id':'1','alpha':'0.01','transform':'','status':'ok'},
           {'trial_id':'2','alpha':'0.1','transform':'','status':'ok'}]
    headers=['trial_id','alpha','transform','status']
    displayed=[['2','0.10','None','ok'],['1','0.01','None','ok']]
    return headers,displayed,saved


def test_sorted_display_matches_all_cells_not_row_positions():
    headers,displayed,saved=fixture()
    assert check_display(headers,displayed,saved)==dict(rows=2,columns=4,cells=8,passed=True)
    assert same_value(1,'1.0') and same_value(None,'')
    assert not same_value(float('inf'),'inf')
    assert not same_value('ok','failed')


@pytest.mark.parametrize('case',['headers','no_id','repeated_header','saved_duplicate',
    'short_row','display_duplicate','unknown','wrong_value','missing_column','missing_row'])
def test_specific_display_corruption_after_positive_counterpart(case):
    headers,displayed,saved=deepcopy(fixture())
    assert check_display(headers,displayed,saved)['passed']
    if case=='headers':headers=[]
    elif case=='no_id':headers[0]='id'
    elif case=='repeated_header':headers[1]='trial_id'
    elif case=='saved_duplicate':saved.append(saved[0].copy())
    elif case=='short_row':displayed[0].pop()
    elif case=='display_duplicate':displayed.append(displayed[0].copy())
    elif case=='unknown':displayed[0][0]='3'
    elif case=='wrong_value':displayed[0][1]='1.0'
    elif case=='missing_column':del saved[0]['alpha']
    else:displayed.pop()
    with pytest.raises(ValueError):check_display(headers,displayed,saved)


def test_significance_count_is_strict_and_missing_is_not_a_hit():
    rows=[{'q_value':x} for x in ('0','.049','.05','.1','1','',None,'nan')]
    assert count_result_rows(rows)==dict(n_results=8,n_below_alpha=2)
    assert count_result_rows(rows,.1)['n_below_alpha']==3


@pytest.mark.parametrize('value',['-0.1','1.1','inf'])
def test_bad_adjusted_probability_after_positive_counterpart(value):
    assert count_result_rows([{'q_value':'0'}])['n_below_alpha']==1
    with pytest.raises(ValueError,match='adjusted P'):count_result_rows([{'q_value':value}])


def test_bad_threshold_after_positive_counterpart():
    assert count_result_rows([])==dict(n_results=0,n_below_alpha=0)
    with pytest.raises(ValueError,match='fractional'):count_result_rows([],5)


def family_fixture():
    saved=[dict(feature='Intercept',coefficient='0',p_value='1',level='grna'),
           dict(feature='guide_A',coefficient='2',p_value='.01',level='grna'),
           dict(feature='guide_B',coefficient='3',p_value='0',level='grna'),
           dict(feature='gene_A',coefficient='4',p_value='.1',level='gene')]
    for row,value in zip(saved,('0','2','inf','1')):row['-log10(p_value)']=value
    snapshot=dict(level='grna',p_axis='raw',headers=list(saved[0]),
        rows=[list(r.values()) for r in saved[:3]],keys=['guide_A','guide_B'],
        points=[[2.,2.,0],[3.,5.,1]])
    return snapshot,saved


def test_whole_family_cells_and_points_not_just_number_of_dots():
    snapshot,saved=family_fixture()
    result=check_result_family(snapshot,saved,'grna')
    assert result['table']['cells']==15 and result['plotted_points']==2
    assert result['maximum_coordinate_error']==0
    assert not result['inference_validated']
    snapshot['points'].reverse()
    assert check_result_family(snapshot,saved,'grna')==result


@pytest.mark.parametrize('case',['level','axis','columns','cell','keys','count','duplicate_index','bad_index','x','y','log_inf'])
def test_family_corruption_fails_after_positive_counterpart(case):
    snapshot,saved=family_fixture();assert check_result_family(snapshot,saved,'grna')['passed']
    if case=='level':snapshot['level']='gene'
    elif case=='axis':snapshot['p_axis']='adjusted'
    elif case=='columns':
        snapshot['headers'].pop(2)
        for row in snapshot['rows']:row.pop(2)
    elif case=='cell':snapshot['rows'][1][1]='999'
    elif case=='keys':snapshot['keys'].reverse()
    elif case=='count':snapshot['points'].pop()
    elif case=='duplicate_index':snapshot['points'][1]=snapshot['points'][0].copy()
    elif case=='bad_index':snapshot['points'][1][2]=9
    elif case=='x':snapshot['points'][1][0]=999.
    elif case=='log_inf':snapshot['rows'][2][-1]='999'
    else:snapshot['points'][1][1]=999.
    with pytest.raises(ValueError):check_result_family(snapshot,saved,'grna')


def support_fixture():
    prepared = [dict(prc=f'p1_r{well}_c1', plateID='p1', rowID=f'r{well}',
        columnID='c1', grna=guide, gene=gene, fraction='.5', pred=str(well / 10),
        cell_count='100', screenID='screen1')
        for well in range(1, 4) for guide, gene in (('a', 'A'), ('b', 'B'))]
    settings = dict(regression_type='ridge', model_data_layout='long', level='both',
        dependent_variable='pred', inference='parametric', analysis_mode='regression',
        analysis_unit='well', agg_type='mean', intercept='fitted',
        model_plate_position=False, random_row_column_effects=False,
        batch_correction='none', transform='None', alpha=.01)
    row = dict(trial_id='1', folder='/trial1', status='ok', alpha='.01', transform='None')
    counts = dict(n_rows_fitted=6, n_wells=3, n_guides=2, n_genes=2)
    for key, value in counts.items():
        row['n_rows_prepared' if key == 'n_rows_fitted' else key + '_prepared'] = str(value)
        for suffix in ('', '_grna', '_gene'):
            row[key + suffix] = str(value)
    row.update(n_design_columns_grna='3', n_design_columns_gene='3')
    return prepared, row, settings, dict(settings)


def test_prepared_support_matches_both_actual_fit_levels_with_optional_qc():
    prepared, row, actual, requested = support_fixture()
    proof = check_fitted_support(prepared, row, actual, requested)
    assert proof['prepared']['n_rows_fitted'] == 6
    assert proof['fitted']['grna']['n_wells'] == 3
    assert proof['fitted']['gene']['n_guides'] == 2
    assert proof['qc_levels'] == [] and not proof['inference_validated']
    qc = {'grna': dict(regression_type='ridge', n_observations=6,
                      n_unique_wells=3, n_predictors=3)}
    assert check_fitted_support(prepared, row, actual, requested, qc)['qc_levels'] == ['grna']


@pytest.mark.parametrize('key', ['n_rows_prepared', 'n_wells_prepared', 'n_guides_prepared',
    'n_genes_prepared', 'n_rows_fitted', 'n_rows_fitted_grna', 'n_rows_fitted_gene',
    'n_wells_grna', 'n_wells_gene', 'n_guides_grna', 'n_guides_gene',
    'n_genes_grna', 'n_genes_gene', 'n_design_columns_grna', 'n_design_columns_gene'])
@pytest.mark.parametrize('bad', [None, '', 'nan', 'inf', '-1', '1.5'])
def test_missing_or_malformed_fit_counts_fail(key, bad):
    args = support_fixture()
    assert check_fitted_support(*args)['passed']
    if bad is None:
        del args[1][key]
    else:
        args[1][key] = bad
    with pytest.raises(ValueError):
        check_fitted_support(*args)


def test_old_coefficient_row_count_and_wrong_per_level_support_fail():
    args = support_fixture()
    args[1]['n_rows_fitted'] = '3'
    with pytest.raises(ValueError, match='Fitted support'):
        check_fitted_support(*args)
    args = support_fixture()
    args[1]['n_wells_gene'] = '2'
    with pytest.raises(ValueError, match='n_wells_gene'):
        check_fitted_support(*args)


@pytest.mark.parametrize('key,bad', [('regression_type', 'ols'),
    ('model_data_layout', 'wide'), ('level', 'gene'), ('transform', 'log'),
    ('dependent_variable', 'other'), ('inference', 'nonparametric'),
    ('analysis_mode', 'guide_permutation'), ('analysis_unit', 'cell'),
    ('agg_type', 'median'), ('intercept', 'control'),
    ('model_plate_position', True), ('random_row_column_effects', True),
    ('batch_correction', 'combat'), ('alpha', .2)])
@pytest.mark.parametrize('which', [1, 2, 3])
def test_changed_actual_requested_or_saved_model_fails(key, bad, which):
    args = support_fixture()
    args[which][key] = bad
    with pytest.raises(ValueError):
        check_fitted_support(*args)


@pytest.mark.parametrize('key', ['prc', 'grna', 'gene', 'plateID', 'rowID', 'columnID',
                               'screenID', 'fraction', 'pred', 'cell_count'])
@pytest.mark.parametrize('bad', ['', None, 'nan', 'inf'])
def test_incomplete_formula_inputs_cannot_be_assumed_fitted(key, bad):
    args = support_fixture()
    args[0][0][key] = bad
    with pytest.raises(ValueError):
        check_fitted_support(*args)


@pytest.mark.parametrize('key', ['n_observations', 'n_unique_wells', 'n_predictors'])
def test_actual_qc_must_agree_with_independent_support(key):
    args = support_fixture()
    qc = {'gene': dict(regression_type='ridge', n_observations=6,
                      n_unique_wells=3, n_predictors=3)}
    qc['gene'][key] = 1
    with pytest.raises(ValueError, match='QC disagrees'):
        check_fitted_support(*args, qc)


@pytest.mark.parametrize('case', ['missing_level', 'extra_level', 'layout',
                                 'missing_count', 'wrong_count', 'fractional_count'])
def test_saved_fit_record_must_match_independent_input_counts(case):
    args = support_fixture()
    fits = {level: dict(n_rows_fitted=6, n_wells=3, n_guides=2, n_genes=2,
                       n_design_columns=3, layout='long') for level in ('grna', 'gene')}
    assert check_fitted_support(*args, fit_designs=fits)['saved_fit_records_checked']
    if case == 'missing_level': del fits['gene']
    elif case == 'extra_level': fits['other'] = dict(fits['gene'])
    elif case == 'layout': fits['gene']['layout'] = 'wide'
    elif case == 'missing_count': del fits['gene']['n_rows_fitted']
    elif case == 'wrong_count': fits['gene']['n_rows_fitted'] = 3
    else: fits['gene']['n_rows_fitted'] = 6.5
    with pytest.raises(ValueError):
        check_fitted_support(*args, fit_designs=fits)


@pytest.mark.parametrize('case', ['missing', 'extra', 'failed', 'identity', 'folder',
                                 'penalty', 'duplicate_penalty'])
def test_exact_two_successful_distinct_penalties_required(case):
    rows = [dict(trial_id='1', folder='/one', status='ok', alpha='.01'),
            dict(trial_id='2', folder='/two', status='ok', alpha='.1')]
    assert check_trial_set(rows)['passed']
    if case == 'missing': rows.pop()
    elif case == 'extra': rows.append(dict(rows[0]))
    elif case == 'failed': rows[1]['status'] = 'failed'
    elif case == 'identity': rows[1]['trial_id'] = '1'
    elif case == 'folder': rows[1]['folder'] = '/one'
    elif case == 'penalty': rows[1]['alpha'] = 'nan'
    else: rows[1]['alpha'] = '.01'
    with pytest.raises(ValueError):
        check_trial_set(rows)
