from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sweep_evidence import same_value,check_display,count_result_rows,check_result_family


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
