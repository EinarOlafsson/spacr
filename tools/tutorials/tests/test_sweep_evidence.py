from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sweep_evidence import same_value,check_display,count_result_rows


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
