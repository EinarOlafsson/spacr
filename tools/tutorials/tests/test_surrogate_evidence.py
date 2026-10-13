from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from surrogate_evidence import binary_counts, check_metric_values, check_formatted_table, check_well_partition


def test_binary_arithmetic_is_not_the_backend_scorer():
    rows=[dict(cv_prediction=a,surrogate_prediction=b) for a,b in [(0,0),(0,1),(1,1),(1,1)]]
    value=binary_counts(rows)
    assert value['confusion']==[[1,1],[0,2]]
    assert value['fidelity']==.75 and value['balanced_accuracy']==.75
    assert value['f1_macro']==pytest.approx((2/3+.8)/2)
    assert value['class_metrics'][0]['support']==2
    single=binary_counts([dict(cv_prediction=1,surrogate_prediction=1)])
    assert single['balanced_accuracy']==1 and single['f1_macro']==1


def test_every_four_object_binary_call_pattern_matches_manual_counts():
    from itertools import product
    for calls in product((0,1),repeat=8):
        actual,predicted=calls[:4],calls[4:]
        rows=[dict(cv_prediction=a,surrogate_prediction=p) for a,p in zip(actual,predicted)]
        found=binary_counts(rows)
        assert found['fidelity']==sum(a==p for a,p in zip(actual,predicted))/4
        assert sum(map(sum,found['confusion']))==4
        assert 0<=found['balanced_accuracy']<=1 and 0<=found['f1_macro']<=1
        assert binary_counts(rows[::-1])==found


@pytest.mark.parametrize('rows',[[],[dict(cv_prediction=2,surrogate_prediction=0)],
    [dict(cv_prediction=0,surrogate_prediction=float('nan'))]])
def test_invalid_calls_fail_after_a_positive_case(rows):
    assert binary_counts([dict(cv_prediction=0,surrogate_prediction=0)])['fidelity']==1
    with pytest.raises(ValueError):binary_counts(rows)


@pytest.mark.parametrize('value',[.3,float('nan'),float('inf')])
def test_wrong_metric_after_correct_value(value):
    assert check_metric_values({'fidelity':.5},{'fidelity':.5})['passed']
    with pytest.raises(ValueError):check_metric_values({'fidelity':value},{'fidelity':.5})


def table_fixture():
    columns=['identity','value','optional']
    rows=[['a',.1234567,None],['b',2,float('nan')]]
    actual=dict(columns=columns[:],rows=[['b','2',''],['a','0.12346','']])
    return actual,columns,rows


def test_all_formatted_cells_and_exact_cap_are_checked():
    assert check_formatted_table(*table_fixture())['cells']==6
    rows=[[i] for i in range(1001)]
    actual=dict(columns=['row'],rows=[[str(i)] for i in range(1000)])
    result=check_formatted_table(actual,['row'],rows)
    assert result['truncated'] and result['source_rows']==1001


@pytest.mark.parametrize('change',['header','duplicate_header','short_row','missing','duplicate','value','nonfinite'])
def test_bad_table_after_a_positive_counterpart(change):
    actual,columns,rows=table_fixture()
    assert check_formatted_table(actual,columns,rows)['passed']
    if change=='header':actual['columns'][0]='wrong'
    elif change=='duplicate_header':
        actual['columns'][1]=actual['columns'][0];columns[1]=columns[0]
    elif change=='short_row':actual['rows'][0].pop();rows[1].pop()
    elif change=='missing':actual['rows'].pop()
    elif change=='duplicate':actual['rows'][0]=actual['rows'][1][:]
    elif change=='value':actual['rows'][0][1]='3'
    else:
        rows[0][1]=float('inf');actual['rows'][1][1]='inf'
    with pytest.raises(ValueError):check_formatted_table(actual,columns,rows)


def partition_fixture():
    population=[dict(prcfo=str(i),plateID='p',rowID='r',columnID='1' if i<2 else '2') for i in range(5)]
    held=deepcopy(population[:2])
    report=dict(group_by='well',train_cells=3,test_cells=2,train_groups=1,test_groups=1,
        total_groups=2,cell_fraction=.4,group_fraction=.5)
    return population,held,report


def test_whole_well_partition_uses_database_identities():
    result=check_well_partition(*partition_fixture())
    assert result['shared_wells']==0 and result['held_out_objects']==2
    assert result['held_out_wells']==[['p','r','1']]


@pytest.mark.parametrize('change',['population_duplicate','test_duplicate','unknown','empty','all',
    'partial_well','group','cells','groups','fraction'])
def test_bad_split_after_positive_counterpart(change):
    population,held,report=partition_fixture();assert check_well_partition(population,held,report)['passed']
    if change=='population_duplicate':population.append(population[0].copy())
    elif change=='test_duplicate':
        held.append(held[0].copy());report.update(train_cells=2,test_cells=3,cell_fraction=.6)
    elif change=='unknown':held[0]['prcfo']='unknown'
    elif change=='empty':held=[]
    elif change=='all':held=deepcopy(population)
    elif change=='partial_well':
        held.pop();report.update(train_cells=4,test_cells=1,cell_fraction=.2)
    elif change=='group':report['group_by']='plate'
    elif change=='cells':report['test_cells']=3
    elif change=='groups':report['test_groups']=2
    else:report['cell_fraction']=.5
    with pytest.raises(ValueError):check_well_partition(population,held,report)
