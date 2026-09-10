"""Positive numerical examples before rejecting corrupted ranking claims."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from feature_explorer_evidence import independent_scores, verify_ranking


@pytest.fixture
def example():
    rows=[{'rowID':label,'x':value,'constant':7} for label,value in
          [('r12',0),('r12',2),('r5',1),('r5',3)]]
    score=SimpleNamespace(feature='x',auc=.75,ks=.5,score=.5,
                          n_by_class={'r12':2,'r5':2},higher_in='r5')
    result=SimpleNamespace(n_rows=4,label='rowID',spec=SimpleNamespace(statistic='auc',top=20),
                           n_considered=1,scores=[score])
    return result,rows,['x','constant']


def test_pair_count_and_tied_cdfs_have_hand_calculated_positive_values(example):
    _,rows,features=example
    assert independent_scores(rows,features)==[dict(feature='x',auc=.75,ks=.5,score=.5,
                                                  n_by_class={'r12':2,'r5':2},higher_in='r5')]
    ties=[{'rowID':label,'x':value} for label,value in [('a',0),('a',1),('b',1),('b',1)]]
    score=independent_scores(ties,['x'])[0]
    assert (score['auc'],score['ks'])==(.75,.5)


@pytest.mark.parametrize('field,value',[('auc',.5),('ks',.25),('score',.75),('auc',float('nan')),
                                       ('ks',float('inf')),('score',float('nan'))])
def test_numerical_corruption_is_rejected_after_positive_proof(example,field,value):
    result,rows,features=example
    assert verify_ranking(result,rows,features)['max_numeric_error']==0
    setattr(result.scores[0],field,value)
    with pytest.raises(ValueError,match='independent pair-count'):
        verify_ranking(result,rows,features)


@pytest.mark.parametrize('field,value',[('feature','wrong'),('n_by_class',{'r12':1,'r5':3}),('higher_in','r12')])
def test_same_count_is_not_same_feature_or_class(example,field,value):
    result,rows,features=example
    assert verify_ranking(result,rows,features)['returned']==1
    setattr(result.scores[0],field,value)
    with pytest.raises(ValueError,match='feature order, class identity'):
        verify_ranking(result,rows,features)


@pytest.mark.parametrize('change',['rows','label','statistic','top','considered','missing'])
def test_population_and_top_cannot_drift(example,change):
    result,rows,features=example
    assert verify_ranking(result,rows,features)['rows']==4
    if change=='rows':result.n_rows=3
    elif change=='label':result.label='wrong'
    elif change=='statistic':result.spec.statistic='ks'
    elif change=='top':result.spec.top=4
    elif change=='considered':result.n_considered=2
    else:result.scores=[]
    with pytest.raises(ValueError,match='population, statistic, Top'):
        verify_ranking(result,rows,features)


def test_missing_values_change_per_feature_n_not_total_rows(example):
    result,rows,features=example
    assert verify_ranking(result,rows,features)['rows']==4
    extended=rows+[{'rowID':'r12','x':None,'constant':7},{'rowID':'r5','x':float('inf'),'constant':7}]
    result.n_rows=6
    assert verify_ranking(result,extended,features)['rows']==6
    assert result.scores[0].n_by_class=={'r12':2,'r5':2}


def test_one_class_is_a_refusal_after_actual_two_class_success(example):
    _,rows,features=example
    assert len(independent_scores(rows,features))==1
    with pytest.raises(ValueError,match='exactly two real classes'):
        independent_scores([r for r in rows if r['rowID']=='r12'],features)


def test_positive_scaling_preserves_ranks_without_creating_information(example):
    _,rows,_=example
    scaled=[dict(r,y=r['x']/1000) for r in rows]
    a,b=independent_scores(scaled,['x','y'])
    assert (a['auc'],a['ks'],a['score'])==(b['auc'],b['ks'],b['score'])


def test_changed_comparison_must_be_named_not_counted_as_the_old_one(example):
    result,rows,features=example
    assert verify_ranking(result,rows,features)['label']=='rowID'
    alternate=[dict(r,columnID=r['rowID']) for r in rows]
    result.label='columnID'
    with pytest.raises(ValueError,match='population, statistic, Top'):
        verify_ranking(result,alternate,features)
    assert verify_ranking(result,alternate,features,label='columnID')['label']=='columnID'
