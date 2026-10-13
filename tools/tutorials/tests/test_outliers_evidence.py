"""Hand-calculated positives and intentional corruption of tutorial evidence."""
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from outliers_evidence import expected_scores, verify_scan, verify_unscored_wells


@pytest.fixture
def example():
    rows=[dict(x=x,plateID='p',rowID='r1',columnID='c1') for x in [1,2,3,4,100]]
    result=SimpleNamespace(n_rows_in=5,n_scored=5,features=('x',),method='iqr',
        transform='none',threshold=1.5,scores=[.5,0,-.5,0,48],
        flags=[False,False,False,False,True],centres={'x':3},scales={'x':2},
        fences={'x':(-1,7)},well_keys=('plateID','rowID','columnID'),n_wells_scored=0)
    wells=pd.DataFrame([dict(plateID='p',rowID='r1',columnID='c1',n_objects=5,
        n_scored_objects=5,n_flagged_objects=1,flagged_share=.2,x_median=3,
        well_scored=False,well_outlier=False,well_outlier_score=float('nan'),
        well_outlier_reason='not scored: only 1 well qualifies')])
    result.well_frame=lambda:wells.copy()
    return result,rows,wells


def test_hand_calculated_quartiles_and_mad(example):
    result,rows,_=example
    assert verify_scan(result,rows,'x',method='iqr',threshold=1.5)['max_numeric_error']==0
    assert verify_unscored_wells(result,rows,'x')[0]['median']==3
    score=expected_scores(rows,'x')
    assert score['centre']==3
    assert score['spread']==pytest.approx(1.4826022185056018)
    assert score['flags']==[False,False,False,False,True]


@pytest.mark.parametrize('field,value',[('n_rows_in',4),('n_scored',4),('features',('z',)),
    ('method','mad'),('transform','log10'),('threshold',3.5),('scores',[0,1])])
def test_scope_corruption_fails_after_positive(example,field,value):
    r,rows,_=example
    assert verify_scan(r,rows,'x',method='iqr',threshold=1.5)['rows']==5
    setattr(r,field,value)
    with pytest.raises(ValueError,match='population, feature'):
        verify_scan(r,rows,'x',method='iqr',threshold=1.5)


@pytest.mark.parametrize('field',['score','nan','centre','spread','fence','flag','missing_flag'])
def test_numeric_or_flag_corruption_fails_after_positive(example,field):
    r,rows,_=example
    assert verify_scan(r,rows,'x',method='iqr',threshold=1.5)['flagged']==1
    if field=='score':r.scores[0]=.6
    elif field=='nan':r.scores[0]=float('nan')
    elif field=='centre':r.centres['x']=4
    elif field=='spread':r.scales['x']=3
    elif field=='fence':r.fences['x']=(-2,7)
    elif field=='flag':r.flags[0]=True
    else:r.flags.pop()
    with pytest.raises(ValueError,match='independent'):
        verify_scan(r,rows,'x',method='iqr',threshold=1.5)


@pytest.mark.parametrize('field,value',[('n_objects',4),('n_scored_objects',4),
    ('n_flagged_objects',0),('flagged_share',0),('x_median',4),('well_scored',True),
    ('well_outlier',True),('well_outlier_score',0),('well_outlier_reason','clean')])
def test_not_scored_cannot_be_relabelled_clean(example,field,value):
    r,rows,wells=example
    assert verify_unscored_wells(r,rows,'x')[0]['well_scored'] is False
    wells.loc[0,field]=value
    with pytest.raises(ValueError,match='Well medians'):
        verify_unscored_wells(r,rows,'x')


def test_log_refusal_has_a_positive_real_domain_counterpart(example):
    _,rows,_=example
    assert len(expected_scores(rows,'x',transform='log10')['scores'])==5
    changed=deepcopy(rows);changed[0]['x']=0
    with pytest.raises(ValueError,match='non-positive'):
        expected_scores(changed,'x',transform='log10')
