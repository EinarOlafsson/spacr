"""Synthetic checks test the oracle only; recording uses actual downloaded cells."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pca_evidence import verify, verify_csv, reference
from spacr.qt.widgets.pca_model import pca, PCASpec


@pytest.fixture
def example():
    rng=np.random.default_rng(29)
    values=rng.normal(size=(121,3)) @ np.array([[3,.7,.1],[0,1.1,.2],[0,0,.5]])
    rows=[dict(object_label=i+1,fieldID='f1',x=a,y=b,z=c) for i,(a,b,c) in enumerate(values)]
    rows[5]['z']=None
    return rows


def actual(rows, **kwargs):
    return pca(pd.DataFrame(rows),PCASpec(features=('x','y','z'),n_components=3,**kwargs))


@pytest.mark.parametrize('policy',['auto','complete','drop_features','mean'])
@pytest.mark.parametrize('scaling',['zscore','none'])
def test_actual_svd_matches_independent_covariance_eigenvectors(example,policy,scaling):
    r=actual(example,nan_policy=policy,scaling=scaling)
    check=verify(r,example,('x','y','z'),policy=policy,scaling=scaling)
    assert check['analysed_rows']==(120 if policy in ('auto','complete') else 121)
    assert check['features']==(['x','y'] if policy=='drop_features' else ['x','y','z'])


@pytest.mark.parametrize('field',['centre','scale','scores','loadings','correlations','explained_variance','explained_variance_ratio','total_variance'])
def test_numeric_corruption_fails_after_positive(example,field):
    r=actual(example);verify(r,example,('x','y','z'))
    bad=np.array(getattr(r,field),copy=True)+.2
    with pytest.raises(ValueError,match='numeric discrepancy'):
        verify(replace(r,**{field:bad}),example,('x','y','z'))


@pytest.mark.parametrize('kind',['identity','policy'])
def test_identity_and_policy_corruption_fail_after_positive(example,kind):
    r=actual(example);verify(r,example,('x','y','z'))
    bad=replace(r,rows=r.rows[::-1]) if kind=='identity' else replace(r,scaling='none')
    with pytest.raises(ValueError,match='identities' if kind=='identity' else 'population'):
        verify(bad,example,('x','y','z'))


def export(stem,rows,r):
    r.scores_frame(pd.DataFrame(rows)).to_csv(str(stem)+'_scores.csv',index=False)
    r.loadings_frame().to_csv(str(stem)+'_loadings.csv',index=False)
    r.variance_frame().to_csv(str(stem)+'_variance.csv',index=False)


@pytest.mark.parametrize('kind',['row','source','loading','component','number'])
def test_export_corruption_fails_after_positive(example,tmp_path,kind):
    r=actual(example);stem=tmp_path/'test';export(stem,example,r)
    assert verify_csv(stem,example,r)['scores']['rows']==120
    suffix={'row':'scores','source':'scores','loading':'loadings','component':'variance','number':'variance'}[kind]
    path=Path(str(stem)+'_'+suffix+'.csv');f=pd.read_csv(path)
    if kind=='row':f=f.iloc[:-1]
    elif kind=='source':f.loc[0,'object_label']=999
    elif kind=='loading':f.loc[0,'feature']='wrong'
    elif kind=='component':f.loc[0,'component']='PC99'
    else:f.loc[0,'cumulative_ratio']=.99
    f.to_csv(path,index=False)
    with pytest.raises(ValueError,match='PCA'):
        verify_csv(stem,example,r)


def test_constant_and_degenerate_inputs_are_not_claimed_valid(example):
    reference(example,('x','y','z'))
    with pytest.raises(ValueError,match='constant'):
        reference([dict(r,z=1) for r in example],('x','y','z'))
    with pytest.raises(ValueError,match='full-rank'):
        reference([dict(r,z=r['x']) for r in example],('x','y','z'))
