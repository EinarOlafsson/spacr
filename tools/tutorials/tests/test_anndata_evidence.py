"""Tiny synthetic records exercise the independent verifier, not tutorial data."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import anndata
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from anndata_evidence import FIELD, LABELS, reference, verify, verify_file


@pytest.fixture
def source():
    data={}
    for role in ('cell','cytoplasm','nucleus','pathogen'):
        rows=[]
        for i in range(1,5):
            if role=='pathogen' and i==4:continue
            r=dict(zip(FIELD,('plate1','r1','c1','f1')))
            r.update(object_label=i,measurement_units='px',object_type=role,
                     cell_id=i,**{role+'_area':None if i==3 else float(i*2)})
            rows.append(r)
        data[role]=dict(rows=rows,features=[role+'_area'])
    data['png_list']=dict(rows=[dict(r,cell_id='o'+str(r['object_label']),png_path=f'/example/{i}.png',
        **{c:(1 if c=='infected' else None) for c in LABELS}) for i,r in enumerate(data['cell']['rows'])])
    return data


def artifact(source,policy='keep',single='cell'):
    e=reference(source,single,policy);rows=e['rows'];m=e['missing']
    obs=pd.DataFrame([{c:r[c] for c in (*FIELD,'object_label','measurement_units','object_type')} for r in rows],index=e['keys'])
    if e['anchor']=='cell':
        for c in LABELS:obs[c]=[1 if c=='infected' else np.nan]*len(obs)
        if e['joined']:
            for role in ('nucleus','pathogen'):
                obs['count_'+role]=[len(e['children'][role][k]) or np.nan for k in e['keys']]
            obs['png_path']=[e['crops'][k]['png_path'] for k in e['keys']]
    else:obs['cell_id']=[r['cell_id'] for r in rows]
    obs['n_missing_features']=m.sum(axis=1) if policy=='mean' else np.isnan(e['X']).sum(axis=1)
    var=pd.DataFrame(dict(source_table=e['tables'],is_aggregated=[e['joined'] and t in ('nucleus','pathogen') for t in e['tables']],
        n_missing_raw=m.sum(axis=0)[e['keep_features']],n_missing=np.isnan(e['X']).sum(axis=0)),index=e['features'])
    a=anndata.AnnData(e['X'].copy(),obs=obs,var=var)
    if policy=='mean':a.layers['missing']=m.copy()
    a.uns['spacr']=dict(anchor_object=e['anchor'],joined=e['joined'],source_database='/example.db',
        n_objects_before_filter=len(e['raw']),n_objects=len(rows),n_features=len(e['features']),
        relationships={'parent':{'obs_column':'cell_id'}},nan=dict(policy=policy,n_missing=int(m.sum()),
        n_objects_counted=len(e['raw']),n_features_counted=e['raw'].shape[1],n_features_with_missing=int(m.any(axis=0).sum()),
        n_objects_with_missing=int(m.any(axis=1).sum()),imputed=policy=='mean',dropped_objects=int((~e['keep_rows']).sum())))
    a.uns['spacr_settings']={'anndata_compute_umap':False}
    return a,e,dict(database='/example.db',settings={'anndata_nan_policy':policy})


@pytest.mark.parametrize('single',['','cell','nucleus'])
@pytest.mark.parametrize('policy',['keep','mean','drop_features','drop_objects'])
def test_positive_small_matrix_policies_and_relationships(source,single,policy):
    a,e,kw=artifact(source,policy,single)
    assert verify(a,e,**kw)['accepted']


@pytest.mark.parametrize('kind',['row','feature','value','dtype','metadata','aggregation','missing_report',
    'raw_missing','final_missing','object_missing','mask','embedding','source','population','settings','label','label_missing','parent'])
def test_corruption_is_rejected_after_positive(source,kind):
    a,e,kw=artifact(source,'mean' if kind=='mask' else 'keep','nucleus' if kind=='parent' else 'cell')
    verify(a,e,**kw)
    if kind=='row':a.obs_names=['wrong',*list(a.obs_names)[1:]]
    elif kind=='feature':a.var_names=['wrong']
    elif kind=='value':a.X[0,0]+=5
    elif kind=='dtype':a.X=a.X.astype('float64')
    elif kind=='metadata':a.obs.loc[a.obs_names[0],'object_label']=99
    elif kind=='aggregation':a.var.loc[a.var_names[0],'is_aggregated']=True
    elif kind=='missing_report':a.uns['spacr']['nan']['n_missing']=99
    elif kind=='raw_missing':a.var['n_missing_raw']=99
    elif kind=='final_missing':a.var['n_missing']=99
    elif kind=='object_missing':a.obs['n_missing_features']=99
    elif kind=='mask':del a.layers['missing']
    elif kind=='embedding':a.obsm['X_umap']=np.zeros((len(a),2))
    elif kind=='source':a.uns['spacr']['source_database']='/wrong.db'
    elif kind=='population':a.uns['spacr']['n_objects']=99
    elif kind=='settings':a.uns['spacr_settings']['anndata_compute_umap']=True
    elif kind=='label':a.obs['infected']=0
    elif kind=='label_missing':a.obs['annotate']=0
    elif kind=='parent':a.obs['cell_id']=99
    with pytest.raises(ValueError):verify(a,e,**kw)


def test_real_hdf5_roundtrip_and_compression_guard(source,tmp_path):
    a,e,kw=artifact(source);path=tmp_path/'small.h5ad'
    record={'single_table':'cell','nan_policy':'keep','settings':kw['settings']}
    a.write_h5ad(path,compression='gzip')
    assert verify_file(path,source,record,'/example.db')['accepted']
    a.write_h5ad(path,compression=None)
    with pytest.raises(ValueError,match='compression'):verify_file(path,source,record,'/example.db')


def test_crop_loss_is_not_silently_modelled_as_success(source):
    assert reference(source)['X'].shape==(4,4)
    source['png_list']['rows'].pop()
    with pytest.raises(ValueError,match='counterparts'):reference(source)
