"""Independent, bounded SQLite oracle for this real AnnData tutorial.

No spaCR exporter, joiner, feature selector or missing-value function is used.
The scope is current 2-D tables with unique full-field object keys, numeric
role-prefixed measurements and cell-only crop labels. Unsupported schemas fail.
"""
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
from capture_database import _readonly, _digest

FIELD=('plateID','rowID','columnID','fieldID')
LABELS=('annotate','al_prob_0','al_prob_1','infected',
        *(f'tutorial_demo_{i:02d}' for i in range(1,7)))


def key(row,role,label='object_label'):
    return '_'.join(str(row[c]) for c in FIELD)+f'_{role}{int(row[label])}'


def read_source(path):
    result={}
    with _readonly(path) as con:
        con.row_factory=__import__('sqlite3').Row
        for table in ('cell','cytoplasm','nucleus','pathogen','png_list'):
            rows=[dict(r) for r in con.execute('SELECT * FROM '+table)]
            columns=[(r[1],r[2]) for r in con.execute('PRAGMA table_info('+table+')')]
            features=[c for c,t in columns if t in ('INTEGER','REAL') and
                      c.startswith(table+'_') and c!=table+'_cell_id']
            result[table]=dict(rows=rows,features=features)
    for role in ('cell','cytoplasm','nucleus','pathogen'):
        rows=result[role]['rows']
        if len({key(r,role) for r in rows})!=len(rows):
            raise ValueError('Independent source has duplicate typed keys')
        if not result[role]['features'] or any(r['measurement_ndim']!=2 for r in rows):
            raise ValueError('Oracle only covers nonempty numeric 2-D role tables')
    return result


def reference(source,single='',policy='keep'):
    if single not in ('','cell','nucleus') or policy not in ('keep','mean','drop_features','drop_objects'):
        raise ValueError('Unsupported tutorial export request')
    anchor=single or 'cell';rows=source[anchor]['rows'];joined=not single
    roles=('cell','cytoplasm','nucleus','pathogen') if joined else (anchor,)
    features=[c for role in roles for c in source[role]['features']]
    tables=[role for role in roles for _ in source[role]['features']]
    by_key={role:{key(r,role):r for r in source[role]['rows']} for role in roles}
    children={role:defaultdict(list) for role in ('nucleus','pathogen')}
    for role in children:
        for r in source[role]['rows']:
            children[role][key(r,'cell','cell_id')].append(r)
    # This acquired source has one cytoplasm and at least one nucleus per
    # cell, and one matching cell crop. Do not silently generalise join loss.
    crop={key(dict(r,object_label=str(r['cell_id']).removeprefix('o')),'cell'):r
          for r in source['png_list']['rows']}
    if len(crop)!=len(source['png_list']['rows']):
        raise ValueError('Independent crop identities are not unique')
    if joined and any(key(r,'cell') not in crop or key(r,'cytoplasm') not in by_key['cytoplasm']
                      or not children['nucleus'][key(r,'cell')] for r in rows):
        raise ValueError('This tutorial oracle requires all cell join counterparts')
    matrix=[]
    for row in rows:
        values=[]
        for role in roles:
            group=(children[role][key(row,'cell')] if joined and role in children
                   else [by_key[role][key(row,role)]])
            for feature in source[role]['features']:
                finite=[float(r[feature]) for r in group if r[feature] is not None and math.isfinite(float(r[feature]))]
                values.append(math.fsum(finite)/len(finite) if finite else math.nan)
        matrix.append(values)
    raw=np.asarray(matrix,dtype=float);missing=np.isnan(raw)
    keep_rows=np.ones(len(rows),bool);keep_features=np.ones(len(features),bool)
    if policy=='drop_objects':keep_rows=~missing.any(axis=1)
    if policy=='drop_features':keep_features=~missing.any(axis=0)
    values=raw.copy()
    if policy=='mean':
        for i in range(len(features)):
            observed=[float(v) for v in raw[:,i] if math.isfinite(float(v))]
            values[missing[:,i],i]=math.fsum(observed)/len(observed) if observed else 0.
    values=values[keep_rows][:,keep_features]
    return dict(anchor=anchor,joined=joined,rows=[r for r,k in zip(rows,keep_rows) if k],
                keys=[key(r,anchor) for r,k in zip(rows,keep_rows) if k],
                features=[f for f,k in zip(features,keep_features) if k],
                tables=[t for t,k in zip(tables,keep_features) if k],
                X=values.astype('float32'),raw=raw,missing=missing,
                keep_rows=keep_rows,keep_features=keep_features,crops=crop,children=children,policy=policy)


def numeric(actual,expected,name):
    a=np.asarray(actual,dtype=float);b=np.asarray(expected,dtype=float)
    if a.shape!=b.shape or not np.allclose(a,b,rtol=2e-7,atol=1e-7,equal_nan=True):
        raise ValueError('AnnData numeric discrepancy: '+name)
    good=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[good]-b[good]),initial=0))


def verify(a,expected,*,database,settings):
    e=expected;raw=e['raw'];missing=e['missing'];policy=e['policy'];rows=e['rows']
    if list(a.obs_names)!=e['keys'] or list(a.var_names)!=e['features'] or tuple(a.shape)!=e['X'].shape:
        raise ValueError('AnnData row or feature identity differs')
    if str(a.X.dtype)!='float32':raise ValueError('AnnData dtype differs')
    error=numeric(a.X,e['X'],'every exported matrix cell')
    for c in (*FIELD,'object_label','measurement_units','object_type'):
        wanted=[e['anchor'] if c=='object_type' else r[c] for r in rows]
        if list(a.obs[c])!=wanted:raise ValueError('AnnData observation metadata differs: '+c)
    if list(a.var['source_table'])!=e['tables'] or list(a.var['is_aggregated'])!=[
            e['joined'] and t in ('nucleus','pathogen') for t in e['tables']]:
        raise ValueError('AnnData feature aggregation metadata differs')
    nan=a.uns['spacr']['nan']
    counts={'policy':policy,'n_missing':int(missing.sum()),'n_objects_counted':len(raw),
            'n_features_counted':raw.shape[1],'n_features_with_missing':int(missing.any(axis=0).sum()),
            'n_objects_with_missing':int(missing.any(axis=1).sum()),'imputed':policy=='mean' and bool(missing.any()),
            'dropped_objects':int((~e['keep_rows']).sum())}
    if any(nan[k]!=v for k,v in counts.items()):raise ValueError('AnnData missing provenance differs')
    numeric(a.var.n_missing_raw,missing.sum(axis=0)[e['keep_features']],'raw missing per feature')
    numeric(a.var.n_missing,np.isnan(e['X']).sum(axis=0),'final missing per feature')
    numeric(a.obs.n_missing_features,
            missing.sum(axis=1) if policy=='mean' else np.isnan(e['X']).sum(axis=1),'missing per object')
    # AnnData 0.13 exposes its primary X matrix as layers[None]. It is not
    # an extra exported layer; older supported AnnData versions omit it.
    named_layers=set(a.layers)-{None}
    if policy=='mean':
        if named_layers!={'missing'}:raise ValueError('AnnData imputation mask missing')
        numeric(a.layers['missing'],missing,'imputation positions')
    elif named_layers:raise ValueError('Unexpected AnnData layer')
    if len(a.obsm) or len(a.obsp):raise ValueError('Unexpected embedding or observation graph')
    p=a.uns['spacr']
    if p['anchor_object']!=e['anchor'] or p['joined']!=e['joined'] or p['source_database']!=str(database):
        raise ValueError('AnnData source provenance differs')
    if int(p['n_objects_before_filter'])!=len(raw) or int(p['n_objects'])!=len(rows) or int(p['n_features'])!=len(e['features']):
        raise ValueError('AnnData population provenance differs')
    if p['nan']['policy']!=settings['anndata_nan_policy'] or bool(a.uns['spacr_settings']['anndata_compute_umap']):
        raise ValueError('AnnData saved settings differ')
    if e['anchor']=='cell':
        for column in LABELS:
            if column not in a.obs or column in a.var_names:raise ValueError('Annotation/prediction crossed X boundary')
            for actual,row in zip(a.obs[column],rows):
                wanted=e['crops'][key(row,'cell')][column]
                if wanted is None:
                    if not __import__('pandas').isna(actual):raise ValueError('AnnData missing label changed')
                elif float(actual)!=float(wanted):raise ValueError('AnnData source label changed')
        if e['joined']:
            for role in ('nucleus','pathogen'):
                wanted=[len(e['children'][role][key(row,'cell')]) or np.nan for row in rows]
                numeric(a.obs['count_'+role],wanted,'child counts; absent remains missing')
            if list(a.obs.png_path)!=[e['crops'][key(r,'cell')]['png_path'] for r in rows]:
                raise ValueError('AnnData source crop path changed')
    else:
        numeric(a.obs.cell_id,[r['cell_id'] for r in rows],'nucleus parent links')
        if any(c in a.obs for c in LABELS):raise ValueError('Cell crop labels attached to nuclei')
        if p['relationships']['parent']['obs_column']!='cell_id':raise ValueError('Missing typed parent relationship')
    return dict(accepted=True,shape=list(a.shape),matrix_cells_checked=int(a.X.size),
                maximum_float32_discrepancy=error,missing_before=int(missing.sum()),
                missing_written=int(np.isnan(a.X).sum()),source_objects=len(raw),
                policy=policy,anchor=e['anchor'],joined=e['joined'],
                annotation_values_checked=len(rows)*len(LABELS) if e['anchor']=='cell' else 0,
                no_embedding_requested=True,image_bytes_included=False)


def verify_file(path,source,record,database):
    import anndata
    import h5py
    expected=reference(source,record['single_table'] or '',record['nan_policy'])
    result=verify(anndata.read_h5ad(path),expected,database=database,settings=record['settings'])
    with h5py.File(path,'r') as h:
        if h['X'].compression!='gzip':raise ValueError('AnnData actual X compression differs')
    result.update(path=str(path),sha256=_digest(path),bytes=Path(path).stat().st_size)
    return result
