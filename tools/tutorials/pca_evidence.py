"""Independent covariance-eigenvalue oracle for the small, full-rank PCA lesson.

The live application uses SVD on these 2,341 rows. This checks a different
factorisation, identities, finite-value policy and exported values; it is not
a validation of arbitrary rank-deficient or nearly degenerate inputs.
"""
import csv
import math
import numpy as np


def reference(rows, features, *, scaling='zscore', policy='auto', components=3):
    names = tuple(sorted(features))
    if scaling not in ('zscore', 'none') or policy not in ('auto','complete','drop_features','mean'):
        raise ValueError('Unsupported bounded PCA policy')
    matrix = np.array([[r[n] if r[n] is not None else np.nan for n in names] for r in rows], float)
    matrix[~np.isfinite(matrix)] = np.nan
    fractions = np.isnan(matrix).mean(axis=0)
    keep = fractions <= (.02 if policy=='auto' else 0) if policy in ('auto','drop_features') else fractions < 1
    if policy=='complete': keep = np.ones(len(names), bool)
    kept = tuple(n for n,k in zip(names,keep) if k)
    matrix = matrix[:,keep]
    if policy=='mean':
        for j in range(matrix.shape[1]):
            finite = matrix[:,j][np.isfinite(matrix[:,j])]
            if len(finite): matrix[np.isnan(matrix[:,j]),j] = math.fsum(finite)/len(finite)
    positions = np.flatnonzero(np.isfinite(matrix).all(axis=1))
    matrix = matrix[positions]
    if len(kept)<2 or len(positions)<3:
        raise ValueError('Oracle requires at least two features and three finite objects')
    means = np.array([math.fsum(col)/len(col) for col in matrix.T])
    sd = np.array([math.sqrt(math.fsum((v-m)**2 for v in col)/(len(col)-1)) for col,m in zip(matrix.T,means)])
    if np.any(sd<=np.maximum(abs(means),1)*1e-12):
        raise ValueError('Oracle does not silently accept constant columns')
    scales = sd if scaling=='zscore' else np.ones(len(kept))
    x = (matrix-means)/scales
    covariance = x.T@x/(len(x)-1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]; values=values[order]; vectors=vectors[:,order]
    if values[-1]<=values[0]*1e-10 or np.min(-np.diff(values))<=values[0]*1e-10:
        raise ValueError('Oracle requires distinct, well-resolved full-rank eigenvalues')
    k=min(components,len(kept)); vectors=vectors[:,:k]
    for j in range(k):
        if vectors[np.argmax(abs(vectors[:,j])),j]<0: vectors[:,j]*=-1
    scores=x@vectors
    correlations=np.array([[np.corrcoef(matrix[:,i],scores[:,j])[0,1] for j in range(k)] for i in range(len(kept))])
    return dict(features=kept,rows=positions,centre=means,scale=scales,scores=scores,
                loadings=vectors,correlations=correlations,explained_variance=values[:k],
                explained_variance_ratio=values[:k]/sum(values),total_variance=float(sum(values)),
                rank=len(kept),dropped_features=tuple(n for n,k in zip(names,keep) if not k))


def close(actual, expected, label):
    a,b=np.asarray(actual,float),np.asarray(expected,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,rtol=2e-8,atol=2e-8):
        raise ValueError('PCA numeric discrepancy: '+label)
    return float(np.max(abs(a-b))) if a.size else 0.


def verify(result, rows, features, *, scaling='zscore', policy='auto', components=3):
    expected=reference(rows,features,scaling=scaling,policy=policy,components=components)
    if (result.features!=expected['features'] or result.n_rows_in!=len(rows) or
        result.n_features_in!=len(features) or result.rank!=expected['rank'] or
        result.scaling!=scaling or result.nan_policy!=policy or
        result.dropped_rows!=len(rows)-len(expected['rows']) or
        set(result.dropped_features)!=set(expected['dropped_features'])):
        raise ValueError('PCA population, feature or policy mismatch')
    if not np.array_equal(result.rows,expected['rows']):
        raise ValueError('PCA retained object identities differ')
    errors={n:close(getattr(result,n),expected[n],n) for n in
            ('centre','scale','scores','loadings','correlations','explained_variance','explained_variance_ratio','total_variance')}
    return dict(input_rows=len(rows),analysed_rows=len(result),features=list(result.features),
                dropped_rows=result.dropped_rows,dropped_features=dict(result.dropped_features),
                components=result.n_components,scaling=scaling,policy=policy,
                explained_variance_ratio=result.explained_variance_ratio.tolist(),
                centre=result.centre.tolist(),scale=result.scale.tolist(),numeric_errors=errors,
                independent_method='covariance eigendecomposition versus application SVD')


def verify_csv(stem, rows, result):
    """Check every original score-row cell and every numerical PCA export."""
    from pathlib import Path
    output={}
    for suffix in ('scores','loadings','variance'):
        path=Path(str(stem)+'_'+suffix+'.csv')
        with path.open(newline='') as f:
            reader=csv.DictReader(f); records=list(reader); columns=reader.fieldnames
        output[suffix]=dict(path=str(path),rows=len(records),columns=len(columns))
        if suffix=='scores':
            pcs=['PC'+str(i+1) for i in range(result.n_components)]
            if columns!=list(rows[0])+pcs or len(records)!=len(result.rows):
                raise ValueError('PCA scores export population or columns differ')
            for record,position,scores in zip(records,result.rows,result.scores):
                source=rows[position]
                for name,value in source.items():
                    if value is None or isinstance(value,float) and math.isnan(value): good=record[name]==''
                    elif isinstance(value,(int,float)):
                        good=record[name]!='' and math.isclose(float(record[name]),value,rel_tol=1e-12,abs_tol=1e-12)
                    else: good=record[name]==str(value)
                    if not good:
                        raise ValueError('PCA score export changed source object cell: '+name)
                close([record[p] for p in pcs],scores,'exported scores')
        elif suffix=='loadings':
            if [r['feature'] for r in records]!=list(result.features):
                raise ValueError('PCA loading export feature identities differ')
            for j in range(result.n_components):
                close([r[f'PC{j+1}_loading'] for r in records],result.loadings[:,j],'exported loadings')
                close([r[f'PC{j+1}_r'] for r in records],result.correlations[:,j],'exported correlations')
        else:
            if [r['component'] for r in records]!=[f'PC{j+1}' for j in range(result.n_components)]:
                raise ValueError('PCA variance export component identities differ')
            for name,values in [('explained_variance',result.explained_variance),('explained_variance_ratio',result.explained_variance_ratio),('cumulative_ratio',np.cumsum(result.explained_variance_ratio))]:
                close([r[name] for r in records],values,'exported '+name)
    return output
