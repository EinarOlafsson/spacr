"""Independent checks for the real, single-family permutation volcano lesson."""
import csv
import math
from pathlib import Path
import numpy as np


def read_family(path):
    with Path(path).open(newline='') as stream:rows=list(csv.DictReader(stream))
    if not rows or len({r['guide'] for r in rows})!=len(rows):
        raise ValueError('Volcano needs a nonempty unique guide family')
    for name in ('outcome','minimum_wells_threshold','multiple_testing_method','alpha'):
        if len({r[name] for r in rows})!=1:
            raise ValueError('Volcano must not pool correction families')
    if rows[0]['multiple_testing_method']!='fdr_bh':
        raise ValueError('Independent oracle only checks this BH-corrected family')
    p=[float(r['permutation_p_value']) for r in rows]
    order=sorted(range(len(rows)),key=lambda i:p[i]);q=[0.]*len(rows);ceiling=1.
    for rank in range(len(rows),0,-1):
        i=order[rank-1];ceiling=min(ceiling,len(rows)*p[i]/rank);q[i]=ceiling
    error=0.
    for i,r in enumerate(rows):
        wanted=(int(r['permutation_exceedances'])+1)/(int(r['permutations'])+1)
        if (not 0<p[i]<=1 or abs(p[i]-wanted)>1e-12 or
            abs(q[i]-float(r['adjusted_p_value']))>1e-12):
            raise ValueError('Permutation P or independent BH correction differs')
        expected=q[i]<float(r['alpha'])
        if r['significant'] not in ('True','False') or (r['significant']=='True')!=expected:
            raise ValueError('Stored significance does not match original family alpha')
        error=max(error,abs(q[i]-float(r['adjusted_p_value'])))
    return rows,dict(rows=len(rows),outcome=rows[0]['outcome'],support=int(rows[0]['minimum_wells_threshold']),
                     original_alpha=float(rows[0]['alpha']),minimum_raw_p=min(p),minimum_adjusted_p=min(q),
                     significant_at_original_alpha=sum(r['significant']=='True' for r in rows),
                     max_BH_error=error,method='stdlib plus-one permutation P and monotone BH adjustment')


def close(a,b,message):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.allclose(a,b,rtol=1e-10,atol=1e-10):
        raise ValueError('Volcano differs: '+message)
    return float(np.max(abs(a-b))) if a.size else 0.


def check_frame(frame,rows):
    if list(frame.columns)!=list(rows[0]) or len(frame)!=len(rows):
        raise ValueError('Volcano result population or column order changed')
    for actual,original in zip(frame.to_dict('records'),rows):
        for key,text in original.items():
            value=actual[key]
            if text in ('True','False'):
                same=isinstance(value,(bool,np.bool_)) and bool(value)==(text=='True')
            elif isinstance(value,(int,float)):
                same=math.isnan(value) if text=='' else math.isclose(float(text),value,rel_tol=1e-12,abs_tol=1e-12)
            else:same=str(value)==text
            if not same:
                raise ValueError('Volcano source cell changed: '+key)


def check_plot(explorer,rows,*,alpha=.05,y_column='adjusted_p_value',neglog=True,
               cut=None,multiplier=3.,colour=None):
    check_frame(explorer.results(),rows);s=explorer.style()
    if (s.x_column!='standardized_marginal_effect' or s.y_column!=y_column or
        s.y_neg_log10!=neglog or s.alpha!=alpha or s.threshold_method!='value' or
        s.effect_threshold!=cut or s.threshold_multiplier!=multiplier or s.color_by!=colour or
        s.split_axis or s.shape_by or s.localizations):
        raise ValueError('Volcano actual style differs from requested bounded view')
    if explorer.problems() or len(explorer._panels)!=1:
        raise ValueError('Volcano fell back or did not draw one normal panel')
    x=np.array([float(r[s.x_column]) for r in rows]);raw=np.array([float(r[y_column]) for r in rows])
    y=np.array([-math.log10(v) for v in raw]) if neglog else raw
    axis=explorer._panels[0]
    points=np.concatenate([np.asarray(c.get_offsets(),float) for c in axis.collections if len(c.get_offsets())])
    ordered=lambda a:a[np.lexsort((a[:,1],a[:,0]))]
    error=close(ordered(points),ordered(np.column_stack((x,y))),'painted point coordinates')
    expected_label=f'-log10({y_column})' if neglog else y_column
    if axis.get_ylabel()!=expected_label or axis.get_xlabel()!='Standardized marginal effect':
        raise ValueError('Volcano axis label does not name current quantity')
    significant=np.array([r['significant']=='True' for r in rows])
    resolved=None if cut is None else abs(cut)*multiplier
    if resolved is not None:significant&=abs(x)>=resolved
    if colour is None:
        groups={c.get_label():len(c.get_offsets()) for c in axis.collections}
        wanted={k:n for k,n in [('not significant',int((~significant).sum())),('significant',int(significant.sum()))] if n}
        if groups!=wanted:
            raise ValueError('Volcano significance display does not preserve source flags')
    else:
        values=np.concatenate([c.get_array() for c in axis.collections if c.get_array() is not None])
        close(values,[float(r[colour]) for r in rows],'actual colour mapping')
    horizontal=[];vertical=[]
    for line in axis.lines:
        lx,ly=np.asarray(line.get_xdata(),float),np.asarray(line.get_ydata(),float)
        if line.get_transform()==axis.get_yaxis_transform() and np.all(ly==ly[0]):horizontal.append(float(ly[0]))
        if line.get_transform()==axis.get_xaxis_transform() and np.all(lx==lx[0]):vertical.append(float(lx[0]))
    close(horizontal,[-math.log10(alpha) if neglog else alpha],'significance reference line')
    expected_vertical=sorted([0.]+([] if not resolved else [-resolved,resolved]))
    close(sorted(vertical),expected_vertical,'zero and effect reference lines')
    return dict(rows=len(rows),x_column=s.x_column,y_column=y_column,negative_log10=neglog,
                significance_reference=horizontal[0],effect_lines=vertical,significant_by_source_flags=int(significant.sum()),
                below_current_y_cut=int((raw<alpha).sum()),alpha=alpha,colour=colour,
                plotted_points=len(points),maximum_coordinate_error=error,
                figures_are_readers_not_new_statistical_tests=True)
