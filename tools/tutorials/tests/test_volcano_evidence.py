"""Small synthetic guard tests; the tutorial itself uses the prior real run."""
import csv
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from volcano_evidence import read_family,check_plot,check_frame
from spacr.volcano_style import VolcanoStyle,render_volcano


@pytest.fixture
def example(tmp_path):
    probabilities=[.001,.1,.2,.4,.5,.6,.7,.8,.9,1.]
    adjusted=[.01,.5,2/3,1.,1.,1.,1.,1.,1.,1.]
    rows=[dict(outcome='y',guide=f'g{i}',wells_with_guide=i%3+2,
               standardized_marginal_effect=(i-4.5)/10,permutation_exceedances=round(p*1000)-1,
               permutations=999,permutation_p_value=p,minimum_wells_threshold=2,
               multiple_testing_method='fdr_bh',adjusted_p_value=q,significant=q<.05,alpha=.05)
          for i,(p,q) in enumerate(zip(probabilities,adjusted))]
    path=tmp_path/'family.csv'
    def save(values):
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(values)
    save(rows);records,proof=read_family(path)
    assert proof['significant_at_original_alpha']==1
    return path,save,rows,records


def explorer(rows,**options):
    frame=pd.DataFrame(rows);style=VolcanoStyle(**options)
    figure,panels=render_volcano(frame,style,figure=Figure())
    return SimpleNamespace(results=lambda:frame.copy(),style=lambda:style,
                           problems=lambda:{},_panels=panels,figure=figure)


@pytest.mark.parametrize('column',['adjusted_p_value','permutation_p_value'])
@pytest.mark.parametrize('log',[True,False])
def test_real_renderer_matches_transforms_and_keeps_source_flags(example,column,log):
    _,_,rows,records=example;e=explorer(rows,y_column=column,y_neg_log10=log)
    p=check_plot(e,records,y_column=column,neglog=log)
    assert p['significant_by_source_flags']==1 and p['plotted_points']==10


@pytest.mark.parametrize('kind',['guide','family','p','q','flag'])
def test_csv_corruption_after_positive(example,kind):
    path,save,rows,records=example;read_family(path)
    bad=[dict(r) for r in rows]
    key,value={'guide':('guide','g1'),'family':('outcome','other'),'p':('permutation_p_value',.4),
               'q':('adjusted_p_value',.9),'flag':('significant',False)}[kind]
    bad[0][key]=value;save(bad)
    with pytest.raises(ValueError):read_family(path)


@pytest.mark.parametrize('kind',['row','source','style','fallback','point','label','flags','reference'])
def test_render_corruption_after_positive(example,kind):
    _,_,rows,records=example;e=explorer(rows);check_plot(e,records)
    a=e._panels[0]
    if kind=='row':e.results=lambda:pd.DataFrame(rows[:-1])
    elif kind=='source':
        bad=[dict(r) for r in rows];bad[0]['guide']='wrong';e.results=lambda:pd.DataFrame(bad)
    elif kind=='style':e.style().alpha=.1
    elif kind=='fallback':e.problems=lambda:{'x':'bad'}
    elif kind=='point':a.collections[0].set_offsets(np.array(a.collections[0].get_offsets())+1)
    elif kind=='label':a.set_ylabel('wrong')
    elif kind=='flags':a.collections[0].set_label('all hits')
    else:a.lines[0].set_ydata([9,9])
    with pytest.raises(ValueError,match='Volcano'):check_plot(e,records)


def test_real_threshold_multiplier_and_numeric_colour(example):
    _,_,rows,records=example
    e=explorer(rows,effect_threshold=.1,threshold_multiplier=3.)
    assert sorted(check_plot(e,records,cut=.1)['effect_lines'])==pytest.approx([-.3,0,.3])
    e=explorer(rows,color_by='wells_with_guide')
    assert check_plot(e,records,colour='wells_with_guide')['plotted_points']==10
    e._panels[0].collections[0].set_array(np.zeros(10))
    with pytest.raises(ValueError,match='colour'):check_plot(e,records,colour='wells_with_guide')
