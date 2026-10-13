import json,sys,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
root=Path(__file__).parent
repo=Path('/tmp/spacr-implementation-20261001/suggest-capture');sys.path.insert(0,str(repo))
from spacr.sp_stats import screen_wells,score_screen
screen=pd.read_csv(root/'screen.csv');reference=pd.read_csv(root/'reference.csv')
reports={}
for feature in ['cell_count_proxy','nuclear_area']:
    wells=screen_wells(screen,feature,plate_column='plateID',control_column='well_type',negative_levels=['negcon'])
    for scope in ['plate','pooled']:
        result=score_screen(wells,scope=scope,rank_by='b_score')
        actual=result.wells.sort_values(['plateID','well']).reset_index(drop=True)
        expected=reference.sort_values(['plateID','well']).reset_index(drop=True)
        assert actual['plateID'].equals(expected['plateID']) and actual['well'].equals(expected['well'])
        a=actual['b_score'].to_numpy();b=expected[f'{feature}_{scope}'].to_numpy();sample=actual['role'].eq('sample').to_numpy()
        finite=np.isfinite(a)&np.isfinite(b)
        calls=sample&(np.abs(b)>=3)
        report={'n':len(a),'sample_wells':int(sample.sum()),'finite_mask_equal':bool(np.array_equal(np.isfinite(a),np.isfinite(b))),'max_absolute_difference':float(np.max(np.abs(a[finite]-b[finite]))),'spearman_all':float(spearmanr(a[finite],b[finite]).statistic),'sample_spearman_abs':float(spearmanr(np.abs(a[sample]),np.abs(b[sample])).statistic),'calls_actual':int(actual['hit_b_score'].sum()),'calls_reference':int(calls.sum()),'call_disagreements':int(np.count_nonzero(actual['hit_b_score'].to_numpy()!=calls)),'all_scores_equal_1e_10':bool(np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True))}
        report['sample_spearman_abs_numeric_ties_1e_10']=float(spearmanr(np.round(np.abs(a[sample]),10),np.round(np.abs(b[sample]),10)).statistic)
        ranked=np.flatnonzero(sample)[np.argsort(-np.abs(a[sample]),kind='stable')]
        report['distinct_reference_rank_inversions_over_1e_10']=int(np.count_nonzero(np.diff(np.abs(b[ranked]))>1e-10))
        if sys.argv[1]=='final':
            assert report['all_scores_equal_1e_10'] and report['finite_mask_equal'],report
            assert report['call_disagreements']==0 and report['distinct_reference_rank_inversions_over_1e_10']==0,report
        reports[f'{feature}_{scope}']=report
        actual['reference_b_score']=b;actual.to_csv(root/f'{sys.argv[1]}-{feature}-{scope}.csv',index=False)
print(json.dumps(reports,indent=2));(root/f'{sys.argv[1]}.json').write_text(json.dumps(reports,indent=2)+'\n')
