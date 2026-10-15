from pathlib import Path
import sys,json,hashlib,subprocess
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f540-finite-threshold-20261015');scratch=Path('/mnt/wd4tb/scratch/f540-finite-threshold-20261015');sys.path.insert(0,str(root))
import numpy as np
import pandas as pd
from spacr import measure as m
from spacr.tabular import write_database,read_table
from tests.test_live_dead_viability import _signal_table
objects=_signal_table(n_live=30,n_dead=10);before=objects.copy(deep=True);cases={}
for name,threshold in [('nan',float('nan')),('positive_inf',float('inf')),('negative_inf',float('-inf')),('finite',100.),('blank',None)]:
 db=scratch/(name+'.db');write_database(objects,db,'nucleus',if_exists='replace',canonicalise=False);raw=read_table(db,table='nucleus',report=None)
 config={'channels':[0,1,2],'viability_dead_channel':1,'viability_live_channel':None,'viability_thresholds':[threshold,None],'plot':False};table,report=m._classify_viability(str(db),config,plot=False)
 pd.testing.assert_frame_equal(read_table(db,table='nucleus',report=None),raw)
 cuts=m._viability_manual([threshold,None]);qc=report['qc']
 cases[name]={'parsed_dead':str(cuts[0]),'states':table.viability_state.value_counts().to_dict(),'finite_dead_thresholds':int(np.isfinite(table.dead_threshold).sum()),'raw_nucleus_table_unchanged':True,'qc_columns':list(qc.columns)}
pd.testing.assert_frame_equal(objects,before)
assert cases['nan']['states']=={'live':40} and cases['positive_inf']['states']=={'live':40} and cases['negative_inf']['states']=={'dead':40}
assert cases['finite']['states']=={'live':30,'dead':10}
result={'source_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':hashlib.sha256((root/'spacr/measure.py').read_bytes()).hexdigest(),'cases':cases,'input_table_unchanged':True,'limits':'Private declared synthetic measured signals; no biological accuracy, actual model/training/GPU, original DB/data writes or automatic-cut tuning. Classifier writes only separate private per-case results.'};(scratch/'counterexample.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
