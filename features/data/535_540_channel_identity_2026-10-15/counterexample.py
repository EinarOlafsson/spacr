from pathlib import Path
import sys,json,subprocess,hashlib
root=Path('/mnt/wd4tb/spacr-worktrees/codex-measured-channel-identity-20261015');scratch=Path('/mnt/wd4tb/scratch/measured-channel-identity-20261015');sys.path.insert(0,str(root))
import numpy as np
import pandas as pd
from spacr import measure as m
from spacr.tabular import write_database,read_table
from tests.test_live_dead_viability import _signal_table
resolution={}
for name,channels,requested in [('fractional_requested',[0,1,2],.5),('fractional_measured',[.5,1.5,2.5],1),('boolean_requested',[0,1,2],True),('legal_reordered',[2,0,1],1),('legal_integer_strings',['2','0','1'],'1'),('legal_numpy_integer',[2,0,1],np.int64(1)),('legal_integral_float',[2,0,1],1.)]:
 config={'channels':channels};before=list(channels);resolution[name]={'accepted_position':m._measured_channel_column(config,requested),'requested_repr':repr(requested),'measured_repr':repr(channels)};assert config['channels']==before
objects=_signal_table(n_live=30,n_dead=10);before=objects.copy(deep=True);cases={}
for name,requested in [('correct_pi',1),('fractional_selects_dna',.5),('boolean_selects_pi',True)]:
 db=scratch/(name+'.db');write_database(objects,db,'nucleus',if_exists='replace',canonicalise=False);original=read_table(db,table='nucleus',report=None)
 config={'channels':[0,1,2],'viability_dead_channel':requested,'viability_live_channel':None,'viability_thresholds':[100,None],'plot':False};table,_=m._classify_viability(str(db),config,plot=False)
 pd.testing.assert_frame_equal(read_table(db,table='nucleus',report=None),original)
 cases[name]={'requested_repr':repr(requested),'states':table.viability_state.value_counts().to_dict(),'mean_dead_signal':float(table.dead_signal.mean()),'original_measured_nucleus_table_unchanged':True}
pd.testing.assert_frame_equal(objects,before);assert cases['correct_pi']['states']=={'live':30,'dead':10};assert cases['fractional_selects_dna']['states']=={'dead':40}
result={'source_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'measure_sha256':hashlib.sha256((root/'spacr/measure.py').read_bytes()).hexdigest(),'actual_resolver':resolution,'actual_classifier':cases,'limits':'Private declared synthetic measured signals only. This demonstrates silent channel coercion, not biological performance or validation. Original frame/raw nucleus table unchanged; result DBs private. No source edits/model/GPU.'};(scratch/'counterexample.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
