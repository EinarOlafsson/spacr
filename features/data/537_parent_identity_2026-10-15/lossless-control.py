from pathlib import Path
import sys,json,hashlib,importlib.util
r=Path('/mnt/wd4tb/spacr-worktrees/codex-f537-parent-identity-20261015');s=Path('/mnt/wd4tb/scratch/f537-parent-identity-20261015');sys.path.insert(0,str(r))
import pandas as pd
from spacr._lineage_trees import _lineage_explicit_parents
spec=importlib.util.spec_from_file_location('spacr._lineage_proposal',s/'proposed-_lineage_trees.py');proposal=importlib.util.module_from_spec(spec);spec.loader.exec_module(proposal)
a=2**53+1;b=2**53+3;child=2**53+5
observations=[]
for dtype,values in [('object',[None,a,a]),('Int64',[None,a,a]),('object',[None,str(a),str(a)])]:
 table=pd.DataFrame({'track_id':pd.Series([child]*3,dtype='int64'),'parent_track_id':pd.Series(values,dtype=dtype)})
 spans=pd.DataFrame({'start':[0,0,3],'end':[2,2,5]},index=pd.Index([a,b,child],dtype='int64'));before=table.copy(deep=True)
 old=_lineage_explicit_parents(table,spans);new=proposal._lineage_explicit_parents(table,spans);assert old==new=={child:a};pd.testing.assert_frame_equal(table,before)
 observations.append({'dtype':dtype,'value_type':type(values[1]).__name__,'existing':old,'proposed':new})
for dtype,values in [('object',[None,a,b]),('Int64',[None,a,b]),('object',[None,str(a),str(b)])]:
 table=pd.DataFrame({'track_id':pd.Series([child]*3,dtype='int64'),'parent_track_id':pd.Series(values,dtype=dtype)})
 try:proposal._lineage_explicit_parents(table,spans)
 except ValueError as error:assert str(child) in str(error)
 else:raise AssertionError('Large known contradiction accepted')
 observations.append({'dtype':dtype,'value_type':type(values[1]).__name__,'large_contradiction_refused':True})
converted=pd.to_numeric(pd.Series([None,a,b],dtype=object),errors='coerce');assert int(converted.iloc[1])!=a
result={'large_id_controls':observations,'rejected_coercion_example':{'original':a,'lossy_float':int(converted.iloc[1])},'production_source_unchanged':hashlib.sha256((r/'spacr/_lineage_trees.py').read_bytes()).hexdigest()==json.loads((s/'proposal.json').read_text())['original_sha256'],'limits':'Actual existing and scratch-proposed helper only, declared IDs/no tracking/training/GPU.'};assert result['production_source_unchanged'];(s/'lossless-control.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
