import sys,json,hashlib,subprocess
from pathlib import Path
root=Path('/mnt/wd4tb/spacr-worktrees/codex-f537-parent-identity-20261015');sys.path.insert(0,str(root))
import pandas as pd
from spacr._lineage_trees import _lineage_explicit_parents,_lineage_segments
rows=[]
for track in (1,2):
 for frame in (0,1,2):rows.append(dict(track_id=track,frame=frame,x=10*track,y=10,parent_track_id=0,parent_track_id_source='trackastra'))
for frame,parent in [(3,1),(4,2),(5,2)]:rows.append(dict(track_id=3,frame=frame,x=15,y=10,parent_track_id=parent,parent_track_id_source='trackastra'))
original=pd.DataFrame(rows);before=original.copy(deep=True);cases={}
for name,table in [('forward',original),('reversed',original.iloc[::-1].copy())]:
 spans=table.groupby('track_id')['frame'].agg(start='min',end='max')
 links=_lineage_explicit_parents(table,spans);segments=_lineage_segments(table)
 daughter=segments[segments.track_id.eq(3)].iloc[0]
 parent=segments[segments.segment_id.eq(daughter.parent_segment_id)].iloc[0]
 cases[name]={'accepted_parent_links':links,'actual_daughter_parent_track_id':int(parent.track_id),'segments':segments[['segment_id','track_id','parent_segment_id','n_daughters']].to_dict('records')}
assert cases['forward']['actual_daughter_parent_track_id']==1
assert cases['reversed']['actual_daughter_parent_track_id']==2
pd.testing.assert_frame_equal(original,before)
controls={}
for name,values in [('repeated_identical',[1,1,1]),('all_zero',[0,0,0]),('all_missing',[None,None,None]),('missing_then_known',[None,1,1]),('unknown_absent_track',[99,99,99]),('zero_then_known',[0,1,1])]:
 table=original.copy();table.loc[table.track_id.eq(3),'parent_track_id']=values
 spans=table.groupby('track_id')['frame'].agg(start='min',end='max')
 controls[name]=_lineage_explicit_parents(table,spans)
result={'source_pin':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'source_sha256':hashlib.sha256((root/'spacr/_lineage_trees.py').read_bytes()).hexdigest(),'actual':cases,'legacy_controls':controls,'original_table_unchanged':True,'limits':'Private declared metadata only, actual reader/segment functions; no tracking, inference models, biological data, output file writes or algorithm tuning.'}
Path('/mnt/wd4tb/scratch/f537-parent-identity-20261015/counterexample.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
