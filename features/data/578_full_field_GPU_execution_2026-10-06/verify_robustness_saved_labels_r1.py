from pathlib import Path
import gzip
import hashlib
import json
import os
import re
import shutil
import sys

import numpy as np
import pandas as pd

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0,str(checkout))
import spacr
from spacr import object as objects
from spacr.schema import canonicalise_frame
from spacr.seg_qc import _score_robustness, _robustness_grid
assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'578-robustness-plate-r1'
target=root/'cuda-r2'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((root/'plan.json').read_text())
recipe=json.loads((root/'recipe.json').read_text())
freeze=json.loads((root/'GPU-freeze.json').read_text())
assert digest(root/'plan.json') == freeze['plan_sha256']
assert digest(scratch/'benchmark_robustness_plate_gpu_r2.py') == freeze['benchmark_sha256']
assert digest(freeze['model_path']) == freeze['model_sha256']
for path,sha in plan['source_sha256'].items(): assert digest(path) == sha
for row in plan['inputs']:
    assert digest(row['original_path']) == row['original_sha256']
    assert digest(row['export']) == row['export_sha256']
fields=objects._robustness_sample(str(root/'plate/masks'),recipe,'nucleus')
grid=_robustness_grid(recipe,'nucleus')
assert len(fields)==16 and grid==plan['grid']
for name,image in fields: np.testing.assert_array_equal(image,np.load(root/(name+'-normalized-intensity.npy'),allow_pickle=False))
raw=scratch/'robustness-plate-gpu-r2.log'
log=raw.read_text()
assert 'FINISH rc=1' in log and 'DataFrame.columns are different' in log
assert "[left]:  Index(['fieldID'" in log and "[right]: Index(['field'" in log
lines=[s for s in log.splitlines() if s.startswith('actual normal GPU segmentation ')]
assert len(lines)==128
assert len(list(target.glob('call-*-labels.npz')))==128
profiles={}
for index in (0,127):
    path=target/f'call-{index:03d}-profile.json'
    trace=json.loads(path.read_text())
    events=[e for e in trace['traceEvents'] if str(e.get('cat','')).lower()=='kernel']
    assert events and all(e.get('dur',0)>=0 for e in events)
    profiles[str(index)]={'positive_actual_CUDA_kernel_events':len(events),'sha256':digest(path),'bytes':path.stat().st_size}
calls=[]
for i,line in enumerate(lines):
    field_index,grid_index=divmod(i,len(grid))
    name,image=fields[field_index]
    point=grid[grid_index]
    assert line.startswith(f'actual normal GPU segmentation {i+1} /128 {name} {point["parameter"]} {point["value"]} objects ')
    label_path=target/f'call-{i:03d}-labels.npz'
    with np.load(label_path,allow_pickle=False) as saved: labels=saved['labels']
    assert labels.shape==(1994,1994) and labels.dtype.kind in 'iu'
    objects_count=len(np.unique(labels[labels>0]))
    assert objects_count==int(line.rsplit(' ',1)[1])
    calls.append({'index':i,'field':name,'grid_index':grid_index,'point':point,'image_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'labels_path':str(label_path),'labels_sha256':digest(label_path),'objects':objects_count,'log_line':line})
replay_index=0
def replay(image,point):
    global replay_index
    call=calls[replay_index]
    assert point==call['point'] and hashlib.sha256(image.tobytes()).hexdigest()==call['image_sha256']
    assert digest(call['labels_path'])==call['labels_sha256']
    replay_index+=1
    with np.load(call['labels_path'],allow_pickle=False) as saved: return saved['labels']
per_field,summary=_score_robustness(fields,replay,grid,recipe['robustness_tolerance'])
assert replay_index==128
qc=root/'plate/qc'
summary_path=qc/'segmentation_robustness_nucleus.csv'
field_path=qc/'segmentation_robustness_nucleus_fields.csv'
pdf_path=qc/'segmentation_robustness_nucleus.pdf'
normal_canonical=canonicalise_frame(per_field)
assert normal_canonical.columns[0]=='fieldID' and per_field.columns[0]=='field'
pd.testing.assert_frame_equal(pd.read_csv(summary_path).fillna({'reason':''}),summary,check_dtype=False)
pd.testing.assert_frame_equal(pd.read_csv(field_path),normal_canonical,check_dtype=False)
assert pdf_path.stat().st_size>1000
baseline=summary[summary.parameter=='baseline'].iloc[0]
fragile=summary[(summary.parameter=='cellprob_threshold') & (summary.value=='6')].iloc[0]
assert not bool(baseline.fragile) and bool(fragile.fragile)
for path,sha in plan['source_sha256'].items(): assert digest(path)==sha
record={'all_128_actual_normal_GPU_calls_accepted_after_independent_CPU_saved_label_replay':True,'original_GPU_turn_terminal_rc':1,'original_failure_private_CSV_column_expectation':{'observed':'fieldID','expected':'field'},'normal_table_canonicalisation_field_to_fieldID_verified':True,'no_GPU_rerun_or_application_source_change':True,'normal_summary_and_canonical_field_csv_equal_saved_original_GPU_masks':True,'full_acquired_fields':16,'shape':[1994,1994],'grid':grid,'calls':calls,'CUDA_profiles':profiles,'source_sha256':plan['source_sha256'],'model_sha256':freeze['model_sha256'],'plan_sha256':digest(root/'plan.json'),'benchmark_sha256':freeze['benchmark_sha256'],'verifier_sha256':digest(__file__),'raw_terminal_log_sha256':digest(raw),'summary':summary.to_dict(orient='records'),'baseline_stable':True,'deliberately_fragile_threshold_6_flagged':True,'normal_PDF_written_visual_review_pending':True,'full_production_screen_or_independent_biological_accuracy_claimed':False,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in (summary_path,field_path,pdf_path)}}
(target/'independent-acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
print('PASS: all 128 saved original full-field GPU masks independently reconstruct normal summary and canonical field CSV; real first/last CUDA profiles and exact source/model verified. Original private postflight failure retained; no GPU rerun.',flush=True)
