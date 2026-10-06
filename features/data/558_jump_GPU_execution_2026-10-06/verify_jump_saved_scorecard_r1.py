from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkout=Path.cwd().resolve()
sys.meta_path[:]=[f for f in sys.meta_path if not getattr(f,'__module__','').startswith('__editable__')]
sys.path.insert(0,str(checkout))
import spacr
from spacr import deep_spacr as deep
from spacr.scorecard import match_objects
from spacr.schema import canonicalise_frame
assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'558-jump-current-r1'
target=root/'cuda-r1'
read=lambda p:json.loads(Path(p).read_text())
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=read(root/'plan.json');freeze=read(root/'GPU-freeze.json');actual=read(target/'acceptance.json')
assert actual['actual_GPU_training_and_scorecard_complete'] and actual['independent_saved_label_rescoring_pending']
assert len(actual['actual_Cellpose_calls'])==60
assert digest(root/'plan.json')==freeze['plan_sha256']==actual['source_plan_sha256']
assert digest(scratch/'benchmark_jump_virtual_stain_gpu.py')==freeze['benchmark_sha256']==actual['benchmark_sha256']
assert digest(freeze['cpsam_path'])==freeze['cpsam_sha256']==actual['cpsam_sha256']
for p,sha in plan['source_sha256'].items(): assert digest(p)==sha
for r in plan['original_downloads']: assert digest(r['path'])==r['sha256']
for r in plan['training']+plan['held_out']: assert digest(r['field_path'])==r['field_sha256']
assert {r['plate'] for r in plan['training']}.isdisjoint({r['plate'] for r in plan['held_out']})
assert len(plan['training'])==48 and len(plan['held_out'])==20
model=target/'virtual_stain_c0.pt'
assert digest(model)==actual['trained_model_sha256']
loaded=deep._load_virtual_stain(model,device='cpu')
assert len(loaded['losses'])==len(actual['training_progress'])==20
np.testing.assert_array_equal(loaded['losses'],[r['loss'] for r in actual['training_progress']])
for key in ('sources','target','scale','crop','per_field','epochs','base','depth','seed'):
    assert loaded[key]==plan['training_parameters'][key],(key,loaded.get(key),plan['training_parameters'][key])
profiles={}
for name in ('actual-first-training-forward-profile.json','actual-cellpose-call-00-profile.json','actual-cellpose-call-59-profile.json'):
    path=target/name
    events=[r for r in read(path)['traceEvents'] if r.get('cat')=='kernel']
    assert events
    profiles[name]={'positive_CUDA_kernel_events':len(events),'sha256':digest(path),'bytes':path.stat().st_size}
rows=[];pairs=[];prediction_cpu_checks=[]
for index,input_row in enumerate(plan['held_out']):
    name=Path(input_row['field_path']).stem
    field=np.load(input_row['field_path'],allow_pickle=False)
    assert field.shape==(1080,1080,4)
    real=deep._vs_normalize(field[...,loaded['target']])
    source=deep._vs_normalize(field[...,loaded['sources'][0]])
    prediction_path=target/(name+'-prediction.npy')
    predicted=np.load(prediction_path,allow_pickle=False)
    predicted_cpu=deep._predict_virtual_stain(loaded,field)
    assert predicted.shape==predicted_cpu.shape==(1080,1080)
    np.testing.assert_allclose(predicted_cpu,predicted,rtol=2e-5,atol=2e-5)
    prediction_cpu_checks.append({'field':name,'saved_actual_GPU_prediction_sha256':digest(prediction_path),'normal_CPU_saved_model_replay_max_abs_error':float(np.max(np.abs(predicted-predicted_cpu)))})
    labels={}
    for kind,plane in (('real',real),('predicted',predicted),('input_baseline',source)):
        offset={'real':0,'predicted':1,'input_baseline':2}[kind]
        call=actual['actual_Cellpose_calls'][index*3+offset]
        assert call['field']==name and call['kind']==kind and call['model_device'].startswith('cuda')
        assert call['plane_sha256']==hashlib.sha256(plane.tobytes()).hexdigest()
        label_path=Path(call['labels_path'])
        assert digest(label_path)==call['labels_sha256']
        with np.load(label_path,allow_pickle=False) as saved: mask=saved['labels']
        assert mask.shape==plane.shape and mask.dtype.kind in 'iu'
        assert len(np.unique(mask[mask>0]))==call['objects']
        labels[kind]=mask
    for kind,plane in (('predicted',predicted),('input_baseline',source)):
        row={'field':name,'kind':kind,**deep._vs_pixel_metrics(real,plane)}
        for iou in (.5,.75):
            match=match_objects(labels['real'],labels[kind],iou)
            tag=int(round(iou*100))
            row[f'f1_{tag}']=match.f1;row[f'precision_{tag}']=match.precision;row[f'recall_{tag}']=match.recall
            pairs.append({'field':name,'kind':kind,'iou':iou,'tp':match.true_positives,'fp':match.false_positives,'fn':match.false_negatives,'f1':match.f1,'matched_pairs':match.pairs,'matched_ious':match.ious})
        row['n_real']=match.n_truth;row['n_pred']=match.n_pred
        rows.append(row)
    print('independently rescored saved actual GPU field',index+1,'/20',name,flush=True)
recomputed=pd.DataFrame(rows)
pd.testing.assert_frame_equal(recomputed,pd.DataFrame(actual['scorecard']))
pd.testing.assert_frame_equal(pd.read_csv(target/'actual-Cellpose-SAM-scorecard.csv'),canonicalise_frame(recomputed),check_dtype=False)
means=recomputed.groupby('kind')[['pearson','ssim','f1_50','f1_75']].mean().to_dict(orient='index')
assert means==actual['summary_mean_per_field']
for p,sha in plan['source_sha256'].items(): assert digest(p)==sha
assert digest(model)==actual['trained_model_sha256']
raw=scratch/'jump-virtual-stain-gpu-r1.log'
assert 'FINISH rc=0' in raw.read_text()
result={'independent_item_532_saved_label_and_pixel_metrics_rescoring_accepted':True,'all_40_scorecard_rows_match_actual_GPU_report_and_normal_canonical_CSV':True,'all_60_original_Cellpose_label_plane_source_hashes_and_object_counts_verified':True,'all_20_GPU_predictions_reproduced_from_saved_model_on_CPU_with_predeclared_2e_5_tolerance':prediction_cpu_checks,'actual_CUDA_profiles':profiles,'training_fields':48,'held_out_fields':20,'plate_disjoint_split':True,'normal_training_metadata_and_all_20_epoch_losses_verified':True,'trained_model_sha256':digest(model),'summary_mean_per_field':means,'independent_matches':pairs,'source_sha256':plan['source_sha256'],'plan_sha256':digest(root/'plan.json'),'raw_terminal_log_sha256':digest(raw),'verifier_sha256':digest(__file__),'reference':'Real Hoechst Cellpose-SAM labels are algorithmic reference, not human truth','no_historical_model_reuse_second_cell_line_patient_holdout_or_four_biological_replicates_claim':True}
(target/'independent-acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: all 20 saved model/predictions, all 60 original CUDA labels and all 40 normal Cellpose scorecard rows independently verified.',means,flush=True)
