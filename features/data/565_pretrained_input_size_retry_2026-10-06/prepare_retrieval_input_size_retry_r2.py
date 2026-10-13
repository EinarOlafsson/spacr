from pathlib import Path
import ast
import gzip
import hashlib
import json
import shutil

scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'565-pbc-retrieval-r4'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((root/'plan.json').read_text())
cpu=json.loads((scratch/'565-pretrained-input-size-CPU-verification-r2.json').read_text())
assert cpu['actual_normal_CPU_embed_array_first_original_cell_passed'] and cpu['actual_normal_pretrained_model_input_size']==[518,518]
plan=json.loads(json.dumps(original))
plan['embedding_policy']['resize']='Whole original RGB image resized to the actual pretrained model input 518x518 using PIL bilinear; no crop or simulated noise'
plan['embedding_policy']['input_size']=[518,518]
plan['embedding_policy']['batch_size']=32
plan['preflight_correction']={'original_plan_sha256':digest(root/'plan.json'),'actual_input_size_CPU_verification_sha256':digest(scratch/'565-pretrained-input-size-CPU-verification-r2.json'),'terminal_GPU_r1_failed_first_forward_shape_check_224_vs_required_518':True,'no_completed_GPU_embeddings_or_FAISS_acceptance_from_r1':True,'primary_originals_labels_duplicate_policy_and_frozen_pretrained_weights_unchanged':True}
new_plan=root/'plan-518-r2.json'
assert not new_plan.exists()
new_plan.write_text(json.dumps(plan,indent=2)+'\n')
assert plan['all_original_images']==original['all_original_images'] and plan['retrieval_images']==original['retrieval_images']
assert plan['source_sha256']==original['source_sha256']
encoder=scratch/'benchmark_pbc_embeddings_gpu_r1.py'
s=encoder.read_text().replace("root / 'plan.json'","root / 'plan-518-r2.json'").replace("root / 'GPU-freeze.json'","root / 'GPU-freeze-518-r2.json'").replace("root / 'cuda-embeddings-r1'","root / 'cuda-embeddings-r2'").replace('batch_size=64','batch_size=32').replace('(3,224,224)','(3,518,518)').replace('len(rows), 64','len(rows), 32').replace('start + 64','start + 32').replace('(224,224)','(518,518)')
assert '224' not in s
new_encoder=scratch/'benchmark_pbc_embeddings_gpu_r2.py'
ast.parse(s)
new_encoder.write_text(s)
assert 'selected = rows[start:start + 32]' in s and 'for start in range(0, len(rows), 32)' in s
freeze={'files_sha256':{str(p):digest(p) for p in (new_plan,new_encoder,scratch/'565-retrieval-encoder-preparation.json',scratch/'565-pretrained-input-size-CPU-verification-r2.json')},'source_sha256':plan['source_sha256'],'real_GPU_embedding_execution_pending':True}
(root/'GPU-freeze-518-r2.json').write_text(json.dumps(freeze,indent=2)+'\n')
old_faiss=scratch/'benchmark_pbc_faiss_gpu_r1.py'
s=old_faiss.read_text().replace("root/'plan.json'","root/'plan-518-r2.json'").replace("root/'FAISS-GPU-freeze.json'","root/'FAISS-GPU-freeze-518-r2.json'").replace("root/'cuda-embeddings-r1'","root/'cuda-embeddings-r2'")
new_faiss=scratch/'benchmark_pbc_faiss_gpu_r2.py'
ast.parse(s)
new_faiss.write_text(s)
(root/'FAISS-GPU-freeze-518-r2.json').write_text(json.dumps({'benchmark_sha256':digest(new_faiss),'plan_sha256':digest(new_plan),'environment_preparation_sha256':digest(scratch/'565-faiss-environment-preparation-r2.json')},indent=2)+'\n')
out=Path('features/data/565_pretrained_input_size_retry_2026-10-06')
out.mkdir(exist_ok=False)
for p in (new_plan,root/'GPU-freeze-518-r2.json',root/'FAISS-GPU-freeze-518-r2.json',new_encoder,new_faiss,scratch/'565-pretrained-input-size-CPU-verification-r2.json',Path(__file__)):
    target=out/p.name
    if p==new_plan:
        target=target.with_name(target.name+'.gz');target.write_bytes(gzip.compress(p.read_bytes(),mtime=0))
    else:shutil.copyfile(p,target)
raw=scratch/'pbc-dinov2-embeddings-gpu-r1.log'
assert 'FINISH rc=1' in raw.read_text() and "Input height (224) doesn't match model (518)." in raw.read_text()
(out/(raw.name+'.gz')).write_bytes(gzip.compress(raw.read_bytes(),mtime=0))
receipt={'item':565,'actual_normal_CPU_pretrained_encoder_input_shape_preflight_passed':True,'required_input_shape':[518,518,3],'dimensions':384,'GPU_batch_size':32,'first_GPU_224_vs_518_shape_failure_preserved':True,'original_17092_images_17074_unique_labels_and_pretrained_weights_unchanged':True,'no_application_source_or_normal_encoder_repair':True,'actual_GPU_retry_and_FAISS_scoring_pending':True,'normal_turn_label':'565-pbc-dinov2-embeddings-20261006-r2','artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
(out.with_suffix('.json')).write_text(json.dumps(receipt,indent=2)+'\n')
note='\n2026-10-06 workstation DINOv2 input preflight correction: actual normal turn r1 is terminal rc=1 on the first forward; the current pretrained timm model requires 518x518, whereas the private preparation supplied 224x224. No completed embedding or FAISS result is claimed. A CUDA-hidden normal _backbone_encoder/embed_array run on the first original acquired cell verifies the actual 518x518 model input and 384 finite features. The separate plan-518-r2, source-bound benchmark and CPU verification are frozen, with whole-RGB PIL bilinear resize and batches of 32. All 17,092 original files, 17,074 unique unambiguous labels, duplicate/conflict policy, source and full pretrained weight freeze stay exact. The historical plan/freeze/failed turn are retained; no application source or normal encoder change is made. Receipt 565_pretrained_input_size_retry_2026-10-06.json archives this preflight and planned normal retry 565-pbc-dinov2-embeddings-20261006-r2, real HOME, 24 GiB and unchanged idle/handoff. The separately frozen FAISS r2 will queue only after the exact embedding handle finishes and its actual arrays are verified. Home should honour each actual timm model input size in the remaining embedding GUI/pipeline integration. All GPU processes and API/docs/tutorial/translation work stay workstation-owned; protected jobs remain untouched.\n'
for p in ('features/future/565_similarity_search_find_cells_like_this.txt','features/325_two_sessions_one_repo_working_protocol.temp'):
    with Path(p).open('a') as stream:stream.write(note)
print('PASS: actual CPU-tested 518x518 normal encoder retry frozen; primary originals/labels/weights unchanged; failed first shape probe archived.')
