from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time

import faiss
import numpy as np
import pandas as pd

checkout=Path.cwd().resolve()
sys.meta_path[:]=[f for f in sys.meta_path if not getattr(f,'__module__','').startswith('__editable__')]
sys.path.insert(0,str(checkout))
import spacr
from spacr.active_learning import _SimilarityIndex, _similarity_agreement
from spacr.tabular import write_table

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
assert os.environ.get('SPACR_DEVICE')=='cuda' and os.environ.get('CUDA_VISIBLE_DEVICES')!=''
holder=dict(part.split('=',1) for part in (Path.home()/'.spacr/gpu/holder').read_text().split())
ancestor=os.getpid()
while ancestor and ancestor!=int(holder['pid']):
    ancestor=int(next(line.split(':',1)[1] for line in Path(f'/proc/{ancestor}/status').read_text().splitlines() if line.startswith('PPid:')))
assert ancestor==int(holder['pid'])
assert faiss.get_num_gpus()>=1
scratch=Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root=scratch/'565-pbc-retrieval-r4'
encoded=root/'cuda-embeddings-r2'
digest=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((root/'plan-518-r2.json').read_text())
freeze=json.loads((root/'FAISS-GPU-freeze-518-r2.json').read_text())
acceptance=json.loads((encoded/'acceptance.json').read_text())
assert digest(__file__)==freeze['benchmark_sha256']
assert digest(root/'plan-518-r2.json')==freeze['plan_sha256']
environment=json.loads((scratch/'565-faiss-environment-preparation-r2.json').read_text())
assert digest(scratch/'565-faiss-environment-preparation-r2.json')==freeze['environment_preparation_sha256']
for path,sha in environment['actual_FAISS_shared_library_sha256'].items(): assert digest(path)==sha
assert acceptance['real_GPU_embeddings_accepted'] and acceptance['rows']==17074
for path,record in acceptance['artifacts'].items(): assert digest(path)==record['sha256']
for path,sha in plan['source_sha256'].items(): assert digest(path)==sha
vectors=np.load(encoded/'embeddings.npy',allow_pickle=False)
with np.load(encoded/'keys-and-labels.npz',allow_pickle=False) as data:
    keys=data['keys']; labels=data['labels']; columns=data['columns']
assert vectors.shape==(17074,384) and len(keys)==len(labels)==17074
assert keys.tolist()==[row['key'] for row in plan['retrieval_images']]
assert labels.tolist()==[row['label'] for row in plan['retrieval_images']]
target=root/'cuda-faiss-r1'
target.mkdir(exist_ok=False)
features=pd.DataFrame(vectors,index=keys,columns=columns)
start=time.perf_counter()
index=_SimilarityIndex(features,backend='faiss')
assert index.backend=='faiss-gpu'
assert index._faiss.ntotal==17074
human_build_seconds=time.perf_counter()-start
scores=[]; neighbors=[]
for start in range(0,len(keys),256):
    score,row=index.search(index._matrix[start:start+256],11)
    selected=[]; selected_scores=[]
    for offset,(s,r) in enumerate(zip(score,row)):
        keep=r!=start+offset
        selected.append(r[keep][:10]); selected_scores.append(s[keep][:10])
    assert all(len(r)==10 for r in selected)
    neighbors.extend(selected); scores.extend(selected_scores)
    print('actual normal FAISS CUDA labelled neighbors',min(start+256,len(keys)),'/',len(keys),flush=True)
near=np.asarray(neighbors,dtype=np.int64)
near_scores=np.asarray(scores,dtype=np.float32)
assert near.shape==(17074,10) and not (near==np.arange(len(keys))[:,None]).any()
hits=(labels[near]==labels[:,None]).mean(axis=1)
np.savez_compressed(target/'all-human-labelled-FAISS-neighbors.npz',row=near,score=near_scores,precision_at_10=hits)
counts=pd.Series(labels).value_counts()
rows=[]
for name in sorted(counts.index):
    mask=labels==name
    chance=(counts[name]-1)/(len(keys)-1)
    precision=float(hits[mask].mean())
    rows.append({'class':name,'n':int(mask.sum()),'precision_at_k':precision,'chance':float(chance),'lift':precision/chance})
chance_all=float((counts*(counts-1)).sum()/(len(keys)*(len(keys)-1)))
rows.append({'class':'all','n':len(keys),'precision_at_k':float(hits.mean()),'chance':chance_all,'lift':float(hits.mean())/chance_all})
actual=pd.DataFrame(rows)
normal=_similarity_agreement(index,dict(zip(keys,labels)),k=10)
write_table(actual,target/'FAISS-human-class-agreement.csv')
write_table(normal,target/'normal-independent-NumPy-human-class-agreement.csv')
assert actual['class'].tolist()==normal['class'].tolist()
assert actual['n'].tolist()==normal['n'].tolist()
np.testing.assert_allclose(actual['chance'],normal['chance'],rtol=0,atol=1e-12)
agreement_delta=float(np.max(np.abs(actual['precision_at_k']-normal['precision_at_k'])))
assert agreement_delta<1e-3
control=_SimilarityIndex(features,backend='numpy',block=4096)
np.testing.assert_array_equal(control._matrix,index._matrix)
selected=np.linspace(0,len(keys)-1,64,dtype=int)
cs,cr=control.search(control._matrix[selected],11)
fs,fr=index.search(index._matrix[selected],11)
np.testing.assert_allclose(fs,cs,rtol=1e-5,atol=3e-6)
index_match=float((cr==fr).mean())
assert index_match>.999
human_summary={'backend':index.backend,'actual_faiss_index_type':type(index._faiss).__name__,'build_seconds':human_build_seconds,'rows':17074,'class_agreement':rows,'independent_normal_numpy_agreement_max_absolute_delta':agreement_delta,'independent_64_query_neighbour_index_match_fraction':index_match,'predeclared_high_retrieval_overall_precision_0_75_met':float(hits.mean())>=.75,'every_class_precision_above_its_chance':bool((actual.iloc[:-1].precision_at_k>actual.iloc[:-1].chance).all())}
del index,control,features,vectors
import gc
gc.collect()
random=np.random.default_rng(0).standard_normal((1200000,128),dtype=np.float32)
features=pd.DataFrame(random)
del random
started=time.perf_counter()
large=_SimilarityIndex(features,backend='faiss')
assert large.backend=='faiss-gpu' and large._faiss.ntotal==1200000
build_seconds=time.perf_counter()-started
queries=np.linspace(0,len(large)-1,20,dtype=int)
large.search(large.vector(str(queries[0])),100)
records=[]; saved_rows=[]; saved_scores=[]
for row in queries:
    query=large.vector(str(row))
    started=time.perf_counter()
    score,near=large.search(query,100)
    seconds=time.perf_counter()-started
    assert score.shape==near.shape==(1,100) and np.isfinite(score).all()
    np.testing.assert_allclose(score[0],large._matrix[near[0]]@query,rtol=1e-5,atol=3e-6)
    assert int(near[0,0])==int(row)
    records.append({'query_row':int(row),'seconds':seconds,'under_one_second':seconds<1.0})
    saved_rows.append(near[0]);saved_scores.append(score[0])
np.savez_compressed(target/'synthetic-million-FAISS-timing-neighbors.npz',queries=queries,row=np.asarray(saved_rows),score=np.asarray(saved_scores))
large_summary={'backend':large.backend,'actual_faiss_index_type':type(large._faiss).__name__,'rows':1200000,'dimensions':128,'seed':0,'fixture':'Explicit random float32 timing data, not acquired cell embeddings or biological validation','queries':20,'neighbors':100,'build_seconds':build_seconds,'query_timings':records,'all_queries_under_one_second':all(r['under_one_second'] for r in records),'synchronous_FAISS_returned_scores_verified_against_actual_normalized_index_vectors':True}
for path,sha in plan['source_sha256'].items(): assert digest(path)==sha
for path,sha in environment['actual_FAISS_shared_library_sha256'].items(): assert digest(path)==sha
result={'actual_normal_FAISS_GPU_execution_complete':True,'FAISS_GPUs':faiss.get_num_gpus(),'human_class_retrieval':human_summary,'synthetic_million_timing':large_summary,'FAISS_version':faiss.__version__,'packages':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy')},'script_sha256':digest(__file__),'source_plan_sha256':digest(root/'plan-518-r2.json'),'normal_spacr_source_sha256':plan['source_sha256'],'embedding_acceptance_sha256':digest(encoded/'acceptance.json'),'no_patient_split_medical_diagnosis_screen_wide_GUI_or_million_acquired_cells_claim':True,'independent_saved_neighbour_rescoring_pending':True,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(target.iterdir())}}
(target/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n')
print('COMPLETE: actual normal FAISS CUDA on all independent human-labelled embeddings and explicitly synthetic 1.2M timing fixture; independent saved-neighbor replay remains separate.',human_summary,large_summary,flush=True)
