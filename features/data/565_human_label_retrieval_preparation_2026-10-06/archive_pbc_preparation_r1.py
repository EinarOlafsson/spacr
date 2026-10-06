from pathlib import Path
import ast
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
benchmark = scratch / 'benchmark_pbc_embeddings_gpu_r1.py'
ast.parse(benchmark.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((root / 'plan.json').read_text())
encoder = scratch / '565-retrieval-encoder-preparation.json'
assert len(plan['all_original_images']) == 17092
assert len(plan['retrieval_images']) == 17074
assert len(plan['pixel_identical_duplicates_excluded']) == 16
assert sum(len(r) for r in plan['cross_label_pixel_conflicts_excluded_entire_groups'].values()) == 2
assert json.loads(encoder.read_text())['CPU_weight_preparation_complete']
freeze = {'files_sha256':{str(p):digest(p) for p in (root/'plan.json', benchmark,encoder)},'source_sha256':plan['source_sha256'],'real_GPU_embedding_execution_pending':True}
(root/'GPU-freeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
out = Path('features/data/565_human_label_retrieval_preparation_2026-10-06')
out.mkdir(exist_ok=False)
raw = [root/'plan.json',root/'GPU-freeze.json',root/'independent-archive-inventory.json',scratch/'pbc-primary-public-network.json',encoder,scratch/'prepare_pbc_retrieval.py',scratch/'prepare_pbc_retrieval_r2.py',scratch/'prepare_pbc_retrieval_r3.py',scratch/'prepare_pbc_retrieval_r4.py',scratch/'prepare_retrieval_encoder.py',benchmark,Path(__file__)]
logs = ['pbc-retrieval-preparation-r1.log','pbc-retrieval-preparation-r2.log','pbc-preparation-r3.log','pbc-preparation-r4.log','pbc-primary-discovery-r1.log','retrieval-encoder-preparation-r1.log','retrieval-encoder-preparation-r2.log','retrieval-encoder-installation-r1.log','faiss-gpu-environment-r1.log']
raw.extend(scratch/p for p in logs)
for source in raw:
    assert source.is_file(), source
    target = out/source.name
    if source.name in ('plan.json','pbc-primary-public-network.json') or source.suffix=='.log':
        target=target.with_name(target.name+'.gz')
        target.write_bytes(gzip.compress(source.read_bytes(),mtime=0))
        assert gzip.decompress(target.read_bytes()) == source.read_bytes()
    else:
        shutil.copyfile(source,target)
receipt={'item':565,'actual_original_dataset_and_pretrained_encoder_preparation_accepted':True,'dataset_primary_url':plan['dataset_primary_url'],'dataset_DOI':plan['dataset_DOI'],'dataset_license':plan['license'],'dataset_archive_sha256':plan['dataset_archive_sha256'],'dataset_archive_bytes':plan['dataset_archive_bytes'],'all_original_clinician_labelled_images':17092,'unique_unambiguous_retrieval_images':17074,'same_label_extra_pixel_duplicates_excluded':16,'cross_label_conflicting_original_images_excluded_as_whole_group':2,'all_original_geometry_pixels_and_classes_preserved':True,'independent_original_archive_inventory_verified':True,'pretrained_encoder':json.loads(encoder.read_text()),'normal_CPU_main_env_not_modified_private_timm_venv':True,'FAISS_python_313_resolution_failed_retained_python_312_isolated_retry_pending':True,'first_preparation_failures_retained_no_original_repair_or_label_selection':True,'actual_GPU_embeddings_FAISS_timing_class_agreement_and_GUI_screen_index_still_pending':True,'million_scale_fixture_explicitly_synthetic_not_biological_validation':True,'no_patient_holdout_or_independent_medical_accuracy_claim':True,'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.iterdir())}}
(out.with_suffix('.json')).write_text(json.dumps(receipt,indent=2)+'\n')
note='''
2026-10-06 workstation human-labelled retrieval GPU preparation: original public PBC archive (DOI 10.17632/snkd93bnjr.1, CC BY 4.0) matches its published 281,366,219 bytes and SHA256. Independent complete archive inventory verifies all 17,092 original RGB cells/eight expert-labelled classes and seven actual geometries. All originals remain intact. One cross-label pixel-identical group (neutrophil/eosinophil, two images) is excluded in full before embedding; sixteen additional same-label copies are excluded without choosing labels, leaving 17,074 unique unambiguous cells. Failed hidden-file/geometry/conflicting-label preflights are retained. Actual pretrained DINOv2 weights and full normal CPU model state are frozen in a private timm venv; no production environment or application source was changed. Normal spaCR embed_array will use its real pretrained encoder cached once, with explicit whole-RGB resize/scale and actual first/last CUDA profiles. Planned normal GPU turn 565-pbc-dinov2-embeddings-20261006-r1 follows the already queued 578 robustness/558 virtual staining turns with real HOME, 24 GiB and unchanged idle/handoff. GPU execution, independent class retrieval and FAISS million-scale timing remain pending; random million-scale timing vectors will be labelled synthetic. FAISS's current package rejects Python 3.13; the terminal failed solve is retained and an isolated Python 3.12 retry is preparing. Receipt 565_human_label_retrieval_preparation_2026-10-06.json archives accepted inputs/model/benchmark and failed preflights. Home retains CPU CI/coverage/Qt and remaining screen-wide GUI/pipeline source integration; workstation retains all GPU, API, documentation, translations and tutorials. Protected livecell/cellposeTIME jobs remain untouched. No patient-split or medical accuracy claim is made.
'''
for p in (Path('features/future/565_similarity_search_find_cells_like_this.txt'),Path('features/325_two_sessions_one_repo_working_protocol.temp')):
    with p.open('a') as stream: stream.write(note)
print('PASS: original human-label cohort, independent duplicate/geometry inventory, pretrained model state and exact normal GPU benchmark frozen and archived; no GPU acceptance claimed.')
