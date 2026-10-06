from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
encoded = root / 'cuda-embeddings-r2'
verification = root / 'independent-CUDA-embedding-verification-r1.json'
independent = json.loads(verification.read_text())
acceptance = json.loads((encoded / 'acceptance.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert independent['independent_originals_rows_labels_full_preprocessing_and_CUDA_evidence_verified']
assert independent['acceptance_sha256'] == digest(encoded / 'acceptance.json')
out = Path('features/data/565_human_labelled_CUDA_embeddings_2026-10-06')
out.mkdir(exist_ok=False)
sources = [encoded / 'embeddings.npy', encoded / 'keys-and-labels.npz', encoded / 'acceptance.json', verification, scratch / 'benchmark_pbc_embeddings_gpu_r2.py', scratch / 'verify_pbc_cuda_embeddings_r1.py', scratch / 'pbc-dinov2-embeddings-gpu-r2.log', scratch / 'pbc-CUDA-embedding-independent-r1.log', root / 'GPU-freeze-518-r2.json', root / 'FAISS-GPU-freeze-518-r2.json', scratch / 'benchmark_pbc_faiss_gpu_r2.py', Path(__file__)]
sources.extend(sorted(encoded.glob('*-profile.json')))
for source in sources:
    target = out / source.name
    if source.suffix in ('.npy', '.log') or source.name == 'acceptance.json' or source.name.endswith('-profile.json'):
        target = target.with_name(target.name + '.gz')
        target.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        assert gzip.decompress(target.read_bytes()) == source.read_bytes()
    else:
        shutil.copyfile(source, target)
receipt = {'item': 565, 'actual_normal_pretrained_DINOv2_CUDA_embeddings_independently_accepted': True, 'dataset_primary_URL': 'https://data.mendeley.com/datasets/snkd93bnjr/1', 'dataset_DOI': '10.17632/snkd93bnjr.1', 'original_expert_labelled_cells': 17092, 'unique_unambiguous_cells_encoded': 17074, 'dimensions': 384, 'model': 'vit_small_patch14_dinov2.lvd142m', 'actual_required_whole_RGB_input_size': [518, 518], 'all_original_file_pixels_keys_labels_and_independently_recomputed_derived_input_hashes_exact': True, 'actual_forward_batches': independent['normal_forward_batches'], 'every_real_model_parameter_and_forward_input_on_CUDA': True, 'actual_first_last_CUDA_profiles': independent['full_first_and_last_actual_CUDA_profiles'], 'pretrained_weights_full_state_sha256': acceptance['frozen_model_state_sha256'], 'actual_device': acceptance['device_name'], 'CUDA_version': acceptance['torch_cuda_version'], 'elapsed_embedding_seconds': acceptance['elapsed_seconds'], 'label_counts': independent['label_counts'], 'all_normal_model_source_and_plan_hashes_exact': True, 'no_production_environment_or_application_source_changed': True, 'earlier_failed_224_input_attempt_preserved_in_separate_preflight_receipt': True, 'normal_FAISS_GPU_class_retrieval_and_million_timing_pending_separate_turn': True, 'synthetic_million_scale_timing_fixture_not_acquired_cell_or_patient_accuracy_claim': True, 'GUI_pipeline_and_screen_wide_source_integration_remains_Home_owned': True, 'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}}
out.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = f'''\n2026-10-06 workstation human-labelled DINOv2 CUDA acceptance: corrected normal turn 565-pbc-dinov2-embeddings-20261006-r2 is terminal rc=0 at 07:47:40 UTC. All 17,074 unique unambiguous expert-labelled original PBC cells now have 384 finite features from actual normal spaCR embed_array and the frozen real pretrained DINOv2 encoder, using its actual 518x518 whole-RGB input. Independent CPU readback rechecks all 17,092 original file hashes, every retained key/label/original pixel array, all 17,074 resized-pixel hashes, all {independent['normal_forward_batches']} model/input CUDA forwards, complete positive first/last CUDA traces and source/plan/full pretrained model-state hashes. Receipt 565_human_labelled_CUDA_embeddings_2026-10-06.json archives all actual feature arrays, ordered labels/keys, complete compressed traces, raw terminal log, independent verifier and frozen next benchmark. The failed original 224-input run remains separately archived. This completes genuine crop-embedding GPU execution on the human-labelled cohort; class agreement and million-scale FAISS timing are separate and pending. Next planned turn is 565-pbc-FAISS-human-labels-million-20261006-r1, private official FAISS-GPU/Python3.12 environment, real HOME, 24 GiB and unchanged six-minute idle/ten-minute handoff; the 1.2M timing fixture is explicitly synthetic. Home retains remaining embedding GUI/pipeline/screen-wide source integration and CPU CI/Qt. Workstation retains all GPU/API/docs/tutorial/translation tasks; protected jobs remain untouched.\n'''
for path in (Path('features/future/565_similarity_search_find_cells_like_this.txt'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp')):
    with path.open('a') as stream:
        stream.write(note)
print('PASS independently accepted actual human-labelled CUDA embeddings archived; next FAISS acceptance remains separate.', flush=True)
