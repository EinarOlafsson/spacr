from pathlib import Path
import gzip
import hashlib
import json
import shutil

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
target = root / 'cuda-faiss-r1'
verified = root / 'independent-FAISS-saved-neighbor-verification-r1.json'
actual = json.loads((target / 'acceptance.json').read_text())
independent = json.loads(verified.read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert independent['all_required_GPU_retrieval_and_timing_criteria_met']
assert independent['actual_GPU_acceptance_sha256'] == digest(target / 'acceptance.json')
assert actual['human_class_retrieval']['backend'] == actual['synthetic_million_timing']['backend'] == 'faiss-gpu'
out = Path('features/data/565_human_label_FAISS_GPU_acceptance_2026-10-06')
out.mkdir(exist_ok=False)
sources = list(sorted(target.iterdir())) + [verified, scratch / 'benchmark_pbc_faiss_gpu_r2.py', scratch / 'verify_pbc_FAISS_saved_neighbors_r1.py', scratch / 'pbc-FAISS-human-million-gpu-r1.log', scratch / 'pbc-FAISS-independent-neighbor-replay-r1.log', root / 'FAISS-GPU-freeze-518-r2.json', scratch / '565-faiss-environment-preparation-r2.json', Path(__file__)]
for source in sources:
    output = out / source.name
    if source.suffix == '.log':
        output = output.with_name(output.name + '.gz')
        output.write_bytes(gzip.compress(source.read_bytes(), mtime=0))
        assert gzip.decompress(output.read_bytes()) == source.read_bytes()
    else:
        shutil.copyfile(source, output)
human = actual['human_class_retrieval']
million = actual['synthetic_million_timing']
receipt = {'item': 565, 'actual_normal_FAISS_GPU_human_class_retrieval_and_million_timing_independently_accepted': True, 'backend': human['backend'], 'actual_index_type': human['actual_faiss_index_type'], 'FAISS_version': actual['FAISS_version'], 'human_labelled_original_DINOv2_cells': human['rows'], 'human_class_precision_at_10': human['class_agreement'][-1]['precision_at_k'], 'human_class_chance_precision': human['class_agreement'][-1]['chance'], 'every_human_class_above_chance': human['every_class_precision_above_its_chance'], 'complete_per_class_results': human['class_agreement'], 'all_170740_original_human_neighbor_scores_labels_and_hits_independently_recomputed': True, 'all_normal_class_agreement_and_CSV_rows_reproduced': True, 'independent_64_query_numpy_neighbor_match_fraction': independent['64_query_independent_normal_numpy_neighbor_match_fraction'], 'synthetic_timing_fixture': million, 'synthetic_million_seed_matrix_normalization_and_all_20_top100_queries_independently_recreated_on_CPU': True, 'synthetic_million_independent_numpy_neighbor_match_fraction': independent['million_normal_numpy_neighbor_match_fraction'], 'all_twenty_actual_GPU_queries_under_one_second': independent['twenty_actual_GPU_queries_under_one_second_passed'], 'GPU_query_seconds_min_max': independent['actual_GPU_timing_min_max_seconds'], 'source_inputs_pretrained_encoder_features_FAISS_library_and_plan_hashes_exact': True, 'no_million_acquired_cell_patient_split_diagnostic_accuracy_or_GUI_screen_integration_claim': True, 'all_pending_GPU_retrieval_and_million_timing_checks_closed_GUI_pipeline_screen_wide_integration_still_Home_owned': True, 'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in sorted(out.iterdir())}}
out.with_suffix('.json').write_text(json.dumps(receipt, indent=2) + '\n')
note = '''
2026-10-06 workstation human-labelled FAISS GPU acceptance: normal turn 565-pbc-FAISS-human-labels-million-20261006-r1 is terminal rc=0 at 07:58:00 UTC. The actual normal spaCR similarity index is faiss-gpu/GpuIndexFlat (official FAISS1.15.1), with no CPU fallback. All 17,074 unique unambiguous expert-labelled original PBC DINOv2 feature rows have self-excluded exact top-ten neighbours. Overall precision@10 is 0.8568876654562493 versus chance 0.14397477263482286; each of eight classes beats its prevalence baseline, including monocyte 0.719662 versus 0.083055 and platelet 0.983475 versus 0.137469. The normal independent NumPy class agreement matches exactly, and every sampled neighbour index matches. The separately declared synthetic seed-zero 1.2M x128 index returns all twenty synchronous top100 queries in 0.002894-0.003926 seconds (build 3.243079 seconds); it is not 1.2M acquired cells. Independent CPU readback recomputes all 170,740 saved human neighbour scores, labels/hits/class rows and canonical CSV, reproduces the 64 real queries, and reconstructs the entire synthetic matrix/normalization and all twenty top100 searches with exact neighbour-index matches. Source/input/actual feature/FAISS-library/plan hashes remain exact. Receipt 565_human_label_FAISS_GPU_acceptance_2026-10-06.json archives all actual neighbours/scores, full class/timing records, raw terminal log, frozen benchmark and independent replay. This closes the genuine human-labelled embedding/GPU-FAISS/million-timing gaps; item565 remains open for Home's remaining embedding GUI/pipeline and multi-plate screen-wide index integration. No patient split, clinical diagnosis accuracy or million acquired cell claim is made. No GPU turn remains live from this benchmark; all further GPU processes stay workstation-owned and protected livecell/cellposeTIME jobs remain untouched. Home retains CPU CI/coverage/Qt/source and workstation retains API/docs/tutorial/translations.
'''
for path in (Path('features/future/565_similarity_search_find_cells_like_this.txt'), Path('features/new/615_requested_batch_api_docs_translations_tutorials_and_acceptance.txt'), Path('features/325_two_sessions_one_repo_working_protocol.temp')):
    with path.open('a') as stream:
        stream.write(note)
print('PASS all pending GPU human-labelled retrieval/million timing checks independently accepted and archived; whole item565 remains open for GUI/pipeline/screen-wide integration.', flush=True)
