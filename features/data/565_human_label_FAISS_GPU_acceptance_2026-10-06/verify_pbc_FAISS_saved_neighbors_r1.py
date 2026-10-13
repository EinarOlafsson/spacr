from pathlib import Path
from collections import Counter
import hashlib
import json
import sys

import numpy as np
import pandas as pd

checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr.active_learning import _SimilarityIndex
from spacr.schema import canonicalise_frame

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
encoded = root / 'cuda-embeddings-r2'
target = root / 'cuda-faiss-r1'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((root / 'plan-518-r2.json').read_text())
actual = json.loads((target / 'acceptance.json').read_text())
assert actual['actual_normal_FAISS_GPU_execution_complete'] and actual['FAISS_GPUs'] >= 1
assert actual['human_class_retrieval']['backend'] == 'faiss-gpu'
assert actual['synthetic_million_timing']['backend'] == 'faiss-gpu'
assert actual['embedding_acceptance_sha256'] == digest(encoded / 'acceptance.json')
assert actual['source_plan_sha256'] == digest(root / 'plan-518-r2.json')
for path, sha in plan['source_sha256'].items():
    assert digest(path) == sha
for path, record in actual['artifacts'].items():
    assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
vectors = np.load(encoded / 'embeddings.npy', allow_pickle=False)
with np.load(encoded / 'keys-and-labels.npz', allow_pickle=False) as data:
    keys, labels, columns = data['keys'], data['labels'], data['columns']
assert keys.tolist() == [r['key'] for r in plan['retrieval_images']]
assert labels.tolist() == [r['label'] for r in plan['retrieval_images']]
index = _SimilarityIndex(pd.DataFrame(vectors, index=keys, columns=columns), backend='numpy', block=4096)
with np.load(target / 'all-human-labelled-FAISS-neighbors.npz', allow_pickle=False) as data:
    near, scores, stored_hits = data['row'], data['score'], data['precision_at_10']
assert near.shape == scores.shape == (17074, 10)
assert ((near >= 0) & (near < 17074)).all()
assert not (near == np.arange(17074)[:, None]).any()
assert all(len(set(r)) == 10 for r in near)
assert (np.diff(scores, axis=1) <= 0).all()
independent_scores = np.einsum('ijk,ik->ij', index._matrix[near], index._matrix)
np.testing.assert_allclose(scores, independent_scores, rtol=1e-5, atol=3e-6)
hits = (labels[near] == labels[:, None]).mean(axis=1)
np.testing.assert_array_equal(stored_hits, hits)
counts = Counter(labels.tolist())
rows = []
for name, count in sorted(counts.items()):
    precision = float(hits[labels == name].mean())
    chance = (count - 1) / 17073
    rows.append({'class': name, 'n': count, 'precision_at_k': precision, 'chance': chance, 'lift': precision / chance})
chance_all = sum(n * (n - 1) for n in counts.values()) / (17074 * 17073)
rows.append({'class': 'all', 'n': 17074, 'precision_at_k': float(hits.mean()), 'chance': chance_all, 'lift': float(hits.mean()) / chance_all})
recomputed = pd.DataFrame(rows)
pd.testing.assert_frame_equal(pd.DataFrame(actual['human_class_retrieval']['class_agreement']), recomputed, check_dtype=False)
pd.testing.assert_frame_equal(pd.read_csv(target / 'FAISS-human-class-agreement.csv'), canonicalise_frame(recomputed), check_dtype=False)
selected = np.linspace(0, 17073, 64, dtype=int)
control_scores, control_rows = index.search(index._matrix[selected], 11)
control = np.asarray([r[r != own][:10] for r, own in zip(control_rows, selected)])
neighbor_fraction = float((control == near[selected]).mean())
assert neighbor_fraction > .999
large_record = actual['synthetic_million_timing']
assert (large_record['rows'], large_record['dimensions'], large_record['seed'], large_record['queries'], large_record['neighbors']) == (1200000, 128, 0, 20, 100)
del index, vectors
features = pd.DataFrame(np.random.default_rng(0).standard_normal((1200000, 128), dtype=np.float32))
large = _SimilarityIndex(features, backend='numpy', block=65536)
del features
with np.load(target / 'synthetic-million-FAISS-timing-neighbors.npz', allow_pickle=False) as data:
    queries, large_near, large_scores = data['queries'], data['row'], data['score']
np.testing.assert_array_equal(queries, np.linspace(0, 1199999, 20, dtype=int))
assert large_near.shape == large_scores.shape == (20, 100)
matrix_queries = large._matrix[queries]
control_scores, control_rows = large.search(matrix_queries, 100)
np.testing.assert_allclose(large_scores, control_scores, rtol=1e-5, atol=3e-6)
million_neighbor_fraction = float((large_near == control_rows).mean())
assert million_neighbor_fraction > .999
for query, record, neighbors in zip(queries, large_record['query_timings'], large_near):
    assert query == record['query_row'] == neighbors[0]
    assert record['seconds'] > 0 and record['under_one_second'] == (record['seconds'] < 1)
speed_passed = all(r['under_one_second'] for r in large_record['query_timings'])
assert large_record['all_queries_under_one_second'] == speed_passed
human_passed = float(hits.mean()) >= .75
classes_passed = bool((recomputed.iloc[:-1].precision_at_k > recomputed.iloc[:-1].chance).all())
assert actual['human_class_retrieval']['predeclared_high_retrieval_overall_precision_0_75_met'] == human_passed
assert actual['human_class_retrieval']['every_class_precision_above_its_chance'] == classes_passed
output = root / 'independent-FAISS-saved-neighbor-verification-r1.json'
assert not output.exists()
record = {'independent_all_saved_human_neighbour_scores_labels_class_counts_and_normal_CSV_verified': True, 'all_170740_human_neighbour_cosine_scores_recomputed_from_original_features': True, '64_query_independent_normal_numpy_neighbor_match_fraction': neighbor_fraction, 'synthetic_million_fixture_seed_matrix_normalization_and_20_top100_searches_recreated_on_CPU': True, 'million_normal_numpy_neighbor_match_fraction': million_neighbor_fraction, 'human_class_agreement': rows, 'predeclared_human_precision_0_75_passed': human_passed, 'each_class_above_chance_passed': classes_passed, 'twenty_actual_GPU_queries_under_one_second_passed': speed_passed, 'actual_GPU_timing_min_max_seconds': [min(r['seconds'] for r in large_record['query_timings']), max(r['seconds'] for r in large_record['query_timings'])], 'all_required_GPU_retrieval_and_timing_criteria_met': speed_passed and human_passed and classes_passed, 'source_plan_embeddings_FAISS_artifact_hashes_exact': True, 'actual_GPU_acceptance_sha256': digest(target / 'acceptance.json'), 'verifier_sha256': digest(__file__), 'no_million_acquired_cells_patient_split_or_GUI_screen_integration_claim': True}
output.write_text(json.dumps(record, indent=2) + '\n')
print('PASS independent saved actual FAISS neighbour replay; required GPU retrieval/timing criteria met:', record['all_required_GPU_retrieval_and_timing_criteria_met'], 'human precision', float(hits.mean()), 'actual timing range', record['actual_GPU_timing_min_max_seconds'], flush=True)
