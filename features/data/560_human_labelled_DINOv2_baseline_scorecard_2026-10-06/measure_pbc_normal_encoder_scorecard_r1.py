from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr.active_learning import _SimilarityIndex
from spacr.embeddings import EmbeddingSpec, _scored_encoder_entry, _embedding_classifier_scorecard

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
encoded = root / 'cuda-embeddings-r2'
output = root / 'normal-human-label-encoder-scorecard-r1'
output.mkdir(exist_ok=False)
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((root / 'plan-518-r2.json').read_text())
accepted = json.loads((encoded / 'acceptance.json').read_text())
retrieval = json.loads((root / 'independent-FAISS-saved-neighbor-verification-r1.json').read_text())
assert retrieval['all_required_GPU_retrieval_and_timing_criteria_met']
for path, sha in plan['source_sha256'].items():
    assert digest(path) == sha
for path, record in accepted['artifacts'].items():
    assert digest(path) == record['sha256']
values = np.load(encoded / 'embeddings.npy', allow_pickle=False)
with np.load(encoded / 'keys-and-labels.npz', allow_pickle=False) as data:
    keys, labels, columns = data['keys'], data['labels'], data['columns']
assert values.shape == (17074, 384) and len(set(keys)) == 17074
assert keys.tolist() == [row['key'] for row in plan['retrieval_images']]
assert labels.tolist() == [row['label'] for row in plan['retrieval_images']]
features = pd.DataFrame(values, index=keys, columns=columns)
named = dict(zip(keys, labels))
spec = EmbeddingSpec(backbone=plan['embedding_policy']['backbone'], channel_policy='project', channels=(0, 1, 2), channel_scale=(255., 255., 255.), device='cuda', batch_size=32)
assert spec.fingerprint() == accepted['embedding_spec_fingerprint']
started = time.monotonic()
entry = _scored_encoder_entry(spec, features, named, k=10)
assert entry.metrics['n'] == 17074 and entry.metrics['classes'] == 8
assert entry.sha256 and Path(entry.path).is_file() and digest(entry.path) == entry.sha256
normal_seconds = time.monotonic() - started
print('normal full human-label scorecard complete', entry.metrics, 'seconds', normal_seconds, flush=True)
index = _SimilarityIndex(features, backend='numpy', block=4096)
matrix = index._matrix.astype(np.float64)
hits = []
ap = []
correct = 0
ranks = np.arange(1, len(keys))
for start in range(0, len(keys), 256):
    end = min(start + 256, len(keys))
    scores = matrix[start:end] @ matrix.T
    scores[np.arange(end - start), np.arange(start, end)] = -np.inf
    ordered = np.argsort(-scores, axis=1, kind='stable')[:, :-1]
    same = labels[ordered] == labels[start:end, None]
    hits.extend(same[:, :10].mean(axis=1).tolist())
    precision = np.cumsum(same, axis=1) / ranks
    ap.extend(((precision * same).sum(axis=1) / same.sum(axis=1)).tolist())
    for own, near in zip(labels[start:end], labels[ordered[:, :10]]):
        counts = {c: int((near == c).sum()) for c in near}
        largest = max(counts.values())
        vote = next(c for c in near if counts[c] == largest)
        correct += vote == own
counts = pd.Series(labels).value_counts()
chance = float((counts * (counts - 1)).sum() / (len(keys) * (len(keys) - 1)))
independent = {'knn_accuracy': correct / len(keys), 'map': float(np.mean(ap)), 'precision_at_k': float(np.mean(hits)), 'chance_map': chance, 'chance_precision': chance, 'n': float(len(keys)), 'classes': float(len(counts))}
assert set(entry.metrics) == set(independent)
for key in independent:
    np.testing.assert_allclose(entry.metrics[key], independent[key], rtol=0, atol=1e-12)
assert abs(independent['precision_at_k'] - retrieval['human_class_agreement'][-1]['precision_at_k']) < 1e-3
np.savez_compressed(output / 'all-human-retrieval-AP-and-precision.npz', keys=keys, labels=labels, average_precision=np.asarray(ap), precision_at_10=np.asarray(hits))
print('independent complete blocked rank/AP/vote replay exact', independent, flush=True)
classifier = _embedding_classifier_scorecard(features, named, folds=5, seed=0)
assert classifier['n'] == 17074 and classifier['classes'] == 8 and classifier['folds'] == 5
print('normal five-fold diagnostic linear classifier complete', classifier, flush=True)
for path, sha in plan['source_sha256'].items():
    assert digest(path) == sha
record = {'item': 560, 'normal_full_human_labelled_DINOv2_baseline_scorecard_and_model_zoo_entry_complete': True, 'dataset_DOI': '10.17632/snkd93bnjr.1', 'all_unique_unambiguous_expert_labelled_cells': 17074, 'normal_retrieval_scorecard': entry.metrics, 'independent_complete_blocked_rank_AP_and_vote_replay_exact': independent, 'normal_model_zoo_encoder_entry': asdict(entry), 'normal_linear_classifier_scorecard': classifier, 'classifier_scope': 'Five stratified crop folds, seed zero, StandardScaler and default LogisticRegression fitted only inside each training fold. Original dataset has no patient identifiers; this is a within-cohort diagnostic, not patient/plate/well held-out accuracy.', 'retrieval_scope': 'Leave-self-out transductive crop retrieval; all exact pixel duplicates and cross-label identical groups were removed before encoding under the existing frozen input policy. No backbone or metric parameter is selected using these labels.', 'no_new_GPU_inference_or_application_source_change': True, 'actual_GPU_feature_acceptance_sha256': digest(encoded / 'acceptance.json'), 'independent_FAISS_acceptance_sha256': digest(root / 'independent-FAISS-saved-neighbor-verification-r1.json'), 'source_sha256': plan['source_sha256'], 'normal_scorecard_seconds': normal_seconds, 'script_sha256': digest(__file__), 'foundation_backbone_comparisons_Cell_DINO_and_screen_integration_remain_separate': True, 'artifacts': {str(p): {'sha256': digest(p), 'bytes': p.stat().st_size} for p in output.iterdir()}}
(output / 'acceptance.json').write_text(json.dumps(record, indent=2, default=str) + '\n')
print('PASS normal human-labelled DINOv2 baseline kNN/mAP encoder scorecard, full independent replay, cached-weight checksum and five-fold within-cohort classifier; no foundation model or independent-patient claim.', flush=True)
