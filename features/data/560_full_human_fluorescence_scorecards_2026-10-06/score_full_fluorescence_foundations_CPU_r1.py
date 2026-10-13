from pathlib import Path
from collections import Counter
import dataclasses
import hashlib
import json
import os
import sys
import time
import warnings

sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import pandas as pd
from spacr.embeddings import EmbeddingSpec, _scored_encoder_entry, _embedding_classifier_scorecard
from spacr.tabular import write_table

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
encoded = root / 'full-fluorescence-foundation-CUDA-r1'
output = root / 'full-fluorescence-foundation-CPU-scorecards-r1'
output.mkdir(exist_ok=False)
def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()
plan = json.loads((root / 'frozen-full-fluorescence-foundation-plan-r1.json').read_text())
assert digest('spacr/embeddings.py') == plan['application_source_sha256']
execution = json.loads((encoded / 'complete-execution.json').read_text())
assert execution['passed'] and len(execution['comparisons']) == 12
assert execution['plan_sha256'] == digest(root / 'frozen-full-fluorescence-foundation-plan-r1.json')
cards = []
table = []
per_class_table = []
for run in execution['comparisons']:
    name = run['backbone'] + '-' + run['cohort']
    assert digest(run['full_features']['path']) == run['full_features']['sha256']
    for profile in run['first_last_actual_CUDA_profiles']:
        assert profile['CUDA_kernel_events'] > 0 and digest(profile['trace_path']) == profile['sha256']
    values = np.load(run['full_features']['path'], mmap_mode='r')
    keys = np.asarray([row['key'] for row in run['ordered_rows']])
    labels = np.asarray([row['label'] for row in run['ordered_rows']])
    assert values.shape == (len(keys), run['dimensions']) and len(set(keys)) == len(keys)
    assert run['ordered_rows'] == plan['cohorts'][run['cohort']]['rows']
    counts = Counter(labels)
    assert dict(counts) == plan['cohorts'][run['cohort']]['class_counts']
    frame = pd.DataFrame(values, index=keys, columns=run['columns'])
    named = dict(zip(keys, labels))
    spec = EmbeddingSpec(**{**run['normal_spec'], 'channels': tuple(run['normal_spec']['channels']), 'channel_scale': tuple(run['normal_spec']['channel_scale'])})
    before = time.perf_counter()
    entry = _scored_encoder_entry(spec, frame, named, k=10)
    normal_seconds = time.perf_counter() - before
    assert entry.metrics['n'] == len(keys) and entry.metrics['classes'] == len(counts)
    print('PASS normal full-cohort scorecard', name, entry.metrics, 'seconds', normal_seconds, flush=True)
    if entry.sha256:
        assert digest(entry.path) == entry.sha256
        assert any(metadata['sha256'] == entry.sha256 for metadata in plan['models'][run['backbone']]['receipt']['actual_cached_files'].values())
    else:
        assert run['backbone'] in ('openphenom', 'chada_vit', 'subcell')
    raw = np.asarray(values, dtype=np.float32).copy()
    assert np.isfinite(raw).all()
    centre, spread = np.median(raw, axis=0), raw.std(axis=0)
    usable = np.isfinite(spread) & (spread > 0)
    matrix = raw[:, usable]
    matrix -= centre[usable]
    matrix /= spread[usable]
    length = np.linalg.norm(matrix, axis=1, keepdims=True)
    length[length == 0] = 1
    matrix /= length
    matrix = np.ascontiguousarray(matrix, dtype=np.float64)
    n = len(keys)
    ap = np.empty(n, dtype=np.float64)
    precision = np.empty(n, dtype=np.float64)
    vote_correct = np.empty(n, dtype=bool)
    neighbor_indices = np.empty((n, 10), dtype=np.int32)
    neighbor_scores = np.empty((n, 10), dtype=np.float64)
    unique_classes, integer_labels = np.unique(labels, return_inverse=True)
    rank = np.arange(1, n, dtype=np.float64)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        similarity = matrix[start:stop] @ matrix.T
        similarity[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        ordered = np.argsort(-similarity, axis=1, kind='stable')[:, :-1]
        matching = integer_labels[ordered] == integer_labels[start:stop, None]
        positive_count = matching.sum(axis=1)
        assert np.all(positive_count == np.asarray([counts[label] - 1 for label in labels[start:stop]]))
        ap[start:stop] = ((np.cumsum(matching, axis=1, dtype=np.float64) / rank) * matching).sum(axis=1) / positive_count
        precision[start:stop] = matching[:, :10].mean(axis=1)
        neighbor_indices[start:stop] = ordered[:, :10]
        neighbor_scores[start:stop] = np.take_along_axis(similarity, ordered[:, :10], axis=1)
        for j, own in enumerate(integer_labels[start:stop]):
            nearby = integer_labels[ordered[j, :10]]
            votes = np.bincount(nearby, minlength=len(unique_classes))
            picked = next(label for label in nearby if votes[label] == votes.max())
            vote_correct[start + j] = picked == own
    chance = sum(count * (count - 1) for count in counts.values()) / (n * (n - 1))
    candidate_n = n - 1
    harmonic = float(np.sum(1.0 / np.arange(1, candidate_n + 1)))
    exact_random_ap_by_class = {label: (count - 2) / (candidate_n - 1) + (candidate_n - count + 1) * harmonic / (candidate_n * (candidate_n - 1)) for label, count in counts.items()}
    exact_random_map = sum(counts[label] * value for label, value in exact_random_ap_by_class.items()) / n
    replay = {'knn_accuracy': float(vote_correct.mean()), 'map': float(ap.mean()), 'precision_at_k': float(precision.mean()),
              'chance_map': chance, 'chance_precision': chance, 'n': float(n), 'classes': float(len(counts))}
    assert set(replay) == set(entry.metrics)
    for metric in replay:
        np.testing.assert_allclose(entry.metrics[metric], replay[metric], rtol=0, atol=1e-12, err_msg=name + '/' + metric)
    assert np.all(neighbor_indices != np.arange(n)[:, None])
    np.savez_compressed(output / (name + '-all-query-retrieval-replay.npz'), keys=keys, labels=labels,
                        average_precision=ap, precision_at_10=precision, correct_knn_vote=vote_correct,
                        neighbor_indices=neighbor_indices, neighbor_cosine_scores=neighbor_scores)
    print('PASS independent all-row full-rank AP, top-ten and majority-vote replay', name, flush=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        classifier = _embedding_classifier_scorecard(frame, named, folds=5, seed=0)
    warning_records = [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in caught]
    assert classifier['n'] == n and classifier['classes'] == len(counts) and classifier['folds'] == 5
    print('PASS normal five-fold crop classifier', name, classifier, 'warnings', len(warning_records), flush=True)
    classes = []
    for label in sorted(counts):
        chosen = labels == label
        row = {'cohort': run['cohort'], 'backbone': run['backbone'], 'class': label, 'n': int(chosen.sum()),
               'map': float(ap[chosen].mean()), 'precision_at_10': float(precision[chosen].mean()),
               'knn_accuracy': float(vote_correct[chosen].mean()), 'class_prevalence_chance': (counts[label] - 1) / (n - 1)}
        classes.append(row)
        per_class_table.append(row)
    card = {'backbone': run['backbone'], 'cohort': run['cohort'], 'all_original_eligible_rows_scored': n,
            'normal_model_zoo_entry': dataclasses.asdict(entry), 'normal_retrieval': entry.metrics,
            'independent_all_row_retrieval': replay, 'normal_classifier': classifier, 'classifier_warnings_retained': warning_records,
            'per_class_retrieval': classes, 'normal_scorecard_seconds': normal_seconds,
            'analytical_random_ranking_expected_map': exact_random_map,
            'normal_chance_map_is_prevalence_approximation_API_gap_preserved': True,
            'full_features_sha256': run['full_features']['sha256'], 'complete_model_state_sha256': run['complete_ordered_model_state_sha256'],
            'foundation_model_zoo_empty_hash_and_timm_attribution_gap_preserved_not_claimed_fixed': not bool(entry.sha256)}
    (output / (name + '-scorecard.json')).write_text(json.dumps(card, indent=2) + '\n')
    cards.append(card)
    table.append({'cohort': run['cohort'], 'backbone': run['backbone'], 'n': n, 'classes': len(counts), 'dimensions': values.shape[1],
                  **entry.metrics, 'classifier_accuracy': classifier['accuracy'], 'classifier_accuracy_sd': classifier['accuracy_sd'],
                  'analytical_random_ranking_expected_map': exact_random_map,
                  'classifier_majority_chance': classifier['chance'], 'actual_CUDA_embed_seconds': run['normal_embed_seconds_including_boundary_profiles']})
    del frame, raw, matrix, values
write_table(pd.DataFrame(table), output / 'all-twelve-full-cohort-scorecards.csv')
write_table(pd.DataFrame(per_class_table), output / 'all-original-class-scorecards.csv')
assert digest('spacr/embeddings.py') == plan['application_source_sha256']
receipt = {'passed': True, 'comparisons': cards, 'source_sha256': plan['application_source_sha256'],
           'plan_sha256': execution['plan_sha256'], 'original_complete_GPU_execution_sha256': digest(encoded / 'complete-execution.json'),
           'all_twelve_full_normal_scorecards_and_all_row_independent_rank_AP_vote_replays_complete': True,
           'normal_five_fold_crop_classifier_all_twelve_complete': True,
           'no_patient_plate_gene_holdout_clinical_manual_mask_or_Cell_DINO_claim': True,
           'foundation_model_zoo_provenance_API_gap_remains_separate': True,
           'script_sha256': digest(__file__)}
(output / 'complete-scorecards.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS ALL TWELVE NORMAL FULL-COHORT SCORECARDS, INDEPENDENT FULL-RANK REPLAYS AND FIVE-FOLD CROP DIAGNOSTICS', flush=True)
