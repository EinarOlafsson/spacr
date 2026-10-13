from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import os
import shutil
import sys

sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import pandas as pd
from spacr.tabular import read_table

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '560-human-fluorescence-primary-r1'
scored = root / 'full-fluorescence-foundation-CPU-scorecards-r2'
destination = Path('features/data/560_full_human_fluorescence_scorecards_2026-10-06')
destination.mkdir(exist_ok=False)
digest = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
receipt = json.loads((scored / 'complete-scorecards.json').read_text())
assert receipt['passed'] and len(receipt['comparisons']) == 12
assert digest('spacr/embeddings.py') == receipt['source_sha256']
assert digest(scratch / 'score_full_fluorescence_foundations_CPU_r2.py') == receipt['script_sha256']
plan = json.loads((root / 'frozen-full-fluorescence-foundation-plan-r1.json').read_text())
table = read_table(scored / 'all-twelve-full-cohort-scorecards.csv')
classes = read_table(scored / 'all-original-class-scorecards.csv')
assert len(table) == 12 and len(classes) == 96
for card in receipt['comparisons']:
    name = card['backbone'] + '-' + card['cohort']
    cohort = plan['cohorts'][card['cohort']]
    with np.load(scored / (name + '-all-query-retrieval-replay.npz'), allow_pickle=False) as arrays:
        labels, keys = arrays['labels'], arrays['keys']
        assert keys.tolist() == [row['key'] for row in cohort['rows']]
        assert labels.tolist() == [row['label'] for row in cohort['rows']]
        counts = Counter(labels)
        assert dict(counts) == cohort['class_counts']
        n = len(keys)
        neighbors = arrays['neighbor_indices']
        assert neighbors.shape == (n, 10) and np.all(neighbors != np.arange(n)[:, None])
        precision = (labels[neighbors] == labels[:, None]).mean(axis=1)
        np.testing.assert_array_equal(precision, arrays['precision_at_10'])
        votes = []
        for own, near in zip(labels, labels[neighbors]):
            counted = Counter(near)
            maximum = max(counted.values())
            voted = next(label for label in near if counted[label] == maximum)
            votes.append(voted == own)
        np.testing.assert_array_equal(votes, arrays['correct_knn_vote'])
        recomputed = {'map': float(arrays['average_precision'].mean()), 'precision_at_k': float(precision.mean()), 'knn_accuracy': float(np.mean(votes))}
        for metric, value in recomputed.items():
            np.testing.assert_allclose(value, card['normal_retrieval'][metric], rtol=0, atol=1e-12)
        row = table.loc[(table['cohort'] == card['cohort']) & (table['backbone'] == card['backbone'])]
        assert len(row) == 1 and float(row['n'].iloc[0]) == n
        for metric in ('map', 'precision_at_k', 'knn_accuracy'):
            np.testing.assert_allclose(float(row[metric].iloc[0]), card['normal_retrieval'][metric], rtol=0, atol=1e-12)
        assert card['normal_classifier']['n'] == n and card['normal_classifier']['folds'] == 5
        assert not card['classifier_warnings_retained']
        assert card['normal_retrieval']['map'] > card['analytical_random_ranking_expected_map']
    print('PASS original saved all-query keys/labels/neighbours/votes/AP reductions and canonical CSV', name, flush=True)
for path in sorted(scored.iterdir()):
    if path.suffix == '.json':
        (destination / (path.name + '.gz')).write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    else:
        shutil.copyfile(path, destination / path.name)
for name in ('score_full_fluorescence_foundations_CPU_r1.py', 'score_full_fluorescence_foundations_CPU_r2.py',
             'plot_full_fluorescence_foundation_scorecards_r2.py', 'plot_full_fluorescence_foundation_scorecards_r3.py',
             'archive_full_fluorescence_foundation_scorecards_r2.py'):
    shutil.copyfile(scratch / name, destination / name)
for name in ('full-fluorescence-foundation-CPU-scorecards-r1.log', 'full-fluorescence-foundation-CPU-scorecards-r2.log',
             'full-fluorescence-foundation-normal-figure-r2.log', 'full-fluorescence-foundation-normal-figure-r3.log'):
    (destination / (name + '.gz')).write_bytes(gzip.compress((scratch / name).read_bytes(), mtime=0))
artifacts = {str(path): {'sha256': digest(path), 'bytes': path.stat().st_size} for path in sorted(destination.iterdir())}
summary = {'item': 560, 'all_twelve_normal_full_cohort_scorecards_complete': True,
           'all_twelve_independent_all_row_full_rank_AP_and_neighbour_vote_replays_match_normal_at_1e_minus12': True,
           'all_twelve_normal_five_fold_stratified_crop_classifier_diagnostics_complete_without_warnings': True,
           'actual_class_scorecard_rows': 96, 'all_twelve_aggregate_MAP_values_above_exact_random_ranking_expectation': True,
           'original_source_application_sha256': receipt['source_sha256'], 'frozen_plan_sha256': receipt['plan_sha256'],
           'normal_tabular_and_figure_style_exports_used': True,
           'final_figure_r3_visually_reviewed': False,
           'plot_definitions': 'Full-rank self-excluded mAP with exact random-ranking expectation; five-fold crop classifier accuracy with majority-class baseline.',
           'comparison_scope': 'Model-specific frozen spaCR pipelines, including three native yeast channels for other encoders and two for SubCell; not architecture-only.',
           'SubCell_epithelial_is_DNA_only_zero_protein_ablation': True,
           'provided_native_yeast_split_not_plate_or_field_independent': True,
           'original_failed_private_C_order_replay_and_corrected_F_order_replay_retained_without_guard_weakening': True,
           'model_zoo_provenance_and_exact_random_AP_chance_API_repairs_remain_open': True,
           'Cell_DINO_four_channel_SubCell_and_remaining_Home_source_integration_remain_open': True,
           'no_patient_plate_gene_holdout_clinical_or_manual_mask_accuracy_claim': True,
           'artifacts': artifacts}
Path(str(destination) + '.json').write_text(json.dumps(summary, indent=2) + '\n')
print('PASS all twelve real normal scorecards, full independent rank replay and saved-query/canonical-export verification archived.', flush=True)
