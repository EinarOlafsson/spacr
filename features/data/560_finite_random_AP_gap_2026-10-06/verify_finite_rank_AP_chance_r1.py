from pathlib import Path
from itertools import permutations
import hashlib
import json
import os
import sys

sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
import numpy as np
import pandas as pd
from spacr.embeddings import _retrieval_scorecard

labels = ['a', 'a', 'b', 'b']
keys = ['a0', 'a1', 'b0', 'b1']
features = pd.DataFrame([[1., 0.], [1.1, 0.], [0., 1.], [0., 1.1]], index=keys)
normal = _retrieval_scorecard(features, dict(zip(keys, labels)), k=1)
means = []
for query in range(len(labels)):
    candidates = [labels[index] for index in range(len(labels)) if index != query]
    aps = []
    for order in permutations(candidates):
        hits = np.asarray([label == labels[query] for label in order])
        aps.append(float(((np.cumsum(hits) / np.arange(1, 4)) * hits).sum() / hits.sum()))
    means.append(float(np.mean(aps)))
expected = float(np.mean(means))
assert abs(expected - 11 / 18) < 1e-12
assert abs(normal['chance_map'] - 1 / 3) < 1e-12
assert abs(normal['chance_precision'] - 1 / 3) < 1e-12
for size in range(2, 8):
    harmonic = sum(1 / rank for rank in range(1, size + 1))
    for positives in range(1, size + 1):
        formula = (positives - 1) / (size - 1) + (size - positives) * harmonic / (size * (size - 1))
        average = []
        for order in set(permutations([True] * positives + [False] * (size - positives))):
            hits = np.asarray(order)
            average.append(float(((np.cumsum(hits) / np.arange(1, size + 1)) * hits).sum() / positives))
        np.testing.assert_allclose(formula, np.mean(average), rtol=0, atol=1e-12)
receipt = {'application_source_sha256': hashlib.sha256(Path('spacr/embeddings.py').read_bytes()).hexdigest(),
           'normal_four_crop_scorecard': normal, 'exact_random_rank_AP_enumerated_all_permutations': expected,
           'closed_form_random_AP_formula_matches_every_binary_ranking_for_candidate_sizes_two_through_seven': True,
           'normal_chance_precision_is_correct': True,
           'normal_chance_map_uses_class_prevalence_not_exact_random_AP_expectation': True,
           'no_GPU_biological_model_performance_or_application_source_change': True,
           'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
target = root / 'normal-scorecard-finite-rank-chance-gap-r2.json'
assert not target.exists()
target.write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS original API chance-mAP defect reproduced; exact analytical random-AP baseline exhaustively verified for all binary rankings through seven candidates.', flush=True)
