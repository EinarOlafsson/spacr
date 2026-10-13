from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import os
import sys
import time
import resource

import numpy as np
import pandas as pd

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
assert os.environ.get('OPENBLAS_NUM_THREADS') == '2'
checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr.embeddings import EmbeddingSpec, _scored_encoder_entry

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
encoded = root / 'cuda-embeddings-r2'
baseline = root / 'normal-human-label-encoder-scorecard-r3'
original = json.loads((baseline / 'acceptance.json').read_text())
accepted = json.loads((encoded / 'acceptance.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, record in accepted['artifacts'].items():
    assert digest(path) == record['sha256']
for path, sha in original['source_sha256'].items():
    if path != 'spacr/embeddings.py':
        assert digest(path) == sha
assert digest('spacr/embeddings.py') != original['source_sha256']['spacr/embeddings.py']
with np.load(encoded / 'keys-and-labels.npz', allow_pickle=False) as data:
    keys, labels, columns = data['keys'], data['labels'], data['columns']
with np.load(baseline / 'all-human-retrieval-AP-and-precision.npz', allow_pickle=False) as data:
    np.testing.assert_array_equal(data['keys'], keys)
    np.testing.assert_array_equal(data['labels'], labels)
    assert len(data['average_precision']) == 17074
features = pd.DataFrame(np.load(encoded / 'embeddings.npy', allow_pickle=False), index=keys, columns=columns)
spec = EmbeddingSpec(backbone='vit_small_patch14_dinov2.lvd142m', channel_policy='project', channels=(0, 1, 2), channel_scale=(255., 255., 255.), device='cuda', batch_size=32)
assert spec.fingerprint() == accepted['embedding_spec_fingerprint']
started = time.monotonic()
entry = _scored_encoder_entry(spec, features, dict(zip(keys, labels)), k=10)
elapsed = time.monotonic() - started
assert entry.metrics['n'] == 17074
for key, value in original['normal_retrieval_scorecard'].items():
    np.testing.assert_allclose(entry.metrics[key], value, rtol=0, atol=1e-12)
assert entry.sha256 == original['normal_model_zoo_encoder_entry']['sha256']
assert digest(entry.path) == entry.sha256
assert entry.size_bytes == 88240510
output = scratch / 'scorecard-api-repair-full-cohort-r1.json'
assert not output.exists()
record = {'normal_repaired_full_17074_cell_scorecard_two_threads_passed': True, 'full_unchanged_original_GPU_features_keys_and_human_labels': True, 'every_original_dense_one_thread_metric_matches_to_1e_12': True, 'no_top_k_mAP_approximation_or_label_cohort_reduction': True, 'actual_CPU_thread_settings': {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}, 'normal_model_zoo_encoder_entry': asdict(entry), 'elapsed_seconds': elapsed, 'peak_process_RSS_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'current_source_sha256': {path: digest(path) for path in ('spacr/embeddings.py', 'spacr/qt/screens/embeddings.py')}, 'prior_normal_full_cohort_acceptance_sha256': digest(baseline / 'acceptance.json'), 'original_GPU_feature_acceptance_sha256': digest(encoded / 'acceptance.json'), 'verification_script_sha256': digest(__file__), 'no_new_GPU_inference_pretrained_weight_or_environment_change': True, 'bare_numpy_two_thread_large_matrix_library_failure_not_claimed_fixed': True}
output.write_text(json.dumps(record, indent=2) + '\n')
print('PASS repaired normal full 17074-cell scorecard on two threads; all original dense baseline metrics unchanged to 1e-12; seconds', elapsed, 'peak KiB', record['peak_process_RSS_KiB'], flush=True)
