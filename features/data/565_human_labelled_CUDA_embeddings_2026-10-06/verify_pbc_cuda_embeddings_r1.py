from pathlib import Path
from collections import Counter
import hashlib
import json

import numpy as np
from PIL import Image

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
encoded = root / 'cuda-embeddings-r2'
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((root / 'plan-518-r2.json').read_text())
freeze = json.loads((root / 'GPU-freeze-518-r2.json').read_text())
weights = json.loads((scratch / '565-retrieval-encoder-preparation.json').read_text())
acceptance = json.loads((encoded / 'acceptance.json').read_text())
assert acceptance['real_GPU_embeddings_accepted']
assert acceptance['source_plan_sha256'] == digest(root / 'plan-518-r2.json')
assert acceptance['source_sha256'] == plan['source_sha256']
assert acceptance['frozen_model_state_sha256'] == weights['normal_timm_pretrained_state_sha256']
assert acceptance['encoder_preparation_sha256'] == digest(scratch / '565-retrieval-encoder-preparation.json')
for path, sha in freeze['files_sha256'].items():
    assert digest(path) == sha
for path, sha in plan['source_sha256'].items():
    assert digest(path) == sha
for path, record in weights['files'].items():
    assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
for path, record in acceptance['artifacts'].items():
    assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
vectors = np.load(encoded / 'embeddings.npy', allow_pickle=False)
with np.load(encoded / 'keys-and-labels.npz', allow_pickle=False) as data:
    keys, labels, columns = data['keys'], data['labels'], data['columns']
assert vectors.shape == (17074, 384) and vectors.dtype == np.float32
assert np.isfinite(vectors).all() and (np.linalg.norm(vectors, axis=1) > 0).all()
assert len(set(keys)) == len(keys) == 17074
assert len(columns) == len(set(columns)) == 384
assert keys.tolist() == [row['key'] for row in plan['retrieval_images']]
assert labels.tolist() == [row['label'] for row in plan['retrieval_images']]
batches = acceptance['batches']
forward = acceptance['all_normal_forward_devices']
assert len(batches) == len(forward) == (17074 + 31) // 32
prepared = []
for ordinal, (batch, observed) in enumerate(zip(batches, forward)):
    start = ordinal * 32
    count = min(32, 17074 - start)
    assert batch['start'] == start and batch['count'] == count
    assert observed['shape'] == [count, 3, 518, 518]
    assert observed['model_device'].startswith('cuda') and observed['input_device'].startswith('cuda')
    assert batch['seconds'] > 0
    assert len(batch['derived_inputs']) == count
    prepared.extend(batch['derived_inputs'])
assert len(prepared) == 17074
original_count = 0
for row in plan['all_original_images']:
    assert digest(row['path']) == row['sha256']
    original_count += 1
assert original_count == 17092
for expected, row in zip(prepared, plan['retrieval_images']):
    assert expected['key'] == row['key']
    with Image.open(row['path']) as image:
        pixels = np.array(image)
        assert list(pixels.shape) == row['shape']
        assert hashlib.sha256(pixels.tobytes()).hexdigest() == row['RGB_pixels_sha256']
        resized = np.asarray(image.resize((518, 518), Image.Resampling.BILINEAR))
    assert hashlib.sha256(resized.tobytes()).hexdigest() == expected['whole_RGB_resized_pixels_sha256']
profiles = []
for batch in (batches[0], batches[-1]):
    profile_path = encoded / f"batch-{batch['start']:05d}-profile.json"
    profile = json.loads(profile_path.read_text())
    events = profile['traceEvents']
    kernels = [event for event in events if event.get('cat') == 'kernel']
    assert kernels and batch['cuda_kernel_events'] > 0
    profiles.append({'path': str(profile_path), 'sha256': digest(profile_path), 'bytes': profile_path.stat().st_size, 'actual_CUDA_kernel_trace_events': len(kernels), 'reported_profiler_CUDA_events': batch['cuda_kernel_events']})
record = {'independent_originals_rows_labels_full_preprocessing_and_CUDA_evidence_verified': True, 'all_original_images': original_count, 'unique_unambiguous_human_labelled_cells': len(keys), 'label_counts': dict(sorted(Counter(labels.tolist()).items())), 'dimensions': 384, 'normal_forward_batches': len(batches), 'all_forward_inputs_and_model_parameters_CUDA': True, 'all_independently_recomputed_whole_RGB_preprocessing_hashes_exact': True, 'all_source_pretrained_model_plan_and_array_hashes_exact': True, 'full_first_and_last_actual_CUDA_profiles': profiles, 'acceptance_sha256': digest(encoded / 'acceptance.json'), 'verifier_sha256': digest(__file__), 'FAISS_class_agreement_and_million_scale_acceptance_remain_separate': True}
output = root / 'independent-CUDA-embedding-verification-r1.json'
assert not output.exists()
output.write_text(json.dumps(record, indent=2) + '\n')
print('PASS independent exact 17092 originals, 17074 rows/labels/preprocessing hashes, 384 finite features, full CUDA first/last traces and every real model/input forward device; FAISS remains separate.', flush=True)
