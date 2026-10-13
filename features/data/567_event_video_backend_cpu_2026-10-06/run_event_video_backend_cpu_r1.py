from pathlib import Path
import hashlib
import json
import os
import time

import numpy as np
from spacr import _segmentation_backends as backend

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
output = scratch / '567-event-video-backend-cpu-r1'
output.mkdir(exist_ok=False)
root = output / 'backends'
prepared = scratch / '567-videomae-preparation-r1'
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
started = time.time()
state = backend._install_backend('videomae', root=root,
    torch_index='https://download.pytorch.org/whl/cpu',
    progress=lambda step, count, text: print(f'{step}/{count}: {text}', flush=True))
assert state.ready and not state.in_process
print('PASS normal isolated optional-backend installation', state.env, flush=True)
inputs = np.load(prepared / 'synthetic-16-frame-RGB-uint8.npy', allow_pickle=False)
inputs = inputs.transpose(0, 3, 1, 2)[None].astype(np.float32) / 255
np.save(output / 'input-normalized-clip.npy', inputs, allow_pickle=False)
try:
    features, provenance = backend._event_video_features(inputs, prepared / 'model',
        [0, 1, 2], device='cpu', root=root)
    repeated, second = backend._event_video_features(inputs, prepared / 'model',
        [0, 1, 2], device='cpu', root=root)
    assert provenance == second
    np.testing.assert_array_equal(features, repeated)
    np.save(output / 'actual-worker-features.npy', features, allow_pickle=False)
    reference = np.load(prepared / 'synthetic-CPU-pooled-features.npy', allow_pickle=False)
    difference = float(np.abs(features - reference).max())
    print('Actual normalized-float versus original uint8 preparation max difference:', difference, flush=True)
    np.testing.assert_allclose(features, reference, rtol=0, atol=2e-5)
    report = {'item': 567, 'passed': True, 'CPU_only': True,
        'normal_optional_environment_install': True, 'actual_parent_worker_protocol': True,
        'input_scope': 'Declared synthetic RGB clip; no microscopy or event-accuracy claim',
        'feature_shape': list(features.shape), 'repeat_bit_exact': True,
        'uint8_preparation_max_absolute_difference': difference,
        'provenance': provenance, 'environment': state.env,
        'seconds': time.time() - started,
        'source_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in (Path('spacr/_segmentation_backends.py'), Path('spacr/timelapse.py'))}}
    (output / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS normal installed VideoMAE parent/worker CPU extraction and exact repeat', flush=True)
finally:
    backend._shutdown_workers()
