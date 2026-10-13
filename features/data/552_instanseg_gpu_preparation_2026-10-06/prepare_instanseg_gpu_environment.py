from pathlib import Path
import json
import os
import sys

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(Path.cwd()))
from spacr import _segmentation_backends as backend

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/552-instanseg-gpu-r1')
root.mkdir(exist_ok=True)
index = backend._torch_index_url()
assert index == 'https://download.pytorch.org/whl/cu128'
state = backend._install_backend('instanseg', root=root / 'backends',
    progress=lambda step, steps, message: print(f'{step}/{steps} {message}', flush=True))
assert state.ready and not state.in_process
(root / 'installation.json').write_text(json.dumps({
    'normal_isolated_installation_complete': True, 'torch_index': index,
    'backend': 'instanseg', 'environment': state.env, 'record': state.record,
    'cuda_inference_acceptance': False,
    'scope': 'Normal installer with CUDA hidden; GPU inference must run separately through gpu_turn.sh.'
}, indent=2) + '\n')
print('PASS: normal isolated InstanSeg environment prepared; real GPU operation remains pending.', flush=True)
