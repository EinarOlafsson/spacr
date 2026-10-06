from pathlib import Path
import hashlib
import importlib.metadata
import json
import os

import timm
import torch
from huggingface_hub import HfApi

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
name = 'vit_small_patch14_dinov2.lvd142m'
repository = 'timm/' + name
info = HfApi().model_info(repository)
model = timm.create_model(name, pretrained=True, num_classes=0)
model.eval()
assert next(model.parameters()).device.type == 'cpu'
sha = hashlib.sha256()
for key, value in sorted(model.state_dict().items()):
    sha.update(key.encode())
    sha.update(str(value.dtype).encode())
    sha.update(str(tuple(value.shape)).encode())
    sha.update(value.detach().cpu().contiguous().numpy().tobytes())
repo = Path.home() / '.cache/huggingface/hub' / ('models--timm--' + name)
files = {}
for path in sorted((repo / 'snapshots' / info.sha).glob('*')):
    if path.is_file():
        files[str(path)] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'bytes': path.stat().st_size}
assert any('safetensors' in path or path.endswith('.bin') for path in files)
record = {'CPU_weight_preparation_complete': True, 'no_GPU_embedding_or_retrieval_run': True,
          'model': name, 'model_repository': repository, 'current_primary_revision': info.sha,
          'normal_timm_pretrained_state_sha256': sha.hexdigest(), 'files': files,
          'packages': {p: importlib.metadata.version(p) for p in ('torch', 'torchvision', 'timm', 'huggingface_hub')},
          'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(scratch / '565-retrieval-encoder-preparation.json').write_text(json.dumps(record, indent=2) + '\n')
print('PASS: normal pretrained DINOv2 encoder weights acquired and exact complete CPU model state frozen; real GPU encoding remains pending.', flush=True)
