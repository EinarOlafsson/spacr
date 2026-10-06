from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import sys
import time

import numpy as np
from PIL import Image
import torch

checkout = Path.cwd().resolve()
sys.meta_path[:] = [f for f in sys.meta_path if not getattr(f, '__module__', '').startswith('__editable__')]
sys.path.insert(0, str(checkout))
import spacr
from spacr.embeddings import EmbeddingSpec, _backbone_encoder, embed_array

assert Path(spacr.__file__).resolve().is_relative_to(checkout)
assert os.environ.get('SPACR_DEVICE') == 'cuda' and torch.cuda.is_available()
holder = dict(part.split('=', 1) for part in (Path.home() / '.spacr/gpu/holder').read_text().split())
ancestor = os.getpid()
while ancestor and ancestor != int(holder['pid']):
    ancestor = int(next(line.split(':', 1)[1] for line in Path(f'/proc/{ancestor}/status').read_text().splitlines() if line.startswith('PPid:')))
assert ancestor == int(holder['pid'])
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '565-pbc-retrieval-r4'
plan = json.loads((root / 'plan-518-r2.json').read_text())
freeze = json.loads((root / 'GPU-freeze-518-r2.json').read_text())
weights = json.loads((scratch / '565-retrieval-encoder-preparation.json').read_text())
digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
for path, sha in freeze['files_sha256'].items():
    assert digest(path) == sha
assert plan['prepared'] and len(plan['retrieval_images']) == 17074
for path, sha in plan['source_sha256'].items():
    assert digest(path) == sha
for path, record in weights['files'].items():
    assert digest(path) == record['sha256'] and Path(path).stat().st_size == record['bytes']
for row in plan['all_original_images']:
    assert digest(row['path']) == row['sha256']
target = root / 'cuda-embeddings-r2'
target.mkdir(exist_ok=False)
spec = EmbeddingSpec(backbone=plan['embedding_policy']['backbone'], channel_policy='project', channels=(0,1,2), channel_scale=(255.0,255.0,255.0), device='cuda', batch_size=32)
run = _backbone_encoder(spec)
closure = dict(zip(run.__code__.co_freevars, (c.cell_contents for c in run.__closure__)))
model = closure['model']
assert next(model.parameters()).device.type == 'cuda'

def state_hash():
    sha = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        sha.update(key.encode())
        sha.update(str(value.dtype).encode())
        sha.update(str(tuple(value.shape)).encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()

assert state_hash() == weights['normal_timm_pretrained_state_sha256']
observed = []
def observe(module, args):
    assert next(module.parameters()).device.type == 'cuda' and args[0].device.type == 'cuda'
    assert args[0].shape[1:] == (3,518,518)
    observed.append({'input_device':str(args[0].device),'model_device':str(next(module.parameters()).device),'shape':list(args[0].shape)})

hook = model.register_forward_pre_hook(observe)
rows = plan['retrieval_images']
values = None
columns = None
batches = []
started = time.monotonic()
for start in range(0, len(rows), 32):
    selected = rows[start:start + 32]
    crops = []
    prepared = []
    for row in selected:
        with Image.open(row['path']) as image:
            original = np.array(image)
            assert list(original.shape) == row['shape']
            assert hashlib.sha256(original.tobytes()).hexdigest() == row['RGB_pixels_sha256']
            resized = np.array(image.resize((518,518), Image.Resampling.BILINEAR))
        crops.append(resized)
        prepared.append({'key':row['key'],'whole_RGB_resized_pixels_sha256':hashlib.sha256(resized.tobytes()).hexdigest()})
    crops = np.stack(crops)
    call_start = time.monotonic()
    profiled = start == 0 or start + 32 >= len(rows)
    if profiled:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profile:
            result = embed_array(crops, spec, encoder=run)
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(target / f'batch-{start:05d}-profile.json'))
        kernels = sum(1 for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA)
        assert kernels > 0
    else:
        result = embed_array(crops, spec, encoder=run)
        torch.cuda.synchronize()
        kernels = None
    assert result.values.shape == (len(selected),384) and np.isfinite(result.values).all()
    assert result.spec.fingerprint() == spec.fingerprint()
    if values is None:
        values = np.lib.format.open_memmap(target / 'embeddings.npy', mode='w+',dtype=np.float32,shape=(len(rows),384))
        columns = result.columns
    assert columns == result.columns
    values[start:start + len(selected)] = result.values
    batches.append({'start':start,'count':len(selected),'seconds':time.monotonic()-call_start,'cuda_kernel_events':kernels,'derived_inputs':prepared})
    print('actual normal pretrained CUDA embeddings', start+len(selected),'/',len(rows),flush=True)
values.flush()
hook.remove()
assert len(observed) == len(batches)
assert state_hash() == weights['normal_timm_pretrained_state_sha256']
for path, sha in freeze['files_sha256'].items():
    assert digest(path) == sha
for path, record in weights['files'].items():
    assert digest(path) == record['sha256']
np.savez_compressed(target / 'keys-and-labels.npz', keys=np.asarray([r['key'] for r in rows]),labels=np.asarray([r['label'] for r in rows]),columns=np.asarray(columns))
acceptance = {'real_GPU_embeddings_accepted':True,'rows':len(rows),'dimensions':384,'all_normal_forward_devices':observed,'batches':batches,'normal_pretrained_encoder_cached_once_explicitly_passed_to_normal_embed_array':True,'no_fake_or_test_encoder':True,'source_plan_sha256':digest(root / 'plan-518-r2.json'),'frozen_model_state_sha256':weights['normal_timm_pretrained_state_sha256'],'encoder_preparation_sha256':digest(scratch/'565-retrieval-encoder-preparation.json'),'source_sha256':plan['source_sha256'],'script_sha256':digest(__file__),'embedding_spec_fingerprint':spec.fingerprint(),'elapsed_seconds':time.monotonic()-started,'torch_cuda_version':torch.version.cuda,'device_name':torch.cuda.get_device_name(),'packages':{p:importlib.metadata.version(p) for p in ('torch','torchvision','timm','numpy','Pillow')},'artifacts':{str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for p in (target/'embeddings.npy',target/'keys-and-labels.npz')},'FAISS_GPU_timing_and_independent_human_label_retrieval_still_pending':True}
(target / 'acceptance.json').write_text(json.dumps(acceptance,indent=2)+'\n')
print('PASS: all 17074 unique unambiguous clinician-labelled original cells encoded by actual normal spaCR pretrained DINOv2 on CUDA; FAISS and class agreement remain separate.',flush=True)
