from pathlib import Path
import gc
import hashlib
import importlib.metadata
import json
import os
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
import torch
from spacr.embeddings import EmbeddingSpec, _backbone_encoder, _FOUNDATION_MODELS

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
packages = {name: importlib.metadata.version(name) for name in
            ('torch', 'torchvision', 'transformers', 'tokenizers', 'timm', 'huggingface_hub', 'numpy', 'einops')}
for backbone in ('resnet18', 'resnet50', 'vit_small_patch14_dinov2.lvd142m', 'openphenom', 'chada_vit', 'subcell'):
    target = root / (backbone + '-CPU-weight-preparation.json')
    assert not target.exists()
    torch.manual_seed(0)
    spec = EmbeddingSpec(backbone=backbone, device='cpu', batch_size=32)
    run = _backbone_encoder(spec)
    candidates = [cell.cell_contents for cell in run.__closure__ if isinstance(cell.cell_contents, torch.nn.Module)]
    assert len(candidates) == 1
    model = candidates[0]
    assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
    state_digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        state_digest.update(name.encode())
        state_digest.update(str(value.dtype).encode())
        state_digest.update(str(tuple(value.shape)).encode())
        state_digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    files = {}
    if backbone == 'subcell':
        cache_path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(_FOUNDATION_MODELS[backbone]['url']).name
        files[str(cache_path)] = {'sha256': digest(cache_path), 'bytes': cache_path.stat().st_size}
        source = {'url': _FOUNDATION_MODELS[backbone]['url']}
        input_size = _FOUNDATION_MODELS[backbone]['size']
    else:
        source = (_FOUNDATION_MODELS[backbone] if backbone in _FOUNDATION_MODELS else
                  {'repo': model.pretrained_cfg['hf_hub_id']})
        cache = Path.home() / '.cache/huggingface/hub' / ('models--' + source['repo'].replace('/', '--'))
        revision = source.get('revision') or (cache / 'refs/main').read_text().strip()
        source = {**source, 'revision': revision}
        for path in sorted((cache / 'snapshots' / revision).iterdir()):
            if path.is_file():
                files[str(path)] = {'sha256': digest(path), 'bytes': path.stat().st_size}
        assert any(path.endswith(('.safetensors', '.bin')) for path in files)
        input_size = (_FOUNDATION_MODELS[backbone]['size'] if backbone in _FOUNDATION_MODELS else
                      model.pretrained_cfg['input_size'][-1])
    receipt = {'backbone': backbone, 'normal_CPU_factory_loaded': True,
               'no_GPU_context_or_forward_inference': True,
               'complete_ordered_model_state_sha256': state_digest.hexdigest(),
               'actual_parameters': sum(p.numel() for p in model.parameters()),
               'normal_input_size': input_size, 'normal_input_channels': getattr(run, 'in_channels', 3),
               'source': source, 'actual_cached_files': files, 'packages': packages,
               'application_source_sha256': digest(Path('spacr/embeddings.py')),
               'preparation_script_sha256': digest(Path(__file__))}
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print('PASS normal CPU factory', backbone, 'parameters', receipt['actual_parameters'],
          'input_size', input_size, 'state_sha256', state_digest.hexdigest(), flush=True)
    del model, run, candidates
    gc.collect()
