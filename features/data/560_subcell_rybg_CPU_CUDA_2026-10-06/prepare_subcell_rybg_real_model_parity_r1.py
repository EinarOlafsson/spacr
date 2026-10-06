from pathlib import Path
import ast
import dataclasses
import gc
import hashlib
import importlib.util
import json
import os
import resource
import sys
import numpy as np
import torch
import yaml
from spacr.embeddings import CHANNEL_PROJECT, EmbeddingSpec, _foundation_encoder, embed_array, encoder_entry
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
base = scratch / '560-subcell-rybg-preparation-r1'
out = base / 'CPU-mapping-parity-r1'
out.mkdir(exist_ok=False)
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and not torch.cuda.is_available()
assert Path(torch.hub.get_dir()) == base / 'torch/hub'
torch.set_num_threads(2)
def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()
def state_digest(model):
    sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        sha.update(name.encode())
        sha.update(str(value.dtype).encode())
        sha.update(str(tuple(value.shape)).encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()
acquisition = json.loads((base / 'acquisition.json').read_text())
for name, row in acquisition['files'].items():
    assert digest(base / name) == row['sha256']
source_sha = digest('spacr/embeddings.py')
assert source_sha == json.loads((scratch / 'subcell-rybg-documentation-inventory-r4.json').read_text())['source_sha256']
fixtures = {}
for height, width in ((16, 16), (48, 64), (64, 64), (512, 512)):
    yy, xx = np.indices((height, width), dtype=np.float32)
    planes = [((xx * (channel + 1) + yy * (channel + 3)) % 113) * (channel + 1) + 100 * channel for channel in range(5)]
    array = np.stack(planes, axis=-1)[None].astype(np.float32)
    array[0, 0, 0, 4] = 0
    array[0, -1, -1, 1] = 1000
    array[..., 3] = 9999
    fixtures[str(height) + 'x' + str(width)] = array
fixtures['constant-64x64'] = np.full((1, 64, 64, 5), 100, dtype=np.float32)
spec = EmbeddingSpec(backbone='subcell_rybg', channel_policy=CHANNEL_PROJECT, channels=(4, 0, 2, 1), normalize=False, batch_size=1, device='cpu')
normal = _foundation_encoder(spec)
models = [cell.cell_contents for cell in normal.__closure__ or () if isinstance(cell.cell_contents, torch.nn.Module)]
assert len(models) == 1
model = models[0]
assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
model_sha = state_digest(model)
results = {}
for key, array in fixtures.items():
    np.save(out / (key + '-original.npy'), array, allow_pickle=False)
    result = embed_array(array, spec, encoder=normal)
    assert result.spec == spec and result.values.shape == (1, 1536)
    assert np.isfinite(result.values).all()
    results[key] = result.values.copy()
    np.save(out / (key + '-spacr-features.npy'), result.values, allow_pickle=False)
entry = encoder_entry(spec)
assert entry.sha256 == acquisition['files']['torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth']['sha256']
assert entry.size_bytes == 349009018
assert entry.uri == acquisition['files']['torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth']['original_url']
assert entry.verified is False
assert entry.trained_on == 'UNKNOWN'
del models, model, normal
gc.collect()
tree = ast.parse((base / 'upstream_dataset.py').read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'min_max_norm_fn')
namespace = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(base / 'upstream_dataset.py'), 'exec'), namespace)
normalize = namespace['min_max_norm_fn']
module_spec = importlib.util.spec_from_file_location('subcell_original_author_CPU_reference', base / 'vit_model.py')
module = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = module
module_spec.loader.exec_module(module)
config = yaml.safe_load((base / 'model_config.yaml').read_text())['model_config']
reference_model = module.ViTPoolClassifier(config)
reference_model.load_model_dict(str(base / 'torch/hub/checkpoints/all_channels_ViT-ProtS-Pool.pth'), [])
reference_model.eval()
comparisons = {}
with torch.no_grad():
    for key, array in fixtures.items():
        selected = array[0, ..., list(spec.channels)].transpose(2, 0, 1)
        prepared = normalize(selected)
        assert prepared.shape == (4, array.shape[1], array.shape[2])
        tensor = torch.from_numpy(prepared[None].copy())
        expected = reference_model(tensor).pool_op.detach().cpu().numpy()
        np.save(out / (key + '-authors-features.npy'), expected, allow_pickle=False)
        delta = np.abs(results[key] - expected)
        comparison = {'original_shape': list(array.shape), 'model_input_shape': list(tensor.shape),
                      'features_shape': list(expected.shape), 'max_abs_difference': float(delta.max()),
                      'mean_abs_difference': float(delta.mean()), 'predeclared_max_abs_guard': 1e-5}
        comparisons[key] = comparison
        print('CPU original author parity', key, comparison, flush=True)
        assert expected.shape == (1, 1536) and np.isfinite(expected).all()
        assert float(delta.max()) < 1e-5
assert digest('spacr/embeddings.py') == source_sha
report = {'passed': True, 'application_source_sha256': source_sha, 'script_sha256': digest(__file__),
          'spec': dataclasses.asdict(spec), 'complete_ordered_model_state_sha256': model_sha,
          'checkpoint_sha256': entry.sha256, 'actual_normal_entry': dataclasses.asdict(entry),
          'comparisons': comparisons, 'peak_rss_KiB': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
          'no_GPU_run_or_biological_accuracy_or_human_labels_claim': True,
          'input_scope': 'Five deterministic synthetic normalization/mapping fixtures; original five planes retained, four explicitly permuted selected planes, an unselected extreme plane, minimum/rectangular/native dimensions and constant input.',
          'files': {str(path.name): {'bytes': path.stat().st_size, 'sha256': digest(path)} for path in sorted(out.glob('*.npy'))}}
(out / 'acceptance.json').write_text(json.dumps(report, indent=2) + '\n')
print('PASS real strict-loaded four-channel CPU model, public embed_array mapping and original author normalization/geometry parity; no biological or CUDA claim', flush=True)
