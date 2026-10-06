from pathlib import Path
import contextlib
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
import os
import sys
import time

sys.meta_path = [finder for finder in sys.meta_path if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
import spacr
assert Path(spacr.__file__).resolve().parent == Path('spacr').resolve()
import numpy as np
from PIL import Image
import torch
from spacr.embeddings import EmbeddingSpec, _backbone_encoder, embed_array

root = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/560-human-fluorescence-primary-r1')
plan_path = root / 'frozen-full-fluorescence-foundation-plan-r1.json'
plan_sha = 'cf7321646d184fb475c243349c87066cb2ec16bb244242d7700288964e409746'
def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            sha.update(block)
    return sha.hexdigest()

class TerminalLog:
    def __init__(self, stream, output):
        self.stream, self.output = stream, output
    def write(self, value):
        self.stream.write(value)
        self.output.write(value)
        self.output.flush()
        return len(value)
    def flush(self):
        self.stream.flush()
        self.output.flush()

log = (root / 'full-fluorescence-foundation-CUDA-r1.log').open('x')
sys.stdout = TerminalLog(sys.stdout, log)
sys.stderr = TerminalLog(sys.stderr, log)
assert digest(plan_path) == plan_sha
plan = json.loads(plan_path.read_text())
assert digest('spacr/embeddings.py') == plan['application_source_sha256']
assert os.environ.get('HF_HUB_OFFLINE') == '1'
assert torch.cuda.is_available()
torch.set_num_threads(2)
device = torch.cuda.get_device_properties(0)
print('ACTUAL CUDA', device.name, 'memory', device.total_memory, 'Torch', torch.__version__, 'plan', plan_sha, flush=True)
output = root / 'full-fluorescence-foundation-CUDA-r1'
output.mkdir(exist_ok=False)
arrays = {}
for name, cohort in plan['cohorts'].items():
    assert digest(cohort['inventory_path']) == cohort['inventory_sha256']
    assert digest(cohort['pixel_path']) == cohort['pixel_sha256']
    arrays[name] = np.load(cohort['pixel_path'], mmap_mode='r')
records = []
for backbone, preparation in plan['models'].items():
    assert digest(preparation['preparation_path']) == preparation['preparation_sha256']
    prep = preparation['receipt']
    for name, version in prep['packages'].items():
        assert importlib.metadata.version(name) == version, name
    for path, metadata in prep['actual_cached_files'].items():
        assert digest(path) == metadata['sha256'] and Path(path).stat().st_size == metadata['bytes'], path
    torch.manual_seed(plan['random_seed_before_each_normal_factory'])
    normal = _backbone_encoder(EmbeddingSpec(backbone=backbone, device='cuda', batch_size=16))
    candidates = [cell.cell_contents for cell in normal.__closure__ if isinstance(cell.cell_contents, torch.nn.Module)]
    assert len(candidates) == 1
    model = candidates[0]
    assert all(parameter.device.type == 'cuda' for parameter in model.parameters())
    state_sha = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        state_sha.update(name.encode())
        state_sha.update(str(value.dtype).encode())
        state_sha.update(str(tuple(value.shape)).encode())
        state_sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    assert state_sha.hexdigest() == prep['complete_ordered_model_state_sha256'], backbone
    print('PASS frozen real CUDA factory and all cached weights', backbone, state_sha.hexdigest(), flush=True)
    for comparison in [row for row in plan['comparisons'] if row['backbone'] == backbone]:
        cohort_name = comparison['cohort']
        cohort = plan['cohorts'][cohort_name]
        rows = cohort['rows']
        spec = EmbeddingSpec(backbone=backbone, channel_policy=comparison['channel_policy'], channels=tuple(comparison['channels']),
                             channel_scale=tuple(comparison['channel_scale']), device='cuda', batch_size=16)
        tensor_input_shapes = []
        def recorded(stack):
            assert stack.dtype == np.float32 and np.isfinite(stack).all()
            assert float(stack.min()) >= 0 and float(stack.max()) <= 1
            if backbone == 'subcell' and cohort_name == 'HEp2-expert-DAPI':
                assert stack.shape[-1] == 2 and np.count_nonzero(stack[..., 1]) == 0
            tensor_input_shapes.append(list(stack.shape))
            return normal(stack)
        recorded.in_channels = getattr(normal, 'in_channels', 3)
        features = None
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        embed_seconds = 0.0
        profile_receipts = []
        for start in range(0, len(rows), 16):
            chosen = rows[start:start + 16]
            raw = arrays[cohort_name][[row['pixel_row'] for row in chosen]]
            if cohort_name == 'HEp2-expert-DAPI':
                raw = raw[..., None]
            else:
                raw = raw[..., :3]
            resize = comparison['external_resize']
            if resize:
                size = resize['size']
                if cohort_name == 'HEp2-expert-DAPI':
                    raw = np.stack([np.asarray(Image.fromarray(image[..., 0]).resize((size, size), Image.Resampling.BILINEAR)) for image in raw])[..., None]
                else:
                    raw = np.stack([np.stack([np.asarray(Image.fromarray(image[..., channel].astype(np.float32)).resize((size, size), Image.Resampling.BILINEAR)) for channel in range(3)], axis=-1) for image in raw])
            boundary = 'first' if start == 0 else ('last' if start + len(chosen) == len(rows) else None)
            profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True) if boundary else contextlib.nullcontext()
            torch.cuda.synchronize()
            before = time.perf_counter()
            with profile:
                result = embed_array(raw, spec, encoder=recorded)
                torch.cuda.synchronize()
            embed_seconds += time.perf_counter() - before
            assert result.spec == spec and result.values.shape[0] == len(chosen)
            assert np.isfinite(result.values).all() and result.values.dtype == np.float32
            if features is None:
                feature_path = output / f'{backbone}-{cohort_name}-features.npy'
                features = np.lib.format.open_memmap(feature_path, mode='w+', dtype=np.float32, shape=(len(rows), result.values.shape[1]))
                columns = list(result.columns)
            assert result.values.shape[1] == features.shape[1]
            features[start:start + len(chosen)] = result.values
            if boundary:
                trace_path = output / f'{backbone}-{cohort_name}-{boundary}-CUDA-trace.json'
                profile.export_chrome_trace(str(trace_path))
                trace = json.loads(trace_path.read_text())
                kernels = sum(event.get('cat') == 'kernel' for event in trace['traceEvents'])
                assert kernels > 0, (backbone, cohort_name, boundary)
                profile_receipts.append({'boundary': boundary, 'rows': len(chosen), 'CUDA_kernel_events': kernels, 'trace_path': str(trace_path), 'sha256': digest(trace_path)})
                print('PASS actual CUDA profile', backbone, cohort_name, boundary, 'kernels', kernels, flush=True)
                del trace
            if (start // 16) % 64 == 0:
                print('PROGRESS', backbone, cohort_name, 'completed', start + len(chosen), 'of', len(rows), flush=True)
        features.flush()
        receipt = {'backbone': backbone, 'cohort': cohort_name, 'rows': len(rows), 'dimensions': features.shape[1],
                   'normal_spec': dataclasses.asdict(spec), 'columns': columns, 'ordered_rows': rows,
                   'input_stack_shapes': dict((str(shape), sum(tuple(value) == shape for value in tensor_input_shapes)) for shape in {tuple(shape) for shape in tensor_input_shapes}),
                   'all_actual_model_parameters_on_CUDA': True, 'complete_ordered_model_state_sha256': state_sha.hexdigest(),
                   'full_features': {'path': str(feature_path), 'bytes': feature_path.stat().st_size, 'sha256': digest(feature_path)},
                   'wall_seconds_including_adapters_profiles_and_output': time.perf_counter() - started,
                   'normal_embed_seconds_including_boundary_profiles': embed_seconds,
                   'peak_CUDA_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_CUDA_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'first_last_actual_CUDA_profiles': profile_receipts, 'frozen_plan_sha256': plan_sha,
                   'application_source_sha256': plan['application_source_sha256'], 'script_sha256': digest(__file__)}
        (output / f'{backbone}-{cohort_name}-execution.json').write_text(json.dumps(receipt, indent=2) + '\n')
        records.append(receipt)
        print('PASS all actual CUDA embeddings', backbone, cohort_name, len(rows), 'dims', features.shape[1], 'seconds', receipt['normal_embed_seconds_including_boundary_profiles'], flush=True)
        del features
    del candidates, model, normal
    gc.collect()
    torch.cuda.empty_cache()
(output / 'complete-execution.json').write_text(json.dumps({'passed': True, 'plan_sha256': plan_sha, 'comparisons': records, 'actual_CUDA_device': device.name}, indent=2) + '\n')
print('PASS ALL TWELVE FULL EXPERT-LABELLED CUDA COMPARISONS', flush=True)
