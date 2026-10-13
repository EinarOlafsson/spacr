import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch

source = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005/spacr/_segmentation_backends.py')
spec = importlib.util.spec_from_file_location('spacr_profiled_backend', source)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
original = backend._worker_segment
root = Path(os.environ['SPACR_INSTANSEG_PROFILE_ROOT'])
root.mkdir(parents=True, exist_ok=False)
counter = 0

def profiled_segment(name, request, adapters):
    global counter
    requested = request.get('device')
    gpu = str(requested).startswith(('cuda', 'gpu'))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if gpu:
        assert torch.cuda.is_available()
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        result = original(name, request, adapters)
        if gpu:
            torch.cuda.synchronize()
    counter += 1
    events = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [event for event in events if not any(word in event.name.lower() for word in ('memcpy', 'memset'))]
    record = {'request': counter, 'requested_device': str(requested), 'actual_worker_reply_device': result['device'],
              'cuda_events': len(events), 'cuda_kernel_events': len(kernels),
              'cuda_kernel_names': sorted({event.name for event in kernels}),
              'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'model': request.get('model'), 'params': request.get('params'),
              'input_file_sha256': [hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in request['inputs']],
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'output_file_sha256': [hashlib.sha256(Path(row['mask']).read_bytes()).hexdigest() for row in result['outputs']]}
    (root / f'segment-{counter:02d}.json').write_text(json.dumps(record, indent=2) + '\n')
    return result

backend._worker_segment = profiled_segment
raise SystemExit(backend._worker_main())
