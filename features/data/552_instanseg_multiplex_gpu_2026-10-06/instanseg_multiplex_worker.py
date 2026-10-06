import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

source = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005/spacr/_segmentation_backends.py')
spec = importlib.util.spec_from_file_location('spacr_profiled_backend', source)
backend = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = backend
spec.loader.exec_module(backend)
original = backend._worker_segment
root = Path(os.environ['SPACR_MULTIPLEX_PROFILE_ROOT'])
root.mkdir(parents=True, exist_ok=False)
counter = 0

def profiled_segment(name, request, adapters):
    global counter
    counter += 1
    saved = []
    for index, path in enumerate(request['inputs'], 1):
        destination = root / f'request-{counter:02d}-input-{index:02d}.npy'
        shutil.copyfile(path, destination)
        saved.append({'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                      'shape': list(np.load(destination, allow_pickle=False).shape)})
    gpu = str(request.get('device')).startswith(('cuda', 'gpu'))
    activities = [torch.profiler.ProfilerActivity.CPU]
    if gpu:
        assert torch.cuda.is_available()
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities) as profile:
        result = original(name, request, adapters)
        if gpu:
            torch.cuda.synchronize()
    events = [event for event in profile.events() if event.device_type == torch.autograd.DeviceType.CUDA]
    kernels = [event for event in events if not any(word in event.name.lower() for word in ('memcpy', 'memset'))]
    outputs = []
    for index, row in enumerate(result['outputs'], 1):
        destination = root / f'request-{counter:02d}-mask-{index:02d}.npy'
        shutil.copyfile(row['mask'], destination)
        labels = np.load(destination, allow_pickle=False)
        outputs.append({'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
                        'objects': int(len(np.unique(labels[labels > 0])))})
    record = {'request': counter, 'requested_device': str(request.get('device')),
              'actual_worker_reply_device': result['device'], 'cuda_events': len(events),
              'cuda_kernel_events': len(kernels), 'cuda_kernel_names': sorted({event.name for event in kernels}),
              'torch_version': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'model': request.get('model'), 'params': request.get('params'), 'inputs': saved, 'outputs': outputs,
              'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
    (root / f'segment-{counter:02d}.json').write_text(json.dumps(record, indent=2) + '\n')
    return result

backend._worker_segment = profiled_segment
raise SystemExit(backend._worker_main())
