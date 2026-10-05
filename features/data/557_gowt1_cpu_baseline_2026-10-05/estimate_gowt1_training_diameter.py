import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from cellpose import models

assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
torch.set_num_threads(2)
scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
root = scratch / '557-gowt1'
environment = root / 'backends/cellpose3'
assert models.MODEL_DIR.resolve() == (environment / 'models').resolve()
target = root / 'training-diameter-r1.json'
assert not target.exists()
plan_path = root / 'prepared-r1/plan.json'
plan = json.loads(plan_path.read_text())
assert len(plan['training']) == 20
assert all('/02/' in record['image'] for record in plan['training'])
model = models.Cellpose(gpu=False, model_type='nuclei', device=torch.device('cpu'))
assert str(model.cp.device) == 'cpu'

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

package = Path(models.__file__).parent
paths = [Path(__file__), plan_path, environment / 'spacr-backend.json',
         *package.rglob('*.py'), *(environment / 'models').rglob('*')]
paths += [Path(row[key]) for row in plan['training'] for key in ('image', 'exact_pixel_export')]
frozen = {str(path.resolve()): digest(path) for path in paths if path.is_file()}
records = []
for row in plan['training']:
    path = Path(row['exact_pixel_export'])
    assert digest(path) == row['export_sha256'] and digest(row['image']) == row['image_sha256']
    image = np.load(path, allow_pickle=False)
    assert image.shape == (1024, 1024) and image.dtype == np.uint8
    started = time.perf_counter()
    diameter, style_diameter = model.sz.eval(image, channels=[0, 0], channel_axis=None,
                                           normalize=True, augment=False, batch_size=8)
    assert np.isfinite(diameter) and float(diameter) > 0
    record = {'training_image': row['image'], 'image_sha256': row['image_sha256'],
              'learned_diameter_px': float(diameter), 'style_diameter_px': float(style_diameter),
              'seconds': time.perf_counter() - started}
    records.append(record)
    print(record, flush=True)
assert all(digest(path) == expected for path, expected in frozen.items())
receipt = {'schema': 1, 'accepted': True, 'device': 'cpu',
           'policy': 'Median of normal Cellpose3 nuclei size-model estimates on all 20 raw training-sequence frames, fixed for every evaluation arm before held-out predictions. No evaluation image, gold label or score is used for selection.',
           'model': 'cellpose3:nuclei', 'cellpose': '3.1.1.3',
           'diameter_px': float(np.median([record['learned_diameter_px'] for record in records])),
           'records': records, 'source_input_and_model_sha256': frozen,
           'original_inputs_and_models_unchanged': True}
target.write_text(json.dumps(receipt, indent=2) + '\n')
print('Fixed training-only median diameter:', receipt['diameter_px'], flush=True)
