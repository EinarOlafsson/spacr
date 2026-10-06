"""A one-percent physical radius can target one actual seeded paper centre."""
import hashlib
import importlib.util
import os
import json
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

sys.path.insert(0, os.environ['SPACR_REPRO_REPO'])
import spacr.qt.widgets
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._paper_proof', Path(__file__).with_name('after.py'))
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)

app = QApplication([])
width, height = 3840, 2160
engine = ambient.make_engine('data_art_tissue_facets', 'spacr', '#101418',
                             seed=42, resolution=2)
engine.set_max_pixels(width * height)
engine.set_gravity_radius(.01)
image = engine.shade(width, height)
resting = np.frombuffer(image.bits(), np.uint8).reshape(height, width, 4).copy()
cells = next(value for key, value in engine._material_cache.items()
             if key[0] == 'tissue_facets')
chosen = min(cells, key=lambda cell: (cell[0] - width * .5) ** 2
             + (cell[1] - height * .5) ** 2)
x, y = chosen[:2]
reach = min(width, height) * .01
hits = [(cx, cy) for cx, cy, *_ in cells if (cx-x)**2 + (cy-y)**2 < reach**2]
assert hits == [(x, y)]
engine.set_pointer((x/width, y/height))
image = engine.shade(width, height)
lifted = np.frombuffer(image.bits(), np.uint8).reshape(height, width, 4).copy()
changed = int(np.count_nonzero(np.any(lifted != resting, axis=2)))
assert changed > 0
engine.set_pointer(None)
image = engine.shade(width, height)
assert np.array_equal(np.frombuffer(image.bits(), np.uint8).reshape(height, width, 4), resting)
receipt = {'source_sha256': hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest(),
           'native_size': [width, height], 'seed': 42, 'radius': .01,
           'physical_radius_pixels': reach, 'pointer': [x/width, y/height],
           'affected_primitive_centres': len(hits), 'changed_pixels': changed,
           'restored_exactly_after_clear': True,
           'scope': 'Primitive-centre radius intentionally may hit zero centres between tiles; no expanded radius or minimum was introduced.'}
Path(__file__).with_name('paper_center_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt, indent=2))
