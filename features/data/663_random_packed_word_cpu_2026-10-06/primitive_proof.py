"""Check every rank, duplicates, clipping, owned alpha and failure recovery."""
import gc
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, sys.argv[1])
importlib.import_module('spacr.qt.widgets')
app = QApplication([])
modules = {}
for label in ['before', 'after']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._rank_' + label,
                                                ROOT / (label + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._warm_packed_scatter()
    assert module._COLORED_SCATTER is not None
    modules[label] = module

records = []
rng = np.random.default_rng(1189)
x = rng.integers(-1, 18, size=(12, 8192)).astype(np.float32)
y = rng.integers(-1, 11, size=x.shape).astype(np.float32)
light = np.broadcast_to(np.arange(8192, dtype=np.float32) % 256 / 255, x.shape).copy()
light[:, :4] = [-1, 0, 1, 2]
for background in ['#101418', '#f6f7f9']:
    for spread in [False, True]:
        engines = {label: module.make_engine('data_art_impulse_lens', 'random',
                                             background, seed=42)
                   for label, module in modules.items()}
        frames = {}
        for label, engine in engines.items():
            frames[label] = engine._point_material(17, 10, x, y, light, spread)
            pixels = np.frombuffer(frames[label].bits(), np.uint32)
            assert np.all(pixels >> 24 == 255)
        assert frames['before'].bits().tobytes() == frames['after'].bits().tobytes()
        old = engines['before']._material_cache[('random_grain_palette', engines['before'].dark)]
        new = engines['after']._material_cache[('random_grain_palette', engines['after'].dark)]
        for old_table, new_table in zip(old, new):
            packed = (((old_table >> 32) << 24) | (old_table & 0xffffff)).astype(np.uint32)
            assert np.array_equal(packed, new_table)
            assert np.array_equal(np.argsort(old_table, kind='stable'),
                                  np.argsort(new_table, kind='stable'))
        original = frames['after'].bits().tobytes()
        calls = []

        def fail(output, *_arguments):
            calls.append(True)
            output[:100] = 0x1355ff77
            raise RuntimeError('injected partially packed working frame')

        after = modules['after']
        kernel = after._COLORED_SCATTER
        after._COLORED_SCATTER = fail
        recovered = engines['after']._point_material(17, 10, x, y, light, spread)
        assert recovered.bits().tobytes() == original
        assert len(calls) == 1 and after._COLORED_SCATTER_FAILED
        repeated = engines['after']._point_material(17, 10, x, y, light, spread)
        assert repeated.bits().tobytes() == original and len(calls) == 1
        after._COLORED_SCATTER = kernel
        after._COLORED_SCATTER_FAILED = False
        records.append({'background': background, 'spread': spread,
                        'all_256_ranks': True, 'clipped_duplicate_points': int(x.size),
                        'complete_alpha_255': True, 'partial_failure_restored': True,
                        'packed_table_order_exact': True,
                        'pixel_sha256': hashlib.sha256(original).hexdigest()})
        del engines, frames, recovered, repeated
        gc.collect()
receipt = {'source_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                            for name, module in modules.items()}, 'records': records}
(ROOT / 'primitive_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
