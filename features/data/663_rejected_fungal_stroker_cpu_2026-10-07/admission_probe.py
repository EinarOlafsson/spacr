import collections
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

import spacr.qt.widgets

scratch = Path(__file__).resolve().parent
frozen = Path('/mnt/wd4tb/scratch/fungal-stroker-parity-20261007')
app = QApplication([])
rows = []

class CountedDict(dict):
    def __init__(self):
        super().__init__()
        self.counts = collections.Counter()

    def get(self, key, default=None):
        self.counts['lookup'] += 1
        self.counts['key_hit' if key in self else 'key_miss'] += 1
        return super().get(key, default)

    def __setitem__(self, key, value):
        self.counts['replace' if key in self else 'insert'] += 1
        super().__setitem__(key, value)

    def __delitem__(self, key):
        self.counts['delete'] += 1
        super().__delitem__(key)

for source, palette in [('before', 'spacr'), ('before', 'random'), ('candidate', 'spacr')]:
    path = frozen / (source + '.py')
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._admission_' + source + palette, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    engine = module.make_engine('data_art_fungal_growth', palette, '#101418', seed=42,
                                resolution=2, density=3, size=2.5, blur=0)
    engine.set_max_pixels(3840 * 2160)
    engine._fungal_observed = CountedDict()
    if source == 'candidate':
        engine._fungal_strokes = CountedDict()
    counts = collections.Counter()
    samples = []
    original_reuse = engine._reuse_fungal_raster
    original_warm = engine._warm_fungal_raster
    original_observe = engine._observe_fungal_path
    original_paths = engine._fungal_paths

    def reuse(key, path, color, target):
        entry = engine._fungal_rasters.get(key)
        reason = 'key_miss' if entry is None else 'path_mismatch' if entry[0] != path else 'headroom_refusal'
        result = original_reuse(key, path, color, target)
        counts['reuse_hit' if result else reason] += 1
        return result

    def warm(key, path, color, stroke, width, height):
        previous = set(engine._fungal_rasters)
        result = original_warm(key, path, color, stroke, width, height)
        counts['warm_calls'] += 1
        counts['raster_evictions'] += len(previous - set(engine._fungal_rasters))
        return result

    def observe(key, path):
        previous = engine._fungal_observed.get(key)
        counts['observed_unchanged' if previous is not None and previous[0] == path else 'observed_changed_or_missing'] += 1
        result = original_observe(key, path)
        counts['observation_ready' if result else 'observation_pending'] += 1
        return result

    def paths(width, height):
        result = original_paths(width, height)
        counts['path_groups'] += len(result[0])
        counts['mature_groups'] += sum(result[1].values())
        return result

    engine._reuse_fungal_raster = reuse
    engine._warm_fungal_raster = warm
    engine._observe_fungal_path = observe
    engine._fungal_paths = paths
    for index in range(72):
        engine.set_time(95 + index / 24)
        image = engine.shade(3840, 2160)
        if index in (0, 2, 7, 23, 47, 71):
            samples.append({'frame': index, 'clock': 95 + index / 24, 'counts': dict(counts),
                            'raster_entries': len(engine._fungal_rasters),
                            'raster_array_bytes': sum(v[1].nbytes + v[2].nbytes for v in engine._fungal_rasters.values()),
                            'observed_entries': len(engine._fungal_observed),
                            'stroke_entries': len(getattr(engine, '_fungal_strokes', {}))})
        del image
    row = {'source': source, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'palette': palette, 'viewport': [3840, 2160], 'resolution': 2, 'density': 3,
           'size': 2.5, 'seed': 42, 'frames': 72, 'counts': dict(counts),
           'observed_operations': dict(engine._fungal_observed.counts),
           'stroke_operations': dict(engine._fungal_strokes.counts) if source == 'candidate' else None,
           'samples': samples}
    rows.append(row)
    print(json.dumps(row), flush=True)
    del engine

receipt = {'scope': 'Scratch-only instrumented admission counters, not FPS or a parity rerun.',
           'limits': {'cache_entries': 64, 'sparse_array_bytes': 8 * 1024 ** 2},
           'rows': rows}
(scratch / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
