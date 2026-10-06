"""Alternate frozen native renderers against the same active field clocks."""
import gc
import hashlib
import importlib.util
import json
import math
import resource
import statistics
import sys
import time
from pathlib import Path

from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
importlib.import_module('spacr.qt.widgets')
app = QApplication([])
modules = {}
for label in ['before', 'after']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._balanced_' + label,
                                                ROOT / (label + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._warm_packed_scatter()
    assert module._COLORED_SCATTER is not None
    modules[label] = module


def summary(values):
    ordered = sorted(values)
    return {'samples': len(values), 'median_ms': statistics.median(values),
            'p95_ms': ordered[math.ceil(len(values) * .95) - 1], 'max_ms': max(values)}


records = []
for background in ['#101418', '#f6f7f9']:
    engines, measurements, grain_sizes = {}, {}, {}
    for label, module in modules.items():
        engine = module.make_engine('data_art_genetic_advection', 'random', background,
                                    seed=42, resolution=2)
        engine.set_gravity_radius(.25)
        engine.set_max_pixels(3840 * 2160)
        engine.set_time(8)
        engine.set_pointer((.5, .5))
        engines[label] = engine
        measurements[label] = {'shade': [], 'material': []}
        material = engine._colored_point_material

        def timed(*arguments, material=material, label=label):
            started = time.perf_counter()
            image = material(*arguments)
            measurements[label]['material'].append((time.perf_counter() - started) * 1000)
            grain_sizes[label] = list(arguments[2].shape)
            return image

        engine._colored_point_material = timed
        for _ in range(3):
            engine.shade(3840, 2160)
        measurements[label]['material'].clear()
    hashes = []
    for index in range(48):
        order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
        images = {}
        for label in order:
            engine = engines[label]
            engine.set_time(8 + index / 24)
            started = time.perf_counter()
            images[label] = engine.shade(3840, 2160)
            measurements[label]['shade'].append((time.perf_counter() - started) * 1000)
        old = hashlib.sha256(images['before'].constBits()).hexdigest()
        new = hashlib.sha256(images['after'].constBits()).hexdigest()
        assert old == new, (background, index)
        hashes.append(new)
        del images
    records.append({'background': background, 'native_size': [3840, 2160],
                    'gravity_radius': .25, 'pointer': [.5, .5], 'exact_pairs': len(hashes),
                    'grain_array_shapes': grain_sizes,
                    'measurements': {label: {stage: summary(values)
                                              for stage, values in stages.items()}
                                     for label, stages in measurements.items()},
                    'pixel_sha256': hashes})
    del engines, engine, material, timed
    gc.collect()
receipt = {'source_sha256': {label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                             for label, module in modules.items()},
           'scope': 'balanced same-process warmed CPU shade/material, not GUI cadence',
           'records': records, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / 'balanced_stages.json').write_text(json.dumps(receipt, indent=2) + '\n')
for record in records:
    print(json.dumps({k: v for k, v in record.items() if k != 'pixel_sha256'}), flush=True)
