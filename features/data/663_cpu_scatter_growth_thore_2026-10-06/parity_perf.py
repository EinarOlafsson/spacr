"""Exact raster parity and paired native-4K scatter/shade measurements."""

import hashlib
import importlib.util
import json
import math
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path('/mnt/wd4tb/scratch/theme-refinement-20261006/numba-integration')
sys.path.insert(0, '/mnt/wd4tb/spacr-worktrees/codex-theme-clock-20261006')
import spacr.qt.widgets

modules = []
for label in ('before', 'after'):
    path = ROOT / 'ambient.py' if label == 'before' else Path('/mnt/wd4tb/spacr-worktrees/codex-theme-numba-20261006/spacr/qt/widgets/ambient.py')
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._numba_' + label, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules.append(module)

app = QApplication.instance() or QApplication([])
rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
started = time.perf_counter()
compiler = threading.Thread(target=modules[1]._warm_packed_scatter, name='scatter-compile')
compiler.start()
compiler.join()
cold = {'compile_thread': compiler.name, 'kernel_ready': modules[1]._PACKED_SCATTER is not None, 'wall_ms': (time.perf_counter() - started) * 1000,
        'rss_before_kib': rss_before, 'rss_after_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
records, primitive = [], []
styles = (('#101418', None), ('#fafafa', ('#172adb', '#db792b')),
          ('#101418', ('#ff0000', '#00ffff')))


def engine(module, family, backdrop, colors, detail, seed=42):
    result = module.make_engine('data_art_' + family, 'spacr', backdrop, seed=seed,
                                blur=0, speed=1, size=1, density=1, resolution=detail)
    if colors:
        result.set_colors(colors)
    result.set_max_pixels(3840 * 2160)
    return result


def configure(result, family, clock):
    result.set_time(clock)
    result.set_pointer((0.38, 0.59))
    if family == 'impulse_lens':
        result._gravity_impulses = [
            (8 - i * 0.1, (0.5 + 0.12 * math.sin(i * 0.07),
                           0.5 + 0.09 * math.cos(i * 0.1)), 1.0 if i == 0 else 0.24)
            for i in range(24)]


for backdrop, colors in styles:
    engines = [engine(module, 'point_atlas', backdrop, colors, 2) for module in modules]
    x = np.tile(np.arange(-2, 19), 300)
    y = np.repeat(np.arange(-2, 13), 420)
    values = (np.arange(x.size) % 256) / 255.0
    for spread in (False, True):
        frames = []
        for item in engines:
            image = item._point_material(17, 11, x, y, values, spread)
            frames.append(image.bits().tobytes())
        identical = frames[0] == frames[1]
        primitive.append({'backdrop': backdrop, 'colors': colors, 'spread': spread,
                          'inputs': x.size, 'byte_identical': identical})
        assert identical

for family in ('point_atlas', 'impulse_lens', 'genetic_advection'):
    for backdrop, colors in styles:
        for detail in (1.0, 2.0):
            engines = [engine(module, family, backdrop, colors, detail) for module in modules]
            for clock in (8.0, 11.5):
                frames = []
                for item in engines:
                    configure(item, family, clock)
                    image = item.shade(3840, 2160)
                    assert image.width() == 3840 and image.height() == 2160
                    frames.append(image.bits().tobytes())
                identical = frames[0] == frames[1]
                records.append({'family': family, 'backdrop': backdrop, 'colors': colors,
                                'detail': detail, 'clock': clock, 'display': [3840, 2160],
                                'byte_identical': identical,
                                'pixel_sha256': hashlib.sha256(frames[0]).hexdigest()})
                assert identical
            print(json.dumps(records[-1]), flush=True)

perf = []
for family in ('point_atlas', 'impulse_lens', 'genetic_advection'):
    engines = [engine(module, family, '#101418', None, 2) for module in modules]
    shade, scatter_times = [[], []], [[], []]
    inputs = [[], []]
    for index, item in enumerate(engines):
        method = item._point_material
        def timed(*args, slot=index, original=method, **kwargs):
            started = time.perf_counter()
            image = original(*args, **kwargs)
            scatter_times[slot].append((time.perf_counter() - started) * 1000)
            inputs[slot].append(len(args[2]))
            return image
        item._point_material = timed
        configure(item, family, 8)
        item.shade(3840, 2160)
        scatter_times[index].clear()
    for frame in range(28):
        for index in (frame % 2, 1 - frame % 2):
            item = engines[index]
            configure(item, family, 8 + frame / 24)
            started = time.perf_counter()
            item.shade(3840, 2160)
            shade[index].append((time.perf_counter() - started) * 1000)
    result = {'family': family, 'display': [3840, 2160], 'inputs': inputs[0][0],
              'shade_median_ms': [statistics.median(values) for values in shade],
              'shade_p95_ms': [sorted(values)[math.ceil(.95 * len(values)) - 1] for values in shade],
              'point_material_median_ms': [statistics.median(values) for values in scatter_times],
              'point_material_p95_ms': [sorted(values)[math.ceil(.95 * len(values)) - 1] for values in scatter_times]}
    perf.append(result)
    print(json.dumps(result), flush=True)

receipt = {'renderer_sha256': hashlib.sha256((ROOT / 'ambient.py').read_bytes()).hexdigest(),
           'production_renderer_sha256': hashlib.sha256(Path(modules[1].__file__).read_bytes()).hexdigest(),
           'cold_compiler': cold, 'primitive': primitive, 'native_frame_parity': records,
           'paired_warm_performance': perf, 'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           'scope': 'exact CPU uint32 scatter; same samples, lookups, kernels, resolution; no fastmath'}
(ROOT / 'parity_perf.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'cold': cold, 'parity_frames': len(records), 'max_rss_kib': receipt['max_rss_kib']}), flush=True)
