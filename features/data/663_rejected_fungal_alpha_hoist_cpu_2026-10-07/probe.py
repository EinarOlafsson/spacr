"""Qualify one frame-invariant alpha calculation against frozen native pixels."""
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

import spacr.qt.widgets

root = Path(__file__).resolve().parent
before = Path('/mnt/wd4tb/scratch/fungal-stroker-parity-20261007/before.py').read_text()
start = before.index('    def geometry(', before.index('class _FungalGrowthEngine('))
end = before.index('    def _fungal_paths(', start)
body = before[start:end]
needle = '        branch_count = self.element_count(90, 240)\n'
assert body.count(needle) == 1
body = body.replace(needle, needle + '        fractional_alpha = self._fractional_alpha_scale(90)\n')
assert body.count('* fade * self._fractional_alpha_scale(90)') == 1
body = body.replace('* fade * self._fractional_alpha_scale(90)', '* fade * fractional_alpha')
sources = {'before': before, 'candidate': before[:start] + body + before[end:]}
mods = {}
for name, source in sources.items():
    path = root / (name + '.py')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._frame_invariant_' + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    mods[name] = module
app = QApplication([])
rows = []
for palette, resolution, density, background in [('spacr', 2, 3, '#101418'),
                                               ('random', 2, 3, '#101418'),
                                               ('custom', 1, 3, '#101418'),
                                               ('spacr', 2, 0.01, '#ffffff')]:
    engines = {name: mod.make_engine('data_art_fungal_growth', palette, background, seed=42,
                resolution=resolution, density=density, size=2.5, blur=0) for name, mod in mods.items()}
    for engine in engines.values():
        engine.set_max_pixels(3840 * 2160)
        if palette == 'custom':
            engine.set_colors(['#ff2200', '#00ff44', '#2288ff'])
    geometry_cases = 0
    for clock in (0, 2, 9, 30, 95, 98, 3600.33):
        for engine in engines.values():
            engine.set_time(clock)
        assert engines['before'].geometry(3840, 2160) == engines['candidate'].geometry(3840, 2160)
        geometry_cases += 1
    pairs = []
    for clock in (95, 98, 3600.33):
        for engine in engines.values():
            engine.set_time(clock)
        frames = {name: engine.shade(3840, 2160) for name, engine in engines.items()}
        pixels = {name: np.frombuffer(frame.constBits(), dtype=np.uint32) for name, frame in frames.items()}
        difference = int(np.count_nonzero(pixels['before'] != pixels['candidate']))
        assert difference == 0
        pairs.append({'clock': clock, 'different_pixels': difference,
                      'hashes': {name: hashlib.sha256(frame.constBits()).hexdigest() for name, frame in frames.items()}})
        del pixels, frames
    for engine in engines.values():
        engine.set_time(95)
    geometry_times = {name: [] for name in engines}
    for index in range(120):
        for name in (list(engines) if index % 2 == 0 else list(engines)[::-1]):
            start = time.perf_counter()
            engines[name].geometry(3840, 2160)
            geometry_times[name].append((time.perf_counter() - start) * 1000)
    for engine in engines.values():
        for _ in range(6):
            engine.shade(3840, 2160)
    shader_times = {name: [] for name in engines}
    for index in range(24):
        for name in (list(engines) if index % 2 == 0 else list(engines)[::-1]):
            start = time.perf_counter()
            frame = engines[name].shade(3840, 2160)
            shader_times[name].append((time.perf_counter() - start) * 1000)
            del frame
    row = {'palette': palette, 'resolution': resolution, 'density': density, 'background': background,
           'geometry_equal_cases': geometry_cases, 'native_pairs': pairs,
           'geometry_median_ms': {name: statistics.median(values) for name, values in geometry_times.items()},
           'shader_median_ms': {name: statistics.median(values) for name, values in shader_times.items()}}
    print(json.dumps(row), flush=True)
    rows.append(row)
receipt = {'source_sha256': {name: hashlib.sha256(source.encode()).hexdigest() for name, source in sources.items()},
           'rows': rows, 'scope': 'One frame-invariant alpha hoist only; exact native geometry/frame qualification and balanced direct-stage/shader timing; not actual live FPS.'}
(root / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
