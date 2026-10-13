"""Exact full-native mature-mycelium cache parity across palettes and controls."""
import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, sys.argv[1])
importlib.import_module('spacr.qt.widgets')
app = QApplication([])
modules = {}
for label in ['before', 'after']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._raster_gate_' + label, ROOT / (label + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    modules[label] = module



def install_tally(engine):
    """Count real successful private-cache reuse without changing its work."""
    original = engine._reuse_fungal_raster
    engine._proof_hits = 0

    def reuse(*args):
        """Delegate unchanged then record the successful-reuse result."""
        result = original(*args)
        engine._proof_hits += bool(result)
        return result

    engine._reuse_fungal_raster = reuse

results = {}
for variant in ['after']:
    rows = []
    rejected = False
    for background, palette, resolution, density in itertools.product(
            ['#101418', '#f6f7f9'], ['spacr', 'random', 'white', 'rgb16'], [1, 2], [.01, 1, 3]):
        engines = [modules[label].make_engine('data_art_fungal_growth', 'spacr' if palette in ('white', 'rgb16') else palette, background,
                                              seed=42, density=density, resolution=resolution, size=2.5)
                   for label in ['before', variant]]
        install_tally(engines[1])
        for engine in engines:
            if palette == 'white':
                engine.set_colors(['#ffffff'])
            elif palette == 'rgb16':
                engine.set_colors(['#fffe00017777', '#0303fffefefe'])
            engine.set_max_pixels(3840 * 2160)
        for clock in [.25, 95, 155]:
            frames = []
            for engine in engines:
                engine.set_time(clock)
                if engine is engines[1]:
                    for _ in range(8):
                        engine.shade(3840, 2160)
                if engine is engines[1]:
                    engine._proof_hits = 0
                frame = engine.shade(3840, 2160)
                assert frame.size().width() == 3840 and frame.size().height() == 2160
                assert frame.format() == QImage.Format_RGB32
                frames.append(frame)
            assert sum(v[1].nbytes + v[2].nbytes for v in getattr(engines[1], '_fungal_rasters', {}).values()) <= 8 * 1024**2
            assert engines[0].geometry(3840, 2160) == engines[1].geometry(3840, 2160)
            planes = [np.frombuffer(frame.constBits(), dtype=np.uint32) for frame in frames]
            assert all(np.all(plane >> 24 == 255) for plane in planes)
            differences = np.flatnonzero(planes[0] != planes[1])
            row = {'background': background, 'palette': palette, 'resolution': resolution,
                   'density': density, 'size': 2.5, 'clock': clock,
                   'differing_pixels': len(differences),
                   'cache_hits': getattr(engines[1], '_proof_hits', 0),
                   'first_difference_offsets': differences[:12].tolist(),
                   'pixel_sha256': [hashlib.sha256(frame.constBits()).hexdigest() for frame in frames]}
            rows.append(row)
            if len(differences):
                rejected = True
                print(variant, 'REJECT', json.dumps(row), flush=True)
                break
        if rejected:
            break
    results[variant] = {'all_pairs_exact': not rejected, 'pairs': len(rows), 'records': rows}
    print(variant, 'pairs', len(rows), 'all_exact', not rejected, flush=True)

receipt = {'source_sha256': {label: hashlib.sha256((ROOT / (label + '.py')).read_bytes()).hexdigest()
                             for label in modules}, 'native_size': [3840, 2160],
           'cpu_only': True, 'max_pixels': 3840 * 2160, 'variants': results}
(ROOT / 'native_parity.json').write_text(json.dumps(receipt, indent=2) + '\n')
