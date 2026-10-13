"""Source-bound native density and radius receipts, no simulated FPS."""
import gc
import hashlib
import importlib.util
import json
import resource
import os
import sys
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

WORKTREE = Path(os.environ['SPACR_REPRO_REPO'])
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))
import spacr.qt.widgets

app = QApplication([])
modules = {}
for label, path in [('before', ROOT / 'before.py'),
                    ('after', ROOT / 'after.py')]:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._graded_' + label, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._SATIN_COMPILER.failed = True
    module._PACKED_SCATTER_FAILED = True
    modules[label] = module


def raster(engine):
    if hasattr(engine, 'shade'):
        image = engine.shade(3840, 2160)
    else:
        image = QImage(3840, 2160, QImage.Format_RGB32)
        image.fill(engine.identity)
        painter = QPainter(image)
        painter.setCompositionMode(engine.mode)
        try:
            engine.paint(painter, 3840, 2160)
        finally:
            painter.end()
    pixels = np.frombuffer(image.bits(), np.uint8).reshape(image.height(), image.width(), 4)
    return pixels.copy()


def engine(module, theme, background='#101418', density=1):
    result = module.make_engine(theme, 'spacr', background, seed=42,
                                resolution=2, density=density)
    result.set_max_pixels(3840 * 2160)
    result.set_time(97)
    return result


rows = []
for theme in ['blobs', 'aurora', 'drift', 'data_art_point_atlas', 'data_art_tissue_facets',
              'data_art_chromatin_ribbon', 'data_art_genetic_advection',
              'data_art_impulse_lens', 'data_art_fungal_growth', 'data_art_thore']:
    current = engine(modules['after'], theme)
    previous = None
    for density in [.01, .10, .50]:
        current.set_density(density)
        pixels = raster(current)
        rgb = pixels[:, :, :3]
        rows.append({'theme': theme, 'density': density,
                     'raster_size': [pixels.shape[1], pixels.shape[0]],
                     'rgb_sum': int(rgb.sum()),
                     'ink_pixels': int(np.count_nonzero(np.any(rgb != 0, axis=2))),
                     'changed_pixels_previous': None if previous is None else
                     int(np.count_nonzero(np.any(pixels != previous, axis=2))),
                     'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
        previous = pixels
    del current, previous, pixels, rgb
    gc.collect()

parity = []
for theme in ['blobs', 'aurora', 'data_art_tissue_facets', 'data_art_chromatin_ribbon',
              'data_art_thore']:
    for background in ['#101418', '#f6f7f9']:
        for density in [.50, 1, 2]:
            old = engine(modules['before'], theme, background, density)
            new = engine(modules['after'], theme, background, density)
            a, b = raster(old), raster(new)
            assert np.array_equal(a, b), (theme, background, density)
            parity.append({'theme': theme, 'background': background, 'density': density,
                           'pixel_sha256': hashlib.sha256(b.tobytes()).hexdigest()})
            del old, new, a, b
            gc.collect()

radius = []
for theme in ['data_art_point_atlas', 'data_art_genetic_advection', 'data_art_impulse_lens',
              'data_art_tissue_facets']:
    for value in [.01, .10, .50]:
        current = engine(modules['after'], theme)
        current.set_gravity_radius(value)
        a = raster(current)
        current.set_pointer((.5, .5))
        b = raster(current)
        radius.append({'theme': theme, 'radius': value,
                       'changed_pixels': int(np.count_nonzero(np.any(a != b, axis=2))),
                       'changed_fraction': float(np.mean(np.any(a != b, axis=2))),
                       'pixel_sha256': hashlib.sha256(b.tobytes()).hexdigest()})
        del current, a, b
        gc.collect()

receipt = {'source_sha256': {label: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                             for label, module in modules.items()},
           'scope': 'Actual 3840x2160 display; native DataArt, existing classic buffer sizes; CPU NumPy fallback. Pixel sum/occupancy are descriptive, not acceptance thresholds.',
           'density': rows, 'unchanged_default_higher_pairs': parity,
           'gravity_radius': radius,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'density_frames': len(rows), 'exact_unchanged_pairs': len(parity),
                  'radius': radius, 'peak_rss_kib': receipt['peak_rss_kib']}, indent=2))
