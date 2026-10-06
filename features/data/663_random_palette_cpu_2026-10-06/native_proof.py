"""Full native random kernel parity and unchanged existing palette pixels."""
import gc
import hashlib
import importlib.util
import json
import resource
import sys
from pathlib import Path

import numpy as np
from numba import njit
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
importlib.import_module('spacr.qt.widgets')

app = QApplication([])
modules = {}
for label in ['before', 'after']:
    spec = importlib.util.spec_from_file_location('spacr.qt.widgets._palette_' + label, ROOT/(label+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._PACKED_SCATTER_FAILED = True
    module._SATIN_COMPILER.failed = True
    modules[label] = module


def make(module, theme, background, palette, seed=42):
    selected = palette if module.is_valid_palette(theme, palette) else 'spacr'
    engine = module.make_engine(theme, selected, background, seed=seed, resolution=2)
    if selected != palette:
        engine.set_colors(['#3b82f6', '#ff00ff'])
    engine.set_max_pixels(3840*2160)
    engine.set_time(47)
    return engine


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
    return image


unchanged = []
for theme in modules['after'].AMBIENT_THEMES:
    for background in ['#101418', '#f6f7f9']:
        for palette in ['spacr', 'custom']:
            old = make(modules['before'], theme, background, palette)
            new = make(modules['after'], theme, background, palette)
            a, b = raster(old), raster(new)
            assert a.bits().tobytes() == b.bits().tobytes(), (theme, background, palette)
            unchanged.append({'theme':theme, 'background':background, 'palette':palette,
                              'raster_size':[b.width(),b.height()],
                              'recipe':'factory' if modules['before'].is_valid_palette(theme,palette)
                              else 'custom_colors_setter',
                              'pixel_sha256':hashlib.sha256(b.bits().tobytes()).hexdigest()})
            del old, new, a, b
            gc.collect()

after = modules['after']
kernel = njit(nogil=True, cache=False)(after._scatter_colored_grains)
coordinates=np.zeros(1,np.int32); table=np.zeros(256,np.uint64)
kernel(np.zeros(1,np.uint32),np.zeros(1,np.uint8),coordinates,coordinates,
       np.zeros(1,np.uint16),table,table,table,1,1,True,True)
exact = []
for theme in ['data_art_point_atlas', 'data_art_impulse_lens', 'data_art_genetic_advection']:
    for background in ['#101418', '#f6f7f9']:
        for seed in [17,42]:
            engine = make(after, theme, background, 'random', seed)
            for clock in [17.,47.]:
                engine.set_time(clock)
                after._COLORED_SCATTER = None
                a = raster(engine)
                reference = a.bits().tobytes()
                after._COLORED_SCATTER = kernel
                b = raster(engine)
                assert b.bits().tobytes() == reference
                exact.append({'theme':theme,'background':background,'seed':seed,'clock':clock,
                              'pixel_sha256':hashlib.sha256(reference).hexdigest()})
            if seed==42:
                b.save(str(ROOT/((theme+'-dark' if engine.dark else theme+'-light')+'.png')))
            del engine,a,b,reference
            gc.collect()
receipt={'source_sha256':{label:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for label,module in modules.items()},
         'native_size':[3840,2160],'unchanged_existing_palette_pairs':unchanged,
         'full_native_random_compiled_fallback_pairs':exact,
         'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT/'native_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('unchanged:',len(unchanged),'random exact:',len(exact),'peakRSSKiB',receipt['peak_rss_kib'])
