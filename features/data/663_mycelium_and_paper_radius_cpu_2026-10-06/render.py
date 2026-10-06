"""Actual native common-origin branching frames and low-density receipt."""
import hashlib
import os
import importlib.util
import json
import resource
import sys
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.environ['SPACR_REPRO_REPO'])
import spacr.qt.widgets

app = QApplication([])
path = ROOT / 'after.py'
spec = importlib.util.spec_from_file_location('spacr.qt.widgets._mycelium_visual', path)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)
rows = []
for background in ['#101418', '#f6f7f9']:
    engine = a.make_engine('data_art_fungal_growth', 'spacr', background, seed=42,
                           resolution=2, density=1)
    engine.set_max_pixels(3840 * 2160)
    for clock in [5., 18., 47., 97., 3600.33]:
        engine.set_time(clock)
        image = QImage(3840, 2160, QImage.Format_RGB32)
        image.fill(engine.background)
        painter = QPainter(image)
        try:
            engine.paint(painter, 3840, 2160)
        finally:
            painter.end()
        name = ('dark' if engine.dark else 'light') + '-' + str(clock) + '.png'
        image.save(str(ROOT / name))
        pixels = np.frombuffer(image.bits(), np.uint8).reshape(2160, 3840, 4)[:, :, :3]
        page = np.array([engine.background.blue(), engine.background.green(), engine.background.red()])
        occupancy = float(np.any(pixels != page, axis=2).mean())
        assert occupancy <= .25
        geometry = engine.geometry(3840, 2160)
        rows.append({'background': background, 'clock': clock, 'file': name,
                     'size': [3840, 2160], 'occupied_fraction': occupancy,
                     'edges': len(geometry), 'advancing_tips': sum(edge[6] < 1 for edge in geometry),
                     'pixel_sha256': hashlib.sha256(image.bits().tobytes()).hexdigest()})
        if clock == 18 and engine.dark:
            origin = engine._origin
            image.copy(round(origin[0] * 3840) - 400, round(origin[1] * 2160) - 700,
                       800, 750).save(str(ROOT / 'native-origin-crop.png'))

engine = a.make_engine('data_art_fungal_growth', 'spacr', '#101418', seed=42, resolution=2)
engine.set_max_pixels(3840 * 2160)
engine.set_time(80)
density = []
for value in [.01, .1, .5]:
    engine.set_density(value)
    image = engine.shade(3840, 2160)
    density.append({'density': value, 'edges': len(engine.geometry(3840, 2160)),
                    'pixel_sha256': hashlib.sha256(image.bits().tobytes()).hexdigest()})
receipt = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
           'scope': 'Actual CPU native3840x2160 deterministic frames, no aesthetic acceptance claim',
           'frames': rows, 'density': density,
           'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
(ROOT / 'frames.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
