"""Measure the currently imported advection renderer at fixed seed and times."""

import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
ambient = importlib.import_module('spacr.qt.widgets.ambient')


width, height = 1920, 1080
engine = ambient.make_engine(
    'data_art_genetic_advection', 'spacr', '#101418', seed=42)
engine.set_max_pixels(width * height)
frames = []


def capture(w, h, x, y, intensity, spread=False):
    frames.append((np.asarray(x).copy(), np.asarray(y).copy()))
    image = QImage(w, h, QImage.Format_RGB32)
    image.fill(engine.identity)
    return image


engine._point_material = capture
records = []
for radius in (.01, .1, .5):
    engine.set_gravity_radius(radius)
    for instant in (2.0, 2.2):
        engine.set_time(instant)
        engine.set_pointer(None)
        engine.shade(width, height)
        old_x, old_y = frames[-1]
        engine.set_pointer((.5, .5))
        engine.shade(width, height)
        new_x, new_y = frames[-1]
        distance = np.hypot(old_x - width / 2, old_y - height / 2)
        outside = distance >= radius * height + 2
        inside = distance < radius * height * .8
        new_distance = np.hypot(new_x[inside] - width / 2,
                                new_y[inside] - height / 2)
        changed = (new_x != old_x) | (new_y != old_y)
        records.append({
            'radius': radius,
            'time': instant,
            'inside_samples': int(inside.sum()),
            'changed_samples': int(changed.sum()),
            'outside_samples': int(outside.sum()),
            'outside_identical': bool(np.array_equal(old_x[outside], new_x[outside])
                                      and np.array_equal(old_y[outside], new_y[outside])),
            'inside_inward_fraction': float(np.mean(new_distance < distance[inside])),
            'inside_mean_inward_pixels': float(np.mean(distance[inside] - new_distance)),
        })
print(json.dumps(records, indent=2))
