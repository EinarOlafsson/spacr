"""Capture an actual native advection image from a frozen renderer source."""

import importlib
import importlib.util
import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
importlib.import_module('spacr.qt.widgets')
source = Path(os.environ['AMBIENT_SOURCE'])
spec = importlib.util.spec_from_file_location(
    'spacr.qt.widgets._advection_capture', source)
ambient = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ambient
spec.loader.exec_module(ambient)
app = QApplication.instance() or QApplication([])
engine = ambient.make_engine(
    'data_art_genetic_advection', 'spacr', '#101418', seed=42)
engine.set_max_pixels(1920 * 1080)
engine.set_time(8.0)
engine.set_gravity_radius(.25)
engine.set_pointer((.5, .5))
image = engine.shade(1920, 1080)
path = source.with_suffix('.png')
assert image.save(str(path))
print(path)
