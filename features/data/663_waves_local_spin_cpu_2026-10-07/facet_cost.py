"""Bounded facet-only native comparison without grain/compiler warm-up."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

from PySide6.QtWidgets import QApplication

repo = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
from spacr.qt.widgets import ambient

source = Path(ambient.__file__).read_text()
tree = ast.parse(source)
node = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
            and cls.name == "_DataArtEngine" for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_paint_tissue_facets")
method = "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])
vector_source = source.replace(method, method.replace(
    "tile, extent_x, extent_y))", "tile, extent_x, extent_y, (triangles, outline)))"
).replace(
    "cx, cy, rx, ry, phase, tile, extent_x, extent_y = cell",
    "cx, cy, rx, ry, phase, tile, extent_x, extent_y, vector = cell"
).replace(
    "painter.setRenderHint(QPainter.SmoothPixmapTransform, True)\n                    painter.drawImage(QPointF(-extent_x, -extent_y), tile)",
    "painter.setPen(Qt.NoPen)\n                    for triangle, color in vector[0]:\n                        painter.setBrush(color)\n                        painter.drawPolygon(triangle)\n                    painter.setBrush(Qt.NoBrush)\n                    painter.setPen(QPen(self._ink(0, 0.23), 0.65))\n                    painter.drawPolygon(vector[1])"
))
assert vector_source != source
(root / "vector_candidate.py").write_text(vector_source)
fast_source = source.replace("painter.setRenderHint(QPainter.SmoothPixmapTransform, True)\n                    painter.drawImage", "painter.setRenderHint(QPainter.SmoothPixmapTransform, False)\n                    painter.drawImage")
(root / "nearest_candidate.py").write_text(fast_source)


def load(path, name):
    spec = importlib.util.spec_from_file_location("spacr.qt.widgets._facet_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._begin_ambient_startup()
    return module


modules = {"original": load(root / "before.py", "before"),
           "native_bilinear": ambient,
           "native_nearest": load(root / "nearest_candidate.py", "nearest"),
           "native_vector": load(root / "vector_candidate.py", "vector")}
ambient._begin_ambient_startup()
app = QApplication([])
engines = {}
for label, module in modules.items():
    engine = module.make_engine("data_art_tissue_facets", "spacr", "#101418", seed=42,
                                resolution=1, blur=0)
    engine.set_max_pixels(3840 * 2160)
    engine.set_gravity_radius(.5)
    engine.set_pointer((.5, .5))
    engine.shade(3840, 2160)
    engine.advance(.1)
    engine.shade(3840, 2160)
    engines[label] = engine
samples = {label: [] for label in engines}
for index in range(8):
    labels = list(engines)
    if index % 2:
        labels.reverse()
    for label in labels:
        engines[label].advance(1 / 24)
        started = time.perf_counter()
        image = engines[label].shade(3840, 2160)
        samples[label].append((time.perf_counter() - started) * 1000)
        if index == 7:
            image.save(str(root / (label + "-native.png")))
receipt = {"scope": "8 alternating facet-only native4K direct shaders per variant, compiler gates closed; not GUI/worker FPS",
           "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
           "load_average": os.getloadavg(),
           "samples_ms": samples,
           "median_ms": {label: statistics.median(values) for label, values in samples.items()},
           "no_production_change": True}
(root / "facet_cost.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt), flush=True)
