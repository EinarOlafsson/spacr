"""Actual native renderer review, default parity and bounded direct-shade costs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import statistics
import sys
import time

import numpy as np
from PySide6.QtWidgets import QApplication

repo = Path(sys.argv[1]).resolve()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo))
from spacr.qt.widgets import ambient

spec = importlib.util.spec_from_file_location("spacr.qt.widgets._spin_before", root / "before.py")
before = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = before
spec.loader.exec_module(before)
app = QApplication([])
width, height = 3840, 2160
assert Path(ambient.__file__).resolve() == repo / "spacr/qt/widgets/ambient.py"
source_sha = hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest()


def engine(module, family, background, palette="spacr"):
    result = module.make_engine("data_art_" + family, palette, background,
                                 seed=42, resolution=1, size=1, density=1, blur=0)
    result.set_max_pixels(width * height)
    return result


def pixels(image):
    return np.frombuffer(image.constBits(), np.uint32).reshape(height, width)


records = []
for background in ("#101418", "#f6f7f9"):
    for family in ("point_atlas", "tissue_facets"):
        current = engine(ambient, family, background)
        reference = engine(before, family, background)
        current.set_time(2)
        reference.set_time(2)
        started = time.perf_counter()
        resting = current.shade(width, height)
        cold_ms = (time.perf_counter() - started) * 1000
        expected = reference.shade(width, height)
        current.set_time(2.25)
        later = current.shade(width, height)
        record = {"family": family, "background": background,
                  "native": [resting.width(), resting.height()],
                  "cold_direct_shade_ms": cold_ms,
                  "changed_default_pixels_vs_previous": int(np.count_nonzero(pixels(resting) != pixels(expected))),
                  "clock_motion_pixels": int(np.count_nonzero(pixels(resting) != pixels(later)))}
        if family == "tissue_facets":
            assert record["changed_default_pixels_vs_previous"] == 0
            assert record["clock_motion_pixels"] == 0
        else:
            assert record["clock_motion_pixels"] > 1000
        current.set_gravity_radius(.5)
        current.set_pointer((.5, .5))
        initial = current.shade(width, height)
        current.set_time(2.5)
        active = current.shade(width, height)
        record["mouse_changed_pixels"] = int(np.count_nonzero(pixels(active) != pixels(later)))
        record["stationary_mouse_animation_pixels"] = int(np.count_nonzero(pixels(active) != pixels(initial)))
        assert record["mouse_changed_pixels"] > 1000
        assert record["stationary_mouse_animation_pixels"] > 1000
        palette_samples = []
        for index in range(8):
            current.advance(1 / 24)
            started = time.perf_counter()
            frame = current.shade(width, height)
            palette_samples.append((time.perf_counter() - started) * 1000)
            assert np.all((pixels(frame) >> 24) == 255)
        label = "dark" if background == "#101418" else "light"
        frame.save(str(root / (family + "-" + label + "-native.png")))
        record["direct_active_shade_median_ms"] = statistics.median(palette_samples)
        record["direct_active_shade_max_ms"] = max(palette_samples)
        record["frame_sha256"] = hashlib.sha256(frame.constBits()).hexdigest()
        if family == "tissue_facets":
            cells = next(value for key, value in current._material_cache.items() if key[0] == "tissue_facets")
            angles = next(value[1] for key, value in current._material_cache.items() if key[0] == "tissue_rotation")
            record["cached_facets"] = len(cells)
            record["spinning_facets"] = sum(angle != 0 for angle in angles)
            record["owned_angle_count"] = len(angles)
            record["owned_resting_tile_bytes"] = sum(cell[5].sizeInBytes() for cell in cells)
            current.set_gravity_radius(0)
            restored = current.shade(width, height)
            assert np.array_equal(pixels(resting), pixels(restored))
            record["zero_radius_restores_default"] = True
        records.append(record)
        print(json.dumps(record), flush=True)
        del current, reference, resting, expected, later, initial, active, frame
receipt = {"source_sha256": source_sha,
           "before_sha256": hashlib.sha256((root / "before.py").read_bytes()).hexdigest(),
           "scope": "Actual native 4K renderer only; eight direct active samples per family/background, not worker/GUI FPS acceptance",
           "controls": {"resolution": 1, "size": 1, "density": 1, "blur": 0, "radius": .5},
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "records": records, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
           "hard_24fps_acceptance": False, "human_aesthetic_acceptance": False}
assert hashlib.sha256(Path(ambient.__file__).read_bytes()).hexdigest() == source_sha
(root / "native_review.json").write_text(json.dumps(receipt, indent=2) + "\n")
