"""Measure every spaCR animation's shading cost and record its GPU coverage (N684).

Run headless on the CPU path:

    CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen \
        tools/run_capped.sh 8G python tools/audit_animation_performance.py OUT.json

Each theme is shaded on a thread named like the application's producer, at
1920x1080 and 3840x2160, at its shipped density and at density 1.0, with the
CPU renderer. Frame times are a distribution (median, p90, max) of complete
``advance`` plus ``shade`` calls, after warm-up. Peak RSS is reported for the
whole run. ``--gpu`` additionally measures the optional GPU renderer where
a theme has one; run that only through tools/gpu_turn.sh. Spinn
(``data_art_tissue_facets``) only moves under the pointer, so a pointer
sweeps across it; without one it republishes its resting frame (~0 ms).
``--cost-table RECEIPT`` prints Automatic's cost table from a ``--gpu``
receipt.
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

GPU_COVERAGE = {
    "data_art_impulse_lens": "GPU point renderer (monochrome palettes)",
    "data_art_genetic_advection": "GPU point renderer (monochrome palettes)",
    "data_art_point_atlas": "GPU point renderer (monochrome palettes)",
    "data_art_spaceout_field": "GPU point renderer outside colour-wave events",
    "blobs": "GPU radial-gradient discs",
    "drift": "GPU star discs on a shading thread (Animation GPU On only)",
    "aurora": "GPU ray texture and fold strips",
    "data_art_fungal_growth": "GPU filament strokes and tips",
    "data_art_tissue_facets": "GPU paper-tile atlas while the pointer spins it",
    "fractal": "CPU in this backdrop; the native spaceout fractal has its own GPU backend",
}

#: Themes that only move under the pointer; the audit moves one across them.
POINTER_THEMES = {"data_art_tissue_facets"}


def _measure(ambient, theme, size, density, backend, frames):
    """Return per-frame milliseconds for one theme on the shading thread.

    Engines without an off-thread ``shade`` (the starfield) are timed
    painting a full-size canvas, which is what the GUI thread pays.
    """
    from PySide6.QtGui import QImage, QPainter

    engine = ambient.make_engine(theme, ambient.default_palette_for(theme),
                                 "#191919", seed=684, density=density)
    engine.set_max_pixels(size[0] * size[1])
    if hasattr(engine, "_graphics_backend"):
        engine._graphics_backend = backend
    if theme in POINTER_THEMES:
        engine.set_gravity_radius(0.35)
    samples, used = [], []
    buffered = isinstance(engine, ambient._BufferedEngine)

    def run():
        try:
            for index in range(frames + 3):
                if theme in POINTER_THEMES:
                    engine.set_pointer((0.2 + 0.6 * (index % 12) / 11, 0.5))
                begin = time.perf_counter()
                engine.advance(1 / 30)
                if buffered or (backend != "cpu" and hasattr(engine, "shade")):
                    engine.shade(*size)
                else:
                    canvas = QImage(size[0], size[1], QImage.Format_ARGB32_Premultiplied)
                    painter = QPainter(canvas)
                    engine.paint(painter, *size)
                    painter.end()
                if index >= 3:
                    samples.append((time.perf_counter() - begin) * 1000)
                    used.append(getattr(engine, "_point_graphics", None) is not None)
        finally:
            release = getattr(engine, "_release_point_graphics", None)
            if release is not None:
                release()

    thread = threading.Thread(target=run, name="spacr-ambient-shade")
    thread.start()
    thread.join()
    samples.sort()
    return {"median_ms": round(statistics.median(samples), 3),
            "p90_ms": round(samples[int(0.9 * (len(samples) - 1))], 3),
            "max_ms": round(samples[-1], 3),
            "gpu_frames": sum(used), "frames": len(samples)}


def cost_table(report) -> str:
    """The ``_GRAPHICS_COST`` literal for every theme a receipt timed on both paths."""
    table = {}
    for row in report["rows"]:
        gpu = row.get("gpu")
        if not gpu or gpu.get("gpu_frames", 0) < gpu.get("frames", 1):
            continue
        table.setdefault(row["theme"], []).append(
            (row["size"][0] * row["size"][1], row["density"],
             row["cpu"]["median_ms"], gpu["median_ms"]))
    lines = ["_GRAPHICS_COST = {"]
    for theme, rows in table.items():
        lines.append(f"    {theme!r}: (")
        lines.extend(f"        {tuple(item)!r}," for item in rows)
        lines.append("    ),")
    lines.append("}")
    return "\n".join(lines)


def main(argv=None) -> int:
    """Write the audit receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--cost-table", type=Path, default=None, metavar="RECEIPT",
                        help="print ambient._GRAPHICS_COST rows from a --gpu receipt and exit")
    args = parser.parse_args(argv)
    if args.cost_table is not None:
        print(cost_table(json.loads(args.cost_table.read_text())))
        return 0
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["spacr-animation-audit"])
    from spacr.qt.widgets import ambient

    themes = [theme for theme in ambient.ANIMATION_CHOICES if theme != ambient.NO_ANIMATION]
    themes += [theme for theme in ambient.SPACEOUT_ONLY_THEMES + (ambient.SPACEOUT_THEME,)
               if theme not in themes]
    rows = []
    for theme in themes:
        for size in ((1920, 1080), (3840, 2160)):
            for density in sorted({ambient._default_density_for(theme), 1.0}):
                row = {"theme": theme, "label": ambient.theme_label(theme)
                       if hasattr(ambient, "theme_label") else theme,
                       "size": list(size), "density": density,
                       "detail": 1.0, "palette": ambient.default_palette_for(theme),
                       "gpu_coverage": GPU_COVERAGE.get(theme, "CPU only"),
                       "pointer": theme in POINTER_THEMES}
                row["cpu"] = _measure(ambient, theme, size, density, "cpu", args.frames)
                if args.gpu and theme in GPU_COVERAGE:
                    row["gpu"] = _measure(ambient, theme, size, density, "gpu", args.frames)
                rows.append(row)
                print(json.dumps(row), flush=True)
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        source = "unknown"
    report = {"schema": 1, "source_commit": source,
              "machine": platform.processor() or platform.machine(),
              "platform": platform.platform(), "python": sys.version.split()[0],
              "qt_platform": app.platformName(), "display_scaling": 1.0,
              "peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
              "qualification": ("Offscreen shading-thread cost only; no displayed frame "
                                "rate, compositor or native-display claim."),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
