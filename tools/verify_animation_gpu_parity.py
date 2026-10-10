"""Compare every GPU-capable animation's graphics frames with its CPU frames (N684).

For each theme, background (dark and light), canvas size and density, two
engines are built from the same seed and driven through the same clock
and pointer. One shades with Animation GPU off (the CPU path), the other
with it on (the moderngl renderer). Both shade on a thread named like the
application's producer, which is the only thread the renderer accepts.
Each frame pair is compared channel by channel; the run also times both
paths. A frame the renderer refused (no context, a probe failure, a busy
GPU, a draw error) is recorded as refused, never as a pass.

Workstation (hardware GL), through the GPU queue:

    QT_QPA_PLATFORM=offscreen tools/gpu_turn.sh n684-gpu-parity \\
        tools/run_capped.sh 8G python tools/verify_animation_gpu_parity.py \\
        features/data/684_animation_gpu_parity_<date>/receipt.json --images

Leave ``CUDA_VISIBLE_DEVICES`` unset there: NVIDIA's EGL device list can
follow it, and an empty value would leave no hardware device.

Shader and fallback checks without a GPU (Mesa llvmpipe, CPU only). This
never touches a hardware device: it forces the Mesa EGL vendor and its
software device, and refuses to run if the renderer is not software:

    __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json \\
        CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen \\
        tools/run_capped.sh 8G python tools/verify_animation_gpu_parity.py \\
        OUT.json --software-device 1

(``--software-device`` is the EGL device index of Mesa's software device;
``eglinfo -B`` or the error message lists the devices.) Software results
check the shaders and the comparison only; they say nothing about speed.

A pair passes when the mean absolute channel difference is at most
``--mean`` (default 1.0 of 255), the 99th percentile at most ``--p99``
(default 12) and at most ``--outliers`` (default 0.5 %) of pixels differ by
more than 48 in some channel. QPainter and OpenGL rasterise edges and
bilinear samples slightly differently, so exact equality is not expected;
the receipt keeps the full distribution so the thresholds can be judged.
Exit status is 0 only when every compared pair passed and none was refused.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

THEMES = ("blobs", "drift", "aurora", "data_art_fungal_growth",
          "data_art_tissue_facets", "data_art_impulse_lens",
          "data_art_genetic_advection", "data_art_point_atlas")

BACKGROUNDS = ("#191919", "#f2f2f2")


class _Producer:
    """One persistent thread named like the application's producer.

    An OpenGL context is current on the thread that made it, so every
    frame, like the application's shading thread, runs on this one thread.
    """

    def __init__(self):
        """Start the worker thread."""
        import queue

        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="spacr-ambient-shade",
                                        daemon=True)
        self._thread.start()

    def _run(self):
        """Run submitted jobs until told to stop."""
        while True:
            job = self._jobs.get()
            if job is None:
                return
            function, box, done = job
            try:
                box["value"] = function()
            except BaseException as error:
                box["error"] = error
            done.set()

    def __call__(self, function):
        """Run ``function`` on the producer thread and return its result."""
        box, done = {}, threading.Event()
        self._jobs.put((function, box, done))
        done.wait()
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def stop(self):
        """Finish the worker thread."""
        self._jobs.put(None)
        self._thread.join()


_on_producer = None


def _pixels(image):
    """An ``(h, w, 4)`` uint8 copy of ``image`` as premultiplied ARGB32 bytes."""
    import numpy as np
    from PySide6.QtGui import QImage

    image = image.convertToFormat(QImage.Format_ARGB32_Premultiplied)
    raw = np.frombuffer(image.constBits(), dtype=np.uint8)
    rows = raw.reshape(image.height(), image.bytesPerLine())
    return rows[:, :image.width() * 4].reshape(image.height(), image.width(), 4).copy()


def _compare(cpu, gpu):
    """Channel difference statistics between two frames."""
    import numpy as np

    a, b = _pixels(cpu), _pixels(gpu)
    if a.shape != b.shape:
        return {"shape_cpu": list(a.shape), "shape_gpu": list(b.shape)}
    channels = slice(0, 4) if cpu.hasAlphaChannel() else slice(0, 3)
    diff = np.abs(a[:, :, channels].astype(np.int16) - b[:, :, channels].astype(np.int16))
    worst = diff.max(axis=2)
    return {"mean": round(float(diff.mean()), 4),
            "p99": int(np.percentile(worst, 99)),
            "max": int(worst.max()),
            "over_16": round(float((worst > 16).mean()), 6),
            "over_48": round(float((worst > 48).mean()), 6),
            "diff": worst}


def _engines(ambient, theme, background, density, size):
    """Two identically seeded engines, CPU and GPU, at ``size``."""
    pair = []
    for backend in ("cpu", "gpu"):
        engine = ambient.make_engine(theme, ambient.default_palette_for(theme),
                                     background, seed=684, density=density)
        engine.set_max_pixels(size[0] * size[1])
        engine._graphics_backend = backend
        if theme == "data_art_tissue_facets":
            engine.pointer = (0.45, 0.55)
            engine.gravity_radius = 0.35
        pair.append(engine)
    return pair


def _frame(engine, size):
    """Shade one frame on the producer thread; return ``(image, ms, gpu)``."""
    def run():
        begin = time.perf_counter()
        image = engine.shade(*size)
        elapsed = (time.perf_counter() - begin) * 1000
        return image, elapsed, engine._point_graphics is not None
    return _on_producer(run)


def main(argv=None) -> int:
    """Write the parity receipt; return 0 only when everything passed."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", type=Path)
    parser.add_argument("--themes", nargs="*", default=list(THEMES))
    parser.add_argument("--sizes", nargs="*", default=["1920x1080", "3840x2160"])
    parser.add_argument("--densities", nargs="*", type=float, default=None,
                        help="default: each theme's shipped density and 1.0")
    parser.add_argument("--times", nargs="*", type=float, default=[0.0, 7.25, 41.5])
    parser.add_argument("--timing-frames", type=int, default=8)
    parser.add_argument("--mean", type=float, default=1.0)
    parser.add_argument("--p99", type=int, default=12)
    parser.add_argument("--outliers", type=float, default=0.005)
    parser.add_argument("--images", action="store_true",
                        help="save CPU, GPU and difference PNGs of the worst pair per theme")
    parser.add_argument("--software-device", type=int, default=None,
                        help="Mesa EGL software device index (shader checks without a GPU)")
    args = parser.parse_args(argv)

    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(["spacr-gpu-parity"])
    from spacr.qt.widgets import ambient

    global _on_producer
    _on_producer = _Producer()
    ambient._RESOURCE_POLICY = None
    if args.software_device is not None:
        options = ambient._graphics_context_options()
        options["device_index"] = args.software_device
        ambient._graphics_context_options = lambda: dict(options)
        ambient._flow_graphics_preflight = lambda: True
        ambient._flow_compute_idle = lambda: True
    elif not ambient._flow_graphics_preflight():
        print("no hardware OpenGL 4.3 context (moderngl missing, software "
              "renderer, or probe failure); nothing compared", file=sys.stderr)
        return 2

    def describe():
        renderer = ambient._FlowPointGraphics()
        try:
            return {key: str(renderer._context.info.get(key))
                    for key in ("GL_RENDERER", "GL_VENDOR", "GL_VERSION")}
        finally:
            renderer._close()

    gl_info = _on_producer(describe)
    software = any(name in gl_info["GL_RENDERER"].lower()
                   for name in ambient._SOFTWARE_RENDERERS)
    if args.software_device is not None and not software:
        print(f"device {args.software_device} is {gl_info['GL_RENDERER']!r}, "
              "not a software renderer; refusing", file=sys.stderr)
        return 2
    print(json.dumps(gl_info), flush=True)

    rows, worst, ok = [], {}, True
    for theme in args.themes:
        densities = args.densities or sorted({ambient._default_density_for(theme), 1.0})
        for background in BACKGROUNDS:
            for text in args.sizes:
                size = tuple(int(part) for part in text.lower().split("x"))
                for density in densities:
                    cpu, gpu = _engines(ambient, theme, background, density, size)
                    pairs, cpu_ms, gpu_ms, refused = [], [], [], 0
                    for moment in args.times:
                        for engine in (cpu, gpu):
                            engine.set_time(moment)
                            if theme == "data_art_tissue_facets":
                                for step in range(1, 4):
                                    engine.set_time(moment + step / 30)
                                    _frame(engine, size)
                        cpu_image, _, _ = _frame(cpu, size)
                        gpu_image, _, used = _frame(gpu, size)
                        if not used or gpu._graphics_failed:
                            refused += 1
                            continue
                        pairs.append(_compare(cpu_image, gpu_image))
                        if args.images:
                            score = pairs[-1].get("mean", 1e9)
                            if score >= worst.get(theme, (-1,))[0]:
                                worst[theme] = (score, cpu_image.copy(), gpu_image.copy(),
                                                pairs[-1].get("diff"), background, size)
                    for engine, sink in ((cpu, cpu_ms), (gpu, gpu_ms)):
                        for index in range(args.timing_frames + 2):
                            engine.advance(1 / 30)
                            _image, elapsed, _used = _frame(engine, size)
                            if index >= 2:
                                sink.append(elapsed)
                    _on_producer(gpu._release_point_graphics)
                    for pair in pairs:
                        pair.pop("diff", None)
                    passed = bool(pairs) and not refused and all(
                        "mean" in pair and pair["mean"] <= args.mean
                        and pair["p99"] <= args.p99
                        and pair["over_48"] <= args.outliers for pair in pairs)
                    ok = ok and passed
                    row = {"theme": theme, "background": background,
                           "size": list(size), "density": density,
                           "refused": refused, "pairs": pairs, "passed": passed,
                           "cpu_median_ms": round(statistics.median(cpu_ms), 3),
                           "gpu_median_ms": round(statistics.median(gpu_ms), 3),
                           "gpu_p90_ms": round(sorted(gpu_ms)[int(0.9 * (len(gpu_ms) - 1))], 3)}
                    rows.append(row)
                    print(json.dumps(row), flush=True)

    if args.images and worst:
        import numpy as np
        from PySide6.QtGui import QImage

        folder = args.output.parent / "images"
        folder.mkdir(parents=True, exist_ok=True)
        for theme, (_score, cpu_image, gpu_image, diff, background, size) in worst.items():
            stem = f"{theme}_{background.strip('#')}_{size[0]}x{size[1]}"
            cpu_image.save(str(folder / f"{stem}_cpu.png"))
            gpu_image.save(str(folder / f"{stem}_gpu.png"))
            if diff is not None:
                scaled = np.ascontiguousarray(np.clip(diff.astype(np.int32) * 4, 0, 255)
                                              .astype(np.uint8))
                QImage(scaled.data, scaled.shape[1], scaled.shape[0], scaled.strides[0],
                       QImage.Format_Grayscale8).save(str(folder / f"{stem}_diff_x4.png"))
    _on_producer.stop()
    try:
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        source = "unknown"
    report = {"schema": 1, "source_commit": source, "gl": gl_info,
              "software_renderer": software,
              "machine": platform.processor() or platform.machine(),
              "platform": platform.platform(), "python": sys.version.split()[0],
              "qt_platform": app.platformName(),
              "thresholds": {"mean": args.mean, "p99": args.p99,
                             "over_48": args.outliers},
              "qualification": ("Offscreen producer-thread frames compared with the "
                                "CPU path; no displayed frame rate or compositor "
                                "claim. Software-renderer timings are not GPU timings."),
              "all_passed": ok, "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
