"""Measure the spaceout backdrop's frame time, windowed against fullscreen.

Instruction 327 (1) is a report that "the fullscreen mode fractal is super
choppy while the normal background one is super smoothe", and it asks for
the difference to be MEASURED before anything is changed -- and for both
surfaces to be measured again afterwards, because a change that makes
fullscreen better and the backdrop worse is a bad trade.

KEPT AS A SCRIPT for the same reason `find_mandelbrot_regions.py` is: the
numbers in the instruction file are only worth what re-running them is,
and "it feels smoother now" is what this replaces.

    python tools/measure_the_backdrop.py                # everything it can
    python tools/measure_the_backdrop.py --no-gpu       # skip the GL part

Four measurements, cheapest first:

  * WHAT IS SHADED. `target_render_size` for each surface, which is where
    the area appears as a number rather than as an impression.
  * THE CPU KERNEL. Median render time per surface, at the adaptive
    scale's top and at its 0.58 floor, so what the adaptive loop buys is
    visible next to what it costs.
  * THE WHOLE PIPELINE. The real widget, frames actually delivered over a
    few seconds -- worker thread, QImage conversion and paint included --
    because a shader time is not a frame rate.
  * THE GPU SHADERS. Each pattern into an offscreen framebuffer, so
    nothing appears on the desktop and neither vsync nor the compositor
    is in the number.

The GPU part needs a real GL context. Under Wayland Qt hands out an
OpenGL ES context that these GLSL 120 shaders will not compile on, which
is why `spacr.qt` asks for XWayland at launch; run this with
``QT_QPA_PLATFORM=xcb`` for the same reason.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

#: The surfaces worth comparing: a backdrop panel behind a settled window,
#: a window, and the two fullscreen sizes people actually have.
SURFACES = (
    ("backdrop panel 900x600", 900, 600),
    ("backdrop panel 1280x800", 1280, 800),
    ("windowed 1600x1000", 1600, 1000),
    ("fullscreen 2560x1440", 2560, 1440),
    ("fullscreen 3840x2160", 3840, 2160),
)

#: The CPU orbit fold's own numbers, from `_make_cpu_widget`. Repeated
#: here rather than imported because they live inside a closure that needs
#: a widget: keep them in step with that builder.
ORBIT_BASE_PIXELS = 460_000.0
ORBIT_ITERATIONS = 5
ORBIT_TARGET_FPS = 30

#: The bottom of the adaptive range, from `_adapt_resolution`.
ADAPTIVE_FLOOR = 0.58


def _headline(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def what_is_shaded(scale: float) -> None:
    """Pixels shaded per surface, which is where the area becomes a number."""
    from spacr.qt.widgets.fractal_travel import target_render_size

    _headline(f"What is shaded, at render scale {scale}")
    base = ORBIT_BASE_PIXELS * scale * scale
    panel = None
    for label, width, height in SURFACES:
        shaded = target_render_size(width, height, 1.0, scale, base, 1.0)
        pixels = shaded[0] * shaded[1]
        panel = pixels if panel is None else panel
        print(f"  {label:26s} {shaded[0]:5d}x{shaded[1]:<5d} "
              f"{pixels:9,d} px   {pixels / panel:5.2f}x the panel")


def the_cpu_kernel(threads: int, frames: int = 12) -> None:
    """Median render time per surface, at the top and floor of the loop."""
    from spacr.qt.widgets.fractal_travel import (OrbitEngine,
                                                 target_render_size)

    budget = 1000.0 / ORBIT_TARGET_FPS
    for adaptive in (1.0, ADAPTIVE_FLOOR):
        _headline(f"The CPU orbit fold, adaptive scale {adaptive:.2f} "
                  f"({threads} threads, budget {budget:.1f} ms)")
        for label, width, height in SURFACES:
            shaded = target_render_size(width, height, 1.0, 1.0,
                                        ORBIT_BASE_PIXELS, adaptive)
            engine = OrbitEngine(threads)
            # The first frame allocates the ring and compiles; it is not a
            # frame rate and does not belong in the median.
            engine.render(shaded[0], shaded[1], 0.0, 1.0, 1.5,
                          ORBIT_ITERATIONS)
            times = []
            for index in range(frames):
                started = time.perf_counter()
                engine.render(shaded[0], shaded[1], 0.05 * index, 1.0, 1.5,
                              ORBIT_ITERATIONS)
                times.append(time.perf_counter() - started)
            ms = 1000.0 * statistics.median(times)
            print(f"  {label:26s} {shaded[0] * shaded[1]:9,d} px  "
                  f"{ms:7.2f} ms  {1000.0 / ms:6.1f} fps  "
                  f"{'OVER' if ms > budget else 'within'} budget")


def the_whole_pipeline(seconds: float = 8.0) -> None:
    """Frames the real widget actually delivers, worker and paint included.

    FRAMES PAINTED, not the timer interval: a timer that asks for sixty
    frames a second on a machine that can shade twenty-four still says
    sixty. The adaptive loop is disabled for one run of each size so the
    trade it makes is visible rather than asserted.
    """
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication

    from spacr.qt.widgets.fractal_travel import (RuntimeControls, Settings,
                                                 create_fractal_widget)

    application = QApplication.instance() or QApplication(sys.argv)
    _headline(f"The whole pipeline, {seconds:.0f}s per run")
    for label, width, height in SURFACES[:1] + SURFACES[-2:-1]:
        for adapting in (False, True):
            widget = create_fractal_widget(
                Settings(pattern="orbit", backend="cpu"), RuntimeControls())
            widget.resize(width, height)
            widget.show()
            if not adapting:
                widget._adapt_resolution = lambda: None
            started = time.perf_counter()
            QTimer.singleShot(int(1000 * seconds), application.quit)
            application.exec()
            elapsed = time.perf_counter() - started
            shaded = widget._render_size
            print(f"  {label:26s} "
                  f"{'adaptive' if adapting else 'held at 1.0'}: "
                  f"{widget._frames / elapsed:5.1f} fps, shading "
                  f"{shaded[0]}x{shaded[1]}, "
                  f"{1000.0 * (widget._render_ema or 0.0):5.1f} ms a frame")
            widget.shutdown()
            widget.deleteLater()
            application.processEvents()


def the_gpu_shaders(frames: int = 24) -> None:
    """Each GPU pattern into an offscreen framebuffer, at every surface."""
    import numpy as np
    from vispy import app as vispy_app, gloo

    vispy_app.use_app("pyside6")
    from vispy.app import Canvas

    from spacr.qt.widgets import (fractal_cascade, fractal_orbit_gpu,
                                  fractal_space, fractal_travel)

    # THE DETAIL EACH ONE IS BUILT WITH at `balanced` quality, from
    # `_make_gpu_widget`. It is the shader's iteration count where the
    # shader has one, and leaving it at its unset zero would time a loop
    # that never runs -- which is how this script first reported the
    # cascade at twenty thousand frames a second.
    patterns = (("orbit_gpu", fractal_orbit_gpu.FRAGMENT_SHADER, 4),
                ("cascade", fractal_cascade.FRAGMENT_SHADER, 5),
                ("space", fractal_space.FRAGMENT_SHADER, 4),
                ("orbit (shared)", fractal_travel.FRAGMENT_SHADER, 6))
    # NOT SHOWN. A canvas with no window still has a context, and drawing
    # into a framebuffer of the size being measured keeps the desktop out
    # of it -- both the compositor's cost and the user's screen.
    canvas = Canvas(keys=None, size=(64, 64), show=False)
    with canvas:
        renderer = gloo.gl.glGetParameter(gloo.gl.GL_RENDERER)
        for name, source, detail in patterns:
            _headline(f"The GPU {name} shader on {renderer}")
            program = gloo.Program(fractal_travel.VERTEX_SHADER, source)
            program["a_position"] = np.asarray(
                [(-1, -1), (1, -1), (-1, 1), (1, 1)], dtype=np.float32)
            for uniform, value in (("u_time", 3.0), ("u_speed", 1.0),
                                   ("u_dream", 1.5), ("u_pointer_x", 0.2),
                                   ("u_pointer_y", 0.1), ("u_pull", 1.0),
                                   ("u_push", 0.0)):
                try:
                    program[uniform] = np.float32(value)
                except KeyError:
                    pass                # this pattern does not declare it
            try:
                program["u_detail"] = np.int32(detail)
            except KeyError:
                pass
            for label, width, height in SURFACES:
                target = gloo.FrameBuffer(
                    color=gloo.Texture2D(shape=(height, width, 4),
                                         format="rgba",
                                         internalformat="rgba8"))
                program["u_resolution"] = (float(width), float(height))
                with target:
                    gloo.set_viewport(0, 0, width, height)
                    for _ in range(3):
                        program.draw("triangle_strip")
                    gloo.gl.glFinish()
                    times = []
                    for _ in range(frames):
                        started = time.perf_counter()
                        program.draw("triangle_strip")
                        # THE FENCE IS THE MEASUREMENT. `draw` only queues
                        # the work; without this the number is how fast the
                        # driver accepts commands.
                        gloo.gl.glFinish()
                        times.append(time.perf_counter() - started)
                ms = 1000.0 * statistics.median(times)
                print(f"  {label:26s} {width * height:9,d} px  "
                      f"{ms:7.2f} ms  {1000.0 / ms:7.1f} fps")
                del target


def main(argv=None) -> int:
    """Run the measurements and print them. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=float, default=1.0,
                        help="render scale to size the CPU frames with")
    parser.add_argument("--seconds", type=float, default=8.0,
                        help="seconds per end-to-end run")
    parser.add_argument("--no-gpu", action="store_true",
                        help="skip the part that needs a GL context")
    parser.add_argument("--no-pipeline", action="store_true",
                        help="skip the part that builds a real widget")
    arguments = parser.parse_args(argv)

    from spacr.qt.widgets.fractal_travel import (HardwareProfile, Settings,
                                                 resolved_cpu_threads)

    hardware = HardwareProfile.detect()
    threads = resolved_cpu_threads(Settings().validated(), hardware)
    print(f"{hardware.logical_cpus} logical CPUs, {threads} render threads")

    what_is_shaded(arguments.scale)
    the_cpu_kernel(threads)
    if not arguments.no_pipeline:
        the_whole_pipeline(arguments.seconds)
    if not arguments.no_gpu:
        try:
            the_gpu_shaders()
        except Exception as error:                           # noqa: BLE001
            # A machine with no usable GL context is the ordinary case for
            # half the reports this script exists to answer, so it is a
            # line of output rather than a traceback.
            print(f"\nNo GPU measurement: {type(error).__name__}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
