#!/usr/bin/env python3
"""Measure what spaCR costs on THIS machine, and print it.

Run this on the machine that feels slow and send the result back.

    python tools/spacr_hardware_report.py

It prints the report AND saves a copy next to it, so a long report can be
attached instead of copied out of a terminal. The saved path is printed on
the last line.

    python tools/spacr_hardware_report.py --out somewhere/else.txt

IT READS THE MACHINE AND NOTHING ELSE. It opens no project, touches no
data, and the only file it writes is its own report.

WHY A SCRIPT AND NOT A GUESS. The same code is fast on one machine and
unusable on another, and the difference is never where it is assumed to be.
This times each stage separately -- the imports, the Qt application, each
screen -- so the answer is a number rather than a theory.
"""
from __future__ import annotations

import os
import platform
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "")


def _t():
    return time.perf_counter()


class _Tee:
    """Write to the terminal and collect the same text for the file.

    Holds its OWN reference to the builtin. The module swaps its global
    ``print`` for one of these while the report runs, so a tee that reached
    for ``print`` by name would call itself.
    """

    _write = staticmethod(__builtins__["print"]
                          if isinstance(__builtins__, dict)
                          else __builtins__.print)

    def __init__(self):
        self.lines: list = []

    def __call__(self, text: str = "") -> None:
        _Tee._write(text)
        self.lines.append(str(text))


def _report_path(argv) -> "os.PathLike | str":
    """Where the copy goes: --out if given, else beside the user's home."""
    if "--out" in argv:
        return argv[argv.index("--out") + 1]
    import datetime
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    folder = os.path.join(os.path.expanduser("~"), ".spacr", "reports")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"hardware-{stamp}.txt")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    say = _Tee()
    global print  # noqa: PLW0603 - the body below prints through the tee
    _real_print, print = print, say
    try:
        return _run(argv, say)
    finally:
        print = _real_print


def _run(argv, say) -> int:
    print("=" * 68)
    print("spaCR hardware report")
    print("=" * 68)
    print(f"python      {sys.version.split()[0]}")
    print(f"platform    {platform.platform()}")
    print(f"machine     {platform.machine()}")
    print(f"processor   {platform.processor() or '(not reported)'}")
    try:
        import multiprocessing
        print(f"cpu count   {multiprocessing.cpu_count()}")
    except Exception:
        pass
    try:
        import psutil
        print(f"memory      {psutil.virtual_memory().total / 2**30:.1f} GiB")
    except Exception:
        pass

    # Rosetta / emulation is the single most likely cause of a Mac being
    # many times slower than it should be, and it is invisible unless asked.
    if platform.system() == "Darwin":
        try:
            import subprocess
            out = subprocess.run(
                ["sysctl", "-n", "sysctl.proc_translated"],
                capture_output=True, text=True, timeout=5).stdout.strip()
            print(f"translated  {out} "
                  f"({'RUNNING UNDER ROSETTA' if out == '1' else 'native'})")
        except Exception:
            print("translated  (could not ask)")

    print()
    print("-- imports " + "-" * 56)
    for name in ("numpy", "pandas", "PySide6.QtWidgets", "torch",
                 "sklearn", "statsmodels", "scipy", "matplotlib"):
        t0 = _t()
        try:
            __import__(name)
            print(f"  {name:24} {_t() - t0:7.2f} s")
        except Exception as exc:                              # noqa: BLE001
            print(f"  {name:24} not installed ({type(exc).__name__})")

    # The numeric library underneath everything. A build without an
    # accelerated BLAS is the other classic reason for a slow machine.
    try:
        import numpy as np
        cfg = getattr(np, "__config__", None)
        blas = ""
        if cfg is not None and hasattr(cfg, "show"):
            import contextlib, io
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                cfg.show()
            text = buf.getvalue().lower()
            for candidate in ("accelerate", "openblas", "mkl", "blis"):
                if candidate in text:
                    blas = candidate
                    break
        print(f"  numpy BLAS               {blas or '(not reported)'}")
        t0 = _t()
        a = np.random.rand(1200, 1200)
        a @ a
        print(f"  1200x1200 matmul         {_t() - t0:7.2f} s")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  numpy benchmark failed: {exc}")

    print()
    print("-- the application " + "-" * 48)
    t0 = _t()
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    print(f"  QApplication             {_t() - t0:7.2f} s")

    t0 = _t()
    from spacr.qt.screens.app_screen import AppScreen
    print(f"  import AppScreen         {_t() - t0:7.2f} s")

    for key in ("regression", "mask", "measure", "classify_merged"):
        t0 = _t()
        try:
            screen = AppScreen(key)
            app.processEvents()
            rows = len(screen._settings_model.collect())
            took = _t() - t0
            per = (took / rows * 1000) if rows else 0.0
            print(f"  build {key:18} {took:7.2f} s  "
                  f"({rows} rows, {per:.1f} ms/row)")
        except Exception as exc:                              # noqa: BLE001
            print(f"  build {key:18} FAILED ({type(exc).__name__}: {exc})")

    print()
    print("-- the backdrop " + "-" * 51)
    try:
        from spacr.qt.widgets import ambient
        from spacr.qt import theme
        page = theme.page_colour("dark")
        for name in (getattr(ambient, "SPACEOUT_THEME", None), "glow"):
            if not name:
                continue
            try:
                engine = ambient.make_engine(
                    name, getattr(ambient, "SPACEOUT_PALETTE", None) or "",
                    page, seed=1)
                t0 = _t()
                for _ in range(20):
                    engine.advance(1 / 24)
                print(f"  {str(name):24} {(_t() - t0) / 20 * 1000:7.2f} ms/frame")
            except Exception as exc:                          # noqa: BLE001
                print(f"  {str(name):24} failed ({type(exc).__name__})")
    except Exception as exc:                                  # noqa: BLE001
        print(f"  backdrop check failed: {exc}")

    print()
    try:
        path = _report_path(argv)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(say.lines) + "\n")
        print(f"Saved to {path}")
        print("Send that file back, or paste the text above.")
    except Exception as exc:                                  # noqa: BLE001
        print(f"(could not save a copy: {exc})")
        print("Paste the text above instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
