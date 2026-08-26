#!/usr/bin/env python3
"""Measure what spaCR costs on THIS machine, and print it.

Run this on the machine that feels slow and send the output back. It is
read-only: it builds screens, times them, and writes nothing.

    python tools/spacr_hardware_report.py

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


def main() -> int:
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
    print("Send this whole output back. Nothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
