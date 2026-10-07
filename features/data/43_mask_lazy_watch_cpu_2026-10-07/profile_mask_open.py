"""Measure a real fresh Mask open with the package's timing spans."""
from __future__ import annotations

import cProfile
import hashlib
import json
import os
import pstats
import time
from pathlib import Path


def main() -> None:
    from spacr.qt import timing

    timing.begin()
    from PySide6.QtWidgets import QApplication
    from spacr.qt.app import MainWindow
    import spacr.qt.app as app_module

    root = Path(os.environ["SPACR_PROOF_REPO"]).resolve()
    output = Path(os.environ["SPACR_PROOF_OUT_DIR"]).resolve()
    label = os.environ["SPACR_PROOF_LABEL"]
    assert Path(app_module.__file__).resolve() == root / "spacr/qt/app.py"
    output.mkdir(parents=True, exist_ok=True)
    sources = {}
    for relative in ("spacr/qt/app.py", "spacr/qt/screens/app_screen.py",
                     "spacr/qt/widgets/object_settings_grid.py"):
        sources[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()

    qapp = QApplication([])
    window = MainWindow()
    window.resize(1400, 900)
    window.show()
    for _ in range(5):
        qapp.processEvents()
    assert "mask" not in window._screens

    profiler = cProfile.Profile() if os.environ.get("SPACR_PROOF_PROFILE") else None
    started = time.perf_counter()
    timing.mark("proof mask open began")
    if profiler is not None:
        profiler.enable()
    window._on_nav_selected("mask")
    for _ in range(3):
        qapp.processEvents()
    if profiler is not None:
        profiler.disable()
    elapsed = time.perf_counter() - started
    timing.mark("proof mask open ended")
    assert window._stack.currentWidget() is window._screens["mask"]

    snapshot = timing.snapshot()
    summary = {"label": label, "pid": os.getpid(), "sources": sources,
               "elapsed_s": elapsed, "profiled": profiler is not None,
               "screen_present_before": False,
               "config_home": os.environ.get("XDG_CONFIG_HOME", ""),
               "snapshot": snapshot}
    (output / f"{label}.json").write_text(json.dumps(summary, indent=2))
    if profiler is not None:
        profiler.dump_stats(str(output / f"{label}.pstats"))
        pstats.Stats(profiler).sort_stats("cumulative").print_stats(45)
    print("MASK_PROOF", label, "elapsed_s", elapsed, "profiled", profiler is not None,
          "sources", sources, flush=True)
    window.close()
    window.deleteLater()
    qapp.processEvents()


if __name__ == "__main__":
    main()
