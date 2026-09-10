#!/usr/bin/env python3
"""Capture a native-4K Model Zoo scan, benchmark, and compare hand-off."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
MODEL_ROOT = ROOT / "derived" / "train_cellpose" / "models"
FIELDS = ROOT / "derived" / "cellpose_masks"
LESSON = "22_model_zoo"


def main() -> int:
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-model-zoo-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-model-zoo-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.preferences import apply_preferences_to_app

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")
    window._on_nav_selected("model_zoo")
    wait_until(app, lambda: window._screens.get("model_zoo") is not None,
               15.0, "Model Zoo screen")
    screen = window._screens["model_zoo"]
    settle(app, 40)
    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)

    capture.save(
        "01_overview",
        {
            "nav": nav_button(window, "model_zoo"),
            "screen": screen,
            "scan_controls": [screen._scan_edit, screen._btn_scan],
            "catalogue": screen._table,
        },
    )
    screen.scan(str(MODEL_ROOT), include_catalogue=True)
    wait_until(app, lambda: not screen._busy and len(screen.entries()) >= 2,
               90.0, "model scan")
    settle(app, 35)
    capture.save(
        "02_scanned",
        {
            "scan_controls": [screen._scan_edit, screen._btn_scan],
            "scan": screen._btn_scan,
            "catalogue": screen._table,
            "status": screen._status,
        },
    )

    local_rows = [
        index for index, entry in enumerate(screen.entries())
        if entry.exists and str(entry.path).lower().endswith((".cp_model", ".cpmodel"))
        and str(MODEL_ROOT) in str(entry.path)
    ]
    if len(local_rows) < 2:
        raise RuntimeError(f"expected two local tutorial checkpoints, got {local_rows}")
    screen.select(local_rows[0])
    settle(app, 20)
    capture.save(
        "03_provenance",
        {
            "catalogue": screen._table,
            "detail": screen._detail,
            "download": [screen._dest_edit, screen._btn_download,
                         screen._allow_unverified, screen._progress],
        },
    )

    screen._fields_box.setValue(3)
    screen.set_fields_source(str(FIELDS))
    wait_until(app, lambda: not screen._busy and len(screen._images) == 3,
               60.0, "benchmark fields")
    settle(app, 25)
    capture.save(
        "04_fields",
        {
            "field_controls": [screen._fields_box, screen._fields_edit,
                               screen._btn_test, screen._btn_compare],
            "status": screen._status,
        },
    )
    capture.save(
        "05_test",
        {
            "test": screen._btn_test,
            "field_controls": [screen._fields_box, screen._fields_edit,
                               screen._btn_test],
        },
    )
    screen._btn_test.click()
    wait_until(app, lambda: screen._busy, 15.0, "benchmark start")
    settle(app, 20)
    capture.save(
        "06_running",
        {"status": screen._status, "test": screen._btn_test},
    )
    wait_until(app, lambda: not screen._busy and screen.result() is not None,
               240.0, "benchmark completion")
    settle(app, 35)
    capture.save(
        "07_benchmark",
        {
            "benchmark": screen._bench_table,
            "preview": screen._preview,
            "summary": screen._summary,
        },
    )
    if screen._bench_table.rowCount() > 1:
        screen._bench_table.setCurrentCell(1, 0)
        screen.select_field(1)
        settle(app, 20)
    capture.save(
        "08_field_two",
        {"benchmark": screen._bench_table, "preview": screen._preview},
    )

    screen.select(local_rows[0], local_rows[1])
    settle(app, 20)
    capture.save(
        "09_two_selected",
        {
            "catalogue": screen._table,
            "detail": screen._detail,
            "compare": screen._btn_compare,
        },
    )
    screen._btn_compare.click()
    wait_until(
        app,
        lambda: window._screens.get("model_compare") is not None
        and window._stack.currentWidget() is window._screens.get("model_compare"),
        30.0,
        "Model Compare hand-off",
    )
    comparison = window._screens["model_compare"]
    wait_until(app, lambda: not comparison._busy and len(comparison._images) == 3,
               60.0, "comparison fields loaded")
    settle(app, 30)
    capture.save(
        "10_compare_handoff",
        {
            "comparison": comparison,
            "source_controls": [comparison._path_edit, comparison._fields_box],
            "model_panels": [comparison._panel_a, comparison._panel_b],
        },
    )
    capture.write_geometry()

    result = screen.result()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "model_zoo",
        "model_root": str(MODEL_ROOT),
        "fields": str(FIELDS),
        "entries": [entry.describe() for entry in screen.entries()],
        "local_rows": local_rows,
        "benchmark_summary": result.summary if result is not None else "",
        "benchmark_rows": screen.benchmark_rows(),
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    window.close()
    settle(app, 30)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
