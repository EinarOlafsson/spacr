#!/usr/bin/env python3
"""Capture a native-4K Model Compare run on real tutorial fields."""
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
SOURCE = ROOT / "derived" / "cellpose_masks"
CUSTOM_MODEL = (
    ROOT
    / "derived"
    / "train_cellpose"
    / "models"
    / "cellpose_model"
    / "models"
    / "new_model_cpsam_e1_X256_Y256.CP_model"
)
LESSON = "21_model_compare"


def main() -> int:
    if not CUSTOM_MODEL.exists():
        raise FileNotFoundError(CUSTOM_MODEL)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-model-compare-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-model-compare-mpl")
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
    window._on_nav_selected("model_compare")
    wait_until(
        app,
        lambda: window._screens.get("model_compare") is not None,
        15.0,
        "Model Compare screen",
    )
    screen = window._screens["model_compare"]
    settle(app, 40)

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save(
        "01_overview",
        {
            "nav": nav_button(window, "model_compare"),
            "screen": screen,
            "source_controls": [screen._path_edit, screen._fields_box],
            "model_panels": [screen._panel_a, screen._panel_b],
        },
    )

    screen._fields_box.setValue(3)
    screen.set_source(str(SOURCE))
    wait_until(
        app,
        lambda: not screen._busy and len(screen._images) == 3,
        60.0,
        "three fields loaded",
    )
    settle(app, 25)
    capture.save(
        "02_fields_loaded",
        {
            "source_controls": [screen._path_edit, screen._fields_box],
            "status": screen._status,
        },
    )

    screen._panel_a.model_edit.setText("cpsam")
    screen._panel_a.diameter_box.setValue(30.0)
    screen._panel_b.model_edit.setText(str(CUSTOM_MODEL))
    screen._panel_b.diameter_box.setValue(30.0)
    settle(app, 20)
    capture.save(
        "03_models",
        {
            "model_a": screen._panel_a,
            "model_b": screen._panel_b,
            "model_panels": [screen._panel_a, screen._panel_b],
        },
    )
    capture.save(
        "04_compare",
        {
            "compare": screen._btn_compare,
            "model_panels": [screen._panel_a, screen._panel_b],
        },
    )
    screen._btn_compare.click()
    wait_until(app, lambda: screen._busy, 15.0, "comparison start")
    settle(app, 25)
    capture.save(
        "05_running",
        {
            "status": screen._status,
            "compare": screen._btn_compare,
        },
    )
    wait_until(
        app,
        lambda: not screen._busy and screen.report() is not None,
        240.0,
        "comparison completion",
    )
    settle(app, 40)
    capture.save(
        "06_parameters",
        {
            "parameters": screen._param_table,
            "warnings": screen._warnings,
        },
    )
    capture.save(
        "07_metrics",
        {
            "metrics": screen._row_table,
            "summary": screen._summary,
            "status": screen._status,
        },
    )
    capture.save(
        "08_field_one",
        {
            "previews": [screen._preview_a, screen._preview_b],
            "captions": [screen._caption_a, screen._caption_b],
            "metrics": screen._row_table,
        },
    )
    if screen._row_table.rowCount() > 1:
        screen._row_table.setCurrentCell(1, 0)
        screen.select_field(1)
        settle(app, 25)
    capture.save(
        "09_field_two",
        {
            "previews": [screen._preview_a, screen._preview_b],
            "captions": [screen._caption_a, screen._caption_b],
            "metrics": screen._row_table,
        },
    )
    capture.write_geometry()

    report = screen.report()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "model_compare",
        "source": str(SOURCE),
        "model_a": "cpsam",
        "model_b": str(CUSTOM_MODEL),
        "fields": list(screen.field_names()),
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "summary": report.summary if report is not None else "",
        "total_objects_a": report.total_objects_a if report is not None else None,
        "total_objects_b": report.total_objects_b if report is not None else None,
        "mean_ari": report.mean_ari if report is not None else None,
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    screen.close()
    settle(app, 30)
    window.close()
    settle(app, 30)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
