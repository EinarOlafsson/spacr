#!/usr/bin/env python3
"""Capture native-4K Annotator Agreement on the real tutorial database."""
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
DATABASE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials/measurements/measurements.db"
)
LESSON = "23_agreement"
COLUMNS = ("ER_Bulbs", "predictions_1")


def main() -> int:
    if not DATABASE.exists():
        raise FileNotFoundError(DATABASE)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-agreement-4k-config")
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
    window._on_nav_selected("agreement")
    wait_until(app, lambda: window._screens.get("agreement") is not None,
               15.0, "Agreement screen")
    screen = window._screens["agreement"]
    settle(app, 35)
    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)

    capture.save(
        "01_overview",
        {
            "nav": nav_button(window, "agreement"),
            "screen": screen,
            "source_controls": [screen._path_edit, screen._btn_open],
        },
    )
    if screen.set_database(str(DATABASE)) is False:
        raise RuntimeError(screen.last_error)
    screen.select_columns(COLUMNS)
    settle(app, 25)
    capture.save(
        "02_database",
        {
            "source_controls": [screen._path_edit, screen._btn_open],
            "status": screen._status,
        },
    )
    capture.save(
        "03_columns",
        {
            "columns": screen._columns_list,
            "compute": screen._btn_compute,
        },
    )
    capture.save(
        "04_compute",
        {
            "columns": screen._columns_list,
            "compute": screen._btn_compute,
        },
    )
    screen._btn_compute.click()
    settle(app, 2)
    capture.save(
        "05_running",
        {"status": screen._status, "compute": screen._btn_compute},
    )
    wait_until(app, lambda: not screen._busy and screen.report() is not None,
               120.0, "agreement completion")
    settle(app, 35)
    capture.save(
        "06_pairwise",
        {
            "pairwise": screen._kappa_table,
            "summary": screen._summary,
            "status": screen._status,
        },
    )
    capture.save(
        "07_confusion",
        {
            "pair_selector": screen._pair_combo,
            "confusion": screen._confusion_table,
        },
    )
    capture.save(
        "08_disagreements",
        {
            "review": screen._review_table,
            "review_label": screen._review_label,
            "limit": screen._limit_box,
        },
    )
    capture.save(
        "09_crop_one",
        {
            "review": screen._review_table,
            "crop": screen._crop_label,
            "caption": screen._crop_caption,
        },
    )
    if screen._review_table.rowCount() > 1:
        screen._review_table.setCurrentCell(1, 0)
        screen.select_disagreement(1)
        settle(app, 20)
    capture.save(
        "10_crop_two",
        {
            "review": screen._review_table,
            "crop": screen._crop_label,
            "caption": screen._crop_caption,
        },
    )
    capture.write_geometry()

    report = screen.report()
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "agreement",
        "database": str(DATABASE),
        "available_columns": screen.available_columns(),
        "selected_columns": screen.selected_columns(),
        "kappa_rows": screen.kappa_rows(),
        "confusion_rows": screen.confusion_rows(),
        "disagreement_rows": screen.disagreement_rows(),
        "overall_method": report.overall_method if report is not None else "",
        "overall_kappa": report.overall_kappa if report is not None else None,
        "percent_agreement": report.percent_agreement if report is not None else None,
        "n_complete": report.n_complete if report is not None else None,
        "n_disagreements": report.n_disagreements if report is not None else None,
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
