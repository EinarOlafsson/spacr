#!/usr/bin/env python3
"""Capture Plate Viewer against the transferred experiment at native 4K."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
DATABASE = DATASET / "measurements" / "measurements.db"
GENERATED = ROOT / "generated" / "plate_viewer"
EXPORT = GENERATED / "plate1_cell_counts_min350.csv"
LESSON = "33_plate_viewer"
MEASUREMENT = "cell_channel_1_mean_intensity"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report(report) -> dict:
    row = report.gradient("row")
    column = report.gradient("column")
    return {
        "plate": report.plate,
        "n_wells": int(report.n_wells),
        "n_dropped_min_count": int(report.n_dropped_min_count),
        "edge_detected": bool(report.edge_detected),
        "pct_difference": (float(report.pct_difference)
                           if report.pct_difference is not None else None),
        "cliffs_delta": (float(report.cliffs_delta)
                         if report.cliffs_delta is not None else None),
        "p_value": (float(report.p_value)
                    if report.p_value is not None else None),
        "row_rho": (float(row.spearman_rho)
                    if row and row.spearman_rho is not None else None),
        "column_rho": (float(column.spearman_rho)
                       if column and column.spearman_rho is not None else None),
        "summary": report.summary,
    }


def _wait_idle(app, screen, label: str, timeout: float = 180.0) -> None:
    wait_until(app, lambda: not screen.is_busy(), timeout, label)
    wait_until(app, lambda: screen.active_jobs() == 0, 30.0,
               f"{label} worker retirement")


def _recompute(app, screen, label: str) -> None:
    screen._recompute_timer.stop()
    screen.recompute()
    _wait_idle(app, screen, label)


def main() -> int:
    if not DATABASE.is_file():
        raise FileNotFoundError(DATABASE)
    if GENERATED.exists():
        raise FileExistsError(
            f"tutorial output already exists; remove only {GENERATED} to rerun"
        )

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-plate-viewer-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-plate-viewer-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog
    from spacr.qt.dnd_handlers import ResultsDatabaseDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None

    before_hash = _sha256(DATABASE)
    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    window._on_nav_selected("plate_view")
    wait_until(app, lambda: window._screens.get("plate_view") is not None,
               15.0, "Plate Viewer screen")
    screen = window._screens["plate_view"]
    settle(app, 40)
    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav_button(window, "plate_view"),
        "screen": screen,
        "grid": screen._grid,
        "report": screen._report_view,
    })

    ResultsDatabaseDropHandler().apply(DATABASE, screen)
    _wait_idle(app, screen, "database metadata")
    wait_until(app, lambda: screen._value_combo.count() > 0,
               30.0, "numeric measurement columns")
    capture.save("02_database", {
        "source": [screen._path_edit, screen._btn_pick_db,
                   screen._btn_pick_src, screen._btn_open],
        "path": screen._path_edit,
    })

    screen.set_table("cell")
    _wait_idle(app, screen, "cell table metadata")
    screen.set_value_column(MEASUREMENT)
    screen._loading = True
    screen._grouping_combo.setCurrentText("mean")
    screen._scale_combo.setCurrentIndex(0)
    screen._min_count_box.setValue(100)
    screen._loading = False
    settle(app, 30)
    capture.save("03_measurement", {
        "selection": [screen._table_combo, screen._value_combo,
                      screen._plate_combo],
        "measurement": screen._value_combo,
    })
    capture.save("04_options", {
        "options": [screen._grouping_combo, screen._scale_combo,
                    screen._min_count_box],
        "render": screen._btn_render,
    })
    capture.save("05_render", {
        "render_controls": [screen._grouping_combo, screen._scale_combo,
                            screen._min_count_box, screen._btn_render],
        "render": screen._btn_render,
    })
    screen._btn_render.click()
    wait_until(app, lambda: screen._layout_df is not None,
               180.0, "real plate layout")
    _wait_idle(app, screen, "real plate render")
    if len(screen._layout_df) != 196 or screen._report is None:
        raise RuntimeError("expected all 196 measured wells in the first render")
    mean_report = _report(screen._report)
    settle(app, 35)
    capture.save("06_mean_result", {
        "result": [screen._grid, screen._report_view],
        "grid": screen._grid,
        "report": screen._report_view,
        "status": screen._status,
    })

    first_point = screen._grid.cell_rect(7, 7).center().toPoint()
    QTest.mouseClick(screen._grid, Qt.LeftButton, Qt.NoModifier, first_point)
    settle(app, 20)
    capture.save("07_well_detail", {
        "inspection": [screen._grid, screen._report_view,
                       screen._well_label],
        "grid": screen._grid,
        "well_detail": screen._well_label,
    })
    first_global = screen._grid.mapTo(window, first_point)
    capture.frames["07_well_detail"]["well_click"] = [
        int(first_global.x() - 12), int(first_global.y() - 12), 24, 24
    ]

    screen._loading = True
    screen._grouping_combo.setCurrentText("count")
    screen._loading = False
    settle(app, 20)
    capture.save("08_count_setting", {
        "grouping": screen._grouping_combo,
        "options": [screen._grouping_combo, screen._scale_combo,
                    screen._min_count_box],
    })
    _recompute(app, screen, "count aggregation")
    count_report = _report(screen._report)
    if not screen._report.edge_detected:
        raise RuntimeError("the real count view should expose its edge effect")
    settle(app, 30)
    capture.save("09_count_result", {
        "result": [screen._grid, screen._report_view],
        "grid": screen._grid,
        "report": screen._report_view,
        "status": screen._status,
    })

    screen._loading = True
    screen._min_count_box.setValue(350)
    screen._loading = False
    settle(app, 20)
    capture.save("10_minimum_setting", {
        "minimum": screen._min_count_box,
        "options": [screen._grouping_combo, screen._scale_combo,
                    screen._min_count_box],
    })
    _recompute(app, screen, "minimum-object filter")
    filtered_report = _report(screen._report)
    if len(screen._layout_df) != 168 or screen._report.n_dropped_min_count != 28:
        raise RuntimeError(
            f"expected 168 wells and 28 drops, got {len(screen._layout_df)} "
            f"and {screen._report.n_dropped_min_count}"
        )
    settle(app, 30)
    capture.save("11_filtered_result", {
        "result": [screen._grid, screen._report_view],
        "grid": screen._grid,
        "report": screen._report_view,
        "status": screen._status,
    })

    dropped_point = screen._grid.cell_rect(3, 14).center().toPoint()
    QTest.mouseClick(screen._grid, Qt.LeftButton, Qt.NoModifier, dropped_point)
    settle(app, 20)
    if "blank" not in screen.well_info_text():
        raise RuntimeError(f"expected C14 to be blank: {screen.well_info_text()}")
    capture.save("12_blank_well", {
        "inspection": [screen._grid, screen._report_view,
                       screen._well_label],
        "grid": screen._grid,
        "well_detail": screen._well_label,
    })
    dropped_global = screen._grid.mapTo(window, dropped_point)
    capture.frames["12_blank_well"]["well_click"] = [
        int(dropped_global.x() - 12), int(dropped_global.y() - 12), 24, 24
    ]

    GENERATED.mkdir(parents=True)
    capture.save("13_export", {
        "footer": [screen._scale_label, screen._btn_export],
        "export": screen._btn_export,
    })
    original_dialog = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *args, **kwargs: (str(EXPORT), "CSV files (*.csv)"))
    try:
        QTest.mouseClick(screen._btn_export, Qt.LeftButton)
    finally:
        QFileDialog.getSaveFileName = original_dialog
    wait_until(app, EXPORT.is_file, 15.0, "well-grid CSV export")
    settle(app, 25)
    capture.save("14_exported", {
        "footer": [screen._scale_label, screen._btn_export],
        "status": screen._status,
    })
    capture.write_geometry()

    after_hash = _sha256(DATABASE)
    if before_hash != after_hash:
        raise RuntimeError("Plate Viewer modified its source database")
    rows = sum(1 for _ in EXPORT.open()) - 1
    if rows != 168:
        raise RuntimeError(f"expected 168 exported rows, got {rows}")
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "plate_view",
        "source": str(DATABASE),
        "source_sha256_before": before_hash,
        "source_sha256_after": after_hash,
        "table": "cell",
        "measurement": MEASUREMENT,
        "initial": {"grouping": "mean", "min_count": 100,
                    "report": mean_report},
        "count": {"grouping": "count", "min_count": 100,
                  "report": count_report},
        "filtered": {"grouping": "count", "min_count": 350,
                     "layout_rows": 168, "report": filtered_report},
        "export": {"path": str(EXPORT), "rows": rows},
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    screen.close()
    settle(app, 20)
    window.close()
    settle(app, 20)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
