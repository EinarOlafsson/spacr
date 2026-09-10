#!/usr/bin/env python3
"""Capture a real Align & Stitch plan and write at native 4K."""
from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DERIVED = ROOT / "derived" / "align_stitch"
TILES = DERIVED / "tiles"
SOURCE_MANIFEST = DERIVED / "source_manifest.json"
GENERATED = ROOT / "generated" / "align_stitch"
STACKS = GENERATED / "stacks"
DATABASE = GENERATED / "measurements.db"
LESSON = "32_align_stitch"


def main() -> int:
    if not SOURCE_MANIFEST.is_file() or len(list(TILES.glob("*.npy"))) != 9:
        raise FileNotFoundError(
            "run prepare_align_stitch_tutorial_data.py before capture"
        )
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
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-align-stitch-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-align-stitch-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from spacr.align import read_coordinates
    from spacr.qt.dnd_handlers import AlignDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    window._on_nav_selected("align")
    wait_until(app, lambda: window._screens.get("align") is not None,
               15.0, "Align & Stitch screen")
    screen = window._screens["align"]
    settle(app, 40)
    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)

    capture.save("01_overview", {
        "nav": nav_button(window, "align"),
        "screen": screen,
        "source": screen._src_edit,
        "layout": screen._layout_view,
        "report": screen._report_view,
    })

    AlignDropHandler().apply(TILES, screen)
    screen.apply_settings({
        "src": str(TILES),
        "dst": str(STACKS),
        "db_path": str(DATABASE),
        "grid": (3, 3),
        "overlap": 0.25,
        "order": "row-major",
        "reference_channel": 1,
        "min_confidence": 0.30,
        "neighbour_radius": 1,
        "blend": "feather",
        "max_buffer_bytes": 64 << 20,
        "overwrite": False,
    })
    settle(app, 30)
    capture.save("02_source", {
        "source": screen._src_edit,
        "source_row": [screen._src_edit, screen._btn_pick_src],
    })
    capture.save("03_layout_settings", {
        "grid": [screen._rows_box, screen._cols_box],
        "layout_settings": [screen._rows_box, screen._cols_box,
                            screen._overlap_box, screen._order_combo,
                            screen._ref_box],
    })
    capture.save("04_quality_settings", {
        "quality": [screen._conf_box, screen._radius_box,
                    screen._blend_combo, screen._budget_box],
        "plan": screen._btn_plan,
    })
    capture.save("05_plan_run", {
        "quality": [screen._conf_box, screen._radius_box,
                    screen._blend_combo, screen._budget_box],
        "plan_controls": [screen._conf_box, screen._radius_box,
                          screen._blend_combo, screen._budget_box,
                          screen._btn_plan],
        "plan": screen._btn_plan,
    })
    screen._btn_plan.click()
    wait_until(app, lambda: screen.plan() is not None and not screen.is_busy(),
               180.0, "registration plan")
    wait_until(app, lambda: screen.active_jobs() == 0,
               30.0, "plan worker retirement")
    plan = screen.plan()
    if plan is None or len(plan.placements) != 9 or plan.n_registered != 9:
        raise RuntimeError(
            f"unexpected plan: placements={len(plan.placements) if plan else 0} "
            f"registered={plan.n_registered if plan else 0}"
        )
    settle(app, 35)
    capture.save("06_plan_result", {
        "layout": screen._layout_view,
        "report": screen._report_view,
        "plan_result": [screen._layout_view, screen._report_view],
        "status": screen._status,
    })

    tile_index, tile_rect = screen._layout_view.tile_rects()[4]
    tile_point = tile_rect.center().toPoint()
    QTest.mouseClick(screen._layout_view, Qt.LeftButton, Qt.NoModifier, tile_point)
    settle(app, 20)
    capture.save("07_tile_detail", {
        "layout": screen._layout_view,
        "report": screen._report_view,
        "inspection": [screen._layout_view, screen._report_view,
                       screen._tile_label],
        "tile_detail": screen._tile_label,
    })
    # Record the exact clicked tile, not the centre of the whole layout.
    local = screen._layout_view.mapTo(window, tile_point)
    capture.frames["07_tile_detail"]["tile_click"] = [
        int(local.x() - 12), int(local.y() - 12), 24, 24
    ]

    capture.save("08_output_settings", {
        "output": [screen._dst_edit, screen._btn_pick_dst, screen._db_edit,
                   screen._overwrite_box, screen._btn_write],
        "write": screen._btn_write,
    })
    capture.save("09_write_run", {
        "output": [screen._dst_edit, screen._btn_pick_dst, screen._db_edit,
                   screen._overwrite_box, screen._btn_write],
        "write": screen._btn_write,
    })
    screen._btn_write.click()
    wait_until(app, lambda: screen.result() is not None and not screen.is_busy(),
               240.0, "incremental stack write")
    wait_until(app, lambda: screen.active_jobs() == 0,
               30.0, "write worker retirement")
    result = screen.result()
    if result is None or result.n_written != 9 or result.n_skipped:
        raise RuntimeError(f"unexpected write result: {result}")
    if not Path(result.stack_path).is_file() or not DATABASE.is_file():
        raise RuntimeError("stitched stack or coordinate database is missing")
    coordinates = read_coordinates(DATABASE)
    if len(coordinates) != 9:
        raise RuntimeError(f"expected 9 coordinate rows, got {len(coordinates)}")
    settle(app, 35)
    capture.save("10_finished", {
        "layout": screen._layout_view,
        "report": screen._report_view,
        "status": screen._status,
        "output": [screen._dst_edit, screen._db_edit],
    })
    capture.write_geometry()

    with sqlite3.connect(DATABASE) as connection:
        tables = sorted(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ))
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "align",
        "source_manifest": json.loads(SOURCE_MANIFEST.read_text()),
        "plan": {
            "tiles": len(plan.placements),
            "registered": plan.n_registered,
            "nominal": plan.n_nominal,
            "unplaced": len(plan.unplaced),
            "canvas_shape": list(plan.canvas_shape),
            "dtype": str(plan.dtype),
            "max_residual": plan.max_residual,
        },
        "result": {
            "stack_path": result.stack_path,
            "stack_shape": list(result.canvas.shape),
            "tiles_written": result.n_written,
            "tiles_skipped": result.n_skipped,
            "peak_buffer_bytes": result.peak_buffer_bytes,
            "band_rows": result.band_rows,
            "writer": result.writer,
            "database": str(DATABASE),
            "coordinate_rows": len(coordinates),
            "tables": tables,
        },
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
