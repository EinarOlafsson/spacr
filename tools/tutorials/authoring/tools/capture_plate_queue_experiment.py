#!/usr/bin/env python3
"""Capture Plate Queue with real settings and a non-destructive runner."""
from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle, wait_until


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
SOURCE = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
ORIG = SOURCE / "orig"
MASK_SETTINGS = ORIG / "settings" / "gen_mask_settings.csv"
QUEUE_PATH = ROOT / "generated" / "plate_queue" / "tutorial_queue.json"
LESSON = "30_plate_queue"


def _status_rect(capture: CaptureSession, table, row: int) -> list[int]:
    """Return one visible status-cell rectangle in capture coordinates."""
    from PySide6.QtCore import QPoint

    item = table.item(row, 3)
    if item is None:
        raise RuntimeError(f"queue row {row} has no status item")
    rect = table.visualItemRect(item)
    root = capture.window.mapToGlobal(QPoint(0, 0))
    point = table.viewport().mapToGlobal(rect.topLeft())
    scale = float(capture.window.grab().devicePixelRatio())
    return [
        int(round((point.x() - root.x()) * scale)),
        int(round((point.y() - root.y()) * scale)),
        int(round(rect.width() * scale)),
        int(round(rect.height() * scale)),
    ]


def main() -> int:
    if not MASK_SETTINGS.is_file():
        raise FileNotFoundError(MASK_SETTINGS)
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_PATH.unlink(missing_ok=True)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-plate-queue-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-plate-queue-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication
    from spacr.utils import load_settings
    from spacr.qt import plate_queue as plate_queue_module
    from spacr.qt.dnd_handlers import PlateQueueDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    # Isolate the lesson completely from the user's persistent queue.
    plate_queue_module._queue_path = lambda: QUEUE_PATH

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None

    # Exercise the real queue thread, status transitions, persistence, stop,
    # and resume behavior without launching segmentation or measurement.  The
    # narration and source manifest disclose this validation-only substitute.
    import spacr.qt.bridge as bridge

    def _validation_entry(_app_key):
        def _validate(settings):
            src = Path(str(settings.get("src", "")))
            if not src.exists():
                raise FileNotFoundError(src)
            time.sleep(1.25)
        return _validate

    bridge.resolve_pipeline_entry = _validation_entry

    settings = load_settings(
        str(MASK_SETTINGS), setting_key="Key", setting_value="Value")
    settings.update({
        "src": str(ORIG),
        "test_mode": True,
        "test_images": 5,
        "delete_intermediate": False,
    })

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError("expected device pixel ratio 1 at native 4K")

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)

    window._on_nav_selected("mask")
    wait_until(app, lambda: window._screens.get("mask") is not None,
               15.0, "Mask screen")
    mask = window._screens["mask"]
    mask.apply_settings_dict(settings)
    settle(app, 35)
    source_widget = mask._settings_model._widgets.get("src")
    if hasattr(source_widget, "setCursorPosition"):
        source_widget.setCursorPosition(0)
    capture.save("01_prepared_mask", {
        "mask_screen": mask,
        "source": source_widget,
        "settings": mask._settings_scroll,
        "actions": mask._actions_row,
    })

    window._on_nav_selected("queue")
    wait_until(app, lambda: window._screens.get("queue") is not None,
               15.0, "Plate Queue screen")
    screen = window._screens["queue"]
    # Keep every functional column legible at native 4K.  The production
    # widget otherwise lets its empty Remove column consume most of a wide
    # screen, which leaves values such as ``measure`` needlessly elided.
    from PySide6.QtWidgets import QHeaderView
    header = screen._table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(2, QHeaderView.Stretch)
    header.setSectionResizeMode(5, QHeaderView.Fixed)
    for column, width in ((0, 220), (1, 220), (3, 190), (4, 190), (5, 260)):
        screen._table.setColumnWidth(column, width)
    settle(app, 30)
    capture.save("02_open_queue", {
        "nav": nav_button(window, "queue"),
        "screen": screen,
        "toolbar": [screen._btn_add, screen._btn_import, screen._btn_clear,
                    screen._btn_run, screen._btn_stop],
        "table": screen._table,
    })

    capture.save("03_add_current", {
        "add": screen._btn_add,
        "toolbar": [screen._btn_add, screen._btn_import],
        "table": screen._table,
    })
    screen._btn_add.click()
    wait_until(app, lambda: len(screen.queue()) == 1,
               10.0, "current Mask snapshot")
    settle(app, 20)
    capture.save("04_mask_queued", {
        "table": screen._table,
        "toolbar": [screen._btn_add, screen._btn_import,
                    screen._btn_run, screen._btn_stop],
    })

    # A real folder drop discovers the Mask and Measure settings snapshots
    # whose filenames identify runnable modules.
    PlateQueueDropHandler().apply(SOURCE, screen)
    wait_until(app, lambda: len(screen.queue()) == 3,
               15.0, "dropped settings snapshots")
    settle(app, 25)
    capture.save("05_folder_drop", {
        "table": screen._table,
        "toolbar": [screen._btn_add, screen._btn_import,
                    screen._btn_clear, screen._btn_run],
    })

    reloaded = plate_queue_module.PlateQueue(path=QUEUE_PATH)
    if len(reloaded) != 3:
        raise RuntimeError("persisted tutorial queue did not reload three items")
    capture.save("06_persistence", {
        "table": screen._table,
        "run_controls": [screen._btn_run, screen._btn_stop],
    })

    capture.save("07_run", {
        "run": screen._btn_run,
        "table": screen._table,
    })
    screen._btn_run.click()
    wait_until(
        app,
        lambda: any(item.status == plate_queue_module.Status.RUNNING
                    for item in screen.queue().items()),
        15.0,
        "first validation item running",
    )
    settle(app, 10)
    capture.save("08_running", {
        "table": screen._table,
        "run_controls": [screen._btn_run, screen._btn_stop],
    })
    running_row = next(
        index for index, item in enumerate(screen.queue().items())
        if item.status == plate_queue_module.Status.RUNNING)
    capture.frames["08_running"]["status"] = _status_rect(
        capture, screen._table, running_row)

    capture.save("09_stop", {
        "stop": screen._btn_stop,
        "table": screen._table,
    })
    screen._btn_stop.click()
    wait_until(
        app,
        lambda: screen._runner is not None and not screen._runner.isRunning(),
        15.0,
        "queue stop after current item",
    )
    settle(app, 25)
    statuses = [item.status for item in screen.queue().items()]
    if statuses.count(plate_queue_module.Status.SUCCESS) != 1:
        raise RuntimeError(f"expected one completed validation, got {statuses}")
    capture.save("10_stopped", {
        "table": screen._table,
        "run_controls": [screen._btn_run, screen._btn_stop],
    })

    capture.save("11_resume", {
        "run": screen._btn_run,
        "table": screen._table,
    })
    screen._btn_run.click()
    wait_until(
        app,
        lambda: screen.queue().is_all_done()
        and screen._runner is not None and not screen._runner.isRunning(),
        30.0,
        "resumed validation queue",
    )
    settle(app, 25)
    capture.save("12_finished", {
        "table": screen._table,
        "toolbar": [screen._btn_clear, screen._btn_run, screen._btn_stop],
    })

    capture.save("13_clear", {
        "clear": screen._btn_clear,
        "table": screen._table,
    })
    screen._btn_clear.click()
    wait_until(app, lambda: len(screen.queue()) == 0,
               10.0, "clear finished jobs")
    settle(app, 20)
    capture.save("14_empty", {
        "table": screen._table,
        "toolbar": [screen._btn_add, screen._btn_import,
                    screen._btn_clear, screen._btn_run, screen._btn_stop],
    })
    capture.write_geometry()

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "queue",
        "source": str(SOURCE),
        "settings_snapshots": [
            str(MASK_SETTINGS),
            str(SOURCE / "settings" / "gen_mask_settings.csv"),
            str(SOURCE / "settings" / "measure_crop_settings.csv"),
        ],
        "execution": (
            "Real QueueScreen, PlateQueue persistence, QThread runner, stop, "
            "resume, and status transitions; tutorial-only source validation "
            "replaced expensive pipelines, so no experiment output was written."
        ),
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")

    screen.close()
    settle(app, 20)
    window.close()
    settle(app, 20)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
