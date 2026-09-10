#!/usr/bin/env python3
"""Capture the real Training Runs screen at native 4K."""
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
SOURCE = ROOT / "synthetic" / "training_runs"
LESSON = "28_training_runs"


def _run_id(screen, folder: str) -> str:
    for run in screen.runs():
        if folder in str(run.path):
            return run.run_id
    raise RuntimeError(f"No discovered run contains {folder!r}")


def _row_rect(capture: CaptureSession, list_widget, run_id: str) -> list[int]:
    from PySide6.QtCore import QPoint, Qt

    for index in range(list_widget.count()):
        item = list_widget.item(index)
        if item.data(Qt.UserRole) != run_id:
            continue
        rect = list_widget.visualItemRect(item)
        root = capture.window.mapToGlobal(QPoint(0, 0))
        point = list_widget.viewport().mapToGlobal(rect.topLeft())
        scale = float(capture.window.grab().devicePixelRatio())
        return [
            int(round((point.x() - root.x()) * scale)),
            int(round((point.y() - root.y()) * scale)),
            int(round(min(rect.width(), list_widget.viewport().width()) * scale)),
            int(round(rect.height() * scale)),
        ]
    raise RuntimeError(f"No list row for {run_id}")


def main() -> int:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-training-runs-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-training-runs-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
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
    window._on_nav_selected("train_compare")
    wait_until(app, lambda: window._screens.get("train_compare") is not None,
               15.0, "Training Runs screen")
    screen = window._screens["train_compare"]
    settle(app, 35)

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav_button(window, "train_compare"),
        "screen": screen,
        "source": [screen._path_edit, screen._btn_pick, screen._btn_scan],
        "runs": screen._runs_list,
        "plot": screen._canvas,
        "diff": [screen._diff_summary, screen._diff_table],
        "status": screen._status,
    })

    screen._path_edit.setText(str(SOURCE))
    settle(app, 20)
    capture.save("02_source", {
        "source": [screen._path_edit, screen._btn_pick, screen._btn_scan],
        "scan": screen._btn_scan,
        "status": screen._status,
    })
    screen._btn_scan.click()
    wait_until(app, lambda: not screen._busy and len(screen.runs()) == 4,
               30.0, "four training runs")
    settle(app, 35)

    baseline = _run_id(screen, "baseline")
    tuned = _run_id(screen, "tuned")
    cv = _run_id(screen, "cross_validated")
    broken = _run_id(screen, "incomplete")
    capture.save("03_discovered", {
        "runs": screen._runs_list,
        "problems": screen._problems,
        "metrics": [screen._metric_combo, screen._fold_combo],
        "status": screen._status,
        "discovery": [screen._runs_list, screen._problems, screen._status],
    })

    capture.save("04_select_baseline", {"runs": screen._runs_list})
    capture.frames["04_select_baseline"]["run_row"] = _row_rect(
        capture, screen._runs_list, baseline)
    screen.select_runs([baseline])
    settle(app, 15)
    capture.save("05_select_tuned", {"runs": screen._runs_list})
    capture.frames["05_select_tuned"]["run_row"] = _row_rect(
        capture, screen._runs_list, tuned)
    screen.select_runs([baseline, tuned])
    settle(app, 15)

    capture.save("06_controls", {
        "metrics": [screen._metric_combo, screen._fold_combo],
        "overlay": screen._btn_overlay,
        "runs": screen._runs_list,
    })
    capture.save("07_overlay", {
        "overlay": screen._btn_overlay,
        "runs": screen._runs_list,
    })
    screen._btn_overlay.click()
    settle(app, 60)
    capture.save("08_accuracy", {
        "plot": screen._canvas,
        "diff": [screen._diff_summary, screen._diff_table],
        "result": [screen._canvas, screen._diff_summary, screen._diff_table],
        "status": screen._status,
    })

    capture.save("09_pick_curve", {"plot": screen._canvas})
    val_labels = [label for label in screen.series_labels() if "val" in label]
    if not val_labels:
        raise RuntimeError("comparison produced no validation series")
    screen.identify_series(val_labels[-1])
    settle(app, 20)
    capture.save("10_best_last", {
        "plot": screen._canvas,
        "picked": screen._picked,
        "status": screen._status,
    })

    capture.save("11_choose_loss", {"metric": screen._metric_combo})
    if not screen.set_metric("loss"):
        raise RuntimeError(screen.status_text())
    settle(app, 40)
    capture.save("12_loss", {
        "plot": screen._canvas,
        "diff": [screen._diff_summary, screen._diff_table],
        "metric": screen._metric_combo,
    })

    screen.select_runs([tuned, cv])
    screen.set_metric("accuracy")
    capture.save("13_choose_fold_mean", {
        "fold": screen._fold_combo,
        "runs": screen._runs_list,
    })
    screen.set_fold_mode("mean")
    screen.overlay()
    settle(app, 55)
    capture.save("14_fold_mean", {
        "plot": screen._canvas,
        "diff": [screen._diff_summary, screen._diff_table],
        "fold": screen._fold_combo,
        "status": screen._status,
    })

    capture.save("15_choose_per_fold", {"fold": screen._fold_combo})
    screen.set_fold_mode("per_fold")
    screen.overlay()
    settle(app, 55)
    capture.save("16_per_fold", {
        "plot": screen._canvas,
        "diff": [screen._diff_summary, screen._diff_table],
        "fold": screen._fold_combo,
        "status": screen._status,
    })
    capture.write_geometry()

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "train_compare",
        "source": str(SOURCE),
        "disclosure": "Synthetic progress CSVs; real spaCR parser and UI.",
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "run_ids": {
            "baseline": baseline, "tuned": tuned,
            "cross_validated": cv, "incomplete": broken,
        },
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
