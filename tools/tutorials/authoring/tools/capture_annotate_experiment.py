#!/usr/bin/env python3
"""Capture the Annotate tutorial from spaCR with real experiment crops."""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import (
    CaptureSession, center_dialog, nav_button, settle, wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DEFAULT_DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
OUTPUT = ROOT / "production" / "09_annotate" / "keyframes"
ANNOTATION_COLUMN = "spaCR_Tutorial"


def wait_for_page(app, screen, label: str, timeout: float = 180.0) -> None:
    wait_until(
        app,
        lambda: (
            not screen.is_busy()
            and screen._page_worker is None
            and screen._pending_page_load is None
            and bool(screen._page_paths)
            and any(pm is not None for pm in screen._thumb_pixmaps)
        ),
        timeout,
        label,
    )
    settle(app, 30)


def reset_tutorial_labels(db_path: Path) -> None:
    """Make reruns deterministic, touching only this tutorial's own column."""
    con = sqlite3.connect(str(db_path), timeout=30)
    try:
        columns = {row[1] for row in con.execute('PRAGMA table_info("png_list")')}
        if ANNOTATION_COLUMN in columns:
            con.execute(f'UPDATE "png_list" SET "{ANNOTATION_COLUMN}" = NULL')
        tables = {
            row[0] for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        if "annotation_rounds" in tables:
            round_columns = {
                row[1] for row in con.execute(
                    'PRAGMA table_info("annotation_rounds")'
                )
            }
            name = next(
                (item for item in ("annotation_column", "column_name")
                 if item in round_columns),
                None,
            )
            if name:
                con.execute(
                    f'DELETE FROM "annotation_rounds" WHERE "{name}" = ?',
                    (ANNOTATION_COLUMN,),
                )
        con.commit()
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    db_path = dataset / "measurements" / "measurements.db"
    crop_count = len(list((dataset / "data").rglob("*.png")))
    if not db_path.is_file() or crop_count == 0:
        raise FileNotFoundError(
            "Annotate needs measurements/measurements.db and real data crops"
        )
    reset_tutorial_labels(db_path)

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/spacr-tutorial-annotate-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-annotate-mpl")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import QPoint, QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QTabWidget

    import spacr.qt.first_run as first_run
    first_run.maybe_show_tour = lambda window: None
    import spacr.qt.walkthrough as walkthrough
    walkthrough.was_seen = lambda app_key: True
    import spacr.qt.app as app_module
    app_module._PipelinePreloader.start = lambda self: None
    from spacr.qt.dnd_handlers import AnnotateDropHandler
    from spacr.qt.preferences import apply_preferences_to_app

    app = QApplication.instance() or QApplication([])
    apply_preferences_to_app(app)
    window = app_module.MainWindow()
    window.apply_dock_mode("locked")
    window.resize(3840, 2160)
    window.show()
    settle(app, 70)
    if abs(float(window.devicePixelRatioF()) - 1.0) > 0.01:
        raise RuntimeError(
            f"capture device pixel ratio is {window.devicePixelRatioF()}, "
            "expected 1.0 for a 100% scale 4K tutorial"
        )
    window._on_nav_selected("annotate")
    settle(app, 80)
    screen = window._screens.get("annotate")
    if screen is None:
        raise RuntimeError("Annotate screen did not open")

    capture = CaptureSession(app, window, OUTPUT)

    # Console and chat match the other core workflows, but Annotate starts
    # grid-first and lets the user reveal this pane only when it is useful.
    screen._console_switch.click()
    settle(app, 30)
    capture.save("01_console", {
        "nav": nav_button(window, "annotate"),
        "console_toggle": screen._console_switch,
        "console": screen._console._console_box,
        "chat": screen._console._chat_row,
        "console_and_chat": [screen._console._console_box, screen._console._chat_row],
    })

    screen._ai_menu.popup(screen._ai_menu_btn.mapToGlobal(
        QPoint(0, screen._ai_menu_btn.height())))
    wait_until(app, screen._ai_menu.isVisible, 5.0, "AI provider menu")
    capture.save("02_ai_menu", {
        "ai_controls": [screen._ai_switch, screen._ai_menu_btn],
        "ai_menu_button": screen._ai_menu_btn,
        "ai_menu": screen._ai_menu,
    }, overlays=(screen._ai_menu,))
    screen._ai_menu.hide()
    settle(app, 12)

    from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
    providers_dialog = _ProvidersDialog(screen)
    center_dialog(window, providers_dialog, 1500, 1200)
    providers_dialog.show()
    settle(app, 30)
    provider_tabs = providers_dialog.findChild(QTabWidget)
    if provider_tabs is None:
        raise RuntimeError("AI providers dialog tabs are unavailable")
    capture.save("03_ai_providers", {
        "dialog": providers_dialog,
        "tabs": provider_tabs,
    }, overlays=(providers_dialog,))
    provider_tabs.setCurrentIndex(1)
    settle(app, 20)
    capture.save("04_ai_settings", {
        "dialog": providers_dialog,
        "tabs": provider_tabs,
    }, overlays=(providers_dialog,))
    providers_dialog.close()
    wait_until(app, lambda: not providers_dialog.isVisible(), 5.0,
               "AI providers dialog close")

    screen._console_switch.click()
    settle(app, 20)
    screen._settings.annotation_column = ANNOTATION_COLUMN
    screen._settings.image_type = "cell_png"
    screen._settings.image_size = (200, 200)
    screen._settings.channels = ["r", "g", "b"]
    screen._settings.normalize_channels = ["r", "g", "b"]
    screen._settings.percentiles = (1.0, 99.0)

    # Drive the real drop handler. It resolves this project to its database,
    # creates the requested column, starts background saving, and loads crops.
    AnnotateDropHandler().apply(dataset, screen)
    wait_for_page(app, screen, "initial annotation grid")
    capture.save("05_folder", {
        "source": screen._src_label,
        "grid": screen._grid_scroll,
        "page": screen._page_label,
    })

    # Open the real Settings dialog via its button, adjust only display and
    # tutorial-column values, capture it, then accept through the normal path.
    def capture_settings() -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog.windowTitle() != "Annotate — Settings":
            raise RuntimeError("Annotate settings dialog was not active")
        dialog._ann_col.setText(ANNOTATION_COLUMN)
        dialog._img_size.setValue(200)
        dialog._image_type.setText("cell_png")
        dialog._channels.setText("r, g, b")
        dialog._norm_channels.setText("r, g, b")
        dialog._pct_lo.setValue(1.0)
        dialog._pct_hi.setValue(99.0)
        dialog._outline.setText("")
        dialog._queue_on.setChecked(False)
        center_dialog(window, dialog, 1640, 1840)
        settle(app, 30)
        capture.save("06_settings", {
            "settings_button": screen._btn_settings,
            "dialog": dialog,
            "source_and_column": [dialog._src_edit, dialog._ann_col],
            "display": [dialog._img_size, dialog._image_type,
                        dialog._channels, dialog._norm_channels,
                        dialog._pct_lo, dialog._pct_hi],
            "outline": [dialog._outline, dialog._outline_method,
                        dialog._out_factor, dialog._out_sigma,
                        dialog._edge_thick, dialog._edge_transp,
                        dialog._edge_image],
            "filter": [dialog._obj_min, dialog._obj_max,
                       dialog._measurement, dialog._threshold,
                       dialog._threshold_dir],
            "queue": [dialog._queue_on, dialog._queue_measure,
                      dialog._queue_diversity, dialog._queue_limit],
        }, overlays=(dialog,))
        dialog.accept()

    QTimer.singleShot(120, capture_settings)
    screen._btn_settings.click()
    wait_for_page(app, screen, "settings-applied annotation grid")

    capture.save("07_grid", {
        "source": screen._src_label,
        "toolbar": [screen._btn_open, screen._btn_settings,
                    screen._btn_prev, screen._btn_next, screen._btn_skip],
        "grid": screen._grid_scroll,
        "page": screen._page_label,
    })

    screen._legend_toggle.click()
    settle(app, 20)
    capture.save("08_keyboard_reference", {
        "legend": screen._legend,
        "legend_button": screen._legend_toggle,
        "grid": screen._grid_scroll,
    })

    # Real keyboard annotation: the coloured rings are the database classes;
    # the white ring is the one crop that the next key will act on.
    screen.handle_key("1")
    settle(app, 12)
    capture.save("09_keyboard_class_one", {
        "legend": screen._legend,
        "crop": screen._thumbs[0],
        "grid": screen._grid_scroll,
    })
    screen.handle_key("2")
    settle(app, 12)
    capture.save("10_keyboard_class_two", {
        "legend": screen._legend,
        "crops": [screen._thumbs[0], screen._thumbs[1]],
        "grid": screen._grid_scroll,
    })

    screen.handle_key("Right")
    screen.handle_key("Space")
    screen.handle_key("Backspace")
    settle(app, 12)
    capture.save("11_navigation", {
        "legend": screen._legend,
        "focused_crop": screen._thumbs[screen.current_slot],
        "grid": screen._grid_scroll,
    })

    screen.handle_key("3")
    screen.handle_key("u")
    settle(app, 12)
    capture.save("12_undo", {
        "legend": screen._legend,
        "hint": screen._kbd_hint,
        "focused_crop": screen._thumbs[screen.current_slot],
    })

    # Mouse labels use the exact same write path. Left assigns class 1;
    # right assigns class 2; repeating a class on a crop clears it.
    screen._on_thumb_left(3)
    settle(app, 10)
    capture.save("13_mouse_class_one", {
        "legend": screen._legend,
        "crop": screen._thumbs[3],
        "grid": screen._grid_scroll,
    })
    screen._on_thumb_right(4)
    settle(app, 10)
    capture.save("14_mouse_class_two", {
        "legend": screen._legend,
        "crop": screen._thumbs[4],
        "grid": screen._grid_scroll,
    })
    screen._flush_pending()
    wait_until(
        app,
        lambda: (screen._worker is not None
                 and not screen._worker.busy
                 and screen._worker.pending_batches == 0),
        60.0,
        "annotation database save",
    )
    settle(app, 20)

    # Capture the app's real modal reports from inside their nested event
    # loops, so these are not mock dialogs and the toolbar buttons are clicked.
    def capture_active_modal(stem: str, button, focus_key: str,
                             expected_title: str) -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog.windowTitle() != expected_title:
            raise RuntimeError(f"{expected_title} dialog was not active")
        center_dialog(window, dialog, 1280, 900)
        settle(app, 25)
        capture.save(stem, {
            "button": button,
            focus_key: dialog,
        }, overlays=(dialog,))
        dialog.reject()

    QTimer.singleShot(
        120,
        lambda: capture_active_modal(
            "15_class_counts", screen._btn_count, "dialog", "Class counts"),
    )
    screen._btn_count.click()

    QTimer.singleShot(
        120,
        lambda: capture_active_modal(
            "16_coverage", screen._btn_coverage, "report",
            "Annotation coverage"),
    )
    screen._btn_coverage.click()

    # Page controls flush before navigating. Return to the first page so the
    # tutorial ends with its four labelled examples visible.
    screen._btn_next.click()
    wait_for_page(app, screen, "next annotation page")
    capture.save("17_next_page", {
        "next": screen._btn_next,
        "navigation": [screen._btn_prev, screen._btn_next, screen._btn_skip],
        "page": screen._page_label,
        "grid": screen._grid_scroll,
    })
    screen._btn_prev.click()
    wait_for_page(app, screen, "previous annotation page")
    capture.save("18_previous_page", {
        "back": screen._btn_prev,
        "navigation": [screen._btn_prev, screen._btn_next, screen._btn_skip],
        "page": screen._page_label,
        "grid": screen._grid_scroll,
    })
    screen._btn_skip.click()
    wait_for_page(app, screen, "last annotated page")
    capture.save("19_skip_to_annotations", {
        "skip": screen._btn_skip,
        "navigation": [screen._btn_prev, screen._btn_next, screen._btn_skip],
        "page": screen._page_label,
        "grid": screen._grid_scroll,
    })

    QTimer.singleShot(
        120,
        lambda: capture_active_modal(
            "20_rounds", screen._btn_curve, "report",
            "Active-learning rounds"),
    )
    screen._btn_curve.click()

    capture.save("21_active_learning", {
        "active_state": screen._al_label,
        "active_controls": [screen._btn_coverage, screen._btn_retrain,
                            screen._btn_curve],
        "retrain": screen._btn_retrain,
        "grid": screen._grid_scroll,
    })

    # The handoff buttons are shown together. They preseed Classify (CV) and
    # ML Analyze; no training is launched during a tutorial capture.
    capture.save("22_training_handoffs", {
        "training": [screen._btn_train_cv, screen._btn_train_xg],
        "train_cv": screen._btn_train_cv,
        "train_xg": screen._btn_train_xg,
        "toolbar": [screen._btn_retrain, screen._btn_curve,
                    screen._btn_train_cv, screen._btn_train_xg],
    })

    def capture_clear_warning() -> None:
        dialog = QApplication.activeModalWidget()
        if dialog is None or dialog.windowTitle() != "Confirm clear":
            raise RuntimeError("Clear-column confirmation was not active")
        center_dialog(window, dialog, 1100, 650)
        settle(app, 25)
        capture.save("23_clear_warning", {
            "clear": screen._btn_clear,
            "dialog": dialog,
        }, overlays=(dialog,))
        # Reject is deliberate: the tutorial never destroys its examples.
        dialog.reject()

    QTimer.singleShot(120, capture_clear_warning)
    screen._btn_clear.click()

    screen._legend_toggle.click()
    settle(app, 12)
    capture.save("24_saved_result", {
        "column": screen._src_label,
        "status": screen._status_label,
        "grid": screen._grid_scroll,
        "toolbar": [screen._btn_count, screen._btn_coverage,
                    screen._btn_retrain, screen._btn_curve,
                    screen._btn_train_cv, screen._btn_train_xg,
                    screen._btn_clear],
    })
    capture.write_geometry()

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    labels = dict(con.execute(
        f'SELECT "{ANNOTATION_COLUMN}", COUNT(*) FROM "png_list" '
        f'WHERE "{ANNOTATION_COLUMN}" IS NOT NULL '
        f'GROUP BY "{ANNOTATION_COLUMN}" ORDER BY 1'
    ).fetchall())
    column_present = ANNOTATION_COLUMN in {
        row[1] for row in con.execute('PRAGMA table_info("png_list")')
    }
    con.close()
    if labels != {1: 2, 2: 2}:
        raise RuntimeError(f"unexpected tutorial labels: {labels}")

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "database": str(db_path),
        "crop_png_count": crop_count,
        "annotation_column": ANNOTATION_COLUMN,
        "annotation_column_present": column_present,
        "saved_class_counts": labels,
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "captures": sorted(capture.frames),
        "destructive_clear_confirmed": False,
    }
    (OUTPUT.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )

    screen.close()
    window.close()
    settle(app, 20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
