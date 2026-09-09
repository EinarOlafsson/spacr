#!/usr/bin/env python3
"""Capture a real External Masks preview and import at native 4K."""
from __future__ import annotations

import json
import multiprocessing
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import (
    CaptureSession, card_named, nav_button, settle, wait_until,
)


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
STAGING = ROOT / "derived" / "external_masks" / "inputs"
SOURCE_MANIFEST = STAGING.parent / "source_manifest.json"
OUTPUT = ROOT / "generated" / "external_masks" / "project"
LESSON = "31_external_masks"


def safe_stem(value: str) -> str:
    return "_".join(value.lower().replace("&", "and").split())


def _tables(path: Path) -> list[str]:
    with sqlite3.connect(path) as connection:
        return sorted(
            row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        )


def main() -> int:
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(
            f"run prepare_external_masks_tutorial_data.py first: {SOURCE_MANIFEST}")
    if OUTPUT.exists():
        raise FileExistsError(
            f"tutorial output already exists; rerun the preparation step: {OUTPUT}")
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-external-masks-4k-config")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-external-masks-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtWidgets import QApplication, QHeaderView
    from spacr.external_masks import plan_external_masks
    from spacr.qt.dnd_handlers import ExternalMasksDropHandler
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
    window._on_nav_selected("external_masks")
    wait_until(app, lambda: window._screens.get("external_masks") is not None,
               15.0, "External Masks screen")
    screen = window._screens["external_masks"]
    settle(app, 40)
    model = screen._settings_model
    inputs = model._widgets["inputs"]
    # Avoid source elision in a wide native-4K capture while retaining the
    # editable role and object-type controls.
    input_header = inputs._table.horizontalHeader()
    input_header.setStretchLastSection(False)
    input_header.setSectionResizeMode(0, QHeaderView.Stretch)
    for column, width in ((1, 180), (2, 170), (3, 190), (4, 100), (5, 140)):
        input_header.setSectionResizeMode(column, QHeaderView.Fixed)
        inputs._table.setColumnWidth(column, width)

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav_button(window, "external_masks"),
        "screen": screen,
        "settings": screen._settings_scroll,
        "console": screen._console._console_box,
        "actions": screen._actions_row,
    })

    ExternalMasksDropHandler().apply(STAGING, screen)
    wait_until(app, lambda: inputs.group_count() == 5,
               30.0, "five detected input groups")
    values = {
        "dst": str(OUTPUT),
        "layout": "flat",
        "z_handling": "max",
        "plate_naming": "index",
        "recursive": True,
        "overwrite": False,
        "preview_only": True,
        "channels": [0, 1, 2, 3],
        "crop_mode": ["cell"],
        "save_measurements": True,
        "save_png": True,
        "cytoplasm": True,
        "plot": False,
        "n_jobs": 1,
        "strict_errors": True,
    }
    for key, value in values.items():
        if not model.set_value_for_key(key, value):
            raise RuntimeError(f"could not set External Masks option {key}")
    settle(app, 35)
    if inputs.file_count() != 24:
        raise RuntimeError(f"expected 24 detected TIFFs, got {inputs.file_count()}")
    capture.save("02_detected", {
        "inputs": inputs,
        "input_table": inputs._table,
        "console": screen._console._console_box,
    })

    plan = plan_external_masks(model.collect())
    if not plan.ok or len(plan.stems) != 3 or plan.n_channels != 4:
        raise RuntimeError(plan.summary())
    capture.save("03_assignment", {
        "input_table": inputs._table,
        "input_buttons": [inputs._add_files, inputs._add_folder, inputs._remove],
    })

    search = screen._settings_search
    search.set_modified_only(False)
    search.set_level("all")
    settle(app, 20)
    category_frames = []
    for index, section in enumerate(screen._settings_sections, start=1):
        for other in screen._settings_sections:
            other.set_expanded(False)
        if not section.isVisible():
            continue
        section.set_expanded(True)
        screen._settings_scroll.ensureWidgetVisible(section.header(), 12, 12)
        settle(app, 25)
        stem = f"{index + 3:02d}_category_{safe_stem(section.title())}"
        capture.save(stem, {
            "category": section,
            "category_header": section.header(),
            "category_hint": screen._category_hint,
        })
        category_frames.append((stem, section.title()))
    if len(category_frames) != 9:
        raise RuntimeError(f"expected 9 categories, found {category_frames}")

    input_section = next(
        section for section in screen._settings_sections
        if section.title() == "INPUT MAPPING")
    for section in screen._settings_sections:
        section.set_expanded(section is input_section)
    screen._settings_scroll.ensureWidgetVisible(input_section.header(), 12, 12)
    settle(app, 25)
    capture.save("13_preview_settings", {
        "category": input_section,
        "preview_only": model._widgets["preview_only"],
        "destination": model._widgets["dst"],
        "actions": screen._actions_row,
    })
    capture.save("14_preview_run", {
        "run": screen._btn_run,
        "actions": screen._actions_row,
        "preview_only": model._widgets["preview_only"],
    })
    screen._btn_run.click()
    wait_until(app, lambda: screen._btn_stop.isEnabled(),
               15.0, "preview start")
    wait_until(
        app,
        lambda: screen._btn_run.isEnabled()
        and not screen._btn_stop.isEnabled()
        and screen._thread is None,
        90.0,
        "read-only preview",
    )
    if screen._last_error_text:
        raise RuntimeError(screen._last_error_text)
    settle(app, 30)
    capture.save("15_preview_result", {
        "console": screen._console._console_box,
        "actions": screen._actions_row,
    })

    if not model.set_value_for_key("preview_only", False):
        raise RuntimeError("could not disable preview-only mode")
    settle(app, 20)
    capture.save("16_import_run", {
        "run": screen._btn_run,
        "actions": screen._actions_row,
        "preview_only": model._widgets["preview_only"],
    })
    screen._btn_run.click()
    wait_until(app, lambda: screen._btn_stop.isEnabled(),
               15.0, "external-mask import start")
    settle(app, 30)
    capture.save("17_importing", {
        "console": screen._console._console_box,
        "actions": screen._actions_row,
        "progress": screen._progress,
    })
    wait_until(
        app,
        lambda: screen._btn_run.isEnabled()
        and not screen._btn_stop.isEnabled()
        and screen._thread is None,
        900.0,
        "external-mask import",
    )
    if screen._last_error_text:
        raise RuntimeError(screen._last_error_text)
    settle(app, 40)
    db_path = OUTPUT / "measurements" / "measurements.db"
    merged = sorted((OUTPUT / "merged").glob("*.npy"))
    if not db_path.is_file() or len(merged) != 3:
        raise RuntimeError(
            f"incomplete import: db={db_path.is_file()} merged={len(merged)}")
    tables = _tables(db_path)
    console_scroll = screen._console._console_box.verticalScrollBar()
    console_scroll.setValue(console_scroll.maximum())
    settle(app, 20)
    capture.save("18_finished", {
        "console": screen._console._console_box,
        "actions": screen._actions_row,
        "system": card_named(screen, "System"),
    })
    capture.write_geometry()

    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "external_masks",
        "source_manifest": json.loads(SOURCE_MANIFEST.read_text()),
        "plan": {
            "summary": plan.summary(),
            "stems": plan.stems,
            "channels": plan.n_channels,
            "object_types": plan.object_types,
            "mask_dims": plan.mask_dims,
            "warnings": plan.warnings,
            "errors": plan.errors,
        },
        "result": {
            "destination": str(OUTPUT),
            "merged": [str(path) for path in merged],
            "database": str(db_path),
            "tables": tables,
            "data": str(OUTPUT / "data"),
        },
        "capture_size": [3840, 2160],
        "device_pixel_ratio": float(window.devicePixelRatioF()),
        "category_frames": category_frames,
        "captures": sorted(capture.frames),
    }
    (output.parent / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    screen.close()
    settle(app, 30)
    window.close()
    settle(app, 20)
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
