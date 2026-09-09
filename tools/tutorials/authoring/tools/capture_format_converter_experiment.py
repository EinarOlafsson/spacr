#!/usr/bin/env python3
"""Capture Format Converter with unchanged pixels from the real experiment."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from capture_mask_experiment import CaptureSession, nav_button, settle


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
ORIGINALS = DATASET / "orig"
GENERATED = ROOT / "generated" / "format_converter"
SOURCE = GENERATED / "source" / "tutorial_plate" / "A01" / "images"
DESTINATION = GENERATED / "converted"
LESSON = "35_converter"
SOURCE_FIELDS = (
    (1, "W0127F0004T0001Z000"),
    (2, "W0315F0003T0001Z000"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_real_source() -> list[dict]:
    if GENERATED.exists():
        raise FileExistsError(
            f"tutorial output already exists; remove only {GENERATED} to rerun"
        )
    SOURCE.mkdir(parents=True)
    records = []
    for field, prefix in SOURCE_FIELDS:
        for channel in range(1, 5):
            original = ORIGINALS / f"{prefix}C{channel}.tif"
            if not original.is_file():
                raise FileNotFoundError(original)
            prepared = SOURCE / f"fov{field:02d}_C{channel}.tif"
            shutil.copy2(original, prepared)
            original_hash = _sha256(original)
            prepared_hash = _sha256(prepared)
            if original_hash != prepared_hash:
                raise RuntimeError(f"pixel source copy differs: {prepared}")
            records.append({
                "field": field,
                "channel": channel,
                "original": str(original),
                "prepared": str(prepared),
                "sha256": original_hash,
            })
    return records


def main() -> int:
    source_records = _prepare_real_source()
    prepared_before = {row["prepared"]: _sha256(Path(row["prepared"]))
                       for row in source_records}

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-format-converter-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-converter-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from spacr import convert as cvt
    from spacr.qt.dnd_handlers import ConvertDropHandler
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

    nav = nav_button(window, "convert")
    if nav is None:
        raise RuntimeError("Format Converter navigation button was not found")
    QTest.mouseClick(nav, Qt.LeftButton)
    settle(app, 45)
    screen = window._screens.get("convert")
    if screen is None:
        raise RuntimeError("Format Converter screen was not created")
    screen._threaded = False

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav,
        "screen": screen,
        "source": [screen._src_edit, screen._btn_pick_src],
        "options": [screen._layout_box, screen._z_box,
                    screen._plate_box, screen._resume],
        "destination": [screen._dst_edit, screen._btn_pick_dst,
                        screen._btn_preview, screen._btn_convert],
    })

    ConvertDropHandler().apply(SOURCE.parent.parent.parent, screen)
    screen.set_destination(str(DESTINATION))
    settle(app, 30)
    if screen.source_path() != str(SOURCE.parent.parent.parent):
        raise RuntimeError(f"wrong source after drop: {screen.source_path()}")
    capture.save("02_source", {
        "source": [screen._src_edit, screen._btn_pick_src],
        "destination": [screen._dst_edit, screen._btn_pick_dst],
    })

    capture.save("03_options", {
        "options": [screen._layout_box, screen._z_box,
                    screen._plate_box, screen._resume],
        "source_destination": [screen._src_edit, screen._dst_edit],
    })
    capture.save("04_preview", {
        "configuration": [screen._src_edit, screen._layout_box,
                          screen._z_box, screen._plate_box,
                          screen._dst_edit, screen._btn_preview,
                          screen._btn_convert],
        "preview": screen._btn_preview,
    })
    QTest.mouseClick(screen._btn_preview, Qt.LeftButton)
    settle(app, 45)
    if screen.preview_row_count() != 8 or not screen.can_convert():
        raise RuntimeError(
            f"expected an eight-file valid plan, got "
            f"{screen.preview_row_count()} rows: {screen.status_text()}"
        )
    plan_targets = screen.preview_targets()
    expected_first = "plate1_A01_T0001F001L01A01Z01C01.tif"
    expected_last = "plate1_A01_T0001F002L01A01Z01C04.tif"
    if plan_targets[0] != expected_first or plan_targets[-1] != expected_last:
        raise RuntimeError(f"unexpected converter targets: {plan_targets}")
    capture.save("05_plan", {
        "mapping": screen._table,
        "summary": screen._summary,
        "status": screen._status,
    })

    capture.save("06_convert", {
        "mapping": screen._table,
        "summary": screen._summary,
        "destination": [screen._dst_edit, screen._btn_preview,
                        screen._btn_convert],
        "convert": screen._btn_convert,
    })
    QTest.mouseClick(screen._btn_convert, Qt.LeftButton)
    settle(app, 50)
    result = screen.result()
    if result is None or result.n_written != 8 or not result.is_complete:
        raise RuntimeError(f"conversion failed: {screen.status_text()}")
    map_frame = cvt.read_map(result.map_path)
    if len(map_frame) != 8 or not map_frame["target"].is_unique:
        raise RuntimeError("conversion map does not contain eight unique targets")
    capture.save("07_result", {
        "mapping": screen._table,
        "summary": screen._summary,
        "status": screen._status,
    })

    QTest.mouseClick(screen._resume, Qt.LeftButton)
    settle(app, 25)
    if not screen.resume_enabled():
        raise RuntimeError("Resume did not switch on")
    capture.save("08_resume", {
        "options": [screen._layout_box, screen._z_box,
                    screen._plate_box, screen._resume],
        "resume": screen._resume,
        "summary": screen._summary,
    })

    capture.save("09_resume_convert", {
        "destination": [screen._dst_edit, screen._btn_preview,
                        screen._btn_convert],
        "convert": screen._btn_convert,
        "summary": screen._summary,
    })
    QTest.mouseClick(screen._btn_convert, Qt.LeftButton)
    settle(app, 50)
    resumed = screen.result()
    if resumed is None or resumed.n_written != 0:
        raise RuntimeError(f"resume unexpectedly rewrote files: {screen.status_text()}")
    if len(resumed.resumed_fields) != 2:
        raise RuntimeError(
            f"expected two resumed fields, got {len(resumed.resumed_fields)}"
        )
    capture.save("10_resumed", {
        "summary": screen._summary,
        "status": screen._status,
        "options": [screen._layout_box, screen._z_box,
                    screen._plate_box, screen._resume],
    })
    capture.write_geometry()

    prepared_after = {path: _sha256(Path(path)) for path in prepared_before}
    if prepared_before != prepared_after:
        raise RuntimeError("Format Converter modified a prepared source TIFF")
    output_tiffs = sorted(DESTINATION.glob("*.tif"))
    if len(output_tiffs) != 8:
        raise RuntimeError(f"expected eight converted TIFFs, got {len(output_tiffs)}")
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "convert",
        "source_dataset": str(DATASET),
        "source_disclosure": (
            "Two real four-channel fields were copied byte-for-byte and "
            "renamed into a plate/well folder tree for the conversion demo; "
            "no synthetic pixel data were used."
        ),
        "source_files": source_records,
        "prepared_sha256_before": prepared_before,
        "prepared_sha256_after": prepared_after,
        "layout": screen.layout_mode(),
        "z_handling": screen.z_handling(),
        "plate_naming": screen.plate_naming(),
        "plan": {
            "rows": len(plan_targets),
            "first_target": plan_targets[0],
            "last_target": plan_targets[-1],
            "targets": plan_targets,
        },
        "conversion": {
            "destination": str(DESTINATION),
            "written": 8,
            "map": result.map_path,
            "map_rows": len(map_frame),
        },
        "resume": {
            "written": int(resumed.n_written),
            "resumed_fields": list(resumed.resumed_fields),
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
