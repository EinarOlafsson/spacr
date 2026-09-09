#!/usr/bin/env python3
"""Capture Import Project with real images, masks, and measurements at 4K."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from capture_mask_experiment import CaptureSession, nav_button, settle


ROOT = Path(__file__).resolve().parents[1]
REPO = Path("/mnt/firecuda2/codex/repo/spacr")
DATASET = Path(
    "/mnt/firecuda2/Claude/toxoplasma_projects/test_datasets/"
    "spacr/tutorials"
)
DATABASE = DATASET / "measurements" / "measurements.db"
GENERATED = ROOT / "generated" / "import_project"
PREPARED = GENERATED / "prepared_foreign_project"
IMAGES = PREPARED / "images"
TABLE = PREPARED / "measurements.csv"
MAPPING = GENERATED / "reviewed_column_map.csv"
DESTINATION = GENERATED / "imported_spacr_project"
LESSON = "36_import"
PIXEL_SIZE = 0.65
OBJECTS = ("cell", "nucleus", "pathogen", "organelle")
FIELDS = (
    {
        "tutorial_field": 1,
        "source_stem": "test_E11_2_1.npy",
        "rowID": "r5",
        "columnID": "c11",
        "fieldID": "f2",
        "well": "E11",
    },
    {
        "tutorial_field": 2,
        "source_stem": "test_M07_3_1.npy",
        "rowID": "r13",
        "columnID": "c7",
        "fieldID": "f3",
        "well": "M07",
    },
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_real_inputs() -> tuple[list[dict], dict[str, str]]:
    """Decode real spaCR arrays into a neutral third-party-style project."""
    if GENERATED.exists():
        raise FileExistsError(
            f"tutorial output already exists; remove only {GENERATED} to rerun"
        )
    IMAGES.mkdir(parents=True)
    mask_dirs = {name: PREPARED / f"{name}_masks" for name in OBJECTS}
    for folder in mask_dirs.values():
        folder.mkdir(parents=True)

    source_records: list[dict] = []
    measurement_frames: list[pd.DataFrame] = []
    connection = sqlite3.connect(str(DATABASE))
    try:
        for spec in FIELDS:
            field = int(spec["tutorial_field"])
            source_stem = str(spec["source_stem"])
            stack_path = DATASET / "orig" / "test" / "stack" / source_stem
            stack = np.load(stack_path, mmap_mode="r")
            if stack.shape != (2000, 2000, 4) or stack.dtype != np.uint16:
                raise RuntimeError(f"unexpected real stack: {stack_path} {stack.shape}")
            for channel in range(4):
                target = IMAGES / f"fov{field:02d}_C{channel + 1}.tif"
                tifffile.imwrite(target, np.asarray(stack[:, :, channel]))

            masks: dict[str, dict] = {}
            for object_type in OBJECTS:
                source = (
                    DATASET / "orig" / "test" / "masks"
                    / f"{object_type}_mask_stack" / source_stem
                )
                array = np.load(source, mmap_mode="r")
                if array.shape != (2000, 2000) or array.dtype != np.uint16:
                    raise RuntimeError(f"unexpected real mask: {source} {array.shape}")
                target = mask_dirs[object_type] / (
                    f"fov{field:02d}_{object_type}_mask.tif"
                )
                tifffile.imwrite(target, np.asarray(array))
                labels = np.unique(array)
                masks[object_type] = {
                    "source": str(source),
                    "source_sha256": _sha256(source),
                    "prepared": str(target),
                    "objects": int(np.count_nonzero(labels)),
                }

            query = (
                'SELECT object_label, cell_area, '
                'cell_channel_1_mean_intensity FROM cell '
                'WHERE rowID = ? AND columnID = ? AND fieldID = ? '
                'ORDER BY object_label'
            )
            frame = pd.read_sql_query(
                query,
                connection,
                params=(spec["rowID"], spec["columnID"], spec["fieldID"]),
            )
            frame.insert(0, "ImageNumber", f"fov{field:02d}_C1.tif")
            frame.rename(columns={
                "object_label": "ObjectNumber",
                "cell_channel_1_mean_intensity": "MeanIntensity_ER",
            }, inplace=True)
            frame["AreaShape_Area_um2"] = frame["cell_area"] * PIXEL_SIZE ** 2
            frame["Metadata_Well"] = str(spec["well"])
            frame = frame[[
                "ImageNumber", "ObjectNumber", "AreaShape_Area_um2",
                "MeanIntensity_ER", "Metadata_Well", "cell_area",
            ]]
            measurement_frames.append(frame)
            source_records.append({
                "tutorial_field": field,
                "source_stack": str(stack_path),
                "source_stack_sha256": _sha256(stack_path),
                "prepared_images": [
                    str(IMAGES / f"fov{field:02d}_C{channel}.tif")
                    for channel in range(1, 5)
                ],
                "well": spec["well"],
                "database_field": {
                    "rowID": spec["rowID"],
                    "columnID": spec["columnID"],
                    "fieldID": spec["fieldID"],
                },
                "measurement_rows": int(len(frame)),
                "masks": masks,
            })
    finally:
        connection.close()

    measurements = pd.concat(measurement_frames, ignore_index=True)
    measurements.to_csv(TABLE, index=False)
    return source_records, {name: str(folder) for name, folder in mask_dirs.items()}


def _source_hashes(records: list[dict]) -> dict[str, str]:
    hashes = {str(DATABASE): _sha256(DATABASE)}
    for record in records:
        hashes[record["source_stack"]] = record["source_stack_sha256"]
        for mask in record["masks"].values():
            hashes[mask["source"]] = mask["source_sha256"]
    return hashes


def main() -> int:
    source_records, mask_dirs = _prepare_real_inputs()
    source_before = _source_hashes(source_records)

    multiprocessing.set_start_method("spawn", force=True)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["QT_SCALE_FACTOR"] = "1"
    os.environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_FONT_DPI"] = "96"
    os.environ.setdefault("SPACR_LANGUAGE", "en")
    os.environ.setdefault(
        "XDG_CONFIG_HOME", "/tmp/spacr-tutorial-import-project-4k-config"
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/spacr-tutorial-import-mpl")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(REPO))

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QFileDialog
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

    nav = nav_button(window, "foreign")
    if nav is None:
        raise RuntimeError("Import Project navigation button was not found")
    QTest.mouseClick(nav, Qt.LeftButton)
    settle(app, 45)
    screen = window._screens.get("foreign")
    if screen is None:
        raise RuntimeError("Import Project screen was not created")
    screen._threaded = False

    output = ROOT / "production" / LESSON / "keyframes"
    capture = CaptureSession(app, window, output)
    capture.save("01_overview", {
        "nav": nav,
        "screen": screen,
        "inputs": [screen._images_edit, screen._object_box,
                   screen._mask_edit, screen._mask_list,
                   screen._table_edit, screen._scale_edit,
                   screen._conflict_box, screen._btn_preview],
        "review": [screen._table, screen._report],
        "actions": [screen._dst_edit, screen._btn_save_map,
                    screen._btn_load_map, screen._btn_import],
    })

    screen.set_images(str(IMAGES))
    screen.set_destination(str(DESTINATION))
    settle(app, 20)
    capture.save("02_images", {
        "images": [screen._images_edit, screen._btn_pick_images],
        "destination": [screen._dst_edit, screen._btn_pick_dst],
    })

    # Use the real Add button for every class; capture the final Add click.
    for object_type in OBJECTS[:-1]:
        screen._object_box.setCurrentIndex(screen._object_box.findData(object_type))
        screen._mask_edit.setText(mask_dirs[object_type])
        QTest.mouseClick(screen._btn_add_mask, Qt.LeftButton)
        settle(app, 12)
    final_type = OBJECTS[-1]
    screen._object_box.setCurrentIndex(screen._object_box.findData(final_type))
    screen._mask_edit.setText(mask_dirs[final_type])
    capture.save("03_masks", {
        "mask_entry": [screen._object_box, screen._mask_edit,
                       screen._btn_pick_mask, screen._btn_add_mask,
                       screen._btn_remove_mask],
        "mask_list": screen._mask_list,
        "add": screen._btn_add_mask,
    })
    QTest.mouseClick(screen._btn_add_mask, Qt.LeftButton)
    settle(app, 20)
    if list(screen.mask_folders()) != list(OBJECTS):
        raise RuntimeError(f"wrong masks on screen: {screen.mask_folders()}")

    screen.set_measurements(str(TABLE))
    screen.set_pixel_size(PIXEL_SIZE)
    screen.set_on_conflict("rename")
    settle(app, 20)
    capture.save("04_measurements", {
        "masks": screen._mask_list,
        "measurements": [screen._table_edit, screen._btn_pick_table],
        "options": [screen._scale_edit, screen._conflict_box],
        "destination": [screen._dst_edit, screen._btn_pick_dst],
    })

    capture.save("05_preview", {
        "configuration": [screen._images_edit, screen._mask_list,
                          screen._table_edit, screen._scale_edit,
                          screen._conflict_box, screen._btn_preview],
        "preview": screen._btn_preview,
    })
    QTest.mouseClick(screen._btn_preview, Qt.LeftButton)
    settle(app, 60)
    plan = screen.plan()
    if plan is None or not plan.ok or len(plan.stems) != 2:
        raise RuntimeError(f"invalid import plan: {screen.status_text()}")
    if (plan.join.rows_total, plan.join.rows_matched) != (207, 206):
        raise RuntimeError(
            f"unexpected join: {plan.join.rows_matched}/{plan.join.rows_total}"
        )
    if plan.join.n_objects_unmeasured != 1198:
        raise RuntimeError(
            f"unexpected unmeasured count: {plan.join.n_objects_unmeasured}"
        )
    if plan.uncalibrated:
        raise RuntimeError(f"unexpected uncalibrated columns: {plan.uncalibrated}")
    capture.save("06_plan", {
        "mapping": screen._table,
        "report": screen._report,
        "status": screen._status,
    })

    row = screen._model.row_of("AreaShape_Area_um2")
    if row < 0:
        raise RuntimeError("area mapping was not inferred")
    target_index = screen._model.index(row, 1)
    screen._table.scrollTo(target_index)
    settle(app, 20)
    edit_rect = screen._table.visualRect(target_index)
    edit_point = edit_rect.center()
    edit_global = screen._table.viewport().mapTo(window, edit_point)
    capture.save("07_edit_mapping", {
        "mapping": screen._table,
        "report": screen._report,
    })
    capture.frames["07_edit_mapping"]["target_cell"] = [
        int(edit_global.x() - 12), int(edit_global.y() - 12), 24, 24
    ]
    if not screen.set_mapping_value(
        row, "target", "foreign_reviewed_cell_area"
    ):
        raise RuntimeError("mapping target edit was rejected")
    settle(app, 25)
    if plan.target_for("AreaShape_Area_um2") == "foreign_reviewed_cell_area":
        raise RuntimeError("stale plan unexpectedly changed in place")
    if screen.plan().target_for("AreaShape_Area_um2") != "foreign_reviewed_cell_area":
        raise RuntimeError("edited mapping did not re-resolve")
    capture.save("08_reviewed", {
        "mapping": screen._table,
        "report": screen._report,
        "status": screen._status,
    })

    capture.save("09_save_mapping", {
        "mapping": screen._table,
        "actions": [screen._dst_edit, screen._btn_save_map,
                    screen._btn_load_map, screen._btn_import],
        "save": screen._btn_save_map,
    })
    original_dialog = QFileDialog.getSaveFileName
    QFileDialog.getSaveFileName = staticmethod(
        lambda *args, **kwargs: (str(MAPPING), "CSV (*.csv)")
    )
    try:
        QTest.mouseClick(screen._btn_save_map, Qt.LeftButton)
    finally:
        QFileDialog.getSaveFileName = original_dialog
    settle(app, 25)
    if not MAPPING.is_file():
        raise RuntimeError("reviewed mapping was not saved")
    capture.save("10_saved", {
        "mapping": screen._table,
        "actions": [screen._dst_edit, screen._btn_save_map,
                    screen._btn_load_map, screen._btn_import],
        "status": screen._status,
    })

    capture.save("11_import", {
        "review": [screen._table, screen._report],
        "actions": [screen._dst_edit, screen._btn_save_map,
                    screen._btn_load_map, screen._btn_import],
        "import": screen._btn_import,
    })
    QTest.mouseClick(screen._btn_import, Qt.LeftButton)
    settle(app, 80)
    result = screen.result()
    if result is None or not result.is_complete:
        raise RuntimeError(f"import failed: {screen.status_text()}")
    if result.n_fields != 2 or len(result.mask_files) != 8:
        raise RuntimeError(
            f"unexpected artifacts: fields={result.n_fields}, masks={len(result.mask_files)}"
        )
    if result.rows != {"foreign_cell": 207, "cell": 207}:
        raise RuntimeError(f"unexpected imported row counts: {result.rows}")
    for path in result.merged:
        merged = np.load(path, mmap_mode="r")
        if merged.shape != (2000, 2000, 8):
            raise RuntimeError(f"unexpected merged shape: {path} {merged.shape}")
    connection = sqlite3.connect(result.db_path)
    try:
        tables = sorted(row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%'"
        ))
        foreign_rows = int(connection.execute(
            "SELECT COUNT(*) FROM foreign_cell"
        ).fetchone()[0])
    finally:
        connection.close()
    if foreign_rows != 207:
        raise RuntimeError(f"unexpected foreign_cell rows: {foreign_rows}")
    capture.save("12_result", {
        "report": screen._report,
        "status": screen._status,
        "destination": [screen._dst_edit, screen._btn_import],
    })
    capture.write_geometry()

    source_after = {path: _sha256(Path(path)) for path in source_before}
    if source_before != source_after:
        raise RuntimeError("Import Project modified a transferred source file")
    manifest = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "app_key": "foreign",
        "source_dataset": str(DATASET),
        "source_disclosure": (
            "Two real four-channel fields and all four real label-mask classes "
            "were decoded from the transferred spaCR arrays into neutral TIFF "
            "inputs. The 207 measurement rows and values came from the real "
            "measurements database; area in square micrometres was derived from "
            "real pixel area using the disclosed 0.65 micrometre pixel size. "
            "No synthetic pixel, mask, or measurement values were used."
        ),
        "source_files": source_records,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "prepared": {
            "images": str(IMAGES),
            "masks": mask_dirs,
            "measurements": str(TABLE),
            "measurement_rows": 207,
            "pixel_size_um": PIXEL_SIZE,
        },
        "preview": {
            "fields": 2,
            "channels": 4,
            "mask_classes": list(OBJECTS),
            "rows_total": 207,
            "rows_matched": 206,
            "mask_objects_without_measurements": 1198,
            "conflict_policy": "rename",
        },
        "reviewed_mapping": str(MAPPING),
        "destination": str(DESTINATION),
        "result": {
            "fields": result.n_fields,
            "converted_images": result.conversion.n_written,
            "mask_arrays": len(result.mask_files),
            "merged_arrays": len(result.merged),
            "merged_shape": [2000, 2000, 8],
            "database": result.db_path,
            "tables": tables,
            "rows": result.rows,
        },
    }
    (output / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    window.close()
    app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
